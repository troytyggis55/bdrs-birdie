# BDRS Birdie – Vision & Control Loop Reference

## Project Overview
Two-wheeled robot that uses hybrid YOLO + classical vision to detect colored balls (red/blue) and drive toward them. MQTT is the communication layer between the Python vision controller and the Teensy hardware.

**Current main goal:** `final_ball_mission.py` — pick up a red ball and deliver it to quadrant A, pick up a blue ball and deliver it to quadrant C, then return to start. ArUco markers on the quadrant walls serve as localization landmarks throughout.

---

## Architecture

### Entry Points
- `mqtt-client.py` – Main dispatcher; CLI args select which mission to run
- `uservice.py` – MQTT service framework (two clients: RX + TX)

### Vision Pipeline
| File | Role |
|------|------|
| `CamVision/visualcontrol.py` | Classical HSV masking + contour detection + YOLO wrapper for localization |
| `CamVision/bucketballsyolo.py` | Ultralytics YOLO wrapper (`get_labels()`) |
| `CamVision/coord_conversion.py` | Pixel→robot-frame 3D coordinate conversion (balls via sphere-radius depth, ArUco via solvePnP) |
| `CamVision/ballcoords.py` | Re-export shim for `coord_conversion` — kept for backwards compatibility |
| `CamVision/pictures.py` | Image I/O, video recording |
| `CamVision/bucketballsmission.py` | Single-ball control loop (reference for fine-approach state machine) |

### Missions
| File | Role |
|------|------|
| `ball_mission/final_ball_mission.py` | **Primary mission** — deliver red→A, blue→C using ArUco landmarks |
| `pathfinding/simulate_ball_mission.py` | Older simulated mission (reference for nav+tracker patterns) |

### Pathfinding & Navigation
| File | Role |
|------|------|
| `pathfinding/pathfinding.py` | `CircleObstacle`, `PlannerConfig`, `find_safe_start` — RRT-style planner |
| `pathfinding/realtime_pathfind.py` | `RealtimePathfinder` — replanning wrapper |
| `pathfinding/pathfind_logger.py` | `PathfindLogger` — event-based debug log writer |
| `odometry/graph_nav.py` | `graph_nav.drive_to()` — waypoint follower using odometry |
| `worldmodel/ball_world_model.py` | `BallWorldModel` — EMA-smoothed world-frame ball position tracking |

### ArUco Detection
| File | Role |
|------|------|
| `aruco/detect_aruco.py` | `detect_aruco(frame)` — OpenCV ArUco marker detection, returns list of dicts with id/corners/center |
| `aruco/aruco_coords.py` | Re-export shim for `aruco_to_robot_frame` from `coord_conversion` |
| `aruco/go_to_aruco_mission.py` | Reference mission: navigate to a single ArUco marker |

### Hardware Interface
- `scam.py` – Camera (Picamera2 raw frames, 820×616). `setup_raw()` + `getRawFrame()` for direct capture. **Always returns 3-channel RGB** (strips the 4th channel from Picamera2's XBGR8888 format). All callers convert to BGR with `cv.cvtColor(img, cv.COLOR_RGB2BGR)`.
- `srobot.py`, `sedge.py`, `spose.py`, `simu.py`, `sir.py`, `sgpio.py` – Sensors/actuators

---

## Arena Layout — Final Ball Mission

```
        +--------+--------+
        |        |        |
        |   A    |   D    |
        |        |        |
        +--------+--------+
        |        |        |
        |   B    |   C    |
        |        |        |
        +--------+--------+
```

Each quadrant is **25 cm × 25 cm**. Quadrants share inner walls; each inner wall carries **two ArUco markers** (one per quadrant face), size **10 cm**, placed so their face normals cross the centre of their respective quadrant.

**ArUco IDs on inner walls (counter-clockwise ordering):**

| Quadrant | Left wall ID | Right wall ID |
|----------|-------------|---------------|
| A        | 10          | 11            |
| B        | 12          | 13            |
| C        | 14          | 15            |
| D        | 16          | 17            |

*Left/Right defined as seen from outside corner of the quadrant, looking toward the centre of the the entire square.*

**Delivery targets:**
- Red ball → **Quadrant A** (IDs 10, 11)
- Blue ball → **Quadrant C** (IDs 14, 15)

**ArUco geometry:** `marker_size_m = 0.10`. The goal point for entering a quadrant is computed as `standoff_m` in front of the marker face, using `aruco_to_robot_frame()` which returns `(x_r, y_r, face_nx, face_ny)`. Goal 40 cm in front: `(x_r + 0.40*face_nx, y_r + 0.40*face_ny)`.

---

## Servo Control — Ball Pickup and Delivery

The robot has a **front servo** (servo ID 1) that controls a ball holder/scoop at the front of the robot.

**MQTT command:** `service.send("robobot/cmd/T0", "servo 1 {position} {speed}")`
- `position`: negative = raised/up, positive = lowered/down, 0 = neutral/forward
- `speed`: rate of movement (higher = faster)

**Key positions (from mqtt-client.py):**
| Position | Meaning | When to use |
|----------|---------|-------------|
| `-800` | Fully raised (up) | Carrying a ball; also delivery release |
| `0` | Neutral/forward | Transit with no ball |
| `100` | Slightly down | Gentle scoop position |
| `500` | Fully lowered (down) | Active pickup scoop |

**Pickup sequence:**
1. Navigate to precisely 30 cm in front of the ball (fine-approach stops here)
2. Lower servo to scoop position: `servo 1 500 200`
3. Drive forward 30 cm so the ball is captured in the holder
4. Ball is now held; servo stays lowered until delivery

**Delivery sequence:**
1. Drive to precisely 30 cm in front of the quadrant entrance (ball holder tip inside opening)
2. Raise servo: `servo 1 -800 300` — ball releases into the quadrant

**Physical geometry — critical:**
- The robot body **never enters the quadrant** (quadrant is only 25 cm wide)
- Only the front ball holder needs to reach inside the quadrant opening
- The standoff distance from the ArUco marker face must be calibrated so that the ball holder tip is at or just inside the quadrant entrance when the robot stops
- The ArUco markers are on the **inner walls** (facing outward toward the approaching robot), so the marker face normal points toward the robot; `standoff_m` places the robot `standoff_m` metres from the wall in the approach direction

---

## final_ball_mission — Design Plan

### Ball Setup
- 4 balls knocked from a cup at mission start: 1 red, 1 blue, 2 white (ignored)
- Balls land randomly in front of the robot
- Servo starts in neutral position (`servo 1 0 0`)

### High-Level State Machine

```
SCAN_FOR_BALLS
  Servo neutral. Stop-and-look rotation until red and blue are localised.
  Also scan for any visible ArUco markers → use as initial pose correction.

PICKUP_RED
  Navigate to 30 cm in front of red ball (odometry + _ClassicalTracker + fine-approach).
  Lower servo: servo 1 500 200.
  Drive forward 30 cm to capture ball in holder.
  Confirm pickup: ball absent from camera for ≥ 3 s.

DELIVER_TO_A
  Navigate to Quadrant A entrance using odometry + ArUco landmark correction.
  Rotate to find ArUco ID 10 or 11; use aruco_to_robot_frame for precise positioning.
  Drive to 30 cm in front of quadrant wall (ball holder tip inside opening).
  Raise servo: servo 1 -800 300 → ball releases into quadrant.
  Confirm delivery: ball absent from camera.

PICKUP_BLUE
  Lower servo. Navigate to blue ball. Fine-approach. Raise servo.

DELIVER_TO_C
  Same as DELIVER_TO_A but target ArUco IDs 14/15.

RETURN_TO_START
  Servo neutral. Plan path back to (x0, y0).

DONE / FAILED
```

### ArUco as Localization Landmarks
Every ArUco detection during the mission gives an absolute robot-pose constraint:
- `aruco_to_robot_frame(marker, marker_size_m=0.10)` → `(x_r, y_r, face_nx, face_ny)` in robot frame
- Combined with the known world-frame position of the marker (derived from arena geometry + marker ID), this gives a corrected world-frame robot pose
- Use this to correct odometry drift before replanning
- Markers 10/11 fix the robot's position relative to Quadrant A; 14/15 relative to Quadrant C
- During delivery, ArUco is the primary (not just corrective) navigation target — the robot homes directly onto the marker using `aruco_to_robot_frame` rather than relying on the odometry world model

### White Balls
Ignored for now. Future: treat as `CircleObstacle` with radius = 0.025 m if classical CV picks them up.

---

## Debug Logging — Secondary Goal

All mission events should be written to a timestamped log file for post-mission replay and analysis.

**Inspired by:** `pathfinding/pathfind_logger.py` (event-based, space-separated, `#` comments), `CamVision/bucketballsmission.py` (per-frame vision state), `worldmodel/ball_world_model.py` (world-model state).

**Log events to capture:**

| Event token | Fields |
|-------------|--------|
| `BM_START` | start_x start_y heading goal_A_id goal_C_id |
| `BM_STATE` | new_state |
| `BM_POSE` | x y heading (periodic ~5 Hz) |
| `BM_BALL_DET` | color cx cy r_px x_r y_r wx wy det_count |
| `BM_ARUCO` | marker_id x_r y_r fnx fny rx ry hdg (each ArUco sighting) |
| `BM_POSE_CORRECT` | old_rx old_ry new_rx new_ry heading (after ArUco-based correction) |
| `BM_PATH` | n x1 y1 x2 y2 … |
| `BM_REPLAN` | reason eff_sx eff_sy n x1 y1 … |
| `BM_PICKUP` | color elapsed_s |
| `BM_DELIVER` | color quadrant |
| `BM_DONE` | 1\|0 elapsed_s |

Log files go to `MissionLogs/` (one file per run, timestamped). A replay script should be able to reconstruct the robot path, ball estimates, and ArUco sightings from the log alone.

---

## Camera Format — Critical Note

Picamera2 `capture_array()` returns **XBGR8888** (4 channels, stored in memory as `[R, G, B, X]` on ARM little-endian). `scam.py:getRawFrame()` strips the 4th channel, always returning 3-channel RGB. All downstream callers apply `cv.cvtColor(img, cv.COLOR_RGB2BGR)` to get BGR for OpenCV/YOLO. **Do not remove this conversion** in any caller.

---

## YOLO Detection — Critical Note: Stop-and-Look

**YOLO must be called on a stationary frame.** On a Raspberry Pi, YOLO inference takes ~300–500 ms. If the robot rotates while inference runs (~14° at 0.3 rad/s), the ball sweeps through the FOV in the gap between frames.

The working pattern (confirmed in `bucketballsmission.py`):
```
stop robot ("rc 0 0")
sleep 0.1s          ← mechanical settle
getRawFrame()       ← sharp image, no blur
YOLO inference      ← high-confidence detection
if not found: rotate pulse (rc 0 0.5 for 0.2s ≈ 5.7°), then loop
```

**Do not run YOLO in a background thread while the robot is moving.**

---

## Color Coding Conventions
- `'B'` = Blue ball, `'R'` = Red ball (single uppercase char throughout)
- YOLO class names are `'blue_ball'` / `'red_ball'`; mapped via `COLOR_MAP = {'B': 'blue_ball', 'R': 'red_ball'}` in `visualcontrol.py`
- White balls have no YOLO class and are ignored in the current implementation

---

## MQTT Protocol
- Send topic: `robobot/cmd/ti`
- Velocity command: `"rc {fwd_m_s:.2f} {turn_rad_s:.3f}"`
- Subscribe: `robobot/drive/#` for robot state (battery, motor feedback)
- MQTT host: `localhost`, port `1883`

---

## Pixel → Robot Coordinate Conversion (`CamVision/coord_conversion.py`)

**Balls:** `pixels_to_robot_coords([(cx, cy, r_px)])` → `[(x_r, y_r, z_r)]` in robot frame [m]. Uses known ball radius (R_REAL = 0.025 m) for depth. Accurate at close range (< 1 m); noisy at distance.

**ArUco:** `aruco_to_robot_frame(marker, marker_size_m=0.10)` → `(x_r, y_r, face_nx, face_ny)` or None. Uses solvePnP for accurate pose at navigation distances.

**Calibration constants:** `F_X=625.48`, `F_Y=622.74`, `C_X=406.11`, `C_Y=324.94`, `PHI=11°` (downward tilt), `C_ZR=0.18 m` (camera height). Camera horizontal half-FOV ≈ 33°.

**Frame conventions:** x_r = forward, y_r = left, z_r = up (robot frame).

---

## BallWorldModel (`worldmodel/ball_world_model.py`)

EMA-smoothed world-frame position estimates for tracked ball colors.
- `update(color, x_r, y_r, robot_x, robot_y, heading)` — fuses a robot-frame detection with odometry into a world-frame estimate (EMA α = 0.3)
- `all_reliable()` — True when every tracked color has ≥ `min_detections` (default 3)
- `absent_for(color)` — seconds since last detection (used for pickup detection)
- `mark_collected(color)` — exclude ball from future estimates
- `robot_to_world(x_r, y_r, rx, ry, hdg)` — utility exported for ArUco world-model conversion

---

## simulate_ball_mission — Key Patterns (Reference)

`pathfinding/simulate_ball_mission.py` contains the navigation scaffolding used as the basis for `final_ball_mission.py`:

- `_ClassicalTracker` — background thread running `localize_ball_lowest_contour()` at ~10 Hz; updates world model continuously while driving
- `_fine_approach()` — close-range P-control using `track_ball_window()` + `pixels_to_robot_coords()`; stops at `_STANDOFF_M = 0.30 m`
- `_rotation_scan()` — YOLO stop-and-look rotation up to 360° to relocate a lost target ball
- `_navigate_to_ball()` — two-phase nav: poll `nav_thread.join(timeout=0.5s)` for mid-segment interruption; outcomes: `close` | `replan` | `lost` | `done`
- Nav interruption via `graph_nav._stop_nav.set()` (threading.Event on `drive_to()`)
