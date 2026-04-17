# BDRS Birdie – Vision & Control Loop Reference


## Project Overview
Two-wheeled robot that uses hybrid YOLO + classical vision to detect colored balls (red/blue) and drive toward them. MQTT is the communication layer between the Python vision controller and the Teensy hardware.

**Current main goal:** `final_ball_mission.py` — pick up a red ball and deliver it to quadrant A, pick up a blue ball and deliver it to quadrant C, then return to start. ArUco markers on the quadrant walls serve as localization landmarks throughout.

## TODOs
- The "plus" is off by 45° for some reason. Right now the plus walls seem to be parallel to the outer perimeter walls, but they should be rotated 45° so they are parallel to the A–C and B–D axes.
- [] The goal nav standoff is supposed to be at the best relative position, but its honestly quite bad. Probably still better than arbitrary fixed standoff, but it could with another simple heuristic like just shortest path to the ball.
- [] We need more image frame logging. Right now its only for yolo and classical during pathinding. I want it to log for basically every part of the mission.
- [] The yolo might still catch onto noise like the lower gripper or other things in the distance. Maybe we need to crop the image fed to the yolo, or ignore values outside a given fov margin.
- [] We need a safe reverse routine after delivering the ball into the quadrant, so we dont rotate and crash into the wall.

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
| `ball_mission/final_ball_mission.py` | **Primary mission** — deliver closest ball first, then second; red→A, blue→C |
| `ball_mission/arena_walls.py` | `ArenaWallModel` — EMA wall model from ArUco; `walls_from_aruco_world()` geometry |
| `ball_mission/final_ball_logger.py` | `FinalBallLogger` — timestamped event logger; saves frames to `MissionLogs/` |
| `ball_mission/final_ball_replay.py` | Replay script: reconstruct path/ball/ArUco from log file |
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

The entire square is 60cm x 60cm.
Each quadrant is **30 cm × 30 cm**. Quadrants share inner walls; each inner wall carries **two ArUco markers** (one per quadrant face), size **10 cm**, placed so their face normals cross around the centre of their respective quadrant.
The center of each ArUco marker is 17cm from the inner corner of the quadrant, meaning they are 13 cm from the outer corner that the robot approaches from.
**ArUco IDs on inner walls (counter-clockwise ordering):**

| Quadrant | Left wall ID | Right wall ID |
|----------|-------------|---------------|
| A        | 10          | 11            |
| B        | 12          | 13            |
| C        | 14          | 15            |
| D        | 16          | 17            |

*Left/Right defined as seen from outside corner of the quadrant, looking toward the centre of the the entire square.*

### Mapping the square walls based on ArUco
We can reconstruct the inner “plus” walls from a single marker pose. For ID 10 (left wall of A), let `(nx, ny)` 
be the marker face normal and let the along-wall direction be `(tx, ty) = (-ny, nx)`. 
Draw the 60cm wall along `(tx, ty)` from `P − 0.13·t` to `P + 0.47·t` (P = marker center). 
Then draw the perpendicular inner wall along `(nx, ny)`, centered at `P + 0.17·t`, spanning `±0.30m` (i.e. from `C − 0.30·n` to `C + 0.30·n`).
This is mirrored for the right wall walls, but the pattern is the same for all quadrants. This way, any ArUco sighting gives us the full arena geometry and our position within it.

To achieve best accuracy, this mapping should be done continously with the latest ArUco sighting based of the closest marker to the robot.

**Plus-center:** The intersection of the two inner walls (center of the 60×60 arena) can be computed from any marker as `P + 0.17·t` (using the marker-appropriate t direction).  `ArenaWallModel.get_plus_center()` averages this across all qualified markers.

### Outer perimiter walls and bucket mapping

Approximate perimeter drawing:

```
                        |
                \A/     |
    /-\        B X D    |   
    | |         /C\     |
    \-/                 |
                        |
                        |
-------------------------
```

C-Wall (bottom): Runs perpendicular to the A–C axis, at 2.55m from the plus center in the C direction (i.e. directly away from quadrant C's outer corner). The wall is 1.75m long, spanning from 0.65m to the left of the plus center (into the B/C side) to 1.10m to the right of the plus center (into the C/D side). In other words, it covers the full bottom of the arena, slightly asymmetrically centered — the plus center projects onto the wall at 0.65m from its left end.
D-Wall (right): Runs parallel to the A–C axis, at 1.04 to the right of the plus center in the D direction (i.e. directly away from quadrant D's outer corner). It forms the right boundary of the arena. Its centered with the plus sign and is 3m long.
Bucket: A circular obstacle positioned 1.02m to the left of the plus center (deep into the B side) and 0.64m downward (toward C). This places it roughly aligned with the left end of the C-wall and below the B/C quadrant boundary — in the open space to the lower-left of the arena square.


### Ball Approach Strategy

**1. ArUco ID filter**
Only markers with IDs 10–17 (the 8 inner-wall markers) are logged and used for wall mapping. All others are silently ignored. Implemented as `_is_arena_aruco()` in `final_ball_mission.py`.

**2. Clearance-maximising approach point**
`_best_approach_point()` samples 36 directions around the ball and picks the standoff point (`_NAV_STANDOFF_M = 0.4 m`) with maximum clearance from walls, other balls, and the bucket. Falls back to the robot-relative direction when no obstacles are present.

**3. Rotate-to-face before fine approach**
After navigation reaches the approach point, `_rotate_to_face_ball()`:
1. Computes bearing to the ball's last known world position and rotates open-loop.
2. Takes one stationary YOLO scan to confirm.
3. Hands off to `_fine_approach()` (classical P-control to 30 cm).

**4. Classical tracker guards**

*FOV edge guard:* If the tracked ball's pixel centre is within `_TRACKING_FOV_X_MARGIN = 150 px` of the left/right edge, `_ClassicalTracker` suspends world-model updates and sets `_needs_yolo`. The main loop must stop the robot, run a YOLO scan, and call `revalidate()` before classical tracking resumes.

*Jump guard:* If a new world-frame estimate would shift the ball position by more than `_JUMP_THRESHOLD_M = 0.30 m`, the tracker sets `_jump_detected` and goes dormant (stops processing frames). The main loop stops navigation, runs one YOLO scan, calls `resolve_jump()`, then resumes the current waypoint. A full replan is triggered only if the YOLO-confirmed position differs from the original plan by more than `_REPLAN_DIST_M`.

**Delivery targets:**
- Red ball → **Quadrant A** (IDs 10, 11)
- Blue ball → **Quadrant C** (IDs 14, 15)

**Second-ball navigation:** `_navigate_to_second_ball()` drives to the last-known world position. When within `_YOLO_CONFIRM_DIST_M = 1.0 m`, it stops, runs `_rotate_to_face_ball()` + YOLO confirm, then hands off to `_navigate_to_ball()`. Falls back to a full `_rotation_scan()` if YOLO fails.

**ArUco geometry:** `marker_size_m = 0.10`. Delivery goal: `_DELIVERY_STANDOFF_M = 0.35 m` in front of the ArUco marker face normal, computed via `aruco_to_robot_frame()` → `(x_r, y_r, face_nx, face_ny)`.

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

## final_ball_mission — Implementation

### Ball Setup
- 4 balls knocked from a cup at mission start: 1 red, 1 blue, 2 white (ignored)
- Balls land randomly in front of the robot
- Servo starts in neutral position (`servo 1 0 0`)

### High-Level State Machine

```
SCAN
  Servo neutral. YOLO stop-and-look pulses until red and blue are localised.
  Passive ArUco scan on every stopped frame → wall model.
  After both balls found (or timeout), pick the closest one first.

NAVIGATE_TO_FIRST_BALL
  _navigate_to_ball(first_color, second_color, world_model)
  _ClassicalTracker + _PassiveArucoScanner run in background.
  Fine approach stops at 30 cm.

WAIT_FIRST_PICKUP
  Lower servo: servo 1 500 200.
  Wait until ball absent for ≥ _PICKUP_ABSENT_S (3 s).
  mark_collected(first_color).

DELIVER_FIRST
  _navigate_to_delivery(first_color, world_model)
  _PassiveArucoScanner refines delivery goal continuously.
  Raise servo: servo 1 -800 300 → ball releases.

NAVIGATE_TO_SECOND_BALL
  Servo neutral (servo 1 0 0).
  _navigate_to_second_ball(second_color, world_model)
  Drives to last-known position → stops at 1 m → YOLO confirm → full approach.

WAIT_SECOND_PICKUP / DELIVER_SECOND
  Same as first ball.

RETURN_TO_START
  Servo neutral. _plan_and_drive((x0, y0), []).

DONE / FAILED
```

### ArUco as Localization
ArUco sightings are accumulated passively throughout the mission via `_PassiveArucoScanner` (background thread while moving) and `_scan_aruco_passive()` (inline on stopped frames). Detections feed `ArenaWallModel` which provides:
- Inner wall segments for pathfinding obstacle avoidance
- Plus-center estimate for perimeter wall/bucket positioning
- Delivery goal positions (average of target marker positions + face normal standoff)

ArUco does **not** directly correct odometry pose in the current implementation; it informs obstacle geometry and delivery goals instead.

### White Balls
Ignored for now. Future: treat as `CircleObstacle` with radius = 0.025 m if classical CV picks them up.

---

## Debug Logging — Secondary Goal

All mission events should be written to a timestamped log file for post-mission replay and analysis.

**Inspired by:** `pathfinding/pathfind_logger.py` (event-based, space-separated, `#` comments), `CamVision/bucketballsmission.py` (per-frame vision state), `worldmodel/ball_world_model.py` (world-model state).

**Log events (implemented in `FinalBallLogger`):**

| Event token | Fields |
|-------------|--------|
| `BM_START` | x0 y0 heading |
| `BM_STATE` | state_name |
| `BM_POSE` | x y heading (periodic, `_POSE_LOG_INTERVAL_S = 0.20 s`) |
| `BM_BALL_DET` | color cx cy r_px x_r y_r wx wy det_count |
| `BM_ARUCO` | marker_id x_r y_r fnx fny rx ry hdg |
| `BM_POSE_CORRECT` | old_rx old_ry new_rx new_ry heading |
| `BM_PATH` | n x1 y1 x2 y2 … |
| `BM_REPLAN` | reason eff_sx eff_sy n x1 y1 … |
| `BM_PICKUP` | color elapsed_s |
| `BM_DELIVER` | color quadrant |
| `BM_DONE` | 1\|0 elapsed_s |
| `BM_FRAME` | filename label (annotated JPEG saved to `<logname>_frames/`) |

Log files go to `MissionLogs/` (one per run, timestamped). Annotated frames are saved every `_IMG_LOG_EVERY_N = 2` classical-tracker iterations and on FOV-edge events. Replay: `python ball_mission/final_ball_replay.py [log_path]`.

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

## final_ball_mission — Key Internal Patterns

These patterns are all implemented in `ball_mission/final_ball_mission.py`:

- `_ClassicalTracker` — background thread running `localize_ball_lowest_contour()` at ~10 Hz while driving; updates world model; has FOV-edge guard (`_needs_yolo`) and jump guard (`_jump_detected`) — both pause the tracker until the main loop resolves them
- `_PassiveArucoScanner` — background thread running `detect_aruco()` at ~10 Hz while driving; feeds `ArenaWallModel` and logs all arena markers; purely observational
- `_fine_approach()` — close-range P-control using `track_ball_window()` + `pixels_to_robot_coords()`; stops at `_FINE_STANDOFF_M = 0.30 m`
- `_rotation_scan()` — YOLO stop-and-look rotation up to 360° to relocate a lost target ball
- `_navigate_to_ball()` — two-phase nav: poll `nav_thread.join(timeout=_POLL_INTERVAL_S)` for mid-segment interruption; outcomes: `done` (→ fine approach) | `replan` (→ next cycle); nav interruption via `graph_nav._stop_nav.set()`
- `_navigate_to_delivery()` — same waypoint loop but goal is refined by `_delivery_goal()` as new ArUco markers are seen; replans when goal shifts > `_REPLAN_DIST_M`

`pathfinding/simulate_ball_mission.py` is an older version kept for reference only.
