# BDRS Birdie – Vision & Control Loop Reference

## Project Overview
Two-wheeled robot that uses hybrid YOLO + classical vision to detect colored balls (red/blue) and drive toward them. MQTT is the communication layer between the Python vision controller and the Teensy hardware.

---

## Architecture

### Entry Points
- `mqtt-client.py` – Main dispatcher; CLI args select which mission to run
- `uservice.py` – MQTT service framework (two clients: RX + TX)

### Vision Pipeline
| File | Role |
|------|------|
| `CamVision/bucketballsmission.py` | Single-ball control loop + state machine (align & drive to one ball) |
| `CamVision/visualcontrol.py` | Classical HSV masking + contour detection + YOLO wrapper for localization |
| `CamVision/bucketballsyolo.py` | Ultralytics YOLO wrapper (`get_labels()`) |
| `CamVision/ballcoords.py` | Pixel→robot-frame 3D coordinate conversion (calibrated camera intrinsics + mounting geometry) |
| `CamVision/bucketballsncnn.py` | Alternative NCNN backend (unused in main loop) |
| `CamVision/pictures.py` | Image I/O, video recording |

### Missions
| File | Role |
|------|------|
| `CamVision/bucketballsmission.py` | `bucketballsmission(color)` — align and drive to a single ball |
| `pathfinding/simulate_ball_mission.py` | `simulate_ball_mission()` — scan for both balls, approach red, wait for pickup, deliver to goal, repeat for blue |

### Pathfinding & Navigation
| File | Role |
|------|------|
| `pathfinding/pathfinding.py` | `CircleObstacle`, `PlannerConfig`, `find_safe_start` — RRT-style planner |
| `pathfinding/realtime_pathfind.py` | `RealtimePathfinder` — replanning wrapper |
| `odometry/graph_nav.py` | `graph_nav.drive_to()` — waypoint follower using odometry |
| `worldmodel/ball_world_model.py` | `BallWorldModel` — EMA-smoothed world-frame ball position tracking |

### ArUco Detection
| File | Role |
|------|------|
| `aruco/detect_aruco.py` | `detect_aruco(frame)` — OpenCV ArUco marker detection |
| `aruco/aruco_coords.py` | Pixel→robot-frame coordinate conversion for markers |
| `aruco/go_to_aruco_mission.py` | Mission: drive to an ArUco marker |

### Hardware Interface
- `scam.py` – Camera (Picamera2 raw frames, 820×616). `setup_raw()` + `getRawFrame()` for direct capture. **Always returns 3-channel RGB** (strips the 4th channel from Picamera2's XBGR8888 format). All callers convert to BGR with `cv.cvtColor(img, cv.COLOR_RGB2BGR)`.
- `srobot.py`, `sedge.py`, `spose.py`, `simu.py`, `sir.py`, `sgpio.py` – Sensors/actuators

---

## Camera Format — Critical Note

Picamera2 `capture_array()` returns **XBGR8888** (4 channels, stored in memory as `[R, G, B, X]` on ARM little-endian). `scam.py:getRawFrame()` strips the 4th channel, always returning 3-channel RGB. All downstream callers apply `cv.cvtColor(img, cv.COLOR_RGB2BGR)` to get BGR for OpenCV/YOLO. **Do not remove this conversion** in any caller.

---

## YOLO Detection — Critical Note: Stop-and-Look

**YOLO must be called on a stationary frame.** On a Raspberry Pi, YOLO inference takes ~300–500 ms. If the robot rotates while inference runs (~14° at 0.3 rad/s), the ball sweeps through the FOV in the gap between frames and will not be detected.

The working pattern (from `bucketballsmission.py` state 0, confirmed working):
```
stop robot ("rc 0 0")
sleep 0.1s          ← mechanical settle
getRawFrame()       ← sharp image, no blur
YOLO inference      ← high-confidence detection
if not found: rotate pulse (rc 0 0.5 for 0.2s ≈ 5.7°), then loop
```

**Do not run YOLO in a background thread while the robot is moving.** Detections will arrive late, sporadically, and mostly as false positives from old frames.

---

## bucketballsmission State Machine (`bucketballsmission.py:72`)

```
STATE 0  LOCALIZATION  (discrete stop-and-look)
  Every 5th frame:
    → stop robot ("rc 0 0")
    → sleep 0.1s
    → grab fresh frame
    → localize_ball_yolo(img, color=target_color)   ← YOLO ONLY
    If found:  last_rect = ball['rect'], state = 1
    If not:    "rc 0 0.5" for 0.2s  (≈ 5.7° rotation), then loop

STATE 1  ALIGNING  (horizontal centering, classical tracker)
  Every frame:
    → track_ball_window(img, prev_window=last_rect)
    error_x = 410 - cX
    turn_vel = Kp_turn * error_x  (P-only, clamped ±0.5)
    If |error_x| ≤ 3px  → state = 2
    If lost             → state = 0

STATE 2  TRACKING  (forward + fine-tune, classical tracker)
  Every frame:
    → track_ball_window(img, prev_window=last_rect)
    error_y = 430 - cY,  error_x = 410 - cX
    fwd_vel  = Kp_fwd * error_y + Kd_fwd * derivative_y  (clamped 0–0.3)
    turn_vel = Kp_turn * error_x  (clamped ±0.5)
    If |error_y| ≤ 10 AND |error_x| ≤ 3  → MISSION COMPLETE
    If lost  → state = 0
```

**Key parameters:**
- Image center X: 410 px, Target Y: 430 px
- Kp_turn = 0.010, Kp_fwd = 0.020, Kd_fwd = 0.050
- Model: `CamVision/YoloModels/Model01/my_model/my_model.pt`
- YOLO confidence threshold: 0.5

---

## simulate_ball_mission State Machine (`pathfinding/simulate_ball_mission.py`)

```
SCAN
  loop:
    stop → sleep 0.1s → grab frame → YOLO both colors → update BallWorldModel
    if both balls reliable (≥3 detections each): → NAVIGATE_TO_RED
    else: rotate pulse (rc 0 0.5 for 0.2s) → loop

NAVIGATE_TO_RED
  Plan path to (red_pos − 20 cm standoff), avoiding blue as obstacle
  Drive path via graph_nav → WAIT_RED_PICKUP

WAIT_RED_PICKUP
  stop; loop:
    grab frame → YOLO red → update world model
    if red absent ≥ 3s: mark collected → DELIVER_RED
    if timeout (30s): FAILED

DELIVER_RED
  Plan path to goal (2 m ahead of start), avoiding blue → NAVIGATE_TO_BLUE

NAVIGATE_TO_BLUE / WAIT_BLUE_PICKUP / DELIVER_BLUE
  Mirror of red sequence, no obstacles
```

**Key parameters:**
- `_GOAL_DIST_M = 2.0` m ahead of start
- `_STANDOFF_M = 0.20` m in front of ball
- `_PICKUP_ABSENT_S = 3.0` s absence to count as collected
- `_DEFAULT_MIN_DETECTIONS = 3` in BallWorldModel before position is reliable

---

## Color Coding Conventions
- `'B'` = Blue ball, `'R'` = Red ball (single uppercase char throughout)
- YOLO class names are `'blue_ball'` / `'red_ball'`; mapped via `COLOR_MAP = {'B': 'blue_ball', 'R': 'red_ball'}` in `visualcontrol.py`

---

## MQTT Protocol
- Send topic: `robobot/cmd/ti`
- Velocity command: `"rc {fwd_m_s:.2f} {turn_rad_s:.3f}"`
- Subscribe: `robobot/drive/#` for robot state (battery, motor feedback)
- MQTT host: `localhost`, port `1883`

---

## Pixel → Robot Coordinate Conversion (`CamVision/ballcoords.py`)

Function `pixels_to_robot_coords(detections)` converts a list of `(x_px, y_px, r_px)` detections to `(x_r, y_r, z_r)` in the robot frame (meters). Calibrated and working.

**Pipeline:** undistort pixel coords → depth from known ball radius → 3D camera coords → rigid transform (axis swap + pitch rotation + mounting offset translation) to robot frame.

**Calibration constants** (top of `ballcoords.py`): `F_X`, `F_Y`, `C_X`, `C_Y`, `DIST_COEFFS`, `R_REAL = 0.025 m`, `PHI = 11°`, `C_ZR = 0.18 m`.

The original `pixel_to_robot_coords()` that was in `visualcontrol.py` has been superseded by this module.

---

## BallWorldModel (`worldmodel/ball_world_model.py`)

EMA-smoothed world-frame position estimates for tracked ball colors.
- `update(color, x_r, y_r, robot_x, robot_y, heading)` — fuses a robot-frame detection with odometry into a world-frame estimate
- `all_reliable()` — True when every tracked color has ≥ `min_detections` (default 3)
- `absent_for(color)` — seconds since last detection (used for pickup detection)
- `mark_collected(color)` — exclude ball from future estimates

---

## Debug Output
- Debug frames saved to `VisionOutput/Debug_mission/`
- Every 10 frames in states 1 & 2 (side-by-side: annotated frame | color mask)
- `bucketballsmissionwithRec()` records video but kills FPS – do not use for real runs
