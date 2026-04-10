# BDRS Birdie – Vision & Control Loop Reference

## Project Overview
Two-wheeled robot that uses hybrid YOLO + classical vision to detect colored balls (red/blue) and drive toward them. MQTT is the communication layer between the Python vision controller and the Teensy hardware.

---

## Architecture

### Entry Points
- `mqtt-client.py` – Main dispatcher, starts `bucketballsmission()` via CLI args
- `uservice.py` – MQTT service framework (two clients: RX + TX)

### Vision Pipeline
| File | Role |
|------|------|
| `CamVision/bucketballsmission.py` | **Main control loop + state machine** |
| `CamVision/visualcontrol.py` | Classical HSV masking + contour detection + YOLO wrapper for localization + pixel→robot coord conversion |
| `CamVision/bucketballsyolo.py` | Ultralytics YOLO wrapper (`get_labels()`) |
| `CamVision/bucketballsncnn.py` | Alternative NCNN backend (unused in main loop) |
| `CamVision/pictures.py` | Image I/O, video recording |

### Hardware Interface
- `scam.py` – Camera (Picamera2 raw frames, 820×616 RGB)
- `srobot.py`, `sedge.py`, `spose.py`, `simu.py`, `sir.py`, `sgpio.py` – Sensors/actuators

---

## State Machine (`bucketballsmission.py:47`)

```
STATE 0  LOCALIZATION  (discrete search)
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

## Pixel → Robot Coordinate Conversion (`visualcontrol.py`)

Function `pixel_to_robot_coords(x_pixel, y_pixel, r_pixel)` converts ball detections to `(x_r, y_r, z_r)` in the robot frame (meters). Calibrated and working.

**Pipeline:** depth from known ball radius → 3D camera coords → rigid transform (axis swap + pitch rotation + mounting offset translation) to robot frame.

---

## Debug Output
- Debug frames saved to `VisionOutput/Debug_mission/`
- Every 10 frames in states 1 & 2 (side-by-side: annotated frame | color mask)
- `bucketballsmissionwithRec()` records video but kills FPS – do not use for real runs
