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
| `CamVision/visualcontrol.py` | Classical HSV masking + contour detection + YOLO wrapper for localization |
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

## Critical Bug – Robot Keeps Spinning Despite YOLO Detection

### Root Cause: Class Name Mismatch (`visualcontrol.py:148`)

```python
# BUG: color is passed as 'B' or 'R' (single char)
# but YOLO model class names are e.g. 'blue_ball' / 'red_ball'
if cls_name == color:   # "blue_ball" == "B"  → always False!
```

This means `localize_ball_yolo()` **always returns `(False, None)`**, so the robot never exits STATE 0 and spins indefinitely.

The code's own comment on line 143 reads:
```python
# Get class name (e.g., 'red_ball' or 'blue_ball')
```
...confirming the mismatch.

### Fix
**First, verify the actual class names** the model was trained with by running:
```python
from ultralytics import YOLO
m = YOLO("CamVision/YoloModels/Model01/my_model/my_model.pt")
print(m.names)   # e.g. {0: 'blue_ball', 1: 'red_ball'}
```

Then fix `visualcontrol.py:148` to match. Options:

```python
# Option A – map single char to full class name (safest, explicit)
COLOR_MAP = {'B': 'blue_ball', 'R': 'red_ball'}
if cls_name == COLOR_MAP.get(color.upper()):

# Option B – partial match
if cls_name.lower().startswith(color.lower()):
```

---

## Secondary Issues (lower priority)

| Issue | Location | Detail |
|-------|----------|--------|
| Spin step is 5.7°, comment says ~45° | `bucketballsmission.py:106` | `0.5 rad/s × 0.2s = 0.1 rad`. To rotate ~45° use `t.sleep(1.57)` |
| No search timeout | `bucketballsmission.py:85` | Spins forever if ball absent. Add rotation counter or time limit |
| STATE 1→2 only checks error_x | `bucketballsmission.py:120` | Doesn't verify ball is close; could transition with a distant ball |
| Derivative reset on re-acquisition | `bucketballsmission.py:188` | `prev_error_y = 0` causes a derivative spike on first frame after re-lock |

---

## MQTT Protocol
- Send topic: `robobot/cmd/ti`
- Velocity command: `"rc {fwd_m_s:.2f} {turn_rad_s:.3f}"`
- Subscribe: `robobot/drive/#` for robot state (battery, motor feedback)
- MQTT host: `localhost`, port `1883`

---

## Color Coding Conventions
- `'B'` = Blue ball, `'R'` = Red ball (single uppercase char throughout)
- `localize_ball_lowest_contour()` and `track_ball_window()` handle `'B'`/`'R'` correctly
- `localize_ball_yolo()` does **not** – see bug above

---

## Debug Output
- Debug frames saved to `VisionOutput/Debug_mission/`
- Every 10 frames in states 1 & 2 (side-by-side: annotated frame | color mask)
- `bucketballsmissionwithRec()` records video but kills FPS – do not use for real runs
