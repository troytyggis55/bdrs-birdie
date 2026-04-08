# BDRS Birdie – Testing Guide

## Pre-run: Stop the camera stream server

The camera stream server must be killed before any high-FPS vision test, as it competes for camera access.

```bash
sudo pkill -f stream_server
```

One of these will error if the process wasn't running — that's fine. Do **not** kill the stream just to take pictures; only do this before visual servoing / control loop tests.

## Run the ball mission

```bash
python3 mqtt-client.py -bbm B    # blue ball
python3 mqtt-client.py -bbm R    # red ball
```

Accepts `R`, `r`, `B`, or `b`.

## Expected output (nominal)

```
% Ball found! Locking on.
% FPS: XX.X | Loop: XX.X ms
% Aligned! Driving to ball...
% MISSION COMPLETE: Ball Reached.
```

## Known bug (as of 2026-04-08)

YOLO detects the ball but the robot keeps spinning. Terminal will print:

```
% Nothing here. Incrementing search...
% Nothing here. Incrementing search...
```

See `CLAUDE.md` for root cause and fix.

## Debug output

Still frames are saved to `VisionOutput/Debug_mission/` every 10 frames while in the aligning and tracking states (side-by-side: annotated frame | color mask).

> **Note:** No frames are saved during STATE 0 (localization/spin). If the bug is present, the folder will be empty after a run.

Full video recording is available via `bucketballsmissionwithRec()` (swap the call in `mqtt-client.py:383`), but it drops FPS enough to break the tracker — use only for post-run visualization, not real tests.
