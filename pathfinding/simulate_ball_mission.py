"""
Simulated ball delivery mission.

Setup
-----
- A virtual goal zone is placed 2 m ahead of the robot at start.
- A red ball and a blue ball are placed (by hand) between the robot and the goal.

Sequence
--------
1. SCAN  — stop-and-look scan: stop, grab fresh frame, run YOLO for both
           colors, rotate a small step, repeat until both balls are localised.
2. Approach red ball (blue = pathfinding obstacle), stop 20 cm in front.
3. WAIT  — camera watches; when red ball disappears for 3 s the human has
           "picked it up".  Robot drives to goal zone (blue still an obstacle).
4. Approach blue ball (no obstacles), stop 20 cm in front.
5. WAIT  — same pickup logic. Robot drives to goal zone. Mission complete.

Run via mqtt-client.py by adding a --simulate-ball flag (or call
simulate_ball_mission() directly).
"""

import cv2 as cv
import math
import sys
import threading
import time as t
from enum import Enum, auto
from pathlib import Path

# Make repo root importable regardless of working directory.
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from pathfinding.pathfinding import CircleObstacle, PlannerConfig, find_safe_start
from pathfinding.realtime_pathfind import RealtimePathfinder

from worldmodel.ball_world_model import BallWorldModel
from CamVision.visualcontrol import (localize_ball_yolo, localize_ball_lowest_contour,
                                     track_ball_window)
from CamVision.ballcoords import pixels_to_robot_coords, C_X as _CAM_CX
from scam import cam
from spose import pose
from uservice import service
from odometry.graph_nav import graph_nav

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

_CFG = PlannerConfig(
    delta=0.15,
    goal_tolerance=0.12,
    clearance=0.00,
    robot_radius=0.12,
    max_steps=3000,
    smooth_path=True,
)

_GOAL_DIST_M      = 2.00   # goal zone distance ahead of start [m]
_GOAL_RADIUS_M    = 0.15   # goal zone radius (30 cm diameter) [m]
_STANDOFF_M       = 0.30   # stop this far in front of a ball [m]
_OBS_BALL_RADIUS  = 0.08   # obstacle radius used in pathfinder for balls [m]
_PICKUP_ABSENT_S  = 3.0    # ball must be absent this long to count as picked up [s]
_PICKUP_TIMEOUT_S = 30.0   # max wait for human pickup before aborting [s]
# Scan step: same pulse used in bucketballsmission (0.5 rad/s for 0.2 s ≈ 5.7°)
_SCAN_PULSE_VEL   = 0.5    # turn rate during each scan pulse [rad/s]
_SCAN_PULSE_S     = 0.2    # duration of each scan pulse [s]
_MISSION_TIMEOUT_S = 120.0 # hard safety timeout for entire mission [s]
_MAX_REPLAN_CYCLES = 5     # safety cap: give up after this many replan cycles per leg
_POLL_INTERVAL_S   = 0.50  # how often the nav loop checks tracker updates mid-segment [s]
_REPLAN_DIST_M     = 0.05  # world-model shift (m) that triggers a mid-drive replan
_LOST_BALL_TIMEOUT_S = 5.0 # seconds without a tracker detection before rotation scan [s]
# Horizontal half-FOV ≈ atan(image_width/2 / F_X) = atan(410/625) ≈ 33°
_CAM_FOV_HALF_RAD  = math.radians(33)
_CLOSE_RANGE_M     = 0.70  # world-model dist at which we switch to classical fine-approach [m]
_KP_TURN_FINE      = 0.010 # fine-approach turn gain  [rad/s per px]  — same as bbm
_KP_FWD_FINE       = 0.30  # fine-approach forward gain [m/s per m of distance error]
_FINE_ALIGN_TOL_PX = 5     # fine-approach stop: max horizontal pixel error [px]
_FINE_DIST_TOL_M   = 0.03  # fine-approach stop: max distance error [m]
_SCAN_START_TIMEOUT_S = 4.0  # start navigation after this scan time if red is found [s]
_SCAN_MIN_RED_DETECTIONS = 1
_SCAN_MIN_BLUE_DETECTIONS = 1
_NAV_SEGMENT_STEP_M = 0.25  # split long planner segments so scan/replan checks happen during travel


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pos() -> tuple[float, float]:
    return pose.pose[0], pose.pose[1]


def _heading() -> float:
    return pose.pose[2]


def _approach_point(ball_x: float, ball_y: float,
                    robot_x: float, robot_y: float,
                    standoff: float = _STANDOFF_M) -> tuple[float, float]:
    """Return the point *standoff* metres behind the ball on the robot→ball line."""
    dx = ball_x - robot_x
    dy = ball_y - robot_y
    dist = math.hypot(dx, dy)
    if dist < 1e-3:
        return ball_x, ball_y
    return ball_x - standoff * dx / dist, ball_y - standoff * dy / dist


def _stop() -> None:
    service.send("robobot/cmd/ti", "rc 0 0")


def _led(r: int, g: int, b: int) -> None:
    service.send("robobot/cmd/T0", f"leds 16 {r} {g} {b}")


def _grab_bgr():
    """Grab one fresh frame from the camera. Returns (ok, bgr_img)."""
    ok, img, _ = cam.getRawFrame()
    if not ok or img is None:
        return False, None
    return True, cv.cvtColor(img, cv.COLOR_RGB2BGR)


def _angle_wrap(a: float) -> float:
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


def _ball_in_fov(ball_wx: float, ball_wy: float) -> bool:
    """True if the ball's last known world position falls within the camera's horizontal FOV."""
    rx, ry = _pos()
    bearing = math.atan2(ball_wy - ry, ball_wx - rx)
    return abs(_angle_wrap(bearing - _heading())) < _CAM_FOV_HALF_RAD


class _ClassicalTracker:
    """
    Background thread that runs localize_ball_lowest_contour() at ~10 Hz while
    the robot drives, continuously updating the world model with fresh detections.

    Full-frame HSV detection (not track_ball_window) is used here so the ball
    can drift across the image freely as the robot moves without losing the window.

    Usage:
        tracker = _ClassicalTracker('R', world_model)
        tracker.start()
        ...
        if tracker.pop_detected():   # True once per detection burst
            replan()
        ...
        tracker.stop()
    """
    _INTERVAL_S = 0.10   # seconds between frames (~10 Hz)

    def __init__(self, target_color: str, world_model: BallWorldModel):
        self._color = target_color
        self._wm = world_model
        self._stop_event = threading.Event()
        self._detected = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._detected.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def pop_detected(self) -> bool:
        """Return True if at least one detection occurred since the last call; resets the flag."""
        if self._detected.is_set():
            self._detected.clear()
            return True
        return False

    def _run(self) -> None:
        while not self._stop_event.is_set():
            ok, img = _grab_bgr()
            if not ok:
                t.sleep(self._INTERVAL_S)
                continue

            found, ball_data = localize_ball_lowest_contour(img, self._color)
            if found:
                cx, cy = ball_data['center']
                _, _, w, h = ball_data['rect']
                r_px = min(w, h) / 2.0
                coords = pixels_to_robot_coords([(cx, cy, r_px)])
                if coords:
                    x_r, y_r, _ = coords[0]
                    rx, ry = _pos()
                    hdg = _heading()
                    self._wm.update(self._color, x_r, y_r, rx, ry, hdg)
                    self._detected.set()
                    print(f"% Tracker [{self._color}]: "
                          f"robot=({x_r:.2f},{y_r:.2f})  "
                          f"world=({self._wm.get(self._color).x:.2f},"
                          f"{self._wm.get(self._color).y:.2f})")

            t.sleep(self._INTERVAL_S)


def _detect_and_update(img, world_model: BallWorldModel) -> None:
    """
    Run YOLO for both ball colors on *img* and update the world model.
    Mirrors the per-frame logic from bucketballsmission.
    """
    rx, ry = _pos()
    hdg = _heading()
    for color in ('R', 'B'):
        ball = world_model.get(color)
        if ball and ball.collected:
            continue
        found, det = localize_ball_yolo(img, color)
        if not found:
            continue
        cx, cy = det['center']
        _, _, w, h = det['rect']
        r_px = min(w, h) / 2.0
        coords = pixels_to_robot_coords([(cx, cy, r_px)])
        if not coords:
            continue
        x_r, y_r, _ = coords[0]
        world_model.update(color, x_r, y_r, rx, ry, hdg)
        est = world_model.get(color)
        print(f"% Vision: {color} detected  center=({cx},{cy})  "
              f"robot=({x_r:.2f},{y_r:.2f})  count={est.detection_count}")


def _detect_color(img, color: str, world_model: BallWorldModel) -> bool:
    """
    Run YOLO for a single *color* and update the world model.
    Returns True if the ball was found.
    """
    ball = world_model.get(color)
    if ball and ball.collected:
        return False
    found, det = localize_ball_yolo(img, color)
    if not found:
        return False
    cx, cy = det['center']
    _, _, w, h = det['rect']
    r_px = min(w, h) / 2.0
    coords = pixels_to_robot_coords([(cx, cy, r_px)])
    if not coords:
        return False
    x_r, y_r, _ = coords[0]
    rx, ry = _pos()
    hdg = _heading()
    world_model.update(color, x_r, y_r, rx, ry, hdg)
    return True


def _fine_approach(target_color: str, world_model: BallWorldModel) -> bool:
    """
    Classical-CV tight control loop for the final ~_CLOSE_RANGE_M to the ball.

    Uses track_ball_window() for per-frame tracking (no YOLO inference lag) and
    pixels_to_robot_coords() for camera-derived distance, which is most reliable
    at close range where coordinate inaccuracies are minimal.

    Stops when the robot is _STANDOFF_M ± _FINE_DIST_TOL_M in front of the ball
    and centred within _FINE_ALIGN_TOL_PX pixels.
    """
    print(f"% SimBallMission: fine approach to {target_color}")
    _stop()
    t.sleep(0.1)

    # Seed the tracking window with a full-frame classical (HSV) detection.
    # At close range the colour mask is reliable and avoids a costly YOLO call.
    ok, img = _grab_bgr()
    if not ok:
        return False
    found, seed = localize_ball_lowest_contour(img, target_color)
    if not found:
        # Rare fallback: YOLO once (robot may be at awkward angle)
        found, seed = localize_ball_yolo(img, target_color)
        if not found:
            print(f"% SimBallMission: fine approach: cannot seed for {target_color}")
            return False

    prev_window = seed['rect']
    lost_frames = 0

    while not service.stop:
        ok, img = _grab_bgr()
        if not ok:
            t.sleep(0.05)
            continue

        ball_data = track_ball_window(img, target_color, prev_window)

        if ball_data is None:
            lost_frames += 1
            if lost_frames > 10:
                print(f"% SimBallMission: ball lost during fine approach")
                _stop()
                return False
            service.send("robobot/cmd/ti", "rc 0 0")
            t.sleep(0.05)
            continue

        lost_frames = 0
        prev_window = ball_data['rect']
        cX, cY = ball_data['center']
        _, _, w, h = ball_data['rect']
        r_px = min(w, h) / 2.0

        # Camera-derived distance is accurate at this range.
        coords = pixels_to_robot_coords([(cX, cY, r_px)])
        if not coords:
            continue
        x_r, y_r, _ = coords[0]
        actual_dist = math.hypot(x_r, y_r)   # horizontal distance to ball [m]

        # Positive error_x → ball left of centre → turn left (CCW, +turn_vel).
        error_x   = _CAM_CX - cX
        # Positive error_dist → too far → drive forward.
        error_dist = actual_dist - _STANDOFF_M

        if abs(error_dist) < _FINE_DIST_TOL_M and abs(error_x) < _FINE_ALIGN_TOL_PX:
            _stop()
            print(f"% SimBallMission: fine approach done — "
                  f"dist={actual_dist:.3f} m  err_x={int(error_x)} px")
            return True

        fwd_vel  = max(-0.10, min(0.25, _KP_FWD_FINE * error_dist))
        turn_vel = max(-0.50, min(0.50, _KP_TURN_FINE * error_x))
        service.send("robobot/cmd/ti", f"rc {fwd_vel:.3f} {turn_vel:.3f}")
        t.sleep(0.05)

    _stop()
    return False


def _rotation_scan(target_color: str, world_model: BallWorldModel) -> bool:
    """
    Rotate up to 360° in stop-and-look YOLO steps to relocate *target_color*.

    Called only when the target ball has been absent from the tracker for
    _LOST_BALL_TIMEOUT_S seconds.  The obstacle ball's world-model position is
    left untouched — its last-seen estimate is good enough for path planning.

    Returns True if the target ball was re-detected (world model updated),
    False if the full rotation exhausted without a detection.
    """
    _stop()
    print(f"% SimBallMission: {target_color} lost — starting rotation scan")

    # Steps required for a full 360°: each pulse turns _SCAN_PULSE_VEL * _SCAN_PULSE_S rad.
    step_rad = _SCAN_PULSE_VEL * _SCAN_PULSE_S          # rad per pulse ≈ 0.1 rad ≈ 5.7°
    steps = int(2 * math.pi / step_rad) + 2              # slight overshoot for safety

    for _ in range(steps):
        if service.stop:
            return False

        t.sleep(0.1)   # mechanical settle
        ok, img = _grab_bgr()
        if ok:
            prev = world_model.get(target_color)
            prev_count = prev.detection_count if prev else 0
            _detect_and_update(img, world_model)   # updates both colors via YOLO
            ball = world_model.get(target_color)
            if ball and ball.detection_count > prev_count:
                print(f"% SimBallMission: {target_color} re-acquired at "
                      f"({ball.x:.2f},{ball.y:.2f})")
                _stop()
                return True

        service.send("robobot/cmd/ti", f"rc 0 {_SCAN_PULSE_VEL:.3f}")
        t.sleep(_SCAN_PULSE_S)
        _stop()

    print(f"% SimBallMission: {target_color} not found after full rotation")
    return False


# ---------------------------------------------------------------------------
# Navigation helper (mirrors pathfinding_mission.py pattern)
# ---------------------------------------------------------------------------

def _drive_path(path: list[tuple[float, float]]) -> bool:
    """
    Execute *path* waypoint by waypoint using graph_nav.drive_to().
    Returns True when all waypoints are reached, False if stopped externally.
    """
    wp_idx = 1
    while not service.stop and wp_idx < len(path):
        tx, ty = path[wp_idx]
        has_next = wp_idx + 1 < len(path)
        nx = path[wp_idx + 1][0] if has_next else None
        ny = path[wp_idx + 1][1] if has_next else None

        graph_nav._stop_nav.clear()
        nav_thread = threading.Thread(
            target=graph_nav.drive_to,
            args=(tx, ty, nx, ny, not has_next),
            daemon=True,
        )
        nav_thread.start()
        nav_thread.join()
        wp_idx += 1

    return wp_idx >= len(path)


def _plan_and_drive(goal: tuple[float, float],
                    obstacles: list[CircleObstacle]) -> bool:
    """
    Plan a path from the current robot position to *goal* avoiding *obstacles*,
    then execute it. Returns True on success.
    """
    eff_start = find_safe_start(_pos(), goal, obstacles, _CFG)
    planner = RealtimePathfinder(goal=goal, cfg=_CFG, replan_cooldown=0.0)
    state = planner.update(eff_start, obstacles)

    if not state.solved:
        print("% SimBallMission: planning failed")
        _stop()
        return False

    print(f"% SimBallMission: path planned, {len(state.path)} waypoints, "
          f"{state.path_length:.2f} m")
    return _drive_path(state.path)


def _navigate_to_ball(target_color: str, obstacle_color: str | None,
                      world_model: BallWorldModel) -> bool:
    """
    Two-phase approach to *target_color*.

    Phase 1 — Odometry navigation with mid-segment interruption.
        A _ClassicalTracker runs localize_ball_lowest_contour() at ~10 Hz in the
        background, continuously updating the world model while the robot drives.

        Rather than waiting for waypoint boundaries (which are too rare with the
        RRT planner), the nav thread is polled every _POLL_INTERVAL_S seconds via
        join(timeout).  On each poll:

          • Close-range check  — if world-model distance ≤ _CLOSE_RANGE_M, stop
            nav and hand off to _fine_approach.
          • Replan check       — if tracker has a fresh detection AND the estimate
            has shifted > _REPLAN_DIST_M from where we planned, stop nav and
            replan from the current position.
          • Lost-ball fallback — if the TARGET ball has been absent from the tracker
            for > _LOST_BALL_TIMEOUT_S (not visible at all), stop nav and rotate
            360° with YOLO until re-acquired.  The obstacle ball's last-seen
            estimate is kept as-is; we never search for it here.

    Phase 2 — Classical fine approach (_fine_approach).

    Parameters
    ----------
    target_color   : 'R' or 'B'
    obstacle_color : other ball used as path obstacle, or None if already collected
    world_model    : shared BallWorldModel
    """
    for cycle in range(_MAX_REPLAN_CYCLES):
        ball = world_model.get(target_color)
        if ball is None:
            print(f"% SimBallMission: no estimate for {target_color}, cannot navigate")
            return False

        rx, ry = _pos()
        if math.hypot(ball.x - rx, ball.y - ry) <= _CLOSE_RANGE_M:
            return _fine_approach(target_color, world_model)

        approach = _approach_point(ball.x, ball.y, rx, ry)
        plan_bx, plan_by = ball.x, ball.y   # snapshot for shift comparison

        obstacles: list[CircleObstacle] = []
        if obstacle_color:
            obs_est = world_model.get(obstacle_color)
            if obs_est and not obs_est.collected:
                obstacles = [CircleObstacle(obs_est.x, obs_est.y, _OBS_BALL_RADIUS)]

        eff_start = find_safe_start(_pos(), approach, obstacles, _CFG)
        planner = RealtimePathfinder(goal=approach, cfg=_CFG, replan_cooldown=0.0)
        state = planner.update(eff_start, obstacles)

        if not state.solved:
            print(f"% SimBallMission: planning failed for {target_color} (cycle {cycle})")
            _stop()
            return False

        path = state.path
        print(f"% SimBallMission: navigating to {target_color} approach "
              f"({approach[0]:.2f},{approach[1]:.2f}), "
              f"{len(path)} waypoints, {state.path_length:.2f} m (cycle {cycle})")

        tracker = _ClassicalTracker(target_color, world_model)
        tracker.start()

        # Outcomes: 'done' | 'close' | 'replan' | 'lost'
        outcome = 'done'
        wp_idx = 1

        try:
            while not service.stop and wp_idx < len(path):
                tx, ty = path[wp_idx]
                has_next = wp_idx + 1 < len(path)
                nx = path[wp_idx + 1][0] if has_next else None
                ny = path[wp_idx + 1][1] if has_next else None

                graph_nav._stop_nav.clear()
                nav_thread = threading.Thread(
                    target=graph_nav.drive_to,
                    args=(tx, ty, nx, ny, not has_next),
                    daemon=True,
                )
                nav_thread.start()

                # Poll nav_thread at _POLL_INTERVAL_S instead of blocking join().
                while nav_thread.is_alive() and not service.stop:
                    nav_thread.join(timeout=_POLL_INTERVAL_S)
                    if not nav_thread.is_alive():
                        break   # waypoint reached naturally

                    rx, ry = _pos()
                    ball = world_model.get(target_color)

                    # ── Close-range: hand off to fine approach ──────────────
                    if ball and math.hypot(ball.x - rx, ball.y - ry) <= _CLOSE_RANGE_M:
                        graph_nav._stop_nav.set()
                        nav_thread.join()
                        outcome = 'close'
                        break

                    # ── Tracker update: replan if estimate has shifted ───────
                    if tracker.pop_detected():
                        ball = world_model.get(target_color)
                        if ball:
                            disp = math.hypot(ball.x - plan_bx, ball.y - plan_by)
                            if disp > _REPLAN_DIST_M:
                                print(f"% SimBallMission: {target_color} estimate shifted "
                                      f"{disp:.2f} m mid-drive — replanning (cycle {cycle})")
                                graph_nav._stop_nav.set()
                                nav_thread.join()
                                outcome = 'replan'
                                break

                    # ── Lost-ball: target absent too long → rotation scan ───
                    if world_model.absent_for(target_color) > _LOST_BALL_TIMEOUT_S:
                        graph_nav._stop_nav.set()
                        nav_thread.join()
                        outcome = 'lost'
                        break

                if outcome != 'done' or service.stop:
                    break

                wp_idx += 1   # waypoint reached naturally; advance

        finally:
            tracker.stop()

        if service.stop:
            return False

        if outcome == 'close' or outcome == 'done':
            return _fine_approach(target_color, world_model)

        if outcome == 'replan':
            continue   # outer for-loop replans from current position

        if outcome == 'lost':
            if not _rotation_scan(target_color, world_model):
                return False
            continue   # replan with re-acquired estimate

    print(f"% SimBallMission: exceeded {_MAX_REPLAN_CYCLES} replan cycles for {target_color}")
    return False


# ---------------------------------------------------------------------------
# Mission states
# ---------------------------------------------------------------------------

class _State(Enum):
    SCAN             = auto()
    NAVIGATE_TO_RED  = auto()
    WAIT_RED_PICKUP  = auto()
    DELIVER_RED      = auto()
    NAVIGATE_TO_BLUE = auto()
    WAIT_BLUE_PICKUP = auto()
    DELIVER_BLUE     = auto()
    DONE             = auto()
    FAILED           = auto()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def simulate_ball_mission() -> None:
    """
    Run the simulated ball delivery mission.
    Called from mqtt-client.py.
    """
    print("% SimBallMission: starting")
    _led(0, 16, 30)   # cyan = active

    # Reset odometry so the world origin is under the robot's start position.
    service.send("robobot/cmd/T0", "enc0")
    t.sleep(0.2)

    cam.setup_raw()

    x0, y0 = _pos()
    h0 = _heading()
    goal = (
        x0 + _GOAL_DIST_M * math.cos(h0),
        y0 + _GOAL_DIST_M * math.sin(h0),
    )
    print(f"% SimBallMission: start=({x0:.3f},{y0:.3f})  "
          f"hdg={math.degrees(h0):.1f}°  goal=({goal[0]:.3f},{goal[1]:.3f})")

    world_model = BallWorldModel(colors=('R', 'B'))
    mission_start = t.monotonic()
    scan_start = mission_start
    state = _State.SCAN

    try:
        while not service.stop:
            if t.monotonic() - mission_start > _MISSION_TIMEOUT_S:
                print("% SimBallMission: timeout!")
                state = _State.FAILED
                break

            # ----------------------------------------------------------
            if state == _State.SCAN:
                # Stop-and-look: identical pattern to bucketballsmission state 0.
                # Stop the robot, wait for vibration to settle, grab a fresh
                # frame, run YOLO for both colors, then rotate a small step.
                _stop()
                t.sleep(0.1)   # mechanical settle

                ok, img = _grab_bgr()
                if ok:
                    _detect_and_update(img, world_model)

                red = world_model.get('R')
                blue = world_model.get('B')
                red_ready = red is not None and red.detection_count >= _SCAN_MIN_RED_DETECTIONS
                blue_ready = blue is not None and blue.detection_count >= _SCAN_MIN_BLUE_DETECTIONS
                scan_elapsed = t.monotonic() - scan_start

                if red_ready and (blue_ready or scan_elapsed >= _SCAN_START_TIMEOUT_S):
                    _stop()
                    r = world_model.get('R')
                    b = world_model.get('B')
                    blue_txt = f"({b.x:.2f},{b.y:.2f})" if b else "unknown"
                    print(f"% SimBallMission: scan ready — "
                          f"R=({r.x:.2f},{r.y:.2f})  B={blue_txt}  "
                          f"after {scan_elapsed:.1f} s")
                    state = _State.NAVIGATE_TO_RED
                else:
                    # Small rotation pulse then loop back
                    service.send("robobot/cmd/ti",
                                 f"rc 0 {_SCAN_PULSE_VEL:.3f}")
                    t.sleep(_SCAN_PULSE_S)
                    _stop()

            # ----------------------------------------------------------
            elif state == _State.NAVIGATE_TO_RED:
                ok = _navigate_to_ball('R', 'B', world_model)
                state = _State.WAIT_RED_PICKUP if ok else _State.FAILED

            # ----------------------------------------------------------
            elif state == _State.WAIT_RED_PICKUP:
                _stop()
                _led(30, 0, 30)   # purple = waiting for pickup
                print("% SimBallMission: waiting for red ball pickup…")
                wait_start = t.monotonic()
                while not service.stop:
                    if t.monotonic() - wait_start > _PICKUP_TIMEOUT_S:
                        print("% SimBallMission: pickup timeout")
                        state = _State.FAILED
                        break
                    # Inline vision: keep world model fresh so absent_for()
                    # correctly measures time since the ball was last seen.
                    ok, img = _grab_bgr()
                    if ok:
                        _detect_color(img, 'R', world_model)
                    if world_model.absent_for('R') >= _PICKUP_ABSENT_S:
                        world_model.mark_collected('R')
                        print("% SimBallMission: red ball collected")
                        state = _State.DELIVER_RED
                        break

            # ----------------------------------------------------------
            elif state == _State.DELIVER_RED:
                _led(0, 16, 30)
                blue = world_model.get('B')
                blue_obs = [CircleObstacle(blue.x, blue.y, _OBS_BALL_RADIUS)] if blue else []
                print(f"% SimBallMission: delivering red to goal {goal}")
                ok = _plan_and_drive(goal, blue_obs)
                if ok:
                    print("% SimBallMission: red delivered!")
                    state = _State.NAVIGATE_TO_BLUE
                else:
                    state = _State.FAILED

            # ----------------------------------------------------------
            elif state == _State.NAVIGATE_TO_BLUE:
                # Red is already collected so no obstacle needed.
                ok = _navigate_to_ball('B', None, world_model)
                state = _State.WAIT_BLUE_PICKUP if ok else _State.FAILED

            # ----------------------------------------------------------
            elif state == _State.WAIT_BLUE_PICKUP:
                _stop()
                _led(30, 0, 30)
                print("% SimBallMission: waiting for blue ball pickup…")
                wait_start = t.monotonic()
                while not service.stop:
                    if t.monotonic() - wait_start > _PICKUP_TIMEOUT_S:
                        print("% SimBallMission: pickup timeout")
                        state = _State.FAILED
                        break
                    ok, img = _grab_bgr()
                    if ok:
                        _detect_color(img, 'B', world_model)
                    if world_model.absent_for('B') >= _PICKUP_ABSENT_S:
                        world_model.mark_collected('B')
                        print("% SimBallMission: blue ball collected")
                        state = _State.DELIVER_BLUE
                        break

            # ----------------------------------------------------------
            elif state == _State.DELIVER_BLUE:
                _led(0, 16, 30)
                print(f"% SimBallMission: delivering blue to goal {goal}")
                ok = _plan_and_drive(goal, [])
                if ok:
                    print("% SimBallMission: blue delivered!")
                    state = _State.DONE
                else:
                    state = _State.FAILED

            # ----------------------------------------------------------
            elif state == _State.DONE:
                break

            elif state == _State.FAILED:
                break

    finally:
        _stop()

    if state == _State.DONE:
        elapsed = t.monotonic() - mission_start
        print(f"% SimBallMission: complete in {elapsed:.1f} s")
        _led(0, 100, 0)   # green = success
    else:
        print("% SimBallMission: failed or aborted")
        _led(100, 0, 0)   # red = failure

    t.sleep(2.0)
    _led(0, 0, 0)
    print("% SimBallMission: done")
