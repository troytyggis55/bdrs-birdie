"""
Final ball mission — Phase 1 test build.

Full mission plan (CLAUDE.md):
  SCAN_FOR_BALLS → PICKUP_RED → DELIVER_TO_A → PICKUP_BLUE → DELIVER_TO_C → RETURN_TO_START

Current test build (Phase 1):
  SCAN → NAVIGATE_TO_RED → DONE
  Only the red-ball approach is active. Delivery and blue-ball states are
  commented out at the bottom of final_ball_mission() — all the underlying
  helpers (_fine_approach, _navigate_to_ball, etc.) remain intact and
  fully logged so the replay tool shows the complete picture.

ArUco markers are scanned passively throughout: during SCAN (inline, per frame)
and during navigation (background thread). Every sighting is written to the
mission log for post-run replay.

Log written to MissionLogs/ball_mission_<timestamp>.log
Replay:  python ball_mission/final_ball_replay.py [log_path]
"""
from json.encoder import INFINITY

import cv2 as cv
import math
import numpy as np
import sys
import threading
import time as t
from enum import Enum, auto
from pathlib import Path

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from pathfinding.pathfinding import CircleObstacle, PlannerConfig, find_safe_start
from pathfinding.realtime_pathfind import RealtimePathfinder

from worldmodel.ball_world_model import BallWorldModel
from CamVision.visualcontrol import (localize_ball_yolo, localize_ball_lowest_contour,
                                     track_ball_window)
from CamVision.ballcoords import pixels_to_robot_coords, C_X as _CAM_CX
from aruco.detect_aruco import detect_aruco
from CamVision.coord_conversion import aruco_to_robot_frame
from ball_mission.final_ball_logger import FinalBallLogger
from ball_mission.arena_walls import ArenaWallModel, walls_from_aruco_world
from scam import cam
from spose import pose
from uservice import service
from odometry.graph_nav import graph_nav

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

_CFG = PlannerConfig(
    delta=0.15,
    goal_tolerance=0.1,
    clearance=0.00,
    robot_radius=0.2,
    max_steps=3000,
    smooth_path=True,
)

_NAV_STANDOFF_M   = 0.30   # pathfinding approach point: this far behind the ball [m]
_FINE_STANDOFF_M  = 0.20   # fine-approach stop distance [m]
_OBS_BALL_RADIUS  = 0.08   # obstacle radius for other balls in pathfinder [m]
_PICKUP_ABSENT_S  = 3.0    # ball must be absent this long to count as picked up [s]
_PICKUP_TIMEOUT_S = 30.0   # max wait for pickup before aborting [s]
_SCAN_PULSE_VEL   = 2      # turn rate during each scan pulse [rad/s]
_SCAN_PULSE_S     = 0.2    # duration of each scan pulse [s]
_MISSION_TIMEOUT_S = 120.0
_MAX_REPLAN_CYCLES = 99999
_POLL_INTERVAL_S   = 0.50
_REPLAN_DIST_M     = 0.05
_CAM_FOV_HALF_RAD  = math.radians(33)
_KP_TURN_FINE      = 0.010
_KP_FWD_FINE       = 0.30
_FINE_ALIGN_TOL_PX = 5
_FINE_DIST_TOL_M   = 0.03
_SCAN_START_TIMEOUT_S = 4.0
_SCAN_MIN_RED_DETECTIONS = 1
_SCAN_MIN_BLUE_DETECTIONS = 1
_POSE_LOG_INTERVAL_S = 0.20  # minimum gap between BM_POSE log entries [s]

_ARUCO_DICT = "DICT_4X4_100"  # dictionary used to print arena markers (IDs 10–17)
_ARUCO_MARKER_SIZE_M = 0.10   # physical side length [m] — matches CLAUDE.md
_ARUCO_ID_MIN = 10            # ignore markers outside this range (reduces noise)
_ARUCO_ID_MAX = 17

_ROTATE_STEP_VEL = 1  # rotation speed during heading-align [rad/s]

# Classical tracking FOV guard
_IMG_WIDTH              = 820   # camera frame width (pixels)
_TRACKING_FOV_X_MARGIN  = 100   # classical track invalid when ball centre is within
                                # this many pixels of the left/right edge; ball can
                                # only re-enter classical tracking after a YOLO rescan

# Debug image capture
_IMG_LOG_EVERY_N = 2   # save an annotated frame every N classical-tracker iterations

# ---------------------------------------------------------------------------
# Module-level singletons (set by final_ball_mission() at start)
# ---------------------------------------------------------------------------

_logger: FinalBallLogger | None = None
_last_pose_log: float = 0.0  # monotonic time of last BM_POSE write
_wall_model: ArenaWallModel | None = None  # grows as ArUco markers are seen


def _log_pose_maybe() -> None:
    """Write a BM_POSE entry if enough time has passed since the last one."""
    global _last_pose_log
    now = t.monotonic()
    if now - _last_pose_log >= _POSE_LOG_INTERVAL_S:
        if _logger is not None:
            x, y = _pos()
            _logger.log_pose(x, y, _heading())
        _last_pose_log = now


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pos() -> tuple[float, float]:
    return pose.pose[0], pose.pose[1]


def _heading() -> float:
    return pose.pose[2]


def _approach_point(ball_x: float, ball_y: float,
                    robot_x: float, robot_y: float,
                    standoff: float = _NAV_STANDOFF_M) -> tuple[float, float]:
    """Return the point *standoff* metres behind the ball on the robot→ball line."""
    dx = ball_x - robot_x
    dy = ball_y - robot_y
    dist = math.hypot(dx, dy)
    if dist < 1e-3:
        return ball_x, ball_y
    return ball_x - standoff * dx / dist, ball_y - standoff * dy / dist


def _approach_point_wall_aware(
    ball_x: float, ball_y: float,
    robot_x: float, robot_y: float,
    standoff: float = _NAV_STANDOFF_M,
) -> tuple[float, float]:
    """
    Compute the pathfinding approach point for a ball.

    When wall mapping is available the approach is placed on the FAR side of
    the ball from the plus-center, so the ball is always between the robot and
    the arena walls.  This prevents the planner from targeting a point that is
    squeezed between the ball and a wall.

    Fallback: standard robot-relative approach when no wall data exists.
    """
    if _wall_model is not None:
        plus_center = _wall_model.get_plus_center()
        if plus_center is not None:
            cx, cy = plus_center
            dx = ball_x - cx
            dy = ball_y - cy
            dist = math.hypot(dx, dy)
            if dist > 0.02:
                nx, ny = dx / dist, dy / dist   # outward unit vector (away from plus)
                ap = (ball_x + standoff * nx, ball_y + standoff * ny)
                print(f"% FinalBallMission: wall-aware approach "
                      f"goal=({ap[0]:.2f},{ap[1]:.2f})  "
                      f"plus=({cx:.2f},{cy:.2f})")
                return ap
    # Fallback
    return _approach_point(ball_x, ball_y, robot_x, robot_y, standoff)


def _rotate_to_face_ball(target_color: str, world_model: BallWorldModel) -> bool:
    """
    Rotate to face the ball using the world-model estimate, then confirm with YOLO.

    1. Compute bearing to ball's last known world position; rotate open-loop.
    2. Take one stationary YOLO scan to confirm.
    """
    _stop()
    t.sleep(0.1)

    # ── 1. Odometry-based rotation ────────────────────────────────────────
    ball = world_model.get(target_color)
    if ball is not None:
        rx, ry = _pos()
        hdg = _heading()
        bearing = math.atan2(ball.y - ry, ball.x - rx)
        delta = _angle_wrap(bearing - hdg)
        print(f"% FinalBallMission: rotate-to-face {target_color}: "
              f"delta={math.degrees(delta):.1f}°  "
              f"ball=({ball.x:.2f},{ball.y:.2f})  hdg={math.degrees(hdg):.1f}°")
        if abs(delta) > math.radians(3):
            turn_vel = _ROTATE_STEP_VEL if delta > 0 else -_ROTATE_STEP_VEL
            service.send("robobot/cmd/ti", f"rc 0 {turn_vel:.3f}")
            t.sleep(abs(delta) / _ROTATE_STEP_VEL)
            _stop()
            t.sleep(0.1)
    else:
        print(f"% FinalBallMission: rotate-to-face {target_color}: "
              f"no world-model estimate — skipping odometry rotation")

    # ── 2. Single YOLO confirmation ───────────────────────────────────────
    ok, img = _grab_bgr()
    if ok:
        _scan_aruco_passive(img)
        found, _ = localize_ball_yolo(img, target_color)
        if found:
            print(f"% FinalBallMission: rotate-to-face {target_color}: confirmed by YOLO")
            return True

    print(f"% FinalBallMission: rotate-to-face {target_color}: not visible after rotation")
    return False


def _stop() -> None:
    service.send("robobot/cmd/ti", "rc 0 0")


def _led(r: int, g: int, b: int) -> None:
    service.send("robobot/cmd/T0", f"leds 16 {r} {g} {b}")


def _grab_bgr():
    """Grab one fresh camera frame. Returns (ok, bgr_img)."""
    ok, img, _ = cam.getRawFrame()
    if not ok or img is None:
        return False, None
    return True, cv.cvtColor(img, cv.COLOR_RGB2BGR)


def _angle_wrap(a: float) -> float:
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


def _ball_in_fov(ball_wx: float, ball_wy: float) -> bool:
    """True if the ball's last known world position falls within the camera FOV."""
    rx, ry = _pos()
    bearing = math.atan2(ball_wy - ry, ball_wx - rx)
    return abs(_angle_wrap(bearing - _heading())) < _CAM_FOV_HALF_RAD


# ---------------------------------------------------------------------------
# Debug frame annotation
# ---------------------------------------------------------------------------

def _annotate_frame(img, ball_data=None, aruco_list=None, label: str = ""):
    """
    Return an annotated copy of *img* (BGR) for debug image logging.

    Draws:
    • Orange vertical lines at the FOV guard margins.
    • Green bounding box + red centroid dot if *ball_data* is provided.
    • Cyan polygon + ID text for each marker in *aruco_list*.
    • White/black label text in the top-left corner.
    """
    out = img.copy()
    h = out.shape[0]
    # FOV boundary markers (orange)
    cv.line(out, (_TRACKING_FOV_X_MARGIN, 0),
            (_TRACKING_FOV_X_MARGIN, h), (0, 165, 255), 1)
    cv.line(out, (_IMG_WIDTH - _TRACKING_FOV_X_MARGIN, 0),
            (_IMG_WIDTH - _TRACKING_FOV_X_MARGIN, h), (0, 165, 255), 1)
    # Ball bounding box
    if ball_data is not None:
        x, y, w, bh = ball_data['rect']
        cX, cY = int(ball_data['center'][0]), int(ball_data['center'][1])
        cv.rectangle(out, (x, y), (x + w, y + bh), (0, 255, 0), 2)
        cv.circle(out, (cX, cY), 5, (0, 0, 255), -1)
        cv.putText(out, f"({cX},{cY})", (x, max(y - 6, 12)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    # ArUco overlays
    if aruco_list:
        for m in aruco_list:
            corners = m.get('corners')
            if corners is not None and len(corners) >= 4:
                pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
                cv.polylines(out, [pts], True, (0, 255, 255), 2)
            center = m.get('center')
            if center:
                ax_c, ay_c = int(center[0]), int(center[1])
                cv.putText(out, f"id:{m['id']}", (ax_c + 4, ay_c - 4),
                           cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    # State / phase label (white outline + black fill for readability)
    if label:
        cv.putText(out, label, (10, 28),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(out, label, (10, 28),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    return out


# ---------------------------------------------------------------------------
# Background classical tracker (ball)
# ---------------------------------------------------------------------------

class _ClassicalTracker:
    """
    Background thread running localize_ball_lowest_contour() at ~10 Hz while
    the robot drives. Updates the world model and the mission log continuously.

    FOV guard: if the ball centre pixel is within _TRACKING_FOV_X_MARGIN of the
    left or right edge, the tracker suspends world-model updates and raises a
    `_needs_yolo` flag.  The main loop must stop the robot, run a YOLO scan, and
    call `revalidate()` before classical tracking resumes.  This prevents the
    tracker from feeding pure noise into the world model when the ball is
    partially or fully off-screen.
    """
    _INTERVAL_S = 0.10

    def __init__(self, target_color: str, world_model: BallWorldModel,
                 frame_label: str = "") -> None:
        self._color = target_color
        self._wm = world_model
        self._frame_label = frame_label
        self._stop_event = threading.Event()
        self._detected = threading.Event()
        self._needs_yolo = threading.Event()   # set when ball exits FOV margin
        self._frame_cnt = 0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._detected.clear()
        self._needs_yolo.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def pop_detected(self) -> bool:
        """Return True (and clear) if a valid in-FOV detection occurred since last call."""
        if self._detected.is_set():
            self._detected.clear()
            return True
        return False

    def is_needs_yolo(self) -> bool:
        """True when the ball has left the FOV margin and needs YOLO reconfirmation."""
        return self._needs_yolo.is_set()

    def revalidate(self) -> None:
        """Call after YOLO confirms the ball position; re-enables classical tracking."""
        self._needs_yolo.clear()
        print(f"% FinalBallMission: Tracker [{self._color}]: revalidated by YOLO")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            # Dormant: wait for YOLO revalidation before resuming classical updates
            if self._needs_yolo.is_set():
                t.sleep(self._INTERVAL_S)
                continue

            ok, img = _grab_bgr()
            if not ok:
                t.sleep(self._INTERVAL_S)
                continue

            found, ball_data = localize_ball_lowest_contour(img, self._color)
            if found:
                cx, cy = ball_data['center']

                # ── FOV edge guard ────────────────────────────────────────
                if cx < _TRACKING_FOV_X_MARGIN or cx > (_IMG_WIDTH - _TRACKING_FOV_X_MARGIN):
                    print(f"% FinalBallMission: Tracker [{self._color}]: "
                          f"ball at FOV edge cx={cx:.0f} — suspending classical tracking")
                    if _logger is not None:
                        ann = _annotate_frame(
                            img, ball_data,
                            label=f"FOV_EDGE {self._color} cx={int(cx)}")
                        _logger.log_frame(ann, label=f"fov_edge_{self._color}")
                    self._needs_yolo.set()
                    t.sleep(self._INTERVAL_S)
                    continue

                # ── Normal in-FOV update ──────────────────────────────────
                _, _, w, h = ball_data['rect']
                r_px = min(w, h) / 2.0
                coords = pixels_to_robot_coords([(cx, cy, r_px)])
                if coords:
                    x_r, y_r, _ = coords[0]
                    rx, ry = _pos()
                    hdg = _heading()
                    self._wm.update(self._color, x_r, y_r, rx, ry, hdg)
                    est = self._wm.get(self._color)
                    if _logger is not None and est is not None:
                        _logger.log_ball_det(
                            self._color, float(cx), float(cy), r_px,
                            x_r, y_r, est.x, est.y, est.detection_count,
                        )
                    self._detected.set()
                    print(f"% FinalBallMission: Tracker [{self._color}]: "
                          f"robot=({x_r:.2f},{y_r:.2f})  "
                          f"world=({est.x:.2f},{est.y:.2f})")

                # ── Periodic debug frame save ─────────────────────────────
                self._frame_cnt += 1
                if _logger is not None and self._frame_cnt % _IMG_LOG_EVERY_N == 0:
                    lbl = self._frame_label or f"track_{self._color}"
                    ann = _annotate_frame(img, ball_data, label=lbl)
                    _logger.log_frame(ann, label=lbl)

            _log_pose_maybe()
            t.sleep(self._INTERVAL_S)


# ---------------------------------------------------------------------------
# Background passive ArUco scanner
# ---------------------------------------------------------------------------

class _PassiveArucoScanner:
    """
    Background thread that runs detect_aruco() on every frame and logs all
    marker sightings. Does NOT drive navigation — purely observational.

    Run this whenever the robot is moving (during navigation phases) so that
    ArUco sightings accumulate in the log without interrupting the ball tracker.
    """
    _INTERVAL_S = 0.10

    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: list[dict] = []  # most recent parsed detections

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def get_latest(self) -> list[dict]:
        """Return the most recent list of ArUco detections (thread-safe snapshot)."""
        with self._lock:
            return list(self._latest)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            ok, img = _grab_bgr()
            if not ok:
                t.sleep(self._INTERVAL_S)
                continue

            try:
                markers, _ = detect_aruco(img, dictionary_name=_ARUCO_DICT)
            except Exception:
                t.sleep(self._INTERVAL_S)
                continue

            rx, ry, hdg = pose.pose[0], pose.pose[1], pose.pose[2]
            parsed: list[dict] = []
            for m in markers:
                if not _is_arena_aruco(m['id']):
                    continue   # ignore IDs outside 10–17
                result = aruco_to_robot_frame(m, marker_size_m=_ARUCO_MARKER_SIZE_M)
                if result is None:
                    continue
                x_r, y_r, fnx, fny = result
                # Convert to world frame for wall model
                fnx_w = fnx * math.cos(hdg) - fny * math.sin(hdg)
                fny_w = fnx * math.sin(hdg) + fny * math.cos(hdg)
                wx = rx + x_r * math.cos(hdg) - y_r * math.sin(hdg)
                wy = ry + x_r * math.sin(hdg) + y_r * math.cos(hdg)
                if _logger is not None:
                    _logger.log_aruco(m['id'], x_r, y_r, fnx, fny, rx, ry, hdg)
                if _wall_model is not None:
                    _wall_model.update(m['id'], wx, wy, fnx_w, fny_w)
                parsed.append({'id': m['id'], 'x_r': x_r, 'y_r': y_r,
                                'fnx': fnx, 'fny': fny,
                                'wx': wx, 'wy': wy})
                print(f"% FinalBallMission: ArUco id={m['id']}  "
                      f"robot_frame=({x_r:.2f},{y_r:.2f})  "
                      f"normal=({fnx:.2f},{fny:.2f})  "
                      f"walls_known={_wall_model.marker_count() if _wall_model else 0}")

            with self._lock:
                self._latest = parsed

            t.sleep(self._INTERVAL_S)


# ---------------------------------------------------------------------------
# Per-frame YOLO detection helpers
# ---------------------------------------------------------------------------

def _detect_and_update(img, world_model: BallWorldModel) -> None:
    """
    Run YOLO for both ball colors on *img* and update the world model + log.
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
        if _logger is not None and est is not None:
            _logger.log_ball_det(color, float(cx), float(cy), r_px,
                                 x_r, y_r, est.x, est.y, est.detection_count)
        print(f"% FinalBallMission: {color} detected  center=({cx},{cy})  "
              f"robot=({x_r:.2f},{y_r:.2f})  count={est.detection_count}")


def _detect_color(img, color: str, world_model: BallWorldModel) -> bool:
    """Run YOLO for a single *color*, update world model + log. Returns True if found."""
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
    est = world_model.get(color)
    if _logger is not None and est is not None:
        _logger.log_ball_det(color, float(cx), float(cy), r_px,
                             x_r, y_r, est.x, est.y, est.detection_count)
    return True


def _is_arena_aruco(marker_id: int) -> bool:
    """Return True only for the 8 inner-wall markers (IDs 10–17) we care about."""
    return _ARUCO_ID_MIN <= marker_id <= _ARUCO_ID_MAX


def _scan_aruco_passive(img) -> None:
    """
    Run detect_aruco() on *img*, log every visible arena marker (IDs 10–17),
    and update the wall model. Call inline on fresh stopped-robot frames.
    """
    try:
        markers, _ = detect_aruco(img, dictionary_name=_ARUCO_DICT)
    except Exception:
        return
    rx, ry, hdg = pose.pose[0], pose.pose[1], pose.pose[2]
    for m in markers:
        if not _is_arena_aruco(m['id']):
            continue   # ignore test markers, camera calibration targets, etc.
        result = aruco_to_robot_frame(m, marker_size_m=_ARUCO_MARKER_SIZE_M)
        if result is None:
            continue
        x_r, y_r, fnx, fny = result
        # Convert face normal to world frame for wall model
        fnx_w = fnx * math.cos(hdg) - fny * math.sin(hdg)
        fny_w = fnx * math.sin(hdg) + fny * math.cos(hdg)
        wx = rx + x_r * math.cos(hdg) - y_r * math.sin(hdg)
        wy = ry + x_r * math.sin(hdg) + y_r * math.cos(hdg)
        if _logger is not None:
            _logger.log_aruco(m['id'], x_r, y_r, fnx, fny, rx, ry, hdg)
        if _wall_model is not None:
            _wall_model.update(m['id'], wx, wy, fnx_w, fny_w)
        print(f"% FinalBallMission: [SCAN] ArUco id={m['id']}  "
              f"robot_frame=({x_r:.2f},{y_r:.2f})  "
              f"walls_known={_wall_model.marker_count() if _wall_model else 0}")


# ---------------------------------------------------------------------------
# Fine approach (classical CV P-control)
# ---------------------------------------------------------------------------

def _fine_approach(target_color: str, world_model: BallWorldModel) -> bool:
    """
    Classical-CV tight control loop for the final approach to the ball.
    Stops at _FINE_STANDOFF_M in front of the ball.
    """
    print(f"% FinalBallMission: fine approach to {target_color}")
    _stop()
    t.sleep(0.1)

    ok, img = _grab_bgr()
    if not ok:
        return False
    found, seed = localize_ball_lowest_contour(img, target_color)
    if not found:
        found, seed = localize_ball_yolo(img, target_color)
        if not found:
            print(f"% FinalBallMission: fine approach: cannot seed for {target_color}")
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
                print(f"% FinalBallMission: ball lost during fine approach")
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

        coords = pixels_to_robot_coords([(cX, cY, r_px)])
        if not coords:
            continue
        x_r, y_r, _ = coords[0]
        actual_dist = math.hypot(x_r, y_r)

        error_x    = _CAM_CX - cX
        error_dist = actual_dist - _FINE_STANDOFF_M

        _log_pose_maybe()

        if abs(error_dist) < _FINE_DIST_TOL_M and abs(error_x) < _FINE_ALIGN_TOL_PX:
            _stop()
            print(f"% FinalBallMission: fine approach done — "
                  f"dist={actual_dist:.3f} m  err_x={int(error_x)} px")
            # Log final pose
            if _logger is not None:
                x, y = _pos()
                _logger.log_pose(x, y, _heading())
            return True

        fwd_vel  = max(-0.10, min(0.25, _KP_FWD_FINE  * error_dist))
        turn_vel = max(-0.50, min(0.50, _KP_TURN_FINE * error_x))
        service.send("robobot/cmd/ti", f"rc {fwd_vel:.3f} {turn_vel:.3f}")
        t.sleep(0.05)

    _stop()
    return False


# ---------------------------------------------------------------------------
# Rotation scan (YOLO stop-and-look up to 360°)
# ---------------------------------------------------------------------------

def _rotation_scan(target_color: str, world_model: BallWorldModel) -> bool:
    """
    Rotate up to 360° in stop-and-look YOLO steps to relocate *target_color*.
    """
    _stop()
    print(f"% FinalBallMission: {target_color} lost — starting rotation scan")

    step_rad = _SCAN_PULSE_VEL * _SCAN_PULSE_S
    steps = int(2 * math.pi / step_rad) + 2

    for _ in range(steps):
        if service.stop:
            return False

        t.sleep(0.1)
        ok, img = _grab_bgr()
        if ok:
            # Passive ArUco scan on every stopped frame
            _scan_aruco_passive(img)
            prev = world_model.get(target_color)
            prev_count = prev.detection_count if prev else 0
            _detect_and_update(img, world_model)
            ball = world_model.get(target_color)
            if ball and ball.detection_count > prev_count:
                print(f"% FinalBallMission: {target_color} re-acquired at "
                      f"({ball.x:.2f},{ball.y:.2f})")
                _stop()
                return True

        service.send("robobot/cmd/ti", f"rc 0 {_SCAN_PULSE_VEL:.3f}")
        t.sleep(_SCAN_PULSE_S)
        _stop()
        _log_pose_maybe()

    print(f"% FinalBallMission: {target_color} not found after full rotation")
    return False


# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------

def _drive_path(path: list[tuple[float, float]]) -> bool:
    """Execute *path* waypoint by waypoint. Returns True when complete."""
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
        _log_pose_maybe()

    return wp_idx >= len(path)


def _plan_and_drive(goal: tuple[float, float],
                    obstacles: list) -> bool:
    """
    Plan a path from current position to *goal*, execute it, log path + result.
    Walls from the ArenaWallModel are automatically appended to *obstacles*.
    """
    all_obs = list(obstacles)
    if _wall_model is not None:
        all_obs.extend(_wall_model.get_walls())
    eff_start = find_safe_start(_pos(), goal, all_obs, _CFG)
    planner = RealtimePathfinder(goal=goal, cfg=_CFG, replan_cooldown=0.0)
    state = planner.update(eff_start, all_obs)

    if not state.solved:
        print("% FinalBallMission: planning failed")
        _stop()
        return False

    if _logger is not None:
        _logger.log_path(state.path)

    print(f"% FinalBallMission: path planned, {len(state.path)} waypoints, "
          f"{state.path_length:.2f} m")
    return _drive_path(state.path)


def _navigate_to_ball(target_color: str, obstacle_color: str | None,
                      world_model: BallWorldModel) -> bool:
    """
    Two-phase approach to *target_color*:
      Phase 1 — Odometry navigation with mid-segment interruption.
                 _ClassicalTracker + _PassiveArucoScanner run in background.
      Phase 2 — Classical fine approach.
    """
    aruco_scanner = _PassiveArucoScanner()

    for cycle in range(_MAX_REPLAN_CYCLES):
        ball = world_model.get(target_color)
        if ball is None:
            print(f"% FinalBallMission: no estimate for {target_color}, cannot navigate")
            return False

        rx, ry = _pos()
        approach = _approach_point_wall_aware(ball.x, ball.y, rx, ry)
        plan_bx, plan_by = ball.x, ball.y

        # Build obstacle list: target ball + other ball + all known arena walls.
        # The target ball must be an obstacle so the planner routes around it to
        # reach the approach point (which is on the far side of the ball).
        obstacles: list = [CircleObstacle(ball.x, ball.y, _OBS_BALL_RADIUS)]
        if obstacle_color:
            obs_est = world_model.get(obstacle_color)
            if obs_est and not obs_est.collected:
                obstacles.append(CircleObstacle(obs_est.x, obs_est.y, _OBS_BALL_RADIUS))
        if _wall_model is not None:
            obstacles.extend(_wall_model.get_walls())
            if _wall_model.marker_count() > 0:
                print(f"% FinalBallMission: planning with "
                      f"{len(_wall_model.get_walls())} wall segments "
                      f"from {_wall_model.marker_count()} marker(s)")

        eff_start = find_safe_start(_pos(), approach, obstacles, _CFG)
        planner = RealtimePathfinder(goal=approach, cfg=_CFG, replan_cooldown=0.0)
        state = planner.update(eff_start, obstacles)

        if not state.solved:
            print(f"% FinalBallMission: planning failed for {target_color} (cycle {cycle})")
            _stop()
            return False

        path = state.path
        if _logger is not None:
            if cycle == 0:
                _logger.log_path(path)
            else:
                _logger.log_replan(f"shift_cycle{cycle}", eff_start[0], eff_start[1], path)

        print(f"% FinalBallMission: navigating to {target_color} approach "
              f"({approach[0]:.2f},{approach[1]:.2f}), "
              f"{len(path)} waypoints, {state.path_length:.2f} m (cycle {cycle})")

        tracker = _ClassicalTracker(
            target_color, world_model,
            frame_label=f"nav_{target_color}_c{cycle}",
        )
        tracker.start()
        aruco_scanner.start()

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

                while nav_thread.is_alive() and not service.stop:
                    nav_thread.join(timeout=_POLL_INTERVAL_S)
                    if not nav_thread.is_alive():
                        break

                    _log_pose_maybe()

                    # ── Replan if world model estimate has shifted ─────────
                    if tracker.pop_detected():
                        ball = world_model.get(target_color)
                        if ball:
                            disp = math.hypot(ball.x - plan_bx, ball.y - plan_by)
                            if disp > _REPLAN_DIST_M:
                                print(f"% FinalBallMission: {target_color} estimate shifted "
                                      f"{disp:.2f} m — replanning (cycle {cycle})")
                                graph_nav._stop_nav.set()
                                nav_thread.join()
                                outcome = 'replan'
                                break

                if outcome != 'done' or service.stop:
                    break

                wp_idx += 1

        finally:
            tracker.stop()
            aruco_scanner.stop()

        if service.stop:
            return False

        if outcome == 'done':
            _rotate_to_face_ball(target_color, world_model)
            return _fine_approach(target_color, world_model)

        if outcome == 'replan':
            continue

    print(f"% FinalBallMission: exceeded {_MAX_REPLAN_CYCLES} replan cycles for {target_color}")
    return False


# ---------------------------------------------------------------------------
# Mission states
# ---------------------------------------------------------------------------

class _State(Enum):
    SCAN             = auto()
    NAVIGATE_TO_RED  = auto()
    # ── Delivery and blue-ball states — commented out for Phase 1 testing ──
    # WAIT_RED_PICKUP  = auto()
    # DELIVER_RED      = auto()
    # NAVIGATE_TO_BLUE = auto()
    # WAIT_BLUE_PICKUP = auto()
    # DELIVER_BLUE     = auto()
    # RETURN_TO_START  = auto()
    DONE             = auto()
    FAILED           = auto()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def final_ball_mission() -> None:
    """
    Run the final ball mission.
    Called from mqtt-client.py.
    """
    global _logger, _last_pose_log, _wall_model

    print("% FinalBallMission: starting")
    _led(0, 16, 30)

    service.send("robobot/cmd/T0", "enc0")
    t.sleep(0.2)

    cam.setup_raw()

    x0, y0 = _pos()
    h0 = _heading()

    _logger = FinalBallLogger()
    _wall_model = ArenaWallModel()
    _last_pose_log = 0.0
    _logger.log_start(x0, y0, h0)

    print(f"% FinalBallMission: start=({x0:.3f},{y0:.3f})  hdg={math.degrees(h0):.1f}°")
    print(f"% FinalBallMission: log → {_logger.path}")

    world_model = BallWorldModel(colors=('R', 'B'))
    mission_start = t.monotonic()
    scan_start = mission_start
    state = _State.SCAN
    _logger.log_state(state.name)

    try:
        while not service.stop:
            if t.monotonic() - mission_start > _MISSION_TIMEOUT_S:
                print("% FinalBallMission: timeout!")
                state = _State.FAILED
                break

            # ----------------------------------------------------------
            if state == _State.SCAN:
                # Stop-and-look: stop, settle, grab frame, run YOLO + ArUco,
                # then rotate a small step. Repeat until red is found.
                _stop()
                t.sleep(0.1)

                ok, img = _grab_bgr()
                if ok:
                    _detect_and_update(img, world_model)
                    _scan_aruco_passive(img)

                _log_pose_maybe()

                red = world_model.get('R')
                blue = world_model.get('B')
                red_ready  = red  is not None and red.detection_count  >= _SCAN_MIN_RED_DETECTIONS
                blue_ready = blue is not None and blue.detection_count >= _SCAN_MIN_BLUE_DETECTIONS
                scan_elapsed = t.monotonic() - scan_start

                if red_ready and (blue_ready or scan_elapsed >= _SCAN_START_TIMEOUT_S):
                    _stop()
                    r = world_model.get('R')
                    b = world_model.get('B')
                    blue_txt = f"({b.x:.2f},{b.y:.2f})" if b else "unknown"
                    print(f"% FinalBallMission: scan ready — "
                          f"R=({r.x:.2f},{r.y:.2f})  B={blue_txt}  "
                          f"after {scan_elapsed:.1f} s")
                    state = _State.NAVIGATE_TO_RED
                    _logger.log_state(state.name)
                else:
                    service.send("robobot/cmd/ti",
                                 f"rc 0 {_SCAN_PULSE_VEL:.3f}")
                    t.sleep(_SCAN_PULSE_S)
                    _stop()

            # ----------------------------------------------------------
            elif state == _State.NAVIGATE_TO_RED:
                ok = _navigate_to_ball('R', 'B', world_model)
                if ok:
                    print("% FinalBallMission: reached red ball — Phase 1 done")
                    state = _State.DONE
                    _logger.log_state(state.name)
                else:
                    state = _State.FAILED
                    _logger.log_state(state.name)

            # ----------------------------------------------------------
            # ── Phase 2+ states (commented out — re-enable after Phase 1 test) ──
            #
            # elif state == _State.WAIT_RED_PICKUP:
            #     _stop()
            #     _led(30, 0, 30)
            #     print("% FinalBallMission: waiting for red ball pickup…")
            #     # Lower servo for pickup
            #     service.send("robobot/cmd/T0", "servo 1 500 200")
            #     # Drive forward 30 cm to capture ball
            #     # ... (implement servo + drive sequence from CLAUDE.md)
            #     wait_start = t.monotonic()
            #     while not service.stop:
            #         if t.monotonic() - wait_start > _PICKUP_TIMEOUT_S:
            #             state = _State.FAILED; break
            #         ok, img = _grab_bgr()
            #         if ok:
            #             _detect_color(img, 'R', world_model)
            #         if world_model.absent_for('R') >= _PICKUP_ABSENT_S:
            #             world_model.mark_collected('R')
            #             if _logger: _logger.log_pickup('R', t.monotonic() - mission_start)
            #             state = _State.DELIVER_RED; break
            #
            # elif state == _State.DELIVER_RED:
            #     _led(0, 16, 30)
            #     # Navigate to ArUco IDs 10/11 (Quadrant A)
            #     # Use aruco_to_robot_frame for final 30 cm positioning
            #     # Raise servo: service.send("robobot/cmd/T0", "servo 1 -800 300")
            #     # if _logger: _logger.log_deliver('R', 'A')
            #     state = _State.NAVIGATE_TO_BLUE
            #
            # elif state == _State.NAVIGATE_TO_BLUE:
            #     # Lower servo. Navigate to blue ball with no obstacles.
            #     ok = _navigate_to_ball('B', None, world_model)
            #     state = _State.WAIT_BLUE_PICKUP if ok else _State.FAILED
            #
            # elif state == _State.WAIT_BLUE_PICKUP:
            #     _stop()
            #     _led(30, 0, 30)
            #     wait_start = t.monotonic()
            #     while not service.stop:
            #         if t.monotonic() - wait_start > _PICKUP_TIMEOUT_S:
            #             state = _State.FAILED; break
            #         ok, img = _grab_bgr()
            #         if ok:
            #             _detect_color(img, 'B', world_model)
            #         if world_model.absent_for('B') >= _PICKUP_ABSENT_S:
            #             world_model.mark_collected('B')
            #             if _logger: _logger.log_pickup('B', t.monotonic() - mission_start)
            #             state = _State.DELIVER_BLUE; break
            #
            # elif state == _State.DELIVER_BLUE:
            #     _led(0, 16, 30)
            #     # Navigate to ArUco IDs 14/15 (Quadrant C)
            #     # Raise servo to release blue ball
            #     # if _logger: _logger.log_deliver('B', 'C')
            #     state = _State.RETURN_TO_START
            #
            # elif state == _State.RETURN_TO_START:
            #     _led(0, 16, 30)
            #     service.send("robobot/cmd/T0", "servo 1 0 0")  # neutral
            #     ok = _plan_and_drive((x0, y0), [])
            #     state = _State.DONE if ok else _State.FAILED

            elif state == _State.DONE:
                break

            elif state == _State.FAILED:
                break

    finally:
        _stop()
        elapsed = t.monotonic() - mission_start
        success = (state == _State.DONE)
        if _logger is not None:
            _logger.log_done(success, elapsed)
            _logger.close()

    if state == _State.DONE:
        elapsed = t.monotonic() - mission_start
        print(f"% FinalBallMission: complete in {elapsed:.1f} s")
        _led(0, 100, 0)
    else:
        print("% FinalBallMission: failed or aborted")
        _led(100, 0, 0)

    t.sleep(2.0)
    _led(0, 0, 0)
    print("% FinalBallMission: done")
