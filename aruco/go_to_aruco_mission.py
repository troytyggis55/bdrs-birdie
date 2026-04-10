"""
Go-to-ArUco mission.

The robot rotates in place until it finds an ArUco marker with a
predetermined ID. Once the marker position is stable (enough detections),
it pathfinds to a goal 40 cm directly in front of the marker face and stops.

Configuration: set TARGET_MARKER_ID and MARKER_SIZE_M below.

Run via mqtt-client.py (add a --go-to-aruco flag or call
go_to_aruco_mission() directly).
"""

import math
import sys
import threading
import time as t
from pathlib import Path

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from pathfinding.pathfinding import PlannerConfig, plan_path, find_safe_start
from aruco.detect_aruco import detect_aruco
from aruco.aruco_coords import aruco_to_robot_frame
from worldmodel.ball_world_model import robot_to_world
from scam import cam
from spose import pose
from uservice import service
from odometry.graph_nav import graph_nav

# ---------------------------------------------------------------------------
# Configuration — edit these before running
# ---------------------------------------------------------------------------

TARGET_MARKER_ID = 53       # ArUco marker ID to seek
MARKER_SIZE_M    = 0.037    # Physical side length of the marker [m]

# ---------------------------------------------------------------------------
# Pathfinding parameters
# ---------------------------------------------------------------------------

_CFG = PlannerConfig(
    delta=0.15,
    goal_tolerance=0.12,
    clearance=0.00,
    robot_radius=0.12,
    max_steps=3000,
    smooth_path=True,
)

_STANDOFF_M       = 0.40   # stop this far in front of the marker [m]
_SCAN_TURN_VEL    = 0.3    # turn rate while searching [rad/s]
_MIN_DETECTIONS   = 5      # detections needed before position is trusted
_EMA_ALPHA        = 0.3    # weight of each new measurement
_MISSION_TIMEOUT_S = 90.0  # hard safety timeout [s]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pos() -> tuple[float, float]:
    return pose.pose[0], pose.pose[1]


def _heading() -> float:
    return pose.pose[2]


def _stop() -> None:
    service.send("robobot/cmd/ti", "rc 0 0")


def _led(r: int, g: int, b: int) -> None:
    service.send("robobot/cmd/T0", f"leds 16 {r} {g} {b}")


# ---------------------------------------------------------------------------
# World-model state (shared between vision thread and mission loop)
# ---------------------------------------------------------------------------

class _ArucoWorldModel:
    """EMA-smoothed world-frame estimate for a single ArUco marker."""

    def __init__(self, alpha: float = _EMA_ALPHA, min_detections: int = _MIN_DETECTIONS):
        self._alpha = alpha
        self._min_det = min_detections
        self._lock = threading.Lock()
        # World-frame state
        self._x: float | None = None
        self._y: float | None = None
        self._fnx: float | None = None   # face normal x (world frame)
        self._fny: float | None = None   # face normal y (world frame)
        self._count: int = 0

    def update(self, x_r: float, y_r: float,
               face_nx_r: float, face_ny_r: float,
               robot_x: float, robot_y: float, heading: float) -> None:
        """Incorporate a new robot-frame detection."""
        # Convert position to world frame
        wx, wy = robot_to_world(x_r, y_r, robot_x, robot_y, heading)

        # Rotate face normal to world frame (direction vector — rotation only)
        wfnx = face_nx_r * math.cos(heading) - face_ny_r * math.sin(heading)
        wfny = face_nx_r * math.sin(heading) + face_ny_r * math.cos(heading)

        with self._lock:
            if self._x is None:
                self._x, self._y = wx, wy
                self._fnx, self._fny = wfnx, wfny
            else:
                a = self._alpha
                self._x   = (1 - a) * self._x   + a * wx
                self._y   = (1 - a) * self._y   + a * wy
                self._fnx = (1 - a) * self._fnx + a * wfnx
                self._fny = (1 - a) * self._fny + a * wfny
                # Re-normalise the face normal after EMA
                mag = math.hypot(self._fnx, self._fny)
                if mag > 1e-3:
                    self._fnx /= mag
                    self._fny /= mag
            self._count += 1

    @property
    def reliable(self) -> bool:
        with self._lock:
            return self._count >= self._min_det

    def goal(self, standoff: float = _STANDOFF_M) -> tuple[float, float] | None:
        """
        Return the world-frame goal: *standoff* metres in front of the marker face.
        Returns None if the estimate is not yet reliable.
        """
        if not self.reliable:
            return None
        with self._lock:
            return (
                self._x + standoff * self._fnx,
                self._y + standoff * self._fny,
            )

    def snapshot(self) -> dict:
        with self._lock:
            return dict(x=self._x, y=self._y,
                        fnx=self._fnx, fny=self._fny, count=self._count)


# ---------------------------------------------------------------------------
# Vision thread
# ---------------------------------------------------------------------------

def _run_vision_thread(model: _ArucoWorldModel, stop_event: threading.Event) -> None:
    """Grab frames continuously and feed ArUco detections into the world model."""
    cam.setup_raw()

    while not stop_event.is_set():
        ok, img, _ = cam.getRawFrame()
        if not ok or img is None:
            t.sleep(0.05)
            continue

        markers, _ = detect_aruco(img)

        for m in markers:
            if m["id"] != TARGET_MARKER_ID:
                continue

            result = aruco_to_robot_frame(m, MARKER_SIZE_M)
            if result is None:
                continue

            x_r, y_r, face_nx, face_ny = result
            rx, ry, hdg = pose.pose[0], pose.pose[1], pose.pose[2]
            model.update(x_r, y_r, face_nx, face_ny, rx, ry, hdg)

        t.sleep(0.05)   # ~20 Hz


# ---------------------------------------------------------------------------
# Navigation helper
# ---------------------------------------------------------------------------

def _drive_path(path: list[tuple[float, float]]) -> bool:
    """Drive a list of waypoints. Returns True when all are reached."""
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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def go_to_aruco_mission() -> None:
    """
    Search for ArUco marker TARGET_MARKER_ID and drive to STANDOFF_M in front of it.
    """
    print(f"% GoToArucoMission: starting  target_id={TARGET_MARKER_ID}  "
          f"marker_size={MARKER_SIZE_M} m  standoff={_STANDOFF_M} m")
    _led(0, 16, 30)   # cyan = active

    service.send("robobot/cmd/T0", "enc0")   # reset odometry
    t.sleep(0.2)

    model = _ArucoWorldModel()
    vision_stop = threading.Event()
    vision_thread = threading.Thread(
        target=_run_vision_thread,
        args=(model, vision_stop),
        daemon=True,
    )
    vision_thread.start()

    mission_start = t.monotonic()
    success = False

    try:
        # ── Phase 1: SCAN ──────────────────────────────────────────────
        print("% GoToArucoMission: scanning for marker…")
        while not service.stop:
            if t.monotonic() - mission_start > _MISSION_TIMEOUT_S:
                print("% GoToArucoMission: timeout during scan")
                break

            if model.reliable:
                _stop()
                snap = model.snapshot()
                print(f"% GoToArucoMission: marker found — "
                      f"world=({snap['x']:.2f},{snap['y']:.2f})  "
                      f"normal=({snap['fnx']:.2f},{snap['fny']:.2f})  "
                      f"detections={snap['count']}")
                break

            service.send("robobot/cmd/ti", f"rc 0 {_SCAN_TURN_VEL:.3f}")
            t.sleep(0.1)

        if service.stop or not model.reliable:
            return

        # ── Phase 2: NAVIGATE ──────────────────────────────────────────
        goal = model.goal()
        if goal is None:
            print("% GoToArucoMission: could not compute goal")
            return

        print(f"% GoToArucoMission: navigating to goal ({goal[0]:.2f},{goal[1]:.2f})")

        start = find_safe_start(_pos(), goal, [], _CFG)
        path, solved = plan_path(start, goal, [], _CFG)

        if not solved:
            print("% GoToArucoMission: pathfinding failed")
            return

        print(f"% GoToArucoMission: path planned, {len(path)} waypoints")
        success = _drive_path(path)

    finally:
        vision_stop.set()
        vision_thread.join(timeout=1.0)
        _stop()

    if success:
        elapsed = t.monotonic() - mission_start
        print(f"% GoToArucoMission: reached goal in {elapsed:.1f} s")
        _led(0, 100, 0)   # green = success
    else:
        print("% GoToArucoMission: failed or aborted")
        _led(100, 0, 0)   # red = failure

    t.sleep(2.0)
    _led(0, 0, 0)
    print("% GoToArucoMission: done")
