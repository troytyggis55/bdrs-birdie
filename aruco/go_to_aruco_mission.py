"""
go_to_aruco_mission.py — ArUco-guided navigation.

Public API
----------
go_to_aruco(marker_id, marker_size_m, standoff_m, aruco_dict)
    Navigate to *standoff_m* in front of the specified ArUco marker.
    The marker must be visible in the current field of view; if it is not,
    the function returns False immediately (no scanning/rotation).
    Replans dynamically if the marker moves during navigation.

go_to_aruco_mission()
    Full mission entry point: rotates in short bursts until the marker
    is found, then calls go_to_aruco() to navigate.
    Wired into mqtt-client.py via --go-to-aruco.

Configuration constants at the top of this file:
    TARGET_MARKER_ID, MARKER_SIZE_M, ARUCO_DICT
"""

import math
import os
import sys
import threading
import time as t
from pathlib import Path

import cv2

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from pathfinding.pathfinding import PlannerConfig, find_safe_start
from pathfinding.realtime_pathfind import RealtimePathfinder

from aruco.detect_aruco import detect_aruco
from CamVision.coord_conversion import aruco_to_robot_frame
from worldmodel.ball_world_model import robot_to_world
from scam import cam
from spose import pose
from uservice import service
from odometry.graph_nav import graph_nav

# ---------------------------------------------------------------------------
# Configuration — edit these before running
# ---------------------------------------------------------------------------

TARGET_MARKER_ID = 53           # ArUco marker ID to seek
MARKER_SIZE_M    = 0.037        # Physical side length of the marker [m]
ARUCO_DICT       = "DICT_4X4_100"  # Must match the dictionary used to print the marker
                                    # DICT_4X4_50 only has IDs 0–49

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

_STANDOFF_M        = 0.15   # default stop distance in front of marker [m]
_SCAN_TURN_VEL     = 0.3    # rotation speed during mission scan [rad/s]
_SCAN_PAUSE_S      = 0.15   # pause after each burst to allow clean detection [s]
_GOAL_SHIFT_M      = 0.10   # replan if marker goal shifts more than this [m]
_MISSION_TIMEOUT_S = 90.0   # hard safety timeout for go_to_aruco_mission [s]
_DEBUG_DIR         = "VisionOutput/Debug_aruco"
_DEBUG_EVERY_N     = 20     # save a debug frame every N frames

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


def _dist(ax, ay, bx, by) -> float:
    return math.hypot(ax - bx, ay - by)


def _preprocess_frame(img):
    """Strip alpha channel if present, then convert Picamera2 RGB → BGR."""
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# World model
# ---------------------------------------------------------------------------

class _ArucoWorldModel:
    """EMA-smoothed world-frame estimate for a single ArUco marker."""

    def __init__(self, alpha: float = 0.3, min_detections: int = 5):
        self._alpha = alpha
        self._min_det = min_detections
        self._lock = threading.Lock()
        self._x: float | None = None
        self._y: float | None = None
        self._fnx: float | None = None
        self._fny: float | None = None
        self._count: int = 0

    def update(self, x_r: float, y_r: float,
               face_nx_r: float, face_ny_r: float,
               robot_x: float, robot_y: float, heading: float) -> None:
        wx, wy = robot_to_world(x_r, y_r, robot_x, robot_y, heading)
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
        if not self.reliable:
            return None
        with self._lock:
            return (self._x + standoff * self._fnx,
                    self._y + standoff * self._fny)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(x=self._x, y=self._y,
                        fnx=self._fnx, fny=self._fny, count=self._count)


# ---------------------------------------------------------------------------
# Vision thread
# ---------------------------------------------------------------------------

def _run_vision_thread(model: _ArucoWorldModel, stop_event: threading.Event,
                       marker_id: int, marker_size_m: float,
                       aruco_dict: str) -> None:
    """Feed ArUco detections into *model* until *stop_event* is set."""
    os.makedirs(_DEBUG_DIR, exist_ok=True)
    frame_idx = 0

    while not stop_event.is_set():
        ok, img, _ = cam.getRawFrame()
        if not ok or img is None:
            t.sleep(0.05)
            continue

        img = _preprocess_frame(img)
        frame_idx += 1
        save_debug = (frame_idx % _DEBUG_EVERY_N == 0)

        markers, _, annotated = detect_aruco(img, dictionary_name=aruco_dict, draw=True)

        target_found = False
        for m in markers:
            if m["id"] != marker_id:
                continue
            target_found = True
            result = aruco_to_robot_frame(m, marker_size_m)
            if result is None:
                continue
            x_r, y_r, face_nx, face_ny = result
            rx, ry, hdg = pose.pose[0], pose.pose[1], pose.pose[2]
            model.update(x_r, y_r, face_nx, face_ny, rx, ry, hdg)

        if save_debug:
            snap = model.snapshot()
            if snap["x"] is not None:
                status = (f"id:{marker_id} {'FOUND' if target_found else 'not seen'}  "
                          f"det:{snap['count']}  "
                          f"pos:({snap['x']:.2f},{snap['y']:.2f})")
            else:
                status = (f"id:{marker_id} {'FOUND' if target_found else 'not seen'}  "
                          f"det:{snap['count']}")
            cv2.putText(annotated, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(f"{_DEBUG_DIR}/{frame_idx:05d}.jpg", annotated)

        t.sleep(0.05)


# ---------------------------------------------------------------------------
# Dynamic navigation
# ---------------------------------------------------------------------------

def _navigate_to_goal(model: _ArucoWorldModel, standoff_m: float) -> bool:
    """
    Drive toward the goal from *model*, replanning whenever the marker
    position shifts more than _GOAL_SHIFT_M.  Returns True on arrival.
    """
    goal = model.goal(standoff_m)
    if goal is None:
        return False

    planner = RealtimePathfinder(goal=goal, cfg=_CFG, replan_cooldown=0.5)
    state = planner.update(find_safe_start(_pos(), goal, [], _CFG), [])

    if not state.solved:
        print("% go_to_aruco: initial planning failed")
        return False

    print(f"% go_to_aruco: path to ({goal[0]:.2f},{goal[1]:.2f}), "
          f"{len(state.path)} waypoints")

    current_path = state.path
    wp_idx = 1

    while not service.stop and wp_idx < len(current_path):
        tx, ty = current_path[wp_idx]
        has_next = wp_idx + 1 < len(current_path)
        nx = current_path[wp_idx + 1][0] if has_next else None
        ny = current_path[wp_idx + 1][1] if has_next else None

        graph_nav._stop_nav.clear()
        nav_thread = threading.Thread(
            target=graph_nav.drive_to,
            args=(tx, ty, nx, ny, not has_next),
            daemon=True,
        )
        nav_thread.start()

        replanned = False
        while nav_thread.is_alive() and not service.stop:
            new_goal = model.goal(standoff_m)
            if new_goal and _dist(*new_goal, *goal) > _GOAL_SHIFT_M:
                graph_nav._stop_nav.set()
                nav_thread.join(timeout=0.5)

                goal = new_goal
                planner.reset(goal=goal)
                eff_start = find_safe_start(_pos(), goal, [], _CFG)
                state = planner.update(eff_start, [], force=True)

                if not state.solved:
                    print("% go_to_aruco: replanning failed after marker move")
                    return False

                print(f"% go_to_aruco: marker moved → replanned to "
                      f"({goal[0]:.2f},{goal[1]:.2f}), {len(state.path)} waypoints")
                current_path = state.path
                wp_idx = 1
                replanned = True
                break

            t.sleep(0.05)

        if not replanned:
            wp_idx += 1

    return wp_idx >= len(current_path)


# ---------------------------------------------------------------------------
# Public library function
# ---------------------------------------------------------------------------

def go_to_aruco(
        marker_id: int,
        marker_size_m: float,
        standoff_m: float = _STANDOFF_M,
        aruco_dict: str = ARUCO_DICT,
        initial_frames: int = 8,
) -> bool:
    """
    Navigate to *standoff_m* in front of the ArUco marker with *marker_id*.

    The marker must be visible in the current field of view — this function
    does NOT rotate to search. If the marker is not found in *initial_frames*
    frames, it returns False immediately.

    Once found, a background vision thread keeps the world model updated.
    If the marker moves during navigation the path is replanned automatically.

    Assumes cam.setup_raw() has already been called.

    Parameters
    ----------
    marker_id     : ArUco marker ID to seek
    marker_size_m : physical side length of the marker [m]
    standoff_m    : stop this far in front of the marker face [m]
    aruco_dict    : ArUco dictionary name (must match how the marker was printed)
    initial_frames: number of frames to attempt detection before giving up

    Returns
    -------
    True on arrival, False if not found or navigation failed.
    """
    model = _ArucoWorldModel(min_detections=3)

    # --- Quick FOV check: no rotation ---
    for _ in range(initial_frames):
        ok, img, _ = cam.getRawFrame()
        if not ok or img is None:
            t.sleep(0.05)
            continue

        img = _preprocess_frame(img)
        markers, _ = detect_aruco(img, dictionary_name=aruco_dict)

        for m in markers:
            if m["id"] != marker_id:
                continue
            result = aruco_to_robot_frame(m, marker_size_m)
            if result is None:
                continue
            x_r, y_r, fnx, fny = result
            rx, ry, hdg = pose.pose[0], pose.pose[1], pose.pose[2]
            model.update(x_r, y_r, fnx, fny, rx, ry, hdg)

        if model.reliable:
            break
        t.sleep(0.05)

    if not model.reliable:
        print(f"% go_to_aruco: marker {marker_id} not in FOV — returning")
        return False

    snap = model.snapshot()
    print(f"% go_to_aruco: marker {marker_id} at world "
          f"({snap['x']:.2f},{snap['y']:.2f}), navigating…")

    # --- Navigate with live vision thread ---
    vision_stop = threading.Event()
    vision_thread = threading.Thread(
        target=_run_vision_thread,
        args=(model, vision_stop, marker_id, marker_size_m, aruco_dict),
        daemon=True,
    )
    vision_thread.start()

    try:
        success = _navigate_to_goal(model, standoff_m)
    finally:
        vision_stop.set()
        vision_thread.join(timeout=1.0)
        _stop()

    return success


# ---------------------------------------------------------------------------
# Mission entry point  (called from mqtt-client.py --go-to-aruco)
# ---------------------------------------------------------------------------

def go_to_aruco_mission() -> None:
    """
    Rotate in short bursts until TARGET_MARKER_ID is found, then call
    go_to_aruco() to navigate to it.
    """
    print(f"% GoToArucoMission: starting  target_id={TARGET_MARKER_ID}  "
          f"marker_size={MARKER_SIZE_M} m  dict={ARUCO_DICT}")
    _led(0, 16, 30)

    service.send("robobot/cmd/T0", "enc0")
    t.sleep(0.2)
    cam.setup_raw()

    mission_start = t.monotonic()
    success = False

    try:
        while not service.stop:
            if t.monotonic() - mission_start > _MISSION_TIMEOUT_S:
                print("% GoToArucoMission: timeout")
                break

            # Try from current FOV — rotate between attempts if not found.
            if go_to_aruco(TARGET_MARKER_ID, MARKER_SIZE_M, _STANDOFF_M):
                success = True
                break

            # Marker not visible — short rotation burst then try again.
            service.send("robobot/cmd/ti", f"rc 0 {_SCAN_TURN_VEL:.3f}")
            t.sleep(0.3)
            _stop()
            t.sleep(_SCAN_PAUSE_S)

    finally:
        _stop()

    if success:
        elapsed = t.monotonic() - mission_start
        print(f"% GoToArucoMission: complete in {elapsed:.1f} s")
        _led(0, 100, 0)
    else:
        print("% GoToArucoMission: failed or aborted")
        _led(100, 0, 0)

    t.sleep(2.0)
    _led(0, 0, 0)
    print("% GoToArucoMission: done")
