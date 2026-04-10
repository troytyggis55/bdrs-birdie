"""
Simulated ball delivery mission.

Setup
-----
- A virtual goal zone is placed 2 m ahead of the robot at start.
- A red ball and a blue ball are placed (by hand) between the robot and the goal.

Sequence
--------
1. SCAN  — rotate until both balls are localised via YOLO.
2. Approach red ball (blue = pathfinding obstacle), stop 20 cm in front.
3. WAIT  — camera watches; when red ball disappears for 3 s the human has
           "picked it up".  Robot drives to goal zone (blue still an obstacle).
4. Approach blue ball (no obstacles), stop 20 cm in front.
5. WAIT  — same pickup logic. Robot drives to goal zone. Mission complete.

Run via mqtt-client.py by adding a --simulate-ball flag (or call
simulate_ball_mission() directly).
"""

import math
import sys
import threading
import time as t
from enum import Enum, auto
from pathlib import Path

# Make sibling packages importable regardless of working directory.
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))   # pathfinding/
sys.path.insert(0, str(_REPO))                    # repo root

from pathfinding import CircleObstacle, PlannerConfig, find_safe_start
from realtime_pathfind import RealtimePathfinder

from worldmodel.ball_world_model import BallWorldModel
from CamVision.visualcontrol import localize_ball_yolo
from CamVision.ballcoords import pixels_to_robot_coords
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
_STANDOFF_M       = 0.20   # stop this far in front of a ball [m]
_OBS_BALL_RADIUS  = 0.08   # obstacle radius used in pathfinder for balls [m]
_PICKUP_ABSENT_S  = 3.0    # ball must be absent this long to count as picked up [s]
_PICKUP_TIMEOUT_S = 30.0   # max wait for human pickup before aborting [s]
_SCAN_TURN_VEL    = 0.3    # turn rate during SCAN phase [rad/s]
_MISSION_TIMEOUT_S = 120.0 # hard safety timeout for entire mission [s]


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


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def _stop() -> None:
    service.send("robobot/cmd/ti", "rc 0 0")


def _led(r: int, g: int, b: int) -> None:
    service.send("robobot/cmd/T0", f"leds 16 {r} {g} {b}")


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


# ---------------------------------------------------------------------------
# Vision thread
# ---------------------------------------------------------------------------

def _run_vision_thread(world_model: BallWorldModel,
                       stop_event: threading.Event) -> None:
    """
    Continuously grab frames and update the world model with YOLO detections.
    Runs as a daemon thread alongside the mission state machine.
    """
    cam.setup_raw()

    while not stop_event.is_set():
        ok, img, _ = cam.getRawFrame()
        if not ok or img is None:
            t.sleep(0.05)
            continue

        rx, ry, hdg = pose.pose[0], pose.pose[1], pose.pose[2]

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

        t.sleep(0.05)   # ~20 Hz vision update


# ---------------------------------------------------------------------------
# Mission states
# ---------------------------------------------------------------------------

class _State(Enum):
    SCAN            = auto()
    NAVIGATE_TO_RED = auto()
    WAIT_RED_PICKUP = auto()
    DELIVER_RED     = auto()
    NAVIGATE_TO_BLUE = auto()
    WAIT_BLUE_PICKUP = auto()
    DELIVER_BLUE    = auto()
    DONE            = auto()
    FAILED          = auto()


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

    x0, y0 = _pos()
    h0 = _heading()
    goal = (
        x0 + _GOAL_DIST_M * math.cos(h0),
        y0 + _GOAL_DIST_M * math.sin(h0),
    )
    print(f"% SimBallMission: start=({x0:.3f},{y0:.3f})  "
          f"hdg={math.degrees(h0):.1f}°  goal=({goal[0]:.3f},{goal[1]:.3f})")

    world_model = BallWorldModel(colors=('R', 'B'))
    vision_stop = threading.Event()
    vision_thread = threading.Thread(
        target=_run_vision_thread,
        args=(world_model, vision_stop),
        daemon=True,
    )
    vision_thread.start()

    mission_start = t.monotonic()
    state = _State.SCAN

    try:
        while not service.stop:
            if t.monotonic() - mission_start > _MISSION_TIMEOUT_S:
                print("% SimBallMission: timeout!")
                state = _State.FAILED
                break

            # ----------------------------------------------------------
            if state == _State.SCAN:
                if world_model.all_reliable():
                    _stop()
                    r = world_model.get('R')
                    b = world_model.get('B')
                    print(f"% SimBallMission: balls found — "
                          f"R=({r.x:.2f},{r.y:.2f})  B=({b.x:.2f},{b.y:.2f})")
                    state = _State.NAVIGATE_TO_RED
                else:
                    service.send("robobot/cmd/ti",
                                 f"rc 0 {_SCAN_TURN_VEL:.3f}")
                    t.sleep(0.1)

            # ----------------------------------------------------------
            elif state == _State.NAVIGATE_TO_RED:
                red  = world_model.get('R')
                blue = world_model.get('B')
                rx, ry = _pos()
                approach = _approach_point(red.x, red.y, rx, ry)
                blue_obs = [CircleObstacle(blue.x, blue.y, _OBS_BALL_RADIUS)]
                print(f"% SimBallMission: navigating to red approach "
                      f"({approach[0]:.2f},{approach[1]:.2f})")
                ok = _plan_and_drive(approach, blue_obs)
                if ok:
                    state = _State.WAIT_RED_PICKUP
                else:
                    state = _State.FAILED

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
                    if world_model.absent_for('R') >= _PICKUP_ABSENT_S:
                        world_model.mark_collected('R')
                        print("% SimBallMission: red ball collected")
                        state = _State.DELIVER_RED
                        break
                    t.sleep(0.1)

            # ----------------------------------------------------------
            elif state == _State.DELIVER_RED:
                _led(0, 16, 30)
                blue = world_model.get('B')
                blue_obs = [CircleObstacle(blue.x, blue.y, _OBS_BALL_RADIUS)]
                print(f"% SimBallMission: delivering red to goal {goal}")
                ok = _plan_and_drive(goal, blue_obs)
                if ok:
                    print("% SimBallMission: red delivered!")
                    state = _State.NAVIGATE_TO_BLUE
                else:
                    state = _State.FAILED

            # ----------------------------------------------------------
            elif state == _State.NAVIGATE_TO_BLUE:
                blue = world_model.get('B')
                rx, ry = _pos()
                approach = _approach_point(blue.x, blue.y, rx, ry)
                print(f"% SimBallMission: navigating to blue approach "
                      f"({approach[0]:.2f},{approach[1]:.2f})")
                ok = _plan_and_drive(approach, [])
                if ok:
                    state = _State.WAIT_BLUE_PICKUP
                else:
                    state = _State.FAILED

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
                    if world_model.absent_for('B') >= _PICKUP_ABSENT_S:
                        world_model.mark_collected('B')
                        print("% SimBallMission: blue ball collected")
                        state = _State.DELIVER_BLUE
                        break
                    t.sleep(0.1)

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
        vision_stop.set()
        vision_thread.join(timeout=1.0)
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
