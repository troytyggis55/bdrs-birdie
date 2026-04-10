"""
Dynamic pathfinding integration test mission.

Drives 2 m straight ahead using A* pathfinding via graph_nav.
When the front IR sensor reads < IR_TRIGGER_M a virtual obstacle (r = OBS_RADIUS_M)
is spawned OBS_FORWARD_M in front of the robot's current heading and the path
is immediately replanned via RealtimePathfinder.

All events are written to a timestamped pathfind_log_*.txt file for replay.

Run from mqtt-client.py state 170:
    python mqtt-client.py --pathfinding
"""
import math
import sys
import threading
import time as t
from pathlib import Path

# Make pathfinding modules importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

from pathfinding import CircleObstacle, PlannerConfig, find_safe_start
from realtime_pathfind import RealtimePathfinder
from pathfind_logger import PathfindLogger

from spose import pose
from sir import ir
from uservice import service
from odometry.graph_nav import graph_nav

# ------------------------------------------------------------------
# Mission parameters
# ------------------------------------------------------------------
_CFG = PlannerConfig(
    delta=0.15,
    goal_tolerance=0.12,
    clearance=0.00,
    robot_radius=0.12,
    max_steps=3000,
    smooth_path=True,
)
_MISSION_DIST_M   = 2.0    # how far to drive (metres)
_IR_TRIGGER_M     = 0.10   # IR distance that signals an obstacle (metres)
_OBS_FORWARD_M    = 0.50   # how far ahead to place the virtual obstacle centre
                           # must exceed clearance + robot_radius + obs_radius (=0.26 m) to
                           # ensure the robot's current position is not inside the inflated zone
_OBS_RADIUS_M     = 0.05   # 10 cm radius → 20 cm diameter
_IR_COOLDOWN_S    = 1.5    # minimum seconds between obstacle injections
_MISSION_TIMEOUT_S = 60    # hard safety timeout


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _pos() -> tuple[float, float]:
    return pose.pose[0], pose.pose[1]


def _heading() -> float:
    return pose.pose[2]


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

def pathfinding_mission() -> None:
    """
    Drive 2 m forward with reactive obstacle avoidance.
    Called directly from the mqtt-client state machine.
    """
    service.send("robobot/cmd/T0", "leds 16 0 30 60")   # cyan = pathfinding active
    print("% PathfindingMission: starting")

    logger = PathfindLogger()

    # ── Record initial pose, compute goal ───────────────────────────
    x0, y0 = _pos()
    h0 = _heading()
    goal = (
        x0 + _MISSION_DIST_M * math.cos(h0),
        y0 + _MISSION_DIST_M * math.sin(h0),
    )
    print(
        f"% PathfindingMission: start=({x0:.3f}, {y0:.3f})  "
        f"hdg={math.degrees(h0):.1f}°  "
        f"goal=({goal[0]:.3f}, {goal[1]:.3f})"
    )
    logger.log_start(goal, _CFG.robot_radius, _CFG.clearance, _OBS_RADIUS_M)

    # ── Initial plan ─────────────────────────────────────────────────
    obstacles: list[CircleObstacle] = []
    planner = RealtimePathfinder(goal=goal, cfg=_CFG, replan_cooldown=0.0)
    state = planner.update(_pos(), obstacles)

    if not state.solved:
        print("% PathfindingMission: initial plan failed — aborting")
        service.send("robobot/cmd/ti", "rc 0 0")
        service.send("robobot/cmd/T0", "leds 16 100 0 0")
        logger.log_done(False)
        logger.close()
        return

    print(f"% PathfindingMission: initial path, {len(state.path)} waypoints")
    logger.log_path(state.path)

    # ── Navigation loop ──────────────────────────────────────────────
    current_path = state.path
    wp_idx = 1             # path[0] == current position; start at [1]
    last_obs_t = 0.0
    mission_start = t.monotonic()
    reached = False

    while not service.stop and wp_idx < len(current_path):
        if t.monotonic() - mission_start > _MISSION_TIMEOUT_S:
            print("% PathfindingMission: timeout!")
            break

        tx, ty = current_path[wp_idx]
        has_next = wp_idx + 1 < len(current_path)
        nx = current_path[wp_idx + 1][0] if has_next else None
        ny = current_path[wp_idx + 1][1] if has_next else None
        is_last = not has_next

        # Drive to this waypoint in a background thread so we can
        # monitor IR in parallel
        graph_nav._stop_nav.clear()
        nav_thread = threading.Thread(
            target=graph_nav.drive_to,
            args=(tx, ty, nx, ny, is_last),
            daemon=True,
        )
        nav_thread.start()

        replanned = False
        while nav_thread.is_alive() and not service.stop:
            now = t.monotonic()
            cx, cy = _pos()
            ch = _heading()

            # ── Periodic pose log ───────────────────────────────────
            logger.log_pose(cx, cy, ch)

            ir_valid = ir.irUpdCnt > 0
            ir_close = ir_valid and ir.ir[0] < _IR_TRIGGER_M
            cooled   = (now - last_obs_t) > _IR_COOLDOWN_S

            if ir_close and cooled:
                # ── IR triggered — stop navigation ──────────────────
                graph_nav._stop_nav.set()
                nav_thread.join(timeout=0.5)

                logger.log_ir(ir.ir[0], (cx, cy), ch)

                # Place virtual obstacle in front of robot
                obs = CircleObstacle(
                    x=cx + _OBS_FORWARD_M * math.cos(ch),
                    y=cy + _OBS_FORWARD_M * math.sin(ch),
                    r=_OBS_RADIUS_M,
                )
                obstacles.append(obs)
                last_obs_t = now
                logger.log_obstacle(
                    obs.x, obs.y, obs.r,
                    obs.r + _CFG.clearance + _CFG.robot_radius,
                )
                print(
                    f"% PathfindingMission: IR={ir.ir[0]:.3f} m — virtual obstacle "
                    f"at ({obs.x:.3f}, {obs.y:.3f})  total obs={len(obstacles)}"
                )

                # ── Replan from current position ─────────────────────
                # find_safe_start handles the case where odometry drift has placed
                # the robot inside a previously placed obstacle's safety zone
                eff_start = find_safe_start(_pos(), goal, obstacles, _CFG)
                p_state = planner.update(eff_start, obstacles, force=True)

                if not p_state.solved:
                    print("% PathfindingMission: replanning failed — stopping")
                    service.send("robobot/cmd/ti", "rc 0 0")
                    service.send("robobot/cmd/T0", "leds 16 100 0 0")
                    logger.log_done(False)
                    logger.close()
                    return

                logger.log_replan(eff_start, p_state.replan_reason, p_state.path)
                print(
                    f"% PathfindingMission: replanned ({p_state.replan_reason}), "
                    f"{len(p_state.path)} waypoints, "
                    f"{p_state.path_length:.2f} m"
                )
                current_path = p_state.path
                wp_idx = 1   # new path[0] is current/effective position
                replanned = True
                break        # restart outer loop with new path

            t.sleep(0.05)

        if not replanned:
            wp_idx += 1      # waypoint reached normally

    # ── Mission complete ─────────────────────────────────────────────
    service.send("robobot/cmd/ti", "rc 0 0")
    reached = wp_idx >= len(current_path)

    if reached:
        elapsed = t.monotonic() - mission_start
        print(
            f"% PathfindingMission: goal reached in {elapsed:.1f} s, "
            f"{len(obstacles)} obstacle(s) detected"
        )
        service.send("robobot/cmd/T0", "leds 16 0 100 0")   # green = success
    else:
        print("% PathfindingMission: ended without reaching goal")
        service.send("robobot/cmd/T0", "leds 16 100 30 0")  # orange = partial

    logger.log_done(reached)
    logger.close()

    t.sleep(1.0)
    service.send("robobot/cmd/T0", "leds 16 0 0 0")
    print("% PathfindingMission: done")
