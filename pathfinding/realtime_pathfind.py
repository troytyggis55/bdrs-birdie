"""
Stateful pathfinding harness for dynamic environments.

Wraps the pure plan_path() snapshot function with:
  - Path validity checking against the current obstacle set
  - Rate-limited replanning to avoid thrashing on noisy updates
  - Segment-prioritised collision checking (nearby segments checked first)

Typical usage in a control loop:

    planner = RealtimePathfinder(goal=(3.0, 2.0), cfg=PlannerConfig())

    while running:
        obstacles = world_model.stable_obstacles()   # from camera / tracker
        state = planner.update(robot_pos=pose.xy, obstacles=obstacles)

        if state.replanned:
            nav.load_waypoints(state.path)
        elif not state.solved:
            handle_no_solution()

Design notes
------------
- `robot_pos` should come from odometry (spose.py). Drift is expected; the
  planner already pads obstacles with clearance + robot_radius.
- `replan_cooldown` prevents back-to-back replans on flickering detections.
  Set to 0.0 in tests / the visualiser where you want immediate feedback.
- Only replans when the current path is blocked. Path *optimisation* on
  obstacle removal is not automatic — call reset() or use force=True.
"""

import time
from dataclasses import dataclass

from pathfinding import CircleObstacle, Obstacle, PlannerConfig, WallObstacle, dist, has_line_of_sight, plan_path


@dataclass
class PlannerState:
    path: list[tuple[float, float]]
    solved: bool
    replanned: bool
    replan_reason: str | None
    timestamp: float

    @property
    def path_length(self) -> float:
        total = 0.0
        for i in range(len(self.path) - 1):
            total += dist(self.path[i], self.path[i + 1])
        return total


class RealtimePathfinder:
    def __init__(
        self,
        goal: tuple[float, float],
        cfg: PlannerConfig,
        replan_cooldown: float = 1.0,
        urgent_lookahead: int = 3,
    ) -> None:
        """
        Args:
            goal: Target position in world coordinates.
            cfg: Planner configuration shared with plan_path().
            replan_cooldown: Minimum seconds between replans (0 = unlimited).
            urgent_lookahead: Number of upcoming path segments to check with
                highest priority. Blocking obstacles here trigger an immediate
                replan even if the cooldown has not elapsed.
        """
        self.goal = goal
        self.cfg = cfg
        self._cooldown = replan_cooldown
        self._lookahead = urgent_lookahead
        self._path: list[tuple[float, float]] = []
        self._solved = False
        self._last_replan: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        robot_pos: tuple[float, float],
        obstacles: list[Obstacle],
        force: bool = False,
    ) -> PlannerState:
        """
        Check whether the current path is still valid; replan if not.

        Args:
            robot_pos: Current robot position from odometry.
            obstacles: Current stable obstacle list.
            force: Bypass cooldown and replan unconditionally.

        Returns:
            PlannerState describing the current path and whether a replan
            occurred this call.
        """
        now = time.monotonic()

        if not self._path:
            return self._do_replan(robot_pos, obstacles, now, "initial plan")

        if force:
            return self._do_replan(robot_pos, obstacles, now, "forced")

        # Urgent check: always run regardless of cooldown
        urgent_blocked, reason = self._check_segments(
            robot_pos, obstacles, urgent_only=True
        )
        if urgent_blocked:
            return self._do_replan(robot_pos, obstacles, now, reason or "urgent block")

        # Non-urgent check: only run once cooldown has elapsed
        if (now - self._last_replan) >= self._cooldown:
            full_blocked, reason = self._check_segments(
                robot_pos, obstacles, urgent_only=False
            )
            if full_blocked:
                return self._do_replan(robot_pos, obstacles, now, reason or "path blocked")

        return PlannerState(self._path, self._solved, False, None, now)

    def reset(self, goal: tuple[float, float] | None = None) -> None:
        """Discard current path so the next update() triggers a fresh plan."""
        if goal is not None:
            self.goal = goal
        self._path = []
        self._solved = False
        self._last_replan = 0.0

    @property
    def current_path(self) -> list[tuple[float, float]]:
        return list(self._path)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_segments(
        self,
        robot_pos: tuple[float, float],
        obstacles: list[Obstacle],
        urgent_only: bool,
    ) -> tuple[bool, str | None]:
        if not self._path:
            return True, "no path"

        seg_start = self._nearest_segment_index(robot_pos)
        n = len(self._path)
        urgent_end = min(seg_start + self._lookahead + 1, n)

        for i in range(seg_start, urgent_end - 1):
            if not has_line_of_sight(self._path[i], self._path[i + 1], obstacles, self.cfg):
                return True, f"obstacle on segment {i} (urgent)"

        if urgent_only:
            return False, None

        for i in range(urgent_end - 1, n - 1):
            if not has_line_of_sight(self._path[i], self._path[i + 1], obstacles, self.cfg):
                return True, f"obstacle on segment {i}"

        return False, None

    def _nearest_segment_index(self, robot_pos: tuple[float, float]) -> int:
        if len(self._path) <= 1:
            return 0
        return min(
            range(len(self._path) - 1),
            key=lambda i: dist(robot_pos, self._path[i]),
        )

    def _do_replan(
        self,
        robot_pos: tuple[float, float],
        obstacles: list[Obstacle],
        now: float,
        reason: str,
    ) -> PlannerState:
        path, solved = plan_path(robot_pos, self.goal, obstacles, self.cfg)
        self._path = path
        self._solved = solved
        self._last_replan = now
        return PlannerState(path, solved, replanned=True, replan_reason=reason, timestamp=now)
