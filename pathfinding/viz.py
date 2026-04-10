"""
Interactive pathfinding visualizer.

Place start, goal, obstacles, and walls with the mouse. The reactive harness
(RealtimePathfinder) automatically checks whether added/removed obstacles
invalidate the current path and replans if they do.

Controls
--------
  s        : enter Start mode    — next click places the start
  g        : enter Goal mode     — next click places the goal
  o        : enter Obstacle mode — click once for centre, again for radius
  w        : enter Wall mode     — click once for first endpoint, again for second
  d        : enter Delete mode   — click near an obstacle or wall to remove it
  c        : clear all obstacles and walls (keep start / goal)
  r        : force replan from current start
  q / Esc  : quit

Tip: zoom / pan with the normal matplotlib toolbar; the keys above still
work in any zoom level.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

sys.path.insert(0, str(Path(__file__).parent.parent))

from pathfinding.pathfinding import CircleObstacle, Obstacle, PlannerConfig, WallObstacle, bounds_from_scene
from pathfinding.realtime_pathfind import PlannerState, RealtimePathfinder

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_BG = "#f7f7f7"
C_GRID = "#e0e0e0"
C_OBS_FILL = "#4a6ecc"
C_OBS_EDGE = "#1e3a80"
C_CLEARANCE = "#8fa8e0"
C_PATH = "#1a1a1a"
C_PATH_REPLANNED = "#d45500"
C_START = "#1e8840"
C_GOAL = "#c03010"
C_PENDING_CENTER = "#4a6ecc"
C_WALL_FILL = "#7a3a3a"
C_WALL_EDGE = "#4a1010"


class _Mode:
    START = "start"
    GOAL = "goal"
    OBSTACLE = "obstacle"
    WALL = "wall"
    DELETE = "delete"


class PathfindingViz:
    def __init__(self, cfg: PlannerConfig) -> None:
        self.cfg = cfg
        self.mode: str = _Mode.START

        # World state
        self.start: tuple[float, float] | None = None
        self.goal: tuple[float, float] | None = None
        self.obstacles: list[CircleObstacle] = []
        self.walls: list[WallObstacle] = []
        self._pending_center: tuple[float, float] | None = None  # first click in obstacle mode
        self._pending_wall_p1: tuple[float, float] | None = None  # first click in wall mode

        # Planner
        self._planner: RealtimePathfinder | None = None
        self._last_state: PlannerState | None = None

        # Matplotlib
        self._fig: Figure
        self._ax: Axes
        self._fig, self._ax = plt.subplots(figsize=(9, 9))
        self._fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._reset_view(-1.0, 4.0, -1.0, 4.0)
        self._redraw()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_key(self, event) -> None:
        key = event.key
        if key == "s":
            self._set_mode(_Mode.START)
        elif key == "g":
            self._set_mode(_Mode.GOAL)
        elif key == "o":
            self._set_mode(_Mode.OBSTACLE)
        elif key == "w":
            self._set_mode(_Mode.WALL)
        elif key == "d":
            self._set_mode(_Mode.DELETE)
        elif key == "c":
            self.obstacles.clear()
            self.walls.clear()
            self._reset_planner()
            self._recompute()
        elif key == "r":
            self._reset_planner()
            self._recompute()
        elif key in ("q", "escape"):
            plt.close("all")

    def _on_click(self, event) -> None:
        if event.inaxes is not self._ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = event.xdata, event.ydata

        if self.mode == _Mode.START:
            self.start = (x, y)
            self._reset_planner()
            self._recompute()
            self._set_mode(_Mode.GOAL)

        elif self.mode == _Mode.GOAL:
            self.goal = (x, y)
            self._reset_planner()
            self._recompute()
            self._set_mode(_Mode.OBSTACLE)

        elif self.mode == _Mode.OBSTACLE:
            if self._pending_center is None:
                self._pending_center = (x, y)
                self._redraw()
            else:
                r = math.hypot(x - self._pending_center[0], y - self._pending_center[1])
                r = max(r, 0.02)
                self.obstacles.append(CircleObstacle(self._pending_center[0], self._pending_center[1], r))
                self._pending_center = None
                self._recompute()

        elif self.mode == _Mode.WALL:
            if self._pending_wall_p1 is None:
                self._pending_wall_p1 = (x, y)
                self._redraw()
            else:
                self.walls.append(WallObstacle(
                    self._pending_wall_p1[0], self._pending_wall_p1[1], x, y
                ))
                self._pending_wall_p1 = None
                self._recompute()

        elif self.mode == _Mode.DELETE:
            # Find nearest circle obstacle
            ci = self._nearest_circle_idx(x, y)
            c_dist = math.hypot(self.obstacles[ci].x - x, self.obstacles[ci].y - y) if ci is not None else float("inf")

            # Find nearest wall (by closest endpoint)
            wi = self._nearest_wall_idx(x, y)
            w_dist = self._wall_click_dist(self.walls[wi], x, y) if wi is not None else float("inf")

            if ci is not None and c_dist <= w_dist and c_dist <= self.obstacles[ci].r + 0.3:
                self.obstacles.pop(ci)
                self._reset_planner()
                self._recompute()
            elif wi is not None and w_dist <= 0.3:
                self.walls.pop(wi)
                self._reset_planner()
                self._recompute()

    # ------------------------------------------------------------------
    # Planner interaction
    # ------------------------------------------------------------------

    def _reset_planner(self) -> None:
        self._planner = None

    def _recompute(self) -> None:
        if self.start is None or self.goal is None:
            self._redraw()
            return

        if self._planner is None:
            self._planner = RealtimePathfinder(
                goal=self.goal,
                cfg=self.cfg,
                replan_cooldown=0.0,   # no cooldown in interactive viz
            )
        else:
            # Goal may have changed — sync it
            self._planner.goal = self.goal

        self._last_state = self._planner.update(
            robot_pos=self.start,
            obstacles=self.obstacles + self.walls,  # type: ignore[arg-type]
            force=True,
        )
        self._fit_view()
        self._redraw()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _redraw(self) -> None:
        ax = self._ax
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.cla()
        ax.set_facecolor(C_BG)
        ax.set_aspect("equal")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
        ax.set_xlabel("x (m)", fontsize=9)
        ax.set_ylabel("y (m)", fontsize=9)

        # Obstacles (circles)
        for obs in self.obstacles:
            self._draw_obstacle(obs)

        # Walls
        for wall in self.walls:
            self._draw_wall(wall)

        # Pending obstacle centre (first click placed, waiting for radius click)
        if self._pending_center is not None:
            ax.plot(
                *self._pending_center, "o",
                color=C_PENDING_CENTER, markersize=7, zorder=6,
            )
            ax.annotate(
                "click to set radius",
                self._pending_center,
                xytext=(self._pending_center[0] + 0.08, self._pending_center[1] + 0.08),
                fontsize=8, color=C_PENDING_CENTER,
            )

        # Pending wall first endpoint
        if self._pending_wall_p1 is not None:
            ax.plot(
                *self._pending_wall_p1, "s",
                color=C_WALL_EDGE, markersize=7, zorder=6,
            )
            ax.annotate(
                "click to set second endpoint",
                self._pending_wall_p1,
                xytext=(self._pending_wall_p1[0] + 0.08, self._pending_wall_p1[1] + 0.08),
                fontsize=8, color=C_WALL_EDGE,
            )

        # Path
        state = self._last_state
        if state is not None and len(state.path) > 1:
            xs = [p[0] for p in state.path]
            ys = [p[1] for p in state.path]
            color = C_PATH_REPLANNED if state.replanned else C_PATH
            ax.plot(xs, ys, "-", color=color, linewidth=2.5, zorder=4)
            ax.plot(xs[1:-1], ys[1:-1], "o", color=color, markersize=4, zorder=4)

        # Start
        if self.start is not None:
            ax.plot(*self.start, "o", color=C_START, markersize=11, zorder=7)
            ax.add_patch(mpatches.Circle(
                self.start, self.cfg.robot_radius,
                fill=False, edgecolor=C_START, linewidth=1.5, zorder=6,
            ))
            ax.annotate(
                "start", self.start,
                xytext=(self.start[0] + 0.06, self.start[1] + 0.06),
                fontsize=9, color=C_START, fontweight="bold",
            )

        # Goal
        if self.goal is not None:
            ax.plot(*self.goal, "s", color=C_GOAL, markersize=11, zorder=7)
            ax.add_patch(mpatches.Circle(
                self.goal, self.cfg.goal_tol_radius(),
                fill=False, edgecolor=C_GOAL, linewidth=1.5, linestyle="--", zorder=6,
            ))
            ax.annotate(
                "goal", self.goal,
                xytext=(self.goal[0] + 0.06, self.goal[1] + 0.06),
                fontsize=9, color=C_GOAL, fontweight="bold",
            )

        # Status overlay
        ax.text(
            0.02, 0.98, "\n".join(self._status_lines()),
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white", alpha=0.88,
                edgecolor="#cccccc",
            ),
            zorder=10,
        )

        mode_str = self.mode.upper()
        ax.set_title(
            f"Mode: [{mode_str}]    "
            "s=start  g=goal  o=obstacle  w=wall  d=delete  c=clear  r=replan  q=quit",
            fontsize=9,
        )

        self._fig.canvas.draw_idle()

    def _draw_obstacle(self, obs: CircleObstacle) -> None:
        ax = self._ax
        ax.add_patch(mpatches.Circle(
            (obs.x, obs.y), obs.r,
            facecolor=C_OBS_FILL, edgecolor=C_OBS_EDGE,
            linewidth=1.5, alpha=0.75, zorder=3,
        ))
        inflated_r = obs.r + self.cfg.clearance + self.cfg.robot_radius
        ax.add_patch(mpatches.Circle(
            (obs.x, obs.y), inflated_r,
            fill=False, edgecolor=C_CLEARANCE,
            linewidth=1.0, linestyle="--", alpha=0.6, zorder=2,
        ))

    def _draw_wall(self, wall: WallObstacle) -> None:
        ax = self._ax
        ax.plot(
            [wall.x1, wall.x2], [wall.y1, wall.y2],
            color=C_WALL_EDGE, linewidth=3,
            solid_capstyle="round", zorder=3,
        )
        # Clearance boundary: two offset lines perpendicular to the wall
        dx, dy = wall.x2 - wall.x1, wall.y2 - wall.y1
        length = math.hypot(dx, dy)
        if length > 1e-9:
            safe_d = self.cfg.clearance + self.cfg.robot_radius
            nx, ny = -dy / length, dx / length  # unit perpendicular
            for sign in (1.0, -1.0):
                ox, oy = nx * safe_d * sign, ny * safe_d * sign
                ax.plot(
                    [wall.x1 + ox, wall.x2 + ox],
                    [wall.y1 + oy, wall.y2 + oy],
                    color=C_CLEARANCE, linewidth=1.0,
                    linestyle="--", alpha=0.6, zorder=2,
                )

    # ------------------------------------------------------------------
    # Status text
    # ------------------------------------------------------------------

    def _status_lines(self) -> list[str]:
        lines: list[str] = []
        lines.append(f"Circles: {len(self.obstacles)}   Walls: {len(self.walls)}")

        state = self._last_state
        if state is None:
            lines.append("Status:    set start and goal to begin")
        elif not state.solved:
            lines.append("Status:    NO SOLUTION")
        elif state.replanned:
            lines.append(f"Status:    REPLANNED")
            lines.append(f"Reason:    {state.replan_reason}")
            lines.append(f"Waypoints: {len(state.path)}  ({state.path_length:.2f} m)")
        else:
            lines.append("Status:    PATH VALID  (no replan needed)")
            lines.append(f"Waypoints: {len(state.path)}  ({state.path_length:.2f} m)")

        lines.append(f"delta={self.cfg.delta}  r={self.cfg.robot_radius}  clr={self.cfg.clearance}")
        return lines

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        if mode != _Mode.OBSTACLE:
            self._pending_center = None
        if mode != _Mode.WALL:
            self._pending_wall_p1 = None
        self._redraw()

    def _nearest_circle_idx(self, x: float, y: float) -> int | None:
        if not self.obstacles:
            return None
        return min(
            range(len(self.obstacles)),
            key=lambda i: math.hypot(self.obstacles[i].x - x, self.obstacles[i].y - y),
        )

    def _nearest_wall_idx(self, x: float, y: float) -> int | None:
        if not self.walls:
            return None
        return min(
            range(len(self.walls)),
            key=lambda i: self._wall_click_dist(self.walls[i], x, y),
        )

    def _wall_click_dist(self, wall: WallObstacle, x: float, y: float) -> float:
        """Distance from click point to nearest endpoint of the wall."""
        return min(
            math.hypot(wall.x1 - x, wall.y1 - y),
            math.hypot(wall.x2 - x, wall.y2 - y),
        )

    def _fit_view(self) -> None:
        if self.start is None or self.goal is None:
            return
        min_x, max_x, min_y, max_y = bounds_from_scene(self.start, self.goal, self.obstacles + self.walls)  # type: ignore[arg-type]
        # Add a bit more breathing room
        pad_x = max((max_x - min_x) * 0.1, 0.3)
        pad_y = max((max_y - min_y) * 0.1, 0.3)
        self._ax.set_xlim(min_x - pad_x, max_x + pad_x)
        self._ax.set_ylim(min_y - pad_y, max_y + pad_y)

    def _reset_view(self, x0: float, x1: float, y0: float, y1: float) -> None:
        self._ax.set_xlim(x0, x1)
        self._ax.set_ylim(y0, y1)

    def run(self) -> None:
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# PlannerConfig helper (goal tolerance circle radius for drawing)
# ---------------------------------------------------------------------------

def _goal_tol_radius(self: PlannerConfig) -> float:
    return self.goal_tolerance


PlannerConfig.goal_tol_radius = _goal_tol_radius  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive pathfinding visualizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--delta", type=float, default=0.2, help="Grid step size (m)")
    p.add_argument("--goal-tol", type=float, default=0.15, help="Goal tolerance radius (m)")
    p.add_argument("--clearance", type=float, default=0.03, help="Extra safety margin around obstacles (m)")
    p.add_argument("--robot-radius", type=float, default=0.1, help="Robot radius (m)")
    p.add_argument("--max-steps", type=int, default=2000, help="A* expansion limit")
    p.add_argument("--clearance-weight", type=float, default=0.0, help="Obstacle-avoidance bias (0 = off)")
    p.add_argument("--no-smooth", action="store_true", help="Disable path smoothing")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = PlannerConfig(
        delta=args.delta,
        goal_tolerance=args.goal_tol,
        clearance=args.clearance,
        robot_radius=args.robot_radius,
        max_steps=args.max_steps,
        clearance_weight=max(0.0, args.clearance_weight),
        smooth_path=not args.no_smooth,
    )
    PathfindingViz(cfg).run()


if __name__ == "__main__":
    main()
