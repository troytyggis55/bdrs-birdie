"""
Pathfinding debug logger.

Writes a timestamped log file capturing all dynamic pathfinding events
during a mission: start conditions, pose stream, obstacles, and replans.

Log format (space-separated, '#' lines are comments):
    timestamp  EVENT  fields...

Events
------
PF_START   goal_x goal_y robot_r clearance obs_r
PF_PATH    n x1 y1 x2 y2 ...         (initial plan)
PF_POSE    x y heading                (periodic ~20 Hz robot position)
PF_IR      ir_m rx ry rh             (IR triggered, before adding obstacle)
PF_OBS     cx cy r safety_r          (virtual obstacle added)
PF_REPLAN  eff_sx eff_sy reason n x1 y1 x2 y2 ...
PF_DONE    1|0                        (1 = goal reached)
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path


class PathfindLogger:
    def __init__(self, log_dir: str = ".") -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(log_dir) / f"pathfind_log.txt"
        self._f = open(path, "w", encoding="utf-8")
        self._write_header()
        print(f"% PathfindLogger: logging to {path}")

    # ------------------------------------------------------------------
    def _write_header(self) -> None:
        self._f.write("# pathfinding debug log\n")
        self._f.write("# timestamp  EVENT  fields\n")
        self._f.write("#\n")
        self._f.write("# PF_START   goal_x goal_y robot_r clearance obs_r\n")
        self._f.write("# PF_PATH    n x1 y1 x2 y2 ...\n")
        self._f.write("# PF_POSE    x y heading\n")
        self._f.write("# PF_IR      ir_m rx ry rh\n")
        self._f.write("# PF_OBS     cx cy r safety_r\n")
        self._f.write("# PF_REPLAN  eff_sx eff_sy reason n x1 y1 x2 y2 ...\n")
        self._f.write("# PF_DONE    1|0\n")
        self._f.write("#\n")

    def _emit(self, event: str, *fields) -> None:
        row = f"{time.time():.4f} {event} " + " ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in fields)
        self._f.write(row + "\n")
        self._f.flush()

    # ------------------------------------------------------------------
    # Event writers
    # ------------------------------------------------------------------

    def log_start(
        self,
        goal: tuple[float, float],
        robot_r: float,
        clearance: float,
        obs_r: float,
    ) -> None:
        self._emit("PF_START", goal[0], goal[1], robot_r, clearance, obs_r)

    def log_path(self, path: list[tuple[float, float]]) -> None:
        fields: list = [len(path)]
        for x, y in path:
            fields += [round(x, 4), round(y, 4)]
        self._emit("PF_PATH", *fields)

    def log_pose(self, x: float, y: float, heading: float) -> None:
        self._emit("PF_POSE", x, y, heading)

    def log_ir(
        self,
        ir_m: float,
        robot_pos: tuple[float, float],
        heading: float,
    ) -> None:
        self._emit("PF_IR", ir_m, robot_pos[0], robot_pos[1], heading)

    def log_obstacle(
        self,
        cx: float,
        cy: float,
        r: float,
        safety_r: float,
    ) -> None:
        self._emit("PF_OBS", cx, cy, r, safety_r)

    def log_replan(
        self,
        eff_start: tuple[float, float],
        reason: str | None,
        path: list[tuple[float, float]],
    ) -> None:
        reason_tok = (reason or "unknown").replace(" ", "_")
        fields: list = [round(eff_start[0], 4), round(eff_start[1], 4), reason_tok, len(path)]
        for x, y in path:
            fields += [round(x, 4), round(y, 4)]
        self._emit("PF_REPLAN", *fields)

    def log_done(self, reached: bool) -> None:
        self._emit("PF_DONE", 1 if reached else 0)

    def close(self) -> None:
        self._f.close()
        print("% PathfindLogger: log closed")
