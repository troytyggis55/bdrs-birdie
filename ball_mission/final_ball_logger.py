"""
final_ball_logger.py — Mission event logger for final_ball_mission.py.

Writes a timestamped log to MissionLogs/ that captures all observable
state during the mission: robot pose, ball detections, ArUco sightings,
planned paths, and state transitions.

Log format (space-separated, '#' lines are comments):
    elapsed_s  EVENT  fields...

Events
------
BM_START        x0 y0 heading
BM_STATE        state_name
BM_POSE         x y heading
BM_BALL_DET     color cx cy r_px x_r y_r wx wy det_count
BM_ARUCO        marker_id x_r y_r fnx fny rx ry hdg
BM_POSE_CORRECT old_rx old_ry new_rx new_ry heading
BM_PATH         n x1 y1 x2 y2 ...
BM_REPLAN       reason eff_sx eff_sy n x1 y1 ...
BM_PICKUP       color elapsed_s
BM_DELIVER      color quadrant
BM_DONE         1|0 elapsed_s
BM_FRAME        filename label
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path


class FinalBallLogger:
    """
    Thread-safe mission event logger.

    Each log_*() call writes and flushes a single line, so concurrent
    callers from different threads never produce interleaved output.
    The GIL makes individual write+flush atomic for CPython, which is
    sufficient here (no binary record format).
    """

    def __init__(self, log_dir: str = "MissionLogs") -> None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = Path(log_dir) / f"ball_mission_{ts}.log"
        self._f = open(self._path, "w", encoding="utf-8")
        self._t0 = time.monotonic()
        # Frames directory: sibling to the log file, named <logname>_frames/
        self._frames_dir = self._path.parent / (self._path.stem + "_frames")
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        self._frame_counter = 0
        self._write_header()
        print(f"% FinalBallLogger: logging to {self._path}")
        print(f"% FinalBallLogger: frames  → {self._frames_dir}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_header(self) -> None:
        lines = [
            "# final_ball_mission debug log",
            f"# created {datetime.now().isoformat()}",
            "# elapsed_s  EVENT  fields...",
            "#",
            "# BM_START        x0 y0 heading",
            "# BM_STATE        state_name",
            "# BM_POSE         x y heading",
            "# BM_BALL_DET     color cx cy r_px x_r y_r wx wy det_count",
            "# BM_ARUCO        marker_id x_r y_r fnx fny rx ry hdg",
            "# BM_POSE_CORRECT old_rx old_ry new_rx new_ry heading",
            "# BM_PATH         n x1 y1 x2 y2 ...",
            "# BM_REPLAN       reason eff_sx eff_sy n x1 y1 ...",
            "# BM_PICKUP       color elapsed_s",
            "# BM_DELIVER      color quadrant",
            "# BM_DONE         1|0 elapsed_s",
            "# BM_FRAME        filename label",
            "#",
        ]
        for line in lines:
            self._f.write(line + "\n")
        self._f.flush()

    def _emit(self, event: str, *fields) -> None:
        elapsed = time.monotonic() - self._t0
        parts = [f"{elapsed:.4f}", event]
        for v in fields:
            if isinstance(v, float):
                parts.append(f"{v:.4f}")
            else:
                parts.append(str(v))
        self._f.write(" ".join(parts) + "\n")
        self._f.flush()

    # ------------------------------------------------------------------
    # Event writers
    # ------------------------------------------------------------------

    def log_start(self, x: float, y: float, heading: float) -> None:
        """Log mission start pose. Call once at mission init."""
        self._emit("BM_START", x, y, heading)

    def log_state(self, state_name: str) -> None:
        """Log a state-machine transition."""
        self._emit("BM_STATE", state_name)

    def log_pose(self, x: float, y: float, heading: float) -> None:
        """Log current robot pose (call periodically, ~5 Hz)."""
        self._emit("BM_POSE", x, y, heading)

    def log_ball_det(
        self,
        color: str,
        cx: float,
        cy: float,
        r_px: float,
        x_r: float,
        y_r: float,
        wx: float,
        wy: float,
        det_count: int,
    ) -> None:
        """
        Log a ball detection and current world model estimate.

        Parameters
        ----------
        color     : 'R' or 'B'
        cx, cy    : pixel centroid of the detection
        r_px      : pixel radius of the detected blob
        x_r, y_r  : robot-frame position [m] from coord_conversion
        wx, wy    : EMA-smoothed world-frame position [m] from BallWorldModel
        det_count : total detection count for this color
        """
        self._emit("BM_BALL_DET", color, cx, cy, r_px, x_r, y_r, wx, wy, det_count)

    def log_aruco(
        self,
        marker_id: int,
        x_r: float,
        y_r: float,
        fnx: float,
        fny: float,
        rx: float,
        ry: float,
        hdg: float,
    ) -> None:
        """
        Log a passive ArUco detection.

        Parameters
        ----------
        marker_id     : ArUco marker ID
        x_r, y_r      : marker centre in robot frame [m]
        fnx, fny      : face normal in robot frame (unit vector)
        rx, ry, hdg   : robot pose at detection time
        """
        self._emit("BM_ARUCO", marker_id, x_r, y_r, fnx, fny, rx, ry, hdg)

    def log_pose_correct(
        self,
        old_rx: float,
        old_ry: float,
        new_rx: float,
        new_ry: float,
        heading: float,
    ) -> None:
        """Log an ArUco-derived odometry correction."""
        self._emit("BM_POSE_CORRECT", old_rx, old_ry, new_rx, new_ry, heading)

    def log_path(self, path: list[tuple[float, float]]) -> None:
        """Log a newly planned path (initial plan)."""
        fields: list = [len(path)]
        for x, y in path:
            fields += [round(x, 4), round(y, 4)]
        self._emit("BM_PATH", *fields)

    def log_replan(
        self,
        reason: str,
        eff_sx: float,
        eff_sy: float,
        path: list[tuple[float, float]],
    ) -> None:
        """Log a replanned path."""
        reason_tok = reason.replace(" ", "_")
        fields: list = [reason_tok, eff_sx, eff_sy, len(path)]
        for x, y in path:
            fields += [round(x, 4), round(y, 4)]
        self._emit("BM_REPLAN", *fields)

    def log_pickup(self, color: str, elapsed_s: float) -> None:
        """Log a successful ball pickup confirmation."""
        self._emit("BM_PICKUP", color, elapsed_s)

    def log_deliver(self, color: str, quadrant: str) -> None:
        """Log a ball delivery."""
        self._emit("BM_DELIVER", color, quadrant)

    def log_done(self, success: bool, elapsed_s: float) -> None:
        """Log mission completion."""
        self._emit("BM_DONE", 1 if success else 0, elapsed_s)

    def log_frame(self, img_bgr, label: str = "") -> None:
        """
        Save *img_bgr* as a JPEG to the frames directory and emit a BM_FRAME
        log entry.  Safe to call from background threads.

        Parameters
        ----------
        img_bgr : numpy ndarray (H×W×3, BGR)  — annotated camera frame
        label   : short description tag (e.g. 'track_R', 'fov_edge_R', ...)
        """
        try:
            import cv2 as cv
            self._frame_counter += 1
            fname = f"frame_{self._frame_counter:05d}.jpg"
            fpath = self._frames_dir / fname
            cv.imwrite(str(fpath), img_bgr)
            label_tok = (label or "nostate").replace(" ", "_")
            self._emit("BM_FRAME", fname, label_tok)
        except Exception as exc:
            print(f"% FinalBallLogger: log_frame failed: {exc}")

    # ------------------------------------------------------------------

    def close(self) -> None:
        self._f.close()
        print(f"% FinalBallLogger: closed {self._path}")

    @property
    def path(self) -> Path:
        """Path to the log file (for printing / passing to replay script)."""
        return self._path
