"""
final_ball_replay.py — Interactive post-run replay of a final_ball_mission log.

Visualizes a top-down world-frame map animated by mission elapsed time.

Usage
-----
    python ball_mission/final_ball_replay.py                   # auto-finds latest log
    python ball_mission/final_ball_replay.py MissionLogs/ball_mission_20240415_123456.log

Controls
--------
    Space          : play / pause
    Left / Right   : step backward / forward by 0.1 s
    , / .          : speed × 0.5 / × 2
    Home / End     : jump to start / end
    q / Esc        : quit

Panels
------
  Left (main map): top-down world-frame view
    • gray dashed trail     — full robot trajectory (BM_POSE)
    • black arrow           — current robot position + heading
    • colored cloud         — all ball world-model estimates (BM_BALL_DET)
    • large colored circle  — latest ball world-model position per color
    • cyan diamonds         — ArUco marker world-frame centroids (BM_ARUCO)
    • orange dashed line    — most recent planned path (BM_PATH / BM_REPLAN)
    • green replanned line  — most recent replan path

  Right (info panel): current state, ball + ArUco positions, event log tail
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec
import numpy as np


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

C_BG        = "#f5f5f5"
C_GRID      = "#dcdcdc"
C_TRAIL     = "#aaaaaa"
C_ROBOT     = "#111111"
C_RED_BALL  = "#e03030"
C_BLUE_BALL = "#2060d0"
C_ARUCO     = "#00aaaa"
C_PATH      = "#e07800"
C_REPLAN    = "#228822"
C_START     = "#1e8840"
C_STATE_BOX = dict(boxstyle="round,pad=0.4", facecolor="white",
                   alpha=0.88, edgecolor="#aaaaaa")

BALL_STYLE = {
    'R': dict(color=C_RED_BALL,  label='Red ball'),
    'B': dict(color=C_BLUE_BALL, label='Blue ball'),
}

ARENA_QUADRANTS = {
    'A': (10, 11),
    'B': (12, 13),
    'C': (14, 15),
    'D': (16, 17),
}
ARUCO_QUADRANT = {mid: q for q, ids in ARENA_QUADRANTS.items() for mid in ids}


# ---------------------------------------------------------------------------
# Log parser
# ---------------------------------------------------------------------------

def _robot_to_world(x_r: float, y_r: float,
                    rx: float, ry: float, hdg: float) -> tuple[float, float]:
    wx = rx + x_r * math.cos(hdg) - y_r * math.sin(hdg)
    wy = ry + x_r * math.sin(hdg) + y_r * math.cos(hdg)
    return wx, wy


def parse_log(path: str) -> dict:
    """
    Parse a ball_mission_*.log file into structured data.

    Returns a dict with keys:
        events     : list of (elapsed_s, event_name, fields_list)
        poses      : list of (elapsed_s, x, y, hdg)
        balls      : dict color → list of (elapsed_s, wx, wy, det_count)
        arucos     : dict marker_id → list of (elapsed_s, wx, wy, fnx_w, fny_w)
        paths      : list of (elapsed_s, kind, [(x,y),...])
        states     : list of (elapsed_s, state_name)
        start      : (elapsed_s, x0, y0, hdg0) or None
        duration   : total log duration in seconds
    """
    events: list = []
    poses: list = []
    balls: dict = defaultdict(list)
    arucos: dict = defaultdict(list)
    paths: list = []
    states: list = []
    start_record = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            tokens = line.split()
            if len(tokens) < 2:
                continue
            try:
                ts = float(tokens[0])
            except ValueError:
                continue
            ev = tokens[1]
            fields = tokens[2:]
            events.append((ts, ev, fields))

            if ev == "BM_START" and len(fields) >= 3:
                x0, y0, hdg = float(fields[0]), float(fields[1]), float(fields[2])
                start_record = (ts, x0, y0, hdg)
                poses.append((ts, x0, y0, hdg))

            elif ev == "BM_POSE" and len(fields) >= 3:
                x, y, hdg = float(fields[0]), float(fields[1]), float(fields[2])
                poses.append((ts, x, y, hdg))

            elif ev == "BM_STATE" and fields:
                states.append((ts, fields[0]))

            elif ev == "BM_BALL_DET" and len(fields) >= 9:
                color = fields[0]
                wx, wy = float(fields[6]), float(fields[7])
                det_count = int(fields[8])
                balls[color].append((ts, wx, wy, det_count))

            elif ev == "BM_ARUCO" and len(fields) >= 8:
                marker_id = int(fields[0])
                x_r, y_r   = float(fields[1]), float(fields[2])
                fnx, fny   = float(fields[3]), float(fields[4])
                rx, ry, hdg = float(fields[5]), float(fields[6]), float(fields[7])
                wx, wy = _robot_to_world(x_r, y_r, rx, ry, hdg)
                # Rotate face normal to world frame
                fnx_w = fnx * math.cos(hdg) - fny * math.sin(hdg)
                fny_w = fnx * math.sin(hdg) + fny * math.cos(hdg)
                arucos[marker_id].append((ts, wx, wy, fnx_w, fny_w))

            elif ev in ("BM_PATH", "BM_REPLAN") and fields:
                if ev == "BM_PATH":
                    n = int(fields[0])
                    coords_flat = fields[1:]
                    kind = "plan"
                else:
                    # BM_REPLAN: reason eff_sx eff_sy n x1 y1 ...
                    n = int(fields[3])
                    coords_flat = fields[4:]
                    kind = "replan"
                pts = []
                for i in range(0, min(n * 2, len(coords_flat)), 2):
                    pts.append((float(coords_flat[i]), float(coords_flat[i + 1])))
                if pts:
                    paths.append((ts, kind, pts))

    duration = events[-1][0] if events else 1.0
    return dict(
        events=events,
        poses=poses,
        balls=dict(balls),
        arucos=dict(arucos),
        paths=paths,
        states=states,
        start=start_record,
        duration=duration,
    )


# ---------------------------------------------------------------------------
# Auto-find latest log
# ---------------------------------------------------------------------------

def _find_latest_log() -> str:
    import glob
    logs = sorted(glob.glob("MissionLogs/ball_mission_*.log"))
    if not logs:
        raise FileNotFoundError("No log files found in MissionLogs/. "
                                "Pass a log path explicitly.")
    return logs[-1]


# ---------------------------------------------------------------------------
# Replay viewer
# ---------------------------------------------------------------------------

class BallMissionReplay:
    _PLAY_SPEED_PRESETS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

    def __init__(self, data: dict, log_path: str) -> None:
        self._d = data
        self._log_path = log_path
        self._t = 0.0            # current replay time [s]
        self._duration = max(data['duration'], 0.1)
        self._playing = False
        self._speed_idx = 2      # index into _PLAY_SPEED_PRESETS (1.0×)
        self._last_wall = 0.0    # wall time of last auto-advance tick

        self._build_figure()
        self._update()

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------

    def _build_figure(self) -> None:
        fig = plt.figure(figsize=(16, 9))
        fig.patch.set_facecolor(C_BG)
        gs = GridSpec(
            2, 3,
            figure=fig,
            left=0.05, right=0.97,
            top=0.95, bottom=0.10,
            wspace=0.25, hspace=0.08,
            height_ratios=[8, 1],
            width_ratios=[5, 2, 0.01],
        )
        self._fig = fig
        self._ax_map  = fig.add_subplot(gs[0, 0])   # main top-down map
        self._ax_info = fig.add_subplot(gs[0, 1])   # text info panel

        # Slider row
        ax_slider = fig.add_axes([0.10, 0.04, 0.62, 0.025])
        ax_play   = fig.add_axes([0.76, 0.03, 0.07, 0.045])
        ax_slower = fig.add_axes([0.84, 0.03, 0.06, 0.045])
        ax_faster = fig.add_axes([0.91, 0.03, 0.06, 0.045])

        self._slider = Slider(ax_slider, 'Time (s)', 0.0, self._duration,
                              valinit=0.0, color='#88aacc')
        self._slider.on_changed(self._on_slider)

        self._btn_play   = Button(ax_play,   'Play',  color='#d0f0d0')
        self._btn_slower = Button(ax_slower, '÷2',    color='#f0f0d0')
        self._btn_faster = Button(ax_faster, '×2',    color='#f0f0d0')
        self._btn_play.on_clicked(self._on_play)
        self._btn_slower.on_clicked(self._on_slower)
        self._btn_faster.on_clicked(self._on_faster)

        fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._timer = fig.canvas.new_timer(interval=50)   # 20 Hz tick
        self._timer.add_callback(self._tick)
        self._timer.start()

        title = f"FinalBallMission Replay — {Path(self._log_path).name}"
        fig.suptitle(title, fontsize=11, fontfamily='monospace')

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_slider(self, val: float) -> None:
        self._t = float(val)
        self._update()

    def _on_play(self, _) -> None:
        self._playing = not self._playing
        self._btn_play.label.set_text('Pause' if self._playing else 'Play')
        self._last_wall = time.monotonic()
        self._fig.canvas.draw_idle()

    def _on_slower(self, _) -> None:
        self._speed_idx = max(0, self._speed_idx - 1)
        self._fig.canvas.draw_idle()

    def _on_faster(self, _) -> None:
        self._speed_idx = min(len(self._PLAY_SPEED_PRESETS) - 1,
                              self._speed_idx + 1)
        self._fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        key = event.key
        if key in ('q', 'escape'):
            plt.close('all')
        elif key == ' ':
            self._on_play(None)
        elif key == 'left':
            self._set_time(self._t - 0.1)
        elif key == 'right':
            self._set_time(self._t + 0.1)
        elif key == 'home':
            self._set_time(0.0)
        elif key == 'end':
            self._set_time(self._duration)
        elif key == ',':
            self._on_slower(None)
        elif key == '.':
            self._on_faster(None)

    def _set_time(self, t: float) -> None:
        self._t = max(0.0, min(self._duration, t))
        self._slider.set_val(self._t)   # triggers _on_slider → _update

    def _tick(self) -> None:
        if not self._playing:
            return
        now = time.monotonic()
        speed = self._PLAY_SPEED_PRESETS[self._speed_idx]
        dt = (now - self._last_wall) * speed
        self._last_wall = now
        new_t = self._t + dt
        if new_t >= self._duration:
            new_t = self._duration
            self._playing = False
            self._btn_play.label.set_text('Play')
        self._set_time(new_t)

    # ------------------------------------------------------------------
    # Data queries (everything ≤ self._t)
    # ------------------------------------------------------------------

    def _poses_up_to(self) -> list:
        return [p for p in self._d['poses'] if p[0] <= self._t]

    def _current_pose(self):
        poses = self._poses_up_to()
        return poses[-1] if poses else None

    def _balls_up_to(self) -> dict:
        """Latest world model estimate per color at current time."""
        result = {}
        for color, entries in self._d['balls'].items():
            seen = [(ts, wx, wy, dc) for ts, wx, wy, dc in entries if ts <= self._t]
            if seen:
                result[color] = seen   # keep all for scatter; last for icon
        return result

    def _arucos_up_to(self) -> dict:
        """Best (latest) world-frame position per marker_id at current time."""
        result = {}
        for mid, entries in self._d['arucos'].items():
            seen = [(ts, wx, wy, fnxw, fnyw)
                    for ts, wx, wy, fnxw, fnyw in entries if ts <= self._t]
            if seen:
                result[mid] = seen[-1]   # latest estimate
        return result

    def _current_path(self):
        """Most recent path or replan at current time."""
        active = [(ts, kind, pts) for ts, kind, pts in self._d['paths'] if ts <= self._t]
        return active[-1] if active else None

    def _current_state(self) -> str:
        seen = [(ts, s) for ts, s in self._d['states'] if ts <= self._t]
        return seen[-1][1] if seen else "—"

    def _recent_events(self, n: int = 12) -> list[str]:
        """Last *n* events up to current time."""
        seen = [(ts, ev, flds) for ts, ev, flds in self._d['events']
                if ts <= self._t and not ev.startswith('#')]
        tail = seen[-n:]
        lines = []
        for ts, ev, flds in tail:
            short_flds = ' '.join(flds[:4])
            if len(flds) > 4:
                short_flds += ' …'
            lines.append(f"{ts:6.2f}  {ev:<18} {short_flds}")
        return lines

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _update(self) -> None:
        self._draw_map()
        self._draw_info()
        self._fig.canvas.draw_idle()

    def _draw_map(self) -> None:
        ax = self._ax_map
        ax.cla()
        ax.set_facecolor(C_BG)
        ax.set_aspect('equal')
        ax.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
        ax.set_xlabel('World X [m] (forward from start)', fontsize=9)
        ax.set_ylabel('World Y [m] (left from start)', fontsize=9)
        ax.set_title(f'Top-down map   t = {self._t:.2f} s', fontsize=10)

        all_x, all_y = [0.0], [0.0]

        # ── Robot trail ──────────────────────────────────────────────
        poses = self._poses_up_to()
        if len(poses) >= 2:
            px = [p[1] for p in poses]
            py = [p[2] for p in poses]
            ax.plot(px, py, '--', color=C_TRAIL, linewidth=1.0,
                    zorder=1, label='Robot trail')
            all_x.extend(px)
            all_y.extend(py)

        # ── Current robot position + heading ─────────────────────────
        cp = self._current_pose()
        if cp:
            _, rx, ry, hdg = cp
            L = 0.15   # arrow length [m]
            dx = L * math.cos(hdg)
            dy = L * math.sin(hdg)
            ax.annotate(
                '', xy=(rx + dx, ry + dy), xytext=(rx, ry),
                arrowprops=dict(arrowstyle='->', color=C_ROBOT, lw=2.5),
                zorder=5,
            )
            ax.plot(rx, ry, 'o', color=C_ROBOT, markersize=8, zorder=6)
            # Robot footprint circle
            ax.add_patch(mpatches.Circle(
                (rx, ry), 0.12, fill=False, edgecolor=C_ROBOT,
                linewidth=1.0, linestyle=':', zorder=4,
            ))
            all_x.append(rx); all_y.append(ry)

        # ── Start marker ─────────────────────────────────────────────
        if self._d['start']:
            _, sx, sy, shdg = self._d['start']
            ax.plot(sx, sy, '*', color=C_START, markersize=14, zorder=7,
                    label='Start')
            ax.annotate('start', (sx, sy),
                        xytext=(sx + 0.06, sy + 0.06),
                        fontsize=8, color=C_START, fontweight='bold')

        # ── Ball scatter + world model icons ─────────────────────────
        balls = self._balls_up_to()
        for color, entries in balls.items():
            style = BALL_STYLE.get(color, dict(color='gray', label='?'))
            # All detection estimates as small scatter
            wxs = [e[1] for e in entries]
            wys = [e[2] for e in entries]
            ax.scatter(wxs, wys, color=style['color'], s=25, alpha=0.35,
                       zorder=3, edgecolors='none')
            # Latest estimate as large icon
            wx, wy = entries[-1][1], entries[-1][2]
            det_count = entries[-1][3]
            ax.plot(wx, wy, 'o', color=style['color'], markersize=14,
                    zorder=6, alpha=0.9,
                    markeredgecolor='black', markeredgewidth=0.8,
                    label=f"{style['label']} (n={det_count})")
            ax.annotate(f"  {color}", (wx, wy),
                        fontsize=9, color=style['color'], fontweight='bold')
            all_x.extend(wxs); all_y.extend(wys)

        # ── ArUco marker world positions ─────────────────────────────
        arucos = self._arucos_up_to()
        for mid, (ts, wx, wy, fnxw, fnyw) in arucos.items():
            quadrant = ARUCO_QUADRANT.get(mid, '?')
            ax.plot(wx, wy, 'D', color=C_ARUCO, markersize=10, zorder=6,
                    markeredgecolor='black', markeredgewidth=0.7)
            ax.annotate(f"  id:{mid}({quadrant})", (wx, wy),
                        fontsize=8, color=C_ARUCO)
            # Face normal arrow (30 cm)
            ax.annotate(
                '', xy=(wx + 0.30 * fnxw, wy + 0.30 * fnyw),
                xytext=(wx, wy),
                arrowprops=dict(arrowstyle='->', color=C_ARUCO, lw=1.5),
                zorder=5,
            )
            all_x.append(wx); all_y.append(wy)

        # ── Planned path ──────────────────────────────────────────────
        path_entry = self._current_path()
        if path_entry:
            _, kind, pts = path_entry
            color = C_REPLAN if kind == 'replan' else C_PATH
            label = 'Replanned path' if kind == 'replan' else 'Planned path'
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, '--', color=color, linewidth=2.0, zorder=4,
                    label=label, alpha=0.8)
            ax.plot(xs[1:-1], ys[1:-1], 'o', color=color, markersize=4,
                    zorder=4, alpha=0.7)
            if pts:
                ax.plot(*pts[-1], 's', color=color, markersize=9, zorder=5)
            all_x.extend(xs); all_y.extend(ys)

        # ── State label ───────────────────────────────────────────────
        state = self._current_state()
        speed = self._PLAY_SPEED_PRESETS[self._speed_idx]
        status_txt = (f"State: {state}\n"
                      f"Speed: {speed:.2g}×   "
                      f"{'▶ PLAYING' if self._playing else '⏸ PAUSED'}")
        ax.text(0.02, 0.98, status_txt,
                transform=ax.transAxes, verticalalignment='top',
                fontsize=9, fontfamily='monospace', bbox=C_STATE_BOX, zorder=10)

        # ── Auto-fit view ─────────────────────────────────────────────
        if len(all_x) > 1:
            xmin, xmax = min(all_x), max(all_x)
            ymin, ymax = min(all_y), max(all_y)
            span_x = max(xmax - xmin, 0.5)
            span_y = max(ymax - ymin, 0.5)
            pad = max(span_x, span_y) * 0.20 + 0.30
            ax.set_xlim(xmin - pad, xmax + pad)
            ax.set_ylim(ymin - pad, ymax + pad)

        # ── Legend ────────────────────────────────────────────────────
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=8, loc='lower right',
                      framealpha=0.85)

    def _draw_info(self) -> None:
        ax = self._ax_info
        ax.cla()
        ax.set_facecolor(C_BG)
        ax.axis('off')
        ax.set_title('Event log (tail)', fontsize=9, loc='left')

        # Ball world-model summary
        lines = []
        balls = self._balls_up_to()
        for color in ('R', 'B'):
            name = 'Red' if color == 'R' else 'Blue'
            if color in balls:
                e = balls[color][-1]
                lines.append(f"{name}: ({e[1]:.2f}, {e[2]:.2f})  n={e[3]}")
            else:
                lines.append(f"{name}: not seen")

        arucos = self._arucos_up_to()
        if arucos:
            ids = sorted(arucos.keys())
            lines.append(f"ArUco seen: {ids}")
            for mid in ids:
                ts, wx, wy, _, _ = arucos[mid]
                q = ARUCO_QUADRANT.get(mid, '?')
                lines.append(f"  id:{mid}({q})  ({wx:.2f},{wy:.2f})")
        else:
            lines.append("ArUco: none seen")

        lines.append("")
        lines.append(f"t = {self._t:.2f} / {self._duration:.2f} s")
        lines.append(f"State: {self._current_state()}")
        lines.append("")
        lines.append("──── recent events ────")
        lines.extend(self._recent_events(14))

        text = "\n".join(lines)
        ax.text(0.02, 0.98, text,
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=7.5,
                fontfamily='monospace',
                wrap=True,
                clip_on=True)

    # ------------------------------------------------------------------

    def run(self) -> None:
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Interactive replay of a final_ball_mission log.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        'log_path', nargs='?', default=None,
        help='Path to ball_mission_*.log file (default: latest in MissionLogs/)',
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    log_path = args.log_path or _find_latest_log()
    log_path = str(Path(log_path))
    print(f"Loading: {log_path}")

    data = parse_log(log_path)
    n_poses  = len(data['poses'])
    n_balls  = sum(len(v) for v in data['balls'].values())
    n_aruco  = sum(len(v) for v in data['arucos'].values())
    n_paths  = len(data['paths'])
    print(f"  poses={n_poses}  ball_dets={n_balls}  aruco_dets={n_aruco}  "
          f"paths={n_paths}  duration={data['duration']:.1f}s")

    if not data['events']:
        print("Log is empty — nothing to replay.")
        return

    BallMissionReplay(data, log_path).run()


if __name__ == '__main__':
    main()
