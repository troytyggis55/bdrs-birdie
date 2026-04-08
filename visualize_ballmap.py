#!/usr/bin/env python3
"""
visualize_ballmap.py — Visualize ball coordinate logs alongside debug images.

For each logged frame, produces a side-by-side figure:
  Left:  debug camera image (annotated frame + color mask)
  Right: 2D top-down coordinate map (robot at origin, balls as colored dots)

Also saves a summary plot showing all ball positions across the entire run.

Usage:
    python visualize_ballmap.py                                          # auto-finds latest run
    python visualize_ballmap.py VisionOutput/BallMap/run_20240408_153042.csv
"""

import sys
import os
import glob
import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEBUG_DIR   = "VisionOutput/Debug_mission"
BALLMAP_DIR = "VisionOutput/BallMap"

# Color styling: CSV stores 'B' or 'R'
BALL_STYLE = {
    'B': dict(color='dodgerblue',  label='Blue ball'),
    'R': dict(color='tomato',      label='Red ball'),
}

# Top-down map axis limits [m] — adjust once you have real calibration data
MAP_X_RANGE = (-1.5, 1.5)   # y_r: left / right
MAP_Y_RANGE = ( 0.0, 3.0)   # x_r: forward distance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_csv():
    csvs = sorted(glob.glob(f"{BALLMAP_DIR}/run_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No run CSV files found in {BALLMAP_DIR}/")
    return csvs[-1]


def load_csv(path):
    """Returns dict: frame_idx (int) → list of {color, x_r, y_r, z_r}"""
    by_frame = defaultdict(list)
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_frame[int(row['frame'])].append({
                'color': row['color'],
                'x_r':   float(row['x_r']),
                'y_r':   float(row['y_r']),
                'z_r':   float(row['z_r']),
            })
    return by_frame


def find_debug_image(frame_idx):
    """Find the debug image saved for a given frame index (any suffix)."""
    pattern = os.path.join(DEBUG_DIR, f"{frame_idx:04d}_*.jpg")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def draw_map(ax, detections, title="Ball positions"):
    """
    Draw a top-down 2D coordinate map on ax.
      Horizontal axis: y_r (robot left = positive, right = negative)
      Vertical axis:   x_r (forward distance from robot)
      Robot at (0, 0), facing up.
    """
    ax.set_xlim(*MAP_X_RANGE)
    ax.set_ylim(*MAP_Y_RANGE)
    ax.set_xlabel("y_r  [m]  ← left · right →")
    ax.set_ylabel("x_r  [m]  (forward)")
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, linestyle='--', alpha=0.4)

    # Robot marker
    ax.plot(0, 0, marker='^', color='black', markersize=12, zorder=5, label='Robot')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')

    for det in detections:
        style = BALL_STYLE.get(det['color'], dict(color='gray', label='?'))
        ax.scatter(det['y_r'], det['x_r'],
                   color=style['color'], s=120, zorder=4,
                   edgecolors='black', linewidths=0.5)
        ax.annotate(f"  {det['x_r']:.2f}m", (det['y_r'], det['x_r']),
                    fontsize=7, color=style['color'])

    # Legend
    handles = [mpatches.Patch(color=s['color'], label=s['label'])
               for s in BALL_STYLE.values()]
    handles.append(plt.Line2D([0], [0], marker='^', color='black',
                               linestyle='None', markersize=8, label='Robot'))
    ax.legend(handles=handles, fontsize=8, loc='upper right')


# ---------------------------------------------------------------------------
# Per-frame combined figures
# ---------------------------------------------------------------------------

def make_frame_figure(frame_idx, detections, img_path, out_dir):
    fig, (ax_img, ax_map) = plt.subplots(1, 2, figsize=(18, 5))
    fig.suptitle(f"Frame {frame_idx:04d}", fontsize=11)

    # Left: camera debug image
    if img_path:
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ax_img.imshow(img_rgb)
        ax_img.set_title(os.path.basename(img_path))
    else:
        ax_img.text(0.5, 0.5, "No image", ha='center', va='center')
        ax_img.set_title("No debug image saved for this frame")
    ax_img.axis('off')

    # Right: coordinate map
    draw_map(ax_map, detections, title=f"Frame {frame_idx:04d} — detected balls")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{frame_idx:04d}_combined.png")
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Summary figure
# ---------------------------------------------------------------------------

def make_summary_figure(by_frame, out_path):
    all_detections = [det for dets in by_frame.values() for det in dets]

    fig, ax = plt.subplots(figsize=(8, 7))
    draw_map(ax, all_detections, title="All ball detections — full run")

    # Annotate frame numbers
    for frame_idx, dets in sorted(by_frame.items()):
        for det in dets:
            ax.annotate(f"f{frame_idx}", (det['y_r'], det['x_r']),
                        fontsize=6, alpha=0.5,
                        xytext=(4, 4), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Summary → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_csv()
    print(f"Loading: {csv_path}")

    by_frame = load_csv(csv_path)
    if not by_frame:
        print("CSV is empty — nothing to visualize.")
        return

    run_id  = os.path.splitext(os.path.basename(csv_path))[0]  # e.g. run_20240408_153042
    out_dir = os.path.join(BALLMAP_DIR, f"viz_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # Per-frame combined images
    for frame_idx, dets in sorted(by_frame.items()):
        img_path  = find_debug_image(frame_idx)
        out_path  = make_frame_figure(frame_idx, dets, img_path, out_dir)
        ball_summary = ", ".join(
            f"{d['color']}({d['x_r']:.2f},{d['y_r']:.2f})" for d in dets
        )
        print(f"  Frame {frame_idx:04d}: {ball_summary}  → {os.path.basename(out_path)}")

    # Summary
    make_summary_figure(by_frame, os.path.join(out_dir, "summary.png"))
    print(f"\nDone. {len(by_frame)} frames visualized.")


if __name__ == "__main__":
    main()
