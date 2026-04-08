"""
ballcoords.py — Pixel detections → robot-frame 3D coordinates

Usage:
    from CamVision.ballcoords import pixels_to_robot_coords

    detections = [(x1, y1, r1), (x2, y2, r2), ...]
    coords = pixels_to_robot_coords(detections)
    # → [(x_r1, y_r1, z_r1), ...]   (meters, robot frame)

All calibration parameters are at the top of this file.
Replace the placeholder values once you run cv2.calibrateCamera()
and have measured the physical setup.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Calibration parameters — replace with measured / calibrated values
# ---------------------------------------------------------------------------

# Camera intrinsics (pixels).
# Get these from cv2.calibrateCamera() with a checkerboard pattern.
# Placeholder: approximate values for Pi Camera v2 at 820×616.
F_X = 625.48093855   # focal length, horizontal  [px]
F_Y = 622.73549921   # focal length, vertical    [px]
C_X = 406.10757421   # optical center, x         [px]  (≈ image_width / 2)
C_Y = 324.93561895   # optical center, y         [px]  (≈ image_height / 2)

# Physical ball radius [m].
# Measure the actual ball with a ruler.
R_REAL = 0.045   # 3 cm — placeholder

# Camera mounting on the robot.
# PHI: downward tilt of the camera from horizontal [radians].
# C_XR: forward offset of camera from robot origin [m].
# C_ZR: vertical offset of camera from robot origin [m].
PHI  = np.radians(11)   # ~11° tilt downward
C_XR = 0.05             # camera is 5 cm in front of robot origin  [m]
C_ZR = 0.10             # camera is 10 cm above robot origin        [m]

# ---------------------------------------------------------------------------
# Pre-built transform matrices (computed once at import time)
# ---------------------------------------------------------------------------

# Frame-change: camera axes → robot axis convention
#   Camera  +Z (forward) → Robot +X
#   Camera  +X (right)   → Robot -Y
#   Camera  +Y (down)    → Robot -Z
_F = np.array([[ 0,  0,  1, 0],
               [-1,  0,  0, 0],
               [ 0, -1,  0, 0],
               [ 0,  0,  0, 1]], dtype=float)

def _build_T(phi, c_xr, c_zr):
    """Rotation (camera pitch) + translation (camera offset on robot)."""
    cp, sp = np.cos(phi), np.sin(phi)
    return np.array([[ cp, 0,  sp, -c_xr],
                     [  0, 1,   0,  0   ],
                     [-sp, 0,  cp, -c_zr],
                     [  0, 0,   0,  1   ]], dtype=float)

_T = _build_T(PHI, C_XR, C_ZR)
_TF = _T @ _F   # combined transform, cached


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pixels_to_robot_coords(detections, f_x=F_X, f_y=F_Y, c_x=C_X, c_y=C_Y,
                           r_real=R_REAL, phi=PHI, c_xr=C_XR, c_zr=C_ZR):
    """
    Convert a list of pixel detections to robot-frame 3D positions.

    Parameters
    ----------
    detections : list of (x_pixel, y_pixel, r_pixel)
        x_pixel, y_pixel — ball center in image pixels
        r_pixel          — ball radius in pixels (use min(w, h)/2 from bbox,
                           or cv2.minEnclosingCircle for better accuracy)

    Optional overrides (use module-level defaults if omitted):
        f_x, f_y   — focal lengths [px]
        c_x, c_y   — optical center [px]
        r_real     — physical ball radius [m]
        phi        — camera tilt downward [radians]
        c_xr       — camera forward offset from robot origin [m]
        c_zr       — camera vertical offset from robot origin [m]

    Returns
    -------
    list of (x_r, y_r, z_r)  — position in robot frame [meters]
        x_r: forward (away from robot)
        y_r: left/right
        z_r: up/down
        Returns an empty list if detections is empty.
    """
    # Rebuild transform only if caller overrides mounting params
    if phi == PHI and c_xr == C_XR and c_zr == C_ZR:
        TF = _TF
    else:
        TF = _build_T(phi, c_xr, c_zr) @ _F

    results = []
    for (x_px, y_px, r_px) in detections:
        if r_px <= 0:
            continue

        # Step 1 — depth from known ball size
        Z_a = (r_real * f_y) / r_px

        # Step 2 — 3D camera coordinates
        X_a = (x_px - c_x) * Z_a / f_x
        Y_a = (y_px - c_y) * Z_a / f_y

        # Step 3 — transform to robot frame
        cam_point = np.array([X_a, Y_a, Z_a, 1.0])
        robot_point = TF @ cam_point
        results.append(tuple(robot_point[:3]))

    return results
