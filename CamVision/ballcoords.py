"""
ballcoords.py — Pixel detections → robot-frame 3D coordinates

Usage:
    from CamVision.ballcoords import pixels_to_robot_coords, undistort_image

    detections = [(x1, y1, r1), (x2, y2, r2), ...]
    coords = pixels_to_robot_coords(detections)
    # → [(x_r1, y_r1, z_r1), ...]   (meters, robot frame)

All calibration parameters are at the top of this file.
Replace the placeholder values once you run cv2.calibrateCamera()
and have measured the physical setup.
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Calibration parameters — replace with measured / calibrated values
# ---------------------------------------------------------------------------

# Camera intrinsics (pixels).
F_X = 625.48093855   # focal length, horizontal  [px]
F_Y = 622.73549921   # focal length, vertical    [px]
C_X = 406.10757421   # optical center, x         [px]
C_Y = 324.93561895   # optical center, y         [px]

# Distortion coefficients [k1, k2, p1, p2, k3] from cv2.calibrateCamera().
DIST_COEFFS = np.array([-6.83778602e-02,  9.19338829e-01,
                          5.58258911e-04, -1.68054345e-03,
                         -1.11717796e+00])

# Physical ball *RADIUS* [m].
# Measure the actual ball with a ruler.
R_REAL = 0.025   # 3 cm — placeholder

# Camera mounting on the robot.
# PHI: downward tilt of the camera from horizontal [radians].
# C_XR: forward offset of camera from robot origin [m].
# C_ZR: vertical offset of camera from robot origin [m].
PHI  = np.radians(11)   # ~11° tilt downward
C_XR = 0.0             # camera is 5 cm in front of robot origin  [m]
C_ZR = 0.18             # camera is 10 cm above robot origin        [m]

# Camera matrix (built once for use with cv2 functions).
_CAMERA_MATRIX = np.array([[F_X,  0,  C_X],
                            [ 0,  F_Y, C_Y],
                            [ 0,   0,   1 ]], dtype=float)

# ---------------------------------------------------------------------------
# Image undistortion utility
# ---------------------------------------------------------------------------

def undistort_image(img, alpha=1):
    """
    Return a lens-undistorted copy of *img*.

    Parameters
    ----------
    img   : numpy array (H×W×C) as returned by cv2.imread / Picamera2
    alpha : 0 → crop to valid pixels only; 1 → keep all pixels (default)

    Returns
    -------
    undistorted image (numpy array, same dtype as input)
    """
    h, w = img.shape[:2]
    new_cam, roi = cv2.getOptimalNewCameraMatrix(
        _CAMERA_MATRIX, DIST_COEFFS, (w, h), alpha, (w, h)
    )
    out = cv2.undistort(img, _CAMERA_MATRIX, DIST_COEFFS, None, new_cam)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        out = out[y:y + rh, x:x + rw]
    return out

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
                           r_real=R_REAL, phi=PHI, c_xr=C_XR, c_zr=C_ZR,
                           undistort=True):
    """
    Convert a list of pixel detections to robot-frame 3D positions.

    Parameters
    ----------
    detections : list of (x_pixel, y_pixel, r_pixel)
        x_pixel, y_pixel — ball center in image pixels
        r_pixel          — ball radius in pixels (use min(w, h)/2 from bbox,
                           or cv2.minEnclosingCircle for better accuracy)

    Optional overrides (use module-level defaults if omitted):
        f_x, f_y    — focal lengths [px]
        c_x, c_y    — optical center [px]
        r_real      — physical ball radius [m]
        phi         — camera tilt downward [radians]
        c_xr        — camera forward offset from robot origin [m]
        c_zr        — camera vertical offset from robot origin [m]
        undistort   — apply lens distortion correction to pixel coords (default True)

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

    if not detections:
        return []

    # Optionally undistort all pixel centers at once before projection.
    centers = np.array([[x, y] for (x, y, _) in detections], dtype=float)
    if undistort:
        # cv2.undistortPoints normalises to camera coords; pass P to get px back.
        cam_mat = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]], dtype=float)
        pts = centers.reshape(-1, 1, 2)
        undistorted = cv2.undistortPoints(pts, cam_mat, DIST_COEFFS, P=cam_mat)
        centers = undistorted.reshape(-1, 2)

    results = []
    for i, (_, __, r_px) in enumerate(detections):
        if r_px <= 0:
            continue
        x_px, y_px = centers[i]

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
