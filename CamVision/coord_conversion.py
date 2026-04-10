"""
coord_conversion.py — Camera pixel detections → robot-frame 3D coordinates.

Handles both ball detections (sphere-radius depth) and ArUco marker pose
(solvePnP), sharing the same camera calibration and rigid transform.

Usage:
    from CamVision.coord_conversion import pixels_to_robot_coords, aruco_to_robot_frame

    # Balls
    coords = pixels_to_robot_coords([(cx, cy, r_px), ...])
    # → [(x_r, y_r, z_r), ...]  in robot frame [m]

    # ArUco
    result = aruco_to_robot_frame(marker, marker_size_m=0.06)
    # → (x_r, y_r, face_nx, face_ny) or None
"""

import math
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Camera calibration parameters
# ---------------------------------------------------------------------------

F_X = 625.48093855   # focal length, horizontal  [px]
F_Y = 622.73549921   # focal length, vertical    [px]
C_X = 406.10757421   # optical center, x         [px]
C_Y = 324.93561895   # optical center, y         [px]

DIST_COEFFS = np.array([-6.83778602e-02,  9.19338829e-01,
                          5.58258911e-04, -1.68054345e-03,
                         -1.11717796e+00])

R_REAL = 0.025   # default physical ball radius [m]

PHI  = np.radians(11)   # camera downward tilt [rad]
C_XR = 0.0              # camera forward offset from robot origin [m]
C_ZR = 0.18             # camera height above robot origin [m]

_CAMERA_MATRIX = np.array([[F_X,  0,  C_X],
                            [ 0,  F_Y, C_Y],
                            [ 0,   0,   1 ]], dtype=float)

# ---------------------------------------------------------------------------
# Rigid transform: camera frame → robot frame (cached at import time)
# ---------------------------------------------------------------------------

# Frame change: camera axes → robot axis convention
#   Camera +Z (forward) → Robot +X
#   Camera +X (right)   → Robot -Y
#   Camera +Y (down)    → Robot -Z
_F = np.array([[ 0,  0,  1, 0],
               [-1,  0,  0, 0],
               [ 0, -1,  0, 0],
               [ 0,  0,  0, 1]], dtype=float)


def _build_T(phi, c_xr, c_zr):
    cp, sp = np.cos(phi), np.sin(phi)
    return np.array([[ cp, 0,  sp, -c_xr],
                     [  0, 1,   0,  0   ],
                     [-sp, 0,  cp, -c_zr],
                     [  0, 0,   0,  1   ]], dtype=float)


_T  = _build_T(PHI, C_XR, C_ZR)
_TF = _T @ _F   # combined camera→robot transform, cached


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def undistort_image(img, alpha=1):
    """Return a lens-undistorted copy of *img*."""
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
# Ball detections → robot frame
# ---------------------------------------------------------------------------

def pixels_to_robot_coords(detections, f_x=F_X, f_y=F_Y, c_x=C_X, c_y=C_Y,
                            r_real=R_REAL, phi=PHI, c_xr=C_XR, c_zr=C_ZR,
                            undistort=True):
    """
    Convert ball pixel detections to robot-frame 3D positions.

    Parameters
    ----------
    detections : list of (x_pixel, y_pixel, r_pixel)

    Returns
    -------
    list of (x_r, y_r, z_r) in robot frame [m]  (x_r = forward, y_r = left)
    """
    if phi == PHI and c_xr == C_XR and c_zr == C_ZR:
        TF = _TF
    else:
        TF = _build_T(phi, c_xr, c_zr) @ _F

    if not detections:
        return []

    centers = np.array([[x, y] for (x, y, _) in detections], dtype=float)
    if undistort:
        cam_mat = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]], dtype=float)
        pts = centers.reshape(-1, 1, 2)
        undistorted = cv2.undistortPoints(pts, cam_mat, DIST_COEFFS, P=cam_mat)
        centers = undistorted.reshape(-1, 2)

    results = []
    for i, (_, __, r_px) in enumerate(detections):
        if r_px <= 0:
            continue
        x_px, y_px = centers[i]
        Z_a = (r_real * f_y) / r_px
        X_a = (x_px - c_x) * Z_a / f_x
        Y_a = (y_px - c_y) * Z_a / f_y
        cam_point = np.array([X_a, Y_a, Z_a, 1.0])
        robot_point = TF @ cam_point
        results.append(tuple(robot_point[:3]))

    return results


# ---------------------------------------------------------------------------
# ArUco marker → robot frame
# ---------------------------------------------------------------------------

def aruco_to_robot_frame(
        marker: dict,
        marker_size_m: float,
) -> tuple[float, float, float, float] | None:
    """
    Estimate position and face normal of an ArUco marker in robot frame.

    Parameters
    ----------
    marker        : dict from detect_aruco() — must have 'corners' (4,2) float array
    marker_size_m : physical side length of the square marker [m]

    Returns
    -------
    (x_r, y_r, face_nx, face_ny) on success, None on failure.

        x_r, y_r          : marker centre in robot frame [m] (x_r=forward, y_r=left)
        face_nx, face_ny  : unit vector of marker face normal in robot XY plane,
                            pointing from the marker surface toward the camera/robot.

    Goal 40 cm in front of marker:
        goal = (x_r + 0.40 * face_nx, y_r + 0.40 * face_ny)
    """
    half = marker_size_m / 2.0

    obj_pts = np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float32)

    img_pts = marker["corners"].astype(np.float32)

    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, _CAMERA_MATRIX, DIST_COEFFS)
    if not ok:
        return None

    t = tvec.flatten()
    cam_pt = np.array([t[0], t[1], t[2], 1.0])
    robot_pt = _TF @ cam_pt
    x_r = float(robot_pt[0])
    y_r = float(robot_pt[1])

    R, _ = cv2.Rodrigues(rvec)
    z_cam = R[:, 2]
    z_robot = _TF[:3, :3] @ z_cam

    nx, ny = float(z_robot[0]), float(z_robot[1])
    mag = math.hypot(nx, ny)
    if mag < 1e-3:
        return None

    return x_r, y_r, nx / mag, ny / mag
