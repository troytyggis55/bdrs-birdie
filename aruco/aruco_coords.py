"""
aruco_coords.py — ArUco marker → robot-frame position and face normal.

Reuses the camera calibration parameters from CamVision/ballcoords.py
and the same camera→robot rigid transform (_TF).

Usage:
    from aruco.aruco_coords import aruco_to_robot_frame

    result = aruco_to_robot_frame(marker, marker_size_m=0.10)
    if result:
        x_r, y_r, face_nx, face_ny = result
"""

import math
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from CamVision.ballcoords import _CAMERA_MATRIX, DIST_COEFFS, _TF


def aruco_to_robot_frame(
        marker: dict,
        marker_size_m: float,
) -> tuple[float, float, float, float] | None:
    """
    Estimate the position and face normal of an ArUco marker in robot frame.

    Parameters
    ----------
    marker        : dict as returned by detect_aruco(), must contain a
                    'corners' key — (4, 2) float array in pixel coords
                    (order: top-left, top-right, bottom-right, bottom-left).
    marker_size_m : physical side length of the square marker [m].

    Returns
    -------
    (x_r, y_r, face_nx, face_ny) on success, None on failure.

        x_r, y_r          : marker centre in robot body frame [m]
                            (x_r = forward, y_r = left).
        face_nx, face_ny  : unit vector of the marker face normal projected
                            onto the robot XY plane. Points from the marker
                            surface toward the camera/robot side.

    Goal computation (40 cm in front of marker):
        goal_robot = (x_r + 0.40 * face_nx, y_r + 0.40 * face_ny)
    """
    half = marker_size_m / 2.0

    # 3D object corners in marker frame: TL, TR, BR, BL
    # X right, Y up, Z out from face toward camera.
    obj_pts = np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float32)

    img_pts = marker["corners"].astype(np.float32)  # (4, 2)

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, _CAMERA_MATRIX, DIST_COEFFS,
    )
    if not ok:
        return None

    # --- marker centre in robot frame ---
    t = tvec.flatten()
    cam_pt = np.array([t[0], t[1], t[2], 1.0])
    robot_pt = _TF @ cam_pt
    x_r = float(robot_pt[0])
    y_r = float(robot_pt[1])

    # --- face normal in robot frame ---
    # R rotates vectors from marker frame to camera frame.
    # The marker Z-axis (face normal) in camera frame: R[:, 2]
    # It points toward the camera when the marker faces the camera.
    R, _ = cv2.Rodrigues(rvec)
    z_cam = R[:, 2]                      # (3,) unit vector in camera frame
    z_robot = _TF[:3, :3] @ z_cam       # (3,) unit vector in robot frame

    # Project to the robot XY plane (ignore tilt) and normalise.
    nx, ny = float(z_robot[0]), float(z_robot[1])
    mag = math.hypot(nx, ny)
    if mag < 1e-3:
        return None

    return x_r, y_r, nx / mag, ny / mag
