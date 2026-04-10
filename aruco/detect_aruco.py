import cv2
import numpy as np
from pathlib import Path


def detect_aruco(
        frame,
        dictionary_name: str = "DICT_4X4_50",
        draw: bool = False,
        *,
        font_scale: float = 1.2,
        font_thickness: int = 2,
        line_thickness: int = 3,
        corner_radius: int = 4,
        center_radius: int = 5,
):
    """
    Detect ArUco markers in a single image frame.

    Args:
        frame: BGR or grayscale image (numpy array).
        dictionary_name: Name of a cv2.aruco predefined dictionary, e.g. "DICT_4X4_50".
        draw: If True, returns a copy of the frame with detected markers drawn.
        font_scale: Text size for the marker id annotation.
        font_thickness: Text stroke thickness for the marker id annotation.
        line_thickness: Thickness of the marker outline.
        corner_radius: Radius (px) of the corner dots.
        center_radius: Radius (px) of the center dot.

    Returns:
        If draw is False:
            markers, rejected
        If draw is True:
            markers, rejected, annotated_frame

        Where:
            markers is a list of dicts, each:
                {
                  "id": int,
                  "corners": (4,2) float ndarray in pixel coords,
                  "center": (2,) float ndarray,
                  "perimeter": float,
                  "area": float
                }
            rejected is a list of (4,2) arrays for rejected candidates.
    """
    if frame is None:
        raise ValueError("frame is None")

    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV was built without the aruco module (opencv-contrib-python required).")

    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    params = cv2.aruco.DetectorParameters()

    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # OpenCV >= 4.7 supports ArucoDetector; keep a fallback for older versions.
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners_list, ids, rejected = detector.detectMarkers(gray)
    else:
        corners_list, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    markers = []
    if ids is not None and len(ids) > 0:
        for marker_id, corners in zip(ids.flatten().tolist(), corners_list):
            # corners is shape (1,4,2) in many OpenCV versions
            c = np.asarray(corners, dtype=np.float32).reshape(4, 2)
            center = c.mean(axis=0)
            perimeter = float(np.linalg.norm(c[0] - c[1]) + np.linalg.norm(c[1] - c[2]) +
                              np.linalg.norm(c[2] - c[3]) + np.linalg.norm(c[3] - c[0]))
            area = float(cv2.contourArea(c))

            markers.append(
                {
                    "id": int(marker_id),
                    "corners": c,
                    "center": center,
                    "perimeter": perimeter,
                    "area": area,
                }
            )

    rejected_out = []
    if rejected is not None and len(rejected) > 0:
        for r in rejected:
            rejected_out.append(np.asarray(r, dtype=np.float32).reshape(4, 2))

    if draw:
        annotated = frame.copy()
        if annotated.ndim == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        if ids is not None and len(ids) > 0:
            # Custom drawing so we can control thickness/sizes
            outline_color = (0, 255, 0)   # green
            corner_color = (255, 0, 0)    # blue
            center_color = (0, 0, 255)    # red

            for m in markers:
                pts = np.round(m["corners"]).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(annotated, [pts], isClosed=True, color=outline_color, thickness=line_thickness)

                for (x, y) in np.round(m["corners"]).astype(np.int32):
                    cv2.circle(annotated, (int(x), int(y)), corner_radius, corner_color, thickness=-1)

                cx, cy = np.round(m["center"]).astype(np.int32)
                cv2.circle(annotated, (int(cx), int(cy)), center_radius, center_color, thickness=-1)

                # Bigger label with a simple outline for readability
                x0, y0 = np.round(m["corners"][0]).astype(np.int32)
                label = f"id:{m['id']}"
                org = (int(x0) + 6, int(y0) - 6)

                cv2.putText(
                    annotated,
                    label,
                    org,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),
                    thickness=max(1, font_thickness + 2),
                    lineType=cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    label,
                    org,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness=font_thickness,
                    lineType=cv2.LINE_AA,
                )

        return markers, rejected_out, annotated

    return markers, rejected_out


if __name__ == "__main__":
    in_dir = Path("examples/images/aruco")
    out_dir = in_dir / "processed"

    for i in range(1, 16):
        in_path = in_dir / f"{i}.png"
        out_path = out_dir / f"{i}.png"

        img = cv2.imread(str(in_path))
        if img is None:
            print(f"Skipping (could not read): {in_path}")
            continue

        markers, rejected, annotated = detect_aruco(img, draw=True)
        ok = cv2.imwrite(str(out_path), annotated)
        if not ok:
            print(f"Failed to write: {out_path}")
        else:
            print(f"Wrote: {out_path}")