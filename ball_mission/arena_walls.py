"""
arena_walls.py — Reconstruct inner-wall geometry from ArUco detections.

The arena has a "plus"-shaped inner wall structure dividing four 30×30 cm
quadrants. Each inner wall face carries one ArUco marker. From a single
marker sighting we can reconstruct both wall segments meeting at the
inner junction.

Formula (from CLAUDE.md — "Mapping the square walls based on ArUco"):
  Given a left-wall marker at world position P with face normal n = (nx, ny):
    t = along-wall direction = (-ny, nx)     [90° CCW from n]
    Wall the marker is on:  P − 0.13·t  →  P + 0.47·t   (60 cm)
    Perpendicular wall:  centre C = P + 0.17·t,  C ± 0.30·n   (60 cm)

  Right-wall markers use the mirrored along-wall direction: t' = (ny, −nx).

Left-wall markers  : IDs 10, 12, 14, 16  (even)
Right-wall markers : IDs 11, 13, 15, 17  (odd)

Public API
----------
walls_from_aruco_world(marker_id, wx, wy, fnx_w, fny_w) -> list[WallObstacle]
    Geometry-only: returns two WallObstacle objects from world-frame inputs.

ArenaWallModel
    Thread-safe EMA world model. Call update() on every ArUco detection,
    then get_walls() to retrieve all current WallObstacle segments for
    pathfinding.
"""

from __future__ import annotations

import math
import threading

from pathfinding.pathfinding import WallObstacle


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _is_right_wall(marker_id: int) -> bool:
    """Odd ArUco IDs in the arena set (11, 13, 15, 17) are right-wall markers."""
    return marker_id % 2 == 1


def walls_from_aruco_world(
    marker_id: int,
    wx: float,
    wy: float,
    fnx_w: float,
    fny_w: float,
) -> list[WallObstacle]:
    """
    Reconstruct the two inner wall segments from a single ArUco detection.

    Parameters
    ----------
    marker_id          : ArUco marker ID (determines left vs right wall)
    wx, wy             : marker centre in world frame [m]
    fnx_w, fny_w       : face normal in world frame (unit vector, pointing
                         away from the wall toward the approaching robot)

    Returns
    -------
    [wall_on_marker, perpendicular_wall]
        Two WallObstacle objects representing the walls that meet at the
        inner junction nearest this marker.
    """
    # Along-wall direction perpendicular to face normal.
    # Left walls:  t = (−ny,  nx)  — 90° CCW from n
    # Right walls: t = ( ny, −nx)  — 90° CW  from n (mirrored)
    if _is_right_wall(marker_id):
        tx, ty = fny_w, -fnx_w
    else:
        tx, ty = -fny_w, fnx_w

    # Wall the marker sits on: 13 cm from outer corner to 47 cm to inner junction.
    wall_on = WallObstacle(
        wx - 0.13 * tx, wy - 0.13 * ty,
        wx + 0.47 * tx, wy + 0.47 * ty,
    )

    # Perpendicular wall: centred 17 cm along t from the marker, spans ±30 cm
    # along the face-normal direction.
    cx = wx + 0.17 * tx
    cy = wy + 0.17 * ty
    wall_perp = WallObstacle(
        cx - 0.30 * fnx_w, cy - 0.30 * fny_w,
        cx + 0.30 * fnx_w, cy + 0.30 * fny_w,
    )

    return [wall_on, wall_perp]


# ---------------------------------------------------------------------------
# World model
# ---------------------------------------------------------------------------

class ArenaWallModel:
    """
    Thread-safe EMA-smoothed world-frame model for all detected ArUco markers.

    Every call to update() refines the position/orientation estimate for
    that marker. get_walls() converts all known estimates to WallObstacle
    objects ready for the pathfinder.

    The EMA gives more stable wall positions than using the last raw detection
    directly. Only markers seen at least min_detections times contribute walls,
    to filter out spurious one-shot detections.
    """

    def __init__(self, alpha: float = 0.35, min_detections: int = 2) -> None:
        self._alpha = alpha
        self._min_det = min_detections
        self._lock = threading.Lock()
        # marker_id → {wx, wy, fnx, fny, count}
        self._markers: dict[int, dict] = {}

    def update(
        self,
        marker_id: int,
        wx: float,
        wy: float,
        fnx_w: float,
        fny_w: float,
    ) -> None:
        """
        Incorporate a new world-frame detection for *marker_id*.

        Parameters
        ----------
        wx, wy         : marker centre in world frame [m]
        fnx_w, fny_w   : face normal in world frame (unit vector)
        """
        a = self._alpha
        with self._lock:
            if marker_id not in self._markers:
                self._markers[marker_id] = {
                    'wx': wx, 'wy': wy,
                    'fnx': fnx_w, 'fny': fny_w,
                    'count': 1,
                }
            else:
                m = self._markers[marker_id]
                m['wx']  = (1 - a) * m['wx']  + a * wx
                m['wy']  = (1 - a) * m['wy']  + a * wy
                m['fnx'] = (1 - a) * m['fnx'] + a * fnx_w
                m['fny'] = (1 - a) * m['fny'] + a * fny_w
                # Re-normalise face normal after EMA to keep it a unit vector.
                mag = math.hypot(m['fnx'], m['fny'])
                if mag > 1e-3:
                    m['fnx'] /= mag
                    m['fny'] /= mag
                m['count'] += 1

    def get_walls(self) -> list[WallObstacle]:
        """
        Return all wall segments inferred from currently known markers.

        Only markers that have been seen at least *min_detections* times are
        included; earlier sightings are likely noisy.
        """
        walls: list[WallObstacle] = []
        with self._lock:
            for mid, m in self._markers.items():
                if m['count'] < self._min_det:
                    continue
                walls.extend(
                    walls_from_aruco_world(mid, m['wx'], m['wy'], m['fnx'], m['fny'])
                )
        return walls

    def get_plus_center(self) -> tuple[float, float] | None:
        """
        Return the best world-frame estimate of the arena plus-center (the point
        where the two inner walls intersect = center of the 60×60 cm arena).

        Derived from all known markers: for each marker the plus-center is
        P + 0.17·t (where t is the along-wall direction).  The average across
        all qualified markers is returned; if no qualified markers exist, None.
        """
        centers: list[tuple[float, float]] = []
        with self._lock:
            for mid, m in self._markers.items():
                if m['count'] < self._min_det:
                    continue
                if mid % 2 == 1:   # right-wall marker → t = (fny, -fnx)
                    tx, ty = m['fny'], -m['fnx']
                else:              # left-wall marker  → t = (-fny, fnx)
                    tx, ty = -m['fny'], m['fnx']
                centers.append((m['wx'] + 0.17 * tx, m['wy'] + 0.17 * ty))
        if not centers:
            return None
        return (
            sum(c[0] for c in centers) / len(centers),
            sum(c[1] for c in centers) / len(centers),
        )

    def get_snapshot(self) -> dict[int, dict]:
        """Return a copy of the current per-marker estimates (for logging/replay)."""
        with self._lock:
            return {mid: dict(m) for mid, m in self._markers.items()}

    def marker_count(self) -> int:
        """Number of distinct markers seen so far."""
        with self._lock:
            return len(self._markers)
