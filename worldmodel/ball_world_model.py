"""
ball_world_model.py — Persistent world-frame ball position tracking.

Fuses camera detections (robot body frame) with odometry to maintain
stable world-frame estimates for each tracked ball color.

Usage:
    from worldmodel.ball_world_model import BallWorldModel

    wm = BallWorldModel(colors=('R', 'B'))
    # In your vision loop:
    wm.update('R', x_r, y_r, robot_x, robot_y, heading)
    # In your mission logic:
    ball = wm.get('R')
    if ball and ball.reliable:
        print(ball.x, ball.y)
"""

import math
import time
from dataclasses import dataclass, field

# Minimum detections before a position is considered reliable.
_DEFAULT_MIN_DETECTIONS = 3
# EMA weight given to new measurements (lower = smoother but slower to converge).
_DEFAULT_EMA_ALPHA = 0.3


@dataclass
class BallEstimate:
    """World-frame position estimate for a single ball."""
    color: str          # 'R' or 'B'
    x: float            # world frame [m], forward from odometry origin
    y: float            # world frame [m], lateral from odometry origin
    last_seen: float    # time.monotonic() timestamp of last detection
    detection_count: int
    collected: bool = False
    _min_detections: int = field(default=_DEFAULT_MIN_DETECTIONS, repr=False, compare=False)

    @property
    def reliable(self) -> bool:
        """True once enough detections have accumulated and ball is not collected."""
        return self.detection_count >= self._min_detections and not self.collected

    @property
    def age(self) -> float:
        """Seconds since this ball was last detected."""
        return time.monotonic() - self.last_seen


def robot_to_world(x_r: float, y_r: float,
                   robot_x: float, robot_y: float, heading: float
                   ) -> tuple[float, float]:
    """
    Convert robot-body-frame coordinates to world/odometry frame.

    Parameters
    ----------
    x_r, y_r   : position in robot frame [m]  (x_r = forward, y_r = left)
    robot_x, robot_y : robot position in world frame [m]
    heading     : robot heading [radians], CCW positive

    Returns
    -------
    (world_x, world_y) in odometry frame [m]
    """
    world_x = robot_x + x_r * math.cos(heading) - y_r * math.sin(heading)
    world_y = robot_y + x_r * math.sin(heading) + y_r * math.cos(heading)
    return world_x, world_y


class BallWorldModel:
    """
    Maintains EMA-smoothed world-frame position estimates for a set of ball colors.

    The caller is responsible for providing new detections (in robot frame)
    together with the current odometry pose. The model handles the frame
    transform and smoothing internally.
    """

    def __init__(self,
                 colors: tuple[str, ...] = ('R', 'B'),
                 ema_alpha: float = _DEFAULT_EMA_ALPHA,
                 min_detections: int = _DEFAULT_MIN_DETECTIONS):
        """
        Parameters
        ----------
        colors          : ball colors to track (single uppercase char each)
        ema_alpha       : weight of each new measurement in the EMA [0–1]
                          (lower = more smoothing, slower convergence)
        min_detections  : detections required before a position is reliable
        """
        self._alpha = ema_alpha
        self._min_det = min_detections
        self._estimates: dict[str, BallEstimate] = {}
        self._colors = colors

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def update(self, color: str,
               x_r: float, y_r: float,
               robot_x: float, robot_y: float, heading: float) -> None:
        """
        Incorporate a new detection for *color*.

        Parameters
        ----------
        color               : 'R' or 'B'
        x_r, y_r            : ball position in robot body frame [m]
        robot_x, robot_y    : robot world-frame position [m]
        heading             : robot heading [radians]
        """
        wx, wy = robot_to_world(x_r, y_r, robot_x, robot_y, heading)
        now = time.monotonic()

        if color not in self._estimates:
            self._estimates[color] = BallEstimate(
                color=color,
                x=wx, y=wy,
                last_seen=now,
                detection_count=1,
                _min_detections=self._min_det,
            )
        else:
            est = self._estimates[color]
            if est.collected:
                return
            # EMA update
            est.x = (1 - self._alpha) * est.x + self._alpha * wx
            est.y = (1 - self._alpha) * est.y + self._alpha * wy
            est.last_seen = now
            est.detection_count += 1

    def mark_collected(self, color: str) -> None:
        """Mark *color* ball as collected so it is excluded from future estimates."""
        if color in self._estimates:
            self._estimates[color].collected = True

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def get(self, color: str) -> BallEstimate | None:
        """Return the current estimate for *color*, or None if never seen."""
        return self._estimates.get(color)

    def all_reliable(self) -> bool:
        """True when every tracked color has a reliable estimate."""
        return all(
            color in self._estimates and self._estimates[color].reliable
            for color in self._colors
        )

    def absent_for(self, color: str) -> float:
        """
        Seconds since *color* was last detected.
        Returns infinity if the ball has never been seen.
        """
        est = self._estimates.get(color)
        return est.age if est is not None else float('inf')
