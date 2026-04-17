"""
Pure A* snapshot planner.

Inputs:  start, goal, obstacles, PlannerConfig
Outputs: (path, solved)

No temporal state, no I/O, no visualisation.
"""
import heapq
import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CircleObstacle:
    x: float
    y: float
    r: float


@dataclass(frozen=True)
class WallObstacle:
    x1: float
    y1: float
    x2: float
    y2: float


Obstacle = CircleObstacle | WallObstacle


@dataclass
class PlannerConfig:
    delta: float = 0.02          # grid step size (m)
    goal_tolerance: float = 0.15
    clearance: float = 0.03     # extra safety margin around obstacles (m)
    robot_radius: float = 0.1   # robot footprint radius (m)
    max_steps: int = 2000
    clearance_weight: float = 0.0  # > 0 biases path away from obstacles
    smooth_path: bool = True


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_segment_distance(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    """Minimum distance from point (px, py) to segment (ax, ay)-(bx, by)."""
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _segments_intersect(
    p0: tuple[float, float], p1: tuple[float, float],
    q0: tuple[float, float], q1: tuple[float, float],
) -> bool:
    """True if segment p0-p1 and q0-q1 properly intersect."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    ex, ey = q1[0] - q0[0], q1[1] - q0[1]
    denom = dx * ey - dy * ex
    if abs(denom) < 1e-12:
        return False  # parallel / collinear
    fx, fy = q0[0] - p0[0], q0[1] - p0[1]
    t = (fx * ey - fy * ex) / denom
    u = (fx * dy - fy * dx) / denom
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0


def segment_intersects_circle(
    p0: tuple[float, float],
    p1: tuple[float, float],
    c: CircleObstacle,
    clearance: float,
    robot_radius: float,
) -> bool:
    ax, ay = p1[0] - p0[0], p1[1] - p0[1]
    bx, by = c.x - p0[0], c.y - p0[1]
    seg_len_sq = ax * ax + ay * ay
    if seg_len_sq == 0.0:
        closest_x, closest_y = p0
    else:
        t = max(0.0, min(1.0, (ax * bx + ay * by) / seg_len_sq))
        closest_x = p0[0] + t * ax
        closest_y = p0[1] + t * ay
    safe_r = c.r + clearance + robot_radius
    return math.hypot(c.x - closest_x, c.y - closest_y) <= safe_r


def segment_intersects_wall(
    p0: tuple[float, float],
    p1: tuple[float, float],
    wall: WallObstacle,
    clearance: float,
    robot_radius: float,
) -> bool:
    safe_d = clearance + robot_radius
    q0 = (wall.x1, wall.y1)
    q1 = (wall.x2, wall.y2)
    if _segments_intersect(p0, p1, q0, q1):
        return True
    return min(
        point_segment_distance(p0[0], p0[1], wall.x1, wall.y1, wall.x2, wall.y2),
        point_segment_distance(p1[0], p1[1], wall.x1, wall.y1, wall.x2, wall.y2),
        point_segment_distance(wall.x1, wall.y1, p0[0], p0[1], p1[0], p1[1]),
        point_segment_distance(wall.x2, wall.y2, p0[0], p0[1], p1[0], p1[1]),
    ) <= safe_d


def is_step_blocked(
    p0: tuple[float, float],
    p1: tuple[float, float],
    obstacles: Iterable[Obstacle],
    clearance: float,
    robot_radius: float,
) -> bool:
    for obs in obstacles:
        if isinstance(obs, CircleObstacle):
            if segment_intersects_circle(p0, p1, obs, clearance, robot_radius):
                return True
        else:
            if segment_intersects_wall(p0, p1, obs, clearance, robot_radius):
                return True
    return False


def has_line_of_sight(
    p0: tuple[float, float],
    p1: tuple[float, float],
    obstacles: Iterable[Obstacle],
    cfg: PlannerConfig,
) -> bool:
    return not is_step_blocked(p0, p1, obstacles, cfg.clearance, cfg.robot_radius)


def min_obstacle_edge_distance(
    point: tuple[float, float],
    obstacles: Iterable[Obstacle],
    robot_radius: float,
) -> float:
    min_d = float("inf")
    for obs in obstacles:
        if isinstance(obs, CircleObstacle):
            min_d = min(min_d, dist(point, (obs.x, obs.y)) - (obs.r + robot_radius))
        else:
            min_d = min(min_d, point_segment_distance(
                point[0], point[1], obs.x1, obs.y1, obs.x2, obs.y2
            ) - robot_radius)
    return min_d


def bounds_from_scene(
    start: tuple[float, float],
    goal: tuple[float, float],
    obstacles: list[Obstacle],
) -> tuple[float, float, float, float]:
    xs = [start[0], goal[0]]
    ys = [start[1], goal[1]]
    for obs in obstacles:
        if isinstance(obs, CircleObstacle):
            xs += [obs.x - obs.r, obs.x + obs.r]
            ys += [obs.y - obs.r, obs.y + obs.r]
        else:
            xs += [obs.x1, obs.x2]
            ys += [obs.y1, obs.y2]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y, 1.0)
    m = 0.2 * span
    return min_x - m, max_x + m, min_y - m, max_y + m


# ---------------------------------------------------------------------------
# A* internals
# ---------------------------------------------------------------------------

def _heading_offsets() -> list[float]:
    return [math.radians(d) for d in (0, 45, -45, 90, -90, 135, -135, 180)]


def _grid_key(
    point: tuple[float, float], origin: tuple[float, float], delta: float
) -> tuple[int, int]:
    return (
        int(round((point[0] - origin[0]) / delta)),
        int(round((point[1] - origin[1]) / delta)),
    )


def _grid_point(
    key: tuple[int, int], origin: tuple[float, float], delta: float
) -> tuple[float, float]:
    return (origin[0] + key[0] * delta, origin[1] + key[1] * delta)


def _reconstruct_path(
    parent: dict[tuple[int, int], tuple[int, int] | None],
    end_key: tuple[int, int],
    origin: tuple[float, float],
    delta: float,
    goal: tuple[float, float],
) -> list[tuple[float, float]]:
    keys: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = end_key
    while cur is not None:
        keys.append(cur)
        cur = parent.get(cur)
    keys.reverse()
    path = [_grid_point(k, origin, delta) for k in keys]
    if dist(path[-1], goal) > 1e-9:
        path.append(goal)
    return path


def _is_point_blocked(
    point: tuple[float, float], obstacles: list[Obstacle], cfg: PlannerConfig
) -> bool:
    for obs in obstacles:
        if isinstance(obs, CircleObstacle):
            if dist(point, (obs.x, obs.y)) <= obs.r + cfg.clearance + cfg.robot_radius:
                return True
        else:
            if point_segment_distance(
                point[0], point[1], obs.x1, obs.y1, obs.x2, obs.y2
            ) <= cfg.clearance + cfg.robot_radius:
                return True
    return False


def find_safe_start(
    start: tuple[float, float],
    goal: tuple[float, float],
    obstacles: list[Obstacle],
    cfg: PlannerConfig,
) -> tuple[float, float]:
    """
    If start is inside one or more obstacle safety zones, project it outward to
    the nearest point that clears all of them.

    On each iteration the most-penetrated obstacle is identified and the point
    is pushed to just outside its inflated boundary.  Iterates until free or a
    maximum count is reached, so overlapping obstacles are handled correctly.

    Returns start unchanged if it is already free.
    """
    point = start
    margin = 0.01  # small clearance beyond the inflated boundary (m)

    for _ in range(10):
        if not _is_point_blocked(point, obstacles, cfg):
            return point

        best_pen = -float("inf")
        best_new_point: tuple[float, float] = point

        for obs in obstacles:
            if isinstance(obs, CircleObstacle):
                inflated_r = obs.r + cfg.clearance + cfg.robot_radius
                d = dist(point, (obs.x, obs.y))
                pen = inflated_r - d
                if pen > best_pen:
                    best_pen = pen
                    if d < 1e-9:
                        dg = dist(point, goal)
                        direction: tuple[float, float] = (
                            ((goal[0] - point[0]) / dg, (goal[1] - point[1]) / dg)
                            if dg >= 1e-9 else (1.0, 0.0)
                        )
                    else:
                        direction = ((point[0] - obs.x) / d, (point[1] - obs.y) / d)
                    best_new_point = (
                        obs.x + (inflated_r + margin) * direction[0],
                        obs.y + (inflated_r + margin) * direction[1],
                    )
            else:  # WallObstacle
                safe_d = cfg.clearance + cfg.robot_radius
                d = point_segment_distance(
                    point[0], point[1], obs.x1, obs.y1, obs.x2, obs.y2
                )
                pen = safe_d - d
                if pen > best_pen:
                    best_pen = pen
                    # Closest point on wall
                    wdx, wdy = obs.x2 - obs.x1, obs.y2 - obs.y1
                    seg_len_sq = wdx * wdx + wdy * wdy
                    if seg_len_sq < 1e-12:
                        cx, cy = obs.x1, obs.y1
                    else:
                        t = max(0.0, min(1.0, (
                            (point[0] - obs.x1) * wdx + (point[1] - obs.y1) * wdy
                        ) / seg_len_sq))
                        cx, cy = obs.x1 + t * wdx, obs.y1 + t * wdy
                    if d < 1e-9:
                        length = math.sqrt(seg_len_sq) if seg_len_sq >= 1e-12 else 0.0
                        direction = (-wdy / length, wdx / length) if length >= 1e-9 else (1.0, 0.0)
                    else:
                        direction = ((point[0] - cx) / d, (point[1] - cy) / d)
                    best_new_point = (
                        cx + (safe_d + margin) * direction[0],
                        cy + (safe_d + margin) * direction[1],
                    )

        point = best_new_point

    return point  # best-effort if still inside after max iterations


# ---------------------------------------------------------------------------
# Path smoothing
# ---------------------------------------------------------------------------

def smooth_waypoints(
    path: list[tuple[float, float]],
    obstacles: list[Obstacle],
    cfg: PlannerConfig,
) -> list[tuple[float, float]]:
    if len(path) <= 2:
        return path
    smoothed: list[tuple[float, float]] = []
    i = 0
    while i < len(path) - 1:
        smoothed.append(path[i])
        j = len(path) - 1
        while j > i + 1:
            if has_line_of_sight(path[i], path[j], obstacles, cfg):
                break
            j -= 1
        i = j
    if smoothed[-1] != path[-1]:
        smoothed.append(path[-1])
    return smoothed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def astar_search(
    start: tuple[float, float],
    goal: tuple[float, float],
    obstacles: list[Obstacle],
    cfg: PlannerConfig,
) -> tuple[list[tuple[float, float]], bool]:
    min_x, max_x, min_y, max_y = bounds_from_scene(start, goal, obstacles)
    margin = max(cfg.delta * 2.0, 0.25)
    min_x -= margin
    min_y -= margin
    max_x += margin
    max_y += margin

    origin = start
    start_key = (0, 0)
    open_heap: list[tuple[float, int, tuple[int, int]]] = []
    g_score: dict[tuple[int, int], float] = {start_key: 0.0}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start_key: None}
    closed: set[tuple[int, int]] = set()
    counter = 0

    heapq.heappush(open_heap, (dist(start, goal), counter, start_key))
    neighbor_offsets = _heading_offsets()

    expanded = 0
    while open_heap and expanded < cfg.max_steps:
        _, _, current_key = heapq.heappop(open_heap)
        if current_key in closed:
            continue
        current = _grid_point(current_key, origin, cfg.delta)
        if dist(current, goal) <= cfg.goal_tolerance:
            return _reconstruct_path(parent, current_key, origin, cfg.delta, goal), True
        closed.add(current_key)
        expanded += 1

        for offset in neighbor_offsets:
            nxt = (
                current[0] + cfg.delta * math.cos(offset),
                current[1] + cfg.delta * math.sin(offset),
            )
            if not (min_x <= nxt[0] <= max_x and min_y <= nxt[1] <= max_y):
                continue
            if is_step_blocked(current, nxt, obstacles, cfg.clearance, cfg.robot_radius):
                continue
            nxt_key = _grid_key(nxt, origin, cfg.delta)
            if nxt_key in closed:
                continue
            tentative_g = g_score[current_key] + dist(current, nxt)
            if tentative_g >= g_score.get(nxt_key, float("inf")):
                continue
            parent[nxt_key] = current_key
            g_score[nxt_key] = tentative_g
            h = dist(nxt, goal)
            f = tentative_g + h
            if cfg.clearance_weight > 0.0:
                clearance_d = min_obstacle_edge_distance(nxt, obstacles, cfg.robot_radius)
                if clearance_d != float("inf"):
                    f += cfg.clearance_weight / max(0.05, clearance_d)
            counter += 1
            heapq.heappush(open_heap, (f, counter, nxt_key))

    return [start], False


def find_safe_goal(
    goal: tuple[float, float],
    start: tuple[float, float],
    obstacles: list[Obstacle],
    cfg: PlannerConfig,
) -> tuple[float, float]:
    """
    If goal is inside one or more obstacle safety zones, project it outward to
    the nearest free point.  Mirror of find_safe_start for the destination end.
    """
    return find_safe_start(goal, start, obstacles, cfg)


def plan_path(
    start: tuple[float, float],
    goal: tuple[float, float],
    obstacles: list[Obstacle],
    cfg: PlannerConfig,
) -> tuple[list[tuple[float, float]], bool]:
    """
    Compute a collision-free path from start to goal.

    Both start and goal are projected to the nearest free point when inside an
    obstacle safety zone.  Returns (path, solved); path always has at least
    [start].  solved is False only when no route exists even after projection.
    """
    if _is_point_blocked(start, obstacles, cfg):
        start = find_safe_start(start, goal, obstacles, cfg)
        if _is_point_blocked(start, obstacles, cfg):
            return [start], False
    if _is_point_blocked(goal, obstacles, cfg):
        goal = find_safe_goal(goal, start, obstacles, cfg)
        if _is_point_blocked(goal, obstacles, cfg):
            return [start], False
    path, solved = astar_search(start, goal, obstacles, cfg)
    if solved and cfg.smooth_path:
        path = smooth_waypoints(path, obstacles, cfg)
    return path, solved
