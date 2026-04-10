# src/odometry/graph_nav.py
"""
Graph-based navigation using odometry.

Loads a JSON file describing waypoints (nodes) and an ordered route,
then drives the robot through each waypoint using proportional
heading + distance control based on pose from spose.py.
"""

import json
import math
import threading
import time as t
from spose import pose
from uservice import service


class GraphNav:
    def __init__(self):
        self.nodes = {}  # name -> (x, y)
        self.route = []  # ordered list of node names
        self.drive_speed = 0.15  # m/s
        self._stop_nav = threading.Event()  # set to interrupt current drive_to / turn_to
        self.waypoint_tol = 0.05  # metres – "close enough"
        # control gains
        self.kp_heading = 4.0  # proportional gain for heading correction (rad/s per rad)
        # turn-in-place: if heading error exceeds this, stop and rotate first
        self.turn_in_place_thresh = math.radians(60)  # 60 degrees
        self.blend_radius = 0.15  # metres – start blending heading to next WP within this distance
        self.pass_through_tol = 0.10  # metres – looser tolerance for intermediate waypoints
        self.loop_dt = 0.05

    # ------------------------------------------------------------------
    def load_waypoints(self, waypoints: list) -> None:
        """Load a path directly from a list of (x, y) tuples, bypassing JSON."""
        self.nodes = {f"wp{i}": (float(x), float(y)) for i, (x, y) in enumerate(waypoints)}
        self.route = [f"wp{i}" for i in range(len(waypoints))]
        print(f"% GraphNav:: loaded {len(self.nodes)} waypoints from path")

    # ------------------------------------------------------------------
    def load(self, path: str):
        """Load a graph JSON file. Expected keys: nodes, route, drive_speed, waypoint_tolerance."""
        if isinstance(path, dict):
            data = path
        else:
            with open(path, "r") as f:
                data = json.load(f)
        for name, coords in data["nodes"].items():
            self.nodes[name] = (float(coords["x"]), float(coords["y"]))
        self.route = list(data["route"])
        self.drive_speed = float(data.get("drive_speed", self.drive_speed))
        self.waypoint_tol = float(data.get("waypoint_tolerance", self.waypoint_tol))
        print(f"% GraphNav:: loaded {len(self.nodes)} nodes, route has {len(self.route)} waypoints")

    # ------------------------------------------------------------------
    def _get_pose(self, zero_x=0.0, zero_y=0.0):
        """Return current (x, y, heading_rad) with tuning scales applied."""
        print(f"XY H pose (raw): {pose.pose[0]:.3f}, {pose.pose[1]:.3f}, {math.degrees(pose.pose[2]):.1f}°")
        return pose.pose[0] - zero_x, pose.pose[1] - zero_y, pose.pose[2]

    # ------------------------------------------------------------------
    @staticmethod
    def _angle_wrap(a: float) -> float:
        """Wrap angle to [-pi, pi]."""
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    # ------------------------------------------------------------------
    def turn_to(self, target_heading: float, heading_tol=math.radians(5)) -> bool:
        """
        Stop and rotate in place until facing target_heading (rad).
        Returns True when aligned, False if service.stop.
        """
        while not service.stop and not self._stop_nav.is_set():
            _, _, ch = self._get_pose()
            heading_err = self._angle_wrap(target_heading - ch)

            if abs(heading_err) < heading_tol:
                # stop rotation
                service.send("robobot/cmd/ti", "rc 0 0")
                t.sleep(0.1)
                return True

            turn_rate = min(self.kp_heading * heading_err, 2)
            max_turn = 4  # rad/s cap for in-place rotation
            max_turn = 2  # rad/s cap for in-place rotation
            turn_rate = max(-max_turn, min(max_turn, turn_rate))

            service.send("robobot/cmd/ti", f"rc 0.0 {turn_rate:.3f}")
            t.sleep(self.loop_dt)

        service.send("robobot/cmd/ti", "rc 0 0")
        return False

    # ------------------------------------------------------------------
    @staticmethod
    def _turn_angle(ax, ay, bx, by, cx, cy) -> float:
        """
        Return the absolute turn angle at point B for the path A -> B -> C.
        0 = straight ahead, pi = full U-turn.
        """
        heading_ab = math.atan2(by - ay, bx - ax)
        heading_bc = math.atan2(cy - by, cx - bx)
        delta = heading_bc - heading_ab
        # wrap to [-pi, pi]
        while delta > math.pi:
            delta -= 2.0 * math.pi
        while delta < -math.pi:
            delta += 2.0 * math.pi
        return abs(delta)

    # ------------------------------------------------------------------
    def _drive_smooth(self, target_x: float, target_y: float,
                      next_x: float = None, next_y: float = None,
                      is_last: bool = True, or_speed: float = None) -> bool:
        """
        Drive towards (target_x, target_y) with smooth heading corrections.
        If a next waypoint is provided and this is not the last waypoint,
        the robot blends its heading toward the next WP when close to the
        current one, creating a smooth arc through intermediate waypoints.
        Returns True when the waypoint is reached / passed, False if service.stop.
        """
        # pre-compute the turn angle at the current waypoint
        has_next = (next_x is not None and not is_last)
        turn_ang = 0.0
        if has_next:
            cx, cy, _ = self._get_pose()
            turn_ang = self._turn_angle(cx, cy, target_x, target_y, next_x, next_y)

        # choose tolerance: tighter for last WP, looser for intermediate pass-through
        tolerance = self.waypoint_tol if is_last else self.pass_through_tol

        while not service.stop and not self._stop_nav.is_set():
            cx, cy, ch = self._get_pose()
            dx = target_x - cx
            dy = target_y - cy
            dist = math.hypot(dx, dy)

            if dist < tolerance:
                if is_last:
                    service.send("robobot/cmd/ti", "rc 0 0")
                    t.sleep(0.1)
                return True

            # heading to current waypoint
            heading_to_wp = math.atan2(dy, dx)

            # blend heading toward next waypoint when within blend_radius
            if has_next and dist < self.blend_radius:
                heading_to_next = math.atan2(next_y - target_y, next_x - target_x)
                blend = 1.0 - (dist / self.blend_radius)  # 0 at edge, 1 at waypoint
                # weighted circular interpolation
                delta = self._angle_wrap(heading_to_next - heading_to_wp)
                desired_heading = self._angle_wrap(heading_to_wp + blend * delta)
            else:
                desired_heading = heading_to_wp

            heading_err = self._angle_wrap(desired_heading - ch)

            # if heading drifts too far, stop and re-align
            if abs(heading_err) > self.turn_in_place_thresh:
                self.turn_to(desired_heading)
                continue

            turn_rate = self.kp_heading * heading_err
            max_turn = 2.0
            turn_rate = max(-max_turn, min(max_turn, turn_rate))

            # slow down based on heading error
            heading_factor = max(0.15, 1.0 - abs(heading_err) / self.turn_in_place_thresh)

            # slow down approaching a sharp corner (scale by turn angle at the waypoint)
            if has_next and dist < self.blend_radius:
                corner_factor = max(0.25, 1.0 - (turn_ang / math.pi))
            else:
                corner_factor = 1.0

            if or_speed is None:
                speed = self.drive_speed * heading_factor * corner_factor
            else:
                speed = or_speed * heading_factor * corner_factor

            service.send("robobot/cmd/ti", f"rc {speed:.3f} {turn_rate:.3f}")
            t.sleep(self.loop_dt)

        service.send("robobot/cmd/ti", "rc 0 0")
        return False

    # ------------------------------------------------------------------
    def drive_to(self, target_x: float, target_y: float,
                 next_x: float = None, next_y: float = None,
                 is_last: bool = True, or_speed: float = None) -> bool:
        """
        Navigate to (target_x, target_y).  For small heading errors the robot
        smoothly adjusts while driving; for large turns it stops and rotates
        in place first.
        """
        # 1) compute heading to target
        cx, cy, ch = self._get_pose()
        dx = target_x - cx
        dy = target_y - cy
        desired_heading = math.atan2(dy, dx)
        heading_err = abs(self._angle_wrap(desired_heading - ch))

        # 2) only stop-and-rotate for large heading errors
        if heading_err > self.turn_in_place_thresh:
            if not self.turn_to(desired_heading):
                return False

        # 3) drive to target with smooth look-ahead
        return self._drive_smooth(target_x, target_y, next_x, next_y, is_last, or_speed)

    # ------------------------------------------------------------------
    def run(self):
        """Execute the full route. Call from the main state machine."""
        if len(self.route) == 0:
            print("% GraphNav:: no route loaded")
            return

        print(f"% GraphNav:: starting route with {len(self.route)} waypoints")
        pose.tripBreset()

        service.send("robobot/cmd/T0", "enc0")  # Reset odometry to zero at start of route
        t.sleep(0.1)

        for idx, node_name in enumerate(self.route):
            if service.stop:
                break
            if node_name not in self.nodes:
                print(f"% GraphNav:: unknown node '{node_name}' – skipping")
                continue

            tx, ty = self.nodes[node_name]
            cx, cy, _ = self._get_pose()
            dist = math.hypot(tx - cx, ty - cy)

            # look ahead: find the next valid waypoint for smooth blending
            is_last = (idx == len(self.route) - 1)
            nx, ny = None, None
            if not is_last:
                for future_name in self.route[idx + 1:]:
                    if future_name in self.nodes:
                        nx, ny = self.nodes[future_name]
                        break

            turn_info = ""
            if nx is not None and not is_last:
                ta = math.degrees(self._turn_angle(cx, cy, tx, ty, nx, ny))
                turn_info = f", turn≈{ta:.0f}°"
            print(f"% GraphNav:: [{idx + 1}/{len(self.route)}] heading to '{node_name}' "
                  f"({tx:.2f}, {ty:.2f}), dist={dist:.2f}m{turn_info}")

            reached = self.drive_to(tx, ty, nx, ny, is_last)
            if reached:
                print(f"% GraphNav:: reached '{node_name}'")
            else:
                print(f"% GraphNav:: aborted on way to '{node_name}'")
                break

        # stop the robot
        service.send("robobot/cmd/ti", "rc 0 0")
        print("% GraphNav:: route finished")

    def move_distance(self, distance_m: float, speed: float = None) -> bool:
        """
        Move straight a given signed distance in metres.
        Positive = forward, negative = backward.
        Returns True when done, False if interrupted by service.stop.
        """
        if speed is None:
            speed = self.drive_speed

        speed = abs(speed)
        if distance_m == 0:
            return True

        start_x, start_y, start_h = self._get_pose()
        direction = 1.0 if distance_m > 0 else -1.0
        target_dist = abs(distance_m)

        while not service.stop:
            cx, cy, ch = self._get_pose()

            traveled = math.hypot(cx - start_x, cy - start_y)
            remaining = target_dist - traveled

            if remaining <= 0.01:  # 1 cm tolerance
                service.send("robobot/cmd/ti", "rc 0 0")
                t.sleep(0.1)
                return True

            heading_err = self._angle_wrap(start_h - ch)
            turn_rate = self.kp_heading * heading_err
            turn_rate = max(-1.0, min(1.0, turn_rate))

            drive = min(speed, max(0.1, remaining * 3))  # only ramp down in last ~3cm
            service.send("robobot/cmd/ti", f"rc {direction * drive:.3f} {turn_rate:.3f}")
            t.sleep(self.loop_dt)

        service.send("robobot/cmd/ti", "rc 0 0")
        return False

    def turn_angle(self, angle_deg: float) -> bool:
        """
        Turn robot by a relative angle in degrees.
        Positive = CCW/left, negative = CW/right.
        Returns True when done, False if interrupted by service.stop.
        """
        _, _, ch = self._get_pose()
        target_heading = self._angle_wrap(ch + math.radians(angle_deg))
        return self.turn_to(target_heading)

    def turn_90_left(self) -> bool:
        return self.turn_angle(90)

    def turn_90_right(self) -> bool:
        return self.turn_angle(-90)

    # ------------------------------------------------------------------
    def _drive_smooth_step(self, target_x: float, target_y: float,
                           next_x: float = None, next_y: float = None,
                           is_last: bool = True):
        """
        One simulation control step.
        Returns:
            reached, v, w
        """
        has_next = (next_x is not None and not is_last)
        turn_ang = 0.0
        if has_next:
            cx, cy, _ = self._get_pose()
            turn_ang = self._turn_angle(cx, cy, target_x, target_y, next_x, next_y)

        tolerance = self.waypoint_tol if is_last else self.pass_through_tol

        cx, cy, ch = self._get_pose()
        dx = target_x - cx
        dy = target_y - cy
        dist = math.hypot(dx, dy)

        if dist < tolerance:
            return True, 0.0, 0.0

        heading_to_wp = math.atan2(dy, dx)

        if has_next and dist < self.blend_radius:
            heading_to_next = math.atan2(next_y - target_y, next_x - target_x)
            blend = 1.0 - (dist / self.blend_radius)
            delta = self._angle_wrap(heading_to_next - heading_to_wp)
            desired_heading = self._angle_wrap(heading_to_wp + blend * delta)
        else:
            desired_heading = heading_to_wp

        heading_err = self._angle_wrap(desired_heading - ch)

        # turn in place
        if abs(heading_err) > self.turn_in_place_thresh:
            turn_rate = min(self.kp_heading * heading_err, 2)
            max_turn = 4
            turn_rate = max(-max_turn, min(max_turn, turn_rate))
            return False, 0.0, turn_rate

        turn_rate = self.kp_heading * heading_err
        max_turn = 2.0
        turn_rate = max(-max_turn, min(max_turn, turn_rate))

        heading_factor = max(0.15, 1.0 - abs(heading_err) / self.turn_in_place_thresh)

        if has_next and dist < self.blend_radius:
            corner_factor = max(0.25, 1.0 - (turn_ang / math.pi))
        else:
            corner_factor = 1.0

        speed = self.drive_speed * heading_factor * corner_factor
        return False, speed, turn_rate

    def run_sim(self, max_steps: int = 20000):
        """
        Simulate running the route, returning:
        - trajectory: list of (x, y, heading)
        - waypoint_hits: list of (name, x, y)
        """
        trajectory = []
        waypoint_hits = []

        if len(self.route) == 0:
            print("% GraphNav:: no route loaded for simulation")
            return trajectory, waypoint_hits

        print(f"% GraphNav:: starting simulation of route with {len(self.route)} waypoints")
        pose.tripBreset()

        current_wp_idx = 0

        for step in range(max_steps):
            if current_wp_idx >= len(self.route):
                break

            node_name = self.route[current_wp_idx]
            if node_name not in self.nodes:
                print(f"% GraphNav:: unknown node '{node_name}' - skipping")
                current_wp_idx += 1
                continue

            tx, ty = self.nodes[node_name]
            is_last = (current_wp_idx == len(self.route) - 1)

            # look ahead for smooth blending
            nx, ny = None, None
            if not is_last:
                for future_name in self.route[current_wp_idx + 1:]:
                    if future_name in self.nodes:
                        nx, ny = self.nodes[future_name]
                        break

            reached, v, w = self._drive_smooth_step(tx, ty, nx, ny, is_last)

            # simulate movement locally instead of calling real robot service
            self._simulate_robot_movement(v, w)

            cx, cy, ch = self._get_pose()
            trajectory.append((cx, cy, ch))

            if reached:
                waypoint_hits.append((node_name, cx, cy))
                print(f"% GraphNav:: simulated hit '{node_name}' at ({cx:.2f}, {cy:.2f})")
                current_wp_idx += 1

        if current_wp_idx < len(self.route):
            print("% GraphNav:: simulation reached max steps limit before finishing route")
        else:
            print("% GraphNav:: simulation finished")

        return trajectory, waypoint_hits

    def _simulate_robot_movement(self, v: float, w: float) -> None:
        """Simulate robot movement for testing. Updates pose based on commanded v (m/s) and w (rad/s)."""
        dt = self.loop_dt
        x, y, h = self._get_pose()
        # simple unicycle model
        x += v * math.cos(h) * dt
        y += v * math.sin(h) * dt
        h += w * dt
        # update the pose directly (bypassing spose's normal updates)
        pose.pose[0] = x / self.distance_scale
        pose.pose[1] = y / self.distance_scale
        pose.pose[2] = h / self.heading_scale

    # ------------------------------------------------------------------
    def terminate(self):
        service.send("robobot/cmd/ti", "rc 0 0")
        print("% GraphNav:: terminated")


# module-level singleton
graph_nav = GraphNav()

