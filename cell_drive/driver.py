"""Cell-by-cell driver: follow axis-aligned waypoints, rotate only at cell centres.

The driver's frame has its origin at the start cell centre with heading 0 along +x;
walls lie on x, y = 0.2 + 0.4k. Each call: correct the pose from the image (wall-line
registration), choose one short command, predict its effect. Systematic motion error
(speed scale, turn scale, curvature while driving) is learnt from the corrections.
"""

from __future__ import annotations

import math

import numpy as np

from cell_drive import perception

FORWARD, BACKWARD, LEFT, RIGHT, IDLE = 1, 2, 4, 8, 0
TURN_RATE = math.radians(3.43774677)  # rad per tick, nominal
SPEED = 0.03  # m per tick, nominal
# Up to this much of each corner is turned while still moving in (a FORWARD|LEFT/RIGHT
# tick moves and turns at once). On the 0.5 m arc radius that is the last 0.09 m before
# the cell centre, under 1 cm sideways, and the sideways part is progress along the
# next segment. The rest of the turn is in place: posts collide ~0.17 m from their
# centre, so beside the centre line there are only ~3 cm of slack, and finishing the
# turn while moving out (20 degrees, or 10 on top of a 1.6 cm offset) hit them.
CORNER_ARC = math.radians(10)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def compress(waypoints):
    """Keep only the points where the direction changes (and the ends)."""
    pts = [np.asarray(p, float) for p in waypoints]
    out = [pts[0]]
    for a, b, c in zip(pts, pts[1:], pts[2:], strict=False):
        d1, d2 = b - a, c - b
        if abs(d1[0] * d2[1] - d1[1] * d2[0]) > 1e-9 or d1 @ d2 < 0:
            out.append(b)
    out.append(pts[-1])
    return out


class CellDriver:
    def __init__(self, waypoints):
        self.route = compress(waypoints)
        self.index = 0  # current segment route[index] -> route[index + 1]
        self.pose = np.zeros(3)
        self.sigma = np.array([0.01, 0.01, math.radians(1.0)])
        self.turn_scale = 1.0
        self.speed = SPEED
        self.curvature = 0.0  # rad per forward tick, learnt
        self.done = False
        self.pending = None  # (action, ticks) awaiting a correction
        self.misses = 0  # consecutive frames whose fix was rejected
        self.along_seen_before = False
        self.points = np.zeros((0, 3))
        self.fix = None
        self.stats = dict(
            fixes=0, rejected=0, decisions=0, rotations=0, arcs=0, backoffs=0, blocked=0
        )
        self.blocked = None
        self.blocked_seen = None  # a blocked doorway seen in the last frame, to confirm
        self.last = {}

    # -- estimation -------------------------------------------------------------
    def _predict(self, action, ticks):
        x, y, th = self.pose
        sx, sy, st = self.sigma
        rate = TURN_RATE * self.turn_scale
        turn = (1 if action & LEFT else 0) - (1 if action & RIGHT else 0)
        move = (1 if action & FORWARD else 0) - (1 if action & BACKWARD else 0)
        for _ in range(ticks):
            dth = turn * rate + (self.curvature if move > 0 and not turn else 0.0)
            mid = th + dth / 2
            x += move * self.speed * math.cos(mid)
            y += move * self.speed * math.sin(mid)
            th += dth
        if move:
            # Speed-scale error is systematic: it adds up linearly, not in quadrature.
            grow = 0.003 * ticks
            sx, sy = sx + grow, sy + grow
            st = math.hypot(st, math.radians(0.9) * ticks)
        if turn:
            st = math.hypot(st, math.radians(0.35) * ticks)
            sx, sy = math.hypot(sx, 0.002), math.hypot(sy, 0.002)
        self.pose = np.array([x, y, wrap(th)])
        self.sigma = np.minimum([sx, sy, st], [0.15, 0.15, math.radians(20)])

    def _correct(self, image):
        self.points = perception.ground_points(image)
        prior = self.pose.copy()
        fix = perception.register(self.points, prior, self.sigma)
        if not fix.used and self.misses >= 2:
            # Several frames disagree with the prediction: the prediction is what is
            # wrong. Search within half a cell / 25 degrees instead of drifting on.
            wide = np.maximum(self.sigma, [0.07, 0.07, math.radians(10)])
            fix = perception.register(self.points, prior, wide)
        self.fix = fix
        if not fix.used:
            self.stats["rejected"] += 1
            self.misses += 1
            self.along_seen_before = False
            return
        self.misses = 0
        self.stats["fixes"] += 1
        innovation = fix.pose - prior
        innovation[2] = wrap(innovation[2])
        if self.pending is not None:
            action, ticks = self.pending
            moved = action & (FORWARD | BACKWARD)
            turned = action & (LEFT | RIGHT)
            if (
                turned
                and not moved
                and ticks >= 3
                and abs(innovation[2]) < math.radians(5)
                and fix.sigma[2] < math.radians(1.5)
            ):
                sign = 1 if action & LEFT else -1
                ratio = 1 + sign * innovation[2] / (TURN_RATE * self.turn_scale * ticks)
                self.turn_scale = float(
                    np.clip(0.9 * self.turn_scale + 0.1 * self.turn_scale * ratio, 0.85, 1.15)
                )
            if moved and not turned and fix.sigma[2] < math.radians(2):
                self.curvature = float(
                    np.clip(
                        0.8 * self.curvature + 0.2 * (self.curvature + innovation[2] / ticks),
                        -0.03,
                        0.03,
                    )
                )
            # Speed scale: only between two frames that both see a near wall across the
            # way, so the correction belongs to this one forward command alone.
            along_seen = fix.info_along > 4000
            if along_seen and self.along_seen_before and moved and not (action & BACKWARD):
                c, s = math.cos(prior[2]), math.sin(prior[2])
                along = innovation[0] * c + innovation[1] * s
                if abs(along) < 0.03:
                    est = self.speed + along / ticks
                    self.speed = float(np.clip(0.9 * self.speed + 0.1 * est, 0.025, 0.035))
        self.along_seen_before = fix.info_along > 4000
        self.pose = fix.pose.copy()
        self.pose[2] = wrap(self.pose[2])
        floor = np.array([0.008, 0.008, math.radians(0.8)])
        self.sigma = np.maximum(fix.sigma, floor)

    # -- control ----------------------------------------------------------------
    def act(self, image):
        """One decision from one camera frame -> (action bits, ticks)."""
        self._correct(image)
        self.pending = None
        if self.done:
            return IDLE, 0
        seen = self.blocked_edge()
        # The same wall in two frames in a row: one frame's feet can be misplaced.
        confirmed = seen is not None and seen == self.blocked_seen
        self.blocked_seen = seen
        if confirmed:
            self.blocked = seen
            self.blocked_seen = None
            self.stats["blocked"] += 1
            return IDLE, 0  # the caller removes that connection and gives a new route
        self.stats["decisions"] += 1
        action, ticks, reason = self._decide()
        self.last = dict(
            reason=reason,
            segment=self.index,
            pose=self.pose.round(4).tolist(),
            sigma=self.sigma.round(4).tolist(),
            turn_scale=round(self.turn_scale, 4),
            speed=round(self.speed, 5),
            curvature_deg=round(math.degrees(self.curvature), 3),
        )
        if ticks:
            self._predict(action, ticks)
            self.pending = (action, ticks)
        return action, ticks

    def set_route(self, waypoints):
        """Replace the remaining route, keeping the pose and the learnt motion."""
        self.route = compress(waypoints)
        self.index = 0
        self.done = False

    def blocked_edge(self):
        """(cell, next cell) when a wall stands between them on the current segment.

        Uses the wall feet of the last frame: a run of feet on the boundary line between
        the robot's cell and the next cell along the segment, across the corridor.
        """
        if self.done or self.index >= len(self.route) - 1 or len(self.points) < 5:
            return None
        a, b = self.route[self.index], self.route[self.index + 1]
        d = (b - a) / np.linalg.norm(b - a)
        if min(abs(d[0]), abs(d[1])) > 1e-6:
            return None
        cell = np.round(self.pose[:2] / 0.4) * 0.4
        nxt = cell + 0.4 * d
        if float((b - cell) @ d) < 0.3:  # the segment ends in this cell
            return None
        # Only facing the doorway: mid-turn, a side wall seen obliquely mapped onto the
        # doorway once (Development run 21) and the detour cost 1100 steps.
        if abs(wrap(self.pose[2] - math.atan2(d[1], d[0]))) > math.radians(20):
            return None
        x, y, th = self.pose
        c, s = math.cos(th), math.sin(th)
        wx = x + c * self.points[:, 0] - s * self.points[:, 1]
        wy = y + s * self.points[:, 0] + c * self.points[:, 1]
        feet = np.stack([wx, wy], axis=1) - (cell + 0.2 * d)
        along = feet @ d
        across = feet @ np.array([-d[1], d[0]])
        # Only the middle of the doorway: a wall across the way covers it, while the
        # corner posts and side walls at +-0.2 m do not.
        on_line = (np.abs(along) < 0.03) & (np.abs(across) < 0.08)
        if on_line.sum() >= 5:
            to_cell = (int(round(cell[0] / 0.4)), int(round(cell[1] / 0.4)))
            to_next = (int(round(nxt[0] / 0.4)), int(round(nxt[1] / 0.4)))
            return to_cell, to_next
        return None

    def turn_to(self, image, heading, tolerance=None):
        """Rotate in place toward a heading: (action, ticks), or (IDLE, 0) once there."""
        tolerance = math.radians(2) if tolerance is None else tolerance
        self._correct(image)
        self.pending = None
        err = wrap(heading - self.pose[2])
        rate = TURN_RATE * self.turn_scale
        if abs(err) < max(tolerance, rate / 2):
            return IDLE, 0
        ticks = int(np.clip(round(abs(err) / rate), 1, 6))
        action = LEFT if err > 0 else RIGHT
        self._predict(action, ticks)
        self.pending = (action, ticks)
        return action, ticks

    def _next_turn(self):
        """Signed turn (rad, left > 0) at the end of the current segment, or None."""
        if self.index + 2 >= len(self.route):
            return None
        a, b, c = self.route[self.index : self.index + 3]
        turn = wrap(math.atan2(*(c - b)[::-1]) - math.atan2(*(b - a)[::-1]))
        return turn if abs(turn) > math.radians(30) else None

    def _decide(self):
        while True:
            if self.index >= len(self.route) - 1:
                self.done = True
                return IDLE, 0, "route_complete"
            a, b = self.route[self.index], self.route[self.index + 1]
            seg = b - a
            length = float(np.linalg.norm(seg))
            d = seg / length
            p = self.pose[:2]
            remaining = float((b - p) @ d)
            if remaining <= 0.015:
                self.index += 1
                continue
            break
        lateral = float(d[0] * (p - a)[1] - d[1] * (p - a)[0])  # left of the line > 0
        psi = math.atan2(d[1], d[0])
        heading_error = wrap(self.pose[2] - psi)
        rate = TURN_RATE * self.turn_scale
        progress = length - remaining
        # Facing the wrong way at a cell centre: rotate in place, re-observing every
        # few ticks. Only near the segment start, where the cell centre gives clearance.
        # A short or diagonal segment (the last step onto a goal on the 0.2 m grid) is
        # started facing it squarely: arcing off the last 12 degrees there pushed the
        # robot into the wall end beside the Development goal on two runs.
        diagonal = min(abs(d[0]), abs(d[1])) > 0.1
        start_tolerance = math.radians(3) if diagonal or length < 0.35 else math.radians(12)
        if abs(heading_error) > start_tolerance and (
            progress < 0.08 or abs(heading_error) > math.radians(30)
        ):
            self.stats["rotations"] += 1
            ticks = int(np.clip(round(abs(heading_error) / rate), 1, 4))
            return (RIGHT if heading_error > 0 else LEFT), ticks, "rotate_at_cell_centre"
        clear = perception.front_clearance(self.points)
        if clear < 0.085:
            self.stats["backoffs"] += 1
            self.sigma[:2] += 0.02
            return BACKWARD, 1, "wall_ahead_back_off"
        # Entering a corner: start the next turn while still moving.
        turn = self._next_turn()
        if turn is not None and length >= 0.3 and CORNER_ARC > 0:
            lead = self.speed / rate * math.sin(CORNER_ARC)  # arc radius * sin(arc)
            turned = heading_error * math.copysign(1.0, turn)
            if remaining < lead + 0.01:
                if turned < min(CORNER_ARC, abs(turn) / 2) - rate / 2:
                    self.stats["arcs"] += 1
                    return FORWARD | (LEFT if turn > 0 else RIGHT), 1, "arc_into_corner"
                if turned > -rate:  # already turned in: finish the segment, no correction
                    ticks = int(np.clip(math.floor((remaining + 0.01) / self.speed), 1, 3))
                    return FORWARD, ticks, "into_corner"
        # Steer back onto the line: aim at a point 0.3 m ahead on the segment.
        correction = math.atan2(-lateral, 0.3)
        error = wrap(heading_error - correction) + self.curvature  # what the next tick adds
        if abs(error) > rate / 2:
            self.stats["arcs"] += 1
            return FORWARD | (RIGHT if error > 0 else LEFT), 1, "arc_to_line"
        ticks = int(np.clip(math.floor((remaining + 0.01) / self.speed), 1, 3))
        return FORWARD, ticks, "forward"
