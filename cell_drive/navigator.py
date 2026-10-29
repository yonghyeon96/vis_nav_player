"""Exploration map + goal candidates -> drive, check the place, refine, face, CHECKIN.

One frame in, one command out; the same object runs locally and on the server.
"""

from __future__ import annotations

import math

import numpy as np

from cell_drive import goal
from cell_drive.driver import BACKWARD, FORWARD, LEFT, RIGHT, TURN_RATE, CellDriver, wrap
from cell_drive.goal_pose import GoalViews

CHECKIN = 16
QUIET = lambda *a: None  # noqa: E731
# A quarter turn costs 0.76 of its 26 ticks, about 1.5 cells: of two routes equally long,
# the one with fewer corners wins, and two corners buy one extra cell at most. (A full
# weight traded cells for corners through open rooms, where the lattice slips along.)
TURN_WEIGHT = 0.76


def plan(cmap, goal_xy, here=None, heading=0.0):
    """Waypoints from `here` (start cell if None) over driven connections to the goal."""
    cells = list(cmap.visits)
    if here is None:
        start = (0, 0)
    else:
        start = min(cells, key=lambda c: math.dist((c[0] * 0.4, c[1] * 0.4), here))
    for end in sorted(cells, key=lambda c: math.dist((c[0] * 0.4, c[1] * 0.4), goal_xy))[:6]:
        path = cmap.route(start, end, heading, turn_weight=TURN_WEIGHT)
        if not path:
            continue
        points = [np.array([c[0] * 0.4, c[1] * 0.4]) for c in path]
        if here is not None and math.dist(points[0], here) > 0.03:
            points.insert(0, np.array(here, float))
        if math.dist(points[-1], goal_xy) > 0.01:
            points.append(np.array(goal_xy, float))
        if len(points) == 1:
            points.append(points[0] + [0.01, 0.0])
        return points
    return None


class GoalNavigator:
    def __init__(self, cmap, candidates, targets, log=QUIET):
        self.cmap = cmap
        self.candidates = candidates  # [(x, y, heading), support], best first
        self.targets = targets
        self.target_features = [goal.features(t) for t in targets]
        self.log = log
        self.choice = 0
        self.goal = candidates[0][0].copy()
        self.driver = None
        self.phase = "plan"
        self.refines = self.servo_steps = self.replans = 0
        self.blocked = []
        self.best_place = (0.0, None)  # (same-place score, pose) of the best place visited
        self.returning = False
        # Last metres: pose relative to the goal straight from the goal photos.
        self.goal_views = GoalViews(targets)
        self.approached = None  # candidate index whose approach has run
        self.approach_steps = self.approach_misses = self.looks = self.disagreed = 0
        self.expected = None  # predicted goal-frame pose after the last command
        self.last_fix = None
        self._first_fix = None  # the fix that started the approach, used once

    def _take_over(self, image) -> bool:
        """Hand the last stretch to the goal-photo approach once the goal is in reach.

        Finishing the route first left the robot a few cm off in the open room and then
        cost a turn on the spot; steering by the goal photos from 0.35 m out goes
        straight to the goal instead.
        """
        if self.approached == self.choice or self.driver.index < len(self.driver.route) - 2:
            return False
        if math.dist(self.driver.pose[:2], self.goal[:2]) > 0.35:
            return False
        fix = self.goal_views.locate(image)
        if fix is None or math.hypot(*fix.pose[:2]) > 0.4:
            return False
        self._first_fix = fix
        self.phase = "approach"
        return True

    def _approach(self, image):
        """One command of the final approach, or None after switching phase.

        The goal photos give the robot's pose in the goal frame (mm-level within 0.25 m
        of the goal on mazes 7-12), with no lattice: in the Development maze's open room
        the lattice let the pose slip a cell. Rotate to face the goal, move at most three
        ticks, look again; at the goal, face its heading and check in.
        """
        if self.approached != self.choice:
            self.approached = self.choice
            self.approach_steps = self.approach_misses = self.looks = self.disagreed = 0
            self.expected = self.last_fix = None
        self.approach_steps += 1
        fix, self._first_fix = self._first_fix or self.goal_views.locate(image), None
        if fix is not None and self.expected is not None:
            # A repeated texture can still fit a turned or shifted pose: trust a fix
            # only if it agrees with the last one moved by what we did since.
            jump = math.dist(fix.pose[:2], self.expected[:2])
            strong = fix.inliers >= 15 and fix.agreement >= 0.9
            if not strong and (
                jump > 0.2 or abs(wrap(fix.pose[2] - self.expected[2])) > math.radians(20)
            ):
                self.disagreed += 1
                if self.disagreed < 2:
                    self.log(f"goal fix ignored: {jump:.2f} m from the expected pose")
                    fix = None
                # Twice in a row: the earlier fix was the wrong one; start from this one.
        if fix is not None:
            self.disagreed = 0
            self.last_fix = fix.pose.copy()
        rate = TURN_RATE * self.driver.turn_scale
        if self.approach_steps > 60:
            if self.last_fix is not None and math.hypot(*self.last_fix[:2]) < 0.1:
                self.log("final approach: out of steps, close enough: checking in")
                self.phase = "checkin"
            else:
                self.phase = "turn"
            return None
        if fix is None:
            self.approach_misses += 1
            if self.approach_misses > 10:
                self.log("final approach: no reliable view of the goal, checking the place")
                self.phase = "turn"
                return None
            if self.expected is not None and self.approach_misses <= 2:
                # Lost it after a move: look again from here, a little to the side.
                return self._turn_rel(LEFT, 3, rate)
            if self.looks == 0:
                # First look along the goal heading, where photo 0 was taken.
                err = wrap(self.goal[2] - self.driver.pose[2])
                if abs(err) > math.radians(8):
                    action = LEFT if err > 0 else RIGHT
                    ticks = int(np.clip(round(abs(err) / rate), 1, 13))
                    return self._turn_rel(action, ticks, rate)
            self.looks += 1
            return self._turn_rel(LEFT, 13, rate)  # look around, 45 degrees at a time
        self.approach_misses = 0
        rel = fix.pose  # robot in the goal frame
        # Keep the driver's map pose in step, for any later phase.
        c, s = math.cos(self.goal[2]), math.sin(self.goal[2])
        self.driver.pose = np.array(
            [
                self.goal[0] + c * rel[0] - s * rel[1],
                self.goal[1] + s * rel[0] + c * rel[1],
                wrap(self.goal[2] + rel[2]),
            ]
        )
        # Full points need <= 0.1 m and <= 10 deg; the fixes are good to ~1 cm. Stop
        # within 5 cm and 3 deg instead of chasing millimetres with extra turns.
        dist = math.hypot(rel[0], rel[1])
        if dist <= 0.05:
            err = wrap(-rel[2])
            if abs(err) > math.radians(3):
                action = LEFT if err > 0 else RIGHT
                return self._turn_rel(action, max(1, round(abs(err) / rate)), rate, rel)
            self.log(f"at the goal: {np.round(rel[:2], 3)} m, {math.degrees(rel[2]):.1f} deg off")
            self.phase = "checkin"
            return None
        # The goal in the robot frame: ahead / to the left.
        c, s = math.cos(rel[2]), math.sin(rel[2])
        ahead = -(c * rel[0] + s * rel[1])
        left = -(-s * rel[0] + c * rel[1])
        speed = self.driver.speed
        if abs(left) <= 0.04:
            # Nearly on the line through the goal: drive or back straight onto it.
            action = FORWARD if ahead > 0 else BACKWARD
            ticks = int(np.clip(round(abs(ahead) / speed), 1, 4))
            step = (1 if action == FORWARD else -1) * ticks * speed
            self.expected = rel + np.array([step * math.cos(rel[2]), step * math.sin(rel[2]), 0.0])
            self.driver._predict(action, ticks)
            return action, ticks
        # Turn onto the line to the goal, facing it or backing toward it, whichever
        # needs less turning now plus at the goal (to its heading).
        bearing = math.atan2(left, ahead)
        options = []
        for travel in (bearing, wrap(bearing + math.pi)):  # forward, backward
            facing = wrap(rel[2] + travel)  # heading while travelling, goal frame
            options.append((abs(travel) + abs(wrap(-facing)), travel))
        _, turn = min(options)
        action = LEFT if turn > 0 else RIGHT
        ticks = int(np.clip(round(abs(turn) / rate), 1, 13))
        return self._turn_rel(action, ticks, rate, rel)

    def _turn_rel(self, action, ticks, rate, rel=None):
        """Rotate in place; predict the goal-frame pose the next fix should show."""
        sign = 1 if action == LEFT else -1
        if rel is not None:
            self.expected = rel + np.array([0.0, 0.0, sign * ticks * rate])
        elif self.expected is not None:
            self.expected = self.expected + np.array([0.0, 0.0, sign * ticks * rate])
        self.driver._predict(action, ticks)
        return action, ticks

    def _next_candidate(self):
        self.choice += 1
        if self.choice >= len(self.candidates) or self.choice > 7:
            if self.best_place[1] is not None and not self.returning:
                self.log("no place passed the check: going back to the closest-looking one")
                self.returning = True
                self.goal = self.best_place[1].copy()
                self.phase = "plan"
            else:
                self.phase = "servo"
        else:
            self.goal = self.candidates[self.choice][0].copy()
            self.log(f"trying goal candidate {self.choice}: {np.round(self.goal[:2], 2)}")
            self.phase = "plan"

    def act(self, image):
        for _ in range(8):  # phase changes do not cost a tick
            if self.phase == "plan":
                here = None if self.driver is None else self.driver.pose[:2]
                heading = 0.0 if self.driver is None else float(self.driver.pose[2])
                points = plan(self.cmap, self.goal[:2], here, heading)
                if points is None:
                    self._next_candidate()
                    continue
                if self.driver is None:
                    self.driver = CellDriver(points)
                else:
                    self.driver.set_route(points)
                self.phase = "drive"
            if self.phase == "drive":
                if self._take_over(image):
                    continue
                action, ticks = self.driver.act(image)
                if self.driver.blocked is not None:
                    # The recorded map was wrong here: drop that connection, re-plan.
                    self.cmap.edges.discard(frozenset(self.driver.blocked))
                    self.blocked.append(self.driver.blocked)
                    self.driver.blocked = None
                    self.replans += 1
                    self.phase = "plan" if self.replans <= 40 else "checkin"
                    continue
                if self.driver.done:
                    self.phase = "approach" if self.approached != self.choice else "turn"
                    continue
                return action, ticks
            if self.phase == "approach":
                step = self._approach(image)
                if step is not None:
                    return step
                continue
            if self.phase == "turn":
                action, ticks = self.driver.turn_to(image, self.goal[2])
                if ticks:
                    return action, ticks
                self.phase = "verify"
            if self.phase == "verify":
                j, shift, matches = goal.same_place(image, self.target_features)
                if j is not None and matches > self.best_place[0]:
                    self.best_place = (matches, self.driver.pose.copy())
                if j is None or matches < 15:
                    self.log(f"place does not look like the goal ({matches:.0f} matches)")
                    self._next_candidate()
                    continue
                self.goal[2] = wrap(self.driver.pose[2] + shift - j * math.pi / 2)
                if self.refines < 2:
                    self.refines += 1
                    xy = goal.refine_here(self.targets, self.driver.pose, self.goal[2])
                    if xy is not None and math.dist(xy, self.driver.pose[:2]) > 0.05:
                        self.log(f"goal refined to {np.round(xy, 2)}")
                        self.goal[:2] = xy
                        self.driver.set_route([self.driver.pose[:2].copy(), np.array(xy)])
                        self.phase = "drive"
                        continue
                self.phase = "servo"
            if self.phase == "servo":
                j, shift, _ = goal.best_view(image, self.target_features)
                self.servo_steps += 1
                if j is None or self.servo_steps > 6:
                    self.phase = "checkin"
                    continue
                err = wrap(shift - j * math.pi / 2)  # turn left by this to face like photo 0
                if abs(err) < math.radians(1.7):
                    self.phase = "checkin"
                    continue
                ticks = max(1, round(abs(err) / TURN_RATE))
                action = LEFT if err > 0 else RIGHT
                self.driver._predict(action, ticks)
                return action, ticks
            if self.phase == "checkin":
                return CHECKIN, 1
        return CHECKIN, 1
