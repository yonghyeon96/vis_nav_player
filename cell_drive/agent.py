"""SDK agent: exploration data -> cell map + goal candidates -> drive -> CHECKIN.

Run it through source/my_agent.py. All map building happens in __init__, before anything
connects to the server.

Every real run (not the SDK's pre-flight) is recorded under artifacts/runs/<time>_<challenge>/:
log.txt (map building, navigator messages, result), trace.tsv (one line per act() call)
and frames/<n>.jpg (the camera frame each decision was made on). The server keeps no
trajectory, so this is what a failed run is diagnosed from. Recording never stops a run:
the first error turns it off and is noted on the terminal.
"""

from __future__ import annotations

import copy
import math
import re
import time
from pathlib import Path

import cv2
import numpy as np
from vis_nav_sdk import Action, Agent

from cell_drive import explore_map, goal
from cell_drive.navigator import CHECKIN, GoalNavigator

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts" / "cell_drive" / "cache"
RUNS = ROOT / "artifacts" / "runs"


class CellAgent(Agent):
    def __init__(self, data_dir: Path, capture: int | None = None, log=print, runs=RUNS):
        self.out = log
        self.runs = runs
        self.lines = []  # map-building messages, written to log.txt once a run starts
        self.run_dir = self.log_file = self.trace = None
        t0 = time.time()
        self.photos = goal.split_targets(cv2.imread(str(Path(data_dir) / "target.jpg")))
        _, tracks, self.cmap, self.capture = explore_map.build(
            Path(data_dir), capture, log=self.log, cache=CACHE, targets=self.photos
        )
        poses = np.vstack([t.poses for t in tracks])
        views = np.vstack([t.views for t in tracks])
        self.candidates = (
            goal.locate_from_views(self.photos, poses, views, top=6, log=self.log) or []
        )
        self.log(
            f"map: {len(self.cmap.visits)} cells, {len(self.cmap.edges)} connections; "
            f"{len(self.candidates)} goal candidates; ready in {time.time() - t0:.0f}s"
        )
        for k, (place, support) in enumerate(self.candidates):
            self.log(
                f"candidate {k}: x {place[0]:+.2f} y {place[1]:+.2f} "
                f"heading {math.degrees(place[2]):+.0f} support {support:.0f}"
            )
        self.nav = None

    def log(self, message):
        self.out(message)
        self.lines.append(f"{time.strftime('%H:%M:%S')} {message}")
        if self.log_file is not None:
            self._record(lambda: self._write_lines())

    def setup(self, info):
        # Called twice (a pre-flight with random frames, then the real run): start fresh.
        targets = [np.asarray(t) for t in info.targets] if len(info.targets) == 4 else self.photos
        self.nav = (
            GoalNavigator(copy.deepcopy(self.cmap), self.candidates, targets, log=self.log)
            if self.candidates
            else None
        )
        self.calls = 0
        self.route = None
        if info.session_id != "preflight" and self.runs is not None:
            self._record(lambda: self._open(info, targets))

    def act(self, obs):
        decision = self._decide(obs)
        if self.trace is not None:
            self._record(lambda: self._trace(obs, decision))
        self.calls += 1
        return decision

    def _decide(self, obs):
        if self.nav is None or obs.steps_left <= 2:
            return Action.CHECKIN
        action, ticks = self.nav.act(obs.image)
        if action == CHECKIN:
            return Action.CHECKIN
        return Action(action), int(max(1, min(ticks, obs.steps_left - 2)))

    def finish(self, result):
        if self.log_file is None:
            return
        self.log(f"result: {result}")
        self.log(f"act() calls: {self.calls}")
        self._record(self._close)

    def hud(self):
        if self.nav is None:
            return ["cell agent: no goal candidate"]
        d = self.nav.driver
        pose = (
            ""
            if d is None
            else f"pose {d.pose[0]:+.2f} {d.pose[1]:+.2f} {math.degrees(d.pose[2]):+.0f}"
        )
        return [
            f"cell agent: {self.nav.phase}, goal candidate {self.nav.choice}",
            f"goal {self.nav.goal[0]:+.2f} {self.nav.goal[1]:+.2f}  {pose}",
            f"replans {self.nav.replans}",
        ]

    # Recording.

    def _record(self, write):
        try:
            write()
        except Exception as exc:  # a full disk or a locked file must not end the run
            self.out(f"recording stopped: {exc!r}")
            self.log_file = self.trace = None

    def _open(self, info, targets):
        challenge = str(info.challenge.get("id", "run"))
        name = f"{time.strftime('%Y%m%d-%H%M%S')}_{re.sub(r'[^A-Za-z0-9_-]', '-', challenge)}"
        self.run_dir = self.runs / name
        (self.run_dir / "frames").mkdir(parents=True, exist_ok=True)
        for j, photo in enumerate(targets):
            cv2.imwrite(str(self.run_dir / f"goal_{j}.jpg"), np.asarray(photo))
        self.log_file = open(self.run_dir / "log.txt", "w", encoding="utf-8")  # noqa: SIM115 (open for the run)
        self.lines[:0] = [
            f"challenge {challenge} session {info.session_id}",
            f"limits {info.limits}",
            f"capture interval {self.capture}",
        ]
        self._write_lines()
        self.trace = open(self.run_dir / "trace.tsv", "w", encoding="utf-8")  # noqa: SIM115
        self.trace.write(
            "call\tsteps_left\tphase\tcandidate\tgoal_x\tgoal_y\tx\ty\theading_deg"
            "\taction\tticks\treason\tsegment\treplans\n"
        )
        self.out(f"recording this run in {self.run_dir}")

    def _write_lines(self):
        for line in self.lines:
            self.log_file.write(line + "\n")
        self.lines.clear()
        self.log_file.flush()

    def _trace(self, obs, decision):
        cv2.imwrite(str(self.run_dir / "frames" / f"{self.calls:05d}.jpg"), np.asarray(obs.image))
        action, ticks = decision if isinstance(decision, tuple) else (decision, 1)
        nav = self.nav
        d = None if nav is None else nav.driver
        if d is not None and (self.route is None or not _same_route(self.route, d.route)):
            self.route = [p.copy() for p in d.route]
            self.log(
                f"route at call {self.calls}: "
                + " ".join(f"({p[0]:+.2f},{p[1]:+.2f})" for p in self.route)
            )
        pose = (
            ("", "", "")
            if d is None
            else (f"{d.pose[0]:+.3f}", f"{d.pose[1]:+.3f}", f"{math.degrees(d.pose[2]):+.1f}")
        )
        row = [
            self.calls,
            obs.steps_left,
            "-" if nav is None else nav.phase,
            "-" if nav is None else nav.choice,
            "" if nav is None else f"{nav.goal[0]:+.2f}",
            "" if nav is None else f"{nav.goal[1]:+.2f}",
            *pose,
            Action(action).name if Action(action).name else int(action),
            ticks,
            "" if d is None else d.last.get("reason", ""),
            "" if d is None else d.index,
            "" if nav is None else nav.replans,
        ]
        self.trace.write("\t".join(str(v) for v in row) + "\n")
        self.trace.flush()

    def _close(self):
        for f in (self.log_file, self.trace):
            if f is not None:
                f.close()
        self.log_file = self.trace = None


def _same_route(a, b):
    return len(a) == len(b) and all(np.allclose(p, q) for p, q in zip(a, b, strict=True))
