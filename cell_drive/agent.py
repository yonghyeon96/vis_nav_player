"""SDK agent: exploration data -> cell map + goal candidates -> drive -> CHECKIN.

Run it through source/my_agent.py. All map building happens in __init__, before anything
connects to the server.
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import cv2
import numpy as np
from vis_nav_sdk import Action, Agent

from cell_drive import explore_map, goal
from cell_drive.navigator import CHECKIN, GoalNavigator

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts" / "cell_drive" / "cache"


class CellAgent(Agent):
    def __init__(self, data_dir: Path, capture: int | None = None, log=print):
        self.log = log
        t0 = time.time()
        self.photos = goal.split_targets(cv2.imread(str(Path(data_dir) / "target.jpg")))
        _, tracks, self.cmap, self.capture = explore_map.build(
            Path(data_dir), capture, log=log, cache=CACHE, targets=self.photos
        )
        poses = np.vstack([t.poses for t in tracks])
        views = np.vstack([t.views for t in tracks])
        self.candidates = goal.locate_from_views(self.photos, poses, views, top=6, log=log) or []
        log(
            f"map: {len(self.cmap.visits)} cells, {len(self.cmap.edges)} connections; "
            f"{len(self.candidates)} goal candidates; ready in {time.time() - t0:.0f}s"
        )
        self.nav = None

    def setup(self, info):
        # Called twice (a pre-flight with random frames, then the real run): start fresh.
        targets = [np.asarray(t) for t in info.targets] if len(info.targets) == 4 else self.photos
        self.nav = (
            GoalNavigator(copy.deepcopy(self.cmap), self.candidates, targets, log=self.log)
            if self.candidates
            else None
        )

    def act(self, obs):
        if self.nav is None or obs.steps_left <= 2:
            return Action.CHECKIN
        action, ticks = self.nav.act(obs.image)
        if action == CHECKIN:
            return Action.CHECKIN
        return Action(action), int(max(1, min(ticks, obs.steps_left - 2)))

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
