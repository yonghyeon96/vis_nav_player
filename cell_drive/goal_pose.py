"""The robot's pose relative to the goal, from the four goal photos alone (no map, no lattice).

A feature matched between the camera image and a goal photo is a point on a wall. In both
images the floor boundary under the feature's column gives that wall's floor position (in
the robot frame and in the goal frame), and the feature's row gives its height above the
floor, which must agree between the two images. Two such pairs fix the rigid transform
between the frames; RANSAC over all pairs keeps the consistent ones.

Goal frame: the goal at the origin, heading 0 = the goal heading; photo j looks j * 90 deg
to the left (front, left, back, right).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cell_drive import perception
from cell_drive.goal import MATCHER, features

MAX_RANGE = 2.5  # m: farther floor boundaries are a pixel or two from the horizon


def wall_columns(image: np.ndarray, height: float):
    """Per image column: boundary row v and the wall's floor point (X, Y); NaN if none."""
    mask = perception.floor_mask(image)
    h, w = mask.shape
    run = np.cumprod(mask[::-1], axis=0).sum(axis=0)
    v = h - run - 0.5
    ok = (run > 0) & (run < h) & (v > perception.CY + 4)
    x = np.where(ok, perception.FY * height / np.maximum(v - perception.CY, 1e-3), np.nan)
    x[x > MAX_RANGE] = np.nan
    y = -(np.arange(w) - perception.CX) * x / perception.FX
    return v, x, y


def wall_features(image: np.ndarray, height: float, rotation: float = 0.0):
    """Features that lie on a wall: (descriptors, floor points (n, 2) rotated into the
    output frame, height above the floor (n,), range (n,))."""
    keypoints, descriptors = features(image)
    if descriptors is None:
        return None
    v_floor, x, y = wall_columns(image, height)
    rows = []
    for i, k in enumerate(keypoints):
        u, v = k.pt
        c = int(round(u))
        if not 0 <= c < len(x) or not np.isfinite(x[c]) or v >= v_floor[c] - 2:
            continue
        # Above the boundary in a column, the nearest wall hides everything behind it.
        z = height - (v - perception.CY) * x[c] / perception.FY
        rows.append((i, x[c], y[c], z))
    if len(rows) < 4:
        return None
    r = np.array(rows)
    c, s = math.cos(rotation), math.sin(rotation)
    pts = np.stack([c * r[:, 1] - s * r[:, 2], s * r[:, 1] + c * r[:, 2]], axis=1)
    return descriptors[r[:, 0].astype(int)], pts, r[:, 3], r[:, 1]


@dataclass
class GoalFix:
    pose: np.ndarray  # robot (x, y, heading) in the goal frame
    inliers: int
    pairs: int
    rms: float  # metres, over the inliers
    agreement: float = 0.0  # share of near camera wall points on the goal photos' walls


class GoalViews:
    def __init__(self, targets):
        self.views = []
        scan = []
        for j, photo in enumerate(targets):
            rot = j * math.pi / 2
            wf = wall_features(photo, perception.GOAL_PHOTO_HEIGHT, rotation=rot)
            if wf is not None:
                self.views.append(wf)
            _, x, y = wall_columns(photo, perception.GOAL_PHOTO_HEIGHT)
            ok = np.isfinite(x)
            c, s = math.cos(rot), math.sin(rot)
            scan.append(np.stack([c * x[ok] - s * y[ok], s * x[ok] + c * y[ok]], axis=1))
        # The walls all around the goal, as the four photos see them (goal frame).
        self.scan = np.vstack(scan) if scan else np.zeros((0, 2))

    def agreement(self, image, pose, near: float = 1.5) -> float:
        """Share of the camera's near wall points that land on the goal photos' walls."""
        _, x, y = wall_columns(image, perception.CAMERA_HEIGHT)
        ok = np.isfinite(x) & (x < near)
        if ok.sum() < 10 or len(self.scan) == 0:
            return 0.0
        pts = np.stack([x[ok], y[ok]], axis=1) @ _rot(pose[2]).T + pose[:2]
        d = np.min(np.linalg.norm(pts[:, None, :] - self.scan[None, :, :], axis=2), axis=1)
        return float((d < 0.04 + 0.03 * x[ok]).mean())

    def pairs(self, image):
        """Matched wall points: robot-frame (n, 2), goal-frame (n, 2), range (n,)."""
        mine = wall_features(image, perception.CAMERA_HEIGHT)
        if mine is None or not self.views:
            return None
        d_me, p_me, z_me, r_me = mine
        a, b, rng = [], [], []
        for d_goal, p_goal, z_goal, r_goal in self.views:
            if len(d_goal) < 2 or len(d_me) < 2:
                continue
            for pair in MATCHER.knnMatch(d_me, d_goal, k=2):
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance > 0.75 * n.distance:
                    continue
                i, k = m.queryIdx, m.trainIdx
                far = max(r_me[i], r_goal[k])
                if abs(z_me[i] - z_goal[k]) > 0.03 + 0.03 * far:  # not the same point
                    continue
                a.append(p_me[i])
                b.append(p_goal[k])
                rng.append(far)
        if len(a) < 3:
            return None
        return np.array(a), np.array(b), np.array(rng)

    def locate(self, image, rng_seed: int = 0, iters: int = 300) -> GoalFix | None:
        """The robot pose in the goal frame, or None when no estimate is trustworthy.

        Repeated wall textures make some wrong transforms consistent too. Of the best
        few RANSAC hypotheses, the one whose camera walls land on the goal photos' walls
        wins; it must agree on >= 70 % of the near wall points (correct fixes: median
        96 %, wrong ones mostly < 50 % on mazes 9-12).
        """
        found = self.pairs(image)
        if found is None:
            return None
        a, b, far = found
        tol = 0.04 + 0.04 * far  # floor positions get worse with range
        weight = 1.0 / (0.02 + 0.03 * far) ** 2
        rng = np.random.default_rng(rng_seed)
        hypotheses = {}
        n = len(a)
        for _ in range(iters):
            i, k = rng.choice(n, 2, replace=False)
            da, db = a[k] - a[i], b[k] - b[i]
            if np.linalg.norm(da) < 0.15 or abs(np.linalg.norm(da) - np.linalg.norm(db)) > 0.1:
                continue
            th = math.atan2(db[1], db[0]) - math.atan2(da[1], da[0])
            R = _rot(th)
            t = b[i] - R @ a[i]
            inl = np.linalg.norm(a @ R.T + t - b, axis=1) < tol
            if inl.sum() >= 4:
                hypotheses[tuple(np.flatnonzero(inl))] = int(inl.sum())
        best = None
        for members in sorted(hypotheses, key=hypotheses.get, reverse=True)[:6]:
            m = np.array(members)
            pose, _ = _fit(a[m], b[m], weight[m])
            inl = np.linalg.norm(a @ _rot(pose[2]).T + pose[:2] - b, axis=1) < tol
            if inl.sum() < 4:
                continue
            pose, rms = _fit(a[inl], b[inl], weight[inl])
            agree = self.agreement(image, pose)
            if best is None or agree > best[0]:
                best = (agree, GoalFix(pose, int(inl.sum()), n, rms, agree))
        if best is None or best[0] < 0.7 or best[1].inliers < 6:
            return None
        return best[1]


def _rot(th):
    return np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])


def _fit(a, b, w):
    """Weighted rigid fit b = R a + t: (x, y, theta) and weighted rms."""
    w = w / w.sum()
    ca, cb = w @ a, w @ b
    A, B = a - ca, b - cb
    s = w @ (A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0])
    c = w @ (A[:, 0] * B[:, 0] + A[:, 1] * B[:, 1])
    th = math.atan2(s, c)
    R = _rot(th)
    t = cb - R @ ca
    rms = float(np.sqrt(w @ (np.linalg.norm(a @ R.T + t - b, axis=1) ** 2)))
    return np.array([t[0], t[1], th]), rms
