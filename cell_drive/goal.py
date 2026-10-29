"""Where the goal is, from the four goal photos and the explorer poses; final heading servo.

The goal photos face front, left, back and right from the goal pose. Each is matched to
the recorded frames (small-image shortlist, then SIFT); the best frame's explorer pose
plus the bearing shift of the matched features gives the photo's pose, and the wall
lattice refines it. Goal positions in this maze lie on a 0.2 m grid.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from cell_drive import perception

SIFT = cv2.SIFT_create(nfeatures=600)
MATCHER = cv2.BFMatcher(cv2.NORM_L2)


def thumb(image: np.ndarray) -> np.ndarray:
    g = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (40, 30), interpolation=cv2.INTER_AREA)
    v = g.astype(np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def features(image):
    return SIFT.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)


def bearing_shift(fa, fb):
    """(heading of b minus heading of a, consistent matches) for two nearby views."""
    ka, da = fa
    kb, db = fb
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return None, 0
    pairs = MATCHER.knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None, 0
    ua = np.array([ka[m.queryIdx].pt[0] for m in good])
    ub = np.array([kb[m.trainIdx].pt[0] for m in good])
    # A point keeps its world bearing: heading_b - heading_a = phi_a - phi_b, phi = -atan.
    shift = np.arctan((ub - perception.CX) / perception.FX) - np.arctan(
        (ua - perception.CX) / perception.FX
    )
    med = float(np.median(shift))
    inliers = int((np.abs(shift - med) < math.radians(2.5)).sum())
    return med, inliers


def split_targets(mosaic: np.ndarray):
    w = mosaic.shape[1] // 4
    return [mosaic[:, i * w : (i + 1) * w] for i in range(4)]


def match_scale(fa, fb):
    """(consistent matches, median size ratio b/a) between two views, or (0, None)."""
    ka, da = fa
    kb, db = fb
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return 0, None
    pairs = MATCHER.knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return 0, None
    ua = np.array([ka[m.queryIdx].pt[0] for m in good])
    ub = np.array([kb[m.trainIdx].pt[0] for m in good])
    shift = np.arctan((ub - perception.CX) / perception.FX) - np.arctan(
        (ua - perception.CX) / perception.FX
    )
    ok = np.abs(shift - np.median(shift)) < math.radians(2.5)
    size = np.array([kb[m.trainIdx].size / max(ka[m.queryIdx].size, 1e-3) for m in good])
    return int(ok.sum()), float(np.median(size[ok])) if ok.any() else None


def locate_from_views(targets, poses, views, top: int = 4, log=print, image_of=None):
    """Goal (x, y, heading) from per-frame goal-photo matches, by agreement of candidates.

    For each goal photo, the frames with the most consistent matches give a prior pose
    (frame pose, heading + bearing shift); the photo's own wall feet then pin it to the
    lattice. Candidates from different photos and frames vote; the largest cluster wins.
    """
    cands = []
    for j, view in enumerate(targets):
        pts = perception.ground_points(view, height=perception.GOAL_PHOTO_HEIGHT)
        inliers = views[:, j, 0]
        view_features = features(view)
        for k in np.argsort(-inliers)[: top * 3 if image_of else top]:
            if inliers[k] < 12:
                continue
            if image_of is not None:
                # Same walls seen from a metre away also match; only a frame taken from
                # about the goal sees the features at the same size.
                _, scale = match_scale(features(image_of(int(k))), view_features)
                if scale is None or not 0.85 < scale < 1.18:
                    continue
            prior = np.array([poses[k, 0], poses[k, 1], poses[k, 2] + views[k, j, 1]])
            fix = perception.register(pts, prior, np.array([0.12, 0.12, math.radians(6)]))
            pose = fix.pose if fix.used else prior
            registered = 1.0 if fix.used else 0.0  # pinned to the wall lattice: exact
            cands.append(
                (pose[0], pose[1], pose[2] - j * math.pi / 2, float(inliers[k]), j, registered)
            )
    if not cands:
        log("goal: no goal photo matched any recorded frame")
        return None
    c = np.array(cands)  # x, y, goal heading, matches, photo index, registered

    def support(mask):
        # One vote per goal photo (its best frame): the true goal is seen from all
        # four sides, while a look-alike spot elsewhere usually matches one or two
        # photos over many consecutive frames.
        best = [c[mask & (c[:, 4] == j), 3].max(initial=0.0) for j in range(4)]
        seen = sum(b > 0 for b in best)
        return sum(best) * seen / 4  # a place seen in more of the photos ranks higher

    left = np.ones(len(c), bool)
    clusters = []
    while left.any():
        # Frames near the goal lie up to ~0.25 m from it: that is one place.
        near = [left & (np.hypot(c[:, 0] - x, c[:, 1] - y) < 0.25) for x, y, *_ in c]
        scores = [support(m) if left[i] else -1.0 for i, m in enumerate(near)]
        best = int(np.argmax(scores))
        group = near[best]
        left &= ~group
        pinned = group & (c[:, 5] > 0)
        member = pinned if pinned.any() else group  # pose from lattice-pinned photos
        w = c[member, 3]
        x = float(np.average(c[member, 0], weights=w))
        y = float(np.average(c[member, 1], weights=w))
        h = math.atan2(
            np.average(np.sin(c[member, 2]), weights=w), np.average(np.cos(c[member, 2]), weights=w)
        )
        clusters.append(
            (np.array([round(x / 0.2) * 0.2, round(y / 0.2) * 0.2, h]), float(scores[best]))
        )
    log(f"goal: {len(clusters)} candidate places, support {[round(s) for _, s in clusters[:4]]}")
    return clusters


def best_view(image, target_features, min_inliers: int = 8):
    """(view index, heading shift to that view, matches) for the best-matching goal photo."""
    mine = features(image)
    best = (None, None, 0)
    for j, tf in enumerate(target_features):
        shift, inliers = bearing_shift(mine, tf)
        if shift is not None and inliers > best[2]:
            best = (j, shift, inliers)
    return best if best[2] >= min_inliers else (None, None, best[2])


def refine_here(targets, pose, goal_heading):
    """Goal position from the goal photos' wall feet, assuming the goal is within ~0.15 m.

    The robot's own pose is accurate, so the photos' feet only need to be placed on the
    lattice near it. Returns (x, y) snapped to the 0.2 m grid, or None.
    """
    found = []
    for j, view in enumerate(targets):
        prior = np.array([pose[0], pose[1], goal_heading + j * math.pi / 2])
        fix = perception.register(
            perception.ground_points(view, height=perception.GOAL_PHOTO_HEIGHT),
            prior,
            np.array([0.12, 0.12, math.radians(8)]),
        )
        if fix.used >= 20:
            found.append((fix.pose[0], fix.pose[1], fix.used))
    if not found:
        return None
    f = np.array(found)
    x, y = np.median(f[:, 0]), np.median(f[:, 1])
    return round(x / 0.2) * 0.2, round(y / 0.2) * 0.2


def same_place(image, target_features):
    """(view, heading shift, score) where score > 0 means "taken from about here".

    Seen from the same spot, matched features keep their size and their height in the
    image; the same walls seen from a metre away look smaller and nearer the horizon.
    """
    ka, da = features(image)
    best = (None, None, 0.0)
    if da is None or len(da) < 8:
        return best
    for j, (kb, db) in enumerate(target_features):
        if db is None or len(db) < 8:
            continue
        pairs = MATCHER.knnMatch(da, db, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            continue
        ua = np.array([ka[m.queryIdx].pt[0] for m in good])
        ub = np.array([kb[m.trainIdx].pt[0] for m in good])
        size = np.array([kb[m.trainIdx].size / max(ka[m.queryIdx].size, 1e-3) for m in good])
        shift = np.arctan((ub - perception.CX) / perception.FX) - np.arctan(
            (ua - perception.CX) / perception.FX
        )
        med = float(np.median(shift))
        ok = np.abs(shift - med) < math.radians(2.5)
        if ok.sum() < 8:
            continue
        # (image heights differ anyway: the goal photos come from a lower camera)
        scale = float(np.median(size[ok]))
        here = 0.85 < scale < 1.18
        score = ok.sum() * (1.0 if here else 0.0)
        if score > best[2]:
            best = (j, med, float(score))
    return best
