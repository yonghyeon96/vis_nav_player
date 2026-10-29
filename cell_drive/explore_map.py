"""Exploration data -> explorer poses in the driver's cell frame -> passable-cell map.

Every recorded drive starts at the spawn facing +x, which is also where a challenge run
starts, so the driver's frame (origin at the start cell centre) is shared. Each frame is
placed by the same wall-lattice registration the driver uses; the action labels only
seed the prediction between frames (they are one sample per capture interval).
A cell is a lattice point (0.4 m); two cells are joined when the explorer drove from
one to the other, so the map is undirected: a corridor can be driven both ways.
"""

from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from cell_drive.driver import BACKWARD, FORWARD, LEFT, RIGHT, CellDriver
from cell_drive.goal import bearing_shift, features, thumb

BITS = {"FORWARD": FORWARD, "BACKWARD": BACKWARD, "LEFT": LEFT, "RIGHT": RIGHT}


def load_records(data_dir: Path):
    """[(image path, action bits)] per trajectory, in recording order."""
    out = []
    for d in sorted(data_dir.glob("traj_*"), key=lambda p: int(p.name.split("_")[1])):
        info = json.loads((d / "data_info.json").read_text(encoding="utf-8"))
        rows = []
        for r in sorted(info, key=lambda r: r["step"]):
            bits = 0
            for name in r["action"]:
                bits |= BITS.get(name, 0)
            rows.append((d / r["image"], bits))
        out.append(rows)
    return out


@dataclass
class Track:
    poses: np.ndarray  # (n, 3) x, y, theta per frame, driver frame
    fixed: np.ndarray  # (n,) bool: the frame's own image confirmed the pose
    thumbs: np.ndarray  # (n, 1200) small normalised images for place lookup
    rejected: int = 0
    views: np.ndarray | None = None  # (n, 4, 2): per goal photo (SIFT matches, heading shift)


def frame_yaw(prev_gray, gray):
    """Heading change between two frames from tracked features (+ = turned left), or None.

    For a turn about the camera the bearing of every point changes by the same angle;
    forward motion pushes left and right points apart, which the median cancels.
    """
    pts = cv2.goodFeaturesToTrack(prev_gray, 200, 0.01, 6)
    if pts is None or len(pts) < 10:
        return None
    nxt, ok, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, winSize=(21, 21), maxLevel=3)
    back, ok2, _ = cv2.calcOpticalFlowPyrLK(
        gray, prev_gray, nxt, None, winSize=(21, 21), maxLevel=3
    )
    good = (ok[:, 0] == 1) & (ok2[:, 0] == 1) & (np.linalg.norm(back - pts, axis=2)[:, 0] < 0.7)
    if good.sum() < 10:
        return None
    u0, u1 = pts[good, 0, 0], nxt[good, 0, 0]
    bearing = np.arctan((u1 - 160.0) / 92.0) - np.arctan((u0 - 160.0) / 92.0)
    return float(np.median(bearing))


def track(
    records, capture: int, limit: int | None = None, image_yaw: bool = True, targets=None
) -> Track:
    """Follow one recorded drive frame by frame (label i moved the robot before frame i).

    Heading change comes from the images (labels sample one tick in `capture`); the
    label only says whether the robot was translating.
    """
    est = CellDriver([np.zeros(2), np.array([0.4, 0.0])])  # route unused: estimator only
    poses, fixed, thumbs, views = [], [], [], []
    target_features = [features(t) for t in targets] if targets is not None else None
    prev_gray = None
    for path, bits in records[:limit]:
        image = cv2.imread(str(path))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dyaw = frame_yaw(prev_gray, gray) if image_yaw and prev_gray is not None else None
        if dyaw is None and bits & (LEFT | RIGHT):
            dyaw = (1 if bits & LEFT else -1) * math.radians(3.4377) * capture
        if dyaw is not None:
            est.pose[2] += dyaw
        if bits & (FORWARD | BACKWARD):
            step = (1 if bits & FORWARD else -1) * 0.03 * capture
            th = est.pose[2] - (dyaw or 0.0) / 2
            est.pose[:2] += step * np.array([math.cos(th), math.sin(th)])
        extra = np.array([0.012 * capture, 0.012 * capture, math.radians(2.0)])
        est.sigma = np.minimum(np.hypot(est.sigma, extra), [0.15, 0.15, math.radians(20)])
        est.pending = None  # no speed/turn learning from labels
        est._correct(image)
        prev_gray = gray
        poses.append(est.pose.copy())
        fixed.append(bool(est.fix is not None and est.fix.used))
        thumbs.append(thumb(image))
        if target_features is not None:
            mine = features(image)
            row = []
            for tf in target_features:
                shift, inliers = bearing_shift(mine, tf)
                row.append((inliers, 0.0 if shift is None else shift))
            views.append(row)
    return Track(
        np.array(poses),
        np.array(fixed),
        np.array(thumbs),
        est.stats["rejected"],
        np.array(views) if views else None,
    )


def estimate_capture(records, frames: int = 600) -> int:
    """Ticks per recorded frame: image-measured turn between two frames that are both
    labelled with the same turn, divided by the turn per tick."""
    turns = []
    prev = None
    for path, bits in records[:frames]:
        gray = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
        if prev is not None and bits in (LEFT, RIGHT) and prev[1] == bits:
            yaw = frame_yaw(prev[0], gray)
            if yaw is not None:
                turns.append(abs(yaw))
        prev = (gray, bits)
    if len(turns) < 10:
        return 3
    return int(np.clip(round(np.median(turns) / math.radians(3.4377)), 1, 10))


def cell_of(xy) -> tuple[int, int]:
    return int(round(xy[0] / 0.4)), int(round(xy[1] / 0.4))


def moved(poses: np.ndarray, k: int, offset) -> np.ndarray:
    """Poses turned by k quarter turns about the origin, then shifted by whole cells."""
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k % 4]
    out = poses.copy()
    out[:, 0] = c * poses[:, 0] - s * poses[:, 1] + offset[0] * 0.4
    out[:, 1] = s * poses[:, 0] + c * poses[:, 1] + offset[1] * 0.4
    out[:, 2] = poses[:, 2] + k * math.pi / 2
    return out


def starts_together(a, b) -> bool:
    """Two start views (features, heading in their drive's frame) taken from the same spot
    in a shared frame: they match, and the bearing shift equals the heading difference."""
    shift, same = bearing_shift(a[0], b[0])
    # Weaker matches are not proof: repeated textures make far-apart start views match too.
    return shift is not None and same >= 30 and abs(shift - (b[1] - a[1])) < math.radians(3)


def align(placed: list[Track], new: Track, log=print):
    """(quarter turns, cell offset, agreeing frames) that put `new` into the frame of the
    already placed tracks, or None.

    A recorded drive does not have to start at the spawn (on the Development data the
    second drive starts 4.8 m away, facing left). Each drive is pinned to the wall
    lattice in its own frame, so the two frames differ by quarter turns and whole cells.
    Pairs of frames that look alike vote for the transform that puts them at the same
    pose; the winner must also look alike where the drives overlap.
    """
    P = np.vstack([t.poses for t in placed])
    T = np.vstack([t.thumbs for t in placed])
    votes = {}
    for j in range(0, len(new.poses), 2):
        sims = T @ new.thumbs[j]
        for i in np.where(sims > 0.92)[0]:
            k = int(round((P[i, 2] - new.poses[j, 2]) / (math.pi / 2))) % 4
            q = moved(new.poses[j : j + 1], k, (0, 0))[0]
            key = (k, *cell_of(P[i, :2] - q[:2]))
            votes[key] = votes.get(key, 0) + 1
    grid = {}
    for i, p in enumerate(P):
        grid.setdefault((*cell_of(p[:2]), int(round(p[2] / (math.pi / 2))) % 4), []).append(i)

    def agreement(k, ox, oy):
        Q = moved(new.poses, k, (ox, oy))
        good = overlap = 0
        for j in range(0, len(Q), 2):
            q = Q[j]
            cand = grid.get((*cell_of(q[:2]), int(round(q[2] / (math.pi / 2))) % 4))
            if not cand:
                continue
            pa = P[cand]
            near = (np.hypot(pa[:, 0] - q[0], pa[:, 1] - q[1]) < 0.12) & (
                np.abs((pa[:, 2] - q[2] + math.pi) % (2 * math.pi) - math.pi) < math.radians(10)
            )
            if near.any():
                overlap += 1
                good += float(np.max(T[np.array(cand)[near]] @ new.thumbs[j])) > 0.85
        return good, overlap

    best = None
    for k, ox, oy in sorted(votes, key=votes.get, reverse=True)[:10]:
        good, overlap = agreement(k, ox, oy)
        if best is None or good > best[2]:
            best = (k, (ox, oy), good, overlap)
    if best is not None:
        k, offset, good, overlap = best
        log(f"  best overlap: turn {k * 90} deg, offset {offset} cells, {good} of {overlap} agree")
        # The best of ten searched transforms can agree by chance: a high bar.
        if good >= 20 and good >= 0.5 * overlap:
            return k, offset, good
    # Starting at the spawn is a single hypothesis, not a search: a lower bar.
    good, overlap = agreement(0, 0, 0)
    log(f"  starting at the spawn: {good} of {overlap} overlapping agree")
    if good >= 5 and good >= 0.5 * overlap:
        return 0, (0, 0), good
    return None


@dataclass
class CellMap:
    edges: set = field(default_factory=set)  # frozenset({cell_a, cell_b})
    visits: dict = field(default_factory=dict)  # cell -> [(trajectory, frame)]
    count: dict = field(default_factory=dict)  # edge -> times the explorer drove it

    def add_track(self, t: int, track_: Track):
        prev = None
        for i, (pose, ok) in enumerate(zip(track_.poses, track_.fixed, strict=True)):
            cell = cell_of(pose[:2])
            self.visits.setdefault(cell, []).append((t, i))
            step = None if prev is None else abs(cell[0] - prev[0]) + abs(cell[1] - prev[1])
            if step == 1 and ok:
                edge = frozenset((prev, cell))
                self.edges.add(edge)
                self.count[edge] = self.count.get(edge, 0) + 1
            prev = cell

    def _route_cells(self, start, goal):
        """Cheapest path counting cells only (the Development-tested planner)."""
        adjacency = {}
        for e in self.edges:
            a, b = tuple(e)
            cost = 1.0 if self.count.get(e, 1) >= 2 else 3.0
            adjacency.setdefault(a, []).append((b, cost))
            adjacency.setdefault(b, []).append((a, cost))
        best = {start: 0.0}
        prev = {start: None}
        heap = [(0.0, start)]
        while heap:
            d, c = heapq.heappop(heap)
            if c == goal:
                path = [c]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return path[::-1]
            if d > best.get(c, math.inf):
                continue
            for n, cost in adjacency.get(c, []):
                if d + cost < best.get(n, math.inf):
                    best[n] = d + cost
                    prev[n] = c
                    heapq.heappush(heap, (d + cost, n))
        return None

    def route(self, start, goal, heading: float = 0.0, turn_weight: float = 0.0):
        """Cheapest path in ticks over driven connections, or None.

        A cell costs 0.4 m of driving (13 ticks) and a quarter turn in place 26 ticks,
        as for the oracle, so of two equally long paths the one with fewer corners wins.
        A connection driven only once costs three cells: on mazes 9-12 every connection
        driven twice or more was real, while about one in eight driven once was a
        tracking slip through a wall.
        """
        if turn_weight == 0:
            return self._route_cells(start, goal)
        cell_ticks = 0.4 / 0.03
        quarter_ticks = 90 / 3.4377 * turn_weight
        adjacency = {}
        for e in self.edges:
            a, b = tuple(e)
            cost = cell_ticks * (1.0 if self.count.get(e, 1) >= 2 else 3.0)
            adjacency.setdefault(a, []).append((b, cost))
            adjacency.setdefault(b, []).append((a, cost))

        def turn_ticks(from_angle, to_angle):
            turn = abs((to_angle - from_angle + math.pi) % (2 * math.pi) - math.pi)
            return turn / (math.pi / 2) * quarter_ticks

        # States are (cell, direction of arrival); the start has the robot's heading.
        start_state = (start, None)
        best = {start_state: 0.0}
        prev = {start_state: None}
        heap = [(0.0, 0, start_state)]
        pushed = 0
        while heap:
            d, _, state = heapq.heappop(heap)
            c, arrived = state
            if c == goal:
                path = [state]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return [s[0] for s in path[::-1]]
            if d > best.get(state, math.inf):
                continue
            facing = heading if arrived is None else arrived * math.pi / 2
            for n, cost in adjacency.get(c, []):
                direction = int(round(math.atan2(n[1] - c[1], n[0] - c[0]) / (math.pi / 2))) % 4
                total = d + cost + turn_ticks(facing, direction * math.pi / 2)
                nxt = (n, direction)
                if total < best.get(nxt, math.inf):
                    best[nxt] = total
                    prev[nxt] = state
                    pushed += 1
                    heapq.heappush(heap, (total, pushed, nxt))
        return None


def build(
    data_dir: Path, capture: int | None = None, log=print, cache: Path | None = None, targets=None
):
    records = load_records(data_dir)
    if capture is None:
        capture = estimate_capture(records[0])
    log(f"capture interval {capture} ticks per frame")
    cmap = CellMap()
    tracks = []
    for t, recs in enumerate(records):
        file = None if cache is None else cache / f"{data_dir.name}_c{capture}_t{t}.npz"
        if file is not None and file.exists():
            z = np.load(file)
            tr = Track(
                z["poses"],
                z["fixed"],
                z["thumbs"],
                int(z["rejected"]),
                z.get("views"),
            )
        else:
            tr = track(recs, capture, targets=targets)
            if file is not None:
                file.parent.mkdir(parents=True, exist_ok=True)
                extra = {} if tr.views is None else {"views": tr.views}
                np.savez(
                    file,
                    poses=tr.poses,
                    fixed=tr.fixed,
                    thumbs=tr.thumbs,
                    rejected=tr.rejected,
                    **extra,
                )
        tracks.append(tr)

    # Drive 0 starts at the spawn (checked on the Development server). A drive whose
    # start views match drive 0's is in the same frame; the others are placed by their
    # overlap with drives already placed, retried until no more can be placed.
    def start_views(t):
        recs = records[t][:8]
        headings = tracks[t].poses[: len(recs), 2]
        return [
            (features(cv2.imread(str(p))), th) for (p, _), th in zip(recs, headings, strict=True)
        ]

    spawn_views = start_views(0)
    placed = {0}
    for t in range(1, len(tracks)):
        agree = sum(starts_together(a, b) for a in spawn_views for b in start_views(t))
        if agree >= 1:
            log(f"trajectory {t}: starts at the spawn ({agree} start views agree)")
            placed.add(t)
    progress = True
    while progress:
        progress = False
        for t in range(len(tracks)):
            if t in placed:
                continue
            found = align([tracks[i] for i in sorted(placed)], tracks[t], log=log)
            if found is not None:
                tracks[t].poses = moved(tracks[t].poses, found[0], found[1])
                placed.add(t)
                progress = True
    for t in range(len(tracks)):
        if t not in placed:
            log(f"trajectory {t}: no overlap with the other drives, left out")
            continue
        cmap.add_track(t, tracks[t])
        log(
            f"trajectory {t}: {len(records[t])} frames, {tracks[t].fixed.mean():.0%} fixed, "
            f"cells {len(cmap.visits)}"
        )
    walled = walled_edges(cmap, {t: (records[t], tracks[t]) for t in placed})
    for e in walled:
        cmap.edges.discard(e)
    log(f"map: {len(walled)} connections removed, a wall seen across them")
    opened = seen_open_edges(cmap, {t: (records[t], tracks[t]) for t in placed})
    for e in opened:
        cmap.edges.add(e)
        cmap.count[e] = 1  # never driven: planned like a connection driven once
    log(f"map: {len(opened)} connections added, seen open but never driven")
    tracks = [tracks[t] for t in sorted(placed)]
    return records, tracks, cmap, capture


def walled_edges(cmap: CellMap, drives: dict, max_count: int = 2, frames: int = 8) -> list:
    """Connections the explorer drove at most `max_count` times whose doorway shows a wall.

    A connection driven once can be a tracking slip through a wall (on the Development
    data one such slip made the run take a dead end). Frames of the drives that look at
    the doorway from up to 1.2 m see that wall: wall feet across the middle of the cell
    boundary, as the driver's blocked-edge test sees it while driving.
    """
    from cell_drive import perception

    poses = []
    for t, (_, tr) in drives.items():
        for i, pose in enumerate(tr.poses):
            poses.append((t, i, pose))
    P = np.array([p for *_, p in poses])
    walled = []
    for e in list(cmap.edges):
        if cmap.count.get(e, 0) > max_count:
            continue
        a, b = (np.array(c, float) * 0.4 for c in e)
        mid, d = (a + b) / 2, (b - a) / 0.4
        rel = mid - P[:, :2]
        dist = np.hypot(rel[:, 0], rel[:, 1])
        bearing = np.abs(wrap_angle(np.arctan2(rel[:, 1], rel[:, 0]) - P[:, 2]))
        looking = np.where((dist > 0.15) & (dist < 1.2) & (bearing < math.radians(30)))[0]
        if len(looking) == 0:
            continue
        pick = looking[np.linspace(0, len(looking) - 1, min(frames, len(looking))).astype(int)]
        seen = 0
        for k in pick:
            t, i, (x, y, th) = poses[k]
            pts = perception.ground_points(cv2.imread(str(drives[t][0][i][0])))
            if len(pts) == 0:
                continue
            c, s = math.cos(th), math.sin(th)
            feet = np.stack(
                [x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]], 1
            )
            along = (feet - mid) @ d
            across = (feet - mid) @ np.array([-d[1], d[0]])
            seen += int(((np.abs(along) < 0.03) & (np.abs(across) < 0.08)).sum() >= 5)
        if seen >= 2 and seen >= 0.5 * len(pick):
            walled.append(e)
    return walled


def seen_open_edges(cmap: CellMap, drives: dict, frames: int = 8) -> list:
    """Connections the explorer never drove but looked straight through.

    On the Midterm data the explorer looked from the goal cell through an open doorway
    and turned away; without that connection the route went round two extra cells.
    A doorway counts as open when frames taken near one cell's centre, facing the
    neighbour, see no wall foot ahead until well past the far side of the neighbour
    cell, and none of them sees one at the doorway.
    """
    from cell_drive import perception

    views = {}  # (cell, direction index) -> [(trajectory, frame, distance to the doorway)]
    for t, (_, tr) in drives.items():
        for i, (x, y, th) in enumerate(tr.poses):
            cell = cell_of((x, y))
            off = np.array([x, y]) - np.array(cell) * 0.4
            if np.hypot(*off) > 0.12:
                continue
            k = round(th / (math.pi / 2)) % 4
            if abs(wrap_angle(th - k * math.pi / 2)) > math.radians(10):
                continue
            d = [(1, 0), (0, 1), (-1, 0), (0, -1)][k]
            views.setdefault((cell, k), []).append((t, i, 0.2 - (off[0] * d[0] + off[1] * d[1])))
    opened = []
    for cell in cmap.visits:
        for k, d in enumerate([(1, 0), (0, 1), (-1, 0), (0, -1)]):
            other = (cell[0] + d[0], cell[1] + d[1])
            e = frozenset((cell, other))
            if other not in cmap.visits or e in cmap.edges or e in opened:
                continue
            seen = views.get((cell, k), [])
            if len(seen) < 2:
                continue
            spread = np.linspace(0, len(seen) - 1, min(frames, len(seen))).astype(int)
            pick = [seen[j] for j in spread]
            clear = []
            for t, i, door in pick:
                pts = perception.ground_points(cv2.imread(str(drives[t][0][i][0])))
                clear.append(perception.front_clearance(pts) - door)  # distance past the doorway
            clear = np.array(clear)
            # Open: no frame sees a wall foot at the doorway, and the next wall ahead is
            # most of the neighbour cell (0.4 m) past it in nearly all of them.
            if (clear > 0.3).mean() >= 0.75 and (clear < 0.12).sum() == 0:
                opened.append(e)
    return opened


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
