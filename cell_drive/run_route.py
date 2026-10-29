"""Local driving test: follow the shortest cell route from the spawn to the goal cell.

    uv run python -m cell_drive.run_route --seeds 7 8 9 --viewer

The evaluator (this file) knows the maze, the route and the true pose; the driver gets
only JPEG camera frames and the route as waypoints relative to its start cell. Motion
noise and per-run motion bias (speed scale, turn scale, curvature) are on, as on the
server. Contact = robot centre within 0.062 m of a wall line (a head-on stop is
0.0595 m), checked after every tick.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import vis_nav_sim as vs

from cell_drive.driver import CellDriver, wrap

ROOT = Path(__file__).resolve().parents[1]
CONTACT = 0.062


def textures():
    found = sorted((ROOT / "cache" / "vis-nav").glob("textures-*"))
    return vs.Textures(found[-1]) if found else vs.Textures.find()


def lattice_route(grid, start, goal):
    """BFS over cell centres (odd row/col) two grid steps apart; returns grid cells."""
    rows, cols = len(grid), len(grid[0])

    def free(r, c):
        return 0 <= r < rows and 0 <= c < cols and grid[r][c] != "#"

    target = (goal[0] | 1, goal[1] | 1) if free(goal[0] | 1, goal[1] | 1) else None
    candidates = [(goal[0] + dr, goal[1] + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    ends = {c for c in candidates if c[0] % 2 and c[1] % 2 and free(*c)}
    if target:
        ends.add(target)
    prev = {start: None}
    queue = deque([start])
    end = None
    while queue:
        cell = queue.popleft()
        if cell in ends:
            end = cell
            break
        r, c = cell
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            if (
                free(r + dr, c + dc)
                and free(r + 2 * dr, c + 2 * dc)
                and (r + 2 * dr, c + 2 * dc) not in prev
            ):
                prev[(r + 2 * dr, c + 2 * dc)] = cell
                queue.append((r + 2 * dr, c + 2 * dc))
    path = [end]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    path = path[::-1]
    if tuple(goal) != end and abs(goal[0] - end[0]) + abs(goal[1] - end[1]) == 1:
        path.append(tuple(goal))  # goal on a connector cell: half a cell further
    return path


class Walls:
    def __init__(self, grid):
        segs = []
        for r, row in enumerate(grid):
            for c, ch in enumerate(row):
                if ch != "#":
                    continue
                segs.append((c * 0.2, r * 0.2, c * 0.2, r * 0.2))
                if c + 1 < len(row) and row[c + 1] == "#":
                    segs.append((c * 0.2, r * 0.2, (c + 1) * 0.2, r * 0.2))
                if r + 1 < len(grid) and grid[r + 1][c] == "#":
                    segs.append((c * 0.2, r * 0.2, c * 0.2, (r + 1) * 0.2))
        s = np.array(segs)
        self.a, self.d = s[:, :2], s[:, 2:] - s[:, :2]
        self.len2 = (self.d**2).sum(1)

    def distance(self, x, y):
        p = np.array([x, y])
        t = np.where(
            self.len2 > 0, ((p - self.a) * self.d).sum(1) / np.maximum(self.len2, 1e-12), 0
        )
        q = self.a + self.d * t.clip(0, 1)[:, None]
        return float(np.sqrt(((q - p) ** 2).sum(1)).min())


def jpeg(sim, frame):
    return cv2.imdecode(np.frombuffer(sim.jpeg(frame, 90, "444"), np.uint8), cv2.IMREAD_COLOR)


def draw(sim, image, driver, route_world, origin, truth_trail, est_trail):
    cam = image.copy()
    for X, _, u in driver.points:
        v = int(round(120 + 92 * 0.214 / X))
        cv2.circle(cam, (int(u), min(v, 239)), 1, (0, 200, 0), -1)
    cv2.putText(
        cam, driver.last.get("reason", ""), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
    )
    grid = sim.grid
    scale = 26
    top = np.full((len(grid) * scale // 2 + 8, len(grid[0]) * scale // 2 + 8, 3), 255, np.uint8)

    def px(x, y):
        return int(x / 0.2 * scale / 2) + 4, int(y / 0.2 * scale / 2) + 4

    walls = Walls(grid)
    for a, d in zip(walls.a, walls.d, strict=False):
        cv2.line(top, px(*a), px(*(a + d)), (60, 60, 60), 2)
    for p, q in zip(route_world, route_world[1:], strict=False):
        cv2.line(top, px(*p), px(*q), (255, 170, 0), 1)
    for trail, colour in ((truth_trail, (0, 160, 0)), (est_trail, (0, 0, 255))):
        for p, q in zip(trail, trail[1:], strict=False):
            cv2.line(top, px(*p[:2]), px(*q[:2]), colour, 1)
        if trail:
            x, y, th = trail[-1]
            cv2.arrowedLine(
                top, px(x, y), px(x + 0.15 * math.cos(th), y + 0.15 * math.sin(th)), colour, 2
            )
    cam = cv2.resize(cam, (cam.shape[1] * 2, cam.shape[0] * 2))
    h = max(cam.shape[0], top.shape[0])

    def pad(im):
        white = (255, 255, 255)
        return cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=white)

    return np.hstack([pad(cam), pad(top)])


def run_one(seed, motion_seed, bias, budget_factor, viewer, delay, out):
    sim = vs.Simulator(
        textures(),
        seed,
        motion_noise=vs.DEFAULT_NOISE,
        motion_bias=bias,
        motion_seed=motion_seed,
        msaa=4,
    )
    frame = sim.initial_frame()
    walls = Walls(sim.grid)
    cells = lattice_route(sim.grid, tuple(sim.spawn_cell), tuple(sim.target_cell))
    origin = np.array(sim.spawn[:2])
    route_world = [np.array([c * 0.2, r * 0.2]) for r, c in cells]
    driver = CellDriver([p - origin for p in route_world])
    budget = int(budget_factor * sim.oracle_steps) + 200
    contact_ticks, min_wall, max_off = 0, math.inf, 0.0
    truth_trail, est_trail = [], []
    termination = "budget_exhausted"
    trace = (out / f"maze{seed}_trace.jsonl").open("w", encoding="utf-8")
    polyline = [(route_world[i], route_world[i + 1]) for i in range(len(route_world) - 1)]

    def off_route(x, y):
        p = np.array([x, y])
        best = math.inf
        for a, b in polyline:
            d = b - a
            t = np.clip((p - a) @ d / (d @ d), 0, 1)
            best = min(best, float(np.linalg.norm(a + t * d - p)))
        return best

    while sim.steps < budget:
        image = jpeg(sim, frame)
        action, ticks = driver.act(image)
        truth = np.array(sim.pose())
        est = driver.pose.copy()
        est[:2] += origin
        truth_trail.append(truth)
        est_trail.append(est)
        err = est - truth
        err[2] = wrap(err[2])
        trace.write(
            json.dumps(
                dict(
                    step=sim.steps,
                    action=action,
                    ticks=ticks,
                    **driver.last,
                    truth_evaluator_only=truth.round(4).tolist(),
                    pose_error_evaluator_only=err.round(4).tolist(),
                )
            )
            + "\n"
        )
        if viewer:
            cv2.imshow(
                "cell_drive (camera | top view: green truth, red estimate)",
                draw(sim, image, driver, route_world, origin, truth_trail, est_trail),
            )
            if cv2.waitKey(delay) == 27:
                termination = "viewer_stopped"
                break
        if driver.blocked is not None:  # the true route is open: a false wall
            termination = "false_blocked_edge"
            break
        if driver.done:
            termination = "route_complete"
            break
        for _ in range(min(ticks, budget - sim.steps)):
            frame = sim.step(action, 1)[-1]
            x, y, _ = sim.pose()
            wall = walls.distance(x, y)
            min_wall = min(min_wall, wall)
            contact_ticks += wall <= CONTACT
            max_off = max(max_off, off_route(x, y))
        if max_off > 0.3:
            termination = "left_the_route"
            break
    trace.close()
    x, y, th = sim.pose()
    final = float(np.linalg.norm(np.array([x, y]) - route_world[-1]))
    success = termination == "route_complete" and final <= 0.10 and contact_ticks == 0
    return dict(
        maze=seed,
        motion_seed=motion_seed,
        success=success,
        termination=termination,
        turns=len(driver.route) - 2,
        segments_done=driver.index,
        segments=len(driver.route) - 1,
        ticks=sim.steps,
        oracle_steps=sim.oracle_steps,
        ticks_per_oracle=round(sim.steps / sim.oracle_steps, 2),
        final_error_m=round(final, 3),
        contact_ticks=contact_ticks,
        min_wall_m=round(min_wall, 3),
        max_off_route_m=round(max_off, 3),
        drawn_bias_evaluator_only=[round(v, 3) for v in sim.drawn_bias()],
        learnt=dict(
            turn_scale=round(driver.turn_scale, 3),
            speed=round(driver.speed, 4),
            curvature_deg_per_tick=round(math.degrees(driver.curvature), 3),
        ),
        **driver.stats,
    )


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(7, 16)))
    p.add_argument("--motion-seed", type=int, default=1000, help="run i uses this + maze seed")
    p.add_argument(
        "--bias",
        type=float,
        nargs=3,
        default=[0.05, 0.05, 0.3],
        help="sigmas: speed scale, turn scale, curvature rad/m (0 0 0 = no bias)",
    )
    p.add_argument(
        "--budget-factor", type=float, default=3.0, help="tick budget = factor x oracle + 200"
    )
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--delay", type=int, default=30, help="viewer ms per decision")
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "cell_drive" / time.strftime("%m%d_%H%M%S"),
    )
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    bias = tuple(args.bias) if any(args.bias) else None
    rows = []
    for seed in args.seeds:
        row = run_one(
            seed,
            args.motion_seed + seed,
            bias,
            args.budget_factor,
            args.viewer,
            args.delay,
            args.output,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    ok = sum(r["success"] for r in rows)
    summary = dict(
        success=f"{ok}/{len(rows)}",
        contact_runs=sum(r["contact_ticks"] > 0 for r in rows),
        bias_sigmas=args.bias,
        rows=rows,
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    contact_runs = summary["contact_runs"]
    print(f"ROUTES {ok}/{len(rows)} complete  contact runs {contact_runs}  -> {args.output}")
    if args.viewer:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
