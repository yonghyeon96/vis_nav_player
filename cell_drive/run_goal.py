"""Local end-to-end test: exploration data -> cell map -> goal -> drive -> CHECKIN.

    uv run python -m cell_drive.run_goal --maze 9 --data artifacts/cell_drive/explore/local-9

The policy side uses only the exploration folder (images, labels, target.jpg) and the
camera frames of the run. The simulator's truth is used to score the check-in and to
report how far the goal estimate was off, nothing else.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import vis_nav_sim as vs
from vis_nav_sdk.local import score

from cell_drive import explore_map, goal
from cell_drive.navigator import CHECKIN, GoalNavigator
from cell_drive.run_route import ROOT, Walls, jpeg, textures

CONTACT = 0.062
CACHE = ROOT / "artifacts" / "cell_drive" / "cache"


def run(maze, data_dir, capture, motion_seed, bias, budget_factor, log=print):
    t0 = time.time()
    targets = goal.split_targets(cv2.imread(str(data_dir / "target.jpg")))
    _, tracks, cmap, cap = explore_map.build(
        data_dir, capture, log=log, cache=CACHE, targets=targets
    )
    poses = np.vstack([t.poses for t in tracks])
    views = np.vstack([t.views for t in tracks])
    candidates = goal.locate_from_views(targets, poses, views, top=6, log=log)
    prep = time.time() - t0
    sim = vs.Simulator(
        textures(),
        maze,
        motion_noise=vs.DEFAULT_NOISE,
        motion_bias=bias,
        motion_seed=motion_seed,
        msaa=4,
    )
    origin = np.array(sim.spawn[:2])
    truth_goal = np.array(sim.target)
    row = dict(
        maze=maze,
        capture=cap,
        map_cells=len(cmap.visits),
        map_edges=len(cmap.edges),
        prep_s=round(prep, 1),
        oracle_steps=sim.oracle_steps,
    )
    if not candidates:
        return dict(row, result="goal_not_found")
    row["candidate_errors_m"] = [
        round(float(np.linalg.norm(c[0][:2] + origin - truth_goal[:2])), 2) for c in candidates[:4]
    ]
    nav = GoalNavigator(cmap, candidates, targets, log=log)
    walls = Walls(sim.grid)
    budget = 10000 if budget_factor <= 0 else int(budget_factor * sim.oracle_steps) + 300
    frame = sim.initial_frame()
    contact = 0
    while sim.steps < budget:
        action, ticks = nav.act(jpeg(sim, frame))
        if action == CHECKIN:
            break
        for _ in range(min(ticks, budget - sim.steps)):
            frame = sim.step(action, 1)[-1]
            x, y, _ = sim.pose()
            contact += walls.distance(x, y) <= CONTACT
    distance, heading = sim.distance_heading()
    efficiency = sim.oracle_steps / max(sim.steps, sim.oracle_steps)
    tier, points_awarded, _ = score(distance, heading, efficiency)
    return dict(
        row,
        result=tier,
        points=points_awarded,
        distance_m=round(distance, 3),
        heading_deg=round(heading, 1),
        ticks=sim.steps,
        spl=round(efficiency, 3),
        contact_ticks=contact,
        phase_at_end=nav.phase,
        candidate_used=nav.choice,
        replans=nav.replans,
        refines=nav.refines,
    )


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--maze", type=int, required=True)
    p.add_argument(
        "--data", type=Path, required=True, help="exploration folder (traj_*, target.jpg)"
    )
    p.add_argument(
        "--capture", type=int, default=None, help="ticks per recorded frame (default: estimate)"
    )
    p.add_argument("--motion-seed", type=int, default=5000)
    p.add_argument("--bias", type=float, nargs=3, default=[0.05, 0.05, 0.3])
    p.add_argument("--budget-factor", type=float, default=0.0, help="0 = server budget 10000")
    args = p.parse_args(argv)
    bias = tuple(args.bias) if any(args.bias) else None
    row = run(
        args.maze, args.data, args.capture, args.motion_seed + args.maze, bias, args.budget_factor
    )
    print(json.dumps(row), flush=True)
    out = ROOT / "artifacts" / "cell_drive" / "goal_runs.jsonl"
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
