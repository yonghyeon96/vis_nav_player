"""cell_drive: the geometry the driver, the map and the final approach rely on."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "source"))

import numpy as np  # noqa: E402

from cell_drive import explore_map, goal_pose, perception  # noqa: E402
from cell_drive.driver import CellDriver, compress  # noqa: E402


def test_compress_keeps_only_corners():
    pts = [np.array(p, float) for p in [(0, 0), (0.4, 0), (0.8, 0), (0.8, 0.4), (0.8, 0.8)]]
    kept = compress(pts)
    assert [tuple(p) for p in kept] == [(0, 0), (0.8, 0), (0.8, 0.8)]


def test_register_pins_a_shifted_pose_to_the_wall_lattice():
    # Feet on the two side walls of a corridor along x (y = +-0.2), seen from a pose that
    # is really at y = 0.03 but predicted at y = 0.
    xs = np.linspace(0.3, 1.2, 30)
    feet = np.concatenate(
        [np.stack([xs, 0.2 - 0.03 + 0 * xs], 1), np.stack([xs, -0.2 - 0.03 + 0 * xs], 1)]
    )
    points = np.column_stack([feet, np.arange(len(feet))])
    fix = perception.register(points, np.zeros(3), np.array([0.05, 0.05, math.radians(3)]))
    assert fix.used > 0
    assert abs(fix.pose[1] - 0.03) < 0.005
    assert abs(fix.pose[2]) < math.radians(0.5)


def test_moved_turns_and_shifts_by_whole_cells():
    poses = np.array([[0.4, 0.0, 0.0]])
    out = explore_map.moved(poses, 1, (12, 0))  # the Development drive 1 placement
    assert np.allclose(out[0], [12 * 0.4, 0.4, math.pi / 2])


def test_route_prefers_connections_driven_twice():
    cmap = explore_map.CellMap()
    once = [((0, 0), (1, 0)), ((1, 0), (2, 0))]
    twice = [((0, 0), (0, 1)), ((0, 1), (1, 1)), ((1, 1), (2, 1)), ((2, 1), (2, 0))]
    for a, b in once:
        e = frozenset((a, b))
        cmap.edges.add(e)
        cmap.count[e] = 1
    for a, b in twice:
        e = frozenset((a, b))
        cmap.edges.add(e)
        cmap.count[e] = 2
    # Two cells driven once cost 6; the four-cell detour driven twice costs 4.
    assert cmap.route((0, 0), (2, 0)) == [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)]


def test_rigid_fit_recovers_the_robot_pose_in_the_goal_frame():
    rng = np.random.default_rng(0)
    a = rng.uniform(-1, 1, (20, 2))  # wall points, robot frame
    th, t = math.radians(25), np.array([0.12, -0.07])
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    b = a @ R.T + t  # the same points, goal frame
    pose, rms = goal_pose._fit(a, b, np.ones(len(a)))
    assert np.allclose(pose, [t[0], t[1], th], atol=1e-9)
    assert rms < 1e-9


def test_driver_turns_in_place_toward_a_route_behind_it():
    d = CellDriver([np.zeros(2), np.array([0.0, 0.8])])  # straight ahead is +x; route is +y
    action, ticks, reason = d._decide()
    assert reason == "rotate_at_cell_centre" and action == 4 and ticks >= 1  # LEFT
