"""Floor/wall boundary -> ground points -> pose registration against the wall lattice.

Every maze wall stands on a line x = 0.2 + 0.4k or y = 0.2 + 0.4k of the driver's
frame (origin at the start cell centre). A floor/wall boundary pixel is the foot of a
wall, so its ground position must fall on one of those lines. Fitting the pose to that
constraint gives heading and the position inside the cell; which cell it is comes from
the motion prediction. No map, simulator truth or recorded data is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FX = FY = 92.0
CX, CY = 160.0, 120.0
CAMERA_HEIGHT = 0.214
# The four goal photos are rendered from a lower camera: their floor boundary follows
# v = 120 + 15.0 / X (fitted on mazes 9-14, residual 0.3 px), i.e. 0.163 m.
GOAL_PHOTO_HEIGHT = 0.163
CELL = 0.4
HALF = 0.2

# BGR floor tiles and the JPEG/MSAA blends between them lie on this segment.
FLOOR_A = np.array([232.0, 232.0, 232.0])
FLOOR_B = np.array([218.0, 181.0, 157.0])


def floor_mask(image: np.ndarray, tol: float = 22.0) -> np.ndarray:
    p = image.astype(np.float32)
    d = FLOOR_B - FLOOR_A
    t = np.clip(((p - FLOOR_A) @ d) / float(d @ d), 0.0, 1.0)
    nearest = FLOOR_A + t[..., None] * d
    return np.linalg.norm(p - nearest, axis=2) < tol


def ground_points(
    image: np.ndarray, step: int = 2, max_range: float = 2.0, height: float | None = None
) -> np.ndarray:
    """Robot-frame wall feet (forward X, left Y, image column u), ordered by column."""
    mask = floor_mask(image)
    h = mask.shape[0]
    cols = np.arange(0, mask.shape[1], step)
    m = mask[:, cols]
    run = np.cumprod(m[::-1], axis=0).sum(axis=0)  # floor run up from the bottom row
    v = h - run - 0.5  # boundary between last floor row and first non-floor row
    keep = (run > 0) & (run < h) & (v > CY + 6)
    v, u = v[keep], cols[keep].astype(float)
    x = FY * (CAMERA_HEIGHT if height is None else height) / (v - CY)
    y = -(u - CX) * x / FX
    near = x < max_range
    return np.stack([x[near], y[near], u[near]], axis=1)


def wall_families(points: np.ndarray, theta: float, step: int = 2):
    """Which wall direction each foot lies on, from the boundary's local tangent.

    Returns (family, tangent_residual): family 0 = wall along world x (y = const),
    1 = wall along world y (x = const), -1 = unknown (edge, occlusion, noise).
    A foot is only used on a wall whose direction its own boundary shows; this keeps a
    long side wall from being matched to a crossing grid line by coincidence.
    """
    n = len(points)
    family = np.full(n, -1)
    residual = np.full(n, np.nan)
    k = 3  # half window: boundary rows are whole pixels, so near walls look like
    #        stairs; a 6-column chord follows the wall, a 1-column step does not.
    if n < 2 * k + 1:
        return family, residual
    d = np.diff(points[:, :2], axis=0)
    adjacent = np.diff(points[:, 2]) <= step + 0.5
    reach = np.hypot(d[:, 0], d[:, 1])
    limit = 0.03 + 0.12 * np.maximum(points[:-1, 0], points[1:, 0]) ** 2
    joined = adjacent & (reach < limit)  # segment j joins point j and j + 1
    breaks = np.concatenate([[0], np.cumsum(~joined)])  # constant along a wall run
    for i in range(k, n - k):
        if breaks[i - k] != breaks[i + k]:
            continue
        chord = points[i + k, :2] - points[i - k, :2]
        angle = math.atan2(chord[1], chord[0]) + theta  # wall direction in the world
        wrapped = (angle + math.pi / 4) % (math.pi / 2) - math.pi / 4  # off-axis error
        if abs(wrapped) < math.radians(20):
            family[i] = int(round((angle - wrapped) / (math.pi / 2))) % 2  # 0: along x
            residual[i] = wrapped
    return family, residual


def _to_lines(a):
    """Signed distance of coordinate a to the nearest wall line 0.2 + 0.4k."""
    return (a - HALF) - CELL * np.round((a - HALF) / CELL)


@dataclass
class Fix:
    pose: np.ndarray  # x, y, theta in the driver's frame
    used: int  # points that agreed with a wall line; 0 = no fix, keep the prediction
    info_along: float  # information on position along the current heading
    info_across: float
    sigma: np.ndarray | None = None  # posterior standard deviations (x, y, theta)


def register(points: np.ndarray, prior: np.ndarray, prior_sigma: np.ndarray, iters: int = 4):
    """MAP pose: wall-line residuals of the ground points plus the motion prior."""
    pose = prior.astype(float).copy()
    P = np.diag(1.0 / prior_sigma**2)
    if len(points) < 6:
        return Fix(pose, 0, 0.0, 0.0)
    family, tangent = wall_families(points, pose[2])
    known = family >= 0
    if known.sum() < 6:
        return Fix(pose, 0, 0.0, 0.0)
    # Heading first, from the wall directions alone (no position needed). Near feet
    # give the cleanest tangents.
    near = known & (points[:, 0] < 1.2)
    if near.sum() >= 6:
        limit = max(2.5 * prior_sigma[2], math.radians(3))
        pose[2] -= float(np.clip(np.median(tangent[near]), -limit, limit))
        family, tangent = wall_families(points, pose[2])
        known = family >= 0
    # A far wall across the heading is measured by its depth, where one pixel is many
    # centimetres; it would set the position along the corridor badly. Far walls along
    # the heading are measured sideways and stay accurate, so they are kept.
    across_family = 1 if abs(math.cos(pose[2])) > abs(math.sin(pose[2])) else 0
    X, Y = points[known, 0], points[known, 1]
    vertical = family[known] == 1
    sigma = 0.006 + 0.6 * X**2 / (FY * CAMERA_HEIGHT)  # ~0.6 px boundary error
    # Neighbouring columns see the same wall: their errors are not independent.
    w_base = 1.0 / (sigma**2 * 6.0)
    # Far walls across the heading share one pixel row, so their depth error is common
    # to all of them: together they are one weak (~6 cm) along-corridor measurement.
    # Enough to stop dead-reckoning drift in a long corridor, not enough to jump.
    far_across = (family[known] == across_family) & (X > 1.0)
    total = w_base[far_across].sum()
    if total > 1.0 / 0.06**2:
        w_base = np.where(far_across, w_base * (1.0 / 0.06**2) / total, w_base)
    used = 0
    H = P
    for _ in range(iters):
        x, y, th = pose
        c, s = math.cos(th), math.sin(th)
        wx = x + c * X - s * Y
        wy = y + s * X + c * Y
        r = np.where(vertical, _to_lines(wx), _to_lines(wy))
        gate = np.abs(r) < 0.08
        used = int(gate.sum())
        J = np.zeros((len(X), 3))
        J[:, 0] = vertical
        J[:, 1] = ~vertical
        J[:, 2] = np.where(vertical, -s * X - c * Y, c * X - s * Y)
        w = np.where(gate, w_base, 0.0)
        H = P + (J * w[:, None]).T @ J
        g = (J * w[:, None]).T @ r + P @ (pose - prior)
        delta = np.linalg.solve(H, g)
        pose = pose - delta
        if np.abs(delta).max() < 1e-4:
            break
    # A fix far outside the motion prediction is a wrong wall-line match (open rooms,
    # walls seen edge-on), not a robot that jumped. Keep the prediction instead.
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    r = np.where(vertical, _to_lines(x + c * X - s * Y), _to_lines(y + s * X + c * Y))
    near_used = (np.abs(r) < 0.08) & (X < 1.0)
    rms = float(np.sqrt(np.mean(r[near_used] ** 2))) if near_used.sum() >= 6 else 0.0
    # The tolerance never drops below 0.12 m / 6 deg: a confident prediction must still
    # accept a real few-centimetre correction (lattice lines are 0.4 m apart).
    tolerance = np.maximum(4.0 * prior_sigma, [0.12, 0.12, math.radians(6)])
    if (np.abs(pose - prior) > tolerance).any() or used < 6:
        return Fix(prior.astype(float).copy(), 0, 0.0, 0.0)
    if rms > 0.02:  # the feet do not sit on wall lines: a wrong match, not a fix
        return Fix(prior.astype(float).copy(), 0, 0.0, 0.0)
    c, s = math.cos(pose[2]), math.sin(pose[2])
    along, across = np.array([c, s]), np.array([-s, c])
    Hxy = H[:2, :2] - P[:2, :2]
    sigma = np.sqrt(np.diag(np.linalg.inv(H)))
    return Fix(pose, used, float(along @ Hxy @ along), float(across @ Hxy @ across), sigma)


def front_clearance(points: np.ndarray, half_width: float = 0.075) -> float:
    """Distance ahead to the nearest wall foot inside the body's swept strip."""
    if not len(points):
        return math.inf
    points = points[:, :2]
    inside = np.abs(points[:, 1]) < half_width
    return float(points[inside, 0].min()) if inside.any() else math.inf
