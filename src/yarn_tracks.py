"""Each yarn of a splatted fabric as its own set of Gaussians - found from the Gaussians alone.

A splat draws yarn with small flat Gaussians lying along its surface, their long axes following the
plies (which twist round the yarn's axis). Two facts tell yarns apart without any rule laid over the
fabric:
  - along one yarn, neighbouring Gaussians are close and point the same way;
  - where two yarns cross, they touch, but point in clearly different directions.

  1. directions   each Gaussian's yarn direction: the main axis of the structure tensor of its
                  neighbours' long axes (weighted by opacity x area), so the plies' twist averages
                  out; its strength says how clearly the spot is one yarn (not a crossing, not fog).
  2. link_pieces  Gaussians joined when close and pointing the same way - closer than CLOSE_RADII
                  yarn radii in any direction, or up to LINK_RADII radii when the step runs along the
                  yarn - and the joined groups (connected components) are yarn pieces.
  3. join_pieces  a piece ends where its yarn passed out of sight (under a crossing, round a sharp
                  turn); pieces whose ends meet head-on - close, the gap continuing both directions -
                  are one yarn (greedy, best pair first, two pieces at most at an end).
  4. Gaussians too uncertain to link (fog, crossings) take the yarn of their nearest linked one.

`find_yarns` runs it all; `order_yarns` sorts the yarns by where they lie across their mean
direction, so colours can be handed out in order. Everything is in the splat's own units; lengths
given in mm are carried through `mm_per_unit`. No imgui here.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

YARN_RADIUS_MM = 0.5          # the first guess; find_yarns measures the fabric's own
NEIGHBOURHOOD_RADII = 2.0     # the structure tensor's reach, in yarn radii
MIN_STRENGTH = 0.65           # weaker directions (where yarns touch or cross, fog) are not linked, only attached at the end
SAME_DIRECTION = 0.85         # |cos| between two Gaussians' directions to be one yarn
CLOSE_RADII = 0.6             # this near (yarn radii), any step links
LINK_RADII = 2.0              # up to this far, only a step along the yarn links
ALONG = 0.8                   # |cos| between the step and the yarn's direction for "along"
MIN_PIECE = 30                # Gaussians a piece needs to stand on its own
END_SPAN_RADII = 3.0          # how much of a piece's end is used for the end's position and direction
JOIN_GAP_RADII = 4.0          # ends this far apart (yarn radii) may be joined
JOIN_BEND_COS = 0.5           # ... when both ends point into the gap at least this well (60 deg)


def long_axes(scales: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """Each Gaussian's longest axis (unit vectors), from its scales and rotation (w, x, y, z)."""
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q.T
    rotation = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1),
    ], axis=1)
    longest = np.argmax(np.asarray(scales), axis=1)
    return rotation[np.arange(len(q)), :, longest]


def directions(positions, axes, weight, reach):
    """Each Gaussian's yarn direction and how clear it is (0-1): the structure tensor of the long axes
    of its neighbours within `reach`."""
    positions = np.asarray(positions, dtype=np.float64)
    tree = cKDTree(positions)
    near = tree.query_ball_point(positions, reach)
    out = np.zeros_like(positions)
    strength = np.zeros(len(positions))
    for i, members in enumerate(near):
        a = axes[members]
        w = weight[members]
        tensor = (a * w[:, None]).T @ a
        values, vectors = np.linalg.eigh(tensor)
        out[i] = vectors[:, 2]
        strength[i] = (values[2] - values[1]) / max(values[2], 1e-12)
    return out, strength


def measure_radius(positions, dirs, strength, reach) -> float:
    """The yarn's radius: the median distance of Gaussians, across their yarn's direction, from the
    middle of their clear-direction neighbours (the Gaussians lie on the yarn's surface)."""
    positions = np.asarray(positions, dtype=np.float64)
    tree = cKDTree(positions)
    sample = np.flatnonzero(strength >= MIN_STRENGTH)
    sample = sample[:: max(1, len(sample) // 3000)]
    distances = []
    for i in sample:
        members = tree.query_ball_point(positions[i], reach)
        if len(members) < 8:
            continue
        offsets = positions[members] - positions[members].mean(axis=0)
        across = offsets - np.outer(offsets @ dirs[i], dirs[i])
        distances.append(np.sqrt((across ** 2).sum(axis=1)).mean())
    return float(np.median(distances)) if distances else float("nan")


def link_pieces(positions, dirs, strength, radius):
    """Yarn pieces: Gaussians with clear directions, linked when close and pointing the same way
    (module docstring, 2). Returns a piece number per Gaussian (-1: not linked)."""
    positions = np.asarray(positions, dtype=np.float64)
    clear = np.flatnonzero(strength >= MIN_STRENGTH)
    pairs = cKDTree(positions[clear]).query_pairs(LINK_RADII * radius, output_type="ndarray")
    if not len(pairs):
        return np.full(len(positions), -1)
    i, j = clear[pairs[:, 0]], clear[pairs[:, 1]]
    same = np.abs((dirs[i] * dirs[j]).sum(axis=1)) >= SAME_DIRECTION
    step = positions[j] - positions[i]
    length = np.linalg.norm(step, axis=1)
    along = np.abs((step * dirs[i]).sum(axis=1)) >= ALONG * np.maximum(length, 1e-12)
    keep = same & ((length <= CLOSE_RADII * radius) | along)
    n = len(clear)
    graph = coo_matrix((np.ones(int(keep.sum())), (pairs[keep, 0], pairs[keep, 1])), shape=(n, n))
    _, label = connected_components(graph, directed=False)
    sizes = np.bincount(label)
    pieces = np.full(len(positions), -1)
    big = sizes[label] >= MIN_PIECE
    renumber = {old: new for new, old in enumerate(np.unique(label[big]))}
    pieces[clear[big]] = [renumber[k] for k in label[big]]
    return pieces


def _ends(points, radius):
    """A piece's two ends: (tip, outward direction) each - the direction of the stretch of the piece
    within END_SPAN_RADII radii of either extreme along its main axis, the tip as far out along it as
    the piece reaches."""
    centre = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centre, full_matrices=False)
    axis = vt[0]
    t = (points - centre) @ axis
    ends = []
    for sign, extreme in ((-1, t.min()), (1, t.max())):
        span = points[np.abs(t - extreme) <= END_SPAN_RADII * radius]
        mid = span.mean(axis=0)
        if len(span) >= 3:
            _, _, v = np.linalg.svd(span - mid, full_matrices=False)
            direction = v[0] * np.sign(v[0] @ axis * sign or 1.0)
        else:
            direction = axis * sign
        tip = mid + direction * max(float(((span - mid) @ direction).max()), 0.0) if len(span) else mid
        ends.append((tip, direction))
    return ends


def join_pieces(positions, pieces, radius):
    """Pieces whose ends meet head-on are one yarn (module docstring, 3). Returns a yarn number per
    piece."""
    count = int(pieces.max()) + 1 if (pieces >= 0).any() else 0
    ends = []
    for k in range(count):
        for which, (point, direction) in enumerate(_ends(positions[pieces == k], radius)):
            ends.append((k, which, point, direction))
    candidates = []
    for a in range(len(ends)):
        for b in range(a + 1, len(ends)):
            ka, _, pa, da = ends[a]
            kb, _, pb, db = ends[b]
            if ka == kb:
                continue
            gap = pb - pa
            distance = np.linalg.norm(gap)
            if distance > JOIN_GAP_RADII * radius:
                continue
            if distance > 1e-9:
                into_a, into_b = (gap @ da) / distance, (-gap @ db) / distance
            else:
                into_a = into_b = 1.0
            head_on = -(da @ db)                       # the two ends point toward each other
            if min(into_a, into_b) < JOIN_BEND_COS and head_on < JOIN_BEND_COS:
                continue
            cost = distance / (JOIN_GAP_RADII * radius) + (1.0 - head_on)
            candidates.append((cost, a, b))
    parent = list(range(count))

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    used = set()
    for cost, a, b in sorted(candidates):
        if a in used or b in used:
            continue
        ra, rb = root(ends[a][0]), root(ends[b][0])
        if ra == rb:                                   # would close a loop through itself
            continue
        parent[rb] = ra
        used.update((a, b))
    roots = sorted({root(k) for k in range(count)})
    number = {r: n for n, r in enumerate(roots)}
    return np.array([number[root(k)] for k in range(count)], dtype=int)


def find_yarns(positions, scales, quaternion, opacity, candidates=None, mm_per_unit: float = 1.0):
    """Each Gaussian's yarn (module docstring): {"yarn": number per Gaussian (-1: not a candidate),
    "radius": the yarn radius found (units), "pieces": pieces before joining, "strength": direction
    clarity per Gaussian, "direction": per Gaussian}. `candidates` (bool per Gaussian) is the fabric's
    yarn - the table and the background left out."""
    positions = np.asarray(positions, dtype=np.float64)
    count = len(positions)
    candidates = np.ones(count, dtype=bool) if candidates is None else np.asarray(candidates, dtype=bool)
    index = np.flatnonzero(candidates)
    p = positions[index]
    axes = long_axes(np.asarray(scales)[index], np.asarray(quaternion)[index])
    s = np.sort(np.asarray(scales, dtype=np.float64)[index], axis=1)
    weight = np.asarray(opacity, dtype=np.float64)[index] * s[:, 1] * s[:, 2]
    radius = YARN_RADIUS_MM / float(mm_per_unit)
    dirs, strength = directions(p, axes, weight, NEIGHBOURHOOD_RADII * radius)
    measured = measure_radius(p, dirs, strength, NEIGHBOURHOOD_RADII * radius)
    if np.isfinite(measured) and 0.3 * radius < measured < 3.0 * radius:
        radius = measured
        dirs, strength = directions(p, axes, weight, NEIGHBOURHOOD_RADII * radius)
    pieces = link_pieces(p, dirs, strength, radius)
    yarn_of_piece = join_pieces(p, pieces, radius) if (pieces >= 0).any() else np.zeros(0, dtype=int)
    yarn = np.where(pieces >= 0, yarn_of_piece[np.maximum(pieces, 0)], -1)
    linked = yarn >= 0
    if linked.any() and (~linked).any():
        _, nearest = cKDTree(p[linked]).query(p[~linked])
        yarn[~linked] = yarn[linked][nearest]
    full = np.full(count, -1)
    full[index] = yarn
    out_pieces = np.full(count, -1)
    out_pieces[index] = pieces
    out_strength = np.zeros(count)
    out_strength[index] = strength
    out_dirs = np.zeros((count, 3))
    out_dirs[index] = dirs
    return {"yarn": full, "radius": radius, "pieces": out_pieces, "strength": out_strength, "direction": out_dirs}


def order_yarns(positions, yarn, dirs, flip: bool = False) -> list[int]:
    """The yarns in order across the fabric: sorted by where their middles lie along the axis they
    spread over most, across their mean direction (both measured, nothing fixed). Returns the yarn
    numbers in that order."""
    numbers = [k for k in np.unique(yarn) if k >= 0]
    if not numbers:
        return []
    positions = np.asarray(positions, dtype=np.float64)
    middles = np.array([positions[yarn == k].mean(axis=0) for k in numbers])
    tensor = sum(np.outer(d, d) for d in np.asarray(dirs)[yarn >= 0][:: max(1, int((yarn >= 0).sum()) // 5000)])
    mean_direction = np.linalg.eigh(tensor)[1][:, 2]
    across = middles - np.outer(middles @ mean_direction, mean_direction)
    if len(numbers) > 1:
        _, _, vt = np.linalg.svd(across - across.mean(axis=0), full_matrices=False)
        key = across @ vt[0]
    else:
        key = np.zeros(1)
    ordered = [numbers[k] for k in np.argsort(key)]
    return ordered[::-1] if flip else ordered


BRUSH_REACH_RADII = 4.0       # how far (yarn radii) a brushed yarn grows out from the stroke
BRUSH_STEP_RADII = 1.2        # neighbours this near (yarn radii) carry the selection on


def brush(positions, dirs, strength, seeds, radius, reach_radii: float = BRUSH_REACH_RADII):
    """The yarn a brush stroke lies on: the Gaussians under the stroke (`seeds`, indices) and those
    reached from them step by step - each step to a neighbour within BRUSH_STEP_RADII yarn radii that
    points the same way (SAME_DIRECTION), carried on only by clear directions (MIN_STRENGTH), never
    farther than `reach_radii` from the stroke. Returns a bool mask over all Gaussians."""
    positions = np.asarray(positions, dtype=np.float64)
    seeds = np.unique(np.asarray(seeds, dtype=int))
    chosen = np.zeros(len(positions), dtype=bool)
    if not len(seeds):
        return chosen
    chosen[seeds] = True
    tree = cKDTree(positions)
    seed_tree = cKDTree(positions[seeds])
    frontier = [int(k) for k in seeds if strength[k] >= MIN_STRENGTH]
    while frontier:
        nxt = []
        for i, members in zip(frontier, tree.query_ball_point(positions[frontier], BRUSH_STEP_RADII * radius)):
            members = np.asarray([m for m in members if not chosen[m]], dtype=int)
            if not len(members):
                continue
            same = np.abs(dirs[members] @ dirs[i]) >= SAME_DIRECTION
            near, _ = seed_tree.query(positions[members])
            take = members[same & (near <= reach_radii * radius)]
            chosen[take] = True
            nxt.extend(int(k) for k in take if strength[k] >= MIN_STRENGTH)
        frontier = nxt
    return chosen
