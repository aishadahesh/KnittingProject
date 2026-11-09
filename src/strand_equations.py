"""The yarn of a scanned splat as equations, and the 3D model those equations make again.

A trained splat is a cloud of Gaussians; the yarn is in it, but as millions of blobs, not as strands.
This finds the strands and writes each as a curve with a closed form, then builds yarn back from the
curves alone so the two can be laid side by side:

  1. The yarn: opaque Gaussians lighter than what shows through the holes (Otsu on brightness).
  2. The cloth's plane: the training cameras look at it on average (the dome's mean viewing direction),
     so its normal is theirs; heights are measured along it, toward the cameras.
  3. A top-down map of the yarn on a 0.1 mm grid (the camera's own resolution at 9 cm): where yarn is,
     and how high its top is. Small gaps between plies are filled; specks dropped.
  4. The strands' centrelines: the map thinned to one-pixel lines (Zhang-Suen), split at crossings into
     strands, short spurs pruned. The yarn's radius along each is the distance to the map's edge.
  5. Each strand as an equation in t from 0 to 1 along it, in mm, per coordinate:
        open strand:   p(t) = p0 + (p1 - p0) t + sum_k b_k sin(k pi t)
        closed loop:   p(t) = a0 + sum_k a_k cos(2 pi k t) + b_k sin(2 pi k t)
     with as few terms as reach the tolerance. The centreline runs one radius below the yarn's top.
  6. The rebuilt yarn: a tube of the strand's radius round each curve, and how far it lies from the
     scan both ways (the scan to the tubes; the tubes' upper half to the scan - the underside of the
     yarn was never photographed, so it is not held against the scan).

No imgui here; the Model mode panel (gui_model.py) draws it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from persistence import write_json

GRID_MM = 0.1
MIN_OPACITY = 0.3
# The cloth's slow bow (a hand-width of cloth sags by millimetres) is taken out at this scale, mm, so
# the relief left is the yarn's own.
BOW_MM = 2.0
# Strands are found as ridges - the lit top of each strand runs brighter than the grooves beside it -
# at about half a strand's width, mm.
RIDGE_MM = 0.4
# Yarn pieces smaller than this (mm^2) are specks, not yarn.
MIN_PIECE_MM2 = 0.3
# A spur off a strand shorter than this many yarn radii is a bump of the outline, not a strand.
SPUR_RADII = 2.5
# The shortest strand kept, mm.
MIN_STRAND_MM = 0.6
TUBE_SIDES = 12
SCALE_FILE = "model_scale.json"


@dataclass
class Strand:
    closed: bool
    radius_mm: float
    length_mm: float
    # Coefficients in mm, one row per coordinate (x, y, z in the cloth frame: x, y along it, z up).
    # Open: [p0, p1 - p0, b_1 .. b_K]; closed: [a0, a_1, b_1, .. a_K, b_K].
    coefficients: np.ndarray
    rms_mm: float
    samples: np.ndarray = field(repr=False)     # the traced centreline it was fitted to, mm

    @property
    def terms(self) -> int:
        return (self.coefficients.shape[1] - 2) if not self.closed else (self.coefficients.shape[1] - 1) // 2

    def evaluate(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        basis = _basis(t, self.closed, self.terms)
        return basis @ self.coefficients.T

    def equations(self) -> list[str]:
        """The three coordinate equations, as text, in mm."""
        lines = []
        for axis, row in zip("xyz", self.coefficients):
            if self.closed:
                parts = [f"{row[0]:.3f}"]
                for k in range(1, self.terms + 1):
                    a, b = row[2 * k - 1], row[2 * k]
                    parts.append(f"{_signed(a)} cos({2 * k}πt) {_signed(b)} sin({2 * k}πt)")
            else:
                parts = [f"{row[0]:.3f}", f"{_signed(row[1])} t"]
                parts += [f"{_signed(b)} sin({k if k > 1 else ''}πt)" for k, b in enumerate(row[2:], start=1)]
            lines.append(f"{axis}(t) = " + " ".join(parts))
        return lines


def _signed(value: float) -> str:
    return f"{'-' if value < 0 else '+'} {abs(value):.3f}"


def _basis(t: np.ndarray, closed: bool, terms: int) -> np.ndarray:
    if closed:
        columns = [np.ones_like(t)]
        for k in range(1, terms + 1):
            columns += [np.cos(2 * math.pi * k * t), np.sin(2 * math.pi * k * t)]
    else:
        columns = [np.ones_like(t), t] + [np.sin(math.pi * k * t) for k in range(1, terms + 1)]
    return np.stack(columns, axis=1)


# --------------------------------------------------------------------------------------------- frame

def load_scale(output_dir) -> float | None:
    """mm per splat unit, as recorded beside a trained model (see SCALE_FILE), or None."""
    try:
        return float(json.loads((Path(output_dir) / SCALE_FILE).read_text(encoding="utf-8"))["mm_per_unit"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


# The robot's close scans hold the camera about this far from the cloth (the lens is sharp at 8-10 cm).
TYPICAL_CAMERA_DISTANCE_MM = 90.0


def guess_scale(splat_positions: np.ndarray, camera_centres: np.ndarray) -> float:
    """mm per splat unit when nothing measured it: the cameras taken to be TYPICAL_CAMERA_DISTANCE_MM
    from the cloth's middle. A guess - a board-measured model_scale.json replaces it."""
    middle = np.median(splat_positions, axis=0)
    distance = float(np.median(np.linalg.norm(camera_centres - middle, axis=1)))
    return TYPICAL_CAMERA_DISTANCE_MM / max(distance, 1e-9)


def camera_centres_and_axes(run_dir, dataset_dir):
    """The training cameras' centres and viewing directions in the splat's (nerfstudio normalised) frame."""
    transforms = json.loads((Path(dataset_dir) / "transforms.json").read_text(encoding="utf-8"))
    parser = json.loads(next(Path(run_dir).rglob("dataparser_transforms.json")).read_text(encoding="utf-8"))
    raw = np.array([np.array(f["transform_matrix"], dtype=np.float64) for f in transforms["frames"]])
    move = np.eye(4)
    move[:3, :] = np.array(parser["transform"])
    # As in scripts/crop_splat.py: nerfstudio folds a COLMAP export's applied_transform into its own.
    if "applied_transform" in transforms:
        applied = np.eye(4)
        applied[:3, :] = np.array(transforms["applied_transform"])
        move = move @ np.linalg.inv(applied)
    moved = np.einsum("ij,njk->nik", move, raw)
    centres = moved[:, :3, 3] * float(parser["scale"])
    looking = -moved[:, :3, 2]          # nerfstudio (OpenGL) cameras look down their -z
    looking /= np.linalg.norm(looking, axis=1, keepdims=True)
    return centres, looking


def _quaternion_of(rotation: np.ndarray) -> np.ndarray:
    """(w, x, y, z) of a proper 3x3 rotation matrix."""
    m = rotation
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    x = math.copysign(math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2.0, m[2, 1] - m[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) / 2.0, m[0, 2] - m[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) / 2.0, m[1, 0] - m[0, 1])
    return np.array([w, x, y, z])


def lay_flat(splat, cloth_normal=None, return_frame: bool = False):
    """The splat turned so the cloth lies in its x-y plane with z up toward the cameras, centred on
    the cloth - a viewer orbiting z then looks down on the cloth, not across it. Each Gaussian's
    orientation is turned with it. `return_frame` also returns the (centre, axes) it used, so points
    can be taken back: p = flat @ axes + centre."""
    opaque = splat["opacity"] >= MIN_OPACITY
    positions = splat["positions"].astype(np.float64)
    centre, axes = cloth_frame(positions[opaque] if opaque.sum() > 100 else positions, cloth_normal)
    flat = transform_splat(splat, axes, centre)
    return (flat, centre, axes) if return_frame else flat


def transform_splat(splat, axes: np.ndarray, centre: np.ndarray) -> dict:
    """The splat in a new frame: each position p becomes axes (p - centre) (axes' rows the new x, y, z),
    and each Gaussian's orientation is turned the same way."""
    moved = dict(splat)
    moved["positions"] = ((splat["positions"].astype(np.float64) - centre) @ np.asarray(axes).T).astype(np.float32)
    if "quaternion" in splat:
        w1, x1, y1, z1 = _quaternion_of(np.asarray(axes))
        q = splat["quaternion"].astype(np.float64)
        w2, x2, y2, z2 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        moved["quaternion"] = np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=1).astype(np.float32)
    return moved


# Yarn and what lies under or beside it (a dark table, a board's black squares) differ by at least this
# much in brightness; two groups closer than this are both yarn, lit and in shadow.
YARN_SPLIT = 0.4


def yarn_mask(splat) -> np.ndarray:
    """The Gaussians that are yarn: opaque, and on the light side of an Otsu split of brightness -
    when the split is a real one (YARN_SPLIT); otherwise every opaque Gaussian is yarn."""
    opacity, colours = splat["opacity"], splat["colours"]
    luminance = colours @ np.array([0.299, 0.587, 0.114])
    opaque = opacity >= MIN_OPACITY
    if opaque.sum() < 100:
        return opaque
    threshold = _otsu(luminance[opaque])
    light, dark = luminance[opaque & (luminance >= threshold)], luminance[opaque & (luminance < threshold)]
    if not len(dark) or light.mean() - dark.mean() < YARN_SPLIT:
        return opaque
    return opaque & (luminance >= threshold)


def _otsu(values: np.ndarray, bins: int = 128) -> float:
    hist, edges = np.histogram(values, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    weight = np.cumsum(hist)
    mean = np.cumsum(hist * centres)
    total, total_mean = weight[-1], mean[-1]
    between = (total_mean * weight - mean * total) ** 2 / np.maximum(weight * (total - weight), 1e-12)
    return float(centres[int(np.argmax(between[:-1]))])


def cloth_frame(points_mm: np.ndarray, normal: np.ndarray | None):
    """Origin and axes (x, y along the cloth, z up toward the cameras) for the yarn's points."""
    centre = np.median(points_mm, axis=0)
    if normal is None:
        _, _, vt = np.linalg.svd(points_mm - centre, full_matrices=False)
        normal = vt[2]
    normal = np.asarray(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    flat = points_mm - centre
    flat = flat - np.outer(flat @ normal, normal)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    x_axis = vt[0] - (vt[0] @ normal) * normal
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    return centre, np.stack([x_axis, y_axis, normal])


# ----------------------------------------------------------------------------------------- the map

def _thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning: the mask worn down to one-pixel-wide lines along its middle."""
    image = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            p = image
            n = [p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:], p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]]
            centre = p[1:-1, 1:-1]
            neighbours = sum(n)
            ring = n + [n[0]]
            transitions = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                c1, c2 = n[0] * n[2] * n[4], n[2] * n[4] * n[6]
            else:
                c1, c2 = n[0] * n[2] * n[6], n[0] * n[4] * n[6]
            remove = (centre == 1) & (neighbours >= 2) & (neighbours <= 6) & (transitions == 1) & (c1 == 0) & (c2 == 0)
            if remove.any():
                centre[remove] = 0
                changed = True
        if not changed:
            return image[1:-1, 1:-1].astype(bool)


_OFFSETS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def _neighbour_count(skeleton: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.int32)
    kernel[1, 1] = 0
    return ndimage.convolve(skeleton.astype(np.int32), kernel, mode="constant") * skeleton


def _trace(skeleton: np.ndarray):
    """The skeleton split into paths (lists of (row, col)) that run between ends and crossings, plus
    whether each is a closed ring with no crossing on it."""
    count = _neighbour_count(skeleton)
    nodes = skeleton & (count != 2)
    node_label, _ = ndimage.label(nodes, structure=np.ones((3, 3)))
    visited = np.zeros_like(skeleton, dtype=bool)
    rows, cols = skeleton.shape
    paths = []

    def neighbours(r, c):
        for dr, dc in _OFFSETS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols and skeleton[rr, cc]:
                yield rr, cc

    def walk(start, first):
        path = [start, first]
        previous, current = start, first
        while not nodes[current]:
            visited[current] = True
            following = [n for n in neighbours(*current) if n != previous and n not in path[-3:]]
            if not following:
                break
            # Prefer a node when there is one, so a path ends at its crossing rather than sliding past it.
            following.sort(key=lambda n: (not nodes[n],))
            previous, current = current, following[0]
            path.append(current)
            if visited[current] and not nodes[current]:
                break
        return path

    touching = set()
    for r, c in zip(*np.nonzero(nodes)):
        for n in neighbours(r, c):
            if nodes[n]:
                # Two different crossings side by side: a one-step strand between them, taken once.
                pair = frozenset((node_label[n], node_label[r, c]))
                if len(pair) == 2 and pair not in touching:
                    touching.add(pair)
                    paths.append([(r, c), n])
                continue
            if visited[n]:
                continue
            paths.append(walk((r, c), n))
    # Rings: skeleton loops with no crossing on them.
    closed = [False] * len(paths)
    remaining = skeleton & ~visited & ~nodes
    for r, c in zip(*np.nonzero(remaining)):
        if visited[r, c]:
            continue
        ring = [(r, c)]
        visited[r, c] = True
        current, previous = (r, c), None
        while True:
            following = [n for n in neighbours(*current) if n != previous and not visited[n]]
            if not following:
                break
            previous, current = current, following[0]
            visited[current] = True
            ring.append(current)
        if len(ring) >= 8:
            paths.append(ring)
            closed.append(True)
    return paths, closed, nodes


# Two pieces meeting at a crossing are one strand going on through it when their directions there
# are within this angle of a straight line.
CONTINUE_DEG = 45.0


def _join_through_crossings(paths: list, closed: list, nodes: np.ndarray):
    """Pieces that carry straight on through a crossing joined into one strand - the yarn runs over
    or under the crossing yarn and on, it does not stop there. A chain that comes back to where it
    started is a closed loop."""
    node_label, _ = ndimage.label(nodes, structure=np.ones((3, 3)))
    ends = {}           # node label -> [(path index, end 0/1, outward direction)]
    for index, (path, ring) in enumerate(zip(paths, closed)):
        if ring or len(path) < 3:
            continue
        for end, (tip, inner) in enumerate(((path[0], path[min(8, len(path) - 1)]), (path[-1], path[-min(9, len(path))]))):
            label = node_label[tip]
            if label:
                direction = np.subtract(tip, inner).astype(float)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    ends.setdefault(label, []).append((index, end, direction / norm))
    partner = {}
    limit = math.cos(math.radians(CONTINUE_DEG))
    for incident in ends.values():
        pairs = sorted(((-(a[2] @ b[2]), i, j) for i, a in enumerate(incident) for j, b in enumerate(incident)
                        if i < j and a[0] != b[0]), reverse=True)
        used = set()
        for straightness, i, j in pairs:
            if straightness < limit or i in used or j in used:
                continue
            used |= {i, j}
            a, b = incident[i], incident[j]
            partner[(a[0], a[1])], partner[(b[0], b[1])] = (b[0], b[1]), (a[0], a[1])
    joined, joined_closed, done = [], [], set()

    def chain(start, entry):
        pieces, index, end = [], start, entry
        while True:
            done.add(index)
            path = paths[index] if end == 0 else paths[index][::-1]
            pieces.append(path)
            following = partner.get((index, 1 - end))
            if following is None:
                return pieces, False
            if following[0] == start:
                return pieces, True
            if following[0] in done:
                return pieces, False
            index, end = following

    # Chains with a free end first (start there), then what is left are loops.
    for index in range(len(paths)):
        if index in done or closed[index]:
            continue
        free = [end for end in (0, 1) if (index, end) not in partner]
        if not free:
            continue
        pieces, ring = chain(index, free[0])
        joined.append([p for piece in pieces for p in piece])
        joined_closed.append(ring)
    for index in range(len(paths)):
        if index in done:
            continue
        if closed[index]:
            joined.append(paths[index])
            joined_closed.append(True)
            done.add(index)
            continue
        pieces, ring = chain(index, 0)
        joined.append([p for piece in pieces for p in piece])
        joined_closed.append(ring)
    return joined, joined_closed


def _prune_spurs(skeleton: np.ndarray, radius_px: np.ndarray, rounds: int = 3) -> np.ndarray:
    """Short dead ends - bumps of the yarn's outline, not strands - removed from the skeleton."""
    skeleton = skeleton.copy()
    for _ in range(rounds):
        paths, closed, nodes = _trace(skeleton)
        count = _neighbour_count(skeleton)
        removed = False
        for path, ring in zip(paths, closed):
            if ring or len(path) < 2:
                continue
            free_start, free_end = count[path[0]] == 1, count[path[-1]] == 1
            if free_start == free_end:      # a strand between crossings, or a piece on its own
                continue
            local = float(np.median([radius_px[p] for p in path]))
            if len(path) < SPUR_RADII * max(local, 1.0):
                for p in (path[:-1] if free_start else path[1:]):     # the crossing it hangs from stays
                    skeleton[p] = False
                removed = True
        if not removed:
            break
    return skeleton


def top_down_maps(local: np.ndarray, colours: np.ndarray, low: np.ndarray, high: np.ndarray) -> dict:
    """The cloth seen from above on a GRID_MM grid, from its Gaussians in the cloth frame (mm):
    where cloth is ("support"), the height of its top ("height", mm), how bright that top is
    ("brightness"), and the strands' ridges ("ridges") - the lines the centrelines are traced along."""
    shape = (int((high[1] - low[1]) / GRID_MM) + 1, int((high[0] - low[0]) / GRID_MM) + 1)
    rows = np.clip(((local[:, 1] - low[1]) / GRID_MM).astype(int), 0, shape[0] - 1)
    cols = np.clip(((local[:, 0] - low[0]) / GRID_MM).astype(int), 0, shape[1] - 1)
    density = np.zeros(shape)
    np.add.at(density, (rows, cols), 1.0)
    # The bow: mean height over BOW_MM; the relief above it is the yarn's.
    heights = np.zeros(shape)
    np.add.at(heights, (rows, cols), local[:, 2])
    bow_px = BOW_MM / GRID_MM
    bow = ndimage.gaussian_filter(heights, bow_px) / np.maximum(ndimage.gaussian_filter(density, bow_px), 1e-9)
    relief = local[:, 2] - bow[rows, cols]
    top = np.full(shape, -np.inf)
    np.maximum.at(top, (rows, cols), relief)
    # Brightness of what is on top: each Gaussian counts less the further it lies under the cell's top.
    weight = np.exp(-np.clip(top[rows, cols] - relief, 0.0, None) / 0.3)
    luminance = colours @ np.array([0.299, 0.587, 0.114])
    light, light_weight = np.zeros(shape), np.zeros(shape)
    np.add.at(light, (rows, cols), luminance * weight)
    np.add.at(light_weight, (rows, cols), weight)
    known = np.isfinite(top)
    blur = 1.5
    known_blur = ndimage.gaussian_filter(known.astype(float), blur)
    height = ndimage.gaussian_filter(np.where(known, top, 0.0), blur) / np.maximum(known_blur, 1e-9) + bow
    brightness = ndimage.gaussian_filter(light, blur) / np.maximum(ndimage.gaussian_filter(light_weight, blur), 1e-9)
    # Cloth wherever Gaussians are near (the Gaussians are about 0.2 mm across, the grid 0.1 mm).
    # A fixed count (0.25 per cell) suits a densely packed splat; one that packs its yarn more thinly
    # was left as specks by it, so the cut comes down with this splat's own density.
    spread = ndimage.gaussian_filter(density, 3.0)
    typical = float(np.percentile(spread[spread > 1e-6], 75)) if (spread > 1e-6).any() else 1.0
    support = ndimage.binary_opening(spread > min(0.25, 0.35 * typical), iterations=2)
    if support.any():
        lo_b, hi_b = np.percentile(brightness[support], [2, 98])
        brightness = np.clip((brightness - lo_b) / max(hi_b - lo_b, 1e-9), 0.0, 1.0)
    # Ridges: where brightness curves down most steeply across a line (the Hessian's most negative eigenvalue).
    sigma = RIDGE_MM / GRID_MM
    hxx = ndimage.gaussian_filter(brightness, sigma, order=(0, 2))
    hyy = ndimage.gaussian_filter(brightness, sigma, order=(2, 0))
    hxy = ndimage.gaussian_filter(brightness, sigma, order=(1, 1))
    lowest = 0.5 * (hxx + hyy) - np.sqrt(0.25 * (hxx - hyy) ** 2 + hxy ** 2)
    strength = np.where(support, np.clip(-lowest, 0.0, None), 0.0) * sigma ** 2
    ridges = np.zeros(shape, dtype=bool)
    if (strength > 0).sum() > 100:
        # A little under Otsu's split, so a strand dimmed for a stretch is not cut in two.
        ridges = strength > 0.7 * _otsu(strength[strength > 0])
    ridges = ndimage.binary_closing(ridges, np.ones((3, 3)))
    label, count = ndimage.label(ridges)
    if count:
        area = ndimage.sum(np.ones_like(label), label, index=np.arange(1, count + 1)) * GRID_MM ** 2
        ridges &= np.r_[False, area >= MIN_PIECE_MM2][label]
    return {"support": support, "height": height, "brightness": brightness, "ridge_strength": strength, "ridges": ridges}


# ----------------------------------------------------------------------------------------- fitting

def _resample(points: np.ndarray, closed: bool, step_mm: float = 0.05):
    ring = np.vstack([points, points[:1]]) if closed else points
    segment = np.linalg.norm(np.diff(ring, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(segment)])
    length = float(arc[-1])
    if length <= 0:
        return points, 0.0
    count = max(int(length / step_mm), 8)
    s = np.linspace(0.0, length, count, endpoint=not closed)
    resampled = np.stack([np.interp(s, arc, ring[:, k]) for k in range(3)], axis=1)
    return resampled, length


def fit_strand(points_mm: np.ndarray, closed: bool, radius_mm: float, tolerance_mm: float, max_terms: int) -> Strand:
    """The fewest-term equation (module docstring, step 5) within `tolerance_mm` RMS of the traced centreline."""
    samples, length = _resample(points_mm, closed)
    t = np.linspace(0.0, 1.0, len(samples), endpoint=not closed)
    best = None
    for terms in range(0 if not closed else 1, max_terms + 1):
        basis = _basis(t, closed, terms)
        if basis.shape[1] >= len(samples):
            break
        if closed:
            coefficients = np.linalg.lstsq(basis, samples, rcond=None)[0].T
        else:
            # The ends are held exactly (p0 and p1 - p0 fixed), the sines fitted to what is left.
            p0, p1 = samples[0], samples[-1]
            line = p0 + np.outer(t, p1 - p0)
            if terms:
                sines = np.linalg.lstsq(basis[:, 2:], samples - line, rcond=None)[0]
            else:
                sines = np.zeros((0, 3))
            coefficients = np.vstack([p0, p1 - p0, sines]).T
        rms = float(np.sqrt(np.mean(np.sum((basis @ coefficients.T - samples) ** 2, axis=1))))
        best = Strand(closed, radius_mm, length, coefficients, rms, samples)
        if rms <= tolerance_mm:
            break
    return best


def _smooth_path(path_mm: np.ndarray, closed: bool, sigma_samples: float = 2.0, sigma_height: float = 5.0) -> np.ndarray:
    """The traced pixels smoothed along the path: lightly across the cloth (the pixel steps), more in
    height, where the top of a 0.1 mm grid cell is the noisiest thing measured."""
    if len(path_mm) < 5:
        return path_mm
    mode = "wrap" if closed else "nearest"
    smooth = ndimage.gaussian_filter1d(path_mm, sigma_samples, axis=0, mode=mode)
    smooth[:, 2] = ndimage.gaussian_filter1d(path_mm[:, 2], sigma_height, mode=mode)
    return smooth


# ----------------------------------------------------------------------------------------- the run

@dataclass
class Extraction:
    strands: list[Strand]
    centre: np.ndarray            # cloth frame origin, mm, in splat axes
    axes: np.ndarray              # rows: x, y, z (up) of the cloth frame, in splat axes
    mm_per_unit: float
    yarn_points_mm: np.ndarray    # the scan's yarn, in the cloth frame
    yarn_colours: np.ndarray
    map_mask: np.ndarray          # top-down yarn map (rows along y, cols along x)
    map_skeleton: np.ndarray
    map_origin: np.ndarray        # (x, y) mm of map pixel (0, 0)
    tube: dict = field(default_factory=dict)
    comparison: dict = field(default_factory=dict)
    maps: dict = field(default_factory=dict)

    def to_splat_frame(self, points_mm: np.ndarray) -> np.ndarray:
        """Cloth-frame mm back to the splat's own coordinates (to be drawn with it, same camera)."""
        return (points_mm @ self.axes + self.centre) / self.mm_per_unit

    def summary(self) -> dict:
        lengths = [s.length_mm for s in self.strands]
        return {
            "strands": len(self.strands),
            "closed_loops": int(sum(s.closed for s in self.strands)),
            "total_length_mm": float(sum(lengths)),
            "median_radius_mm": float(np.median([s.radius_mm for s in self.strands])) if self.strands else 0.0,
            "median_terms": float(np.median([s.terms for s in self.strands])) if self.strands else 0.0,
            "coefficients": int(sum(s.coefficients.size + 1 for s in self.strands)),
            "fit_rms_mm": float(np.sqrt(np.mean([s.rms_mm ** 2 for s in self.strands]))) if self.strands else 0.0,
            **self.comparison,
        }


def extract(splat, mm_per_unit: float, cloth_normal=None, tolerance_mm: float = 0.05, max_terms: int = 10,
            progress=lambda message: None) -> Extraction:
    """The splat's yarn as strand equations, the yarn rebuilt from them, and how far apart the two are."""
    progress("Finding the yarn")
    opaque = splat["opacity"] >= MIN_OPACITY
    yarn = yarn_mask(splat)[opaque]
    points = splat["positions"][opaque].astype(np.float64) * mm_per_unit
    colours = splat["colours"][opaque]
    if yarn.sum() < 200:
        raise RuntimeError(f"only {int(yarn.sum())} yarn Gaussians found - is this a cropped cloth splat?")
    centre, axes = cloth_frame(points[yarn], cloth_normal)
    local = (points - centre) @ axes.T
    # A few stray Gaussians far off the cloth would stretch the map: the map covers the middle 99%.
    low, high = np.percentile(local[:, :2], 0.5, axis=0), np.percentile(local[:, :2], 99.5, axis=0)
    inside = np.all((local[:, :2] >= low) & (local[:, :2] < high), axis=1)
    local, colours, yarn = local[inside], colours[inside], yarn[inside]

    progress("Mapping the yarn from above")
    # The map is of the yarn alone: the table showing through the openings (dark blue, opaque in the
    # splat) would otherwise fill every opening in.
    maps = top_down_maps(local[yarn], colours[yarn], low, high)
    mask, height = maps["ridges"], maps["height"]

    progress("Tracing the strands")
    radius_px = ndimage.distance_transform_edt(mask)
    skeleton = _prune_spurs(_thin(mask), radius_px)
    paths, closed, nodes = _trace(skeleton)
    paths, closed = _join_through_crossings(paths, closed, nodes)

    progress(f"Fitting equations to {len(paths)} strands")
    strands = []
    for path, ring in zip(paths, closed):
        pixels = np.array(path)
        if len(pixels) < 3:
            continue
        radius = float(np.median(radius_px[pixels[:, 0], pixels[:, 1]])) * GRID_MM
        xs = low[0] + (pixels[:, 1] + 0.5) * GRID_MM
        ys = low[1] + (pixels[:, 0] + 0.5) * GRID_MM
        zs = height[pixels[:, 0], pixels[:, 1]] - radius
        centreline = _smooth_path(np.stack([xs, ys, zs], axis=1), ring)
        if np.sum(np.linalg.norm(np.diff(centreline, axis=0), axis=1)) < MIN_STRAND_MM:
            continue
        strands.append(fit_strand(centreline, ring, radius, tolerance_mm, max_terms))

    # The scan's yarn as the pictures saw it: its top surface. Gaussians deeper than a yarn's thickness
    # under the top are the splat's filling of what no camera saw, and no strand was traced there.
    rows = np.clip(((local[:, 1] - low[1]) / GRID_MM).astype(int), 0, mask.shape[0] - 1)
    cols = np.clip(((local[:, 0] - low[0]) / GRID_MM).astype(int), 0, mask.shape[1] - 1)
    thickness = 2.0 * (float(np.median([s.radius_mm for s in strands])) if strands else 0.3)
    surface = yarn & (local[:, 2] >= height[rows, cols] - thickness)
    result = Extraction(strands, centre, axes, mm_per_unit, local[surface], colours[surface], mask, skeleton, np.array(low))
    result.maps = maps
    progress("Rebuilding the yarn from the equations")
    result.tube = polyline_tubes([(s.evaluate(np.linspace(0.0, 1.0, max(int(s.length_mm * 20.0), 6), endpoint=not s.closed)),
                                   s.radius_mm, s.closed) for s in strands])
    progress("Comparing the rebuilt yarn with the scan")
    result.comparison = compare_tube(result.tube, result.yarn_points_mm)
    return result


def polyline_tubes(curves) -> dict:
    """A tube round each (points mm, radius mm, closed) curve: vertices, faces, normals, and which curve each vertex is."""
    vertices, normals, faces, owner = [], [], [], []
    offset = 0
    angles = np.linspace(0.0, 2 * math.pi, TUBE_SIDES, endpoint=False)
    for index, (curve, radius, closed) in enumerate(curves):
        curve = np.asarray(curve, dtype=np.float64)
        count = len(curve)
        if count < 2:
            continue
        tangent = np.gradient(curve, axis=0)
        if closed:
            tangent = np.roll(curve, -1, axis=0) - np.roll(curve, 1, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
        # Parallel transport: each ring's frame turned from the last one's, so the tube does not twist.
        reference = np.array([0.0, 0.0, 1.0]) if abs(tangent[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        side = np.cross(tangent[0], reference)
        side /= np.linalg.norm(side)
        sides = []
        for k in range(count):
            side = side - (side @ tangent[k]) * tangent[k]
            side /= max(np.linalg.norm(side), 1e-9)
            sides.append(side)
        sides = np.array(sides)
        ups = np.cross(tangent, sides)
        ring_normals = (np.cos(angles)[None, :, None] * sides[:, None, :] + np.sin(angles)[None, :, None] * ups[:, None, :])
        ring_points = curve[:, None, :] + radius * ring_normals
        vertices.append(ring_points.reshape(-1, 3))
        normals.append(ring_normals.reshape(-1, 3))
        owner.append(np.full(count * TUBE_SIDES, index))
        segments = count if closed else count - 1
        i = np.arange(segments)[:, None]
        a = np.arange(TUBE_SIDES)[None, :]
        j, b = (i + 1) % count, (a + 1) % TUBE_SIDES
        p, q, r, t = offset + i * TUBE_SIDES + a, offset + i * TUBE_SIDES + b, offset + j * TUBE_SIDES + b, offset + j * TUBE_SIDES + a
        faces.append(np.stack([np.stack([p, q, r], -1), np.stack([p, r, t], -1)], axis=2).reshape(-1, 3))
        offset += count * TUBE_SIDES
    if not vertices:
        return {"vertices": np.zeros((0, 3)), "normals": np.zeros((0, 3)), "faces": np.zeros((0, 3), int), "strand": np.zeros(0, int)}
    return {"vertices": np.vstack(vertices), "normals": np.vstack(normals), "faces": np.vstack(faces).astype(np.int64),
            "strand": np.concatenate(owner)}


def compare_tube(tube: dict, yarn_points_mm: np.ndarray) -> dict:
    if not len(tube["vertices"]):
        return {}
    tube_tree = cKDTree(tube["vertices"])
    scan_to_tube, _ = tube_tree.query(yarn_points_mm, k=1)
    upper = tube["normals"][:, 2] > 0.2
    scan_tree = cKDTree(yarn_points_mm)
    tube_to_scan, _ = scan_tree.query(tube["vertices"], k=1)
    tube["error_mm"] = tube_to_scan
    tube["seen"] = upper
    return {
        "scan_to_rebuilt_mean_mm": float(scan_to_tube.mean()),
        "scan_to_rebuilt_p95_mm": float(np.percentile(scan_to_tube, 95)),
        "rebuilt_to_scan_mean_mm": float(tube_to_scan[upper].mean()) if upper.any() else 0.0,
        "rebuilt_to_scan_p95_mm": float(np.percentile(tube_to_scan[upper], 95)) if upper.any() else 0.0,
        "scan_within_0.2mm": float(np.mean(scan_to_tube <= 0.2)),
    }


def heat_colours(error_mm: np.ndarray, full_scale_mm: float = 0.5) -> np.ndarray:
    """Blue (on the scan) through green and yellow to red (`full_scale_mm` or more off it)."""
    x = np.clip(np.asarray(error_mm) / max(full_scale_mm, 1e-6), 0.0, 1.0)
    stops = np.array([[0.15, 0.35, 0.95], [0.15, 0.80, 0.45], [0.95, 0.85, 0.20], [0.90, 0.20, 0.15]])
    position = x * (len(stops) - 1)
    low = np.floor(position).astype(int).clip(0, len(stops) - 2)
    frac = (position - low)[:, None]
    return stops[low] * (1 - frac) + stops[low + 1] * frac


def tube_as_splat(result: Extraction, colours: np.ndarray, opacity: float = 1.0) -> dict:
    """The rebuilt tubes as a splat-shaped dict (round Gaussians on the tube surface) so the same
    renderer, from the same camera, draws them beside or over the scan."""
    vertices = result.tube["vertices"]
    positions = result.to_splat_frame(vertices).astype(np.float32)
    spacing = (2 * math.pi * (result.summary()["median_radius_mm"] or 0.5) / TUBE_SIDES) / result.mm_per_unit
    count = len(positions)
    sigma = np.full(count, spacing * 0.6, dtype=np.float32)
    return {"positions": positions, "colours": np.asarray(colours, dtype=np.float32), "opacity": np.full(count, opacity, dtype=np.float32),
            "radius": sigma, "sigma": sigma, "count": count}


def save_mesh(result: Extraction, path) -> Path:
    """The rebuilt yarn as a triangle mesh (binary PLY, mm, cloth frame) for any 3D program."""
    path = Path(path)
    vertices = np.asarray(result.tube["vertices"], dtype="<f4")
    faces = np.asarray(result.tube["faces"], dtype="<i4")
    header = (f"ply\nformat binary_little_endian 1.0\ncomment units mm, z up toward the cameras\n"
              f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n"
              f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
    rows = np.zeros(len(faces), dtype=[("n", "u1"), ("i", "<i4", (3,))])
    rows["n"], rows["i"] = 3, faces
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertices.tofile(handle)
        rows.tofile(handle)
    return path


def save(result: Extraction, path) -> Path:
    """The equations as JSON (coefficients in mm, cloth frame and scale included) - enough to rebuild the yarn."""
    path = Path(path)
    payload = {
        "units": "mm",
        "frame": {"origin_in_splat_units": (result.centre / result.mm_per_unit).tolist(), "axes_rows_xyz": result.axes.tolist(),
                  "mm_per_splat_unit": result.mm_per_unit, "z": "up, toward the cameras"},
        "forms": {"open": "p(t) = p0 + (p1 - p0) t + sum_k b_k sin(k pi t), t in [0, 1]",
                  "closed": "p(t) = a0 + sum_k (a_k cos(2 pi k t) + b_k sin(2 pi k t)), t in [0, 1)"},
        "summary": result.summary(),
        "strands": [{"closed": s.closed, "radius_mm": s.radius_mm, "length_mm": s.length_mm, "terms": s.terms, "rms_mm": s.rms_mm,
                     "coefficients": {axis: row.tolist() for axis, row in zip("xyz", s.coefficients)}, "equations": s.equations()}
                    for s in result.strands],
    }
    write_json(path, payload, indent=1)
    return path
