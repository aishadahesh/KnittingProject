"""The scanned lace read as filet crochet (crochet_core.py): which way its rows run, how tall a row and
how wide a stitch is, how many yarn-overs a post has, and which cells are open - then the whole cloth
rebuilt from those numbers.

  1. The repeat comes from the openings (lace_lattice.hole_lattice): how far apart they are along and
     across. Every part of the scan is folded onto one repeat and averaged (lace_lattice.fold) - what
     yarn is where, and how high its top stands above the cloth's bow.
  2. Each way of filling the repeat with crochet is built from the equations and laid on the folded
     scan: the rows along one repeat direction or the other; so many rows, so many stitches; one or
     two yarn-overs. Which stitches are posts and which are chains across an opening is read from the
     folded scan (a stitch's place mostly empty is open). Where it lines up best (a circular cross-correlation finds the
     shift) its score is the match of yarn with yarn and of height with height. The best one is kept.
  3. That repeat, rebuilt over the whole cloth on the cloth's bow, as yarn tubes - compared with the
     scan as the traced strands are.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

import crochet_core
import lace_lattice
import strand_equations

GRID_MM = strand_equations.GRID_MM
ROW_MM = (1.2, 6.5)          # a row of crochet (half double up to double treble) is this tall
STITCH_MM = (0.6, 2.2)       # and its posts this far apart


@dataclass
class CrochetFit:
    params: crochet_core.CrochetParams
    angle_deg: float              # the rows' direction in the cloth frame
    cells: np.ndarray             # the repeat's stitches, rows x stitches, row 0 at the bottom; 1 a post, 0 open
    shift: tuple                  # (along, across) mm: where the repeat's corner sits in the lattice
    score: float
    candidates: list = field(default_factory=list)     # (score, description) of every way tried
    mm_per_unit: float = 1.0
    centre: np.ndarray = field(default_factory=lambda: np.zeros(3))
    axes: np.ndarray = field(default_factory=lambda: np.eye(3))
    yarn_points_mm: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    tile_scan: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    tile_model: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    tube: dict = field(default_factory=dict)
    comparison: dict = field(default_factory=dict)

    @property
    def stitch_name(self) -> str:
        return {1: "double crochet", 2: "treble crochet", 3: "double treble"}.get(self.params.wraps, f"{self.params.wraps} yarn-overs")

    def to_splat_frame(self, points_mm: np.ndarray) -> np.ndarray:
        return (points_mm @ self.axes + self.centre) / self.mm_per_unit

    def summary(self) -> dict:
        return {"median_radius_mm": self.params.yarn_radius, **self.comparison}


def _rasterise(curves, radius: float, period_u: float, period_v: float, shape):
    """The model's yarn on one repeat, seen from above: where it is (0/1) and its top's height (mm)."""
    points = np.vstack([c for c, _ in curves])
    iu = (np.mod(points[:, 0], period_u) / period_u * shape[1]).astype(int) % shape[1]
    iv = (np.mod(points[:, 1], period_v) / period_v * shape[0]).astype(int) % shape[0]
    top = np.full(shape, -np.inf)
    np.maximum.at(top, (iv, iu), points[:, 2])
    marks = np.isfinite(top)
    big = np.tile(marks, (3, 3))
    distance, nearest = ndimage.distance_transform_edt(~big, return_indices=True)
    h, w = shape
    distance = distance[h:2 * h, w:2 * w] * GRID_MM
    near_rows, near_cols = nearest[0][h:2 * h, w:2 * w] % h, nearest[1][h:2 * h, w:2 * w] % w
    presence = distance <= radius
    # The top of a tube of this radius over its centreline.
    height = np.where(presence, top[near_rows, near_cols] + np.sqrt(np.maximum(radius ** 2 - distance ** 2, 0.0)), 0.0)
    return presence.astype(float), height


def _match(model_presence, model_height, scan_presence, scan_relief):
    """How well the model lines up with the folded scan at its best shift: (score, shift rows, shift cols)."""
    def normalised(a):
        a = a - a.mean()
        return a / max(float(np.sqrt((a * a).sum())), 1e-12)

    total = np.zeros(model_presence.shape)
    for model, scan in ((model_presence, scan_presence), (model_height, scan_relief)):
        m, s = normalised(model), normalised(scan)
        total += np.real(np.fft.ifft2(np.fft.fft2(s) * np.conj(np.fft.fft2(m))))
    best = np.unravel_index(int(np.argmax(total)), total.shape)
    return float(total[best] / 2.0), int(best[0]), int(best[1])


def opening_mask(support: np.ndarray) -> np.ndarray:
    """The lace's openings as a mask: the empty patches the yarn closes round, of an opening's size."""
    holes = lace_lattice.lace_holes(support)
    label, count = ndimage.label(holes)
    if not count:
        return holes
    area = ndimage.sum(np.ones_like(label), label, range(1, count + 1)) * GRID_MM ** 2
    return np.r_[False, area >= lace_lattice.MIN_OPENING_MM2][label]


# How much a millimetre of difference between the model's openings and the scan's costs a candidate,
# against its match score (which runs 0..1 and differs little between candidates).
HOLE_SIZE_WEIGHT = 0.08


def hole_size(holes: np.ndarray, angle_deg: float):
    """The typical opening's size (along the rows, across them), mm - the median over openings of about
    the usual area (two run together are left out), each measured edge to edge along the two axes."""
    label, count = ndimage.label(holes)
    if not count:
        return None
    a = math.radians(angle_deg)
    sizes, areas = [], []
    for index in range(1, count + 1):
        rows, cols = np.nonzero(label == index)
        x, y = cols * GRID_MM, rows * GRID_MM
        u, v = x * math.cos(a) + y * math.sin(a), -x * math.sin(a) + y * math.cos(a)
        sizes.append((np.ptp(u) + GRID_MM, np.ptp(v) + GRID_MM))
        areas.append(len(rows) * GRID_MM ** 2)
    areas, sizes = np.array(areas), np.array(sizes)
    usual = (areas > 0.5 * np.median(areas)) & (areas < 2.0 * np.median(areas))
    return tuple(np.median(sizes[usual if usual.any() else slice(None)], axis=0))


def model_hole_size(stitches: np.ndarray, p: crochet_core.CrochetParams):
    """The model's opening (along, across) mm: the longest run of chains in a row, between the posts
    either side of it, and the rows it stays open for, less the braid along each row's top."""
    open_ = np.asarray(stitches) < 0.5
    if not open_.any():
        return 0.0, 0.0
    best_run = max(max((len(r) for r in "".join("1" if o else "0" for o in row).split("0")), default=0) for row in open_)
    rows_open = max(max((len(r) for r in "".join("1" if o else "0" for o in column).split("0")), default=0) for column in open_.T)
    width = (best_run + 1) * p.stitch_width - 4.0 * p.yarn_radius
    height = rows_open * p.row_height - 2.6 * p.yarn_radius
    return width, height


def _candidates(repeat_along: float, repeat_across: float):
    """Every (rows, stitches) split of the repeat into crochet-sized rows and stitches."""
    for rows in range(1, 13):
        if ROW_MM[0] <= repeat_across / rows <= ROW_MM[1]:
            for posts in range(2, 30):
                if STITCH_MM[0] <= repeat_along / posts <= STITCH_MM[1]:
                    yield rows, posts


def fit(extraction, progress=lambda message: None, stitches=None, rows=None, wraps=None, rows_along_holes=None,
        grid=None, refine=True) -> CrochetFit:
    """The scan (a strand_equations.Extraction: its maps, frame and yarn) as crochet lace.

    What someone holding the lace can count settles what the pictures cannot: `stitches` and `rows` per
    repeat, `wraps` (yarn-overs: 1 double crochet, 2 treble) and whether the rows run along the rows of
    openings. Each one left None is searched for, as in the module docstring. `grid` (rows x stitches,
    1 a post) keeps a stitch grid drawn by hand instead of reading the openings from the scan; `refine`
    also fits the yarn's thickness, the posts' twist and their lean (see refine_shape)."""
    if grid is not None:
        grid = np.asarray(grid, dtype=float)
        rows, stitches = grid.shape
    maps = extraction.maps
    support, height = maps["support"], maps["height"]
    low = extraction.map_origin
    progress("Finding the repeat from the openings")
    lattice = lace_lattice.hole_lattice(support)
    if lattice is None:
        raise RuntimeError("no regular openings found - a filet repeat needs a few open cells in view")
    hole_angle, repeat_u, repeat_v = lattice
    cloth = lace_lattice.cloth_region(support)
    bow = lace_lattice.bow_map(support, height)
    radius = float(np.clip(extraction.summary()["median_radius_mm"] or 0.3, 0.15, 0.6))
    holes = opening_mask(support)
    measured = {}
    best, tried = None, []
    orientations = [(hole_angle, repeat_u, repeat_v), ((hole_angle + 90.0) % 180.0, repeat_v, repeat_u)]
    if rows_along_holes is not None:
        orientations = orientations[:1] if rows_along_holes else orientations[1:]
    for angle, along, across in orientations:
        progress(f"Folding the scan with the rows at {angle:.0f} deg")
        u, v = lace_lattice._lattice_coords(support.shape, low, angle)
        # Where yarn is, as the openings say: a stitch's place is open where the folded openings cover
        # it. The splat's own small gaps between plies would read as open everywhere.
        presence = 1.0 - lace_lattice.fold(holes.astype(float), cloth, u, v, along, across)
        relief = lace_lattice.fold(height - bow, cloth & support, u, v, along, across)
        measured[angle] = hole_size(holes, angle)
        splits = list(_candidates(along, across))
        if stitches is not None or rows is not None:
            splits = [(r, s) for r, s in splits if (rows is None or r == rows) and (stitches is None or s == stitches)]
            if not splits:
                splits = [(rows or 3, stitches or 3)]
        for n_rows, posts in splits:
            if grid is not None:
                cells = grid
            else:
                cells, _, _ = lace_lattice.bitmap_from_tile(presence, n_rows, posts, open_below=0.5)
            for n_wraps in ((wraps,) if wraps is not None else (1, 2, 3)):
                params = crochet_core.CrochetParams(stitch_width=along / posts, row_height=across / n_rows, wraps=n_wraps,
                                                    yarn_radius=radius)
                progress(f"Trying {n_rows} rows x {posts} stitches, {n_wraps} yarn-over{'s' if n_wraps > 1 else ''}, rows at {angle:.0f} deg")
                curves = crochet_core.build(np.tile(cells, (1, 1)), params)
                model_presence, model_height = _rasterise(curves, radius, along, across, presence.shape)
                score, shift_rows, shift_cols = _match(model_presence, model_height, presence, relief)
                if measured[angle] is not None:
                    model_along, model_across = model_hole_size(cells, params)
                    score -= HOLE_SIZE_WEIGHT * (abs(model_along - measured[angle][0]) + abs(model_across - measured[angle][1]))
                description = f"rows at {angle:.0f} deg, {n_rows} rows x {posts} stitches per repeat, {n_wraps} yarn-over(s)"
                tried.append((score, description))
                if best is None or score > best[0]:
                    shift = (shift_cols * GRID_MM, shift_rows * GRID_MM)
                    best = (score, params, angle, cells, shift, presence, model_presence, (u, v, along, across), relief)
    score, params, angle, cells, shift, presence, model_presence, (u, v, along, across), relief = best
    if refine:
        progress("Fitting the yarn's thickness, the posts' twist and lean")
        params, score, shift, model_presence = refine_shape(cells, params, along, across, presence, relief)
    result = CrochetFit(params, angle, cells, shift, score, sorted(tried, reverse=True), extraction.mm_per_unit, extraction.centre,
                        extraction.axes, extraction.yarn_points_mm, presence, model_presence)
    progress("Building the whole cloth from the repeat")
    build_cloth(result, cloth, u, v, low, bow, along, across)
    progress("Comparing it with the scan")
    result.comparison = strand_equations.compare_tube(result.tube, result.yarn_points_mm)
    return result


# The shape numbers refine_shape tries, around what the counts and the traced yarn gave.
RADIUS_STEPS = np.round(np.arange(0.6, 1.81, 0.2), 2)       # times the traced yarn radius
TWIST_STEPS = (0.25, 0.5, 0.75, 1.0, 1.5)
LEAN_STEPS = np.round(np.arange(-0.4, 0.41, 0.2), 2)        # times the stitch width


def refine_shape(cells, params, along: float, across: float, presence, relief):
    """The yarn's thickness, then the posts' twist, then their lean: each the value whose model lines up
    best with the folded scan, the others held. Returns (params, score, shift mm, model presence)."""
    best = None

    def trial(**changes):
        nonlocal best
        trial_params = crochet_core.CrochetParams(**{**params.as_dict(), **changes})
        curves = crochet_core.build(cells, trial_params)
        model_presence, model_height = _rasterise(curves, trial_params.yarn_radius, along, across, presence.shape)
        score, shift_rows, shift_cols = _match(model_presence, model_height, presence, relief)
        if best is None or score > best[1]:
            best = (trial_params, score, (shift_cols * GRID_MM, shift_rows * GRID_MM), model_presence)

    for factor in RADIUS_STEPS:
        trial(yarn_radius=float(np.clip(params.yarn_radius * factor, 0.08, 1.0)))
    params = best[0]
    for twist in TWIST_STEPS:
        trial(yarn_radius=params.yarn_radius, twist=twist)
    params = best[0]
    for lean in LEAN_STEPS * params.stitch_width:
        trial(yarn_radius=params.yarn_radius, twist=params.twist, lean=float(lean))
    return best


def build_cloth(result: CrochetFit, cloth, u, v, low, bow, along: float, across: float) -> None:
    """The repeat laid over the whole cloth, on its bow, cut to it, as yarn tubes."""
    inside_u, inside_v = u[cloth] - result.shift[0], v[cloth] - result.shift[1]
    first_u, first_v = math.floor(inside_u.min() / along) - 1, math.floor(inside_v.min() / across) - 1
    tiles_u = int(math.ceil(inside_u.max() / along)) - first_u + 1
    tiles_v = int(math.ceil(inside_v.max() / across)) - first_v + 1
    curves = crochet_core.build(np.tile(result.cells, (tiles_v, tiles_u)), result.params)
    a = math.radians(result.angle_deg)
    near_cloth = ndimage.binary_dilation(cloth, iterations=5)
    tubes = []
    for points, closed in curves:
        lu = points[:, 0] + first_u * along + result.shift[0]
        lv = points[:, 1] + first_v * across + result.shift[1]
        x, y = lu * math.cos(a) - lv * math.sin(a), lu * math.sin(a) + lv * math.cos(a)
        col = np.clip(((x - low[0]) / GRID_MM).astype(int), 0, cloth.shape[1] - 1)
        row = np.clip(((y - low[1]) / GRID_MM).astype(int), 0, cloth.shape[0] - 1)
        if not near_cloth[row, col].all():
            continue                         # a stitch off the cloth's edge is left out whole
        tubes.append((np.stack([x, y, bow[row, col] + points[:, 2]], axis=1), result.params.yarn_radius, closed))
    result.tube = strand_equations.polyline_tubes(tubes)
