"""A scanned cloth as Edit Mode's repeating knit: complete loops, joined to each other, that can be edited.

The traced strands (strand_equations.py) are only what the pictures showed clearly - pieces. But a
knitted or crocheted cloth repeats: the same loop, row after row, the same openings every few stitches.
This finds that repeat in the scan and fills in the rest from it, as the model Edit Mode builds
(knitting_core.build_parametric_control_rows): each stitch the loop

    x = (col + bulge sin 2t + t / 2pi) * stitch_width
    y = row * dy - loop_height (cos t - 1) / 2
    z = stitch_z (cos 2t - 1) / 2

joined along the row into one periodic curve, rows stacked dy apart, and a bitmap tile of which
stitches are worked (a cell left out is an opening; the stitch below reaches up past it).

  1. The repeat: the brightness map's spectrum. Its strongest long wavelength in two directions is the
     pattern's repeat (hole to hole); the strongest short wavelengths along those directions are one
     stitch and one row. The axis with the wider short one runs along the rows.
  2. The tile: every part of the scan folded onto one repeat and averaged - noise and the patches the
     pictures missed average out - then cut into rows x stitches; a cell mostly empty is an opening.
  3. The loop's shape (height, bulge) and its exact place in the tile: the candidates' yarn laid on the
     folded tile, the best match kept; the depth (stitch_z) from the folded height relief.
  4. The complete model: the tile repeated over the whole cloth, laid along the cloth's own bow, as
     yarn tubes - compared with the scan the same way the traced strands are.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

import knitting_core
import strand_equations

GRID_MM = strand_equations.GRID_MM
# The pattern's repeat (hole to hole) lies in this range of wavelengths, mm; one stitch or row in the next.
REPEAT_MM = (5.0, 25.0)
MIN_OPENING_MM2 = 2.0
STITCH_MM = (1.2, 5.0)
# How far a short wavelength may point from its axis and still count as a stitch along it, degrees.
AXIS_TOLERANCE_DEG = 25.0
SAMPLES_PER_LOOP = 5            # as Edit Mode builds it
CENTRELINE_PER_STITCH = 32      # knitting_core: loop_res samples per stitch
# Loop height tried, as multiples of the row pitch, and bulge tried.
HEIGHT_STEPS = np.round(np.arange(1.0, 2.81, 0.15), 3)
BULGE_STEPS = np.round(np.arange(0.0, 0.61, 0.1), 3)
NAMES = ("stitch_bulge", "stitch_z", "dy", "stitch_width", "loop_height_1")


@dataclass
class PeriodicFit:
    angle_deg: float              # the rows' direction in the cloth frame (x axis = 0)
    stitch_width: float           # mm
    row_pitch: float              # mm (Edit Mode's dy)
    bitmap: np.ndarray            # rows x stitches, row 0 at the bottom; 1 = stitch worked, 0 = opening
    loop_height: float            # mm
    stitch_bulge: float
    stitch_z: float               # mm
    radius: float                 # mm
    origin: np.ndarray            # cloth-frame (x, y) mm of the tile's corner
    score: float                  # how well the yarn lies on the folded scan (correlation, 0..1)
    mm_per_unit: float = 1.0
    centre: np.ndarray = field(default_factory=lambda: np.zeros(3))
    axes: np.ndarray = field(default_factory=lambda: np.eye(3))
    yarn_points_mm: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    tile_scan: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    tile_model: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    tube: dict = field(default_factory=dict)
    comparison: dict = field(default_factory=dict)
    rows_mm: list = field(default_factory=list)     # the complete model's row centrelines, cloth frame

    @property
    def n_rows(self) -> int:
        return int(self.bitmap.shape[0])

    @property
    def n_cols(self) -> int:
        return int(self.bitmap.shape[1])

    def to_splat_frame(self, points_mm: np.ndarray) -> np.ndarray:
        return (points_mm @ self.axes + self.centre) / self.mm_per_unit

    def summary(self) -> dict:
        return {"median_radius_mm": self.radius, **self.comparison}

    def edit_parameters(self) -> dict:
        """Edit Mode's parameters (mm), by name."""
        values = {"stitch_width": self.stitch_width, "dy": self.row_pitch, "stitch_bulge": self.stitch_bulge,
                  "stitch_z": self.stitch_z, "radius": self.radius}
        values.update({f"loop_height_{row + 1}": self.loop_height for row in range(self.n_rows)})
        return values

    def equations(self) -> list[str]:
        """The loop as Edit Mode builds it, with the fitted numbers in (mm, t from 0 to 2 pi per stitch)."""
        return [f"x(t) = (col + {self.stitch_bulge:.2f} sin 2t + t/2π) × {self.stitch_width:.2f}",
                f"y(t) = row × {self.row_pitch:.2f} − {self.loop_height:.2f} (cos t − 1)/2",
                f"z(t) = {self.stitch_z:+.2f} (cos 2t − 1)/2"]


# ----------------------------------------------------------------------------------------- 1. repeat

def _power_spectrum(brightness: np.ndarray, support: np.ndarray):
    """The brightness map's power spectrum (lightly smoothed) and its frequency grids, cycles per mm."""
    values = np.where(support, brightness - brightness[support].mean(), 0.0)
    window = np.outer(np.hanning(values.shape[0]), np.hanning(values.shape[1]))
    power = np.abs(np.fft.fftshift(np.fft.fft2(values * window))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(values.shape[0], GRID_MM))
    fx = np.fft.fftshift(np.fft.fftfreq(values.shape[1], GRID_MM))
    FX, FY = np.meshgrid(fx, fy)
    return ndimage.gaussian_filter(power, 1.5), FX, FY


def stitches_in_repeat(spectrum, direction_deg: float, repeat: float) -> list[int]:
    """How many stitches (or rows) one repeat may hold along a direction: the whole numbers that give a
    stitch-sized step, strongest in the spectrum first. The strongest is not always the stitch - a
    knitted stitch's two legs make its half as strong - so the loop fit decides among the first few."""
    power, FX, FY = spectrum
    unit = np.array([math.cos(math.radians(direction_deg)), math.sin(math.radians(direction_deg))])
    step_x, step_y = FX[0, 1] - FX[0, 0], FY[1, 0] - FY[0, 0]
    ranked = []
    for count in range(1, 13):
        pitch = repeat / count
        if not STITCH_MM[0] <= pitch <= STITCH_MM[1]:
            continue
        fx, fy = unit * count / repeat
        if fy < 0:
            fx, fy = -fx, -fy
        col = int(round((fx - FX[0, 0]) / step_x))
        row = int(round((fy - FY[0, 0]) / step_y))
        near = power[max(row - 1, 0):row + 2, max(col - 1, 0):col + 2]
        ranked.append((float(near.max()) if near.size else 0.0, count))
    return [count for _, count in sorted(ranked, reverse=True)] or [max(1, int(round(repeat / 3.0)))]


def _spectrum_peaks(brightness: np.ndarray, support: np.ndarray, limit: int = 40):
    """The brightness map's strongest spatial wavelengths: (wavelength mm, direction deg in [0, 180), power)."""
    power, FX, FY = _power_spectrum(brightness, support)
    radius = np.hypot(FX, FY)
    band = (radius > 1.0 / REPEAT_MM[1]) & (radius < 1.0 / STITCH_MM[0]) & (FY >= 0)
    smooth = power * band
    peaks = (smooth == ndimage.maximum_filter(smooth, 9)) & band & (smooth > 0)
    ys, xs = np.nonzero(peaks)
    order = np.argsort(smooth[ys, xs])[::-1][:limit]
    top = smooth[ys, xs].max() if len(ys) else 1.0
    return [(1.0 / radius[ys[k], xs[k]], math.degrees(math.atan2(FY[ys[k], xs[k]], FX[ys[k], xs[k]])) % 180.0,
             smooth[ys[k], xs[k]] / top) for k in order]


def _angle_gap(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def cloth_region(support: np.ndarray) -> np.ndarray:
    """The cloth on the yarn map: where yarn is, with its openings, but not the empty table round it
    or the ragged edge."""
    piece = lace_piece(support)        # the lace alone: a printed board beside it is yarn-white too
    cloth = ndimage.binary_erosion(ndimage.binary_fill_holes(ndimage.binary_closing(piece, iterations=4)), iterations=10)
    return cloth if cloth.sum() >= 100 else support


def bow_map(support: np.ndarray, height: np.ndarray) -> np.ndarray:
    """The cloth's slow bow on the yarn map (mm): the yarn's top height averaged over 3 mm."""
    return ndimage.gaussian_filter(np.where(support, height, 0.0), 30) / np.maximum(ndimage.gaussian_filter(support.astype(float), 30), 1e-9)


def lace_piece(support: np.ndarray) -> np.ndarray:
    """The yarn map's largest connected piece, its gaps between plies closed: the lace itself, not a
    printed board beside it (whose white squares read as yarn, its black ones as openings)."""
    yarn = ndimage.binary_closing(support, iterations=3)
    label, count = ndimage.label(yarn)
    if count <= 1:
        return yarn
    sizes = ndimage.sum(np.ones_like(label), label, range(1, count + 1))
    return label == int(np.argmax(sizes)) + 1


def lace_holes(support: np.ndarray) -> np.ndarray:
    """The empty patches the lace's yarn closes round (its openings, not the gaps between plies)."""
    yarn = lace_piece(support)
    return ndimage.binary_opening(ndimage.binary_fill_holes(yarn) & ~yarn, iterations=3)


def openings(support: np.ndarray):
    """The cloth's openings: (centres mm in map coordinates from the map's corner, areas mm^2) of the
    empty patches the yarn closes round, the lace's holes - not the gaps between plies."""
    holes = lace_holes(support)
    label, count = ndimage.label(holes)
    if not count:
        return np.zeros((0, 2)), np.zeros(0)
    area = ndimage.sum(np.ones_like(label), label, range(1, count + 1)) * GRID_MM ** 2
    centres = np.array(ndimage.center_of_mass(holes, label, range(1, count + 1)))[:, ::-1] * GRID_MM
    return centres, area


def hole_lattice(support: np.ndarray):
    """The repeat from the openings' centres (step 1): (row direction deg, repeat along the rows mm,
    repeat across them mm), or None when there are too few openings. The two shortest common steps
    between openings are the lattice; when the next row of openings sits half a repeat along, the
    rectangular repeat Edit Mode tiles holds two rows of them."""
    centres, area = openings(support)
    if len(area) < 4:
        return None
    # Openings of the pattern, not the small gaps every loop leaves.
    big = centres[(area >= 0.4 * np.median(np.sort(area)[::-1][:10])) & (area >= MIN_OPENING_MM2)]
    if len(big) < 4:
        return None
    steps = (big[:, None] - big[None]).reshape(-1, 2)
    length = np.linalg.norm(steps, axis=1)
    steps = steps[(length >= REPEAT_MM[0]) & (length <= REPEAT_MM[1])]
    if len(steps) < 4:
        return None
    steps = steps * np.where(steps[:, :1] < 0, -1.0, 1.0)       # one of each +/- pair
    support_count = np.array([(np.linalg.norm(steps - s, axis=1) < 1.5).sum() for s in steps])

    def cluster(candidates):
        if not candidates.any():
            return None
        # Among the well-supported steps, the shortest (a multiple of the repeat is supported too).
        strong = candidates & (support_count >= 0.6 * support_count[candidates].max())
        shortest = steps[strong][np.argmin(np.linalg.norm(steps[strong], axis=1))]
        return steps[np.linalg.norm(steps - shortest, axis=1) < 1.5].mean(axis=0)

    along = cluster(np.ones(len(steps), dtype=bool))
    if along is None:
        return None
    direction = along / np.linalg.norm(along)
    repeat_u = float(np.linalg.norm(along))
    # Rows of openings: the openings grouped by how far across the rows they lie.
    across_of = big @ np.array([-direction[1], direction[0]])
    order = np.argsort(across_of)
    groups = np.split(order, np.flatnonzero(np.diff(across_of[order]) > 0.3 * repeat_u) + 1)
    # A row of the pattern's openings is as long as the longest; a pair of smaller openings between
    # them is part of the pattern's detail, not a row of its own.
    longest = max(len(g) for g in groups)
    groups = [g for g in groups if len(g) >= max(2, 0.5 * longest)]
    if len(groups) < 2:
        return None
    levels = np.array([np.median(across_of[g]) for g in groups])
    repeat_v = float(np.median(np.diff(levels)))
    # Where along the row each row of openings sits (a circular mean over the repeat), and how far
    # one row is slid from the next. Half a repeat: the rectangular tile holds two rows of openings.
    phase = np.array([np.angle(np.exp(2j * np.pi * (big[g] @ direction) / repeat_u).mean()) / (2 * np.pi) for g in groups])
    slide = float(np.median(np.abs((np.diff(phase) + 0.5) % 1.0 - 0.5)))
    if 0.35 < slide <= 0.5:
        repeat_v *= 2.0
    return math.degrees(math.atan2(direction[1], direction[0])) % 180.0, repeat_u, repeat_v


def fold_sharpness(values: np.ndarray, coordinate: np.ndarray, period: float, bins: int = 48) -> float:
    """How sharp `values` come out folded onto `period` along `coordinate`: the share of their variance
    the folded average keeps. A true period keeps the pattern; a wrong one averages it away."""
    phase = (np.mod(coordinate, period) / period * bins).astype(int) % bins
    count = np.bincount(phase, minlength=bins)
    mean = np.bincount(phase, weights=values, minlength=bins) / np.maximum(count, 1)
    centred = mean - values.mean()
    return float((count * centred ** 2).sum() / max(len(values) * values.var(), 1e-12))


def best_period(values: np.ndarray, coordinate: np.ndarray, low: float, high: float) -> float:
    """The period in [low, high] mm the values fold onto most sharply - the smallest that does, since
    folding onto two or three periods is sharp too."""
    periods = np.arange(low, high, 0.01)
    sharpness = np.array([fold_sharpness(values, coordinate, p) for p in periods])
    smooth = ndimage.uniform_filter1d(sharpness, 3)
    peaks = np.flatnonzero((smooth >= np.roll(smooth, 1)) & (smooth >= np.roll(smooth, -1)) & (smooth >= 0.8 * smooth.max()))
    return float(periods[peaks[0]] if len(peaks) else periods[int(np.argmax(smooth))])


def _reduce(a: np.ndarray, b: np.ndarray):
    """Gauss's reduction: the same lattice from its shortest, most nearly square pair of steps."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    while True:
        if b @ b < a @ a:
            a, b = b, a
        k = round(float(a @ b) / float(a @ a))
        if k == 0:
            return a, b
        b = b - k * a


def _refine_period(values: np.ndarray, coordinate: np.ndarray, guess: float) -> float:
    """The period within 15% of `guess` the values fold onto most sharply."""
    return float(max(np.arange(0.85 * guess, 1.15 * guess, 0.005), key=lambda p: fold_sharpness(values, coordinate, p)))


def find_lattice(brightness: np.ndarray, support: np.ndarray):
    """The repeat (step 1): (row direction deg, repeat along the rows mm, repeat across mm, candidate
    stitches per repeat, candidate rows per repeat), the likeliest candidates first.

    The rows' direction and the pattern's repeat come from the openings; a plain cloth without any
    takes them from its stitch lattice, three stitches and three rows making its repeat."""
    angle, repeats = _repeat_and_direction(brightness, support)
    if repeats != (None, None):
        spectrum = _power_spectrum(brightness, support)
        return (angle, repeats[0], repeats[1], stitches_in_repeat(spectrum, angle, repeats[0])[:3],
                stitches_in_repeat(spectrum, angle + 90.0, repeats[1])[:3])
    region = support & ndimage.binary_erosion(ndimage.binary_fill_holes(support), iterations=10)
    if region.sum() < 1000:
        region = support
    rows, cols = np.nonzero(region)
    values = brightness[region]
    x, y = (cols + 0.5) * GRID_MM, (rows + 0.5) * GRID_MM
    # A plain cloth: the stitch lattice from its two strongest stitch wavelengths, which may be
    # diagonals of it - the lattice they make is reduced to its shortest, squarest pair of steps.
    first, second = angle
    steps = np.linalg.inv(np.array([first, second])).T          # real steps: a_i . g_j = delta_ij
    a1, a2 = _reduce(steps[0], steps[1])
    across_a1 = float(abs(a1[0] * a2[1] - a1[1] * a2[0]) / np.linalg.norm(a1))
    angle = math.degrees(math.atan2(a1[1], a1[0])) % 180.0
    r = math.radians(angle)
    u, v = x * math.cos(r) + y * math.sin(r), -x * math.sin(r) + y * math.cos(r)
    stitch_width, row_pitch = _refine_period(values, u, float(np.linalg.norm(a1))), _refine_period(values, v, across_a1)
    # Along the rows: the axis with the wider stitch (a stitch is wider than a row is tall).
    if row_pitch > stitch_width:
        angle, stitch_width, row_pitch = (angle + 90.0) % 180.0, row_pitch, stitch_width
    return angle, 3 * stitch_width, 3 * row_pitch, [3], [3]


def _repeat_and_direction(brightness: np.ndarray, support: np.ndarray):
    """(direction deg, (repeat along it mm or None, repeat across mm or None)): from the openings when
    there are enough, else from the spectrum's long wavelengths, else just the strongest stitch direction."""
    from_holes = hole_lattice(support)
    if from_holes is not None:
        angle, repeat_u, repeat_v = from_holes
        return angle, (repeat_u, repeat_v)
    # No openings: a plain cloth. Its long wavelengths are the patch's shading, not a repeat; the
    # stitches' own wavelength gives the direction, refined in find_lattice.
    shorts = [p for p in _spectrum_peaks(brightness, support) if STITCH_MM[0] <= p[0] < REPEAT_MM[0]]
    if not shorts:
        raise RuntimeError("no stitch-sized repeat in the scan - is the model's scale right?")
    first = shorts[0]
    second = next((p for p in shorts[1:] if _angle_gap(p[1], first[1]) >= 30.0), None)
    if second is None:
        second = (first[0], (first[1] + 90.0) % 180.0, 0.0)

    def frequency(peak):
        return np.array([math.cos(math.radians(peak[1])), math.sin(math.radians(peak[1]))]) / peak[0]

    return (frequency(first), frequency(second)), (None, None)


# ----------------------------------------------------------------------------------------- 2. the tile

def _lattice_coords(shape, low, angle_deg):
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    x = low[0] + (cols + 0.5) * GRID_MM
    y = low[1] + (rows + 0.5) * GRID_MM
    a = math.radians(angle_deg)
    return x * math.cos(a) + y * math.sin(a), -x * math.sin(a) + y * math.cos(a)


def fold(values: np.ndarray, region: np.ndarray, u: np.ndarray, v: np.ndarray, period_u: float, period_v: float):
    """`values` averaged onto one repeat (pixels GRID_MM), over the pixels in `region`."""
    shape = (max(int(round(period_v / GRID_MM)), 1), max(int(round(period_u / GRID_MM)), 1))
    iu = (np.mod(u[region], period_u) / period_u * shape[1]).astype(int) % shape[1]
    iv = (np.mod(v[region], period_v) / period_v * shape[0]).astype(int) % shape[0]
    total, count = np.zeros(shape), np.zeros(shape)
    np.add.at(total, (iv, iu), values[region])
    np.add.at(count, (iv, iu), 1.0)
    # A tile pixel no scan pixel landed on borrows its neighbours' average.
    mean = ndimage.gaussian_filter(total, 1.0, mode="wrap") / np.maximum(ndimage.gaussian_filter(count, 1.0, mode="wrap"), 1e-9)
    return np.where(count > 0, total / np.maximum(count, 1e-9), mean)


def cell_occupancy(tile: np.ndarray, n_rows: int, n_cols: int, shift_rows: float, shift_cols: float) -> np.ndarray:
    """How much yarn each cell holds, the cell grid shifted by a fraction of a cell."""
    h, w = tile.shape
    rolled = np.roll(tile, (-int(round(shift_rows * h / n_rows)), -int(round(shift_cols * w / n_cols))), axis=(0, 1))
    rows = np.minimum((np.arange(h) * n_rows) // h, n_rows - 1)
    cols = np.minimum((np.arange(w) * n_cols) // w, n_cols - 1)
    occupancy = np.zeros((n_rows, n_cols))
    np.add.at(occupancy, (rows[:, None].repeat(w, 1), cols[None, :].repeat(h, 0)), rolled)
    return occupancy / ((h / n_rows) * (w / n_cols))


def bitmap_from_tile(tile: np.ndarray, n_rows: int, n_cols: int, open_below: float | None = None):
    """The openings: the cell grid placed where it splits the tile into the most clearly full and empty
    cells, then a cell is an opening when it holds clearly less yarn than the full ones. The grid's row 0
    is put where no opening is (Edit Mode's bottom row is always worked). Returns (bitmap, row shift, col shift)."""
    best = None
    for shift_rows in np.arange(0.0, 1.0, 0.25):
        for shift_cols in np.arange(0.0, 1.0, 0.25):
            occupancy = cell_occupancy(tile, n_rows, n_cols, shift_rows, shift_cols)
            contrast = float(occupancy.std())
            if best is None or contrast > best[0]:
                best = (contrast, occupancy, shift_rows, shift_cols)
    _, occupancy, shift_rows, shift_cols = best
    full = float(np.percentile(occupancy, 75))
    if open_below is not None:          # the tile is already a 0..1 share of yarn: a fixed cut
        bitmap = (occupancy >= open_below).astype(np.float32)
    else:
        bitmap = (occupancy >= 0.6 * full).astype(np.float32) if np.ptp(occupancy) > 0.2 * full else np.ones_like(occupancy, np.float32)
    # Row 0 where the rows are fullest.
    start = int(np.argmax([bitmap[(r + np.arange(n_rows)) % n_rows][0].sum() for r in range(n_rows)]))
    bitmap = np.roll(bitmap, -start, axis=0)
    bitmap, _ = knitting_core.anchor_bitmap(bitmap)
    return bitmap, shift_rows + start, shift_cols


# ----------------------------------------------------------------------------------------- 3. shape

def control_rows(bitmap, stitch_width, row_pitch, loop_height, bulge, stitch_z=1.0):
    """Edit Mode's own control rows for these numbers (knitting_core)."""
    params = np.array([bulge, stitch_z, row_pitch, stitch_width, loop_height], dtype=np.float32)
    return knitting_core.build_parametric_control_rows(params, bitmap, {n: i for i, n in enumerate(NAMES)}, [4], SAMPLES_PER_LOOP)


def row_centrelines(rows, period_x: float, n_cols: int):
    """Each row's smooth periodic centreline, as Edit Mode evaluates it."""
    period = np.array([period_x, 0.0, 0.0])
    count = CENTRELINE_PER_STITCH * n_cols + 1
    lines = []
    for row in rows:
        row = np.asarray(row, dtype=float)
        # A row knitted right to left lists its points that way; closed onto the next period to the
        # right, its spline ran straight back across the whole row. Read left to right it is the same loops.
        if len(row) > 1 and row[0, 0] > row[-1, 0]:
            row = row[::-1]
        lines.append(knitting_core.eval_centerline(row, period, count))
    return lines


def _model_tile(bitmap, stitch_width, row_pitch, loop_height, bulge, radius, shape):
    """Where the model's yarn lies on one repeat (a 0/1 tile of `shape`)."""
    n_rows, n_cols = bitmap.shape
    period_u, period_v = n_cols * stitch_width, n_rows * row_pitch
    lines = row_centrelines(control_rows(bitmap, stitch_width, row_pitch, loop_height, bulge), period_u, n_cols)
    points = np.vstack(lines)
    iu = (np.mod(points[:, 0], period_u) / period_u * shape[1]).astype(int) % shape[1]
    iv = (np.mod(points[:, 1], period_v) / period_v * shape[0]).astype(int) % shape[0]
    marks = np.zeros(shape, dtype=bool)
    marks[iv, iu] = True
    # Yarn within a radius of the centreline, the tile seen as wrapping round on itself.
    big = np.tile(marks, (3, 3))
    near = ndimage.distance_transform_edt(~big)[shape[0]:2 * shape[0], shape[1]:2 * shape[1]] * GRID_MM <= radius
    return near.astype(float)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-12))


def fit_shape(tile, bitmap, stitch_width, row_pitch, radius):
    """Loop height, bulge and the cell grid's exact place (step 3): the best match of the model's yarn
    with the folded scan. Returns (height, bulge, shift_u mm, shift_v mm, score, model tile)."""
    best = None

    def score(height, bulge, du, dv):
        model = _model_tile(bitmap, stitch_width, row_pitch, height, bulge, radius, tile.shape)
        model = np.roll(model, (int(round(dv / GRID_MM)), int(round(du / GRID_MM))), axis=(0, 1))
        return _correlation(model, tile), model

    for height in HEIGHT_STEPS * row_pitch:
        for bulge in BULGE_STEPS:
            value, model = score(height, bulge, 0.0, 0.0)
            if best is None or value > best[4]:
                best = (height, bulge, 0.0, 0.0, value, model)
    height, bulge = best[0], best[1]
    for du in np.linspace(-0.5, 0.5, 7) * stitch_width:
        for dv in np.linspace(-0.5, 0.5, 7) * row_pitch:
            value, model = score(height, bulge, du, dv)
            if value > best[4]:
                best = (height, bulge, du, dv, value, model)
    height, bulge, du, dv = best[:4]
    for h in height + np.linspace(-0.1, 0.1, 5) * row_pitch:
        for b in np.clip(bulge + np.linspace(-0.05, 0.05, 3), 0.0, 1.0):
            value, model = score(h, b, du, dv)
            if value > best[4]:
                best = (h, b, du, dv, value, model)
    return best


def fit_depth(relief_tile, bitmap, stitch_width, row_pitch, loop_height, bulge, shift_u, shift_v) -> float:
    """stitch_z: the height relief along the loops against the loop's own (cos 2t - 1)/2 shape."""
    n_rows, n_cols = bitmap.shape
    period_u, period_v = n_cols * stitch_width, n_rows * row_pitch
    lines = row_centrelines(control_rows(bitmap, stitch_width, row_pitch, loop_height, bulge, stitch_z=1.0), period_u, n_cols)
    points = np.vstack(lines)
    shape = relief_tile.shape
    iu = (np.mod(points[:, 0] + shift_u, period_u) / period_u * shape[1]).astype(int) % shape[1]
    iv = (np.mod(points[:, 1] + shift_v, period_v) / period_v * shape[0]).astype(int) % shape[0]
    measured = relief_tile[iv, iu]
    shape_z = points[:, 2]
    shape_z = shape_z - shape_z.mean()
    if float((shape_z ** 2).sum()) < 1e-9:
        return 0.0
    return float(np.clip((shape_z * (measured - measured.mean())).sum() / (shape_z ** 2).sum(), -1.0, 1.0))


# ----------------------------------------------------------------------------------------- 4. model

# Knitting's proportions, used to split the repeat into stitches and rows when the spectrum leaves a
# choice (its strongest wavelength along a row is often half a stitch - a stitch has two legs): a
# stitch at least this many yarn radii wide, and rows about this share of a stitch apart.
MIN_STITCH_RADII = 5.0
ROW_TO_STITCH = 0.7


def choose_counts(repeat_u: float, repeat_v: float, col_counts, row_counts, radius: float):
    """(stitches, rows) per repeat: of the spectrum's candidates (strongest first), the ones wide
    enough for two legs of yarn; then the pair nearest knitting's row-to-stitch proportion, a weaker
    candidate paying a little for its rank so the proportion only decides between near-equals."""
    cols = [int(np.clip(c, 1, 32)) for c in col_counts]
    rows = [int(np.clip(r, 1, 8)) for r in row_counts]
    wide = [(rank, c) for rank, c in enumerate(cols) if repeat_u / c >= MIN_STITCH_RADII * radius] or [(0, min(cols))]

    def cost(choice):
        (col_rank, c), (row_rank, r) = choice
        proportion = (repeat_v / r) / (repeat_u / c)
        # Along a row the spectrum is the surer (a row of stitches is one clear wave); across, rows
        # blur into each other, so their ranking counts for less.
        return abs(proportion - ROW_TO_STITCH) / ROW_TO_STITCH + 0.25 * col_rank + 0.1 * row_rank

    (_, n_cols), (_, n_rows) = min(((c, r) for c in wide for r in enumerate(rows)), key=cost)
    return n_cols, n_rows


def fit(extraction, counts=None, progress=lambda message: None) -> PeriodicFit:
    """The scan (a strand_equations.Extraction - its maps, frame and yarn) as a repeating knit.
    `counts` (stitches, rows) per repeat fixes the split of the repeat when the automatic one is wrong."""
    maps = extraction.maps
    support, brightness, height = maps["support"], maps["brightness"], maps["height"]
    low = extraction.map_origin
    progress("Finding the repeat")
    angle, repeat_u, repeat_v, col_counts, row_counts = find_lattice(brightness, support)
    if counts is not None:
        col_counts, row_counts = [int(counts[0])], [int(counts[1])]
    u, v = _lattice_coords(support.shape, low, angle)
    cloth = cloth_region(support)
    progress("Folding the scan onto one repeat")
    tile = fold(support.astype(float), cloth, u, v, repeat_u, repeat_v)
    bow = bow_map(support, height)
    relief_tile = fold(height - bow, cloth & support, u, v, repeat_u, repeat_v)
    radius = float(np.clip(extraction.summary()["median_radius_mm"] or 0.3, 0.05, 0.5))
    n_cols, n_rows = choose_counts(repeat_u, repeat_v, col_counts, row_counts, radius)
    stitch_width, row_pitch = repeat_u / n_cols, repeat_v / n_rows
    bitmap, shift_rows, shift_cols = bitmap_from_tile(tile, n_rows, n_cols)
    progress(f"Fitting the loops: {n_rows} rows x {n_cols} stitches per repeat")
    # The bitmap was cut with the cell grid shifted; the tile is rolled the same way before the shape fit.
    roll = (-int(round(shift_rows * tile.shape[0] / n_rows)), -int(round(shift_cols * tile.shape[1] / n_cols)))
    shifted = np.roll(tile, roll, axis=(0, 1))
    loop_height, bulge, du, dv, score, model_tile = fit_shape(shifted, bitmap, stitch_width, row_pitch, radius)
    stitch_z = fit_depth(np.roll(relief_tile, roll, axis=(0, 1)), bitmap, stitch_width, row_pitch, loop_height, bulge, du, dv)
    # The tile's corner in lattice coordinates: the cell-grid shift, less the fine shift of the loops.
    corner_u = shift_cols * stitch_width - du
    corner_v = shift_rows * row_pitch - dv
    a = math.radians(angle)
    origin = np.array([corner_u * math.cos(a) - corner_v * math.sin(a), corner_u * math.sin(a) + corner_v * math.cos(a)])
    result = PeriodicFit(angle, stitch_width, row_pitch, bitmap, float(loop_height), float(bulge), stitch_z, radius, origin, score,
                         extraction.mm_per_unit, extraction.centre, extraction.axes, extraction.yarn_points_mm, shifted, model_tile)
    progress("Building the complete loops over the cloth")
    build_cloth(result, cloth, u, v, low, height, bow)
    progress("Comparing the complete loops with the scan")
    result.comparison = strand_equations.compare_tube(result.tube, result.yarn_points_mm)
    return result


def build_cloth(result: PeriodicFit, cloth: np.ndarray, u: np.ndarray, v: np.ndarray, low: np.ndarray, height: np.ndarray, bow: np.ndarray) -> None:
    """The tile repeated over the cloth (step 4): row centrelines in the cloth frame, cut to the cloth, as tubes."""
    a = math.radians(result.angle_deg)
    corner_u = result.origin[0] * math.cos(a) + result.origin[1] * math.sin(a)
    corner_v = -result.origin[0] * math.sin(a) + result.origin[1] * math.cos(a)
    period_u, period_v = result.n_cols * result.stitch_width, result.n_rows * result.row_pitch
    inside_u, inside_v = u[cloth], v[cloth]
    first_u = math.floor((inside_u.min() - corner_u) / period_u) - 1
    first_v = math.floor((inside_v.min() - corner_v) / period_v) - 1
    tiles_u = int(math.ceil((inside_u.max() - corner_u) / period_u)) - first_u + 1
    tiles_v = int(math.ceil((inside_v.max() - corner_v) / period_v)) - first_v + 1
    big = np.tile(result.bitmap, (tiles_v, tiles_u))
    big, _ = knitting_core.anchor_bitmap(big)
    rows = control_rows(big, result.stitch_width, result.row_pitch, result.loop_height, result.stitch_bulge, result.stitch_z)
    lines = row_centrelines(rows, tiles_u * period_u, tiles_u * result.n_cols)
    near_cloth = ndimage.binary_dilation(cloth, iterations=5)
    smooth_bow = np.where(np.isfinite(bow), bow, np.nanmean(bow))
    curves, rows_mm = [], []
    for line in lines:
        lu = line[:, 0] + corner_u + first_u * period_u
        lv = line[:, 1] + corner_v + first_v * period_v
        x = lu * math.cos(a) - lv * math.sin(a)
        y = lu * math.sin(a) + lv * math.cos(a)
        col = np.clip(((x - low[0]) / GRID_MM).astype(int), 0, cloth.shape[1] - 1)
        row = np.clip(((y - low[1]) / GRID_MM).astype(int), 0, cloth.shape[0] - 1)
        on = near_cloth[row, col] & (x >= low[0]) & (y >= low[1])
        z = smooth_bow[row, col] + line[:, 2] - result.radius
        points = np.stack([x, y, z], axis=1)
        rows_mm.append(points)
        # Cut to the cloth: each stretch of the row that lies on it becomes one tube.
        edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
        for start, stop in zip(edges[::2], edges[1::2]):
            if stop - start >= 4:
                curves.append((points[start:stop], result.radius, False))
    result.rows_mm = rows_mm
    result.tube = strand_equations.polyline_tubes(curves)
