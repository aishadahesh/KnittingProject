"""Fit Edit Mode's own equations to a scanned model.

Edit Mode builds a fabric from equations (knit loops, knitting_core; crochet posts and chains,
crochet_core) and a grid of stitches. This takes a trained splat of a real cloth and finds the numbers
that make those equations reproduce it, keeping what the person editing has set - how many stitches and
rows the repeat holds, and, if they ask, the stitch grid they drew:

  prepare        the splat read, laid flat (the cloth's up from the training cameras), and its yarn
                 mapped (strand_equations.extract), once per model - the fits below reuse it;
  fit_crochet    crochet_fit with the grid's size as the counts, and its thickness, twist and lean fitted;
  fit_knit       lace_lattice with the bitmap's size as the counts;
  scan_picture   the scan from straight above, turned so its rows run left to right as Edit Mode shows
                 the fabric - to set beside the model;
  PhotoFit       the slow, best fit: scripts/fit_stitch_model.py run in the splat environment (it needs
                 the GPU renderer), fitting the stitch's own curves to the photos; its result read back
                 as Edit Mode's crochet settings (load_stitch_model).

The stitch grid on a photo (for lining it up by eye, before the photo fit): a lattice is the rows'
direction, where the grid starts and the stitch width and row height (cloth-frame mm); every stitch's
post stands at (column x width, row x height) on it. load_cameras, to_splat_frame and project carry
cloth millimetres into any training picture; grid_picture draws the posts, the rows' tops and the
openings over the most square-on one; photo_openings reads each stitch's place in every picture.
"""

from __future__ import annotations

import json
import math
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

import crochet_core
import gaussian_splatting
import lace_lattice
import paths
import strand_equations
from persistence import write_json

GRID_MM = strand_equations.GRID_MM
ROW_MM = (1.2, 6.5)
STITCH_MM = (0.6, 2.2)
HOLE_SIZE_WEIGHT = 0.08
RADIUS_STEPS = np.round(np.arange(0.6, 1.81, 0.2), 2)
TWIST_STEPS = (0.25, 0.5, 0.75, 1.0, 1.5)
LEAN_STEPS = np.round(np.arange(-0.4, 0.41, 0.2), 2)


@dataclass
class CrochetFit:
    params: crochet_core.CrochetParams
    angle_deg: float
    cells: np.ndarray
    shift: tuple
    score: float
    candidates: list = field(default_factory=list)
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


def hole_size(holes: np.ndarray, angle_deg: float):
    """The typical opening's size (along the rows, across them), mm."""
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
    """The model's opening (along, across) mm."""
    open_ = np.asarray(stitches) < 0.5
    if not open_.any():
        return 0.0, 0.0
    best_run = max(max((len(r) for r in "".join("1" if o else "0" for o in row).split("0")), default=0) for row in open_)
    rows_open = max(max((len(r) for r in "".join("1" if o else "0" for o in column).split("0")), default=0) for column in open_.T)
    width = (best_run + 1) * p.stitch_width - 4.0 * p.yarn_radius
    height = rows_open * p.row_height - 2.6 * p.yarn_radius
    return width, height


def fit(extraction, progress=lambda message: None, stitches=None, rows=None, wraps=None, rows_along_holes=None,
        grid=None, refine=True) -> CrochetFit:
    """The scan (a strand_equations.Extraction: its maps, frame and yarn) as crochet lace."""
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
        presence = 1.0 - lace_lattice.fold(holes.astype(float), cloth, u, v, along, across)
        relief = lace_lattice.fold(height - bow, cloth & support, u, v, along, across)
        measured[angle] = hole_size(holes, angle)
        splits = [(r, p) for r in range(1, 13) if ROW_MM[0] <= across / r <= ROW_MM[1]
                  for p in range(2, 30) if STITCH_MM[0] <= along / p <= STITCH_MM[1]]
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


def refine_shape(cells, params, along: float, across: float, presence, relief):
    """The yarn's thickness, then the posts' twist, then their lean."""
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
            continue
        tubes.append((np.stack([x, y, bow[row, col] + points[:, 2]], axis=1), result.params.yarn_radius, closed))
    result.tube = strand_equations.polyline_tubes(tubes)


def find_models(root: Path = paths.ROBOT_SCANS_DIR) -> list[dict]:
    """Every trained splat under robot_scans: its PLY (the cropped one when there is one), the training
    run and dataset it came from (for the cameras), its scale and its held-out PSNR."""
    models = []
    for export in sorted(Path(root).glob("*/splat_output*/export")):
        ply = export / "splat_clean.ply"
        if not ply.is_file():
            ply = export / "splat.ply"
        if not ply.is_file():
            continue
        output, session = export.parent, export.parent.parent
        runs = [p for p in output.glob("*/splatfacto/*")
                if (p / "dataparser_transforms.json").is_file() and not p.name.endswith("_cloth")]
        run = max(runs, key=lambda p: p.stat().st_mtime) if runs else None
        dataset = session / run.parent.parent.name if run is not None else None
        try:
            psnr = float(json.loads((output / "eval.json").read_text(encoding="utf-8"))["results"]["psnr"])
        except (OSError, ValueError, KeyError, TypeError):
            psnr = None
        models.append({
            "name": f"{session.name} / {output.name}" + ("" if ply.name == "splat_clean.ply" else " (uncropped)"),
            "ply": ply, "output": output, "run": run,
            "dataset": dataset if dataset is not None and (dataset / "transforms.json").is_file() else None,
            "mm_per_unit": strand_equations.load_scale(output), "psnr": psnr,
        })
    return models


def prepare(model: dict, progress=lambda message: None) -> dict:
    """`model` as find_models lists it. Returns the laid-flat splat, its scale and its yarn map."""
    progress(f"Reading {model['ply'].name}")
    splat = gaussian_splatting.load_splat_ply(model["ply"])
    normal, mm_per_unit = None, model.get("mm_per_unit")
    if model.get("run") is not None and model.get("dataset") is not None:
        centres, looking = strand_equations.camera_centres_and_axes(model["run"], model["dataset"])
        normal = -looking.mean(axis=0)
        if mm_per_unit is None:
            mm_per_unit = strand_equations.guess_scale(splat["positions"], centres)
    flat, flat_centre, flat_axes = strand_equations.lay_flat(splat, normal, return_frame=True)
    progress("Mapping the yarn")
    extraction = strand_equations.extract(flat, float(mm_per_unit or 1.0), (0.0, 0.0, 1.0), progress=progress)
    maps = extraction.maps
    return {"model": model, "splat": flat, "extraction": extraction, "mm_per_unit": float(mm_per_unit or 1.0),
            "flat_centre": flat_centre, "flat_axes": flat_axes, "low": extraction.map_origin,
            "cloth": lace_lattice.cloth_region(maps["support"]), "bow": lace_lattice.bow_map(maps["support"], maps["height"])}


def fit_crochet(prepared: dict, grid: np.ndarray, wraps: int, read_openings: bool, rows_along_openings: bool,
                progress=lambda message: None):
    """Crochet's numbers for this scan with `grid`'s size as the repeat (rows x stitches), `wraps`
    yarn-overs; the openings read from the scan, or `grid`'s kept."""
    grid = np.asarray(grid, dtype=float)
    return fit(prepared["extraction"], progress=progress, stitches=int(grid.shape[1]), rows=int(grid.shape[0]),
               wraps=int(wraps), rows_along_holes=bool(rows_along_openings),
               grid=None if read_openings else grid, refine=True)


def fit_knit(prepared: dict, bitmap: np.ndarray, progress=lambda message: None):
    """Knit's numbers for this scan with `bitmap`'s size as the repeat (rows x stitches)."""
    bitmap = np.asarray(bitmap)
    return lace_lattice.fit(prepared["extraction"], counts=(int(bitmap.shape[1]), int(bitmap.shape[0])), progress=progress)


def scan_picture(prepared: dict, rows_angle_deg: float, size=(360, 300), zoom: float = 1.6):
    """The scan from straight above, its rows turned to run left to right (a PIL image)."""
    extraction = prepared["extraction"]
    # The cloth frame the fit measured its angle in, then turned by that angle.
    a = math.radians(rows_angle_deg)
    turn = np.array([[math.cos(a), math.sin(a), 0.0], [-math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    axes = turn @ extraction.axes
    splat = strand_equations.transform_splat(prepared["splat"], axes, extraction.centre / extraction.mm_per_unit)
    keep = splat["opacity"] >= 0.1
    frame = gaussian_splatting.orbit_frame(splat["positions"][keep])
    return gaussian_splatting.render_splat(splat, -90.0, 89.9, zoom, size=size, frame=frame)


def load_stitch_model(path) -> tuple[dict, np.ndarray]:
    """A fit_stitch_model.py result as Edit Mode's crochet settings: (crochet params dict, stitch grid)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    params = {"stitch_width": data["stitch_width"], "row_height": data["row_height"], "yarn_radius": data["yarn_radius"],
              "wraps": int(data.get("wraps", 1)), "twist": float(data.get("twist", 0.75)), "lean": float(data.get("lean", 0.0)),
              "templates": [{key: t[key] for key in ("kind", "parity", "part", "closed", "coefficients", "slot") if key in t}
                            for t in data["templates"]]}
    return params, np.asarray(data["stitches"], dtype=np.float32)


class PhotoFit:
    """fit_stitch_model.py as a background process, its progress followed line by line."""

    def __init__(self, output_dir, stitches: int, rows: int, wraps: int, candidates: str = "", lattice_path=None):
        self.output_dir = Path(output_dir)
        self.out = self.output_dir / "stitch_fit"
        arguments = (f"--stitches {int(stitches)} --rows {int(rows)} --wraps {int(wraps)} --per-stitch"
                     + (f" --try {candidates}" if candidates else ""))
        if lattice_path is not None:
            arguments += f" --lattice '{lattice_path}'"
        script = r". .\scripts\splat_env.ps1; & .\.venv-splat\Scripts\python.exe scripts\fit_stitch_model.py "
        self.command = script + f"'{self.output_dir}' {arguments} --out '{self.out}'"
        self.lines: list[str] = []
        self.returncode: int | None = None
        self._process = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="photo-fit")
        self._thread.start()

    def _run(self) -> None:
        try:
            self._process = subprocess.Popen(["powershell", "-ExecutionPolicy", "Bypass", "-Command", self.command], cwd=paths.PROJECT_ROOT,
                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                                             creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))
            for line in self._process.stdout:
                line = line.rstrip()
                # The GPU kernels' build chatter is not progress.
                if line and not any(noise in line for noise in ("cpp_extension", "vswhere", "CategoryInfo", "FullyQualified", "NativeCommand")):
                    self.lines.append(line)
            self.returncode = self._process.wait()
        except Exception as exc:
            self.lines.append(f"Could not start the fit: {exc}")
            self.returncode = -1

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    @property
    def progress(self) -> str:
        """The last line that says where the fit is."""
        for line in reversed(self.lines):
            if line.startswith(("step ", "Trying", "Start", "Before", "After", "Best", "Reading", "Mapping", "Outlining", "  ")):
                return line.strip()
        return self.lines[-1] if self.lines else "Starting"

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()

    def result(self):
        """(params dict, stitch grid, summary line) when it finished well, else None."""
        if self.running or self.returncode != 0:
            return None
        return read_photo_fit(self.out)


def read_photo_fit(fit_dir):
    """A finished photo fit's (params dict, stitch grid, summary line), or None when there is none."""
    path = Path(fit_dir) / "stitch_model.json"
    if not path.is_file():
        return None
    params, stitches = load_stitch_model(path)
    scores = json.loads(path.read_text(encoding="utf-8")).get("scores", {})
    summary = (f"held-out pictures: {scores.get('model_psnr', float('nan')):.2f} dB "
               f"(the free splat {scores.get('splat_psnr', float('nan')):.2f} dB, before fitting {scores.get('before_model_psnr', float('nan')):.2f} dB)")
    return params, stitches, summary


# ----------------------------------------------------------------------------------- the stitch grid on a photo

def load_cameras(run, dataset):
    """Every training picture's camera in the splat's frame (OpenCV axes: world -> camera 4x4) and the
    lens; plus nerfstudio's split (train, held-out indices)."""
    run, dataset = Path(run), Path(dataset)
    transforms = json.loads((dataset / "transforms.json").read_text(encoding="utf-8"))
    parser = json.loads((run / "dataparser_transforms.json").read_text(encoding="utf-8"))
    move = np.eye(4)
    move[:3, :] = np.array(parser["transform"])
    if "applied_transform" in transforms:
        applied = np.eye(4)
        applied[:3, :] = np.array(transforms["applied_transform"])
        move = move @ np.linalg.inv(applied)
    cameras = []
    for frame in transforms["frames"]:
        c2w = move @ np.array(frame["transform_matrix"], dtype=np.float64)
        c2w[:3, 3] *= float(parser["scale"])
        c2w = c2w @ np.diag([1.0, -1.0, -1.0, 1.0])          # OpenGL -> OpenCV axes
        path = (dataset / frame["file_path"]).resolve()
        cameras.append({"viewmat": np.linalg.inv(c2w), "path": path, "name": path.stem})
    lens = {k: float(transforms[k]) for k in ("fl_x", "fl_y", "cx", "cy", "w", "h")}
    lens["dist"] = np.array([transforms.get(k, 0.0) for k in ("k1", "k2", "p1", "p2")])
    count = len(cameras)
    train = set(np.linspace(0, count - 1, int(math.ceil(count * 0.9)), dtype=int).tolist())
    return cameras, lens, [i for i in range(count) if i in train], [i for i in range(count) if i not in train]


def to_splat_frame(prepared: dict, points_mm: np.ndarray) -> np.ndarray:
    """Cloth-frame mm into the splat's (and its cameras') frame."""
    ex = prepared["extraction"]
    flat = (np.asarray(points_mm, dtype=float) @ ex.axes + ex.centre) / prepared["mm_per_unit"]
    return flat @ prepared["flat_axes"] + prepared["flat_centre"]


def project(points_splat: np.ndarray, camera: dict, lens: dict):
    """Pixels (x, y) of splat-frame points in a picture, and which lie in front of its camera."""
    cam = (camera["viewmat"][:3, :3] @ np.asarray(points_splat, dtype=float).T + camera["viewmat"][:3, 3:4]).T
    ahead = cam[:, 2] > 1e-6
    z = np.where(ahead, cam[:, 2], 1.0)
    return np.stack([lens["fl_x"] * cam[:, 0] / z + lens["cx"], lens["fl_y"] * cam[:, 1] / z + lens["cy"]], 1), ahead


def bow_height(prepared: dict, x, y):
    """The cloth's bow (mm) under cloth-frame points."""
    bow, low = prepared["bow"], prepared["low"]
    cols = np.clip(((np.asarray(x) - low[0]) / GRID_MM).astype(int), 0, bow.shape[1] - 1)
    rows = np.clip(((np.asarray(y) - low[1]) / GRID_MM).astype(int), 0, bow.shape[0] - 1)
    return bow[rows, cols]


def lattice_to_cloth(lattice: dict, u, v):
    """Lattice coordinates (along the rows, across them; mm) to the cloth frame's x, y."""
    a = math.radians(lattice["angle_deg"])
    u, v = np.asarray(u, dtype=float) + lattice["shift"][0], np.asarray(v, dtype=float) + lattice["shift"][1]
    return u * math.cos(a) - v * math.sin(a), u * math.sin(a) + v * math.cos(a)


def lattice_from_fit(fit) -> dict:
    """A crochet_fit result's lattice."""
    return {"angle_deg": float(fit.angle_deg), "shift": [float(fit.shift[0]), float(fit.shift[1])],
            "stitch_width": float(fit.params.stitch_width), "row_height": float(fit.params.row_height)}


def stitch_places(prepared: dict, lattice: dict) -> list:
    """Every stitch (column, row) whose post stands on the cloth."""
    cloth, low = prepared["cloth"], prepared["low"]
    rows_px, cols_px = np.nonzero(cloth)
    x, y = low[0] + cols_px * GRID_MM, low[1] + rows_px * GRID_MM
    a = math.radians(lattice["angle_deg"])
    u = x * math.cos(a) + y * math.sin(a) - lattice["shift"][0]
    v = -x * math.sin(a) + y * math.cos(a) - lattice["shift"][1]
    w, h = lattice["stitch_width"], lattice["row_height"]
    places = []
    for row in range(int(math.floor(v.min() / h)), int(math.ceil(v.max() / h)) + 1):
        for column in range(int(math.floor(u.min() / w)), int(math.ceil(u.max() / w)) + 1):
            xc, yc = lattice_to_cloth(lattice, column * w, (row + 0.5) * h)
            col_px, row_px = int((xc - low[0]) / GRID_MM), int((yc - low[1]) / GRID_MM)
            if 0 <= row_px < cloth.shape[0] and 0 <= col_px < cloth.shape[1] and cloth[row_px, col_px]:
                places.append((column, row))
    return places


def square_on_camera(prepared: dict, cameras: list) -> int:
    """The picture looking most straight down at the middle of the cloth (and centred on it)."""
    cloth, low = prepared["cloth"], prepared["low"]
    rows_px, cols_px = np.nonzero(cloth)
    middle_x, middle_y = low[0] + np.median(cols_px) * GRID_MM, low[1] + np.median(rows_px) * GRID_MM
    middle = to_splat_frame(prepared, np.array([[middle_x, middle_y, float(bow_height(prepared, middle_x, middle_y))]]))[0]
    up = to_splat_frame(prepared, np.array([[middle_x, middle_y, 10.0]]))[0] - middle
    up /= np.linalg.norm(up)
    best, best_score = 0, -np.inf
    for index, camera in enumerate(cameras):
        rotation, translation = camera["viewmat"][:3, :3], camera["viewmat"][:3, 3]
        centre = -rotation.T @ translation
        looking = rotation.T @ np.array([0.0, 0.0, 1.0])
        to_middle = (middle - centre) / np.linalg.norm(middle - centre)
        score = float(-looking @ up) + float(looking @ to_middle)          # straight down, and aimed at the middle
        if score > best_score:
            best, best_score = index, score
    return best


def grid_picture(prepared: dict, lattice: dict, camera: dict, lens: dict, photo: np.ndarray, open_places=()):
    """The photo with the stitch grid drawn on it: each post (green, foot to top), each row's top (yellow)
    and each stitch read as an opening (red ring)."""
    from PIL import Image, ImageDraw

    image = Image.fromarray(np.asarray(photo, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    w, h = lattice["stitch_width"], lattice["row_height"]
    places = stitch_places(prepared, lattice)
    open_places = set(map(tuple, open_places))

    def pixels(u, v):
        x, y = lattice_to_cloth(lattice, u, v)
        points = np.stack([x, y, bow_height(prepared, x, y)], -1)
        xy, ahead = project(to_splat_frame(prepared, points), camera, lens)
        return [tuple(p) for p in xy], bool(ahead.all())

    for column, row in places:
        foot_top, ok = pixels(np.array([column * w, column * w]), np.array([row * h, (row + 1) * h]))
        if not ok:
            continue
        top_line, _ = pixels(np.array([(column - 0.5) * w, (column + 0.5) * w]), np.array([(row + 1) * h] * 2))
        draw.line(top_line, fill=(255, 220, 0), width=1)
        if (column, row) in open_places:
            centre, _ = pixels(np.array([column * w]), np.array([(row + 0.5) * h]))
            cx, cy = centre[0]
            draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), outline=(255, 40, 40), width=2)
        else:
            draw.line(foot_top, fill=(40, 255, 80), width=2)
    return image


def read_photos(cameras: list) -> np.ndarray:
    """Every training picture (RGB uint8)."""
    import cv2
    return np.stack([cv2.cvtColor(cv2.imread(str(c["path"])), cv2.COLOR_BGR2RGB) for c in cameras])


def photo_openings(prepared: dict, lattice: dict, places: list, cameras: list, lens: dict, photos: np.ndarray, views,
                   table, yarn=(0.85, 0.85, 0.82)) -> np.ndarray:
    """For each stitch place, the share of looks at it (a 4 x 4 patch round its post, in every picture
    in `views` that sees it) that see the table's colour rather than the yarn's: high = an opening."""
    w, h = lattice["stitch_width"], lattice["row_height"]
    du, dv = np.meshgrid(np.linspace(-0.35, 0.35, 4) * w, np.linspace(0.2, 0.75, 4) * h)
    du, dv = du.ravel(), dv.ravel()
    u = np.concatenate([column * w + du for column, _ in places])
    v = np.concatenate([row * h + dv for _, row in places])
    x, y = lattice_to_cloth(lattice, u, v)
    splat = to_splat_frame(prepared, np.stack([x, y, bow_height(prepared, x, y)], 1))
    table, yarn = np.asarray(table, dtype=float), np.asarray(yarn, dtype=float)
    votes, looks = np.zeros(len(splat)), np.zeros(len(splat))
    for i in views:
        xy, ahead = project(splat, cameras[i], lens)
        seen = ahead & (xy[:, 0] >= 0) & (xy[:, 0] < lens["w"] - 1) & (xy[:, 1] >= 0) & (xy[:, 1] < lens["h"] - 1)
        colour = photos[i][xy[seen, 1].astype(int), xy[seen, 0].astype(int)] / 255.0
        votes[seen] += np.linalg.norm(colour - table, axis=1) < np.linalg.norm(colour - yarn, axis=1)
        looks[seen] += 1
    return (votes / np.maximum(looks, 1)).reshape(len(places), -1).mean(axis=1)


def table_colour(photos: np.ndarray, prepared: dict, cameras: list, lens: dict, views) -> np.ndarray:
    """What shows through the openings: the darkest strongly-blue-or-dark tenth of the pixels over the
    cloth, its median colour (RGB 0..1)."""
    cloth, low = prepared["cloth"], prepared["low"]
    rows_px, cols_px = np.nonzero(cloth)
    pick = np.random.default_rng(0).choice(len(rows_px), min(4000, len(rows_px)), replace=False)
    x, y = low[0] + cols_px[pick] * GRID_MM, low[1] + rows_px[pick] * GRID_MM
    splat = to_splat_frame(prepared, np.stack([x, y, bow_height(prepared, x, y)], 1))
    colours = []
    for i in list(views)[::4]:
        xy, ahead = project(splat, cameras[i], lens)
        seen = ahead & (xy[:, 0] >= 0) & (xy[:, 0] < lens["w"] - 1) & (xy[:, 1] >= 0) & (xy[:, 1] < lens["h"] - 1)
        colours.append(photos[i][xy[seen, 1].astype(int), xy[seen, 0].astype(int)] / 255.0)
    colours = np.concatenate(colours)
    brightness = colours.mean(axis=1)
    return np.median(colours[brightness <= np.percentile(brightness, 10)], axis=0)


GRID_FILE = "stitch_grid.json"


def save_grid(output_dir, lattice: dict) -> Path:
    """The stitch grid lined up by eye, beside the model (fit_stitch_model.py --lattice reads it)."""
    path = Path(output_dir) / GRID_FILE
    write_json(path, {k: lattice[k] for k in ('angle_deg', 'shift', 'stitch_width', 'row_height')}, indent=1)
    return path


def load_grid(output_dir) -> dict | None:
    try:
        return json.loads((Path(output_dir) / GRID_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
