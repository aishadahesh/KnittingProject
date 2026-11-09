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
from pathlib import Path

import numpy as np

import crochet_fit
import gaussian_splatting
import lace_lattice
import strand_equations
from persistence import write_json


def prepare(model: dict, progress=lambda message: None) -> dict:
    """`model` as gui_model.find_models lists it. Returns the laid-flat splat, its scale and its yarn map."""
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
    return crochet_fit.fit(prepared["extraction"], progress=progress, stitches=int(grid.shape[1]), rows=int(grid.shape[0]),
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


ROOT = Path(__file__).resolve().parent.parent
GRID_MM = strand_equations.GRID_MM


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
            self._process = subprocess.Popen(["powershell", "-ExecutionPolicy", "Bypass", "-Command", self.command], cwd=ROOT,
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
