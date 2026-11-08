"""Fit crochet stitch equations to the real photos: the lace rebuilt from a few closed-form curves,
rendered into every calibrated picture and changed until the renders look like the pictures.

    .venv-splat\\Scripts\\python.exe scripts\\fit_stitch_model.py robot_scans\\ur5_20261029_patch\\splat_output_s2_close --stitches 3 --rows 3 --wraps 1
    ... --synthetic        a self-check: fit to renders of a known model instead of the photos
    ... --try 3x3,6x3,9x3  a short fit for each stitches x rows count, ranked by the held-out pictures,
                           then the full fit for the best
    ... --lattice grid.json  the stitch grid as lined up by eye on a photo (Edit Mode, scan_fit): the
                           rows' direction, start, stitch width and row height, kept as they are
    ... --per-stitch       each stitch of the rows x stitches repeat its own template (posts in V pairs,
                           crossed beside an opening), not one shape for every post
    ... --stitch mesh      instead of filet posts and chains, a stitch family round each opening
                           (crochet_stitches: mesh, trellis, vstitch, shell), tiled on the openings'
                           diamond lattice; written to stitch_fit_<family>

The model (all of it differentiable, torch):
  * a lattice - the rows' angle, where the repeat starts, the stitch width w and row height h;
  * a stitch template per stitch type (a post: its two legs, its yarn-over wraps and its V top; a
    chain: its top loop) and per row parity (rows are worked back and forth). Each curve is a Fourier
    series in stitch-local coordinates (x/w, y/h, z in mm), shared by every stitch of that type:
        open:    p(t) = a + b t + sum_k c_k sin(k pi t)
        closed:  p(t) = a + sum_k (c_k cos 2 pi k t + d_k sin 2 pi k t)
    started from crochet_core's hand-made curves;
  * the yarn's radius, its colour (spherical harmonics of degree 1 per curve kind) and the table's
    colour showing through the openings;
  * the cloth's bow: a smooth height grid (4 mm) under the lattice, started from the scan's.
The counts per repeat set the stitch width and row height (crochet_fit's reading of the scan with the
counts given here). Which stitches are posts and which are chains across an opening is read stitch by
stitch over the whole cloth, from the photos themselves (photo_openings): each stitch's place is
looked at in every picture that sees it, and where the table's blue shows through more than the
yarn's white, it is an opening. The rows of openings are staggered (each slides about a stitch along
from the one before), so one rectangular repeat cannot hold them; and the splat's yarn map, which
was read first, missed most of the openings.

Every curve is drawn as a chain of long Gaussians (one per step along it, as wide as the yarn),
rendered with gsplat into a batch of the training pictures; the loss is L1 + D-SSIM inside the cloth's
outline (the cropped splat's coverage, its openings included), plus the silhouette - the model's
coverage against the cropped splat's, over the whole picture, so yarn spilling past the cloth's edge
or filling an opening costs (without it, the fit covered everything in white: the cloth is mostly
white) - and a pull towards smooth templates and a smooth bow. Stitch width and row height stay
within LATTICE_SLACK of the repeat read from the openings. Stages: the lattice, bow and colours at half size; then the templates and radius at
full size; then everything. The pictures nerfstudio held out (every tenth) judge the result, beside
the free splat judged the same way.

Writes <output>/stitch_fit/: stitch_model.json (every number, the templates' coefficients), held-out
render | photo pairs, and progress lines on stdout for the app to follow.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from gsplat import rasterization
from pytorch_msssim import ssim
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import crochet_core  # noqa: E402
import crochet_fit  # noqa: E402
import crochet_stitches  # noqa: E402
import gaussian_splatting  # noqa: E402
import lace_lattice  # noqa: E402
import strand_equations  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OPEN_TERMS, CLOSED_TERMS = 10, 8
STEP_MM = 0.12                 # spacing of the Gaussians along a curve
FIELD_MM = 4.0                 # spacing of the bow grid
BATCH = 4
MIN_RADIUS_MM, MAX_RADIUS_MM = 0.15, 0.8
LATTICE_SLACK = 0.05            # stitch width and row height may move this share from the openings' repeat
SILHOUETTE_WEIGHT = 0.5
MAX_SLIP_MM = 0.7               # one stitch slips at most this far from its place on the lattice
MAX_STRETCH_MM = 1.5            # the stretch grid slides the lattice at most this far (less than a stitch)
BACKING_MM = 1.2                # the shadowed yarn behind a post's gaps, drawn as a sheet this far behind it
RING = 6                        # Gaussians round the yarn at each step, so it is shaded as a round tube


def log(message: str) -> None:
    print(message, flush=True)


# ----------------------------------------------------------------------------------- the scene

def find_run_and_dataset(output: Path):
    runs = [p for p in output.glob("*/splatfacto/*") if (p / "dataparser_transforms.json").is_file() and not p.name.endswith("_cloth")]
    if not runs:
        raise SystemExit(f"no trained run under {output}")
    run = max(runs, key=lambda p: p.stat().st_mtime)
    return run, output.parent / run.parent.parent.name


def load_cameras(run: Path, dataset: Path):
    """Every training picture's camera in the splat's frame (OpenCV axes) and its lens, plus nerfstudio's split."""
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
        cameras.append({"viewmat": np.linalg.inv(c2w), "path": (dataset / frame["file_path"]).resolve()})
    lens = {k: float(transforms[k]) for k in ("fl_x", "fl_y", "cx", "cy", "w", "h")}
    lens["dist"] = np.array([transforms.get(k, 0.0) for k in ("k1", "k2", "p1", "p2")])
    count = len(cameras)
    train = set(np.linspace(0, count - 1, int(math.ceil(count * 0.9)), dtype=int).tolist())   # nerfstudio's "fraction" split
    return cameras, lens, [i for i in range(count) if i in train], [i for i in range(count) if i not in train]


def intrinsics(lens, scale: float = 1.0) -> np.ndarray:
    return np.array([[lens["fl_x"] * scale, 0, lens["cx"] * scale], [0, lens["fl_y"] * scale, lens["cy"] * scale], [0, 0, 1]])


def load_photos(cameras, lens) -> torch.Tensor:
    """The pictures as taken (uint8, on the GPU). Not undistorted: the splat, its poses and these
    pinhole cameras were trained on them as they are (undistorting them cost the splat a dB)."""
    return torch.from_numpy(np.stack([cv2.cvtColor(cv2.imread(str(c["path"])), cv2.COLOR_BGR2RGB) for c in cameras])).to(DEVICE)


def splat_tensors(ply: Path):
    """A splat PLY as gsplat's inputs (its own frame)."""
    _, table = _read_ply(ply)
    means = np.stack([table["x"], table["y"], table["z"]], 1)
    quats = np.stack([table["rot_0"], table["rot_1"], table["rot_2"], table["rot_3"]], 1)
    scales = np.exp(np.stack([table["scale_0"], table["scale_1"], table["scale_2"]], 1))
    opacity = 1.0 / (1.0 + np.exp(-table["opacity"]))
    dc = np.stack([table["f_dc_0"], table["f_dc_1"], table["f_dc_2"]], 1)
    rest = [n for n in table.dtype.names if n.startswith("f_rest_")]
    rest = np.stack([table[n] for n in sorted(rest, key=lambda n: int(n.split("_")[-1]))], 1).reshape(len(table), 3, -1).transpose(0, 2, 1) if rest else np.zeros((len(table), 0, 3))
    sh = np.concatenate([dc[:, None, :], rest], 1)
    t = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=DEVICE)
    return t(means), t(quats / np.linalg.norm(quats, axis=1, keepdims=True)), t(scales), t(opacity), t(sh), int(round(math.sqrt(sh.shape[1]) - 1))


def _read_ply(path: Path):
    kinds = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4"}
    with path.open("rb") as handle:
        header, properties, count = [], [], 0
        while True:
            line = handle.readline().decode("ascii").strip()
            header.append(line)
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("property"):
                _, kind, name = line.split()[:3]
                properties.append((name, kinds[kind]))
            elif line == "end_header":
                break
        return header, np.fromfile(handle, dtype=np.dtype(properties), count=count)


def render(means, quats, scales, opacities, colors, sh_degree, viewmats, Ks, width, height, background):
    images, alphas, _ = rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height,
                                      sh_degree=sh_degree, backgrounds=background)
    return images.clamp(0, 1), alphas


def cloth_masks(ply: Path, cameras, lens) -> torch.Tensor:
    """Where the cloth is in each picture, its openings included: the cropped splat's coverage, filled
    and widened a little. The board and the table round the cloth are left out of the loss."""
    means, quats, scales, opacity, sh, degree = splat_tensors(ply)
    K = torch.tensor(intrinsics(lens), dtype=torch.float32, device=DEVICE)
    masks, openings, coverage = [], [], []
    for camera in cameras:
        view = torch.tensor(camera["viewmat"], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            _, alpha = render(means, quats, scales, opacity, sh, degree, view[None], K[None], int(lens["w"]), int(lens["h"]),
                              torch.zeros(1, 3, device=DEVICE))
        coverage.append(alpha[0, ..., 0])
        yarn = alpha[0, ..., 0].cpu().numpy() > 0.5
        covered = ndimage.binary_fill_holes(ndimage.binary_closing(yarn, iterations=8))
        masks.append(ndimage.binary_erosion(covered, iterations=4))
        openings.append(ndimage.binary_erosion(covered & (alpha[0, ..., 0].cpu().numpy() < 0.2), iterations=2))
    return (torch.tensor(np.stack(masks), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.stack(openings), dtype=torch.bool, device=DEVICE),
            torch.stack(coverage).clamp(0, 1))


def table_colour(photos, openings) -> torch.Tensor:
    """What shows through the openings (the median colour there over every picture), kept fixed: free,
    it learned to be yarn-coloured and stood in for yarn the model had not drawn."""
    pixels = photos[openings].float() / 255.0
    return pixels.median(dim=0).values if len(pixels) else torch.tensor([0.08, 0.08, 0.18], device=DEVICE)


# ----------------------------------------------------------------------------------- the start

def prepare(output: Path, run: Path, dataset: Path, stitches: int, rows: int, wraps: int, lattice: dict | None = None,
            family: str | None = None):
    """The scan read as in Model Mode (laid flat, yarn mapped, crochet_fit with these counts - or the
    `lattice` lined up by eye), and the frames to carry the model's millimetres into the splat's (and
    the cameras') frame."""
    ply = output / "export" / ("splat_clean.ply" if (output / "export" / "splat_clean.ply").is_file() else "splat.ply")
    splat = gaussian_splatting.load_splat_ply(ply)
    mm_per_unit = strand_equations.load_scale(output)
    centres, looking = strand_equations.camera_centres_and_axes(run, dataset)
    if mm_per_unit is None:
        mm_per_unit = strand_equations.guess_scale(splat["positions"], centres)
    flat, flat_centre, flat_axes = strand_equations.lay_flat(splat, -looking.mean(axis=0), return_frame=True)
    log("Mapping the yarn")
    extraction = strand_equations.extract(flat, mm_per_unit, (0.0, 0.0, 1.0))
    if family is not None:
        start = family_start(extraction, family)
    elif lattice is not None:
        log(f"The stitch grid as lined up on a photo: rows at {lattice['angle_deg']:.1f} deg, stitch {lattice['stitch_width']:.2f} mm, "
            f"row {lattice['row_height']:.2f} mm")
        radius = float(np.clip(extraction.summary()["median_radius_mm"] or 0.3, 0.15, 0.8))
        start = SimpleNamespace(params=crochet_core.CrochetParams(stitch_width=lattice["stitch_width"], row_height=lattice["row_height"],
                                                                  wraps=wraps, yarn_radius=radius),
                                angle_deg=float(lattice["angle_deg"]), shift=tuple(lattice["shift"]), cells=np.ones((1, 1)))
    else:
        log(f"Reading the crochet: {rows} rows x {stitches} stitches per repeat, {wraps} yarn-over(s)")
        start = crochet_fit.fit(extraction, stitches=stitches, rows=rows, wraps=wraps, rows_along_holes=True, refine=True)
    maps = extraction.maps
    return {"ply": ply, "mm": float(mm_per_unit), "start": start, "extraction": extraction, "family": family,
            "flat_centre": flat_centre, "flat_axes": flat_axes,
            "cloth": lace_lattice.cloth_region(maps["support"]), "bow": lace_lattice.bow_map(maps["support"], maps["height"]),
            "low": extraction.map_origin}


def family_start(extraction, family: str):
    """A stitch family's start: the openings' lattice - one opening every W mm along a row, the rows of
    openings H mm apart, each slid `slide` mm along from the one below, all read from the openings'
    centres - its origin on the opening nearest the cloth's middle. The families are drawn for rows
    slid half a repeat (crochet_stitches); `shear` carries them onto the slide measured here."""
    support = extraction.maps["support"]
    lattice = lace_lattice.hole_lattice(support)
    if lattice is None:
        raise RuntimeError("no regular openings found")
    angle, W, _ = lattice
    centres, area = lace_lattice.openings(support)
    big = centres[(area >= 0.4 * np.median(np.sort(area)[::-1][:10])) & (area >= lace_lattice.MIN_OPENING_MM2)] + extraction.map_origin
    cloth = lace_lattice.cloth_region(support)
    rows_px, cols_px = np.nonzero(cloth)
    middle = extraction.map_origin + np.array([cols_px.mean(), rows_px.mean()]) * 0.1
    hole = big[np.argmin(np.linalg.norm(big - middle, axis=1))]
    a = math.radians(angle)
    to_uv = lambda xy: np.stack([xy[..., 0] * math.cos(a) + xy[..., 1] * math.sin(a), -xy[..., 0] * math.sin(a) + xy[..., 1] * math.cos(a)], -1)
    shift = tuple(to_uv(hole))
    # Rows of openings: grouped by how far across the rows they lie; then how far apart and how slid.
    uv = to_uv(big) - np.array(shift)
    order = np.argsort(uv[:, 1])
    groups = [g for g in np.split(order, np.flatnonzero(np.diff(uv[order, 1]) > 0.4 * W) + 1) if len(g) >= 2]
    levels = np.array([np.median(uv[g, 1]) for g in groups])
    phases = np.array([np.angle(np.exp(2j * np.pi * uv[g, 0] / W).mean()) / (2 * np.pi) for g in groups])
    H = float(np.median(np.diff(levels)))
    slide = float(np.median((np.diff(phases) + 0.5) % 1.0 - 0.5) * W) % W
    radius = float(np.clip(extraction.summary()["median_radius_mm"] or 0.3, 0.15, 0.8))
    log(f"{family}: openings every {W:.2f} mm along rows at {angle:.1f} deg, rows of openings {H:.2f} mm apart, "
        f"each slid {slide:.2f} mm along; yarn r {radius:.2f} mm")
    return SimpleNamespace(params=crochet_core.CrochetParams(stitch_width=W, row_height=H, yarn_radius=radius),
                           angle_deg=float(angle), shift=shift, cells=np.ones((1, 1)), family=family,
                           slide=slide, shear=(slide - W / 2) / H)


def family_instances(scene, start):
    """Every opening of the lattice over the cloth (and a repeat round it): ("motif", 0) -> (column,
    row), each row's columns `slide` further along."""
    p, a, low = start.params, math.radians(start.angle_deg), scene["low"]
    cloth = scene["cloth"]
    near = ndimage.distance_transform_edt(~cloth) * 0.1 <= 0.75 * max(p.stitch_width, p.row_height)
    rows_px, cols_px = np.nonzero(cloth)
    x, y = low[0] + cols_px * 0.1, low[1] + rows_px * 0.1
    u, v = x * math.cos(a) + y * math.sin(a) - start.shift[0], -x * math.sin(a) + y * math.cos(a) - start.shift[1]
    places = []
    for row in range(int(math.floor(v.min() / p.row_height)) - 1, int(math.ceil(v.max() / p.row_height)) + 2):
        for column in range(int(math.floor(u.min() / p.stitch_width)) - 2, int(math.ceil(u.max() / p.stitch_width)) + 2):
            along = column + row * start.slide / p.stitch_width
            uc, vc = along * p.stitch_width + start.shift[0], row * p.row_height + start.shift[1]
            xc, yc = uc * math.cos(a) - vc * math.sin(a), uc * math.sin(a) + vc * math.cos(a)
            col_px, row_px = int((xc - low[0]) / 0.1), int((yc - low[1]) / 0.1)
            if 0 <= row_px < near.shape[0] and 0 <= col_px < near.shape[1] and near[row_px, col_px]:
                places.append((along, row))
    return {("motif", 0): places}


def hand_template(params: crochet_core.CrochetParams, parity: int):
    """crochet_core's hand-made stitch at the origin: {kind: [(points mm, closed, part)]}."""
    direction = 1 if parity == 0 else -1
    post = [(leg, False, "leg") for leg in crochet_core.post_legs(0.0, 0.0, params, direction)]
    post += [(wrap, False, "wrap") for wrap in crochet_core.post_wraps(0.0, 0.0, params, direction)]
    top = crochet_core.top_loop(0.0, params.row_height, params, direction)
    return {"post": post + [(top, True, "top")], "chain": [(top, True, "top")]}


def basis(samples: int, closed: bool, terms: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, samples, endpoint=not closed)
    return strand_equations._basis(t, closed, terms)


class StitchModel(torch.nn.Module):
    """The lace as equations, drawn as Gaussians in the splat's frame (module docstring)."""

    def __init__(self, scene, instances, width, height):
        super().__init__()
        start = scene["start"]
        p = start.params
        self.w0, self.h0 = p.stitch_width, p.row_height
        self.angle = torch.nn.Parameter(torch.tensor(math.radians(start.angle_deg)))
        self.shift = torch.nn.Parameter(torch.tensor(list(start.shift), dtype=torch.float32))
        self.log_w = torch.nn.Parameter(torch.tensor(math.log(p.stitch_width)))
        self.log_h = torch.nn.Parameter(torch.tensor(math.log(p.row_height)))
        self.log_r = torch.nn.Parameter(torch.tensor(math.log(p.yarn_radius)))
        # Templates: one Fourier curve per part of each stitch the instances name - a kind and row parity,
        # or (--per-stitch) a kind and its place in the repeat, (kind, row, column).
        self.curves, coefficients, starts = [], [], []
        family = scene.get("family")
        for key in sorted(instances):
            kind, parity = key[0], int(key[1]) % 2
            if family:
                parts = [(points + np.outer(points[:, 1], [start.shear, 0.0, 0.0]), closed, part)
                         for points, closed, part in crochet_stitches.motif(family, p.stitch_width, p.row_height, p.yarn_radius)]
            else:
                parts = hand_template(p, parity)[kind]
            for points, closed, part in parts:
                scaled = points / np.array([self.w0, self.h0, 1.0])
                terms = CLOSED_TERMS if closed else OPEN_TERMS
                length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
                samples = max(int(length / STEP_MM), 12)
                B = basis(len(scaled), closed, terms)
                fitted = np.linalg.lstsq(B, scaled, rcond=None)[0]
                self.curves.append({"key": key, "parity": parity, "kind": kind, "part": part, "closed": closed, "terms": terms,
                                    "basis": torch.tensor(basis(samples, closed, terms), dtype=torch.float32, device=DEVICE)})
                coefficients.append(torch.nn.Parameter(torch.tensor(fitted, dtype=torch.float32)))
                starts.append(fitted.copy())
        self.coefficients = torch.nn.ParameterList(coefficients)
        self.start_coefficients = [torch.tensor(c, dtype=torch.float32, device=DEVICE) for c in starts]
        # Colour: each part kind's yarn colour, lit by one light (direction, ambient, diffuse) as a
        # round tube is; the table showing through the openings (set from the photos); opacity.
        parts = sorted({c["part"] for c in self.curves})
        self.part_index = {name: i for i, name in enumerate(parts)}
        self.albedo = torch.nn.Parameter(torch.full((len(parts), 3), 0.9))
        self.light = torch.nn.Parameter(torch.tensor([0.3, 0.3, 1.0]))
        self.ambient = torch.nn.Parameter(torch.tensor(0.45))
        self.diffuse = torch.nn.Parameter(torch.tensor(0.55))
        self.register_buffer("background", torch.tensor([0.08, 0.08, 0.18]))
        self.opacity_logit = torch.nn.Parameter(torch.tensor(3.0))
        # The bow grid over the cloth (cloth-frame mm), started from the scan's bow.
        low, cloth, bow = scene["low"], scene["cloth"], scene["bow"]
        rows_px, cols_px = np.nonzero(cloth)
        x0, x1 = low[0] + cols_px.min() * 0.1 - 5, low[0] + cols_px.max() * 0.1 + 5
        y0, y1 = low[1] + rows_px.min() * 0.1 - 5, low[1] + rows_px.max() * 0.1 + 5
        gx, gy = np.arange(x0, x1 + FIELD_MM, FIELD_MM), np.arange(y0, y1 + FIELD_MM, FIELD_MM)
        sample = lambda x, y: bow[np.clip(((y - low[1]) / 0.1).astype(int), 0, bow.shape[0] - 1), np.clip(((x - low[0]) / 0.1).astype(int), 0, bow.shape[1] - 1)]
        GX, GY = np.meshgrid(gx, gy)
        grid = ndimage.gaussian_filter(np.nan_to_num(sample(GX, GY)), 1.0)
        self.field = torch.nn.Parameter(torch.tensor(grid, dtype=torch.float32)[None, None])
        # The cloth's stretch over the same grid: how far (mm) each part has slid along the cloth from the
        # lattice, so the stitches follow the real ones across the patch instead of drifting off them.
        self.warp = torch.nn.Parameter(torch.zeros((1, 2) + grid.shape, dtype=torch.float32))
        # Shade: yarn lying deeper in the cloth (behind the strands in front) darker, by this much per mm.
        self.depth_k = torch.nn.Parameter(torch.tensor(0.3))
        # Behind every post, the yarn further back seen through the gaps between strands: a flat sheet of
        # one learned (shadowed) colour - so a gap need not show the table's blue, which only openings do.
        self.backing = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))     # logits of its RGB (0.5 grey)
        self.field_box = (float(x0), float(x1), float(y0), float(y1))
        # Stitch instances: (column, row) on the lattice, per kind and parity.
        self.instances = {key: torch.tensor(value, dtype=torch.float32, device=DEVICE) for key, value in instances.items() if len(value)}
        # Each stitch's own small slip from the lattice (mm, along and across the rows): hand-made lace is
        # not exact, and a stitch half a yarn's width off its place blurs the shared shape into a smear.
        self.slip_index = {key: i for i, key in enumerate(self.instances)}
        self.slips = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(len(v), 2)) for v in self.instances.values()])
        self.mm = scene["mm"]
        to_splat = lambda a: torch.tensor(np.asarray(a, dtype=np.float32), device=DEVICE)
        self.ex_centre, self.ex_axes = to_splat(scene["extraction"].centre), to_splat(scene["extraction"].axes)
        self.flat_centre, self.flat_axes = to_splat(scene["flat_centre"]), to_splat(scene["flat_axes"])
        self.to(DEVICE)

    @property
    def w(self):
        return torch.exp(self.log_w).clamp(self.w0 * (1 - LATTICE_SLACK), self.w0 * (1 + LATTICE_SLACK))

    @property
    def h(self):
        return torch.exp(self.log_h).clamp(self.h0 * (1 - LATTICE_SLACK), self.h0 * (1 + LATTICE_SLACK))

    @property
    def r(self):
        # Bounded: free, it grew past a millimetre to cover cloth that too few stitches left bare.
        return torch.exp(self.log_r).clamp(MIN_RADIUS_MM, MAX_RADIUS_MM)

    def curve_points(self, index):
        """One template curve (stitch-local mm) at its samples."""
        local = self.curves[index]["basis"] @ self.coefficients[index]
        return local * torch.stack([self.w, self.h, torch.ones((), device=DEVICE)])

    def centrelines(self):
        """Every yarn curve of every stitch, cloth-frame mm: [(points (instances, samples, 3), closed, part)]."""
        out = []
        c, s = torch.cos(self.angle), torch.sin(self.angle)
        for index, curve in enumerate(self.curves):
            origins = self.instances.get(curve["key"])
            if origins is None:
                continue
            local = self.curve_points(index)[None]                               # (1, S, 3)
            slip = MAX_SLIP_MM * torch.tanh(self.slips[self.slip_index[curve["key"]]] / MAX_SLIP_MM)
            u = origins[:, None, 0] * self.w + local[..., 0] + self.shift[0] + slip[:, None, 0]
            v = origins[:, None, 1] * self.h + local[..., 1] + self.shift[1] + slip[:, None, 1]
            x, y = u * c - v * s, u * s + v * c
            slide = MAX_STRETCH_MM * torch.tanh(self.sample(self.warp, x, y) / MAX_STRETCH_MM)
            x, y = x + slide[0], y + slide[1]
            z = local[..., 2] + self.bow(x, y)
            out.append((torch.stack([x, y, z], -1), curve["closed"], curve["part"], local[..., 2].expand_as(x)))
        return out

    def sample(self, field, x, y):
        """A grid over the cloth (channels, rows, columns) at cloth-frame points: (channels,) + x.shape."""
        x0, x1, y0, y1 = self.field_box
        grid = torch.stack([(x - x0) / (x1 - x0) * 2 - 1, (y - y0) / (y1 - y0) * 2 - 1], -1)
        flat = grid.reshape(1, 1, -1, 2)
        return torch.nn.functional.grid_sample(field, flat, align_corners=True, padding_mode="border").reshape((field.shape[1],) + x.shape)

    def bow(self, x, y):
        return self.sample(self.field, x, y)[0]

    def to_splat(self, points):
        flat = (points @ self.ex_axes + self.ex_centre) / self.mm
        return flat @ self.flat_axes + self.flat_centre

    def gaussians(self):
        """The yarn as round tubes in the splat's frame, for gsplat: at each step along a curve a ring of
        RING long Gaussians round the centreline, each lit by its own side's direction (module docstring)."""
        means, quats, scales, colours = [], [], [], []
        light = self.light / self.light.norm()
        theta = torch.arange(RING, device=DEVICE, dtype=torch.float32) * (2 * math.pi / RING)
        for points, closed, part, depth in self.centrelines():
            nxt = torch.roll(points, -1, dims=1) if closed else points[:, 1:]
            cur = points if closed else points[:, :-1]
            depth = depth if closed else (depth[:, :-1] + depth[:, 1:]) / 2
            step = nxt - cur                                                     # cloth-frame mm
            length = step.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            tangent = step / length
            up_hint = torch.tensor([0.0, 0.0, 1.0], device=DEVICE).expand_as(tangent)
            side = torch.linalg.cross(tangent, up_hint)
            side = side / side.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            up = torch.linalg.cross(side, tangent)
            normal = torch.cos(theta)[:, None, None, None] * side + torch.sin(theta)[:, None, None, None] * up   # (R, I, S, 3)
            centre = (cur + nxt) / 2
            surface = centre[None] + 0.55 * self.r * normal                       # ring of Gaussians round the yarn
            P = self.to_splat(surface)
            # Long axis along the yarn (its direction in the splat's frame), the other two its thickness.
            tangent_s = self.to_splat(centre + tangent) - self.to_splat(centre)
            tangent_s = (tangent_s / tangent_s.norm(dim=-1, keepdim=True)).expand_as(P)
            q = torch.stack([1 + tangent_s[..., 0], torch.zeros_like(tangent_s[..., 0]), -tangent_s[..., 2], tangent_s[..., 1]], -1)
            q = torch.where((1 + tangent_s[..., :1]) < 1e-4, torch.tensor([0.0, 0.0, 1.0, 0.0], device=DEVICE), q)
            across = (0.6 * self.r / self.mm).expand(P.shape[:-1] + (1,))
            scale = torch.cat([(length / self.mm * 0.75).expand(P.shape[:-1] + (1,)), across, across], -1)
            shade = self.ambient + self.diffuse * torch.relu((normal * light).sum(-1, keepdim=True))
            shade = shade * torch.exp(self.depth_k.clamp(0.0, 3.0) * depth.clamp(-3.0, 0.0))[None, ..., None]
            colour = (self.albedo[self.part_index[part]] * shade).clamp(0, 1)
            means.append(P.reshape(-1, 3))
            quats.append((q / q.norm(dim=-1, keepdim=True)).reshape(-1, 4))
            scales.append(scale.reshape(-1, 3))
            colours.append(colour.reshape(-1, 3))
        sheet = self.backing_sheet()
        if sheet is not None:
            for target, value in zip((means, quats, scales, colours), sheet):
                target.append(value)
        means = torch.cat(means)
        return means, torch.cat(quats), torch.cat(scales), torch.sigmoid(self.opacity_logit).expand(len(means)), torch.cat(colours)

    def backing_sheet(self):
        """Flat Gaussians behind every post (BACKING_MM back): means, quats, scales, colours."""
        posts = [self.instances[k] for k in self.instances if k[0] == "post"]
        if not posts:
            return None
        origins = torch.cat(posts)                                              # (P, 2) column, row
        su = (torch.arange(4, device=DEVICE, dtype=torch.float32) + 0.5) / 4 - 0.5
        sv = (torch.arange(8, device=DEVICE, dtype=torch.float32) + 0.5) / 8
        U, V = torch.meshgrid(su, sv, indexing="ij")
        c, s = torch.cos(self.angle), torch.sin(self.angle)
        u = (origins[:, None, 0] + U.reshape(1, -1)) * self.w + self.shift[0]
        v = (origins[:, None, 1] + V.reshape(1, -1)) * self.h + self.shift[1]
        x, y = u * c - v * s, u * s + v * c
        slide = MAX_STRETCH_MM * torch.tanh(self.sample(self.warp, x, y) / MAX_STRETCH_MM)
        x, y = x + slide[0], y + slide[1]
        z = self.bow(x, y) - BACKING_MM
        P = self.to_splat(torch.stack([x, y, z], -1)).reshape(-1, 3)
        normal = self.to_splat(torch.stack([x, y, z + 1], -1)).reshape(-1, 3) - P
        normal = normal / normal.norm(dim=-1, keepdim=True)
        # Rotate z onto the cloth's normal: the sheet's thin axis.
        zaxis = torch.tensor([0.0, 0.0, 1.0], device=DEVICE).expand_as(normal)
        q = torch.cat([1 + (zaxis * normal).sum(-1, keepdim=True), torch.linalg.cross(zaxis, normal)], -1)
        q = q / q.norm(dim=-1, keepdim=True)
        side = 0.8 * torch.maximum(self.w / 4, self.h / 8)            # round: the sheet's turn in its plane is arbitrary
        size = torch.stack([side, side, torch.tensor(0.05, device=DEVICE)]) / self.mm
        scales = size.expand(len(P), 3)
        colours = torch.sigmoid(self.backing).expand(len(P), 3)
        return P, q, scales, colours

    def regulariser(self):
        smooth, drift = 0.0, 0.0
        for index, curve in enumerate(self.curves):
            coefficients = self.coefficients[index]
            terms = curve["terms"]
            k = torch.arange(1, terms + 1, device=DEVICE, dtype=torch.float32)
            if curve["closed"]:
                weight = torch.cat([torch.zeros(1, device=DEVICE), k.repeat_interleave(2)]) ** 2
            else:
                weight = torch.cat([torch.zeros(2, device=DEVICE), k]) ** 2
            smooth = smooth + (weight[:, None] * coefficients ** 2).sum()
            drift = drift + ((coefficients - self.start_coefficients[index]) ** 2).sum()
        f = self.field[0, 0]
        bend = ((f[2:, 1:-1] + f[:-2, 1:-1] + f[1:-1, 2:] + f[1:-1, :-2] - 4 * f[1:-1, 1:-1]) ** 2).mean() if min(f.shape) > 2 else 0.0
        w = self.warp[0]
        stretch = (((w[:, 2:, 1:-1] + w[:, :-2, 1:-1] + w[:, 1:-1, 2:] + w[:, 1:-1, :-2] - 4 * w[:, 1:-1, 1:-1]) ** 2).mean()
                   if min(w.shape[1:]) > 2 else 0.0) + 1e-3 * (w ** 2).mean()
        slips = sum((sl ** 2).mean() for sl in self.slips) / max(len(self.slips), 1)
        return 1e-4 * smooth + 1e-3 * drift + 1e-2 * bend + 1e-1 * stretch + 1e-2 * slips


def stitch_instances(scene, start: crochet_fit.CrochetFit):
    """Every stitch over the cloth: (column, row) per (kind, parity), from the repeat's grid tiled out."""
    cells, p = start.cells, start.params
    cloth, low = ndimage.binary_dilation(scene["cloth"], iterations=10), scene["low"]
    rows_px, cols_px = np.nonzero(cloth)
    x, y = low[0] + cols_px * 0.1, low[1] + rows_px * 0.1
    a = math.radians(start.angle_deg)
    u, v = x * math.cos(a) + y * math.sin(a) - start.shift[0], -x * math.sin(a) + y * math.cos(a) - start.shift[1]
    c0, c1 = int(math.floor(u.min() / p.stitch_width)) - 1, int(math.ceil(u.max() / p.stitch_width)) + 1
    r0, r1 = int(math.floor(v.min() / p.row_height)) - 1, int(math.ceil(v.max() / p.row_height)) + 1
    holes = crochet_fit.opening_mask(scene["extraction"].maps["support"])
    instances, grid = {}, {}
    for row in range(r0, r1):
        for column in range(c0, c1):
            uc, vc = column * p.stitch_width + start.shift[0], (row + 0.5) * p.row_height + start.shift[1]
            xc, yc = uc * math.cos(a) - vc * math.sin(a), uc * math.sin(a) + vc * math.cos(a)
            col_px, row_px = int((xc - low[0]) / 0.1), int((yc - low[1]) / 0.1)
            if not (0 <= row_px < cloth.shape[0] and 0 <= col_px < cloth.shape[1] and cloth[row_px, col_px]):
                continue
            post = opening_share(holes, low, a, start.shift, p, column, row) < OPEN_SHARE
            instances.setdefault(("post" if post else "chain", row % 2), []).append((column, row))
            grid[(row, column)] = 1 if post else 0
    scene["grid"] = grid
    return instances


# A stitch is a chain across an opening when the scan's openings cover this share of its place.
OPEN_SHARE = 0.4
# ... or, read from the photos, when this share of the looks at its place see the table, not yarn.
PHOTO_OPEN_SHARE = 0.35


def photo_openings(scene, start, instances, cameras, lens, photos, table, views):
    """Each stitch's place looked at in every picture that sees it: an opening where the table's colour
    shows more than the yarn's (module docstring). Returns new instances and, per stitch, its share."""
    p, a, low, bow = start.params, math.radians(start.angle_deg), scene["low"], scene["bow"]
    ex, flat_centre, flat_axes, mm = scene["extraction"], scene["flat_centre"], scene["flat_axes"], scene["mm"]
    stitches = [(kind, parity, column, row) for (kind, parity), places in instances.items() for column, row in places]
    du = np.linspace(-0.35, 0.35, 4) * p.stitch_width
    dv = np.linspace(0.2, 0.75, 4) * p.row_height
    DU, DV = [g.ravel() for g in np.meshgrid(du, dv)]
    points = []
    for _, _, column, row in stitches:
        u = column * p.stitch_width + start.shift[0] + DU
        v = row * p.row_height + start.shift[1] + DV
        x, y = u * math.cos(a) - v * math.sin(a), u * math.sin(a) + v * math.cos(a)
        cols = np.clip(((x - low[0]) / 0.1).astype(int), 0, bow.shape[1] - 1)
        rows = np.clip(((y - low[1]) / 0.1).astype(int), 0, bow.shape[0] - 1)
        points.append(np.stack([x, y, bow[rows, cols]], 1))
    points = np.concatenate(points)                                             # cloth-frame mm
    splat = ((points @ ex.axes + ex.centre) / mm) @ flat_axes + flat_centre     # the splat's (cameras') frame
    K = intrinsics(lens)
    yarn = np.array([0.85, 0.85, 0.82])
    table = np.asarray(table, dtype=float)
    votes, looks = np.zeros(len(points)), np.zeros(len(points))
    pictures = photos.cpu().numpy()
    for i in views:
        camera = (cameras[i]["viewmat"][:3, :3] @ splat.T + cameras[i]["viewmat"][:3, 3:4]).T
        ahead = camera[:, 2] > 1e-6
        pixel = (K @ camera.T).T
        px, py = pixel[:, 0] / np.maximum(pixel[:, 2], 1e-9), pixel[:, 1] / np.maximum(pixel[:, 2], 1e-9)
        seen = ahead & (px >= 0) & (px < lens["w"] - 1) & (py >= 0) & (py < lens["h"] - 1)
        colour = pictures[i][py[seen].astype(int), px[seen].astype(int)] / 255.0
        to_table = np.linalg.norm(colour - table, axis=1)
        to_yarn = np.linalg.norm(colour - yarn, axis=1)
        votes[seen] += to_table < to_yarn
        looks[seen] += 1
    per_point = votes / np.maximum(looks, 1)
    share = per_point.reshape(len(stitches), -1).mean(axis=1)
    rebuilt = {}
    for (kind, parity, column, row), open_share in zip(stitches, share):
        post = open_share < PHOTO_OPEN_SHARE
        rebuilt.setdefault(("post" if post else "chain", parity), []).append((column, row))
        scene["grid"][(row, column)] = 1 if post else 0
    return rebuilt, share


def opening_share(holes, low, angle, shift, p, column: int, row: int) -> float:
    """How much of one stitch's place (stitch width x row height on the lattice) the scan's openings cover."""
    u = (column + np.linspace(-0.4, 0.4, 7)) * p.stitch_width + shift[0]
    v = (row + np.linspace(0.1, 0.9, 7)) * p.row_height + shift[1]
    U, V = np.meshgrid(u, v)
    x, y = U * math.cos(angle) - V * math.sin(angle), U * math.sin(angle) + V * math.cos(angle)
    cols, rows = ((x - low[0]) / 0.1).astype(int), ((y - low[1]) / 0.1).astype(int)
    inside = (rows >= 0) & (rows < holes.shape[0]) & (cols >= 0) & (cols < holes.shape[1])
    return float(holes[rows[inside], cols[inside]].mean()) if inside.any() else 0.0


# ----------------------------------------------------------------------------------- fitting

def psnr(a, b, mask):
    mse = (((a - b) ** 2) * mask).sum() / (mask.sum() * 3).clamp_min(1)
    return float(-10 * torch.log10(mse.clamp_min(1e-10)))


def fit(model, photos, masks, coverage, cameras, lens, train, stages, report_every=50):
    viewmats = torch.tensor(np.stack([c["viewmat"] for c in cameras]), dtype=torch.float32, device=DEVICE)
    W, H = int(lens["w"]), int(lens["h"])
    total = sum(s["steps"] for s in stages)
    done, began = 0, time.time()
    rng = np.random.default_rng(0)
    for stage in stages:
        def parameters(names):
            found = []
            for name in names:
                value = getattr(model, name)
                found += list(value) if isinstance(value, torch.nn.ParameterList) else [value]
            return found

        groups = [{"params": parameters(names), "lr": lr} for names, lr in stage["groups"]]
        optimiser = torch.optim.Adam(groups)
        scale = stage["scale"]
        Ks = torch.tensor(intrinsics(lens, scale), dtype=torch.float32, device=DEVICE)
        w, h = int(W * scale), int(H * scale)
        for step in range(stage["steps"]):
            pick = rng.choice(train, BATCH, replace=False)
            target = photos[pick].float() / 255.0
            mask = masks[pick][..., None]
            silhouette = coverage[pick][..., None]
            if scale != 1.0:
                silhouette = torch.nn.functional.interpolate(silhouette.permute(0, 3, 1, 2), size=(h, w), mode="area").permute(0, 2, 3, 1)
                target = torch.nn.functional.interpolate(target.permute(0, 3, 1, 2), size=(h, w), mode="area").permute(0, 2, 3, 1)
                mask = torch.nn.functional.interpolate(mask.permute(0, 3, 1, 2), size=(h, w), mode="area").permute(0, 2, 3, 1)
            means, quats, scales_, opacities, colours = model.gaussians()
            image, alpha = render(means, quats, scales_, opacities, colours, None, viewmats[pick], Ks.expand(BATCH, 3, 3), w, h,
                                  model.background.expand(BATCH, 3))
            l1 = ((image - target).abs() * mask).sum() / (mask.sum() * 3).clamp_min(1)
            d_ssim = 1 - ssim((image * mask).permute(0, 3, 1, 2), (target * mask).permute(0, 3, 1, 2), data_range=1.0, size_average=True)
            loss = 0.8 * l1 + 0.2 * d_ssim + SILHOUETTE_WEIGHT * (alpha - silhouette).abs().mean() + model.regulariser()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            done += 1
            if done % report_every == 0 or done == total:
                log(f"step {done}/{total} ({stage['name']}): loss {float(loss):.4f}, PSNR {psnr(image.detach(), target, mask):.2f} dB, "
                    f"w {float(model.w):.3f} h {float(model.h):.3f} r {float(model.r):.3f} mm, {time.time() - began:.0f} s")


def evaluate(model, photos, masks, cameras, lens, views, splat=None):
    """Held-out pictures: the equation model's PSNR/SSIM in the cloth's outline, and the free splat's."""
    viewmats = torch.tensor(np.stack([c["viewmat"] for c in cameras]), dtype=torch.float32, device=DEVICE)
    K = torch.tensor(intrinsics(lens), dtype=torch.float32, device=DEVICE)
    W, H = int(lens["w"]), int(lens["h"])
    scores, pairs = {"model_psnr": [], "model_ssim": [], "splat_psnr": []}, []
    with torch.no_grad():
        means, quats, scales, opacities, colours = model.gaussians()
        for i in views:
            target, mask = photos[i:i + 1].float() / 255.0, masks[i:i + 1][..., None]
            image, _ = render(means, quats, scales, opacities, colours, None, viewmats[i:i + 1], K[None], W, H, model.background[None])
            scores["model_psnr"].append(psnr(image, target, mask))
            scores["model_ssim"].append(float(ssim((image * mask).permute(0, 3, 1, 2), (target * mask).permute(0, 3, 1, 2), data_range=1.0)))
            if splat is not None:
                free, _ = render(*splat[:5], splat[5], viewmats[i:i + 1], K[None], W, H, torch.zeros(1, 3, device=DEVICE))
                scores["splat_psnr"].append(psnr(free, target, mask))
            pairs.append((i, image[0].cpu().numpy(), target[0].cpu().numpy(), mask[0, ..., 0].cpu().numpy()))
    return {k: float(np.mean(v)) for k, v in scores.items() if v}, pairs


STAGES = [
    {"name": "lattice and bow", "scale": 0.5, "steps": 300,
     "groups": [(("angle",), 2e-3), (("shift",), 3e-2), (("log_w", "log_h"), 3e-3), (("field", "warp"), 3e-2),
                (("albedo", "light", "ambient", "diffuse", "opacity_logit", "depth_k", "backing"), 2e-2)]},
    {"name": "stitch shape", "scale": 1.0, "steps": 600,
     "groups": [(("coefficients",), 2e-3), (("log_r",), 5e-3), (("albedo", "light", "ambient", "diffuse", "depth_k", "backing"), 1e-2), (("field", "warp"), 1e-2), (("slips",), 1e-2)]},
    {"name": "everything", "scale": 1.0, "steps": 300,
     "groups": [(("angle",), 5e-4), (("shift",), 1e-2), (("log_w", "log_h", "log_r"), 1e-3), (("coefficients",), 1e-3),
                (("field", "warp"), 5e-3), (("slips",), 5e-3), (("albedo", "light", "ambient", "diffuse", "opacity_logit", "depth_k", "backing"), 5e-3)]},
]


def template_rms(model_a: StitchModel, model_b: StitchModel) -> float:
    """How far two models' stitches lie apart as shapes (mm, RMS of each yarn point's distance to the
    other stitch's nearest yarn, both ways), per stitch kind and row parity. Point by point along t
    would not do: a curve can trace the same yarn with its samples slid along it, and a post's two
    legs can swap."""
    errors = []
    with torch.no_grad():
        for key in {(c["kind"], c["parity"]) for c in model_a.curves}:
            idx = [i for i, c in enumerate(model_a.curves) if (c["kind"], c["parity"]) == key]
            a = torch.cat([model_a.curve_points(i) for i in idx])
            b = torch.cat([model_b.curve_points(i) for i in idx])
            d = torch.cdist(a, b)
            errors += [d.min(1).values ** 2, d.min(0).values ** 2]
    return float(torch.sqrt(torch.cat(errors).mean()))


def cloth_grid(scene) -> dict:
    """The whole cloth's stitches as a grid (rows bottom up): 1 a post, 0 a chain across an opening,
    -1 off the cloth; with the lattice row and column of its corner."""
    cells = scene.get("grid", {})
    if not cells:
        return {}
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    grid = -np.ones((max(rows) - min(rows) + 1, max(cols) - min(cols) + 1), dtype=int)
    for (r, c), value in cells.items():
        grid[r - min(rows), c - min(cols)] = value
    return {"first_row": min(rows), "first_column": min(cols), "grid": grid.tolist()}


def save(model, scene, start, scores, out: Path, pairs):
    out.mkdir(parents=True, exist_ok=True)
    templates = []
    for index, curve in enumerate(model.curves):
        templates.append({"parity": curve["parity"], "kind": curve["kind"], "part": curve["part"], "closed": curve["closed"],
                          **({"slot": [int(curve["key"][1]), int(curve["key"][2])]} if len(curve["key"]) == 3 else {}),
                          "form": "closed: a + sum_k (c_k cos 2 pi k t + d_k sin 2 pi k t)" if curve["closed"]
                                  else "open: a + b t + sum_k c_k sin(k pi t)",
                          "coordinates": "x / stitch_width, y / row_height, z mm",
                          "coefficients": model.coefficients[index].detach().cpu().numpy().T.tolist()})
    data = {"units": "mm", "rows_angle_deg": math.degrees(float(model.angle)), "shift_mm": model.shift.detach().cpu().tolist(),
            "lattice": {"angle_deg": math.degrees(float(model.angle)), "shift": model.shift.detach().cpu().tolist(),
                        "stitch_width": float(model.w), "row_height": float(model.h)},
            "stitch_width": float(model.w), "row_height": float(model.h), "yarn_radius": float(model.r), "wraps": start.params.wraps,
            "twist": start.params.twist, "lean": start.params.lean,
            "stitches": start.cells.tolist(), "templates": templates,
            "cloth_stitches": cloth_grid(scene),
            "yarn_colour": model.albedo.detach().cpu().tolist(), "light": (model.light / model.light.norm()).detach().cpu().tolist(),
            "ambient": float(model.ambient), "diffuse": float(model.diffuse), "table_colour": model.background.detach().cpu().tolist(),
            "bow_grid_mm": {"box": model.field_box, "values": model.field[0, 0].detach().cpu().tolist()},
            "stretch_grid_mm": {"box": model.field_box, "values": model.warp[0].detach().cpu().tolist()},
            "depth_shading_per_mm": float(model.depth_k), "backing_colour": torch.sigmoid(model.backing).detach().cpu().tolist(),
            # Each stitch's own slip off the lattice (mm, along and across the rows, capped as in centrelines),
            # by its curves' key and its place (column, row): with them the curves are rebuilt exactly.
            "slips_mm": [{"key": [k if isinstance(k, str) else int(k) for k in key],
                          "origins": model.instances[key].detach().cpu().round().int().tolist(),
                          "slip": (MAX_SLIP_MM * torch.tanh(model.slips[model.slip_index[key]] / MAX_SLIP_MM)).detach().cpu().tolist()}
                         for key in model.instances],
            "scores": scores}
    (out / "stitch_model.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    (out / "renders").mkdir(exist_ok=True)
    for i, image, target, mask in pairs:
        side = np.hstack([image, target]) * 255
        cv2.imwrite(str(out / "renders" / f"heldout_{i:03d}.png"), cv2.cvtColor(side.astype(np.uint8), cv2.COLOR_RGB2BGR))
    log(f"written to {out}")


def try_counts(args, run, dataset, cameras, lens, train, heldout):
    """A short fit (a quarter of the steps) for each stitches x rows count; the best on the held-out
    pictures wins. What the pictures say, not a hand-tuned score on the splat."""
    photos = masks = openings = coverage = None
    results = []
    for text in args.candidates.split(","):
        stitches, rows = (int(v) for v in text.lower().split("x"))
        log(f"Trying {stitches} stitches x {rows} rows per repeat")
        try:
            scene = prepare(args.output, run, dataset, stitches, rows, args.wraps)
        except Exception as exc:
            log(f"  {text}: could not read the scan this way ({exc})")
            continue
        if photos is None:
            masks, openings, coverage = cloth_masks(scene["ply"], cameras, lens)
            photos = load_photos(cameras, lens)
            table = table_colour(photos, openings)
        instances, _ = photo_openings(scene, scene["start"], stitch_instances(scene, scene["start"]), cameras, lens, photos,
                                      table.cpu().numpy(), train)
        model = StitchModel(scene, instances, lens["w"], lens["h"])
        model.background.copy_(table)
        fit(model, photos, masks, coverage, cameras, lens, train, [{**s, "steps": max(int(s["steps"] * 0.25), 10)} for s in STAGES], report_every=10 ** 9)
        scores, _ = evaluate(model, photos, masks, cameras, lens, heldout)
        log(f"  {text}: held-out {scores['model_psnr']:.2f} dB, SSIM {scores['model_ssim']:.3f}, "
            f"stitch {float(model.w):.2f} mm, row {float(model.h):.2f} mm, r {float(model.r):.2f} mm")
        results.append((scores["model_psnr"], stitches, rows))
        del model
        torch.cuda.empty_cache()
    if not results:
        raise SystemExit("none of the counts could be read")
    best = max(results)
    log(f"Best: {best[1]} stitches x {best[2]} rows ({best[0]:.2f} dB)")
    return best[1], best[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path, help="the trained splat's output folder (holds export/ and model_scale.json)")
    parser.add_argument("--stitches", type=int, required=True, help="stitches per repeat, from one opening to the next along a row")
    parser.add_argument("--rows", type=int, required=True, help="rows per repeat, from one row of openings to the next")
    parser.add_argument("--wraps", type=int, default=1, help="yarn-overs per post: 1 double crochet, 2 treble")
    parser.add_argument("--steps", type=float, default=1.0, help="scale every stage's number of steps (0.3 for a quick trial)")
    parser.add_argument("--synthetic", action="store_true", help="fit to renders of a known model (a self-check), not the photos")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--per-stitch", action="store_true",
                        help="give each stitch of the repeat (rows x stitches) its own shape, not one per row parity")
    parser.add_argument("--stitch", choices=crochet_stitches.FAMILIES, default=None,
                        help="fit a stitch family round each opening instead of filet posts and chains")
    parser.add_argument("--lattice", type=Path, default=None, help="the stitch grid lined up by eye (JSON: angle_deg, shift, stitch_width, row_height)")
    parser.add_argument("--try", dest="candidates", default="",
                        help="stitches x rows counts to compare first, e.g. 3x3,6x3,9x3 (each a short fit, judged on held-out pictures)")
    args = parser.parse_args()

    run, dataset = find_run_and_dataset(args.output)
    cameras, lens, train, heldout = load_cameras(run, dataset)
    log(f"{len(cameras)} pictures ({len(train)} to fit, {len(heldout)} held out), {DEVICE}")
    if args.candidates and not args.synthetic:
        args.stitches, args.rows = try_counts(args, run, dataset, cameras, lens, train, heldout)
    lattice = json.loads(args.lattice.read_text(encoding="utf-8")) if args.lattice else None
    scene = prepare(args.output, run, dataset, args.stitches, args.rows, args.wraps, lattice, args.stitch)
    start = scene["start"]
    instances = family_instances(scene, start) if args.stitch else stitch_instances(scene, start)
    log(f"Start: rows at {start.angle_deg:.1f} deg, stitch {start.params.stitch_width:.2f} mm, row {start.params.row_height:.2f} mm, "
        f"r {start.params.yarn_radius:.2f} mm; {sum(len(v) for v in instances.values())} stitches over the cloth")
    log("Outlining the cloth in every picture")
    masks, openings, coverage = cloth_masks(scene["ply"], cameras, lens)
    if not args.synthetic:
        log("Reading the pictures")
        photos = load_photos(cameras, lens)
        table = table_colour(photos, openings)
    if not args.synthetic and not args.stitch:
        instances, share = photo_openings(scene, start, instances, cameras, lens, photos, table.cpu().numpy(), train)
        log(f"Openings read from the photos: {int((share >= PHOTO_OPEN_SHARE).sum())} of {len(share)} stitches "
            f"(table seen in {np.percentile(share, 50) * 100:.0f}% of looks at a typical stitch, {np.percentile(share, 90) * 100:.0f}% at the 90th)")
    if args.per_stitch:
        # Each stitch of the repeat its own shape: (kind, row, column) of the repeat (rows x stitches).
        slots = {}
        for (kind, _), places in instances.items():
            for column, row in places:
                slots.setdefault((kind, row % args.rows, column % args.stitches), []).append((column, row))
        instances = slots
        log(f"Per stitch: {len(slots)} stitch shapes in the {args.rows} x {args.stitches} repeat")
    model = StitchModel(scene, instances, lens["w"], lens["h"])
    stages = [{**s, "steps": max(int(s["steps"] * args.steps), 10)} for s in STAGES]
    if lattice is not None:
        # Lined up by eye: the grid stays where it was put; the stitch's shape, the yarn and the bow are fitted.
        stages = [{**s, "groups": [(names, lr) for names, lr in s["groups"] if not set(names) & {"angle", "shift", "log_w", "log_h"}]}
                  for s in stages]
    splat = splat_tensors(scene["ply"])

    if args.synthetic:
        # The truth: the start, stretched and with its posts reshaped (twist 1.25 instead of 0.75).
        truth = StitchModel(scene, instances, lens["w"], lens["h"])
        with torch.no_grad():
            truth.log_w += math.log(1.05)
            truth.log_h += math.log(0.96)
            changed = crochet_core.CrochetParams(**{**start.params.as_dict(), "twist": 1.25})
            hand = {(c["parity"], c["kind"], c["part"]): [] for c in truth.curves}
            for parity in (0, 1):
                for kind, parts in hand_template(changed, parity).items():
                    for points, closed, part in parts:
                        hand[(parity, kind, part)].append(points / np.array([truth.w0, truth.h0, 1.0]))
            seen = {}
            for index, curve in enumerate(truth.curves):
                key = (curve["parity"], curve["kind"], curve["part"])
                points = hand[key][seen.get(key, 0)]
                seen[key] = seen.get(key, 0) + 1
                B = basis(len(points), curve["closed"], curve["terms"])
                truth.coefficients[index].copy_(torch.tensor(np.linalg.lstsq(B, points, rcond=None)[0], dtype=torch.float32))
            viewmats = torch.tensor(np.stack([c["viewmat"] for c in cameras]), dtype=torch.float32, device=DEVICE)
            K = torch.tensor(intrinsics(lens), dtype=torch.float32, device=DEVICE)
            g = truth.gaussians()
            drawn = [render(*g, None, viewmats[i:i + 1], K[None], int(lens["w"]), int(lens["h"]), truth.background[None]) for i in range(len(cameras))]
            photos = torch.stack([(image[0] * 255).to(torch.uint8) for image, _ in drawn])
            coverage = torch.stack([alpha[0, ..., 0] for _, alpha in drawn])      # the truth's own silhouette
        log(f"Synthetic truth: w {float(truth.w):.3f} h {float(truth.h):.3f}; start template off by {template_rms(model, truth):.3f} mm")
    else:
        model.background.copy_(table)
        log(f"The table through the openings: RGB {[round(float(c), 3) for c in model.background]}")

    before, _ = evaluate(model, photos, masks, cameras, lens, heldout, None if args.synthetic else splat)
    log(f"Before fitting, held out: equation model {before['model_psnr']:.2f} dB" +
        (f", free splat {before['splat_psnr']:.2f} dB" if "splat_psnr" in before else ""))
    fit(model, photos, masks, coverage, cameras, lens, train, stages)
    scores, pairs = evaluate(model, photos, masks, cameras, lens, heldout, None if args.synthetic else splat)
    scores["before_model_psnr"] = before["model_psnr"]
    log(f"After fitting, held out: equation model {scores['model_psnr']:.2f} dB, SSIM {scores['model_ssim']:.3f}" +
        (f"; free splat {scores['splat_psnr']:.2f} dB" if "splat_psnr" in scores else ""))
    if args.synthetic:
        scores.update({"w_error": float(model.w / truth.w - 1), "h_error": float(model.h / truth.h - 1), "template_rms_mm": template_rms(model, truth)})
        log(f"Recovered: w off by {scores['w_error'] * 100:+.1f}%, h off by {scores['h_error'] * 100:+.1f}%, template off by {scores['template_rms_mm']:.3f} mm")
    out = args.out or (args.output / ("stitch_fit_synthetic" if args.synthetic else f"stitch_fit_{args.stitch}" if args.stitch else "stitch_fit"))
    save(model, scene, start, scores, out, pairs)


if __name__ == "__main__":
    main()
