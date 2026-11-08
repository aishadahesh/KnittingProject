"""Each yarn of a fabric from its fitted crochet model (scripts/fit_stitch_model.py's stitch_model.json).

The photo fit puts every stitch of the cloth in place - its column and row on the lattice, the learned
curves of its posts, wraps, tops and chains, the cloth's stretch and bow, and (when saved) each stitch's
own slip. In crochet one yarn is worked through a whole row of stitches, so the curves of one row are
one yarn. Each of the fabric's yarn Gaussians takes the curve nearest it, and so that curve's yarn.

  fitted_curves   the fitted curves in cloth-frame mm: [(points (n, 3), row, column, kind)]
  yarn_of         each Gaussian's yarn (its curve's row) and curve, and how far it lies from it (mm)

The cloth frame is the one the fit itself used (scan_fit.prepare), so the curves are carried into the
splat's own coordinates the same way (scan_fit.to_splat_frame). No imgui here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

import crochet_core
import scan_fit

MAX_STRETCH_MM = 1.5            # as fit_stitch_model caps its stretch grid
SAMPLES_PER_POINT = 3           # each fitted curve is resampled this much finer for the nearest-curve search


def _grid_sampler(grid: dict):
    """A saved grid over the cloth ({"box": [x0, x1, y0, y1], "values": (channels,) rows x columns}) as
    a function of cloth-frame x, y - bilinear, held at the border as the fit's grid_sample is."""
    x0, x1, y0, y1 = grid["box"]
    values = np.asarray(grid["values"], dtype=float)
    ny, nx = values.shape[-2:]
    xs, ys = np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)
    channels = values.reshape(-1, ny, nx)
    functions = [RegularGridInterpolator((ys, xs), c, bounds_error=False, fill_value=None) for c in channels]

    def sample(x, y):
        where = np.c_[np.clip(y, y0, y1), np.clip(x, x0, x1)]
        return np.stack([f(where) for f in functions])
    return sample


def fitted_curves(fit_path: Path) -> list:
    """Every fitted curve of the cloth in cloth-frame mm: [(points (n, 3), row, column, kind)]."""
    fit = json.loads(Path(fit_path).read_text(encoding="utf-8"))
    params, _ = scan_fit.load_stitch_model(fit_path)
    p = crochet_core.CrochetParams(**params)
    bow, warp = _grid_sampler(fit["bow_grid_mm"]), _grid_sampler(fit["stretch_grid_mm"])
    angle, shift = math.radians(fit["rows_angle_deg"]), fit["shift_mm"]
    slips = {}
    for entry in fit.get("slips_mm", []):
        for (column, row), slip in zip(entry["origins"], entry["slip"]):
            slips[(tuple(entry["key"]), int(column), int(row))] = slip
    cloth = fit["cloth_stitches"]
    grid = np.asarray(cloth["grid"])
    slots = crochet_core.template_slots(p)
    curves = []
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] < 0:
                continue
            row, column = cloth["first_row"] + i, cloth["first_column"] + j
            kind = "post" if grid[i, j] > 0.5 else "chain"
            slot = (row % slots[0], column % slots[1]) if slots else None
            key = (kind, slot[0], slot[1]) if slot is not None else (kind, row % 2)
            du, dv = slips.get((key, column, row), (0.0, 0.0))
            for local, _closed in crochet_core.template_curves(p, kind, row % 2, slot):
                u = column * p.stitch_width + local[:, 0] + shift[0] + du
                v = row * p.row_height + local[:, 1] + shift[1] + dv
                x, y = u * math.cos(angle) - v * math.sin(angle), u * math.sin(angle) + v * math.cos(angle)
                slide = MAX_STRETCH_MM * np.tanh(warp(x, y) / MAX_STRETCH_MM)
                x, y = x + slide[0], y + slide[1]
                z = local[:, 2] + bow(x, y)[0]
                curves.append((np.stack([x, y, z], axis=1), row, column, kind))
    return curves


def yarn_of(positions_splat: np.ndarray, curves: list, prepared: dict, candidates=None):
    """Each Gaussian's yarn - its nearest fitted curve's row, numbered from 0 at the lowest row - its
    curve, and its distance to it (mm). Gaussians not in `candidates` (the table) get -1."""
    points, curve_of = [], []
    for k, (pts, _row, _column, _kind) in enumerate(curves):
        t = np.linspace(0.0, 1.0, len(pts))
        dense = np.linspace(0.0, 1.0, max(2, len(pts) * SAMPLES_PER_POINT))
        fine = np.stack([np.interp(dense, t, pts[:, c]) for c in range(3)], axis=1)
        points.append(scan_fit.to_splat_frame(prepared, fine))
        curve_of.append(np.full(len(fine), k))
    points, curve_of = np.concatenate(points), np.concatenate(curve_of)
    distance, nearest = cKDTree(points).query(np.asarray(positions_splat, dtype=float))
    curve = curve_of[nearest]
    rows = np.array([row for _pts, row, _c, _k in curves])
    yarn = rows[curve] - rows.min()
    if candidates is not None:
        candidates = np.asarray(candidates, dtype=bool)
        yarn = np.where(candidates, yarn, -1)
        curve = np.where(candidates, curve, -1)
    return yarn, curve, distance * float(prepared["mm_per_unit"])
