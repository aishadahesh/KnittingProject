"""Crochet lace as equations: a grid of stitches, each a post or a chain across, built from closed curves.

Filet crochet is worked in rows of tall stitches (double crochet, or treble for taller rows). Seen
close, every stitch is the same few pieces of yarn, and each has a closed form (mm; x along the row,
y up the fabric, z out of it toward the viewer; s from 0 to 1 up a post, t from 0 to 2 pi round a loop):

  post legs     the loop pulled up through the row below, two strands twisted round each other:
                  x = x_i + R cos(2 pi tau s + k pi),  y = y_row + s H,  z = R sin(2 pi tau s + k pi)   (k = 0, 1)
  yarn-over     each wrap of yarn round the post, worked off as the stitch is finished - one for
  wraps         double crochet, two for treble - a helix round the legs over the post's middle:
                  x = x_i + W cos(2 pi s'),  y = y_row + (a + b s') H,  z = W sin(2 pi s')
  V top         the two loops left on top, lying flat along the row and reaching to the next stitch,
                  so the tops chain into the braid along each row's top edge:
                  x = x_i + A cos t,  y = y_top + B sin t,  z = r + C cos t
  chain         where a stitch is left out, a chain stitch across the top instead - the same flat loop
                as a V top, with nothing under it: an opening. Filet crochet's open mesh is every other
                stitch left out ("dc, ch 1, skip 1"), its wider bars several in a row.

The stitch grid is rows x stitches, row 0 at the bottom: 1 = a post, 0 = a chain across (open).
Rows alternate direction as they are worked, so their twist turns the other way.

A learned stitch (scripts/fit_stitch_model.py) replaces the hand-made curves above with `templates`:
each part of a post or a chain, per row parity (or, fitted with --per-stitch, per place in the repeat:
its "slot", [row, column] of the stitch grid), a Fourier series in stitch-local coordinates
(x / stitch_width, y / row_height, z mm), fitted to the photos:
    open curve:    p(t) = a + b t + sum_k c_k sin(k pi t)
    closed curve:  p(t) = a + sum_k (c_k cos 2 pi k t + d_k sin 2 pi k t)
Stitch width, row height, yarn radius and lean still apply to it; twist and yarn-overs are in the
learned curves themselves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

SAMPLES_PER_MM = 24
TEMPLATE_SAMPLES = 96


@dataclass
class CrochetParams:
    stitch_width: float = 1.0      # mm between neighbouring posts
    row_height: float = 2.2        # mm, one row of posts
    wraps: int = 1                 # yarn-overs per post: 1 double crochet, 2 treble
    yarn_radius: float = 0.3       # mm
    twist: float = 0.75            # turns the two legs make round each other up one post
    lean: float = 0.0              # mm the top of a post sits sideways from its foot
    # A learned stitch, as fitted to photos: [{"kind": "post"|"chain", "parity": 0|1, "part", "closed",
    # "coefficients": [[x terms], [y terms], [z terms]]}, ...] - or None for the hand-made curves.
    templates: list | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def template_basis(samples: int, closed: bool, count: int) -> np.ndarray:
    """The Fourier basis a template's coefficients multiply: `count` columns (see the module docstring)."""
    t = np.linspace(0.0, 1.0, samples, endpoint=not closed)
    if closed:
        columns = [np.ones_like(t)]
        for k in range(1, (count - 1) // 2 + 1):
            columns += [np.cos(2 * math.pi * k * t), np.sin(2 * math.pi * k * t)]
    else:
        columns = [np.ones_like(t), t] + [np.sin(math.pi * k * t) for k in range(1, count - 1)]
    return np.stack(columns, axis=1)


def hand_templates(p: CrochetParams, open_terms: int = 10, closed_terms: int = 8) -> list:
    """The hand-made stitch (posts, wraps, tops) as templates: each curve's Fourier coefficients by
    least squares - the starting point a fit to photos learns from."""
    templates = []
    for parity in (0, 1):
        direction = 1 if parity == 0 else -1
        top = (top_loop(0.0, p.row_height, p, direction), True, "top")
        parts = {"post": [(leg, False, "leg") for leg in post_legs(0.0, 0.0, p, direction)]
                         + [(wrap, False, "wrap") for wrap in post_wraps(0.0, 0.0, p, direction)] + [top],
                 "chain": [top]}
        for kind, curves in parts.items():
            for points, closed, part in curves:
                scaled = points / np.array([p.stitch_width, p.row_height, 1.0])
                count = 1 + 2 * closed_terms if closed else 2 + open_terms
                coefficients = np.linalg.lstsq(template_basis(len(scaled), closed, count), scaled, rcond=None)[0]
                templates.append({"kind": kind, "parity": parity, "part": part, "closed": closed,
                                  "coefficients": coefficients.T.tolist()})
    return templates


def template_slots(p: CrochetParams) -> tuple[int, int] | None:
    """The repeat (rows, columns) a per-stitch learned stitch was fitted on, or None for one shape per parity."""
    slots = [t["slot"] for t in p.templates or [] if "slot" in t]
    return (max(r for r, _ in slots) + 1, max(c for _, c in slots) + 1) if slots else None


def template_curves(p: CrochetParams, kind: str, parity: int, slot=None) -> list:
    """A learned stitch's curves for one kind and row parity (and, per stitch, its place in the repeat:
    `slot` (row, column), or the first place with this kind when that one was never seen as it),
    stitch-local mm: [(points, closed)]."""
    curves = []
    chosen = [t for t in p.templates or [] if t["kind"] == kind and int(t["parity"]) == parity]
    if slot is not None and any("slot" in t for t in chosen):
        here = [t for t in chosen if tuple(t.get("slot", ())) == tuple(slot)]
        first = min((tuple(t["slot"]) for t in chosen if "slot" in t), default=None)
        chosen = here or [t for t in chosen if tuple(t.get("slot", ())) == first]
    for template in chosen:
        coefficients = np.asarray(template["coefficients"], dtype=float)          # 3 x terms
        points = template_basis(TEMPLATE_SAMPLES, bool(template["closed"]), coefficients.shape[1]) @ coefficients.T
        points = points * np.array([p.stitch_width, p.row_height, 1.0])
        points[:, 0] += p.lean * points[:, 1] / max(p.row_height, 1e-9)
        curves.append((points, bool(template["closed"])))
    return curves


def _samples(length_mm: float) -> int:
    return max(int(length_mm * SAMPLES_PER_MM), 12)


def post_legs(x: float, y0: float, p: CrochetParams, direction: int) -> list[np.ndarray]:
    """The two twisted strands of one post, from the row below's top up to this row's."""
    count = _samples(p.row_height * 1.5)
    s = np.linspace(0.0, 1.0, count)
    radius = p.yarn_radius
    legs = []
    for k in (0, 1):
        angle = direction * 2 * math.pi * p.twist * s + k * math.pi
        legs.append(np.stack([x + radius * np.cos(angle) + p.lean * s, y0 + s * p.row_height, radius * np.sin(angle)], axis=1))
    return legs


def post_wraps(x: float, y0: float, p: CrochetParams, direction: int) -> list[np.ndarray]:
    """The yarn-over wraps round a post, spread over its middle (one per yarn-over)."""
    if p.wraps <= 0:
        return []
    count = _samples(p.row_height * 3)
    u = np.linspace(0.0, 1.0, count)
    radius = 2.0 * p.yarn_radius
    start, end = 0.3, 0.75                     # the wraps sit between a third and three quarters up
    turns = float(p.wraps)
    s = start + (end - start) * u
    angle = direction * 2 * math.pi * turns * u + math.pi / 2
    return [np.stack([x + radius * np.cos(angle) + p.lean * s, y0 + s * p.row_height, radius * np.sin(angle)], axis=1)]


def top_loop(x: float, y_top: float, p: CrochetParams, direction: int = 1) -> np.ndarray:
    """A stitch's V top (or a chain stitch): a closed loop lying along the row's top edge - its front
    strand across the face of the fabric, its back strand behind - reaching to the next stitch, so the
    tops overlap into the braid along every row. It dips at its far end, under its neighbour's loop."""
    t = np.linspace(0.0, 2 * math.pi, _samples(p.stitch_width * 4), endpoint=False)
    a, depth = 0.62 * p.stitch_width, 1.3 * p.yarn_radius
    return np.stack([x + a * np.cos(t),
                     y_top + 0.35 * p.yarn_radius * direction * np.cos(t),
                     depth * np.sin(t)], axis=1)


def build(stitches: np.ndarray, p: CrochetParams, with_rows: bool = False):
    """Every curve of the fabric: a list of (points mm, closed) - each row's posts, wraps, V tops and
    chains. `stitches` is rows x stitches, row 0 at the bottom; 1 a post, 0 a chain across (open).
    `with_rows` adds each curve's row: (points, closed, row)."""
    stitches = np.asarray(stitches, dtype=float)
    curves = []
    for row in range(stitches.shape[0]):
        direction = 1 if row % 2 == 0 else -1
        y0, y_top = row * p.row_height, (row + 1) * p.row_height
        slots = template_slots(p) if p.templates else None
        for column in range(stitches.shape[1]):
            x = column * p.stitch_width
            if p.templates:
                kind = "post" if stitches[row, column] > 0.5 else "chain"
                slot = (row % slots[0], column % slots[1]) if slots else None
                for points, closed in template_curves(p, kind, row % 2, slot):
                    curves.append((points + np.array([x, y0, 0.0]), closed, row))
                continue
            if stitches[row, column] > 0.5:
                for leg in post_legs(x, y0, p, direction):
                    curves.append((leg, False, row))
                for wrap in post_wraps(x, y0, p, direction):
                    curves.append((wrap, False, row))
            # The V top, or the chain stitch across an opening: the same flat loop at the row's top.
            curves.append((top_loop(x + p.lean, y_top, p, direction), True, row))
    return curves if with_rows else [(points, closed) for points, closed, _ in curves]


def ring_meshes(stitches: np.ndarray, p: CrochetParams, segments: int = 16, copies=(0, 0)):
    """The fabric as Edit Mode draws a model: one tube per curve, as rings of `segments` points round
    it ((vertices, ring count), as knitting_core.build_spline_mesh gives), and each tube's row. `copies`
    repeats the stitch grid that many times more each side, left/right and up/down, as one fabric."""
    stitches = np.asarray(stitches, dtype=float)
    copies_x, copies_y = int(copies[0]), int(copies[1])
    tiled = np.tile(stitches, (2 * copies_y + 1, 2 * copies_x + 1))
    rows = stitches.shape[0]
    shift = np.array([copies_x * stitches.shape[1] * p.stitch_width, copies_y * rows * p.row_height, 0.0])
    angles = np.linspace(0.0, 2 * math.pi, segments, endpoint=False)
    meshes, mesh_rows = [], []
    for points, closed, row in build(tiled, p, with_rows=True):
        points = points - shift
        if closed:
            points = np.vstack([points, points[:1]])        # Edit Mode's tubes are open; close the loop by hand
        tangent = np.gradient(points, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
        reference = np.array([0.0, 0.0, 1.0]) if abs(tangent[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        side = np.cross(tangent[0], reference)
        side /= np.linalg.norm(side)
        sides = []
        for k in range(len(points)):
            side = side - (side @ tangent[k]) * tangent[k]
            side /= max(np.linalg.norm(side), 1e-9)
            sides.append(side)
        sides = np.array(sides)
        ups = np.cross(tangent, sides)
        rings = points[:, None, :] + p.yarn_radius * (np.cos(angles)[None, :, None] * sides[:, None, :]
                                                      + np.sin(angles)[None, :, None] * ups[:, None, :])
        meshes.append((rings.reshape(-1, 3).astype(np.float32), len(points)))
        mesh_rows.append(int(row) % rows)
    return meshes, mesh_rows


def equations(p: CrochetParams) -> list[str]:
    """The stitch's curves as text, with its numbers in (mm)."""
    r, w, h = p.yarn_radius, p.stitch_width, p.row_height
    if p.templates:
        lines = [f"learned stitch: a stitch every {w:.2f} mm, a row {h:.2f} mm, yarn r {r:.2f} mm; x, y in stitch widths and row heights, z in mm"]
        for template in p.templates:
            if int(template["parity"]) != 0:
                continue
            coefficients = np.asarray(template["coefficients"], dtype=float)
            shown = []
            for axis, row in zip("xyz", coefficients):
                if template["closed"]:
                    terms = [f"{row[0]:+.2f}"] + [f"{row[2 * k - 1]:+.2f} cos{2 * k}πt {row[2 * k]:+.2f} sin{2 * k}πt"
                                                  for k in range(1, min(3, (len(row) - 1) // 2) + 1)]
                else:
                    terms = [f"{row[0]:+.2f}", f"{row[1]:+.2f} t"] + [f"{c:+.2f} sin{k}πt" for k, c in enumerate(row[2:4], start=1)]
                shown.append(f"{axis} = " + " ".join(terms) + " ...")
            lines.append(f"{template['kind']} {template['part']} ({len(coefficients[0])} terms): " + ";  ".join(shown))
        return lines
    return [
        f"post legs:  x = xᵢ + {r:.2f} cos(2π·{p.twist:.2f}·s + kπ),  y = y₀ + {h:.2f} s,  z = {r:.2f} sin(2π·{p.twist:.2f}·s + kπ)",
        f"wraps ({p.wraps}): x = xᵢ + {2 * r:.2f} cos(2π·{p.wraps}·u),  y = y₀ + {h:.2f}(0.3 + 0.45u),  z = {2 * r:.2f} sin(2π·{p.wraps}·u)",
        f"V top:      x = xᵢ + {0.62 * w:.2f} cos t,  y = y₀ + {h:.2f} ± {0.35 * r:.2f} cos t,  z = {1.3 * r:.2f} sin t",
        f"a stitch every {w:.2f} mm along the row; a row {h:.2f} mm tall",
    ]
