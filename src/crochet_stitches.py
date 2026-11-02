"""Crochet lace stitch families as equations: the yarn of one repeat round one opening, for lace whose
openings sit in a diamond (staggered) lattice - each row of openings slid half a repeat along from
the one before.

The repeat is the openings' lattice: W mm from one opening to the next along a row, H mm from one row
of openings to the next (the rows slid W/2). Coordinates are mm round one opening's middle: u along
the rows, v up the fabric, z out of it toward the viewer. A family is a few pieces of yarn per
opening, each a closed form:

  twisted strands   a post's two legs, or a chain's two plies, wound round a line from p0 to p1:
                      c(s) = p0 + s (p1 - p0) + b sin(pi s) n,   s in [0, 1]
                      leg k = c(s) + r (cos(2 pi tau s + k pi) e + sin(2 pi tau s + k pi) z)    (k = 0, 1)
                    (n bows the line sideways by b, e is across it in the fabric's plane, tau turns)
  flat loop         a stitch's top (or a single crochet's knot), lying in the fabric:
                      p(t) = m + A cos t a + B sin t a_perp + C sin t z

  mesh      diamond net: chains along the diamond's edges round each opening, joined by single
            crochet knots where four chains meet. Long, thin, loose sides.
  trellis   the same diamonds with longer chains bowed into arches, each joined at the arch's top by
            a short post - taller openings with upright anchors.
  vstitch   each opening inside a V: two posts from one foot spread up to the row's top, the tops
            braided along the row. Thick, upright sides.
  shell     between the openings, fans of five posts from one foot with their tops braided round
            the fan's edge; each row's fans over the last row's openings. Solid wedges, small gaps.

`motif(name, W, H, r)` gives one opening's share of the yarn: [(points (n, 3) mm, closed, part)].
Tiled at every opening (scripts/fit_stitch_model.py --stitch), it is the whole lace.
"""

from __future__ import annotations

import math

import numpy as np

FAMILIES = ("mesh", "trellis", "vstitch", "shell")
SAMPLES_PER_MM = 12


def _count(length_mm: float) -> int:
    return max(int(length_mm * SAMPLES_PER_MM), 16)


def twisted(p0, p1, r: float, turns: float, bow: float = 0.0) -> list[np.ndarray]:
    """Two strands wound round the line p0 -> p1 (in the fabric's plane), bowed sideways by `bow` mm."""
    p0, p1 = np.array([*p0, 0.0][:3], dtype=float), np.array([*p1, 0.0][:3], dtype=float)
    along = p1 - p0
    length = float(np.linalg.norm(along))
    s = np.linspace(0.0, 1.0, _count(length))
    e = np.array([-along[1], along[0], 0.0]) / max(length, 1e-9)        # across the line, in the fabric
    z = np.array([0.0, 0.0, 1.0])
    centre = p0 + s[:, None] * along + (bow * np.sin(math.pi * s))[:, None] * e
    legs = []
    for k in (0, 1):
        angle = 2 * math.pi * turns * s + k * math.pi
        legs.append(centre + r * (np.cos(angle)[:, None] * e + np.sin(angle)[:, None] * z))
    return legs


def flat_loop(middle, half_width: float, half_height: float, depth: float, angle: float = 0.0) -> np.ndarray:
    """A closed loop lying in the fabric round `middle`, its long axis at `angle` (radians)."""
    t = np.linspace(0.0, 2 * math.pi, _count(4 * max(half_width, half_height)), endpoint=False)
    a = np.array([math.cos(angle), math.sin(angle), 0.0])
    b = np.array([-math.sin(angle), math.cos(angle), 0.0])
    m = np.array([middle[0], middle[1], 0.0])
    return (m + (half_width * np.cos(t))[:, None] * a + (half_height * np.sin(t))[:, None] * b
            + (depth * np.sin(t))[:, None] * np.array([0.0, 0.0, 1.0]))


def chain_turns(length_mm: float, r: float) -> float:
    """A chain's plies twist once round each other per chain stitch, about six yarn radii long; the
    Fourier templates hold up to about four turns."""
    return float(min(length_mm / max(6.0 * r, 1e-6), 4.0))


def _mesh(W: float, H: float, r: float, arch: float, post: bool) -> list:
    # The diamond round the opening at (0, 0): corners (+-W/2, 0) and (0, +-H). Each opening owns its
    # two upper sides and the top corner; the lower ones are the openings' below.
    top = (0.0, H)
    pieces = []
    foot = (0.0, H - 0.3 * H) if post else top
    for side in (-1, 1):
        corner = (side * W / 2, 0.0)
        length = math.dist(corner, foot)
        bow = -side * arch * length            # bowed out, away from the opening
        for strand in twisted(corner, foot, r, chain_turns(length, r), bow):
            pieces.append((strand, False, "chain"))
    if post:
        for leg in twisted(foot, (0.0, H + 0.1 * H), r, 0.75):
            pieces.append((leg, False, "post"))
    pieces.append((flat_loop(top if not post else (0.0, H + 0.1 * H), 2.2 * r, 1.4 * r, 1.2 * r), True, "knot"))
    return pieces


def _vstitch(W: float, H: float, r: float) -> list:
    foot, pieces = (0.0, -H / 2), []
    for side in (-1, 1):
        for leg in twisted(foot, (side * 0.3 * W, H / 2), r, 0.75):
            pieces.append((leg, False, "post"))
    # The row's top: the posts' V tops and the chains between the Vs, braided along the row.
    pieces.append((flat_loop((0.0, H / 2), 0.52 * W, 0.6 * r, 1.3 * r), True, "top"))
    return pieces


def _shell(W: float, H: float, r: float) -> list:
    # The fan beside the opening (at u = W/2), its foot on the row of openings' lower edge.
    foot, pieces = np.array([W / 2, -H / 2]), []
    tops = []
    for angle in np.radians([-60.0, -30.0, 0.0, 30.0, 60.0]):
        top = foot + np.array([0.38 * W * math.sin(angle), H * math.cos(angle)])
        tops.append(top)
        for leg in twisted(foot, top, r, 0.75):
            pieces.append((leg, False, "post"))
    # The tops braided round the fan's edge: an arc through them.
    t = np.linspace(math.radians(-70.0), math.radians(70.0), _count(1.6 * H))
    edge = np.stack([foot[0] + 0.38 * W * np.sin(t), foot[1] + H * np.cos(t), np.zeros_like(t)], axis=1)
    for k in (0, 1):
        turn = 2 * math.pi * chain_turns(1.6 * H, r) * np.linspace(0.0, 1.0, len(t)) + k * math.pi
        normal = np.stack([np.sin(t), np.cos(t) * 0.38 * W / H, np.zeros_like(t)], axis=1)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        strand = edge + r * (np.cos(turn)[:, None] * normal + np.sin(turn)[:, None] * np.array([0.0, 0.0, 1.0]))
        pieces.append((strand, False, "top"))
    return pieces


def motif(name: str, W: float, H: float, r: float) -> list:
    """One opening's share of the yarn for a stitch family (module docstring): [(points mm, closed, part)]."""
    if name == "mesh":
        return _mesh(W, H, r, arch=0.06, post=False)
    if name == "trellis":
        return _mesh(W, H, r, arch=0.16, post=True)
    if name == "vstitch":
        return _vstitch(W, H, r)
    if name == "shell":
        return _shell(W, H, r)
    raise ValueError(f"unknown stitch family {name!r}; one of {', '.join(FAMILIES)}")


def tile(name: str, W: float, H: float, r: float, columns: int = 3, rows: int = 2) -> list:
    """A patch of the lace: the motif at every opening of a columns x rows diamond lattice."""
    curves = []
    for row in range(rows):
        for column in range(columns):
            offset = np.array([(column + 0.5 * (row % 2)) * W, row * H, 0.0])
            curves += [(points + offset, closed, part) for points, closed, part in motif(name, W, H, r)]
    return curves
