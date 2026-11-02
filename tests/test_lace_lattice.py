"""The repeating-loop fit: a cloth made from Edit Mode's own loops, with lace openings cut in it,
scanned as a splat would see it, must come back as the numbers it was made with."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import knitting_core  # noqa: E402
import lace_lattice  # noqa: E402
import strand_equations  # noqa: E402

STITCH, ROW, HEIGHT, BULGE = 3.0, 2.0, 3.4, 0.3
TILE = (6, 3)            # rows x stitches per repeat


def _lace_splat(angle_deg: float, seed: int = 0, points: int = 200_000) -> dict:
    """Yarn round Edit Mode's loops (radius 0.35 mm, lit from above), with one round opening per
    repeat, turned by `angle_deg`. Nothing shows through the openings, as in a cropped scan."""
    rng = np.random.default_rng(seed)
    big, _ = knitting_core.anchor_bitmap(np.ones((TILE[0] * 3, TILE[1] * 4), np.float32))
    lines = lace_lattice.row_centrelines(lace_lattice.control_rows(big, STITCH, ROW, HEIGHT, BULGE, 0.2),
                                         4 * TILE[1] * STITCH, 4 * TILE[1])
    centreline = np.vstack(lines)
    period = np.array([TILE[1] * STITCH, TILE[0] * ROW])
    offset = np.mod(centreline[:, :2] - np.array([1.5 * STITCH, 3.0 * ROW]) + period / 2, period) - period / 2
    centreline = centreline[np.linalg.norm(offset, axis=1) > 1.6]
    pick = rng.integers(0, len(centreline), points)
    around, side = rng.uniform(0, np.pi, points), rng.uniform(-1, 1, points)
    surface = centreline[pick] + np.stack([0.35 * np.cos(around) * side, 0.35 * np.cos(around) * np.sqrt(1 - side ** 2),
                                           0.35 * np.sin(around)], axis=1)
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    surface[:, :2] = surface[:, :2] @ np.array([[c, s], [-s, c]])
    colours = (0.45 + 0.5 * np.sin(around))[:, None] * np.array([1.0, 1.0, 0.95])
    return {"positions": surface.astype(np.float32), "colours": colours.astype(np.float32),
            "opacity": np.full(points, 0.9, np.float32), "radius": np.full(points, 0.05, np.float32),
            "sigma": np.full(points, 0.05, np.float32), "count": points}


def _rows_direction(extraction, angle_deg: float) -> float:
    """The made rows' direction in the cloth frame the strand finder chose."""
    made = np.array([math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg)), 0.0])
    local = extraction.axes @ made
    return math.degrees(math.atan2(local[1], local[0])) % 180.0


@pytest.fixture(scope="module", params=[0.0, 20.0])
def fitted(request):
    extraction = strand_equations.extract(_lace_splat(request.param), 1.0, (0.0, 0.0, 1.0))
    return extraction, lace_lattice.fit(extraction), request.param


def test_the_repeat_and_its_stitches_are_found(fitted):
    extraction, fit, angle = fitted
    gap = abs(fit.angle_deg - _rows_direction(extraction, angle)) % 180.0
    assert min(gap, 180.0 - gap) < 2.0
    assert (fit.n_cols, fit.n_rows) == (TILE[1], TILE[0])
    assert abs(fit.stitch_width - STITCH) < 0.15 and abs(fit.row_pitch - ROW) < 0.15
    assert abs(fit.stitch_bulge - BULGE) <= 0.15
    assert abs(fit.loop_height - HEIGHT) < 1.0


def test_the_complete_loops_cover_the_cloth_and_lie_on_it(fitted):
    _, fit, _ = fitted
    assert len(fit.tube["vertices"]) and fit.tube["faces"].max() < len(fit.tube["vertices"])
    assert fit.comparison["rebuilt_to_scan_mean_mm"] < 0.5
    params = fit.edit_parameters()
    assert params["stitch_width"] == fit.stitch_width and params["dy"] == fit.row_pitch
    assert all(f"loop_height_{row + 1}" in params for row in range(fit.n_rows))


def test_counts_set_by_hand_are_kept():
    extraction = strand_equations.extract(_lace_splat(0.0), 1.0, (0.0, 0.0, 1.0))
    fit = lace_lattice.fit(extraction, counts=(6, 3))
    assert (fit.n_cols, fit.n_rows) == (6, 3)
    assert abs(fit.stitch_width - STITCH / 2) < 0.1


def test_rows_run_left_to_right_without_a_jump_back():
    """A row knitted right to left, closed onto the next period, used to run straight back across itself."""
    lines = lace_lattice.row_centrelines(lace_lattice.control_rows(np.ones((3, 3), np.float32), 3.0, 2.0, 3.5, 0.3), 9.0, 3)
    for line in lines:
        assert np.linalg.norm(np.diff(line[:, :2], axis=0), axis=1).max() < 0.5
        assert line[0, 0] < line[-1, 0]
