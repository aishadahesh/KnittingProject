"""Stitch families round the openings: each one's yarn leaves the opening clear and closes the
lattice's other places, and the families differ as their stitches do."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import crochet_stitches  # noqa: E402

W, H, R = 9.6, 8.6, 0.35


@pytest.mark.parametrize("name", crochet_stitches.FAMILIES)
def test_the_openings_stay_open(name):
    points = np.concatenate([p for p, _, _ in crochet_stitches.tile(name, W, H, R, columns=4, rows=4)])
    # The opening of the middle of the patch (column 1, row 2) and its neighbour in the slid row above.
    for middle in ((1 * W, 2 * H), (1.5 * W, 3 * H)):
        nearest = np.linalg.norm(points[:, :2] - np.array(middle), axis=1).min()
        assert nearest > 1.0, f"{name}: yarn {nearest:.2f} mm from an opening's middle"


@pytest.mark.parametrize("name", crochet_stitches.FAMILIES)
def test_the_curves_are_smooth_and_finite(name):
    for points, closed, part in crochet_stitches.motif(name, W, H, R):
        assert np.isfinite(points).all() and points.shape[1] == 3 and len(points) >= 16
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        assert steps.max() < 0.5, f"{name} {part}: a jump of {steps.max():.2f} mm"
        assert np.abs(points[:, 2]).max() < 3 * R


def test_the_families_differ():
    parts = {name: {part for _, _, part in crochet_stitches.motif(name, W, H, R)} for name in crochet_stitches.FAMILIES}
    assert parts["mesh"] == {"chain", "knot"}
    assert "post" in parts["trellis"] and "chain" in parts["trellis"]
    assert parts["vstitch"] == {"post", "top"} and parts["shell"] == {"post", "top"}
    posts = lambda name: sum(part == "post" for _, _, part in crochet_stitches.motif(name, W, H, R))
    assert posts("shell") > posts("vstitch") > posts("trellis")
    with pytest.raises(ValueError):
        crochet_stitches.motif("filet", W, H, R)
