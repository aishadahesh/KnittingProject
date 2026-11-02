"""Crochet from its equations: what each stitch is made of, an opening where a stitch is left out,
and the meshes Edit Mode draws."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import crochet_core  # noqa: E402

P = crochet_core.CrochetParams(stitch_width=1.0, row_height=2.5, wraps=1, yarn_radius=0.3)


def test_a_post_is_two_legs_a_wrap_per_yarn_over_and_a_top():
    one = crochet_core.build(np.ones((1, 1)), P, with_rows=True)
    assert [closed for _, closed, _ in one] == [False, False, False, True]      # 2 legs, 1 wrap, the V top
    treble = crochet_core.build(np.ones((1, 1)), crochet_core.CrochetParams(wraps=2))
    wrap = treble[2][0]
    turns = np.unwrap(np.arctan2(wrap[:, 2], wrap[:, 0] - wrap[0, 0] + 0.0))
    assert len(treble) == 4                    # still one wrap curve, now going round twice
    legs = one[0][0]
    assert abs(legs[0, 1] - 0.0) < 1e-9 and abs(legs[-1, 1] - P.row_height) < 1e-9     # foot to top of the row


def test_a_stitch_left_out_is_a_chain_across_an_opening():
    stitches = np.array([[1, 0, 1]], dtype=float)
    curves = crochet_core.build(stitches, P, with_rows=True)
    tops = [c for c, closed, _ in curves if closed]
    assert len(tops) == 3                       # V tops and the chain across: every stitch has its top loop
    under_open = [c for c, closed, _ in curves if not closed and abs(np.median(c[:, 0]) - 1.0) < 0.5]
    assert not under_open                       # nothing stands in the open stitch's place


def test_rows_stack_and_tops_join_into_a_braid():
    curves = crochet_core.build(np.ones((2, 4)), P, with_rows=True)
    assert {row for _, _, row in curves} == {0, 1}
    tops = sorted((c for c, closed, row in curves if closed and row == 0), key=lambda c: c[:, 0].mean())
    for left, right in zip(tops, tops[1:]):
        assert left[:, 0].max() > right[:, 0].min()          # each V top reaches over its neighbour's


def test_ring_meshes_are_edit_modes_tubes():
    meshes, rows = crochet_core.ring_meshes(np.array([[1, 0, 1], [1, 1, 1]], dtype=float), P, segments=8, copies=(1, 0))
    assert len(meshes) == len(rows) and set(rows) == {0, 1}
    for vertices, count in meshes:
        assert vertices.shape == (count * 8, 3)
    centre = np.vstack([v for v, _ in meshes]).mean(axis=0)
    assert abs(centre[0] - 1.0) < 1.5           # the copies sit either side of the original


def test_equations_carry_the_numbers():
    text = "\n".join(crochet_core.equations(P))
    assert "2.50" in text and "1.00" in text


def test_a_learned_stitch_rebuilds_the_same_yarn():
    """The hand stitch as Fourier templates, built again from them: the same yarn - typically within half a sample's spacing,
    under 0.01 mm, nowhere more than 0.03 mm (an open curve's ends, where its sine series bends least well)."""
    from scipy.spatial import cKDTree

    stitches = np.array([[1, 0, 1], [1, 1, 1]], dtype=float)
    hand = np.vstack([c for c, _ in crochet_core.build(stitches, P)])
    learned_params = crochet_core.CrochetParams(**{**P.as_dict(), "templates": crochet_core.hand_templates(P)})
    learned = np.vstack([c for c, _ in crochet_core.build(stitches, learned_params)])
    there, _ = cKDTree(hand).query(learned)
    back, _ = cKDTree(learned).query(hand)
    assert max(there.max(), back.max()) < 0.03 and max(np.median(there), np.median(back)) < 0.01
    assert any("terms" in line for line in crochet_core.equations(learned_params))
    stretched = crochet_core.CrochetParams(**{**learned_params.as_dict(), "stitch_width": 2.0})
    wide = np.vstack([c for c, _ in crochet_core.build(stitches, stretched)])
    assert np.ptp(wide[:, 0]) > np.ptp(learned[:, 0]) * 1.6          # the width slider still stretches it


def test_a_per_stitch_learned_stitch_draws_each_place_its_own_shape():
    """Fitted with --per-stitch, every place of the repeat has its own shape: a stitch takes the one of
    its place (row, column mod the repeat), never the others stacked on it."""
    p = crochet_core.CrochetParams(stitch_width=2.0, row_height=4.0)
    line = lambda x: [[x, x], [0.0, 1.0], [0.0, 0.0]]          # open: a + b t, x / w, y / h, z mm
    p.templates = [{"kind": "post", "parity": r % 2, "part": "leg", "closed": False, "slot": [r, c], "coefficients": line(0.1 * (1 + c + 2 * r))}
                   for r in range(2) for c in range(2)]
    assert crochet_core.template_slots(p) == (2, 2)
    curves = crochet_core.build(np.ones((2, 4)), p)
    assert len(curves) == 8                                         # one curve per stitch, not four
    feet = sorted(round(float(points[0, 0]), 3) for points, _ in curves)
    # Row 0: columns 0..3 take slots (0,0),(0,1),(0,0),(0,1); row 1 the same in its own shapes.
    assert feet == sorted([0.2, 2 + 0.4, 4 + 0.2, 6 + 0.4, 0.6, 2 + 0.8, 4 + 0.6, 6 + 0.8])
