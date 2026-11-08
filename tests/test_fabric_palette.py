"""The fabric's colour for a model, putting one colour on the whole fabric, and saved colours."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import fabric_palette  # noqa: E402

LUMA = np.array([0.2126, 0.7152, 0.0722])


def _fabric(tmp_path, labels):
    report = {"colours": [
        {"role": "yarn", "share": 0.7, "rgb": [0.68, 0.74, 0.78], "corrected_name": "light blue"},
        {"role": "yarn", "share": 0.2, "rgb": [0.18, 0.19, 0.2], "corrected_name": "charcoal"},
        {"role": "table", "share": 0.1, "rgb": [0.0, 0.0, 0.2]},
    ]}
    (tmp_path / fabric_palette.COLOURS_FILE).write_text(json.dumps(report))
    np.save(tmp_path / fabric_palette.LABELS_FILE, np.asarray(labels, dtype=np.int8))
    return fabric_palette.load(tmp_path, gaussians=len(labels))


def _yarn(rng, count, colour, spread):
    """Gaussians of one yarn: its colour in light and shade (brightness 0.2-1.0 of it)."""
    lin = fabric_palette.srgb_to_linear(np.asarray(colour)) * rng.uniform(1.0 - spread, 1.0, (count, 1))
    return fabric_palette.linear_to_srgb(lin)


def _cloth(tmp_path):
    rng = np.random.default_rng(0)
    lace = _yarn(rng, 300, (0.7, 0.75, 0.8), 0.8)
    band = _yarn(rng, 200, (0.2, 0.2, 0.22), 0.95)          # dark, mostly near-black
    table = np.tile([0.02, 0.03, 0.3], (50, 1))
    colours = np.concatenate([lace, band, table]).astype(np.float32)
    return colours, _fabric(tmp_path, [0] * 300 + [1] * 200 + [2] * 50)


def test_labels_from_another_splat_are_not_used(tmp_path):
    _fabric(tmp_path, [0, 0, 1])
    assert fabric_palette.load(tmp_path, gaussians=3)["labels"] is not None
    assert fabric_palette.load(tmp_path, gaussians=4)["labels"] is None
    assert fabric_palette.load(tmp_path / "missing", gaussians=3) is None


def test_one_colour_goes_on_the_whole_fabric_and_comes_back_as_measured(tmp_path):
    colours, fabric = _cloth(tmp_path)
    red = np.array([0.8, 0.15, 0.15])
    out = fabric_palette.recolour_all(colours, fabric["labels"], fabric["groups"], red)
    # The fabric's colour, measured the same way, is the colour put on (within rounding).
    assert np.allclose(fabric_palette.fabric_colour(fabric, out), red, atol=0.02)
    lace, band, table = out[:300], out[300:500], out[500:]
    assert (lace[:, 0] > lace[:, 1]).all() and (band[:, 0] >= band[:, 1]).all()        # every yarn Gaussian is red now
    assert np.allclose(table, colours[500:])                                           # the table is not touched
    # The lace keeps its light and shade: brightness order and ratios as before.
    before = fabric_palette.srgb_to_linear(colours[:300]) @ LUMA
    after = fabric_palette.srgb_to_linear(lace) @ LUMA
    assert np.corrcoef(before, after)[0, 1] > 0.999
    # The dark band takes the lace's light and shade: no longer near-black.
    assert np.median(fabric_palette.srgb_to_linear(band) @ LUMA) > 0.5 * np.median(after)


def test_a_colour_taken_from_one_model_looks_the_same_on_another(tmp_path):
    rng = np.random.default_rng(1)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    blue = _yarn(rng, 400, (0.55, 0.65, 0.8), 0.7)
    cream = _yarn(rng, 400, (0.85, 0.84, 0.8), 0.6)
    a = _fabric(tmp_path / "a", [0] * 400)
    b = _fabric(tmp_path / "b", [0] * 400)
    taken = fabric_palette.fabric_colour(a, blue)
    put = fabric_palette.recolour_all(cream, b["labels"], b["groups"], taken)
    assert np.allclose(fabric_palette.fabric_colour(b, put), taken, atol=0.02)


def test_a_colour_is_saved_and_listed(tmp_path):
    folder = tmp_path / "palettes"
    fabric_palette.save_colour("Nov 2: lace", (0.1, 0.5, 0.2), "ur5_20261102_175028", folder)
    (saved,) = fabric_palette.list_palettes(folder)
    assert saved["name"] == "Nov 2_ lace" and saved["source"] == "ur5_20261102_175028"
    assert saved["colours"][0]["hex"] == "#1a8033"


def test_the_yarn_map_paints_each_group_flat(tmp_path):
    colours, fabric = _cloth(tmp_path)
    paint = fabric_palette.yarn_map(fabric, len(colours))
    assert np.allclose(paint[:300], fabric["groups"][0]["measured"], atol=1e-6)
    assert np.allclose(paint[500:], fabric_palette.YARN_MAP_COLOURS["table"])


def test_an_order_of_colour_numbers_is_read():
    assert fabric_palette.parse_sequence("1,2", 2) == [0, 1]
    assert fabric_palette.parse_sequence("1 1 2", 2) == [0, 0, 1]
    assert fabric_palette.parse_sequence("3, 1, 9", 3) == [2, 0]            # numbers past the colours are left out
    assert fabric_palette.parse_sequence("", 3) == [0, 1, 2]
    assert fabric_palette.parse_sequence("abc", 2) == [0, 1]


def _strip(tmp_path, yarn_groups=1):
    """A strip of cloth 20 x 6 mm (1 unit = 1 mm) along x, in light and shade, with `yarn_groups` yarns
    side by side (the first the widest) and a table spot."""
    rng = np.random.default_rng(3)
    xy = rng.uniform([0, 0], [20, 6], (2000, 2))
    positions = np.c_[xy, np.zeros(2000)]
    colours = _yarn(rng, 2000, (0.8, 0.8, 0.78), 0.7)
    labels = np.zeros(2000, dtype=int)
    if yarn_groups == 4:
        labels = np.digitize(xy[:, 1], [2.4, 3.6, 4.8])               # four yarns across the strip, widest first
    table = np.linalg.norm(xy - (10, 3), axis=1) < 0.8
    labels[table] = 9
    colours[table] = (0.02, 0.03, 0.3)
    report = {"colours": [{"role": "yarn", "share": share, "rgb": [0.8, 0.8, 0.78]} for share in (0.4, 0.2, 0.2, 0.2)[:max(yarn_groups, 1)]]}
    report["colours"] += [{"role": "table", "share": 0.01, "rgb": [0.02, 0.03, 0.3]}] * (10 - len(report["colours"]))
    (tmp_path / fabric_palette.COLOURS_FILE).write_text(json.dumps(report))
    np.save(tmp_path / fabric_palette.LABELS_FILE, labels.astype(np.int8))
    return positions, colours, fabric_palette.load(tmp_path, 2000)


def test_colours_spread_by_yarn_follow_the_order_round_and_round(tmp_path):
    positions, colours, fabric = _strip(tmp_path, yarn_groups=4)
    red, blue = np.array([0.8, 0.1, 0.1]), np.array([0.1, 0.1, 0.8])
    targets = fabric_palette.spread(fabric, positions, [red, blue], [0, 1], fabric_palette.BY_YARN)
    for yarn_number, expected in enumerate([red, blue, red, blue]):      # yarn 1 colour 1, yarn 2 colour 2, yarn 3 colour 1...
        assert np.allclose(targets[fabric["labels"] == yarn_number], expected)


def test_colours_spread_in_stripes_across_the_fabric(tmp_path):
    positions, colours, fabric = _strip(tmp_path)
    red, blue = np.array([0.8, 0.1, 0.1]), np.array([0.1, 0.1, 0.8])
    targets = fabric_palette.spread(fabric, positions, [red, blue], [0, 0, 1], fabric_palette.STRIPES, stripe_mm=2.0, angle_deg=0.0)
    yarn = fabric["labels"] != 9
    band = np.floor(positions[:, 0] - positions[yarn, 0].min()) // 2
    expected = np.where((band % 3 == 2)[:, None], blue, red)               # 2 mm bands along x: red, red, blue, red, red, blue...
    assert np.allclose(targets[yarn], expected[yarn])
    out = fabric_palette.recolour_all(colours, fabric["labels"], fabric["groups"], targets)
    assert (out[yarn & (band % 3 == 2), 2] > out[yarn & (band % 3 == 2), 0]).all()           # the blue stripes are blue
    assert np.allclose(out[~yarn], colours[~yarn])                                           # the table untouched
    # Turned 90 degrees, the stripes run the other way: bands along y.
    turned = fabric_palette.spread(fabric, positions, [red, blue], [0, 1], fabric_palette.STRIPES, stripe_mm=2.0, angle_deg=90.0)
    across = np.floor(positions[:, 1] - positions[yarn, 1].min()) // 2
    assert np.allclose(turned[yarn], np.where((across % 2 == 1)[:, None], blue, red)[yarn])


def test_several_colours_are_saved_together(tmp_path):
    folder = tmp_path / "palettes"
    fabric_palette.save_colour("Nov 2", [(0.75, 0.8, 0.8), (0.27, 0.28, 0.27)], "ur5_20261102_175028", folder)
    (saved,) = fabric_palette.list_palettes(folder)
    assert [c["hex"] for c in saved["colours"]] == ["#bfcccc", "#454745"]
