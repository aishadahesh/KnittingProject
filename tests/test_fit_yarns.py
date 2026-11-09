"""fit_yarns: a region's Gaussians, and a fit's saved cloth frame carried back into the splat."""

import json

import numpy as np

import fit_yarns  # noqa: E402
import scan_fit  # noqa: E402


def test_region_takes_its_yarn_and_the_rims_nearest_it():
    # Two yarns side by side (x < 0 lace, x > 0 band), rims beside each, and the table under them.
    rng = np.random.default_rng(0)
    lace = np.c_[rng.uniform(-2, -0.5, 50), rng.uniform(0, 1, 50), np.zeros(50)]
    band = np.c_[rng.uniform(0.5, 2, 50), rng.uniform(0, 1, 50), np.zeros(50)]
    rims = np.array([[-1.0, 0.5, 0.05], [1.0, 0.5, 0.05]])
    table = np.array([[0.0, 0.5, -1.0]])
    positions = np.vstack([lace, band, rims, table])
    labels = np.r_[np.zeros(50, int), np.ones(50, int), [2, 2], [3]]
    roles = ["yarn", "yarn", "edge", "table"]
    lace_part = fit_yarns.region_members(positions, labels, roles, 0)
    band_part = fit_yarns.region_members(positions, labels, roles, 1)
    assert lace_part[:50].all() and not lace_part[50:100].any()
    assert lace_part[100] and not lace_part[101] and band_part[101]
    assert not lace_part[102] and not band_part[102]


def test_saved_frame_matches_scan_fits_carrying(tmp_path):
    rng = np.random.default_rng(1)
    q1, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q2, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    frame = {"mm_per_unit": 24.6, "extraction_centre": [1.0, 2.0, 0.5], "extraction_axes": q1.tolist(),
             "flat_centre": [0.1, -0.2, 0.3], "flat_axes": q2.tolist()}
    path = tmp_path / "stitch_model.json"
    path.write_text(json.dumps({"frame": frame}), encoding="utf-8")
    prepared = fit_yarns.fit_frame(path)
    points = rng.normal(size=(5, 3))
    expected = ((points @ q1 + frame["extraction_centre"]) / 24.6) @ q2 + frame["flat_centre"]
    assert np.allclose(scan_fit.to_splat_frame(prepared, points), expected)
    path.write_text("{}", encoding="utf-8")
    assert fit_yarns.fit_frame(path) is None
