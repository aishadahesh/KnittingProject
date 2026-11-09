"""The strand equations: a fitted equation reproduces a known curve, and a splat of known yarn comes
back as strands that lie on it."""

import math

import numpy as np

import strand_equations  # noqa: E402


def test_open_strand_equation_recovers_a_known_curve():
    t = np.linspace(0.0, 1.0, 200)
    curve = np.stack([4.0 * t, 0.6 * np.sin(math.pi * t) - 0.2 * np.sin(3 * math.pi * t), 0.3 * np.sin(2 * math.pi * t)], axis=1)
    strand = strand_equations.fit_strand(curve, closed=False, radius_mm=0.3, tolerance_mm=0.005, max_terms=8)
    assert strand.rms_mm < 0.005 and strand.terms <= 8
    assert np.allclose(strand.evaluate([0.0, 1.0]), curve[[0, -1]], atol=1e-9)    # the ends are held exactly
    # t runs evenly along the strand's length (not the curve's own parameter), so the terms differ;
    # the shape must not: every point of the equation's curve lies on the original.
    from scipy.spatial import cKDTree
    gap, _ = cKDTree(curve).query(strand.evaluate(np.linspace(0.0, 1.0, 500)))
    assert gap.max() < 0.02
    assert len(strand.equations()) == 3 and strand.equations()[0].startswith("x(t) = ")


def test_closed_loop_equation_recovers_a_circle():
    t = np.linspace(0.0, 1.0, 120, endpoint=False)
    ring = np.stack([2.0 + np.cos(2 * math.pi * t), np.sin(2 * math.pi * t), np.zeros_like(t)], axis=1)
    strand = strand_equations.fit_strand(ring, closed=True, radius_mm=0.2, tolerance_mm=0.01, max_terms=6)
    assert strand.terms == 1 and strand.rms_mm < 0.01
    assert abs(strand.length_mm - 2 * math.pi) < 0.05


def _yarn_splat(mm_per_unit: float) -> dict:
    """Three bright wavy strands of yarn (0.35 mm radius) lying on a dark cloth, as Gaussians."""
    rng = np.random.default_rng(1)
    points, colours = [], []
    for row in range(3):
        s = rng.uniform(0.0, 1.0, 9000)
        angle = rng.uniform(0.0, 2 * math.pi, len(s))
        x = -6.0 + 12.0 * s
        y = 3.0 * row + 0.5 * np.sin(2 * math.pi * s * 2)
        z = 0.35 + 0.35 * np.sin(angle)
        y = y + 0.35 * np.cos(angle)
        top = np.sin(angle) > -0.2          # what a camera above sees
        points.append(np.stack([x, y, z], axis=1)[top])
        colours.append(np.tile((0.9, 0.9, 0.85), (int(top.sum()), 1)))
    # The dark cloth under the yarn.
    floor = np.stack([rng.uniform(-6, 6, 12000), rng.uniform(-1.5, 7.5, 12000), rng.uniform(-0.05, 0.0, 12000)], axis=1)
    points.append(floor)
    colours.append(np.tile((0.08, 0.08, 0.25), (len(floor), 1)))
    positions = np.vstack(points) / mm_per_unit
    count = len(positions)
    return {"positions": positions.astype(np.float32), "colours": np.vstack(colours).astype(np.float32),
            "opacity": np.full(count, 0.9, np.float32), "radius": np.full(count, 0.05, np.float32),
            "sigma": np.full(count, 0.05, np.float32), "count": count}


def test_extraction_traces_known_yarn_and_rebuilds_it_on_the_scan():
    mm_per_unit = 50.0
    result = strand_equations.extract(_yarn_splat(mm_per_unit), mm_per_unit, cloth_normal=[0.0, 0.0, 1.0])
    summary = result.summary()
    # Three strands of about 13 mm each (12 mm across, waving): most of that length is traced, and no
    # strand twice over (once along each edge would be about 80 mm). Pieces overlap a little where
    # they are joined, so the total runs above 39 mm.
    assert 25.0 < summary["total_length_mm"] < 70.0
    assert summary["fit_rms_mm"] < 0.06
    # What the equations rebuild is on the scanned yarn.
    assert summary["rebuilt_to_scan_mean_mm"] < 0.3
    tube = result.tube
    assert len(tube["vertices"]) and tube["faces"].max() < len(tube["vertices"])
    # Back in the splat's own frame, the rebuilt yarn sits where the yarn Gaussians are.
    rebuilt = result.to_splat_frame(tube["vertices"]) * mm_per_unit
    assert abs(np.median(rebuilt[:, 2]) - 0.35) < 0.4


def test_saved_equations_and_mesh(tmp_path):
    mm_per_unit = 50.0
    result = strand_equations.extract(_yarn_splat(mm_per_unit), mm_per_unit, cloth_normal=[0.0, 0.0, 1.0])
    saved = strand_equations.save(result, tmp_path / "eq.json")
    import json
    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert payload["units"] == "mm" and len(payload["strands"]) == len(result.strands)
    first = payload["strands"][0]
    rebuilt = strand_equations.Strand(first["closed"], first["radius_mm"], first["length_mm"],
                                      np.array([first["coefficients"][a] for a in "xyz"]), first["rms_mm"], np.zeros((0, 3)))
    assert np.allclose(rebuilt.evaluate([0.3]), result.strands[0].evaluate([0.3]))
    mesh = strand_equations.save_mesh(result, tmp_path / "yarn.ply")
    assert mesh.read_bytes().startswith(b"ply\nformat binary_little_endian")
