import json
import os
import subprocess
import sys

import numpy as np
import pytest

import paths
from knitting_core import (
    build_parametric_control_rows,
    build_row_spline_jacobian,
    build_spline_mesh,
    build_surface_fiber_meshes,
    compute_bitmap_scale_factors,
    compute_knitting_faces,
    eval_centerline,
    evaluate_centerlines,
)


@pytest.fixture
def model():
    config = json.loads(paths.CONFIG_JSON.read_text())
    definitions = config["knit_parameters"]["parameters"]
    params = np.array([p["initial"] for p in definitions])
    pidx = {p["name"]: i for i, p in enumerate(definitions)}
    heights = tuple(pidx[name] for name in sorted(
        (name for name in pidx if name.startswith("loop_height_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    ))
    bitmap = np.ones((3, 4), dtype=np.float32)
    period = np.array([bitmap.shape[1] * params[pidx["stitch_width"]], 0.0, 0.0])
    return config, params, pidx, heights, bitmap, period


def test_core_import_has_no_application_or_runtime_side_effects():
    script = """
import os
import sys
before = dict(os.environ)
import knitting_core
assert dict(os.environ) == before
assert not {'moderngl', 'imgui_bundle', 'gui', 'app_state', 'jax', 'mitsuba'} & sys.modules.keys()
"""
    subprocess.run(
        [sys.executable, "-c", script], check=True,
        env={**os.environ, "PYTHONPATH": str(paths.SRC_DIR)},
    )


def test_scale_factors_cover_inactive_cells_without_mutating_bitmap():
    bitmap = np.array([[1, 1], [0, 1], [0, 0], [1, 0]], dtype=float)
    original = bitmap.copy()
    np.testing.assert_array_equal(compute_bitmap_scale_factors(bitmap), [[3, 1], [0, 3], [0, 0], [1, 0]])
    np.testing.assert_array_equal(bitmap, original)


def test_mesh_faces_join_adjacent_rings(model):
    config, *_ = model
    segments = config["knit_parameters"]["segments"]
    faces, = compute_knitting_faces(segments, [(np.zeros((2 * segments, 3)), 2)])
    assert faces.shape == (segments, 4)
    np.testing.assert_array_equal(faces[-1], [segments - 1, 0, segments, 2 * segments - 1])


def test_inactive_control_points_ignore_loop_height(model):
    _, params, pidx, heights, bitmap, _ = model
    bitmap[1, 2] = 0
    low = params.copy()
    high = params.copy()
    low[list(heights)] = 0.5
    high[list(heights)] = 8.0
    low_rows = build_parametric_control_rows(low, bitmap, pidx, heights)
    high_rows = build_parametric_control_rows(high, bitmap, pidx, heights)
    np.testing.assert_allclose(low_rows[1][6:11], high_rows[1][6:11])
    np.testing.assert_allclose(high_rows[1][6:11, 1], params[pidx["dy"]])
    np.testing.assert_allclose(high_rows[1][6:11, 2], 0)


@pytest.mark.parametrize("variable_radius", [False, True])
def test_mesh_and_simulation_share_periodic_centerlines(model, variable_radius):
    config, params, pidx, heights, bitmap, period = model
    rows = build_parametric_control_rows(params, bitmap, pidx, heights)
    assert all(row.shape == (bitmap.shape[1] * 5 + 1, 3) for row in rows)
    radius = [np.linspace(0.1, 0.3, len(row)) for row in rows] if variable_radius else None
    mesh = build_spline_mesh(rows, params, config, pidx, period, radius_ctrl_rows=radius)
    vertices, edges, _, nout = evaluate_centerlines(rows, period, config)
    assert len(mesh) == len(rows)
    assert edges.shape == (len(rows) * (nout - 1), 2)
    segments = config["knit_parameters"]["segments"]
    for index, (points, count) in enumerate(mesh):
        assert count == nout
        rings = points.reshape(nout, segments, 3)
        np.testing.assert_allclose(rings.mean(axis=1), vertices[index * nout:(index + 1) * nout], atol=1e-6)
        direction = 1 if index % 2 == 0 else -1
        np.testing.assert_allclose(rings[0] + direction * period, rings[-1], atol=1e-5)
        assert np.isfinite(points).all()


@pytest.mark.parametrize("direction", [1, -1])
def test_spline_jacobian_uses_the_same_row_direction(direction):
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.4, 0.1], [2.0, -0.2, 0.0]])[::direction]
    period = np.array([3.0, 0.0, 0.0])
    signed = direction * period
    augmented = np.vstack([points, points[0] + signed])
    knots = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(augmented, axis=0), axis=1))]
    samples = np.linspace(0.0, knots[-1], 20)
    jacobian = build_row_spline_jacobian(points, period, 20)
    moved = points.copy()
    moved[1, 1] += 1e-4
    difference = (eval_centerline(moved, period, 20, knots, samples)
                  - eval_centerline(points, period, 20, knots, samples)) / 1e-4
    np.testing.assert_allclose(difference[:, 1], jacobian[:, 1], atol=1e-9)


def test_period_is_a_vector():
    with pytest.raises(ValueError, match="3D vector"):
        eval_centerline(np.zeros((2, 3)), 2.0, 5)


@pytest.mark.parametrize("enabled, count", [(False, 1), (True, 3)])
def test_fiber_meshes(model, enabled, count):
    config, params, pidx, heights, bitmap, period = model
    rows = build_parametric_control_rows(params, bitmap, pidx, heights)[:1]
    base = build_spline_mesh(rows, params, config, pidx, period)
    meshes, metadata = build_surface_fiber_meshes(
        base, config["knit_parameters"]["segments"], enabled, 3,
        radius=0.1, radius_scale=1.0, lift=0.0, surface_arc=0.5, randomness=0.0, twist=0.0,
    )
    assert len(meshes) == len(metadata) == count
    assert all(entry["row"] == 0 for entry in metadata)
    assert all(np.isfinite(points).all() for points, _ in meshes)
