import os
import sys

# Add project root to sys.path to resolve knitting_core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force JAX to use CPU backend
import jax
jax.config.update("jax_platform_name", "cpu")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def test_core_module_imports():
    # Verify that importing knitting_core does not pull in GUI-related libraries
    import knitting_core
    assert "moderngl" not in sys.modules
    assert "imgui_bundle" not in sys.modules
    assert "gui" not in sys.modules
    assert "app_state" not in sys.modules


import json
import pytest
import numpy as np

@pytest.fixture
def config_fixture():
    with open("config.json", "r") as f:
        return json.load(f)

@pytest.fixture
def params_fixture():
    with open("params.json", "r") as f:
        return json.load(f)

def test_fixtures_load(config_fixture, params_fixture):
    assert "knit_parameters" in config_fixture
    assert "params" in params_fixture
    assert "bitmap" in params_fixture


import jax.numpy as jnp
from knitting_core import (
    _scale_factors_jax, eval_curve
)

def test_scale_factors_jax():
    bitmap = jnp.array([[1.0, 0.0], [1.0, 1.0]], dtype=jnp.float32)
    factors = _scale_factors_jax(bitmap)
    assert factors.shape == (2, 2)
    assert np.all(factors >= 0.0)

def test_eval_curve():
    t = jnp.linspace(0.0, 2.0 * jnp.pi, 10)
    hl = jnp.ones(10)
    lh = 1.5
    sb = 0.25
    sz = 0.1
    
    pos = eval_curve(t, hl, lh, sb, sz)
    assert pos.shape == (10, 3)


from knitting_core import (
    compute_knitting_faces,
    build_parametric_control_rows, build_spline_mesh
)

def test_mesh_generation(config_fixture):
    seg = config_fixture["knit_parameters"]["segments"]
    # Mock vertex list for 1 row, 2 points
    verts = np.zeros((2 * seg, 3), dtype=np.float32)
    vl = [(verts, 2)]
    fl = compute_knitting_faces(seg, vl)
    assert len(fl) == 1
    faces = fl[0]
    assert isinstance(faces, np.ndarray)
    assert faces.ndim == 2
    assert faces.shape[1] == 4  # quads

def test_spline_construction(config_fixture, params_fixture):
    params_dict = params_fixture["params"]
    config_params = config_fixture["knit_parameters"]["parameters"]
    params_list = [params_dict.get(p["name"], p["initial"]) for p in config_params]
    
    pidx = {p["name"]: i for i, p in enumerate(config_params)}
    lh_params = sorted(
        [p["name"] for p in config_params if p["name"].startswith("loop_height_")],
        key=lambda name: int(name.split("_")[-1])
    )
    lh_idx = tuple(pidx[name] for name in lh_params)
    bitmap = np.asarray(params_fixture["bitmap"], dtype=np.float32)
    
    # Build parametric control rows
    ctrl_rows = build_parametric_control_rows(params_list, bitmap, pidx, lh_idx)
    assert len(ctrl_rows) == bitmap.shape[0]
    for row in ctrl_rows:
        assert isinstance(row, np.ndarray)
        assert row.shape[1] == 3
        
    # Build spline mesh
    spline_mesh = build_spline_mesh(ctrl_rows, params_list, config_fixture, pidx, bitmap.shape[1])
    assert len(spline_mesh) == len(ctrl_rows)
    for pts, nout in spline_mesh:
        assert pts.shape[1] == 3
        assert isinstance(nout, int)


from knitting_core import build_surface_fiber_meshes

def test_fiber_meshes():
    # Mock base vertices: List of (verts, count)
    # Each row contains (n_points * segments) vertices. Let n_points = 2, segments = 6
    segments = 6
    n_points = 2
    verts_row = np.arange(n_points * segments * 3, dtype=np.float32).reshape(-1, 3)
    base_vl = [(verts_row, n_points)]
    
    # 1. Disabled case
    out_vl, meta = build_surface_fiber_meshes(
        base_vl=base_vl,
        segments=segments,
        enabled=False,
        count=3,
        radius=0.1,
        radius_scale=1.0,
        lift=0.0,
        surface_arc=0.5,
        randomness=0.0,
        twist=0.0
    )
    assert len(out_vl) == len(base_vl)
    assert meta[0]["row"] == 0
    
    # 2. Enabled case
    out_vl, meta = build_surface_fiber_meshes(
        base_vl=base_vl,
        segments=segments,
        enabled=True,
        count=3,
        radius=0.1,
        radius_scale=1.0,
        lift=0.0,
        surface_arc=0.5,
        randomness=0.0,
        twist=0.0
    )
    # We expect count=3 fiber meshes produced
    assert len(out_vl) == 3
    assert len(meta) == 3
    for entry in meta:
        assert entry["row"] == 0


def test_spline_mesh_with_variable_radius(config_fixture, params_fixture):
    params_dict = params_fixture["params"]
    config_params = config_fixture["knit_parameters"]["parameters"]
    params_list = [params_dict.get(p["name"], p["initial"]) for p in config_params]
    pidx = {p["name"]: i for i, p in enumerate(config_params)}
    lh_params = sorted(
        [p["name"] for p in config_params if p["name"].startswith("loop_height_")],
        key=lambda name: int(name.split("_")[-1])
    )
    lh_idx = tuple(pidx[name] for name in lh_params)
    bitmap = np.asarray(params_fixture["bitmap"], dtype=np.float32)

    ctrl_rows = build_parametric_control_rows(params_list, bitmap, pidx, lh_idx)
    
    # Define variable radius rows matching ctrl_rows shapes
    radius_ctrl_rows = [
        np.full(len(row), 0.35, dtype=np.float32)
        for row in ctrl_rows
    ]
    
    spline_mesh = build_spline_mesh(
        ctrl_rows, params_list, config_fixture, pidx, bitmap.shape[1],
        radius_ctrl_rows=radius_ctrl_rows
    )
    assert len(spline_mesh) == len(ctrl_rows)
    for pts, nout in spline_mesh:
        assert pts.shape[1] == 3
        assert isinstance(nout, int)


def test_periodic_spline_continuity(config_fixture, params_fixture):
    from knitting_core import build_parametric_control_rows, build_spline_mesh
    params_dict = params_fixture["params"]
    config_params = config_fixture["knit_parameters"]["parameters"]
    params_list = [params_dict.get(p["name"], p["initial"]) for p in config_params]
    pidx = {p["name"]: i for i, p in enumerate(config_params)}
    lh_params = sorted(
        [p["name"] for p in config_params if p["name"].startswith("loop_height_")],
        key=lambda name: int(name.split("_")[-1])
    )
    lh_idx = tuple(pidx[name] for name in lh_params)
    bitmap = np.asarray(params_fixture["bitmap"], dtype=np.float32)
    
    # 1. Verify control point lengths are cols * spl (10)
    ctrl_rows = build_parametric_control_rows(params_list, bitmap, pidx, lh_idx)
    assert ctrl_rows[0].shape[0] == 10
    
    # 2. Build mesh and check periodic boundary alignment
    spline_mesh = build_spline_mesh(ctrl_rows, params_list, config_fixture, pidx, bitmap.shape[1])
    seg = config_fixture["knit_parameters"]["segments"]
    D = np.array([float(bitmap.shape[1]), 0.0, 0.0])
    for pts, nout in spline_mesh:
        # First ring of vertices (start) + period translation must match the last ring (end) exactly
        np.testing.assert_allclose(pts[:seg] + D, pts[-seg:], atol=1e-4)




