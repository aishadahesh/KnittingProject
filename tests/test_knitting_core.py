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
    _scale_factors_jax, eval_curve, eval_curve_derivative, compute_orthonormal_frame
)

def test_scale_factors_jax():
    bitmap = jnp.array([[1.0, 0.0], [1.0, 1.0]], dtype=jnp.float32)
    factors = _scale_factors_jax(bitmap)
    assert factors.shape == (2, 2)
    assert np.all(factors >= 0.0)

def test_eval_curve_and_derivative():
    t = jnp.linspace(0.0, 2.0 * jnp.pi, 10)
    hl = jnp.ones(10)
    lh = 1.5
    sb = 0.25
    sz = 0.1
    
    pos = eval_curve(t, hl, lh, sb, sz)
    der = eval_curve_derivative(t, hl, lh, sb, sz)
    
    assert pos.shape == (10, 3)
    assert der.shape == (10, 3)

def test_compute_orthonormal_frame():
    tangent = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=jnp.float32)
    u, v = compute_orthonormal_frame(tangent)
    assert u.shape == (2, 3)
    assert v.shape == (2, 3)
    
    # Assert orthonormality
    for i in range(2):
        t_val = tangent[i]
        u_val = u[i]
        v_val = v[i]
        assert np.abs(np.dot(t_val, u_val)) < 1e-5
        assert np.abs(np.dot(t_val, v_val)) < 1e-5
        assert np.abs(np.dot(u_val, v_val)) < 1e-5
        assert np.abs(np.linalg.norm(u_val) - 1.0) < 1e-5
        assert np.abs(np.linalg.norm(v_val) - 1.0) < 1e-5


from knitting_core import (
    compute_knitting_vertices, compute_knitting_faces,
    build_parametric_control_rows, build_spline_mesh
)

def test_mesh_generation(config_fixture, params_fixture):
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
    
    # 1. Compute vertices
    vl = compute_knitting_vertices(params_list, bitmap, config_fixture, pidx, lh_idx)
    assert isinstance(vl, list)
    assert len(vl) == bitmap.shape[0]
    for verts, pt_count in vl:
        assert isinstance(verts, np.ndarray)
        assert verts.ndim == 2
        assert verts.shape[1] == 3
        assert isinstance(pt_count, int)
        
    # 2. Compute faces
    seg = config_fixture["knit_parameters"]["segments"]
    fl = compute_knitting_faces(seg, vl)
    assert len(fl) == len(vl)
    for faces in fl:
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
