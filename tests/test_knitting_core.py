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
