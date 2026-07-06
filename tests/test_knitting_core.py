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
