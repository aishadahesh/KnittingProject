import os
import sys

# Force JAX to use CPU backend
import jax
jax.config.update("jax_platform_name", "cpu")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
