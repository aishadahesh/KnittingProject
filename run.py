#%%
import os
import json
import numpy as np
import scipy.sparse
import jax

# Force JAX to use CPU backend
jax.config.update("jax_platform_name", "cpu")

from knitting_core import (
    evaluate_centerlines,
    build_row_spline_jacobian
)
from yarn_simulation import (
    run_simulation_step,
    eval_energy
)

# ──────────────────────────────────────────────────────────────────────────────
# HARDCODED CONFIGURATIONS
# ──────────────────────────────────────────────────────────────────────────────
INPUT_PARAMS_PATH = "params.json"
OUTPUT_PARAMS_PATH = "params_optimized.json"

MAX_ITER = 100
TOL = 1e-5
ENERGY_TOL = 1e-6

# Physical parameter values
K_S = 1000.0
K_B = 10.0
K_C = 1.0
DHAT = 0.1

# If True, keep the rest lengths (L0) fixed at their initial values from the start of the simulation.
# If False, updates L0 at each step (matching the GUI's default simulation loop behavior).
FIXED_L0 = True

# ──────────────────────────────────────────────────────────────────────────────
# SETUP AND LOADING
# ──────────────────────────────────────────────────────────────────────────────
print(f"Loading parameter file: {INPUT_PARAMS_PATH}...")
with open(INPUT_PARAMS_PATH, "r") as f:
    params_data = json.load(f)

print("Loading configuration: config.json...")
with open("config.json", "r") as f:
    config = json.load(f)

# Extract control rows, period offsets
ctrl_rows = [np.array(row, dtype=float) for row in params_data["spline_control_rows"]]
period_offset_x = np.array(params_data["period_offset_x"], dtype=float)
period_offset_y = np.array(params_data["period_offset_y"], dtype=float)

# Extract loop resolution
loop_res = config["knit_parameters"]["loop_res"]

# Evaluate initial geometry
V, edges, _, nout = evaluate_centerlines(ctrl_rows, period_offset_x, config)

# Initialize rest lengths (L0)
v0_pts = V[edges[:, 0]]
v1_pts = V[edges[:, 1]]
L0_array = np.linalg.norm(v1_pts - v0_pts, axis=1)

print(f"Loaded {len(ctrl_rows)} control rows with {nout} sample points per row.")
print(f"Initial stretch rest lengths (L0) computed. Mean rest length: {np.mean(L0_array):.6f}")

# Rebuild Jacobian function
def rebuild_jacobian(rows, px, res):
    J_blocks = []
    bitmap_width = float(np.linalg.norm(px))
    nout_cp = res * int(round(bitmap_width)) + 1
    for cp in rows:
        J_r = build_row_spline_jacobian(cp, px, nout_cp)
        J_blocks.append(J_r)
    J_base = scipy.sparse.block_diag(J_blocks, format="csr")
    return scipy.sparse.kron(J_base, scipy.sparse.identity(3), format="csr")

J_cached = rebuild_jacobian(ctrl_rows, period_offset_x, loop_res)

# ──────────────────────────────────────────────────────────────────────────────
# SIMULATION LOOP
# ──────────────────────────────────────────────────────────────────────────────
print("\nStarting simulation loop...")
print(f"{'Iter':<5} | {'E_total':<12} | {'E_elastic':<12} | {'E_bend':<12} | {'E_collision':<12} | {'||dP||_inf':<12} | {'dE':<12}")
print("-" * 90)

history = []
converged = False
iter_idx = 0

# Compute initial energy
flat_P = np.concatenate(ctrl_rows).astype(float)
e_el, e_b, e_col = eval_energy(flat_P, ctrl_rows, period_offset_x, period_offset_y, config, L0_array, DHAT)
e_old = K_S * e_el + K_B * e_b + K_C * e_col

for iter_idx in range(1, MAX_ITER + 1):
    # Step the simulation
    new_ctrl_rows = run_simulation_step(
        ctrl_rows=ctrl_rows,
        period_offset_x=period_offset_x,
        period_offset_y=period_offset_y,
        config=config,
        J_cached=J_cached,
        L0_array=L0_array,
        k_s=K_S,
        k_b=K_B,
        k_c=K_C,
        dhat=DHAT
    )
    
    # Calculate step change metric
    flat_P_new = np.concatenate(new_ctrl_rows).astype(float)
    flat_P_old = np.concatenate(ctrl_rows).astype(float)
    
    dp_norm = np.max(np.abs(flat_P_new - flat_P_old))
    
    # Calculate energy and changes
    e_el_new, e_b_new, e_col_new = eval_energy(
        flat_P_new, new_ctrl_rows, period_offset_x, period_offset_y, config, L0_array, DHAT
    )
    e_new = K_S * e_el_new + K_B * e_b_new + K_C * e_col_new
    de = e_new - e_old
    
    # Save to history
    history.append({
        "iter": iter_idx,
        "e_total": e_new,
        "e_elastic": e_el_new,
        "e_bend": e_b_new,
        "e_col": e_col_new,
        "delta_P_max": dp_norm,
        "de": de
    })
    
    # Print diagnostics
    print(f"{iter_idx:<5} | {e_new:<12.6e} | {e_el_new:<12.6e} | {e_b_new:<12.6e} | {e_col_new:<12.6e} | {dp_norm:<12.6e} | {de:<+12.6e}")
    
    # Check convergence criteria
    if dp_norm < TOL or abs(de) < ENERGY_TOL:
        print(f"\nConverged at iteration {iter_idx}! (||dP||_inf = {dp_norm:.2e}, |dE| = {abs(de):.2e})")
        converged = True
        ctrl_rows = new_ctrl_rows
        break
        
    # Update variables for next step
    ctrl_rows = new_ctrl_rows
    e_old = e_new
    
    # Update L0 if not fixed
    if not FIXED_L0:
        V, edges, _, _ = evaluate_centerlines(ctrl_rows, period_offset_x, config)
        v0_pts = V[edges[:, 0]]
        v1_pts = V[edges[:, 1]]
        L0_array = np.linalg.norm(v1_pts - v0_pts, axis=1)
        
    # Rebuild Jacobian for the new spline shape
    J_cached = rebuild_jacobian(ctrl_rows, period_offset_x, loop_res)

else:
    print(f"\nReached max iterations ({MAX_ITER}) without full convergence.")

# ──────────────────────────────────────────────────────────────────────────────
# SAVE OUTPUT
# ──────────────────────────────────────────────────────────────────────────────
print(f"Saving optimized parameters to {OUTPUT_PARAMS_PATH}...")
params_data["spline_control_rows"] = [row.tolist() for row in ctrl_rows]
with open(OUTPUT_PARAMS_PATH, "w") as f:
    json.dump(params_data, f, indent=2)

print("Done. Variables (ctrl_rows, V, edges, J_cached, history, converged, etc.) are available for inspection.")
