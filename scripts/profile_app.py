"""Profile the critical hot-path functions of the KnittingProject.

This script imports the real modules and profiles the actual geometry pipeline
end-to-end: parametric control-row generation, spline mesh construction,
fiber meshes, normals, Jacobian, and simulation step — using the project's
own params.json and config.json as data inputs.

Run with:  python scripts/profile_app.py
"""

import os, sys, time, json
import numpy as np
import cProfile, pstats, io

# This script sits in scripts/, so the modules it profiles are one level up in
# src/. Everything else it needs comes from paths.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import paths

project_root = str(paths.PROJECT_ROOT)

# ── Load real project data ────────────────────────────────────────────────────
with open(paths.CONFIG_JSON, "r") as f:
    config = json.load(f)

with open(paths.PARAMS_JSON, "r") as f:
    params_data = json.load(f)

# Build parameter index inline (mirrors app_state.py __init__)
pidx = {p["name"]: i for i, p in enumerate(config["knit_parameters"]["parameters"])}
lh_params = sorted(
    [p["name"] for p in config["knit_parameters"]["parameters"] if p["name"].startswith("loop_height_")],
    key=lambda name: int(name.split("_")[-1])
)
lh_idx = tuple(pidx[name] for name in lh_params)

# Build params array from dict (mirrors app_state.py load_params)
p_dict = params_data.get("params", {})
param_defs = config["knit_parameters"]["parameters"]
params = np.array([pd["initial"] for pd in param_defs], dtype=np.float32)
for i, pd in enumerate(param_defs):
    if pd["name"] in p_dict:
        lo, hi = pd["range"]
        params[i] = float(np.clip(p_dict[pd["name"]], lo, hi))
bitmap = np.array(params_data.get("bitmap", [[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]), dtype=np.float32)
saved_ctrl_rows = [np.array(r, dtype=np.float32) for r in params_data.get("spline_control_rows", [])]
saved_radius_rows = [np.array(r, dtype=np.float32) for r in params_data.get("spline_radius_rows", [])]
period_offset_x_saved = np.array(params_data.get("period_offset_x", [float(bitmap.shape[1]), 0.0, 0.0]), dtype=np.float32)
period_offset_y_saved = np.array(params_data.get("period_offset_y", [0.0, 4.0, 0.0]), dtype=np.float32)

from knitting_core import (
    build_parametric_control_rows,
    build_spline_mesh,
    compute_knitting_faces,
    build_surface_fiber_meshes,
    eval_centerline,
    evaluate_centerlines,
    build_row_spline_jacobian,
    _centerline_sample_count,
)
from rendering import compute_normals

# ── Helpers ───────────────────────────────────────────────────────────────────
def timed(label, func, *args, repeats=3, **kwargs):
    """Run func repeats times, print min/avg time, return last result."""
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    mn = min(times)
    print(f"  {label:45s}  avg={avg*1000:8.2f} ms  min={mn*1000:8.2f} ms")
    return result


# ── PROFILE RUNS ──────────────────────────────────────────────────────────────
print("=" * 78)
print("KnittingProject Performance Profile")
print(f"  bitmap shape : {bitmap.shape}")
print(f"  params count : {len(params)}")
print(f"  segments     : {config['knit_parameters']['segments']}")
print(f"  loop_res     : {config['knit_parameters']['loop_res']}")
print("=" * 78)

spl = config["knit_parameters"].get("samples_per_loop", 5)
loop_heights_raw = params_data.get("loop_heights")
loop_heights = np.array(loop_heights_raw, dtype=np.float32) if loop_heights_raw else None

# 1. Build parametric control rows
print("\n── 1. build_parametric_control_rows ─────────────────────────────────")
ctrl_rows = timed("build_parametric_control_rows",
    build_parametric_control_rows, params, bitmap, pidx, lh_idx, spl,
    loop_heights=loop_heights)
print(f"     rows={len(ctrl_rows)}, points-per-row={[len(r) for r in ctrl_rows]}")

# Use saved ctrl_rows if available (these are what the app actually uses)
if saved_ctrl_rows:
    print(f"     Using SAVED ctrl_rows: {len(saved_ctrl_rows)} rows, pts/row={[len(r) for r in saved_ctrl_rows]}")
    ctrl_rows = saved_ctrl_rows

# 2. Period offset
period_offset_x = period_offset_x_saved
nout = _centerline_sample_count(period_offset_x, config)
print(f"\n     period_offset_x = {period_offset_x}, nout = {nout}")

# 3. eval_centerline (single row)
print("\n── 2. eval_centerline (single row) ─────────────────────────────────")
timed("eval_centerline (row 0)", eval_centerline,
      np.asarray(ctrl_rows[0], dtype=float), period_offset_x, nout)

# 4. evaluate_centerlines (all rows)
print("\n── 3. evaluate_centerlines (all rows) ──────────────────────────────")
result = timed("evaluate_centerlines", evaluate_centerlines,
               ctrl_rows, period_offset_x, config)
V, edges, D, nout_actual = result
print(f"     V.shape={V.shape}, edges.shape={edges.shape}")

# 5. build_spline_mesh
print("\n── 4. build_spline_mesh ────────────────────────────────────────────")
rad = float(params[pidx["radius"]])
radius_profiles = [np.full(len(r), rad, dtype=np.float32) for r in ctrl_rows]
vl = timed("build_spline_mesh", build_spline_mesh,
           ctrl_rows, params, config, pidx, period_offset_x,
           radius_ctrl_rows=radius_profiles)
print(f"     {len(vl)} tube(s), total verts = {sum(len(v) for v,_ in vl)}")

# 6. compute_knitting_faces
print("\n── 5. compute_knitting_faces ───────────────────────────────────────")
fl = timed("compute_knitting_faces", compute_knitting_faces,
           config['knit_parameters']['segments'], vl)
print(f"     {len(fl)} face arrays, total faces = {sum(len(f) for f in fl)}")

# 7. build_surface_fiber_meshes
print("\n── 6. build_surface_fiber_meshes (fibers enabled) ─────────────────")
fiber_vl, fiber_meta = timed("build_surface_fiber_meshes (enabled)",
    build_surface_fiber_meshes,
    vl,
    segments=int(config['knit_parameters']['segments']),
    enabled=True,
    count=4,
    radius=rad,
    radius_scale=0.15,
    lift=0.1,
    surface_arc=0.5,
    randomness=0.3,
    twist=2.0,
)
print(f"     {len(fiber_vl)} fiber tube(s)")

print("\n── 7. build_surface_fiber_meshes (fibers disabled) ────────────────")
timed("build_surface_fiber_meshes (disabled)",
    build_surface_fiber_meshes,
    vl,
    segments=int(config['knit_parameters']['segments']),
    enabled=False,
    count=1, radius=rad, radius_scale=0.15, lift=0.1,
    surface_arc=0.5, randomness=0.3, twist=2.0,
)

# 8. compute_normals
print("\n── 8. compute_normals (per tube) ───────────────────────────────────")
seg = int(config['knit_parameters']['segments'])
for idx, (verts, n_pts) in enumerate(vl[:3]):
    v = np.asarray(verts, dtype=np.float32)
    f = np.asarray(fl[idx], dtype=np.int32)
    tris = np.empty((len(f) * 2, 3), dtype=np.int32)
    tris[0::2] = f[:, [0, 1, 2]]
    tris[1::2] = f[:, [0, 2, 3]]
    timed(f"compute_normals (tube {idx}, {len(v)} verts)", compute_normals, v, tris)

# 9. Full prepare_meshes pipeline (triangulate + normals + pack)
print("\n── 9. Full prepare_meshes equivalent ──────────────────────────────")
def full_prepare(verts_list, faces_list):
    prepared = []
    for (verts, _n_points), faces in zip(verts_list, faces_list):
        v = np.array(verts, dtype=np.float32)
        f = np.array(faces, dtype=np.int32)
        tris = np.empty((len(f) * 2, 3), dtype=np.int32)
        tris[0::2] = f[:, [0, 1, 2]]
        tris[1::2] = f[:, [0, 2, 3]]
        nm = compute_normals(v, tris).astype(np.float32)
        prepared.append((v, tris, v.tobytes(), nm.tobytes(), tris.tobytes()))
    return prepared

timed("prepare_meshes (all tubes)", full_prepare, vl, fl)

# 10. Jacobian build
print("\n── 10. Jacobian build ─────────────────────────────────────────────")
import scipy.sparse

def build_full_jacobian():
    J_blocks = [build_row_spline_jacobian(cp, period_offset_x, nout) for cp in ctrl_rows]
    J_base = scipy.sparse.block_diag(J_blocks, format="csr")
    return scipy.sparse.kron(J_base, scipy.sparse.identity(3), format="csr")

J_cached = timed("build_full_jacobian", build_full_jacobian)
print(f"     J_cached shape = {J_cached.shape}, nnz = {J_cached.nnz}")

# 11. Simulation step
print("\n── 11. run_simulation_step ────────────────────────────────────────")
try:
    from yarn_simulation import run_simulation_step, eval_energy
    # Compute rest lengths
    if len(edges) > 0:
        L0 = np.linalg.norm(V[edges[:, 1]] - V[edges[:, 0]], axis=1)
    else:
        L0 = np.array([])

    period_offset_y = np.array([0.0, max(1, len(ctrl_rows)) * float(params[pidx['dy']]), 0.0], dtype=np.float32)

    # Just time eval_energy first (called in line-search)
    flat_P = np.concatenate(ctrl_rows).astype(float)
    timed("eval_energy (single call)", eval_energy,
          flat_P, ctrl_rows, period_offset_x, period_offset_y, config, L0, 0.01)

    # Full step
    timed("run_simulation_step (1 step)", run_simulation_step,
          ctrl_rows, period_offset_x, period_offset_y, config, J_cached, L0,
          1.0, 1.0, 1.0, 0.01,
          repeats=1)
except Exception as ex:
    print(f"  [SKIPPED] Simulation profiling failed: {ex}")

# 12. cProfile of the full rebuild pipeline
print("\n── 12. cProfile: full rebuild pipeline (5 iterations) ─────────────")
pr = cProfile.Profile()
pr.enable()
for _ in range(5):
    ctrl_rows_copy = [r.copy() for r in ctrl_rows]
    vl2 = build_spline_mesh(ctrl_rows_copy, params, config, pidx, period_offset_x, radius_ctrl_rows=radius_profiles)
    fl2 = compute_knitting_faces(seg, vl2)
    fiber_vl2, _ = build_surface_fiber_meshes(
        vl2, segments=seg, enabled=True, count=4, radius=rad,
        radius_scale=0.15, lift=0.1, surface_arc=0.5, randomness=0.3, twist=2.0)
    fl3 = compute_knitting_faces(seg, fiber_vl2)
    full_prepare(fiber_vl2, fl3)
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(30)
print(s.getvalue())

print("\n── cProfile by tottime (top 20) ────────────────────────────────────")
s2 = io.StringIO()
ps2 = pstats.Stats(pr, stream=s2).sort_stats('tottime')
ps2.print_stats(20)
print(s2.getvalue())

print("=" * 78)
print("Profiling complete.")
