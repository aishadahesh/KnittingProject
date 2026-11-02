"""Verify the new Y period formula"""
import json, numpy as np, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

with open(os.path.join('config', 'config.json')) as f:
    cfg = json.load(f)
with open(os.path.join('config', 'params.json')) as f:
    d = json.load(f)

pidx = {p['name']: i for i, p in enumerate(cfg['knit_parameters']['parameters'])}
p_dict = d.get('params', {})
params = [pd['initial'] for pd in cfg['knit_parameters']['parameters']]
for i, pd in enumerate(cfg['knit_parameters']['parameters']):
    if pd['name'] in p_dict:
        params[i] = float(np.clip(p_dict[pd['name']], pd['range'][0], pd['range'][1]))

ctrl = [np.array(r, dtype=np.float32) for r in d.get('spline_control_rows', [])]
radius = params[pidx['radius']]

from knitting_core import build_spline_mesh
period_x = np.array(d.get('period_offset_x', [2, 0, 0]), dtype=np.float32)
rp = [np.full(len(r), radius, dtype=np.float32) for r in ctrl]
vl = build_spline_mesh(ctrl, np.array(params, dtype=np.float32), cfg, pidx, period_x, radius_ctrl_rows=rp)

# Simulate what the FIXED _display_copy_y_period does
all_v = np.vstack([v for v, _ in vl])
model_h = float(all_v[:, 1].max() - all_v[:, 1].min())
row_centers = sorted(float(np.mean(r[:, 1])) for r in ctrl)
row_pitch = float(np.median(np.diff(row_centers)))
new_y_period = model_h - row_pitch

dy = params[pidx['dy']]
old_y_period = abs(dy) * len(ctrl)

print(f"OLD y_period (dy * rows):              {old_y_period:.4f}  <-- BROKEN")
print(f"NEW y_period (model_h - row_pitch):    {new_y_period:.4f}  <-- FIXED")
print(f"model_height:                          {model_h:.4f}")
print(f"row_pitch:                             {row_pitch:.4f}")
print()
print(f"With OLD: copy 2 starts at Y={old_y_period:.2f}, but model extends to Y={model_h:.2f} -> OVERLAP of {model_h - old_y_period:.2f}")
print(f"With NEW: copy 2 starts at Y={new_y_period:.2f}, gap = {new_y_period - (model_h - row_pitch):.4f} (should be ~0)")
