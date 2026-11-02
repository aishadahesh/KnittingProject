"""Quick diagnostic: compare Y period values"""
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
dy = params[pidx['dy']]
radius = params[pidx['radius']]

print(f"rows: {len(ctrl)}")
print(f"dy: {dy:.4f}")
print(f"radius: {radius:.4f}")
print(f"Y period (dy * rows): {abs(dy) * len(ctrl):.4f}")

if ctrl:
    y_centers = [float(np.mean(r[:, 1])) for r in ctrl]
    print(f"row Y centers: {[f'{c:.3f}' for c in y_centers]}")
    if len(y_centers) > 1:
        pitches = np.diff(np.sort(y_centers))
        print(f"row pitches: {[f'{p:.4f}' for p in pitches]}")
        median_pitch = float(np.median(pitches))
        print(f"median pitch: {median_pitch:.4f}")
        print(f"OLD y_period (pitch * (rows-1)): {median_pitch * (len(y_centers)-1):.4f}")
        print(f"OLD y_period (pitch * rows):     {median_pitch * len(y_centers):.4f}")

    # Also compute actual mesh bounds
    from knitting_core import build_spline_mesh
    period_x = np.array(d.get('period_offset_x', [float(len(ctrl)), 0, 0]), dtype=np.float32)
    radius_profiles = [np.full(len(r), radius, dtype=np.float32) for r in ctrl]
    vl = build_spline_mesh(ctrl, np.array(params, dtype=np.float32), cfg, pidx, period_x, radius_ctrl_rows=radius_profiles)
    all_v = np.vstack([v for v, _ in vl])
    print(f"\nActual mesh bounds:")
    print(f"  Y min: {all_v[:, 1].min():.4f}")
    print(f"  Y max: {all_v[:, 1].max():.4f}")
    print(f"  Y span (model_h): {all_v[:, 1].max() - all_v[:, 1].min():.4f}")
    print(f"  X span (model_w): {all_v[:, 0].max() - all_v[:, 0].min():.4f}")
    print(f"  Z span: {all_v[:, 2].max() - all_v[:, 2].min():.4f}")

    model_h = all_v[:, 1].max() - all_v[:, 1].min()
    INSET_Y = 0.20
    scanner_y_period = max(model_h - 2.0 * INSET_Y * radius, radius)
    print(f"\nScanner y_period: {scanner_y_period:.4f}")
    print(f"dy * rows:        {abs(dy) * len(ctrl):.4f}")
    print(f"DIFFERENCE:       {scanner_y_period - abs(dy) * len(ctrl):.4f}")
