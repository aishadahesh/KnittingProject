# %% IMPORTS
import os
import time
import json
import numpy as np
from PIL import Image
import mitsuba as mi
import drjit as dr
import jax.numpy as jnp
import jax 
import optax
import matplotlib.pyplot as plt
from functools import partial

# Conditional imports for UI components
try:
    from vedo import Plotter, Mesh, Text2D, Sphere, Plane, Spline
except ImportError:
    print("vedo not installed. Interactive App will be unavailable.")

# %% CONFIGURATION LOADING

def load_config(config_path="config.json"):
    """Loads project configuration from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

CONFIG = load_config()

# Set Mitsuba variant from configuration
if not mi.variant():
    try:
        mi.set_variant(CONFIG['rendering']['mitsuba_variant'])
    except Exception:
        mi.set_variant(CONFIG['rendering']['mitsuba_variant_fallback'])

# Initialize output directories
OUTPUT_DIR = CONFIG['rendering']['output_dir']
for sub_dir in ["meshes", "renders"]:
    os.makedirs(os.path.join(OUTPUT_DIR, sub_dir), exist_ok=True)

# %% GEOMETRY ENGINE (JAX)

@jax.jit
def eval_curve_batch(t, scale, stitch_bulge, stitch_z):
    """Vectorized evaluation of the knitting curve geometry."""
    x = stitch_bulge * jnp.sin(2 * t) + t / (2 * jnp.pi)
    y = -(jnp.cos(t) - 1) / 2
    z = stitch_z * (jnp.cos(2 * t) - 1) / 2
    x = jnp.where(scale == 0, t / (2 * jnp.pi), x)
    return jnp.stack([x, y * scale, z * scale], axis=-1)

@jax.jit
def eval_curve_derivative_batch(t, scale, stitch_bulge, stitch_z):
    """Vectorized evaluation of the knitting curve derivatives."""
    dx = 2 * stitch_bulge * jnp.cos(2 * t) + 1 / (2 * jnp.pi)
    dy = 0.5 * jnp.sin(t) * scale
    dz = -stitch_z * jnp.sin(2 * t) * scale
    dx = jnp.where(scale == 0, 1 / (2 * jnp.pi), dx)
    return jnp.stack([dx, dy, dz], axis=-1)

@jax.jit
def compute_orthonormal_frame_batch(tangent):
    """Computes orthonormal frames along the curve for tube generation."""
    tangent = tangent / (jnp.linalg.norm(tangent, axis=-1, keepdims=True) + 1e-8)
    ref = jnp.array([0.0, 0.0, 1.0])
    normal_u = jnp.cross(tangent, ref)
    u_norm = jnp.linalg.norm(normal_u, axis=-1, keepdims=True)
    
    # Handle parallel cases to avoid gimbal lock
    alt_ref = jnp.array([1.0, 0.0, 0.0])
    normal_u = jnp.where(u_norm < 1e-6, jnp.cross(tangent, alt_ref), normal_u)
    
    normal_u = normal_u / (jnp.linalg.norm(normal_u, axis=-1, keepdims=True) + 1e-8)
    normal_v = jnp.cross(tangent, normal_u)
    return normal_u, normal_v

@partial(jax.jit, static_argnums=(2, 3))
def compute_knitting_vertices_jit(geometry_params, bitmap, loop_res, segments):
    """JIT-compiled function to generate all mesh vertices for the pattern."""
    (stitch_bulge, stitch_z, loop_height, dy, radius, 
     _, _, _, _, ellipse_ratio) = geometry_params
    
    def count_consecutive_zeros(row):
        n = len(row)
        indices = jnp.arange(n)
        mask = (indices[:, None] < indices[None, :])
        masked_row = jnp.where(mask, row[None, :], 999)
        nonzero_mask = (masked_row != 0) & mask
        first_nonzero = jnp.argmax(nonzero_mask.astype(jnp.int32), axis=1)
        counts = jnp.where(jnp.any(nonzero_mask, axis=1), 
                           first_nonzero - indices, n - indices - 1)
        return jnp.where(row == 1, counts + 1, 0)
    
    scale_factor = jax.vmap(count_consecutive_zeros)(bitmap)
    scale_factor = jnp.where((scale_factor <= 1), scale_factor, 
                             1 + dy * (scale_factor - 1))
    
    n_rows, n_loops = bitmap.shape
    t_vals = jnp.linspace(0.0, 2 * jnp.pi * n_loops, loop_res * n_loops + 1)
    
    def process_row(row_idx, row_scales):
        x_scale = jnp.append(jnp.repeat(row_scales, loop_res), 1.0)
        pos = eval_curve_batch(t_vals, x_scale, stitch_bulge, stitch_z)
        pos = pos.at[:, 1].add(row_idx * dy)
        d_pos = eval_curve_derivative_batch(t_vals, x_scale, stitch_bulge, stitch_z)
        u_frame, v_frame = compute_orthonormal_frame_batch(d_pos)
        
        angles = jnp.linspace(0, 2 * jnp.pi, segments, endpoint=False)
        offsets = (u_frame[:, None, :] * jnp.cos(angles)[None, :, None] * 
                   radius * ellipse_ratio + 
                   v_frame[:, None, :] * jnp.sin(angles)[None, :, None] * radius)
        
        return (pos[:, None, :] + offsets).reshape(-1, 3)

    return jax.vmap(process_row)(jnp.arange(n_rows), scale_factor)

def compute_knitting_vertices(geometry_params, bitmap):
    """High-level wrapper for JIT vertex computation."""
    res = CONFIG['geometry']['loop_res']
    seg = CONFIG['geometry']['segments']
    vertices = compute_knitting_vertices_jit(
        jnp.array(geometry_params), jnp.array(bitmap), res, seg
    )
    return [(v, len(v) // seg) for v in vertices]

def compute_knitting_faces(segments, verts_list):
    """Computes mesh faces based on vertex counts for each row."""
    faces_list = []
    for _, n_points in verts_list:
        i_grid, j_grid = np.meshgrid(np.arange(n_points - 1), 
                                     np.arange(segments), indexing='ij')
        v0 = i_grid * segments + j_grid
        v1 = i_grid * segments + (j_grid + 1) % segments
        v2 = (i_grid + 1) * segments + (j_grid + 1) % segments
        v3 = (i_grid + 1) * segments + j_grid
        faces_list.append(np.stack([v0, v1, v2, v3], axis=-1).reshape(-1, 4))
    return faces_list

def compute_geometry_jacobian(geometry_params, bitmap):
    """Computes the Jacobian of vertex positions using JAX autodiff."""
    res, seg = CONFIG['geometry']['loop_res'], CONFIG['geometry']['segments']
    def get_all_vertices(params):
        verts = compute_knitting_vertices_jit(params, bitmap, res, seg)
        return verts.flatten()
    jacobian = jax.jacfwd(get_all_vertices)(geometry_params)
    return jacobian.reshape(-1, 3, len(geometry_params))

# %% MESH IO

def save_combined_obj(mesh_data_list, base_filename="knitting_model"):
    """Saves all mesh parts into a single combined OBJ file."""
    combined_filename = f"{base_filename}_combined.obj"
    vertex_offset = 0
    with open(combined_filename, 'w') as f:
        f.write("# Knitting Model\n\n")
        for i, (verts, _, faces, _) in enumerate(mesh_data_list):
            f.write(f"o mesh_{i}\n")
            for v in verts: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                idx_str = ' '.join([str(int(idx) + vertex_offset + 1) for idx in face])
                f.write(f"f {idx_str}\n")
            vertex_offset += len(verts)

def save_per_loop_objs(mesh_data_list, base_filename, loop_res, segments):
    """Saves each stitch loop as a separate OBJ for pattern analysis."""
    obj_info = []
    for row_idx, (verts, _, _, n_points) in enumerate(mesh_data_list):
        n_loops = (n_points - 1) // loop_res
        for loop_idx in range(n_loops):
            v_start = loop_idx * loop_res * segments
            v_end = (loop_idx + 1) * loop_res * segments + segments
            l_verts = verts[v_start:v_end]
            n_l_pts = (v_end - v_start) // segments
            i_g, j_g = np.meshgrid(np.arange(n_l_pts - 1), 
                                   np.arange(segments), indexing='ij')
            v0, v1 = i_g * segments + j_g, i_g * segments + (j_g + 1) % segments
            v2, v3 = ((i_g + 1) * segments + (j_g + 1) % segments, 
                      (i_g + 1) * segments + j_g)
            l_faces = np.stack([v0, v1, v2, v3], axis=-1).reshape(-1, 4)
            path = f"{base_filename}_r{row_idx:02d}_l{loop_idx:02d}.obj"
            with open(path, 'w') as f:
                for v in l_verts: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                for fa in l_faces: 
                    f.write(f"f {' '.join([str(int(x)+1) for x in fa])}\n")
            obj_info.append((row_idx, loop_idx, path))
    return obj_info

# %% OPTIMIZATION ENGINE

def get_loop_color(row_idx, loop_idx):
    """Returns the color for a specific loop following the target pattern."""
    palette = [(0.45, 0.25, 0.15), (0.15, 0.35, 0.75), 
               (0.95, 0.85, 0.20), (0.85, 0.20, 0.20)]
    row_pattern = row_idx % 3
    if row_pattern == 0: return palette[0]
    if row_pattern == 1: return palette[3] if loop_idx % 2 == 0 else palette[2]
    return palette[1]

class KnittingOptimizer:
    """Handles the differentiable rendering and gradient descent loop."""
    def __init__(self, reference_img, bitmap):
        self.bitmap = bitmap
        self.iteration = 0
        opt_cfg = CONFIG['optimization']
        self.optimizer = optax.adam(opt_cfg['learning_rate'])
        self.opt_state = None
        self.ref_array = np.array(reference_img).astype(np.float32) / 255.0
        self.res_height, self.res_width = self.ref_array.shape[:2]
        self.loss_weights = opt_cfg['loss_weights']
        self.loss_mask = mi.TensorXf(self._build_mask(opt_cfg['loss_center_crop']))
        self.ref_tensor = mi.TensorXf(self.ref_array)
        self.loss_history, self.param_history = [], []
        rc = CONFIG['rendering']
        self.camera_params = (rc['camera_dist_mult'], rc['camera_fov'])
        self.row_colors = [tuple(c) for c in CONFIG['ui']['yarn_colors']]
        self.row_colors = [self.row_colors[i % len(self.row_colors)] 
                           for i in range(bitmap.shape[0])]

    def _build_mask(self, crop):
        cw, ch = crop
        mask = np.zeros((self.res_height, self.res_width, 3), dtype=np.float32)
        x0, x1 = int((1 - cw) * 0.5 * self.res_width), int((1 + cw) * 0.5 * self.res_width)
        y0, y1 = int((1 - ch) * 0.5 * self.res_height), int((1 + ch) * 0.5 * self.res_height)
        mask[max(0, y0):min(self.res_height, y1), 
             max(0, x0):min(self.res_width, x1), :] = 1.0
        return mask

    def get_scene_dict(self, obj_path, params, camera_params=None):
        """Generates a Mitsuba scene dictionary for the current model state."""
        dm, fov = camera_params if camera_params else self.camera_params
        verts_list = compute_knitting_vertices(params, self.bitmap)
        all_verts = jnp.concatenate([v for v, _ in verts_list])
        vmin, vmax = np.min(all_verts, axis=0), np.max(all_verts, axis=0)
        center = (vmin + vmax) * 0.5
        dist = max((vmax[1]-vmin[1])/np.tan(np.deg2rad(fov)*0.5), 
                   (vmax[0]-vmin[0])/np.tan(np.deg2rad(fov)*0.5 * 
                   self.res_width/self.res_height)) * 0.5 * dm + (vmax[2]-vmin[2])*0.2
        return {
            "type": "scene", "integrator": {"type": "path", "max_depth": 2},
            "sensor": {
                "type": "perspective", "fov": fov,
                "to_world": mi.ScalarTransform4f.look_at(
                    origin=[float(center[0]), float(center[1]), float(vmax[2]+dist)], 
                    target=[float(center[0]), float(center[1]), float(center[2])], 
                    up=[0, 1, 0]
                ),
                "film": {"type": "hdrfilm", "width": self.res_width, 
                         "height": self.res_height, "pixel_format": "rgb"},
            },
            "emitter": {"type": "constant", "radiance": {"type": "rgb", "value": [0.9, 0.9, 0.9]}},
            "mesh": { "type": "obj", "filename": obj_path, 
                      "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.8, 0.4, 0.3]}} }
        }

    def step(self, params):
        """Performs a single optimization step using differentiable rendering."""
        self.iteration += 1
        params_np = np.array(params)
        verts_list = compute_knitting_vertices(params_np, self.bitmap)
        faces_list = compute_knitting_faces(CONFIG['geometry']['segments'], verts_list)
        mesh_data = [(v, [], f, n) for (v, n), f in zip(verts_list, faces_list)]
        
        base_path = os.path.join(OUTPUT_DIR, "meshes", f"epoch_{self.iteration:03d}")
        save_combined_obj(mesh_data, base_path)
        scene = mi.load_dict(self.get_scene_dict(base_path + "_combined.obj", params_np))
        params_scene = mi.traverse(scene)
        vertex_key = [k for k in params_scene.keys() if 'vertex_positions' in k]
        
        if vertex_key:
            vpos = params_scene[vertex_key[0]]
            dr.enable_grad(vpos)
            params_scene.update()
            img = mi.render(scene, params=params_scene, spp=CONFIG['rendering']['spp_optimization'])
            diff = (img - self.ref_tensor) * self.loss_mask
            loss_dr = dr.sum(dr.sqr(diff)) / (dr.sum(self.loss_mask) + 1e-8)
            dr.backward(loss_dr)
            jacobian = compute_geometry_jacobian(params_np, self.bitmap)
            vertex_grads_np = np.array(dr.grad(vpos)).reshape(-1, 3)
            grads = np.array([np.sum(vertex_grads_np * jacobian[:, :, i]) for i in range(len(params_np))])
            current_loss = float(dr.ravel(loss_dr)[0])
        else:
            grads, current_loss = np.zeros(len(params_np)), 0.0
            
        if self.opt_state is None:
            self.opt_state = self.optimizer.init(jnp.array(params))
        updates, self.opt_state = self.optimizer.update(jnp.array(grads), self.opt_state)
        new_params = jnp.clip(optax.apply_updates(jnp.array(params), updates), 
                              jnp.array([r[0] for r in CONFIG['geometry']['param_ranges']]), 
                              jnp.array([r[1] for r in CONFIG['geometry']['param_ranges']]))
        self.loss_history.append(current_loss)
        self.param_history.append(new_params)
        print(f"Epoch {self.iteration:02d} | Loss: {current_loss:.6f}")
        return new_params

# %% UNIFIED INTERACTIVE APP

class KnittingApp:
    """The unified GUI for model editing and optimization."""
    def __init__(self, optimizer, init_params):
        self.optimizer, self.params = optimizer, np.array(init_params, dtype=np.float64)
        self.bitmap = optimizer.bitmap
        self.param_idx, self.mode = 0, 'parameter'
        self.selected_pt, self.dragging = None, False
        
        verts_list = compute_knitting_vertices(self.params, self.bitmap)
        faces_list = compute_knitting_faces(CONFIG['geometry']['segments'], verts_list)
        self.mesh_faces = []
        for faces in faces_list:
            f_np = np.array(faces)
            tris = np.empty((len(f_np) * 2, 3), dtype=np.int32)
            tris[0::2], tris[1::2] = f_np[:, [0, 1, 2]], f_np[:, [0, 2, 3]]
            self.mesh_faces.append(tris)

        self.plotter = Plotter(bg='blackboard', axes=1, title="Knitting Unified Optimizer")
        self.plotter.interactor.RemoveObservers('CharEvent')
        self.plotter.interactor.RemoveObservers('KeyPressEvent')
        
        self.actors = [Mesh([np.array(v), self.mesh_faces[i]]).color(self.optimizer.row_colors[i]).lighting('plastic').alpha(0.8) for i, (v, n) in enumerate(verts_list)]
        self.plotter.add(self.actors)
        
        self.ctrl_pts, self.ctrl_actors = [], []
        self._init_splines()
        
        self.plane = Plane(pos=[0,0,0], normal=[0,0,1], s=[100,100]).alpha(0).pickable(True)
        self.plotter.add(self.plane)
        self.txt = None
        self._update_display(rebuild=False)
        
        self.plotter.add_callback("KeyPress", self._on_key_press)
        self.plotter.add_callback("LeftButtonPress", self._on_click)
        self.plotter.add_callback("MouseMove", self._on_move)
        self.plotter.add_callback("LeftButtonRelease", self._on_release)

    def _init_splines(self):
        bulge, z, _, dy, _ = self.params[:5]
        n_r, n_c = self.bitmap.shape
        for r in range(n_r):
            pts = []
            cols = range(n_c) if r % 2 == 0 else range(n_c - 1, -1, -1)
            for c in cols:
                t_v = np.linspace(0, 2 * np.pi, 5, endpoint=False)
                if r % 2 != 0: t_v = t_v[::-1]
                for t in t_v:
                    pts.append([c + bulge * np.sin(2*t) + t/(2*np.pi), 
                                r * dy - (np.cos(t)-1)/2, z*(np.cos(2*t)-1)/2])
            pts.append([n_c if r % 2 == 0 else 0, r * dy, 0])
            self.ctrl_pts.append(np.array(pts))
            for p in pts:
                s = Sphere(p, r=0.08).color("white").pickable(True)
                self.ctrl_actors.append(s)
                self.plotter.add(s)

    def _sync_splines_from_params(self):
        """Updates control points to match current global parameters."""
        bulge, z, _, dy, _ = self.params[:5]
        n_r, n_c = self.bitmap.shape
        flat_idx = 0
        for r in range(n_r):
            pts = []
            cols = range(n_c) if r % 2 == 0 else range(n_c - 1, -1, -1)
            for c in cols:
                t_v = np.linspace(0, 2 * np.pi, 5, endpoint=False)
                if r % 2 != 0: t_v = t_v[::-1]
                for t in t_v:
                    p = [c + bulge * np.sin(2 * t) + t / (2 * jnp.pi), 
                         r * dy - (jnp.cos(t) - 1) / 2, z * (jnp.cos(2 * t) - 1) / 2]
                    pts.append(p); self.ctrl_actors[flat_idx].pos(p); flat_idx += 1
            p_end = [n_c if r % 2 == 0 else 0, r * dy, 0]
            pts.append(p_end); self.ctrl_actors[flat_idx].pos(p_end); flat_idx += 1
            self.ctrl_pts[r] = np.array(pts)

    def _sync_mesh_from_splines(self):
        """Vectorized mesh generation from spline interpolated control points."""
        res, seg = CONFIG['geometry']['loop_res'], CONFIG['geometry']['segments']
        radius, ratio = self.params[4], self.params[9]
        for r, row_pts in enumerate(self.ctrl_pts):
            diff = np.diff(row_pts, axis=0)
            mask = np.ones(len(row_pts), dtype=bool)
            mask[1:] = np.linalg.norm(diff, axis=1) > 1e-5
            unique_pts = row_pts[mask]
            if len(unique_pts) < 2: continue
            try:
                pts = Spline(unique_pts, res=res * self.bitmap.shape[1] + 1).points
                T = np.gradient(pts, axis=0)
                T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-8)
                U = np.cross(T, [0,0,1])
                mask_U = np.linalg.norm(U, axis=1) < 1e-6
                U[mask_U] = np.cross(T[mask_U], [1,0,0])
                U /= (np.linalg.norm(U, axis=1, keepdims=True) + 1e-8)
                V = np.cross(T, U)
                angles = np.linspace(0, 2*np.pi, seg, endpoint=False)
                offsets = U[:,None,:] * np.cos(angles)[None,:,None] * radius * ratio + V[:,None,:] * np.sin(angles)[None,:,None] * radius
                self.actors[r].points = (pts[:,None,:] + offsets).reshape(-1, 3)
            except Exception: pass

    def _update_display(self, rebuild=True):
        if rebuild:
            if self.mode == 'parameter':
                verts_list = compute_knitting_vertices(self.params, self.bitmap)
                for i, (v, n) in enumerate(verts_list): self.actors[i].points = np.array(v)
            else: self._sync_mesh_from_splines()
        if self.txt: self.plotter.remove(self.txt)
        param_names = CONFIG['geometry']['param_names']
        mode_label = f"MODE: {self.mode.upper()} (T to toggle)"
        lines = [mode_label, "---"]
        if self.mode == 'parameter':
            lines += [f"{'>>>' if i==self.param_idx else '   '} {param_names[i]}: {self.params[i]:.4f}" for i in range(len(param_names))]
        else:
            lines += [f"Selected Point: {self.selected_pt}", "Use WASD/QE to move point"]
        self.txt = Text2D("\n".join(lines + ["", "KEYS: arrows=adj, R=render, O=optimize, F=finish"]), pos='top-left', s=0.7, bg='black', alpha=0.5)
        self.plotter.add(self.txt).render()

    def _on_key_press(self, event):
        key = event.keypress.lower() if event.keypress else ""
        if key == 't': 
            self.mode = 'spline' if self.mode == 'parameter' else 'parameter'
            if self.mode == 'spline': self._sync_splines_from_params()
            else: self._estimate_params_from_splines()
        elif key == 'r': self._trigger_render()
        elif key == 'o': self._trigger_optimization()
        elif key == 'f': self.plotter.close()
        elif self.mode == 'parameter':
            if key in ['up', 'down']: self.param_idx = (self.param_idx + (-1 if key == 'up' else 1)) % len(self.params)
            elif key in ['left', 'right']:
                self.params[self.param_idx] += (-1 if key == 'left' else 1) * CONFIG['geometry']['param_deltas'][self.param_idx]
                self.params[self.param_idx] = np.clip(self.params[self.param_idx], *CONFIG['geometry']['param_ranges'][self.param_idx])
        elif self.mode == 'spline' and self.selected_pt:
            r, i = self.selected_pt; p = self.ctrl_pts[r][i]; d = 0.05
            if key == 'w': p[1] += d
            elif key == 's': p[1] -= d
            elif key == 'a': p[0] -= d
            elif key == 'd': p[0] += d
            elif key == 'q': p[2] += d
            elif key == 'e': p[2] -= d
            flat_idx = sum(len(x) for x in self.ctrl_pts[:r]) + i
            self.ctrl_actors[flat_idx].pos(p)
        self._update_display()

    def _estimate_params_from_splines(self):
        """Heuristic to update global parameters based on manual spline edits."""
        flat_pts = np.concatenate(self.ctrl_pts)
        self.params[3] = np.mean(np.diff([np.mean(row[:,1]) for row in self.ctrl_pts])) if len(self.ctrl_pts)>1 else self.params[3]
        z_range = np.max(flat_pts[:,2]) - np.min(flat_pts[:,2])
        self.params[1] = -z_range / 2 if z_range > 0 else self.params[1]

    def _on_click(self, event):
        if self.mode != 'spline' or not event.actor: return
        for flat_idx, s in enumerate(self.ctrl_actors):
            if event.actor == s:
                if self.selected_pt:
                    old_r, old_i = self.selected_pt
                    self.ctrl_actors[sum(len(x) for x in self.ctrl_pts[:old_r]) + old_i].color("white")
                cumsum = 0
                for r, row in enumerate(self.ctrl_pts):
                    if flat_idx < cumsum + len(row): self.selected_pt = (r, flat_idx - cumsum); break
                    cumsum += len(row)
                s.color("red"); self.dragging = True; break
        self._update_display(rebuild=False)

    def _on_move(self, event):
        if not self.dragging or self.selected_pt is None or event.picked3d is None: return
        r, i = self.selected_pt; new_p = np.array(event.picked3d)
        new_p[2] = self.ctrl_pts[r][i][2]
        self.ctrl_pts[r][i] = new_p
        flat_idx = sum(len(x) for x in self.ctrl_pts[:r]) + i
        self.ctrl_actors[flat_idx].pos(new_p)
        self._update_display()

    def _on_release(self, event): self.dragging = False

    def _trigger_render(self):
        verts_list = compute_knitting_vertices(self.params, self.bitmap)
        mesh_data = [(v, [], f, n) for (v, n), f in zip(verts_list, compute_knitting_faces(CONFIG['geometry']['segments'], verts_list))]
        save_combined_obj(mesh_data, "temp_preview")
        img = mi.render(mi.load_dict(self.optimizer.get_scene_dict("temp_preview_combined.obj", self.params)), spp=64)
        plt.figure("Preview"); plt.imshow(np.clip(np.array(img), 0, 1)); plt.axis('off'); plt.show()

    def _trigger_optimization(self):
        self.params, _ = run_optimization_loop(self.optimizer, self.params); self._update_display()

    def run(self): self.plotter.show(interactive=True); return self.params

def run_optimization_loop(optimizer, params):
    """Executes the optimization loop based on configuration."""
    cfg = CONFIG['optimization']
    best_l, best_p, count = float('inf'), params, 0
    for _ in range(cfg['max_epochs']):
        params = optimizer.step(params)
        l = optimizer.loss_history[-1]
        if l < best_l - 1e-6: best_l, best_p, count = l, params, 0
        elif (count := count + 1) >= cfg['patience']: break
    return best_p, best_l

if __name__ == "__main__":
    ref = Image.open(CONFIG['ui']['reference_image']).convert("RGB")
    bitmap = jnp.ones((CONFIG['geometry']['bitmap_rows'], CONFIG['geometry']['bitmap_loops']))
    opt = KnittingOptimizer(ref, bitmap)
    final_params = KnittingApp(opt, CONFIG['geometry']['initial_params']).run()
    v_l = compute_knitting_vertices(final_params, bitmap)
    save_combined_obj([(v, [], f, n) for (v, n), f in zip(v_l, compute_knitting_faces(CONFIG['geometry']['segments'], v_l))], "best_model")
