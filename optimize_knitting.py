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

# %% CONFIGURATION LOADING

def load_config(config_path="config.json"):
    """Loads project configuration from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

CONFIG = load_config()

# Ensure Mitsuba is in a differentiable mode
if not mi.variant():
    try:
        mi.set_variant(CONFIG['rendering']['mitsuba_variant'])
    except:
        mi.set_variant(CONFIG['rendering']['mitsuba_variant_fallback'])

OUTPUT_DIR = CONFIG['rendering']['output_dir']
for d in ["meshes", "renders"]: os.makedirs(os.path.join(OUTPUT_DIR, d), exist_ok=True)

# %% GEOMETRY ENGINE (JAX) - FULLY JIT OPTIMIZED

@jax.jit
def eval_curve_batch(t, scale, stitch_bulge, stitch_z):
    """Vectorized evaluation of the knitting curve geometry."""
    x = stitch_bulge * jnp.sin(2*t) + t/(2*jnp.pi)
    y = -(jnp.cos(t) - 1)/2
    z = stitch_z * (jnp.cos(2*t) - 1)/2
    x = jnp.where(scale == 0, t/(2*jnp.pi), x)
    return jnp.stack([x, y * scale, z * scale], axis=-1)

@jax.jit
def eval_curve_derivative_batch(t, scale, stitch_bulge, stitch_z):
    """Vectorized evaluation of the knitting curve derivatives."""
    dx = 2*stitch_bulge*jnp.cos(2*t) + 1/(2*jnp.pi)
    dy = 0.5*jnp.sin(t)*scale
    dz = -stitch_z*jnp.sin(2*t)*scale
    dx = jnp.where(scale == 0, 1/(2*jnp.pi), dx)
    return jnp.stack([dx, dy, dz], axis=-1)

@jax.jit
def compute_orthonormal_frame_batch(T):
    """Computes orthonormal frames along the curve for mesh generation."""
    T = T / (jnp.linalg.norm(T, axis=-1, keepdims=True) + 1e-8)
    ref = jnp.array([0.0, 0.0, 1.0])
    U = jnp.cross(T, ref)
    U_norm = jnp.linalg.norm(U, axis=-1, keepdims=True)
    U = jnp.where(U_norm < 1e-6, jnp.cross(T, jnp.array([1.0, 0.0, 0.0])), U)
    U = U / (jnp.linalg.norm(U, axis=-1, keepdims=True) + 1e-8)
    V = jnp.cross(T, U)
    return U, V

@partial(jax.jit, static_argnums=(2, 3))
def compute_knitting_vertices_jit(geometry_params, bitmap, loop_res, segments):
    """JIT-compiled function to generate all mesh vertices for the knitting pattern."""
    (stitch_bulge, stitch_z, loop_height, dy, radius, _, _, _, _, ellipse_ratio) = geometry_params
    
    # Vectorized bitmap processing to find consecutive zeros
    def count_zeros(row):
        n = len(row)
        indices = jnp.arange(n)
        mask = (indices[:, None] < indices[None, :])
        masked = jnp.where(mask, row[None, :], 999)
        nonzero_mask = (masked != 0) & mask
        first_nz = jnp.argmax(nonzero_mask.astype(jnp.int32), axis=1)
        counts = jnp.where(jnp.any(nonzero_mask, axis=1), first_nz - indices, n - indices - 1)
        return jnp.where(row == 1, counts + 1, 0)
    
    scale_factor = jax.vmap(count_zeros)(bitmap)
    scale_factor = jnp.where((scale_factor <= 1), scale_factor, 1 + dy * (scale_factor - 1))
    
    n_rows, n_loops = bitmap.shape
    t = jnp.linspace(0.0, 2 * jnp.pi * n_loops, loop_res * n_loops + 1)
    
    def process_row(row_idx, row_scales):
        x_scale = jnp.append(jnp.repeat(row_scales, loop_res), 1.0)
        p = eval_curve_batch(t, x_scale, stitch_bulge, stitch_z)
        p = p.at[:, 1].add(row_idx * dy)
        dp = eval_curve_derivative_batch(t, x_scale, stitch_bulge, stitch_z)
        U, V = compute_orthonormal_frame_batch(dp)
        
        angles = jnp.linspace(0, 2 * jnp.pi, segments, endpoint=False)
        offsets = (U[:, None, :] * jnp.cos(angles)[None, :, None] * radius * ellipse_ratio + 
                   V[:, None, :] * jnp.sin(angles)[None, :, None] * radius)
        
        return (p[:, None, :] + offsets).reshape(-1, 3)

    return jax.vmap(process_row)(jnp.arange(n_rows), scale_factor)

def compute_knitting_vertices(geometry_params, bitmap):
    """Wrapper for JIT vertex computation that handles parameter conversion."""
    res = CONFIG['geometry']['loop_res']
    seg = CONFIG['geometry']['segments']
    verts = compute_knitting_vertices_jit(jnp.array(geometry_params), jnp.array(bitmap), res, seg)
    return [(v, len(v) // seg) for v in verts]

def compute_knitting_faces(segments, verts_list):
    """Computes mesh faces based on vertex counts."""
    faces_list = []
    for _, n_points in verts_list:
        i_grid, j_grid = np.meshgrid(np.arange(n_points - 1), np.arange(segments), indexing='ij')
        v0, v1 = i_grid * segments + j_grid, i_grid * segments + (j_grid + 1) % segments
        v2, v3 = (i_grid + 1) * segments + (j_grid + 1) % segments, (i_grid + 1) * segments + j_grid
        faces_list.append(np.stack([v0, v1, v2, v3], axis=-1).reshape(-1, 4))
    return faces_list

def compute_geometry_jacobian(geometry_params, bitmap):
    """Computes the Jacobian of vertex positions with respect to parameters."""
    res, seg = CONFIG['geometry']['loop_res'], CONFIG['geometry']['segments']
    def get_all_verts(p):
        v = compute_knitting_vertices_jit(p, bitmap, res, seg)
        return v.flatten()
    return jax.jacfwd(get_all_verts)(geometry_params).reshape(-1, 3, len(geometry_params))

# %% MESH IO & UTILS

def save_combined_obj(mesh_data_list, base_filename="knitting_model"):
    """Saves multiple mesh parts into a single combined OBJ file."""
    combined_filename = f"{base_filename}_combined.obj"
    vertex_offset = 0
    with open(combined_filename, 'w') as f:
        f.write("# Knitting Model\n\n")
        for i, (verts, edges, faces, n_points) in enumerate(mesh_data_list):
            f.write(f"o mesh_{i}\n")
            for vert in verts: f.write(f"v {vert[0]:.6f} {vert[1]:.6f} {vert[2]:.6f}\n")
            for face in faces:
                f.write(f"f {' '.join([str(int(idx) + vertex_offset + 1) for idx in face])}\n")
            vertex_offset += len(verts)

def save_per_loop_objs(mesh_data_list, base_filename, loop_res, segments):
    """Saves each loop of the knit as a separate OBJ for discrete coloring."""
    obj_info = []
    for row_idx, (verts, _, faces, n_points) in enumerate(mesh_data_list):
        n_loops = (n_points - 1) // loop_res
        for loop_idx in range(n_loops):
            v_start, v_end = loop_idx * loop_res * segments, (loop_idx + 1) * loop_res * segments + segments
            l_verts = verts[v_start:v_end]
            n_l_pts = (v_end - v_start) // segments
            i_g, j_g = np.meshgrid(np.arange(n_l_pts - 1), np.arange(segments), indexing='ij')
            v0, v1, v2, v3 = i_g*segments+j_g, i_g*segments+(j_g+1)%segments, (i_g+1)*segments+(j_g+1)%segments, (i_g+1)*segments+j_g
            l_faces = np.stack([v0, v1, v2, v3], axis=-1).reshape(-1, 4)
            path = f"{base_filename}_r{row_idx}_l{loop_idx}.obj"
            with open(path, 'w') as f:
                for v in l_verts: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                for fa in l_faces: f.write(f"f {' '.join([str(int(x)+1) for x in fa])}\n")
            obj_info.append((row_idx, loop_idx, path))
    return obj_info

# %% OPTIMIZATION ENGINE

def get_loop_color(row_idx, loop_idx):
    """Returns discrete colors for loops based on a repeating pattern."""
    p = [(0.45, 0.25, 0.15), (0.15, 0.35, 0.75), (0.95, 0.85, 0.20), (0.85, 0.20, 0.20)]
    r = row_idx % 3
    if r == 0: return p[0]
    return p[3] if loop_idx % 2 == 0 else p[2] if r == 1 else p[1]

class KnittingOptimizer:
    """Core optimizer for differentiable knitting reconstruction."""
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
        self.row_colors = [self.row_colors[i % len(self.row_colors)] for i in range(bitmap.shape[0])]

    def _build_mask(self, crop):
        cw, ch = crop
        mask = np.zeros((self.res_height, self.res_width, 3), dtype=np.float32)
        x0, x1 = int((1-cw)*0.5*self.res_width), int((1+cw)*0.5*self.res_width)
        y0, y1 = int((1-ch)*0.5*self.res_height), int((1+ch)*0.5*self.res_height)
        mask[max(0,y0):min(self.res_height,y1), max(0,x0):min(self.res_width,x1), :] = 1.0
        return mask

    def get_scene_dict(self, obj_path, params, camera_params=None):
        dm, fov = camera_params if camera_params else self.camera_params
        verts_list = compute_knitting_vertices(params, self.bitmap)
        all_verts = jnp.concatenate([v for v, _ in verts_list])
        vmin, vmax = np.min(all_verts, axis=0), np.max(all_verts, axis=0)
        center = (vmin + vmax) * 0.5
        dist = max((vmax[1]-vmin[1])/np.tan(np.deg2rad(fov)*0.5), 
                   (vmax[0]-vmin[0])/np.tan(np.deg2rad(fov)*0.5*self.res_width/self.res_height)) * 0.5 * dm + (vmax[2]-vmin[2])*0.2
        return {
            "type": "scene", "integrator": {"type": "path", "max_depth": 2},
            "sensor": {
                "type": "perspective", "fov": fov,
                "to_world": mi.ScalarTransform4f.look_at(origin=[float(center[0]), float(center[1]), float(vmax[2]+dist)], 
                                                        target=[float(center[0]), float(center[1]), float(center[2])], up=[0, 1, 0]),
                "film": {"type": "hdrfilm", "width": self.res_width, "height": self.res_height, "pixel_format": "rgb"},
            },
            "emitter": {"type": "constant", "radiance": {"type": "rgb", "value": [0.9, 0.9, 0.9]}},
            "mesh": { "type": "obj", "filename": obj_path, "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.8, 0.4, 0.3]}} }
        }

    def step(self, params):
        self.iteration += 1
        params_np = np.array(params)
        verts_list = compute_knitting_vertices(params_np, self.bitmap)
        mesh_data = [(v, [], f, n) for (v, n), f in zip(verts_list, compute_knitting_faces(CONFIG['geometry']['segments'], verts_list))]
        base_path = os.path.join(OUTPUT_DIR, "meshes", f"epoch_{self.iteration:03d}")
        save_combined_obj(mesh_data, base_path)
        scene = mi.load_dict(self.get_scene_dict(base_path + "_combined.obj", params_np))
        params_scene = mi.traverse(scene)
        vertex_key = [k for k in params_scene.keys() if 'vertex_positions' in k]
        
        if vertex_key:
            vertex_positions = params_scene[vertex_key[0]]
            dr.enable_grad(vertex_positions)
            params_scene.update()
            img = mi.render(scene, params=params_scene, spp=CONFIG['rendering']['spp_optimization'])
            diff = (img - self.ref_tensor) * self.loss_mask
            loss_dr = dr.sum(dr.sqr(diff)) / (dr.sum(self.loss_mask) + 1e-8)
            dr.backward(loss_dr)
            jacobian = compute_geometry_jacobian(params_np, self.bitmap)
            vertex_grads_np = np.array(dr.grad(vertex_positions)).reshape(-1, 3)
            grads = np.array([np.sum(vertex_grads_np * jacobian[:, :, i]) for i in range(len(params_np))])
            current_loss = float(dr.ravel(loss_dr)[0])
        else:
            grads, current_loss = np.zeros(len(params_np)), 0.0
            
        if self.opt_state is None: self.opt_state = self.optimizer.init(jnp.array(params))
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
    """Unified UI for interactive parameter and spline editing."""
    def __init__(self, optimizer, init_params):
        from vedo import Plotter, Mesh, Text2D
        self.optimizer, self.params = optimizer, np.array(init_params, dtype=np.float64)
        self.bitmap = optimizer.bitmap
        self.param_idx, self.mode = 0, 'parameter'
        
        verts_list = compute_knitting_vertices(self.params, self.bitmap)
        faces_list = compute_knitting_faces(CONFIG['geometry']['segments'], verts_list)
        self.mesh_faces = []
        for faces in faces_list:
            f_np = np.array(faces)
            tris = np.empty((len(f_np) * 2, 3), dtype=np.int32)
            tris[0::2], tris[1::2] = f_np[:, [0, 1, 2]], f_np[:, [0, 2, 3]]
            self.mesh_faces.append(tris)

        self.plotter = Plotter(bg='blackboard', axes=1, title="Knitting Unified Optimizer")
        self.actors = [Mesh([np.array(v), self.mesh_faces[i]]).color(self.optimizer.row_colors[i]).lighting('plastic').alpha(0.8) for i, (v, n) in enumerate(verts_list)]
        self.plotter.add(self.actors)
        self.info_text = None
        self._update_display(rebuild=False)
        self.plotter.add_callback("KeyPress", self._on_key_press)

    def _update_display(self, rebuild=True):
        from vedo import Text2D
        if rebuild:
            verts_list = compute_knitting_vertices(self.params, self.bitmap)
            for i, (v, n) in enumerate(verts_list): self.actors[i].points = np.array(v)
        if self.info_text: self.plotter.remove(self.info_text)
        param_names = CONFIG['geometry']['param_names']
        mode_label = f"MODE: {self.mode.upper()} (T to toggle)"
        lines = [mode_label, "---"] + [f"{'>>>' if i==self.param_idx else '   '} {param_names[i]}: {self.params[i]:.4f}" for i in range(len(param_names))]
        self.info_text = Text2D("\n".join(lines + ["", "KEYS: arrows=adj, R=render, O=optimize, F=finish"]), pos='top-left', s=0.7, bg='black', alpha=0.5)
        self.plotter.add(self.info_text).render()

    def _on_key_press(self, event):
        key = event.keypress.lower() if event.keypress else ""
        if key == 't':
            self.mode = 'spline' if self.mode == 'parameter' else 'parameter'
            print(f"Switched mode to: {self.mode}")
        elif key == 'r': self._trigger_render()
        elif key == 'o': self._trigger_optimization()
        elif key == 'f': self.plotter.close()
        elif key in ['up', 'down']:
            delta = -1 if key == 'up' else 1
            self.param_idx = (self.param_idx + delta) % len(self.params)
        elif key in ['left', 'right']:
            delta = -1 if key == 'left' else 1
            self.params[self.param_idx] += delta * CONFIG['geometry']['param_deltas'][self.param_idx]
            self.params[self.param_idx] = np.clip(self.params[self.param_idx], *CONFIG['geometry']['param_ranges'][self.param_idx])
        self._update_display()

    def _trigger_render(self):
        print("Rendering scene...")
        verts_list = compute_knitting_vertices(self.params, self.bitmap)
        mesh_data = [(v, [], f, n) for (v, n), f in zip(verts_list, compute_knitting_faces(CONFIG['geometry']['segments'], verts_list))]
        save_combined_obj(mesh_data, "temp_preview")
        image = mi.render(mi.load_dict(self.optimizer.get_scene_dict("temp_preview_combined.obj", self.params)), spp=64)
        plt.figure("Current Reconstruction Preview"); plt.imshow(np.clip(np.array(image), 0, 1)); plt.axis('off'); plt.show()

    def _trigger_optimization(self):
        print("Starting optimization loop..."); self.params, _ = run_optimization_loop(self.optimizer, self.params); self._update_display()

    def run(self):
        self.plotter.show(interactive=True)
        return self.params

def run_optimization_loop(optimizer, params):
    """Main optimization loop controlled by hyperparameters."""
    opt_cfg = CONFIG['optimization']
    best_loss, best_params, counter = float('inf'), params, 0
    for _ in range(opt_cfg['max_epochs']):
        params = optimizer.step(params)
        loss = optimizer.loss_history[-1]
        if loss < best_loss - 1e-6: best_loss, best_params, counter = loss, params, 0
        elif (counter := counter + 1) >= opt_cfg['patience']: break
    return best_params, best_loss

def save_final_results(optimizer, params):
    """Generates final OBJ and high-quality render of the reconstruction."""
    print("\nSaving final reconstruction results...")
    verts_list = compute_knitting_vertices(params, optimizer.bitmap)
    faces_list = compute_knitting_faces(CONFIG['geometry']['segments'], verts_list)
    mesh_data = [(v, [], f, n) for (v, n), f in zip(verts_list, faces_list)]
    save_combined_obj(mesh_data, os.path.join(OUTPUT_DIR, "meshes", "optimized_reconstruction"))
    scene_dict = optimizer.get_scene_dict(os.path.join(OUTPUT_DIR, "meshes", "optimized_reconstruction_combined.obj"), params)
    final_render = mi.render(mi.load_dict(scene_dict), spp=CONFIG['rendering']['spp_final'])
    mi.util.write_bitmap(os.path.join(OUTPUT_DIR, "final_reconstruction.png"), final_render)
    print(f"Final reconstruction saved. Best Loss: {optimizer.loss_history[-1] if optimizer.loss_history else 0.0:.6f}")

if __name__ == "__main__":
    reference = Image.open(CONFIG['ui']['reference_image']).convert("RGB")
    bitmap = jnp.ones((CONFIG['geometry']['bitmap_rows'], CONFIG['geometry']['bitmap_loops']))
    opt = KnittingOptimizer(reference, bitmap)
    final_params = KnittingApp(opt, CONFIG['geometry']['initial_params']).run()
    save_final_results(opt, final_params)
