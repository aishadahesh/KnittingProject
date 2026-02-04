# %% IMPORTS
import os
import time
import numpy as np
from PIL import Image
import mitsuba as mi
import drjit as dr
import jax.numpy as jnp
import jax 
import optax
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from new_main import save_into_obj_files

# %% CONFIGURATION & VARIANTS
OUTPUT_DIR = "opt_outputs"
for d in ["meshes", "renders"]: os.makedirs(os.path.join(OUTPUT_DIR, d), exist_ok=True)

# Ensure Mitsuba is in a differentiable mode
if not mi.variant():
    try:
        mi.set_variant("cuda_ad_rgb")
    except:
        mi.set_variant("llvm_ad_rgb")

# %% JAX GEOMETRY LOGIC

def eval_curve(t, scale, stitch_bulge=0.30, stitch_z=-0.4):
    t, scale = jnp.asarray(t), jnp.asarray(scale)
    x = stitch_bulge * jnp.sin(2*t) + t/(2*jnp.pi)
    y = -(jnp.cos(t) - 1)/2
    z = stitch_z * (jnp.cos(2*t) - 1)/2
    
    x = jnp.where(scale == 0, t/(2*jnp.pi), x)
    return jnp.column_stack((x, y * scale, z * scale))

def eval_curve_derivative(t, scale, stitch_bulge=0.30, stitch_z=-0.4):
    t, scale = jnp.asarray(t), jnp.asarray(scale)
    dx = 2*stitch_bulge*jnp.cos(2*t) + 1/(2*jnp.pi)
    dy = 0.5*jnp.sin(t)*scale
    dz = -stitch_z*jnp.sin(2*t)*scale
    dx = jnp.where(scale == 0, 1/(2*jnp.pi), dx)
    return jnp.column_stack((dx, dy, dz))

def compute_orthonormal_frame(T):
    T = T / (jnp.linalg.norm(T, axis=1, keepdims=True) + 1e-8)
    ref = jnp.array([0.0, 0.0, 1.0])
    U = jnp.cross(T, ref)
    U_norm = jnp.linalg.norm(U, axis=1, keepdims=True)
    
    # Handle parallel cases (Gimbal Lock avoidance)
    ref_alt = jnp.array([1.0, 0.0, 0.0])
    U_alt = jnp.cross(T, ref_alt)
    U = jnp.where(U_norm < 1e-6, U_alt, U)
    
    U = U / (jnp.linalg.norm(U, axis=1, keepdims=True) + 1e-8)
    V = jnp.cross(T, U)
    return T, U, V

# %% BITMAP PROCESSING

def count_consecutive_zeros_after_jax(A):
    n = len(A)
    indices = jnp.arange(n)
    mask = (indices[:, None] < indices[None, :])
    masked_A = jnp.where(mask, A[None, :], 999)
    nonzero_mask = (masked_A != 0) & mask
    has_nonzero = jnp.any(nonzero_mask, axis=1)
    first_nonzero_pos = jnp.argmax(nonzero_mask.astype(jnp.int32), axis=1)
    
    zeros_count = jnp.where(has_nonzero, first_nonzero_pos - indices, n - indices - 1)
    return jnp.where(A == 1, zeros_count + 1, 0)

def convert_bitmap_to_scales_factors_jax(matrix):
    return jax.vmap(count_consecutive_zeros_after_jax, in_axes=1, out_axes=1)(matrix.T).T

# %% JACOBIAN COMPUTATION (JAX AUTODIFF)

def compute_geometry_jacobian(params, bitmap):
    """Compute Jacobian of vertex positions w.r.t. parameters using JAX autodiff"""
    consts = {'BITMAP': bitmap, 'loop_res': 32, 'segments': 8}
    
    def get_all_vertices(p):
        verts_list = compute_knitting_vertices(p, consts)
        # Flatten all vertices into one array
        all_verts = jnp.concatenate([v for v, _ in verts_list], axis=0)
        return all_verts
    
    # Compute Jacobian: d(vertices)/d(params)
    jacobian_fn = jax.jacfwd(get_all_vertices)
    J = jacobian_fn(params)  # Shape: [num_verts, 3, num_params]
    
    return J

# %% MESH GENERATION

def compute_knitting_vertices(params, consts):
    bitmap = jnp.array(consts["BITMAP"], dtype=float)
    loop_res, segments = consts["loop_res"], consts["segments"]
    stitch_bulge, stitch_z, dy, radius = params

    scale_factor = convert_bitmap_to_scales_factors_jax(bitmap)
    scale_factor = jnp.where((scale_factor <= 1), scale_factor, 1 + dy * (scale_factor - 1))

    n_rows, n_loops = scale_factor.shape
    verts_list = []

    for row_idx in range(n_rows):
        scale_row = scale_factor[row_idx]
        t = jnp.linspace(0.0, 2 * jnp.pi * n_loops, loop_res * n_loops + 1, endpoint=True)
        # Interpolate scales to match vertex resolution
        x_scale = jnp.repeat(scale_row, loop_res)
        x_scale = jnp.append(x_scale, 1.0)

        p = eval_curve(t, x_scale, stitch_bulge, stitch_z).at[:, 1].add(row_idx * dy)
        dp = eval_curve_derivative(t, x_scale, stitch_bulge, stitch_z)
        T, U, V = compute_orthonormal_frame(dp)

        j_indices = jnp.arange(segments)
        angles = 2 * jnp.pi * j_indices / segments
        offsets = (U[:, None, :] * jnp.cos(angles)[None, :, None] + 
                   V[:, None, :] * jnp.sin(angles)[None, :, None]) * radius
        
        verts = (p[:, None, :] + offsets).reshape(-1, 3)
        verts_list.append((verts, len(p)))
    
    return verts_list

def compute_knitting_faces(segments, verts_list):
    """Dynamically calculates faces based on actual vertex count."""
    faces_list = []
    for _, n_points in verts_list:
        i_grid, j_grid = jnp.meshgrid(jnp.arange(n_points - 1), jnp.arange(segments), indexing='ij')
        v0 = i_grid * segments + j_grid
        v1 = i_grid * segments + (j_grid + 1) % segments
        v2 = (i_grid + 1) * segments + (j_grid + 1) % segments
        v3 = (i_grid + 1) * segments + j_grid
        faces_list.append(jnp.stack([v0, v1, v2, v3], axis=-1).reshape(-1, 4))
    return faces_list

# %% PARAMETER VISUALIZATION

def visualize_parameter_effects(params, bitmap, param_names, output_path):
    """Create 3D visualizations showing how each parameter affects geometry"""
    consts = {'BITMAP': bitmap, 'loop_res': 32, 'segments': 8}
    
    fig = plt.figure(figsize=(20, 5))
    
    for i, name in enumerate(param_names):
        ax = fig.add_subplot(1, 4, i+1, projection='3d')
        
        # Create three variants: -20%, current, +20%
        delta = 0.2 * params[i]
        test_params = [
            params.copy().at[i].set(params[i] - delta),
            params,
            params.copy().at[i].set(params[i] + delta)
        ]
        colors = ['blue', 'green', 'red']
        labels = [f'-20%', 'current', '+20%']
        
        for p, color, label in zip(test_params, colors, labels):
            verts_list = compute_knitting_vertices(p, consts)
            # Plot just the first strand for clarity
            if len(verts_list) > 0:
                verts = verts_list[0][0]  # First strand
                # Sample every 4th vertex to reduce clutter
                verts_sample = verts[::4]
                ax.plot(verts_sample[:, 0], verts_sample[:, 1], verts_sample[:, 2], 
                       color=color, label=label, alpha=0.6, linewidth=2)
        
        ax.set_title(f'Effect of {name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(fontsize=8)
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved parameter effect visualization: {output_path}")

# %% OPTIMIZATION ENGINE

class KnittingOptimizer:
    def __init__(self, reference_img, bitmap, learning_rate=0.01):
        self.bitmap = bitmap
        self.consts = {'BITMAP': bitmap, 'loop_res': 32, 'segments': 8}
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = None
        self.iteration = 0
        self.best_loss = float('inf')
        
        # Keep reference at original size and render at same resolution
        # This maintains gradient tracking through the entire pipeline
        self.ref_array = np.array(reference_img).astype(np.float32) / 255.0
        self.ref_img = reference_img  # Keep PIL image for visualization
        self.ref_height, self.ref_width = self.ref_array.shape[:2]
        
        # Use reference resolution for rendering (maintains gradients)
        self.res_height = self.ref_height
        self.res_width = self.ref_width
        
        # Track optimization history
        self.loss_history = []
        self.param_history = []
        self.gradient_history = []  # Track gradients for each parameter

    def get_scene_dict(self, obj_path, params):
        """Create Mitsuba scene with properly framed camera and clean background"""
        n_rows, n_loops = self.bitmap.shape
        _, stitch_z, dy, _ = params
        
        # Center the camera on the mesh
        center = [n_loops / 2.0, ((n_rows - 1) * dy) / 2.0, stitch_z / 2.0]
        
        # Optimal camera distance found from testing
        dist = max(n_loops, n_rows * dy) * 1.05
        
        return {
            "type": "scene",
            "integrator": {"type": "path", "max_depth": 2},
            "sensor": {
                "type": "perspective",
                "fov": 45,
                "to_world": mi.ScalarTransform4f.look_at(
                    origin=[center[0], center[1], center[2] + dist], 
                    target=center, 
                    up=[0, 1, 0]
                ),
                "film": {
                    "type": "hdrfilm", 
                    "width": self.res_width, 
                    "height": self.res_height,
                    "rfilter": {"type": "gaussian"},
                    "pixel_format": "rgb"
                },
            },
            # Use envmap for uniform background
            "emitter": {
                "type": "constant",
                "radiance": {"type": "rgb", "value": [0.9, 0.9, 0.9]}
            },
            # Add point lights for better mesh illumination
            "light1": {
                "type": "point",
                "position": [center[0] + n_loops*0.5, center[1] + n_rows*dy*0.5, center[2] + dist*0.7],
                "intensity": {"type": "rgb", "value": [2.0, 2.0, 2.0]}
            },
            "light2": {
                "type": "point",
                "position": [center[0] - n_loops*0.5, center[1] - n_rows*dy*0.3, center[2] + dist*0.5],
                "intensity": {"type": "rgb", "value": [1.0, 1.0, 1.0]}
            },
            "mesh": {
                "type": "obj", 
                "filename": obj_path,
                "bsdf": {
                    "type": "diffuse", 
                    "reflectance": {"type": "rgb", "value": [0.8, 0.4, 0.3]}
                }
            }
        }
    
    def render_parameter_variations(self, params, param_names):
        """Render the scene with parameter variations to show their effects"""
        params_np = np.array(params)
        variations = {}
        
        for i, name in enumerate(param_names):
            # Create -10% and +10% variations
            delta = 0.1 * params_np[i]
            
            for sign, label in [(-1, 'minus'), (0, 'current'), (1, 'plus')]:
                p_var = params_np.copy()
                if sign != 0:
                    p_var[i] += sign * delta
                
                # Generate and render mesh
                verts_list = compute_knitting_vertices(p_var, self.consts)
                faces_list = compute_knitting_faces(self.consts['segments'], verts_list)
                mesh_data = [(v, [], f, n) for (v, n), f in zip(verts_list, faces_list)]
                
                path = os.path.join(OUTPUT_DIR, "meshes", f"temp_var_{name}_{label}.obj")
                save_into_obj_files(mesh_data, path.replace(".obj", ""))
                
                scene = mi.load_dict(self.get_scene_dict(path.replace(".obj", "_combined.obj"), p_var))
                img = mi.render(scene, spp=16)  # Lower spp for speed
                
                variations[f"{name}_{label}"] = np.array(img)
        
        return variations
    
    def visualize_epoch_summary(self, params, current_img, loss, param_names):
        """Create comprehensive visualization after each epoch"""
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(5, 5, hspace=0.3, wspace=0.3)
        
        params_np = np.array(params)
        
        # Row 1: Current Results
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(current_img)
        ax1.set_title(f'Current Render\nEpoch {self.iteration}', fontsize=11, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(self.ref_img)
        ax2.set_title('Target Reference', fontsize=11, fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        diff = np.abs(current_img - self.ref_array)
        ax3.imshow(diff, cmap='hot')
        ax3.set_title(f'Difference\nLoss: {loss:.6f}', fontsize=11, fontweight='bold')
        ax3.axis('off')
        
        # Loss History
        ax4 = fig.add_subplot(gs[0, 3:])
        if len(self.loss_history) > 0:
            ax4.plot(range(1, len(self.loss_history) + 1), self.loss_history, 
                    'b-o', linewidth=2, markersize=6)
            ax4.scatter([self.iteration], [loss], color='red', s=200, 
                       zorder=5, edgecolors='darkred', linewidths=2, label='Current')
            ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax4.set_ylabel('MSE Loss', fontsize=11, fontweight='bold')
            ax4.set_title('Loss Evolution', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.legend(fontsize=9)
            if len(self.loss_history) > 1:
                ax4.set_yscale('log')
        
        # Rows 2-5: Parameter Effects (one row per parameter)
        for i, name in enumerate(param_names):
            row = i + 1
            
            # Render parameter variations
            delta = 0.1 * params_np[i]
            images = []
            labels = [f'-10%\n({params_np[i]-delta:.3f})', 
                     f'Current\n({params_np[i]:.3f})', 
                     f'+10%\n({params_np[i]+delta:.3f})']
            
            for sign in [-1, 0, 1]:
                p_var = params_np.copy()
                if sign != 0:
                    p_var[i] += sign * delta
                
                # Quick render (reuse epoch mesh for current params)
                if sign == 0:
                    # Reuse the epoch mesh instead of regenerating
                    epoch_obj = os.path.join(OUTPUT_DIR, "meshes", f"epoch_{self.iteration:03d}_combined.obj")
                    if os.path.exists(epoch_obj):
                        scene = mi.load_dict(self.get_scene_dict(epoch_obj, p_var))
                        img = mi.render(scene, spp=16)
                        images.append(np.array(img))
                        continue
                
                # Generate mesh for variation
                verts_list = compute_knitting_vertices(p_var, self.consts)
                faces_list = compute_knitting_faces(self.consts['segments'], verts_list)
                mesh_data = [(v, [], f, n) for (v, n), f in zip(verts_list, faces_list)]
                
                path = os.path.join(OUTPUT_DIR, "meshes", f"temp_var_{self.iteration}_{i}_{sign}.obj")
                save_into_obj_files(mesh_data, path.replace(".obj", ""))
                
                scene = mi.load_dict(self.get_scene_dict(path.replace(".obj", "_combined.obj"), p_var))
                img = mi.render(scene, spp=16)
                images.append(np.array(img))
                
                # Clean up temp variation file
                try:
                    os.remove(path.replace(".obj", "_combined.obj"))
                    os.remove(path)
                except:
                    pass
            
            # Plot the three variations
            for col, (img, label) in enumerate(zip(images, labels)):
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(img)
                ax.set_title(label, fontsize=9)
                ax.axis('off')
            
            # Parameter info and gradient
            ax_info = fig.add_subplot(gs[row, 3:])
            ax_info.axis('off')
            
            # Display parameter info
            if len(self.gradient_history) > 0:
                grad = self.gradient_history[-1][i]
                info_text = f"Parameter: {name}\n\n"
                info_text += f"Current Value: {params_np[i]:.4f}\n"
                info_text += f"Gradient: {grad:+.6f}\n\n"
                info_text += f"Effect: {'←' if grad < 0 else '→'} {'DECREASE' if grad < 0 else 'INCREASE'} {name}\n"
                info_text += f"       to {'REDUCE' if grad < 0 else 'INCREASE'} loss\n\n"
                
                # Parameter history
                if len(self.param_history) > 1:
                    param_vals = [float(p[i]) for p in self.param_history]
                    change = params_np[i] - param_vals[0]
                    percent = (change / param_vals[0] * 100) if param_vals[0] != 0 else 0
                    info_text += f"Change from start: {change:+.4f} ({percent:+.1f}%)"
                
                ax_info.text(0.1, 0.5, info_text, fontsize=10, 
                           verticalalignment='center', family='monospace',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
        fig.suptitle(f'Epoch {self.iteration} Summary - Optimization Progress', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Save figure
        save_path = os.path.join(OUTPUT_DIR, "renders", f"epoch_{self.iteration:03d}_summary.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved epoch summary: {save_path}")
        
        # Display interactively in Jupyter
        plt.show()
        plt.close()

    def step(self, params, epsilon=0.01):
        self.iteration += 1
        params_np = np.array(params)
        
        # Generate mesh geometry from current parameters
        verts_list = compute_knitting_vertices(params_np, self.consts)
        faces_list = compute_knitting_faces(self.consts['segments'], verts_list)
        mesh_data = [(v, [], f, n) for (v, n), f in zip(verts_list, faces_list)]
        
        # Save mesh to OBJ file (one per epoch)
        path = os.path.join(OUTPUT_DIR, "meshes", f"epoch_{self.iteration:03d}.obj")
        save_into_obj_files(mesh_data, path.replace(".obj", ""))
        obj_path = path.replace(".obj", "_combined.obj")
        
        # Build complete scene
        n_rows, n_loops = self.bitmap.shape
        _, stitch_z, dy, _ = params_np
        center = [n_loops / 2.0, ((n_rows - 1) * dy) / 2.0, stitch_z / 2.0]
        dist = max(n_loops, n_rows * dy) * 1.05
        
        scene = mi.load_dict({
            "type": "scene",
            "integrator": {"type": "path", "max_depth": 2},
            "sensor": {
                "type": "perspective",
                "fov": 45,
                "to_world": mi.ScalarTransform4f.look_at(
                    origin=[center[0], center[1], center[2] + dist], 
                    target=center, 
                    up=[0, 1, 0]
                ),
                "film": {
                    "type": "hdrfilm", 
                    "width": self.res_width, 
                    "height": self.res_height, 
                    "rfilter": {"type": "gaussian"}
                },
            },
            "light": {"type": "constant", "radiance": {"type": "rgb", "value": 0.8}},
            "mesh": {
                "type": "obj",
                "filename": obj_path,
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {"type": "rgb", "value": [0.8, 0.4, 0.3]}
                }
            }
        })
        
        # Access mesh vertex positions through scene parameters
        params_scene = mi.traverse(scene)
        vertex_key = [k for k in params_scene.keys() if 'vertex_positions' in k]
        
        if len(vertex_key) > 0:
            # Enable gradient tracking on vertex positions
            vertex_positions = params_scene[vertex_key[0]]
            dr.enable_grad(vertex_positions)
            params_scene[vertex_key[0]] = vertex_positions
            params_scene.update()
            
            # Differentiable rendering (at reference resolution)
            img = mi.render(scene, params=params_scene, spp=32)
            
            # Compute loss with gradient tracking (no resizing needed)
            ref_flat = dr.ravel(mi.TensorXf(self.ref_array))    ##Convert the input into a contiguous flat array.
            img_flat = dr.ravel(img)
            loss_dr = dr.mean(dr.sqr(img_flat - ref_flat))
            
            # Backward pass through renderer
            dr.backward(loss_dr)
            
            # Extract gradients from vertex positions
            vertex_grads = dr.grad(vertex_positions)
            
            # Compute parameter gradients using chain rule with JAX Jacobian
            J = compute_geometry_jacobian(params_np, self.bitmap)
            vertex_grads_np = np.array(vertex_grads).reshape(-1, 3)
            
            # Compute gradients: sum over vertices of (dL/dv_i * dv_i/dp_j)
            grads_np = np.zeros(len(params_np))
            for i in range(len(params_np)):
                grads_np[i] = np.sum(vertex_grads_np * J[:, :, i])
            
            # Convert loss to numpy (properly extract scalar value from DrJit type)
            base_loss = float(dr.sum(loss_dr)[0])
            base_img = np.array(img)
        else:
            # Fallback: use finite differences if gradient tracking fails
            print("  ⚠ Gradient tracking unavailable, using finite differences")
            img = mi.render(scene, spp=32)
            base_loss = np.mean((np.array(img) - self.ref_array)**2)
            base_img = np.array(img)
            
            # Finite difference gradients
            grads_np = np.zeros(len(params_np))
            for i in range(len(params_np)):
                p_eps = params_np.copy()
                p_eps[i] += epsilon
                verts_list_eps = compute_knitting_vertices(p_eps, self.consts)
                faces_list_eps = compute_knitting_faces(self.consts['segments'], verts_list_eps)
                mesh_data_eps = [(v, [], f, n) for (v, n), f in zip(verts_list_eps, faces_list_eps)]
                
                path_eps = os.path.join(OUTPUT_DIR, "meshes", f"temp_eps_{i}.obj")
                save_into_obj_files(mesh_data_eps, path_eps.replace(".obj", ""))
                scene_eps = mi.load_dict(self.get_scene_dict(path_eps.replace(".obj", "_combined.obj"), p_eps))
                img_eps = mi.render(scene_eps, spp=32)
                loss_eps = np.mean((np.array(img_eps) - self.ref_array)**2)
                grads_np[i] = (loss_eps - base_loss) / epsilon
                
                # Clean up temp file
                try:
                    os.remove(path_eps.replace(".obj", "_combined.obj"))
                    os.remove(path_eps)
                except:
                    pass
        
        # Update parameters using Adam optimizer
        if self.opt_state is None: 
            self.opt_state = self.optimizer.init(jnp.array(params))
        updates, self.opt_state = self.optimizer.update(jnp.array(grads_np), self.opt_state)
        new_params = optax.apply_updates(jnp.array(params), updates)
        
        # Clip to physical reality
        new_params = jnp.clip(new_params, jnp.array([0.1, -0.8, 0.1, 0.02]), jnp.array([0.6, -0.1, 1.0, 0.3]))
        
        # Track history
        self.loss_history.append(base_loss)
        self.param_history.append(new_params)
        self.gradient_history.append(grads_np)
        
        # Display progress
        param_names = ['bulge', 'z', 'dy', 'rad']
        grad_str = ' | '.join([f"∂L/∂{name}={g:+.4f}" for name, g in zip(param_names, grads_np)])
        print(f"\nEpoch {self.iteration:02d} | Loss: {base_loss:.6f}")
        print(f"  Gradients: {grad_str}")
        print(f"  Parameters: {np.round(new_params, 4)}")
        print(f"  [Using DrJit autodiff through Mitsuba renderer]")
        
        # Save progress render
        mi.util.write_bitmap(os.path.join(OUTPUT_DIR, "renders", f"iter_{self.iteration:03d}.png"), base_img)
        
        # Create comprehensive visualization
        print(f"  Creating epoch visualization...")
        self.visualize_epoch_summary(new_params, base_img, base_loss, param_names)
        
        return new_params

# %% MAIN EXECUTION

def print_hyperparameters(learning_rate, max_epochs, spp_opt, spp_final, epsilon, patience):
    """Display optimization hyperparameters"""
    print("="*80)
    print("KNITTING PATTERN OPTIMIZATION")
    print("="*80)
    print(f"Learning Rate:       {learning_rate}")
    print(f"Max Epochs:          {max_epochs}")
    print(f"SPP (Optimization):  {spp_opt}")
    print(f"SPP (Final Render):  {spp_final}")
    print(f"Finite Diff Step:    {epsilon}")
    print(f"Early Stop Patience: {patience}")
    print("="*80)

def print_initial_parameters(params):
    """Display initial parameter values"""
    print(f"\nInitial Parameters: {params}")
    print(f"  stitch_bulge: {params[0]:.4f}")
    print(f"  stitch_z:     {params[1]:.4f}")
    print(f"  dy (spacing): {params[2]:.4f}")
    print(f"  radius:       {params[3]:.4f}\n")

def run_optimization_loop(optimizer, params, max_epochs, epsilon, patience):
    """Execute the main optimization loop with early stopping"""
    best_loss = float('inf')
    best_params = None
    patience_counter = 0
    
    for epoch in range(max_epochs):
        params = optimizer.step(params, epsilon=epsilon)
        current_loss = optimizer.loss_history[-1]
        
        # Early stopping logic
        if current_loss < best_loss - 1e-6:
            best_loss = current_loss
            best_params = params
            patience_counter = 0
            print(f"  ✓ New best loss: {best_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n✓ Converged! No improvement for {patience} iterations.")
                params = best_params
                break
    
    return params, best_loss

def create_before_after_comparison(output_dir, ref_img, total_iters):
    """Generate before/after comparison visualization"""
    print("\n" + "="*80)
    print("CREATING BEFORE/AFTER COMPARISON")
    print("="*80)
    
    first_render_path = os.path.join(output_dir, "renders", "iter_001.png")
    last_render_path = os.path.join(output_dir, "renders", f"iter_{total_iters:03d}.png")
    
    if not (os.path.exists(first_render_path) and os.path.exists(last_render_path)):
        print("  ⚠ Render files not found, skipping comparison")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Initial render
    init_img = Image.open(first_render_path)
    axes[0].imshow(init_img)
    axes[0].set_title('Initial Render (Epoch 1)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Reference image
    axes[1].imshow(ref_img)
    axes[1].set_title('Target Reference', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Final render
    final_img = Image.open(last_render_path)
    axes[2].imshow(final_img)
    axes[2].set_title(f'Final Render (Epoch {total_iters})', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "before_after_comparison.png")
    plt.savefig(comparison_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved comparison: {comparison_path}")
    plt.show()
    plt.close()

def create_optimization_summary(optimizer, init_params, final_params, output_dir):
    """Generate comprehensive optimization summary plots"""
    print("\n" + "="*80)
    print("GENERATING OPTIMIZATION SUMMARY")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    param_names = ['stitch_bulge', 'stitch_z', 'dy', 'radius']
    
    # 1. Loss Curve
    ax = axes[0, 0]
    ax.plot(optimizer.loss_history, 'b-o', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('MSE Loss', fontsize=12, fontweight='bold')
    ax.set_title('Optimization Progress', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if len(optimizer.loss_history) > 1:
        ax.set_yscale('log')
    
    # 2. Parameter Evolution
    ax = axes[0, 1]
    param_array = np.array(optimizer.param_history)
    for i, name in enumerate(param_names):
        ax.plot(param_array[:, i], label=name, marker='o', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Parameter Value', fontsize=12, fontweight='bold')
    ax.set_title('Parameter Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. Gradient Magnitudes Over Time
    ax = axes[1, 0]
    grad_array = np.array(optimizer.gradient_history)
    for i, name in enumerate(param_names):
        ax.plot(np.abs(grad_array[:, i]), label=f'|∂L/∂{name}|', marker='x', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gradient Magnitude', fontsize=12, fontweight='bold')
    ax.set_title('Gradient Evolution (Absolute Values)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 4. Parameter Changes (Bar Chart)
    ax = axes[1, 1]
    init_arr = np.array(init_params)
    final_arr = np.array(final_params)
    changes = ((final_arr - init_arr) / np.abs(init_arr)) * 100
    colors = ['green' if c < 0 else 'red' for c in changes]
    ax.bar(param_names, changes, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Change (%)', fontsize=12, fontweight='bold')
    ax.set_title('Parameter Changes from Initial', fontsize=14, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "optimization_summary.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved summary: {save_path}")
    plt.show()
    plt.close()

def visualize_parameter_effects_final(params, bitmap, output_dir):
    """Create 3D visualizations of parameter effects"""
    print("\n" + "="*80)
    print("VISUALIZING PARAMETER EFFECTS ON GEOMETRY")
    print("="*80)
    
    param_names = ['stitch_bulge', 'stitch_z', 'dy', 'radius']
    save_path = os.path.join(output_dir, "parameter_effects.png")
    visualize_parameter_effects(params, bitmap, param_names, save_path)
    
    # Show interactively
    consts = {'BITMAP': bitmap, 'loop_res': 32, 'segments': 8}
    fig = plt.figure(figsize=(20, 5))
    
    for i, name in enumerate(param_names):
        ax = fig.add_subplot(1, 4, i+1, projection='3d')
        delta = 0.2 * params[i]
        test_params = [
            params.copy().at[i].set(params[i] - delta),
            params,
            params.copy().at[i].set(params[i] + delta)
        ]
        colors = ['blue', 'green', 'red']
        labels = [f'-20%', 'current', '+20%']
        
        for p, color, label in zip(test_params, colors, labels):
            verts_list = compute_knitting_vertices(p, consts)
            if len(verts_list) > 0:
                verts = verts_list[0][0]
                verts_sample = verts[::4]
                ax.plot(verts_sample[:, 0], verts_sample[:, 1], verts_sample[:, 2], 
                       color=color, label=label, alpha=0.6, linewidth=2)
        
        ax.set_title(f'Effect of {name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(fontsize=8)
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()
    plt.close()

def compute_jacobian_analysis(params, bitmap):
    """Compute and display JAX Jacobian analysis"""
    print("\n" + "="*80)
    print("COMPUTING GEOMETRY JACOBIAN (JAX AUTODIFF)")
    print("="*80)
    print("Computing ∂(vertices)/∂(parameters) using JAX automatic differentiation...")
    
    J = compute_geometry_jacobian(params, bitmap)
    param_names = ['stitch_bulge', 'stitch_z', 'dy', 'radius']
    
    print(f"Jacobian shape: {J.shape}")
    print(f"  → {J.shape[0]} vertices")
    print(f"  → {J.shape[1]} spatial dimensions (x, y, z)")
    print(f"  → {J.shape[2]} parameters")
    
    print("\nAverage geometric sensitivity per parameter:")
    for i, name in enumerate(param_names):
        sensitivity = np.mean(np.abs(J[:, :, i]))
        print(f"  {name:14s}: {sensitivity:.6f} (units/parameter_unit)")

def print_final_summary(optimizer, init_params, final_params, best_loss):
    """Print final optimization summary"""
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Total Epochs:      {len(optimizer.loss_history)}")
    print(f"Final Loss:        {optimizer.loss_history[-1]:.6f}")
    print(f"Best Loss:         {best_loss:.6f}")
    
    param_names = ['stitch_bulge', 'stitch_z', 'dy', 'radius']
    init_arr = np.array(init_params)
    final_arr = np.array(final_params)
    
    print(f"\nParameter Changes:")
    for i, name in enumerate(param_names):
        change = final_arr[i] - init_arr[i]
        percent = (change / init_arr[i]) * 100 if init_arr[i] != 0 else 0
        print(f"  {name:14s}: {init_arr[i]:7.4f} → {final_arr[i]:7.4f} ({change:+.4f}, {percent:+6.1f}%)")

def test_parameters(params, reference_img, bitmap, output_dir="opt_outputs"):
    """Test if parameters reproduce the reference image (loss should be ~0)"""
    print("\n" + "="*80)
    print("TESTING PARAMETERS - SANITY CHECK & CAMERA CALIBRATION")
    print("="*80)
    
    consts = {'BITMAP': bitmap, 'loop_res': 32, 'segments': 8}
    params_np = np.array(params)
    
    print(f"Testing parameters: {params_np}")
    print(f"Reference image size: {reference_img.size}")
    
    # Generate mesh
    verts_list = compute_knitting_vertices(params_np, consts)
    faces_list = compute_knitting_faces(consts['segments'], verts_list)
    mesh_data = [(v, [], f, n) for (v, n), f in zip(verts_list, faces_list)]
    
    # Save mesh
    path = os.path.join(output_dir, "meshes", "test_params.obj")
    save_into_obj_files(mesh_data, path.replace(".obj", ""))
    obj_path = path.replace(".obj", "_combined.obj")
    
    # Match reference image resolution exactly
    ref_width, ref_height = reference_img.size
    ref_array = np.array(reference_img).astype(np.float32) / 255.0
    
    # Render with matching camera setup - try multiple camera distances to find best match
    n_rows, n_loops = bitmap.shape
    _, stitch_z, dy, _ = params_np
    center = [n_loops / 2.0, ((n_rows - 1) * dy) / 2.0, stitch_z / 2.0]
    
    print("\n📷 Camera calibration: Testing different distances to minimize loss...")
    
    best_loss = float('inf')
    best_dist_mult = 1.05
    best_img = None
    
    # Test around optimal distance
    for dist_mult in [0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]:
        dist = max(n_loops, n_rows * dy) * dist_mult
        
        scene = mi.load_dict({
            "type": "scene",
            "integrator": {"type": "path", "max_depth": 2},
            "sensor": {
                "type": "perspective",
                "fov": 45,
                "to_world": mi.ScalarTransform4f.look_at(
                    origin=[center[0], center[1], center[2] + dist], 
                    target=center, 
                    up=[0, 1, 0]
                ),
                "film": {
                    "type": "hdrfilm", 
                    "width": ref_width, 
                    "height": ref_height, 
                    "rfilter": {"type": "gaussian"},
                    "pixel_format": "rgb"
                },
            },
            # Use envmap for uniform background
            "emitter": {
                "type": "constant",
                "radiance": {"type": "rgb", "value": [0.9, 0.9, 0.9]}
            },
            # Add point lights for better mesh illumination
            "light1": {
                "type": "point",
                "position": [center[0] + n_loops*0.5, center[1] + n_rows*dy*0.5, center[2] + dist*0.7],
                "intensity": {"type": "rgb", "value": [2.0, 2.0, 2.0]}
            },
            "light2": {
                "type": "point",
                "position": [center[0] - n_loops*0.5, center[1] - n_rows*dy*0.3, center[2] + dist*0.5],
                "intensity": {"type": "rgb", "value": [1.0, 1.0, 1.0]}
            },
            "mesh": {
                "type": "obj",
                "filename": obj_path,
                "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.8, 0.4, 0.3]}}
            }
        })
        
        img = mi.render(scene, spp=64)
        img_array = np.array(img)
        
        loss = np.mean((img_array - ref_array)**2)
        
        status = "✓" if loss < best_loss else " "
        print(f"  {status} Distance multiplier {dist_mult:.1f}x: Loss = {loss:.8f}")
        
        if loss < best_loss:
            best_loss = loss
            best_dist_mult = dist_mult
            best_img = img_array
    
    print(f"\n✓ Best camera distance multiplier: {best_dist_mult:.1f}x")
    print(f"✓ Best loss achieved: {best_loss:.8f}")
    print(f"  {'✓ PASS - Excellent match!' if best_loss < 0.01 else '✗ FAIL - Loss still too high'}")
    print(f"\n⚠ RECOMMENDATION: Update get_scene_dict() to use: dist = max(...) * {best_dist_mult:.1f}")
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(reference_img)
    axes[0].set_title('Reference Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(best_img)
    axes[1].set_title(f'Best Match Render\nDistance={best_dist_mult:.1f}x\nLoss: {best_loss:.8f}', 
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    diff = np.abs(best_img - ref_array)
    im = axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Absolute Difference', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    test_path = os.path.join(output_dir, "camera_calibration.png")
    plt.savefig(test_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved calibration result: {test_path}")
    
    return best_loss, best_dist_mult

def save_best_model(params, bitmap, output_dir, spp_final=128):
    """Save the optimized model as OBJ file and create final high-quality render"""
    print("\n" + "="*80)
    print("SAVING BEST MODEL")
    print("="*80)
    
    consts = {'BITMAP': bitmap, 'loop_res': 32, 'segments': 8}
    params_np = np.array(params)
    
    # Generate final mesh with best parameters
    print("Generating final mesh geometry...")
    verts_list = compute_knitting_vertices(params_np, consts)
    faces_list = compute_knitting_faces(consts['segments'], verts_list)
    mesh_data = [(v, [], f, n) for (v, n), f in zip(verts_list, faces_list)]
    
    # Save as OBJ
    obj_path = os.path.join(output_dir, "meshes", "best_model.obj")
    save_into_obj_files(mesh_data, obj_path.replace(".obj", ""))
    final_obj_path = obj_path.replace(".obj", "_combined.obj")
    print(f"✓ Saved optimized mesh: {final_obj_path}")
    
    # Create high-quality final render with matching framing
    print(f"Creating final high-quality render ({spp_final} spp)...")
    n_rows, n_loops = bitmap.shape
    _, stitch_z, dy, _ = params_np
    
    # Use same camera settings as optimization
    center = [n_loops / 2.0, ((n_rows - 1) * dy) / 2.0, stitch_z / 2.0]
    dist = max(n_loops, n_rows * dy) * 1.05
    
    # Calculate rectangular resolution
    mesh_aspect = n_loops / n_rows
    final_height = 512
    final_width = int(final_height * mesh_aspect)
    
    scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 4},
        "sensor": {
            "type": "perspective",
            "fov": 45,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[center[0], center[1], center[2] + dist], 
                target=center, 
                up=[0, 1, 0]
            ),
            "film": {
                "type": "hdrfilm", 
                "width": final_width, 
                "height": final_height, 
                "height": 512,
                "rfilter": {"type": "gaussian"},
                "pixel_format": "rgb"
            },
        },
        # Constant background color
        "background": {
            "type": "constant",
            "radiance": {"type": "rgb", "value": [0.9, 0.9, 0.9]}
        },
        # Add point lights for better mesh illumination
        "light1": {
            "type": "point",
            "position": [center[0] + n_loops*0.5, center[1] + n_rows*dy*0.5, center[2] + dist*0.7],
            "intensity": {"type": "rgb", "value": [2.0, 2.0, 2.0]}
        },
        "light2": {
            "type": "point",
            "position": [center[0] - n_loops*0.5, center[1] - n_rows*dy*0.3, center[2] + dist*0.5],
            "intensity": {"type": "rgb", "value": [1.0, 1.0, 1.0]}
        },
        "mesh": {
            "type": "obj", 
            "filename": final_obj_path,
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.8, 0.4, 0.3]}}
        }
    })
    
    final_render = mi.render(scene, spp=spp_final)
    final_render_path = os.path.join(output_dir, "final_render_best.png")
    mi.util.write_bitmap(final_render_path, final_render)
    print(f"✓ Saved final render: {final_render_path}")
    
    # Display final render
    plt.figure(figsize=(8, 8))
    plt.imshow(np.array(final_render))
    plt.title('Final Optimized Model (High Quality)', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()
    
    return final_obj_path, final_render_path

# %% ENTRY POINT
if __name__ == "__main__":
    # ==================== HYPERPARAMETERS ====================
    LEARNING_RATE = 0.005
    MAX_EPOCHS = 5
    SPP_OPTIMIZATION = 32
    SPP_FINAL = 128
    EPSILON = 0.01
    PATIENCE = 5
    
    # TEST MODE: Set to True to test known parameters without optimization
    TEST_MODE = False
    epsilon = 0.001
    TEST_PARAMS = [ 0.2993+epsilon, -0.3505+epsilon, 0.40109998+epsilon, 0.1497+epsilon]
    # =========================================================
    
    # Load reference image and setup
    # ref = Image.open("referenceImage.jpg").convert("RGB")
    ref = Image.open("referenceImage.jpg").convert("RGB")
    BITMAP = jnp.ones((19, 17))
    # init_params = [0.2749, -0.375, 0.4210, 0.1251]
    epsilon = 0.001
    init_params= [ 0.2993+epsilon, -0.3505+epsilon, 0.40109998+epsilon, 0.1497+epsilon]
    
    if TEST_MODE:
        print("="*80)
        print("RUNNING IN TEST MODE - VALIDATING PARAMETERS")
        print("="*80)
        test_parameters(TEST_PARAMS, ref, BITMAP, OUTPUT_DIR)
    else:
        # Display configuration
        print_hyperparameters(LEARNING_RATE, MAX_EPOCHS, SPP_OPTIMIZATION, 
                             SPP_FINAL, EPSILON, PATIENCE)
        
        print_initial_parameters(init_params)
        
        # Run optimization
        opt = KnittingOptimizer(ref, BITMAP, learning_rate=LEARNING_RATE)
        params, best_loss = run_optimization_loop(opt, init_params, MAX_EPOCHS, 
                                                 EPSILON, PATIENCE)
        
        # Generate all visualizations
        create_before_after_comparison(OUTPUT_DIR, ref, len(opt.loss_history))
        create_optimization_summary(opt, init_params, params, OUTPUT_DIR)
        visualize_parameter_effects_final(params, BITMAP, OUTPUT_DIR)
        compute_jacobian_analysis(params, BITMAP)
        print_final_summary(opt, init_params, params, best_loss)
        
        # Save best model
        save_best_model(params, BITMAP, OUTPUT_DIR, spp_final=SPP_FINAL)
        
        print("\n" + "="*80)
        print("All visualizations complete and displayed interactively!")
        print("="*80)
# %%

