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
import torch
import torchvision.models as models


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

def compute_geometry_jacobian(geometry_params, bitmap):
    """Compute Jacobian of vertex positions w.r.t. parameters using JAX autodiff"""
    consts = {'BITMAP': bitmap, 'loop_res': 32, 'segments': 8}
    
    def get_all_vertices(p):
        verts_list = compute_knitting_vertices(p, consts)
        # Flatten all vertices into one array
        all_verts = jnp.concatenate([v for v, _ in verts_list], axis=0)
        return all_verts
    
    # Compute Jacobian: d(vertices)/d(params)
    jacobian_fn = jax.jacfwd(get_all_vertices)
    J = jacobian_fn(geometry_params)  # Shape: [num_verts, 3, num_params]
    
    return J

# %% MESH GENERATION

def compute_knitting_vertices(geometry_params, consts):
    bitmap = jnp.array(consts["BITMAP"], dtype=float)
    loop_res, segments = consts["loop_res"], consts["segments"]
    (
        stitch_bulge,
        stitch_z,
        loop_height,
        dy,
        radius,
        curve_skew,
        y_sharp,
        x_bias,
        z_bias,
        ellipse_ratio,
    ) = geometry_params

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

        p = eval_curve(
            t,
            x_scale,
            stitch_bulge,
            stitch_z,
        ).at[:, 1].add(row_idx * dy)
        dp = eval_curve_derivative(
            t,
            x_scale,
            stitch_bulge,
            stitch_z,
        )
        T, U, V = compute_orthonormal_frame(dp)
        
        # radius_varying = jnp.clip(radius * (1.0 + 0.3 * jnp.sin(t)), 0.01, 0.5)

        j_indices = jnp.arange(segments)
        angles = 2 * jnp.pi * j_indices / segments
        radius_u = radius * ellipse_ratio
        radius_v = radius
        offsets = (
            U[:, None, :] * jnp.cos(angles)[None, :, None] * radius_u
            + V[:, None, :] * jnp.sin(angles)[None, :, None] * radius_v
        )
        
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
    
    n_params = len(param_names)
    fig = plt.figure(figsize=(5 * n_params, 5))
    
    for i, name in enumerate(param_names):
        ax = fig.add_subplot(1, n_params, i + 1, projection='3d')
        
        # Create three variants: -20%, current, +20%
        delta = 0.2 * params[i]
        
        # Handle both list and JAX array
        params_low = list(params)
        params_low[i] = params[i] - delta
        params_high = list(params)
        params_high[i] = params[i] + delta
        
        test_params = [params_low, list(params), params_high]
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

# %% 3D VIEWER (VEDO)

def view_model(geometry_params, bitmap, row_colors=None, title="Knitting Model Viewer"):
    """
    View the knitting model interactively using vedo.
    
    Args:
        geometry_params: Array of geometry parameters
            [stitch_bulge, stitch_z, loop_height, dy, radius, curve_skew, y_sharp, x_bias, z_bias, ellipse_ratio]
        bitmap: 2D array defining the knitting pattern
        row_colors: Optional list of (r, g, b) tuples per row (values 0-1)
        title: Window title
    """
    try:
        from vedo import Mesh, Plotter, Text2D
    except ImportError:
        print("vedo not installed. Install with: pip install vedo")
        return
    
    consts = {'BITMAP': bitmap, 'loop_res': 32, 'segments': 8}
    segments = consts['segments']
    
    # Compute mesh geometry
    verts_list = compute_knitting_vertices(geometry_params, consts)
    faces_list = compute_knitting_faces(segments, verts_list)
    
    # Default row colors if not provided
    n_rows = len(verts_list)
    if row_colors is None:
        palette = [
            (0.15, 0.35, 0.75),  # Blue
            (0.90, 0.75, 0.20),  # Yellow  
            (0.85, 0.25, 0.25),  # Red
            (0.30, 0.70, 0.40),  # Green
        ]
        row_colors = [palette[i % len(palette)] for i in range(n_rows)]
    
    # Create meshes for each row
    meshes = []
    for row_idx, ((verts, n_points), faces) in enumerate(zip(verts_list, faces_list)):
        verts_np = np.array(verts)
        # vedo expects triangulated faces or can handle quads
        # Convert quad faces to triangles
        faces_np = np.array(faces)
        triangles = []
        for f in faces_np:
            triangles.append([f[0], f[1], f[2]])
            triangles.append([f[0], f[2], f[3]])
        triangles = np.array(triangles)
        
        mesh = Mesh([verts_np, triangles])
        color = row_colors[row_idx % len(row_colors)]
        mesh.color(color)
        mesh.lighting('glossy')
        meshes.append(mesh)
    
    # Create plotter and show
    plt = Plotter(title=title, bg='white', bg2='lightgray')
    plt.add(meshes)
    plt.add(Text2D("Controls: drag=rotate, scroll=zoom, shift+drag=pan", pos='bottom-left', s=0.7))
    plt.show()
    return plt


def view_model_from_mesh_data(mesh_data_list, row_colors=None, title="Knitting Model Viewer"):
    """
    View the knitting model from pre-computed mesh data using vedo.
    
    Args:
        mesh_data_list: List of (verts, edges, faces, n_points) tuples
        row_colors: Optional list of (r, g, b) tuples per row (values 0-1)
        title: Window title
    """
    try:
        from vedo import Mesh, Plotter, Text2D
    except ImportError:
        print("vedo not installed. Install with: pip install vedo")
        return
    
    n_rows = len(mesh_data_list)
    if row_colors is None:
        palette = [
            (0.15, 0.35, 0.75),  # Blue
            (0.90, 0.75, 0.20),  # Yellow
            (0.85, 0.25, 0.25),  # Red
            (0.30, 0.70, 0.40),  # Green
        ]
        row_colors = [palette[i % len(palette)] for i in range(n_rows)]
    
    meshes = []
    for row_idx, (verts, edges, faces, n_points) in enumerate(mesh_data_list):
        verts_np = np.array(verts)
        faces_np = np.array(faces)
        # Convert quads to triangles
        triangles = []
        for f in faces_np:
            triangles.append([f[0], f[1], f[2]])
            triangles.append([f[0], f[2], f[3]])
        triangles = np.array(triangles)
        
        mesh = Mesh([verts_np, triangles])
        color = row_colors[row_idx % len(row_colors)]
        mesh.color(color)
        mesh.lighting('glossy')
        meshes.append(mesh)
    
    plt = Plotter(title=title, bg='white', bg2='lightgray')
    plt.add(meshes)
    plt.add(Text2D("Controls: drag=rotate, scroll=zoom, shift+drag=pan", pos='bottom-left', s=0.7))
    plt.show()
    return plt


def view_obj_file(obj_path, color=(0.5, 0.5, 0.8), title="OBJ Viewer"):
    """
    View an OBJ file using vedo.
    
    Args:
        obj_path: Path to the OBJ file (or list of paths)
        color: RGB tuple (0-1) or list of colors for multiple files
        title: Window title
    """
    try:
        from vedo import load, Plotter, Text2D
    except ImportError:
        print("vedo not installed. Install with: pip install vedo")
        return
    
    if isinstance(obj_path, str):
        obj_path = [obj_path]
        color = [color]
    elif not isinstance(color, list):
        color = [color] * len(obj_path)
    
    meshes = []
    for path, c in zip(obj_path, color):
        mesh = load(path)
        mesh.color(c)
        mesh.lighting('glossy')
        meshes.append(mesh)
    
    plt = Plotter(title=title, bg='white', bg2='lightgray')
    plt.add(meshes)
    plt.add(Text2D("Controls: drag=rotate, scroll=zoom, shift+drag=pan", pos='bottom-left', s=0.7))
    plt.show()
    return plt


# %% INTERACTIVE SPLINE EDITOR (exactly like spline.py - move points & change radius)

class InteractiveSplineEditor:
    """
    Interactive spline editor exactly like spline.py.
    - Click to select control points
    - Drag to move points (position mode)
    - W/S to adjust radius (radius mode)
    - M to toggle between position/radius modes
    - R to render with Mitsuba
    - O to continue optimization
    - F to finish
    """
    
    YARN_COLORS = ["red", "dodgerblue", "gold", "saddlebrown", "forestgreen", "purple"]
    
    def __init__(self, geometry_params, bitmap, optimizer=None, ref_img=None):
        from vedo import Plotter, Text2D, Sphere, Spline, Tube, Line
        
        self.geometry_params = np.array(geometry_params, dtype=np.float64)
        self.bitmap = bitmap
        self.optimizer = optimizer
        self.ref_img = ref_img
        
        # Generate control points from geometry params
        self.all_control_points = []
        self.all_radii = []
        self.default_radius = float(geometry_params[4])  # radius param
        
        self._generate_control_points()
        
        # Editor state
        self.selected_point_idx = None
        self.selected_row = None
        self.point_spheres = []
        self.spline_actors = []
        self.mesh_actors = []
        self.dragging = False
        self.editing_locked = False
        self.edit_mode = 'position'  # 'position' or 'radius'
        
        # Result action
        self.action = 'finish'
        
        # UI elements
        self.mode_text = None
        
        # Plotter
        self.plotter = Plotter(
            bg='blackboard', axes=1,
            title="Knitting Spline Editor - Click to select, drag to move"
        )
        
        # For double-click detection
        self.last_click_time = 0.0
    
    def _generate_control_points(self):
        """Generate control points from geometry parameters (like spline.py's generate_knit_row)."""
        # params: [stitch_bulge, stitch_z, loop_height, dy, radius, ...]
        stitch_bulge = float(self.geometry_params[0])
        stitch_z = float(self.geometry_params[1])
        dy = float(self.geometry_params[3])
        radius = float(self.geometry_params[4])
        
        n_rows, n_cols = int(self.bitmap.shape[0]), int(self.bitmap.shape[1])
        samples_per_stitch = 5
        
        np.random.seed(42)  # Reproducible
        
        for row in range(n_rows):
            pts = []
            # Alternate direction every row
            col_range = range(n_cols) if row % 2 == 0 else range(n_cols - 1, -1, -1)
            
            for c in col_range:
                # Parameter t from 0 to 2*pi for each stitch
                if row % 2 == 0:
                    t_vals = np.linspace(0, 2*np.pi, samples_per_stitch, endpoint=False)
                else:
                    t_vals = np.linspace(2*np.pi, 0, samples_per_stitch, endpoint=False)
                
                for t in t_vals:
                    x = stitch_bulge * np.sin(2*t) + t/(2*np.pi)
                    y = -(np.cos(t) - 1)/2
                    z = stitch_z * (np.cos(2*t) - 1)/2
                    
                    # Offset by column and row
                    pts.append([c + x, row * dy + y, z])
            
            self.all_control_points.append(np.array(pts))
            self.all_radii.append(np.full(len(pts), radius))
    
    def _build_point_spheres(self):
        """Create clickable spheres for each control point."""
        from vedo import Sphere, Plane
        
        # Add a large ground plane for picking during drag (invisible but pickable)
        n_rows = len(self.all_control_points)
        n_cols = int(self.bitmap.shape[1]) if hasattr(self.bitmap, 'shape') else 5
        center = [n_cols/2, n_rows * self.geometry_params[3] / 2, 0]
        self.ground_plane = Plane(pos=center, normal=(0, 0, 1), s=(n_cols * 3, n_rows * 3))
        self.ground_plane.alpha(0.0).pickable(True)  # Invisible but pickable
        self.plotter.add(self.ground_plane)
        
        for r, row_pts in enumerate(self.all_control_points):
            for i, pt in enumerate(row_pts):
                sphere = Sphere(pt, r=0.12).color("white").alpha(0.8)  # Larger spheres for easier clicking
                sphere.pickable(True)
                self.point_spheres.append(sphere)
                self.plotter.add(sphere)
        
        print(f"Created {len(self.point_spheres)} control point spheres")
    
    def rebuild_visuals(self):
        """Rebuild splines and meshes from current control points with varying radii."""
        from vedo import Spline, Tube, Line
        
        # Remove old actors
        for actor in self.spline_actors + self.mesh_actors:
            self.plotter.remove(actor)
        self.spline_actors.clear()
        self.mesh_actors.clear()
        
        # Rebuild splines and meshes
        for r, row_pts in enumerate(self.all_control_points):
            row_color = self.YARN_COLORS[r % len(self.YARN_COLORS)]
            row_radii = self.all_radii[r]
            
            # Create smooth spline
            row_spline = Spline(row_pts, res=200)
            n_spline_pts = len(row_spline.vertices)
            n_ctrl_pts = len(row_pts)
            
            # Interpolate radii to match spline resolution
            if n_ctrl_pts > 1:
                radii_interp = np.interp(
                    np.linspace(0, 1, n_spline_pts),
                    np.linspace(0, 1, n_ctrl_pts),
                    row_radii
                )
            else:
                radii_interp = np.full(n_spline_pts, row_radii[0])
            
            # Spline line visualization
            spline_line = Line(row_spline.vertices).color(row_color).linewidth(3)
            self.spline_actors.append(spline_line)
            self.plotter.add(spline_line)
            
            # Mesh tube with varying radius
            row_mesh = Tube(row_spline.vertices, r=radii_interp, res=8)
            row_mesh.color(row_color).alpha(0.6).lighting("plastic")
            row_mesh.pickable(True)  # Make pickable for dragging
            self.mesh_actors.append(row_mesh)
            self.plotter.add(row_mesh)
        
        self.plotter.render()
    
    def update_point_colors(self):
        """Update point sphere colors based on selection and mode."""
        flat_idx = 0
        for r, row_pts in enumerate(self.all_control_points):
            for i in range(len(row_pts)):
                sphere = self.point_spheres[flat_idx]
                if self.selected_row == r and self.selected_point_idx == i:
                    sphere.color("lime").alpha(1.0)
                elif self.edit_mode == 'radius':
                    sphere.color("orange").alpha(0.9)
                else:
                    sphere.color("white").alpha(0.8)
                flat_idx += 1
    
    def update_mode_display(self):
        """Update the mode indicator text."""
        from vedo import Text2D
        
        if self.mode_text:
            self.plotter.remove(self.mode_text)
        
        if self.edit_mode == 'position':
            mode_str = "MODE: POSITION | M=switch to RADIUS | R=Render | O=Optimize | F=Finish"
            color = "cyan"
        else:
            mode_str = "MODE: RADIUS (+/- or W/S to change) | M=switch to POSITION | R=Render | O=Optimize | F=Finish"
            color = "orange"
        
        self.mode_text = Text2D(mode_str, pos='bottom-center', c=color, s=0.9, bold=True)
        self.plotter.add(self.mode_text)
        self.plotter.render()
    
    def on_key_press(self, evt):
        """Handle keyboard input for moving selected point or changing radius."""
        key = evt.keypress.lower() if evt.keypress else ""
        
        if key:
            print(f"[KEY] '{key}' pressed, mode={self.edit_mode}, selected=({self.selected_row}, {self.selected_point_idx})")
        
        # R to render with Mitsuba
        if key == 'r':
            self.render_mitsuba()
            return
        
        # O to continue optimization
        if key == 'o':
            self.action = 'optimize'
            self.plotter.close()
            return
        
        # F to finish
        if key == 'f':
            self.action = 'finish'
            self.plotter.close()
            return
        
        # M to toggle mode
        if key == 'm':
            if self.edit_mode == 'position':
                self.edit_mode = 'radius'
                print("=== RADIUS MODE === Click a point, then +/- or W/S to change radius")
            else:
                self.edit_mode = 'position'
                print("=== POSITION MODE === Drag points or use WASD to move")
            self.update_mode_display()
            self.update_point_colors()
            return
        
        # Other keys require a selected point
        if self.selected_row is None or self.selected_point_idx is None:
            return
        
        r = self.selected_row
        i = self.selected_point_idx
        
        if self.edit_mode == 'radius':
            # Radius editing mode
            radius_delta = 0.02  # Bigger step for more visible changes
            current_radius = self.all_radii[r][i]
            
            changed = False
            if key in ['w', 'up', 'plus', 'equal', '=', '+']:
                self.all_radii[r][i] = min(0.5, current_radius + radius_delta)
                changed = True
                print(f">>> RADIUS INCREASED at Row {r}, Point {i}: {current_radius:.3f} -> {self.all_radii[r][i]:.3f}")
            elif key in ['s', 'down', 'minus', '-']:
                self.all_radii[r][i] = max(0.01, current_radius - radius_delta)
                changed = True
                print(f">>> RADIUS DECREASED at Row {r}, Point {i}: {current_radius:.3f} -> {self.all_radii[r][i]:.3f}")
            
            if changed:
                self.rebuild_visuals()
        else:
            # Position editing mode
            delta = 0.05
            pt = self.all_control_points[r][i].copy()
            
            moved = False
            if key == 'w':
                pt[1] += delta
                moved = True
            elif key == 's':
                pt[1] -= delta
                moved = True
            elif key == 'a':
                pt[0] -= delta
                moved = True
            elif key == 'd':
                pt[0] += delta
                moved = True
            elif key == 'q':
                pt[2] += delta
                moved = True
            elif key == 'e':
                pt[2] -= delta
                moved = True
            
            if moved:
                self.all_control_points[r][i] = pt
                flat_idx = sum(len(self.all_control_points[rr]) for rr in range(r)) + i
                self.point_spheres[flat_idx].pos(pt)
                self.rebuild_visuals()
                print(f"Moved point to ({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f})")
    
    def on_mouse_move(self, evt):
        """Handle mouse drag for moving selected point (position mode only)."""
        if self.edit_mode != 'position':
            return
        if not self.dragging or self.selected_row is None:
            return
        if evt.picked3d is None:
            return
        
        new_pos = np.array(evt.picked3d)
        r = self.selected_row
        i = self.selected_point_idx
        
        # Keep Z from original point (drag in XY plane)
        orig_z = self.all_control_points[r][i][2]
        new_pos[2] = orig_z
        
        self.all_control_points[r][i] = new_pos
        flat_idx = sum(len(self.all_control_points[rr]) for rr in range(r)) + i
        self.point_spheres[flat_idx].pos(new_pos)
        self.rebuild_visuals()
    
    def on_left_button_release(self, evt):
        """Stop dragging on mouse release."""
        if self.dragging:
            self.dragging = False
    
    def on_left_click(self, evt):
        """Handle click: select point, or double-click to deselect."""
        import time
        
        current_time = time.time()
        double_click = (current_time - self.last_click_time) < 0.3
        self.last_click_time = current_time
        
        if double_click and self.editing_locked:
            # Double-click: deselect
            self.editing_locked = False
            self.selected_row = None
            self.selected_point_idx = None
            self.dragging = False
            self.update_point_colors()
            self.plotter.render()
            return
        
        # In position mode with editing locked, start dragging
        # In radius mode, allow selecting new points
        if self.editing_locked and self.edit_mode == 'position':
            self.dragging = True
            return
        
        if evt.picked3d is None:
            print("No picked3d - click not on a pickable surface")
            return
        
        click_pos = np.array(evt.picked3d)
        
        # Find closest control point
        min_dist = float('inf')
        best_flat_idx = None
        
        for flat_idx, sphere in enumerate(self.point_spheres):
            sphere_pos = np.array(sphere.pos())
            dist = np.linalg.norm(click_pos - sphere_pos)
            if dist < min_dist:
                min_dist = dist
                best_flat_idx = flat_idx
        
        # Select if close enough (increased threshold for easier selection)
        if min_dist < 0.5 and best_flat_idx is not None:
            cumsum = 0
            for r, row_pts in enumerate(self.all_control_points):
                if best_flat_idx < cumsum + len(row_pts):
                    self.selected_row = r
                    self.selected_point_idx = best_flat_idx - cumsum
                    self.editing_locked = True
                    self.dragging = (self.edit_mode == 'position')
                    print(f">>> SELECTED: Row {r}, Point {self.selected_point_idx} (dist={min_dist:.3f})")
                    self.update_point_colors()
                    self.plotter.render()
                    return
                cumsum += len(row_pts)
        else:
            print(f"Click at ({click_pos[0]:.2f}, {click_pos[1]:.2f}, {click_pos[2]:.2f}), nearest dist={min_dist:.3f}")
    
    def save_mesh_obj(self, filepath="knitting_spline_mesh.obj"):
        """Save the current mesh to OBJ file."""
        from vedo import Spline
        
        segments = 8
        with open(filepath, "w") as f:
            f.write(f"# Knitting mesh from spline editor\n\n")
            
            vertex_offset = 0
            total_verts = 0
            total_faces = 0
            
            for r, row_pts in enumerate(self.all_control_points):
                row_radii = self.all_radii[r]
                
                row_spline = Spline(row_pts, res=200)
                spline_pts = row_spline.vertices
                n_spline_pts = len(spline_pts)
                n_ctrl_pts = len(row_pts)
                
                if n_ctrl_pts > 1:
                    radii_interp = np.interp(
                        np.linspace(0, 1, n_spline_pts),
                        np.linspace(0, 1, n_ctrl_pts),
                        row_radii
                    )
                else:
                    radii_interp = np.full(n_spline_pts, row_radii[0])
                
                f.write(f"# Row {r}\n")
                f.write(f"o row_{r}\n")
                
                # Build tube vertices
                for i, pt in enumerate(spline_pts):
                    if i == 0:
                        tangent = spline_pts[1] - spline_pts[0]
                    elif i == n_spline_pts - 1:
                        tangent = spline_pts[-1] - spline_pts[-2]
                    else:
                        tangent = spline_pts[i+1] - spline_pts[i-1]
                    tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
                    
                    up = np.array([0, 0, 1])
                    if abs(np.dot(tangent, up)) > 0.9:
                        up = np.array([1, 0, 0])
                    normal = np.cross(tangent, up)
                    normal = normal / (np.linalg.norm(normal) + 1e-8)
                    binormal = np.cross(tangent, normal)
                    
                    radius = radii_interp[i]
                    for j in range(segments):
                        theta = 2 * np.pi * j / segments
                        offset = radius * (np.cos(theta) * normal + np.sin(theta) * binormal)
                        v = pt + offset
                        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                        total_verts += 1
                
                # Build faces
                for i in range(n_spline_pts - 1):
                    for j in range(segments):
                        v0 = vertex_offset + i * segments + j + 1
                        v1 = vertex_offset + i * segments + (j + 1) % segments + 1
                        v2 = vertex_offset + (i + 1) * segments + (j + 1) % segments + 1
                        v3 = vertex_offset + (i + 1) * segments + j + 1
                        f.write(f"f {v0} {v1} {v2} {v3}\n")
                        total_faces += 1
                
                vertex_offset += n_spline_pts * segments
                f.write("\n")
        
        print(f"*** SAVED: {filepath} ({total_verts} vertices, {total_faces} faces) ***")
        return filepath
    
    def render_mitsuba(self):
        """Render current model with Mitsuba and show result."""
        import matplotlib.pyplot as mplt
        
        print("="*60)
        print("RENDERING WITH MITSUBA...")
        print("="*60)
        
        # Save mesh to OBJ
        obj_path = os.path.join(OUTPUT_DIR, "meshes", "spline_preview.obj")
        self.save_mesh_obj(obj_path)
        
        if self.optimizer is None:
            print("No optimizer - creating basic Mitsuba scene...")
            # Create basic scene
            scene = mi.load_dict({
                'type': 'scene',
                'integrator': {'type': 'path', 'max_depth': 4},
                'sensor': {
                    'type': 'perspective',
                    'fov': 45,
                    'to_world': mi.ScalarTransform4f.look_at(
                        origin=[4, 3, 5],
                        target=[2, 1, 0],
                        up=[0, 1, 0]
                    ),
                    'film': {
                        'type': 'hdrfilm',
                        'width': 512,
                        'height': 512,
                        'pixel_format': 'rgb',
                    },
                    'sampler': {'type': 'independent', 'sample_count': 64},
                },
                'light': {
                    'type': 'constant',
                    'radiance': {'type': 'rgb', 'value': [0.8, 0.85, 0.9]},
                },
                'mesh': {
                    'type': 'obj',
                    'filename': obj_path,
                    'bsdf': {
                        'type': 'principled',
                        'base_color': {'type': 'rgb', 'value': [0.8, 0.4, 0.3]},
                        'roughness': 0.7,
                    },
                },
            })
        else:
            scene = mi.load_dict(self.optimizer.get_scene_dict(obj_path, self.geometry_params))
        
        img = mi.render(scene, spp=64)
        img_np = np.clip(np.array(img), 0, 1)
        
        # Display
        has_ref = self.ref_img is not None
        fig, axes = mplt.subplots(1, 2 if has_ref else 1, figsize=(12 if has_ref else 6, 6))
        
        if has_ref:
            axes[0].imshow(self.ref_img)
            axes[0].set_title('Reference', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            axes[1].imshow(img_np)
            axes[1].set_title('Current Render', fontsize=14, fontweight='bold')
            axes[1].axis('off')
        else:
            axes.imshow(img_np)
            axes.set_title('Current Render', fontsize=14, fontweight='bold')
            axes.axis('off')
        
        mplt.tight_layout()
        mplt.show()
        print("Render complete! Close the window to continue editing.")
    
    def run(self):
        """Run the interactive spline editor."""
        from vedo import Text2D
        
        # Build control point spheres
        self._build_point_spheres()
        
        # Build initial meshes
        self.rebuild_visuals()
        
        # Add instructions
        instructions = Text2D(
            "Controls:\n"
            "  Click: Select point\n"
            "  Drag: Move point (position mode)\n"
            "  Double-click: Deselect\n"
            "  M: Toggle Position/Radius mode\n"
            "  W/S: Move up/down or +/- radius\n"
            "  A/D: Move left/right\n"
            "  Q/E: Move forward/back\n"
            "  R: Render with Mitsuba\n"
            "  O: Continue Optimization\n"
            "  F: Finish",
            pos='top-left', c='white', s=0.7, bg='black', alpha=0.8
        )
        self.plotter.add(instructions)
        
        # Initial mode display
        self.update_mode_display()
        
        # Add callbacks
        self.plotter.add_callback("KeyPress", self.on_key_press)
        self.plotter.add_callback("LeftButtonPress", self.on_left_click)
        self.plotter.add_callback("MouseMove", self.on_mouse_move)
        self.plotter.add_callback("LeftButtonRelease", self.on_left_button_release)
        
        # Show
        self.plotter.show(interactive=True)
        
        # Extract updated params from edited control points/radii
        updated_params = self._extract_updated_params()
        
        return self.all_control_points, self.all_radii, self.action, updated_params
    
    def _extract_updated_params(self):
        """
        Extract updated geometry parameters from edited control points and radii.
        Maps the visual edits back to optimizer parameters.
        """
        params = list(self.geometry_params)  # Copy original params
        
        # Update radius (param[4]) with average of all edited radii
        all_radii_flat = []
        for row_radii in self.all_radii:
            all_radii_flat.extend(row_radii)
        if len(all_radii_flat) > 0:
            avg_radius = float(np.mean(all_radii_flat))
            params[4] = avg_radius
            print(f"Updated radius param: {self.geometry_params[4]:.4f} -> {avg_radius:.4f}")
        
        # Estimate stitch_bulge from X range of control points
        # stitch_bulge affects the sin(2t) amplitude
        x_ranges = []
        for row_pts in self.all_control_points:
            if len(row_pts) > 1:
                x_min, x_max = row_pts[:, 0].min(), row_pts[:, 0].max()
                x_ranges.append(x_max - x_min)
        if len(x_ranges) > 0:
            n_cols = self.bitmap.shape[1]
            avg_x_range = np.mean(x_ranges)
            # Each stitch spans t=0 to 2*pi, x = stitch_bulge*sin(2t) + t/(2*pi)
            # For one stitch: x goes from 0 to 1, with bulge adding amplitude of stitch_bulge
            # Total per row: n_cols stitches, so range ~ n_cols + estimated bulge contribution
            # This is an approximation - bulge contributes ~2*stitch_bulge per stitch
            estimated_bulge = (avg_x_range - n_cols) / (2 * n_cols) if n_cols > 0 else params[0]
            if 0.05 < estimated_bulge < 1.0:  # Sanity check
                params[0] = estimated_bulge
                print(f"Estimated stitch_bulge: {self.geometry_params[0]:.4f} -> {estimated_bulge:.4f}")
        
        # Estimate dy (row spacing, param[3]) from Y positions
        if len(self.all_control_points) > 1:
            row_y_means = []
            for row_pts in self.all_control_points:
                row_y_means.append(np.mean(row_pts[:, 1]))
            dy_estimates = [row_y_means[i+1] - row_y_means[i] for i in range(len(row_y_means)-1)]
            if len(dy_estimates) > 0:
                avg_dy = np.mean(dy_estimates)
                if 0.1 < avg_dy < 2.0:  # Sanity check
                    params[3] = avg_dy
                    print(f"Estimated dy (row spacing): {self.geometry_params[3]:.4f} -> {avg_dy:.4f}")
        
        # Estimate stitch_z from Z range
        z_ranges = []
        for row_pts in self.all_control_points:
            if len(row_pts) > 1:
                z_min, z_max = row_pts[:, 2].min(), row_pts[:, 2].max()
                z_ranges.append(z_max - z_min)
        if len(z_ranges) > 0:
            avg_z_range = np.mean(z_ranges)
            # stitch_z affects cos(2t)-1 amplitude, max amplitude is 2*stitch_z
            estimated_stitch_z = -avg_z_range / 2 if avg_z_range > 0 else params[1]
            if -1.0 < estimated_stitch_z < 0:  # Should be negative
                params[1] = estimated_stitch_z
                print(f"Estimated stitch_z: {self.geometry_params[1]:.4f} -> {estimated_stitch_z:.4f}")
        
        return params


def interactive_spline_edit(geometry_params, bitmap, optimizer=None, ref_img=None):
    """
    Launch interactive spline editor with point dragging and radius editing.
    
    Args:
        geometry_params: Initial geometry parameters
        bitmap: 2D array defining knitting pattern
        optimizer: KnittingOptimizer for Mitsuba rendering
        ref_img: Reference image to compare against
        
    Returns:
        tuple: (updated_params, action) where action is 'optimize' or 'finish'
    """
    editor = InteractiveSplineEditor(geometry_params, bitmap, optimizer, ref_img)
    control_pts, radii, action, updated_params = editor.run()
    return updated_params, action


# %% INTERACTIVE PARAMETER EDITOR (like spline.py)

class InteractiveParamEditor:
    """
    Interactive editor for knitting model parameters with LIVE mesh updates.
    Works like spline.py - changes are visible immediately in the viewport.
    """
    
    YARN_COLORS = ["red", "dodgerblue", "gold", "saddlebrown", "forestgreen", "purple"]
    
    def __init__(self, init_params, bitmap, optimizer=None, ref_img=None):
        from vedo import Plotter, Text2D, Mesh
        
        self.params = np.array(init_params, dtype=np.float64)
        self.bitmap = bitmap
        self.optimizer = optimizer
        self.ref_img = ref_img
        self.consts = {'BITMAP': bitmap, 'loop_res': 32, 'segments': 8}
        
        # Parameter info
        self.param_names = [
            'stitch_bulge', 'stitch_z', 'loop_height', 'dy', 'radius',
            'curve_skew', 'y_sharp', 'x_bias', 'z_bias', 'ellipse_ratio'
        ]
        self.param_deltas = [0.02, 0.02, 0.1, 0.02, 0.01, 0.05, 0.05, 0.02, 0.02, 0.05]
        self.param_ranges = [
            (0.1, 0.6), (-0.8, -0.1), (0.2, 2.5), (0.05, 1.0), (0.02, 0.3),
            (-0.6, 0.6), (0.0, 0.8), (-0.2, 0.2), (-0.2, 0.2), (0.4, 1.6)
        ]
        
        # Current selection
        self.selected_param = 0
        
        # Result action
        self.action = 'finish'
        
        # Mesh actors
        self.mesh_actors = []
        
        # UI elements
        self.info_text = None
        self.mode_text = None
        
        # Plotter
        self.plotter = Plotter(
            bg='blackboard', axes=1, 
            title="Knitting Parameter Editor - Use Arrow Keys to Adjust"
        )
        
    def build_meshes(self):
        """Build mesh actors from current parameters."""
        from vedo import Mesh
        
        verts_list = compute_knitting_vertices(self.params, self.consts)
        faces_list = compute_knitting_faces(self.consts['segments'], verts_list)
        
        meshes = []
        for row_idx, ((verts, n_points), faces) in enumerate(zip(verts_list, faces_list)):
            verts_np = np.array(verts)
            faces_np = np.array(faces)
            # Convert quads to triangles
            triangles = []
            for f in faces_np:
                triangles.append([f[0], f[1], f[2]])
                triangles.append([f[0], f[2], f[3]])
            triangles = np.array(triangles)
            
            mesh = Mesh([verts_np, triangles])
            color = self.YARN_COLORS[row_idx % len(self.YARN_COLORS)]
            mesh.color(color).lighting('plastic').alpha(0.9)
            meshes.append(mesh)
        
        return meshes
    
    def rebuild_visuals(self):
        """Rebuild meshes with current parameters - LIVE UPDATE."""
        # Remove old meshes
        for m in self.mesh_actors:
            self.plotter.remove(m)
        self.mesh_actors.clear()
        
        # Build new meshes
        self.mesh_actors = self.build_meshes()
        for m in self.mesh_actors:
            self.plotter.add(m)
        
        # Update info display
        self.update_info_display()
        
        self.plotter.render()
    
    def update_info_display(self):
        """Update the parameter info text display."""
        from vedo import Text2D
        
        if self.info_text:
            self.plotter.remove(self.info_text)
        
        # Build parameter display string
        lines = ["=== PARAMETERS ==="]
        for i, (name, val) in enumerate(zip(self.param_names, self.params)):
            marker = ">>>" if i == self.selected_param else "   "
            lines.append(f"{marker} {name}: {val:.4f}")
        
        self.info_text = Text2D(
            "\n".join(lines),
            pos='top-left', c='white', s=0.7, bg='black', alpha=0.7
        )
        self.plotter.add(self.info_text)
    
    def update_mode_display(self):
        """Update mode/instructions text."""
        from vedo import Text2D
        
        if self.mode_text:
            self.plotter.remove(self.mode_text)
        
        name = self.param_names[self.selected_param]
        self.mode_text = Text2D(
            f"Selected: {name} | UP/DOWN=select param | LEFT/RIGHT=adjust value | R=Render | O=Optimize | F=Finish",
            pos='bottom-center', c='yellow', s=0.8, bold=True
        )
        self.plotter.add(self.mode_text)
        self.plotter.render()
    
    def on_key_press(self, evt):
        """Handle keyboard input for parameter adjustment."""
        key = evt.keypress.lower() if evt.keypress else ""
        
        changed = False
        
        # Navigation: select parameter
        if key in ['up', 'w']:
            self.selected_param = (self.selected_param - 1) % len(self.params)
            self.update_info_display()
            self.update_mode_display()
            return
        elif key in ['down', 's']:
            self.selected_param = (self.selected_param + 1) % len(self.params)
            self.update_info_display()
            self.update_mode_display()
            return
        
        # Adjust parameter value
        elif key in ['right', 'd', 'equal', 'plus']:
            delta = self.param_deltas[self.selected_param]
            self.params[self.selected_param] += delta
            changed = True
        elif key in ['left', 'a', 'minus']:
            delta = self.param_deltas[self.selected_param]
            self.params[self.selected_param] -= delta
            changed = True
        
        # Render with Mitsuba
        elif key == 'r':
            self.render_mitsuba()
            return
        
        # Continue optimization
        elif key == 'o':
            self.action = 'optimize'
            self.plotter.close()
            return
        
        # Finish
        elif key == 'f':
            self.action = 'finish'
            self.plotter.close()
            return
        
        if changed:
            # Clamp to valid range
            vmin, vmax = self.param_ranges[self.selected_param]
            self.params[self.selected_param] = np.clip(self.params[self.selected_param], vmin, vmax)
            
            name = self.param_names[self.selected_param]
            print(f">>> {name} = {self.params[self.selected_param]:.4f}")
            
            # LIVE UPDATE - rebuild meshes immediately
            self.rebuild_visuals()
    
    def render_mitsuba(self):
        """Render current model with Mitsuba and show result."""
        if self.optimizer is None:
            print("No optimizer provided, cannot render with Mitsuba.")
            return
        
        print("="*60)
        print("RENDERING WITH MITSUBA...")
        print("="*60)
        
        # Generate mesh
        verts_list = compute_knitting_vertices(self.params, self.consts)
        faces_list = compute_knitting_faces(self.consts['segments'], verts_list)
        mesh_data = [(v, [], f, n) for (v, n), f in zip(verts_list, faces_list)]
        
        # Save mesh
        temp_path = os.path.join(OUTPUT_DIR, "meshes", "interactive_preview")
        save_into_obj_files(mesh_data, temp_path)
        obj_path = temp_path + "_combined.obj"
        
        # Render
        scene = mi.load_dict(self.optimizer.get_scene_dict(obj_path, self.params))
        img = mi.render(scene, spp=64)
        img_np = np.clip(np.array(img), 0, 1)
        
        # Display using matplotlib (use mplt to avoid conflict with vedo)
        import matplotlib.pyplot as mplt
        
        has_ref = self.ref_img is not None
        fig, axes = mplt.subplots(1, 2 if has_ref else 1, figsize=(12 if has_ref else 6, 6))
        
        if has_ref:
            axes[0].imshow(self.ref_img)
            axes[0].set_title('Reference', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            axes[1].imshow(img_np)
            axes[1].set_title('Current Render', fontsize=14, fontweight='bold')
            axes[1].axis('off')
        else:
            axes.imshow(img_np)
            axes.set_title('Current Render', fontsize=14, fontweight='bold')
            axes.axis('off')
        
        mplt.tight_layout()
        mplt.show()
        print("Render complete! Close the window to continue editing.")
    
    def run(self):
        """Run the interactive editor."""
        from vedo import Text2D
        
        # Build initial meshes
        self.mesh_actors = self.build_meshes()
        for m in self.mesh_actors:
            self.plotter.add(m)
        
        # Add instructions
        instructions = Text2D(
            "Controls:\n"
            "  UP/DOWN or W/S: Select parameter\n"
            "  LEFT/RIGHT or A/D: Adjust value\n"
            "  R: Render with Mitsuba\n"
            "  O: Continue Optimization\n"
            "  F: Finish & Save",
            pos='top-right', c='white', s=0.7, bg='darkblue', alpha=0.8
        )
        self.plotter.add(instructions)
        
        # Initial displays
        self.update_info_display()
        self.update_mode_display()
        
        # Add key callback
        self.plotter.add_callback("KeyPress", self.on_key_press)
        
        # Show
        self.plotter.show(interactive=True)
        
        return list(self.params), self.action


def interactive_edit_model(geometry_params, bitmap, optimizer=None, ref_img=None):
    """
    Launch interactive parameter editor with LIVE mesh updates.
    
    Args:
        geometry_params: Initial geometry parameters 
        bitmap: 2D array defining the knitting pattern
        optimizer: KnittingOptimizer instance for Mitsuba rendering
        ref_img: Reference image to compare against
        
    Returns:
        tuple: (final_params, action) where action is 'optimize' or 'finish'
    """
    editor = InteractiveParamEditor(geometry_params, bitmap, optimizer, ref_img)
    return editor.run()


# %% OPTIMIZATION ENGINE

# Default colorful yarn palette (similar to reference image)
DEFAULT_YARN_PALETTE = [
    (0.15, 0.35, 0.75),  # Blue
    (0.90, 0.75, 0.20),  # Yellow
    (0.85, 0.25, 0.25),  # Red
    (0.30, 0.70, 0.40),  # Green
    (0.85, 0.45, 0.20),  # Orange
    (0.60, 0.30, 0.70),  # Purple
    (0.20, 0.65, 0.75),  # Cyan
    (0.90, 0.55, 0.65),  # Pink
]

# Color definitions for the specific pattern
COLOR_BROWN = (0.45, 0.25, 0.15)
COLOR_BLUE = (0.15, 0.35, 0.75)
COLOR_YELLOW = (0.95, 0.85, 0.20)
COLOR_RED = (0.85, 0.20, 0.20)

def get_loop_color(row_idx, loop_idx):
    """Get color for a specific loop based on pattern:
    Row 0: Brown (all loops)
    Row 1: Red/Yellow alternating per loop
    Row 2: Blue (all loops)
    Then repeat pattern...
    """
    pattern_row = row_idx % 3
    if pattern_row == 0:
        return COLOR_BROWN
    elif pattern_row == 1:
        # Red for even loops, Yellow for odd
        return COLOR_RED if loop_idx % 2 == 0 else COLOR_YELLOW
    else:  # pattern_row == 2
        return COLOR_BLUE

def save_colored_obj_files(mesh_data_list, base_filename="knitting_model"):
    """Save each row as a separate OBJ file for per-row coloring."""
    obj_paths = []
    for i, (verts, edges, faces, n_points) in enumerate(mesh_data_list):
        filepath = f"{base_filename}_row_{i:02d}.obj"
        with open(filepath, 'w') as f:
            f.write(f"# Knitting Model - Row {i}\n")
            f.write(f"o knittingRow_{i}\n\n")
            for vert in verts:
                if hasattr(vert, 'tolist'):
                    v = vert.tolist()
                else:
                    v = vert
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write("\n")
            for face in faces:
                face_indices = [str(idx + 1) for idx in face]
                f.write(f"f {' '.join(face_indices)}\n")
        obj_paths.append(filepath)
    return obj_paths

def save_per_loop_obj_files(mesh_data_list, base_filename="knitting_model", loop_res=32, segments=8):
    """Save each loop as a separate OBJ file for per-loop coloring.
    
    Returns:
        obj_paths: list of (row_idx, loop_idx, filepath) tuples
    """
    obj_info = []  # (row_idx, loop_idx, filepath)
    
    for row_idx, (verts, edges, faces, n_points) in enumerate(mesh_data_list):
        verts_np = np.array(verts)
        faces_np = np.array(faces)
        
        # n_points = loop_res * n_loops + 1
        n_loops = (n_points - 1) // loop_res
        
        for loop_idx in range(n_loops):
            # Vertex indices for this loop (including overlap at ends for continuity)
            start_pt = loop_idx * loop_res
            end_pt = (loop_idx + 1) * loop_res + 1  # +1 for overlap
            if end_pt > n_points:
                end_pt = n_points
            
            # Vertex range in flattened array (each point has `segments` vertices)
            v_start = start_pt * segments
            v_end = end_pt * segments
            
            loop_verts = verts_np[v_start:v_end]
            
            # Face indices for this loop
            # Each segment ring creates (loop_res) quads per loop
            # Face vertex indices reference points 0 to (end_pt - start_pt - 1)
            n_loop_pts = end_pt - start_pt
            i_grid, j_grid = np.meshgrid(np.arange(n_loop_pts - 1), np.arange(segments), indexing='ij')
            v0 = i_grid * segments + j_grid
            v1 = i_grid * segments + (j_grid + 1) % segments
            v2 = (i_grid + 1) * segments + (j_grid + 1) % segments
            v3 = (i_grid + 1) * segments + j_grid
            loop_faces = np.stack([v0, v1, v2, v3], axis=-1).reshape(-1, 4)
            
            filepath = f"{base_filename}_row_{row_idx:02d}_loop_{loop_idx:02d}.obj"
            with open(filepath, 'w') as f:
                f.write(f"# Knitting Model - Row {row_idx}, Loop {loop_idx}\n")
                f.write(f"o knittingRow_{row_idx}_loop_{loop_idx}\n\n")
                for vert in loop_verts:
                    if hasattr(vert, 'tolist'):
                        v = vert.tolist()
                    else:
                        v = list(vert)
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                f.write("\n")
                for face in loop_faces:
                    face_indices = [str(int(idx) + 1) for idx in face]
                    f.write(f"f {' '.join(face_indices)}\n")
            
            obj_info.append((row_idx, loop_idx, filepath))
    
    return obj_info
#%%

class KnittingOptimizer:
    # Yarn palette matching reference: Brown, Blue, Yellow, Red (repeating)
    YARN_PALETTE = [
        (0.45, 0.25, 0.15),  # Brown
        (0.15, 0.35, 0.75),  # Blue
        (0.95, 0.85, 0.20),  # Yellow
        (0.85, 0.20, 0.20),  # Red
    ]
    
    def __init__(
        self,
        reference_img,
        bitmap,
        learning_rate=0.01,
        optimize_texture=False,
        texture_learning_rate=0.05,
        initial_texture_rgb=(0.8, 0.4, 0.3),
        camera_params=(3.0, 45.0),
        # Clip / crop controls for loss computation
        loss_center_crop=(1.0, 1.0),
        loss_weights=(0.7, 0.3),
        row_colors=None,  # Per-row colors: list of (r, g, b) tuples
        use_colored_rows=True,  # Enable per-row coloring
    ):
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

        # Loss crop (clip) mask: compute loss only on a central region
        self.loss_center_crop = (float(loss_center_crop[0]), float(loss_center_crop[1]))
        self.loss_mask_np = self._build_center_crop_mask(self.loss_center_crop)
        self.loss_mask_tensor = mi.TensorXf(self.loss_mask_np)
        self.ref_tensor = mi.TensorXf(self.ref_array)
        self.loss_weights = (float(loss_weights[0]), float(loss_weights[1]))
        
        # Track optimization history
        self.loss_history = []
        self.param_history = []
        self.gradient_history = []  # Track gradients for each parameter

        # Camera + texture controls
        self.camera_params = (float(camera_params[0]), float(camera_params[1]))  # (dist_mult, fov_deg)

        self.optimize_texture = bool(optimize_texture)
        self.texture_optimizer = optax.adam(texture_learning_rate)
        self.texture_opt_state = None
        self.texture_rgb = jnp.array(initial_texture_rgb, dtype=jnp.float32)
        
        # Per-row coloring
        self.use_colored_rows = use_colored_rows
        n_rows = bitmap.shape[0]
        if row_colors is not None:
            self.row_colors = list(row_colors)
        else:
            # Cycle through palette for all rows
            self.row_colors = [self.YARN_PALETTE[i % len(self.YARN_PALETTE)] for i in range(n_rows)]

        self.texture_history = []

        # Cache for expensive bbox computation
        self._bbox_cache_key = None
        self._bbox_cache = None

    def _build_center_crop_mask(self, center_crop):
        """Create an (H,W,3) mask with ones in the center crop and zeros elsewhere."""
        crop_w, crop_h = float(center_crop[0]), float(center_crop[1])
        crop_w = float(np.clip(crop_w, 0.05, 1.0))
        crop_h = float(np.clip(crop_h, 0.05, 1.0))

        h, w = int(self.res_height), int(self.res_width)
        x0 = int(round((1.0 - crop_w) * 0.5 * w))
        x1 = int(round((1.0 + crop_w) * 0.5 * w))
        y0 = int(round((1.0 - crop_h) * 0.5 * h))
        y1 = int(round((1.0 + crop_h) * 0.5 * h))

        x0 = max(0, min(w, x0))
        x1 = max(0, min(w, x1))
        y0 = max(0, min(h, y0))
        y1 = max(0, min(h, y1))

        mask = np.zeros((h, w, 3), dtype=np.float32)
        mask[y0:y1, x0:x1, :] = 1.0
        return mask

    def _masked_mse_np(self, img_np, ref_np):
        diff = (img_np - ref_np) * self.loss_mask_np
        denom = float(np.sum(self.loss_mask_np))
        denom = denom if denom > 0 else 1.0
        return float(np.sum(diff * diff) / denom)

    def _compute_loss_dr(self, img_tensor):
        mask = self.loss_mask_tensor
        ref = self.ref_tensor
        denom = dr.sum(mask) + 1e-8

        # Pixel MSE on masked region
        diff = (img_tensor - ref) * mask
        pixel_mse = dr.sum(dr.sqr(diff)) / denom

        # Normalized MSE to reduce sensitivity to global exposure/brightness
        img_mean = dr.sum(img_tensor * mask) / denom
        ref_mean = dr.sum(ref * mask) / denom
        img_center = (img_tensor - img_mean) * mask
        ref_center = (ref - ref_mean) * mask
        img_var = dr.sum(dr.sqr(img_center)) / denom
        ref_var = dr.sum(dr.sqr(ref_center)) / denom
        img_norm = img_center / dr.sqrt(img_var + 1e-8)
        ref_norm = ref_center / dr.sqrt(ref_var + 1e-8)
        norm_mse = dr.sum(dr.sqr(img_norm - ref_norm)) / denom

        w_pixel, w_norm = self.loss_weights
        return w_pixel * pixel_mse + w_norm * norm_mse

    def _compute_bbox(self, geometry_params_np):
        """Compute axis-aligned bounding box of the current mesh in world space."""
        key = tuple(np.round(np.asarray(geometry_params_np, dtype=np.float32), 6).tolist())
        if self._bbox_cache_key == key and self._bbox_cache is not None:
            return self._bbox_cache

        verts_list = compute_knitting_vertices(geometry_params_np, self.consts)
        all_verts = jnp.concatenate([v for v, _ in verts_list], axis=0)
        all_verts_np = np.asarray(all_verts, dtype=np.float32)
        vmin = np.min(all_verts_np, axis=0)
        vmax = np.max(all_verts_np, axis=0)

        self._bbox_cache_key = key
        self._bbox_cache = (vmin, vmax)
        return vmin, vmax

    def view(self, geometry_params, title="Knitting Model"):
        """
        View the current model interactively using vedo.
        
        Args:
            geometry_params: Array of geometry parameters
            title: Window title
        """
        return view_model(geometry_params, self.bitmap, row_colors=self.row_colors, title=title)

    def get_scene_dict(self, obj_path, params, camera_params=None, texture_rgb=None):
        """Create Mitsuba scene with properly framed camera and clean background"""
        n_rows, n_loops = self.bitmap.shape
        # params = [stitch_bulge, stitch_z, loop_height, dy, radius, curve_skew, y_sharp, x_bias, z_bias, ellipse_ratio]
        params_np = np.asarray(params, dtype=np.float32)
        dy = float(params_np[3])

        if camera_params is None:
            camera_params = self.camera_params
        dist_mult, fov = float(camera_params[0]), float(camera_params[1])

        if texture_rgb is None:
            texture_rgb = self.texture_rgb
        texture_rgb = [float(texture_rgb[0]), float(texture_rgb[1]), float(texture_rgb[2])]
        
        # Fit camera to mesh bounding box (prevents cropping / too-close framing)
        vmin, vmax = self._compute_bbox(params_np)
        center_np = (vmin + vmax) * 0.5
        width = float(vmax[0] - vmin[0])
        height = float(vmax[1] - vmin[1])
        depth = float(vmax[2] - vmin[2])

        # Aspect ratio from film
        aspect = float(self.res_width) / float(self.res_height)
        vfov = np.deg2rad(float(fov))
        vfov = max(vfov, 1e-3)
        hfov = 2.0 * np.arctan(np.tan(vfov * 0.5) * aspect)

        # Distance needed to fit width/height within fov
        dist_h = (0.5 * height) / max(np.tan(vfov * 0.5), 1e-6)
        dist_w = (0.5 * width) / max(np.tan(hfov * 0.5), 1e-6)
        dist = max(dist_h, dist_w)

        # Tighter padding so the object fills more of the frame
        dist = float(dist) * float(dist_mult)
        dist = dist + float(depth) * 0.20 + 0.01

        center = [float(center_np[0]), float(center_np[1]), float(center_np[2])]
        origin = [center[0], center[1], float(vmax[2]) + dist]
        
        return {
            "type": "scene",
            "integrator": {"type": "path", "max_depth": 2},
            "sensor": {
                "type": "perspective",
                "fov": fov,
                "to_world": mi.ScalarTransform4f.look_at(
                    origin=origin,
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
                    "reflectance": {"type": "rgb", "value": texture_rgb}
                }
            }
        }

    def get_colored_scene_dict(self, obj_paths, params, camera_params=None):
        """Create Mitsuba scene with per-row colored meshes."""
        n_rows, n_loops = self.bitmap.shape
        params_np = np.asarray(params, dtype=np.float32)
        dy = float(params_np[3])

        if camera_params is None:
            camera_params = self.camera_params
        dist_mult, fov = float(camera_params[0]), float(camera_params[1])

        # Fit camera to mesh bounding box
        vmin, vmax = self._compute_bbox(params_np)
        center_np = (vmin + vmax) * 0.5
        width = float(vmax[0] - vmin[0])
        height = float(vmax[1] - vmin[1])
        depth = float(vmax[2] - vmin[2])

        aspect = float(self.res_width) / float(self.res_height)
        vfov = np.deg2rad(float(fov))
        vfov = max(vfov, 1e-3)
        hfov = 2.0 * np.arctan(np.tan(vfov * 0.5) * aspect)

        dist_h = (0.5 * height) / max(np.tan(vfov * 0.5), 1e-6)
        dist_w = (0.5 * width) / max(np.tan(hfov * 0.5), 1e-6)
        dist = max(dist_h, dist_w)
        dist = float(dist) * float(dist_mult)
        dist = dist + float(depth) * 0.20 + 0.01

        center = [float(center_np[0]), float(center_np[1]), float(center_np[2])]
        origin = [center[0], center[1], float(vmax[2]) + dist]

        scene_dict = {
            "type": "scene",
            "integrator": {"type": "path", "max_depth": 2},
            "sensor": {
                "type": "perspective",
                "fov": fov,
                "to_world": mi.ScalarTransform4f.look_at(
                    origin=origin,
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
            "emitter": {
                "type": "constant",
                "radiance": {"type": "rgb", "value": [0.9, 0.9, 0.9]}
            },
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
        }

        # Add each row as a separate colored mesh
        for i, obj_path in enumerate(obj_paths):
            color = self.row_colors[i % len(self.row_colors)]
            scene_dict[f"mesh_row_{i}"] = {
                "type": "obj",
                "filename": obj_path,
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {"type": "rgb", "value": list(color)}
                }
            }

        return scene_dict

    def get_per_loop_colored_scene_dict(self, obj_info, params, camera_params=None):
        """Create Mitsuba scene with per-loop colored meshes.
        
        Args:
            obj_info: list of (row_idx, loop_idx, filepath) tuples from save_per_loop_obj_files
        """
        n_rows, n_loops = self.bitmap.shape
        params_np = np.asarray(params, dtype=np.float32)
        dy = float(params_np[3])

        if camera_params is None:
            camera_params = self.camera_params
        dist_mult, fov = float(camera_params[0]), float(camera_params[1])

        # Fit camera to mesh bounding box
        vmin, vmax = self._compute_bbox(params_np)
        center_np = (vmin + vmax) * 0.5
        width = float(vmax[0] - vmin[0])
        height = float(vmax[1] - vmin[1])
        depth = float(vmax[2] - vmin[2])

        aspect = float(self.res_width) / float(self.res_height)
        vfov = np.deg2rad(float(fov))
        vfov = max(vfov, 1e-3)
        hfov = 2.0 * np.arctan(np.tan(vfov * 0.5) * aspect)

        dist_h = (0.5 * height) / max(np.tan(vfov * 0.5), 1e-6)
        dist_w = (0.5 * width) / max(np.tan(hfov * 0.5), 1e-6)
        dist = max(dist_h, dist_w)
        dist = float(dist) * float(dist_mult)
        dist = dist + float(depth) * 0.20 + 0.01

        center = [float(center_np[0]), float(center_np[1]), float(center_np[2])]
        origin = [center[0], center[1], float(vmax[2]) + dist]

        scene_dict = {
            "type": "scene",
            "integrator": {"type": "path", "max_depth": 2},
            "sensor": {
                "type": "perspective",
                "fov": fov,
                "to_world": mi.ScalarTransform4f.look_at(
                    origin=origin,
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
            "emitter": {
                "type": "constant",
                "radiance": {"type": "rgb", "value": [0.9, 0.9, 0.9]}
            },
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
        }

        # Add each loop as a separate colored mesh
        for row_idx, loop_idx, obj_path in obj_info:
            color = get_loop_color(row_idx, loop_idx)
            scene_dict[f"mesh_row_{row_idx}_loop_{loop_idx}"] = {
                "type": "obj",
                "filename": obj_path,
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {"type": "rgb", "value": list(color)}
                }
            }

        return scene_dict

    def _find_scene_key(self, params_scene, predicate):
        for k in params_scene.keys():
            if predicate(k):
                return k
        return None

    def calibrate_camera_to_reference(
        self,
        geometry_params,
        output_dir=OUTPUT_DIR,
        dist_mult_grid=(0.85, 0.95, 1.05, 1.15, 1.25),
        fov_grid=(30.0, 40.0, 45.0, 50.0, 60.0),
        spp=16,
        refine_with_scipy=False,
        scipy_max_iter=25,
    ):
        """Find camera params (distance multiplier, fov) that best match the reference image.

        This keeps the film resolution equal to the reference resolution.
        """
        geometry_params = np.array(geometry_params, dtype=np.float32)

        # Build mesh once
        verts_list = compute_knitting_vertices(geometry_params, self.consts)
        faces_list = compute_knitting_faces(self.consts['segments'], verts_list)
        mesh_data = [(v, [], f, n) for (v, n), f in zip(verts_list, faces_list)]
        path = os.path.join(output_dir, "meshes", "camera_calib.obj")
        
        if self.use_colored_rows:
            loop_res = self.consts['loop_res']
            segments = self.consts['segments']
            obj_info = save_per_loop_obj_files(mesh_data, path.replace(".obj", ""), loop_res, segments)
        else:
            save_into_obj_files(mesh_data, path.replace(".obj", ""))
            obj_path = path.replace(".obj", "_combined.obj")

        best_loss = float('inf')
        best_params = self.camera_params

        for dist_mult in dist_mult_grid:
            for fov in fov_grid:
                if self.use_colored_rows:
                    scene = mi.load_dict(self.get_per_loop_colored_scene_dict(obj_info, geometry_params, camera_params=(dist_mult, fov)))
                else:
                    scene = mi.load_dict(self.get_scene_dict(obj_path, geometry_params, camera_params=(dist_mult, fov)))
                img = mi.render(scene, spp=spp)
                loss = self._masked_mse_np(np.array(img), self.ref_array)
                if loss < best_loss:
                    best_loss = loss
                    best_params = (float(dist_mult), float(fov))

        # Optional local refinement
        if refine_with_scipy:
            try:
                from scipy.optimize import minimize

                def objective(x):
                    dist_mult, fov = float(x[0]), float(x[1])
                    # keep params in reasonable bounds
                    dist_mult = float(np.clip(dist_mult, 0.5, 2.0))
                    fov = float(np.clip(fov, 20.0, 75.0))
                    if self.use_colored_rows:
                        scene = mi.load_dict(self.get_per_loop_colored_scene_dict(obj_info, geometry_params, camera_params=(dist_mult, fov)))
                    else:
                        scene = mi.load_dict(self.get_scene_dict(obj_path, geometry_params, camera_params=(dist_mult, fov)))
                    img = mi.render(scene, spp=spp)
                    return self._masked_mse_np(np.array(img), self.ref_array)

                result = minimize(
                    objective,
                    x0=np.array(best_params, dtype=np.float32),
                    method='Nelder-Mead',
                    options={'maxiter': int(scipy_max_iter), 'disp': False},
                )
                cand = (float(result.x[0]), float(result.x[1]))
                cand_loss = float(result.fun)
                if cand_loss < best_loss:
                    best_loss = cand_loss
                    best_params = cand
            except Exception as e:
                print(f"  ⚠ SciPy refinement skipped: {e}")

        self.camera_params = best_params
        print(f"  ✓ Calibrated camera: dist_mult={best_params[0]:.3f}, fov={best_params[1]:.1f}°, loss={best_loss:.6f}")
        return best_params, best_loss
    
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
                if self.use_colored_rows:
                    loop_res = self.consts['loop_res']
                    segments = self.consts['segments']
                    obj_info = save_per_loop_obj_files(mesh_data, path.replace(".obj", ""), loop_res, segments)
                    scene = mi.load_dict(self.get_per_loop_colored_scene_dict(obj_info, p_var))
                else:
                    save_into_obj_files(mesh_data, path.replace(".obj", ""))
                    scene = mi.load_dict(self.get_scene_dict(path.replace(".obj", "_combined.obj"), p_var))
                img = mi.render(scene, spp=16)  # Lower spp for speed
                
                variations[f"{name}_{label}"] = np.array(img)
        
        return variations
    
    def visualize_epoch_summary(self, params, current_img, loss, param_names):
        """Create comprehensive visualization after each epoch"""
        n_params = len(param_names)
        fig = plt.figure(figsize=(20, 3 + 2.5 * n_params))
        gs = fig.add_gridspec(n_params + 1, 5, hspace=0.3, wspace=0.3)
        
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
                    if self.use_colored_rows:
                        # Check for per-loop files
                        epoch_loop_0 = os.path.join(OUTPUT_DIR, "meshes", f"epoch_{self.iteration:03d}_row_00_loop_00.obj")
                        if os.path.exists(epoch_loop_0):
                            # Reconstruct obj_info from saved files
                            n_rows, n_loops = self.bitmap.shape
                            obj_info = []
                            for r in range(n_rows):
                                for l in range(n_loops):
                                    fpath = os.path.join(OUTPUT_DIR, "meshes", f"epoch_{self.iteration:03d}_row_{r:02d}_loop_{l:02d}.obj")
                                    if os.path.exists(fpath):
                                        obj_info.append((r, l, fpath))
                            if obj_info:
                                scene = mi.load_dict(self.get_per_loop_colored_scene_dict(obj_info, p_var))
                                img = mi.render(scene, spp=16)
                                images.append(np.array(img))
                                continue
                    else:
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
                if self.use_colored_rows:
                    loop_res = self.consts['loop_res']
                    segments = self.consts['segments']
                    obj_info = save_per_loop_obj_files(mesh_data, path.replace(".obj", ""), loop_res, segments)
                    scene = mi.load_dict(self.get_per_loop_colored_scene_dict(obj_info, p_var))
                else:
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

    # Modification inside KnittingOptimizer.step()
    def compute_hybrid_loss(rendered_tensor, reference_tensor, mask):
        # 1. Standard Pixel MSE (keeps global color/position)
        pixel_loss = dr.sum(dr.sqr((rendered_tensor - reference_tensor) * mask)) / dr.sum(mask)
        
        # 2. Perceptual/Semantic Loss (NeuroDiff3D approach)
        # Convert Mitsuba tensors to Torch to use the feature extractor
        torch_render = self.mi_to_torch(rendered_tensor)
        torch_ref = self.mi_to_torch(reference_tensor)
        
        feat_render = self.feature_extractor(torch_render)
        feat_ref = self.feature_extractor(torch_ref)
        
        perceptual_loss = torch.mean((feat_render - feat_ref)**2)
        
        # Combine (weights derived from NeuroDiff3D's multimodal fusion)
        return 0.5 * pixel_loss + 0.5 * perceptual_loss.item()
    
    def compute_perceptual_loss(self, rendered_mi, reference_mi):
        """Helper to compute Perceptual Loss using PyTorch features."""
        
        # Initialize VGG if not already done (Semantic Encoder [cite: 235])
        if not hasattr(self, 'vgg'):
            self.vgg = models.vgg16(pretrained=True).features[:16].eval().cuda()
            for p in self.vgg.parameters(): p.requires_grad = False

        # Convert Mitsuba to Torch [cite: 259]
        def to_torch(mi_img):
            t = torch.from_numpy(np.array(mi_img)).permute(2, 0, 1).unsqueeze(0).cuda()
            t.requires_grad = True
            return t

        render_t = to_torch(rendered_mi)
        ref_t = to_torch(reference_mi).detach()

        # Extract features (Semantic Information [cite: 221, 234])
        feat_render = self.vgg(render_t)
        feat_ref = self.vgg(ref_t)
        
        loss = torch.mean((feat_render - feat_ref)**2)
        loss.backward()
        
        # Extract gradient dL/dImage to inject back into Mitsuba
        grad_img = render_t.grad.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
        
        return float(loss.item()), grad_img
    
    # def step(self, params, epsilon=0.01):
    #     self.iteration += 1
    #     params_np = np.array(params)
        
    #     # Generate mesh geometry from current parameters
    #     verts_list = compute_knitting_vertices(params_np, self.consts)
    #     faces_list = compute_knitting_faces(self.consts['segments'], verts_list)
    #     mesh_data = [(v, [], f, n) for (v, n), f in zip(verts_list, faces_list)]
        
    #     # Save mesh to OBJ file (one per epoch)
    #     path = os.path.join(OUTPUT_DIR, "meshes", f"epoch_{self.iteration:03d}.obj")
    #     save_into_obj_files(mesh_data, path.replace(".obj", ""))
    #     obj_path = path.replace(".obj", "_combined.obj")

    #     scene = mi.load_dict(self.get_scene_dict(obj_path, params_np))
        
    #     # Access mesh vertex positions through scene parameters
    #     params_scene = mi.traverse(scene)
    #     vertex_key = [k for k in params_scene.keys() if 'vertex_positions' in k]

    #     # Optional: enable texture gradient
    #     tex_key = None
    #     if self.optimize_texture:
    #         tex_key = self._find_scene_key(params_scene, lambda k: ('reflectance' in k and 'value' in k))
    #         if tex_key is None:
    #             tex_key = self._find_scene_key(params_scene, lambda k: 'reflectance' in k)
    #         if tex_key is not None:
    #             tex_val = params_scene[tex_key]
    #             dr.enable_grad(tex_val)
    #             params_scene[tex_key] = tex_val
        
    #     if len(vertex_key) > 0:
    #         # Enable gradient tracking on vertex positions
    #         vertex_positions = params_scene[vertex_key[0]]
    #         dr.enable_grad(vertex_positions)
    #         params_scene[vertex_key[0]] = vertex_positions
    #         params_scene.update()
            
    #         # Differentiable rendering (at reference resolution)
    #         img = mi.render(scene, params=params_scene, spp=32)
            
    #         # Compute masked loss with gradient tracking (clip ignores background)
    #         ref_flat = dr.ravel(mi.TensorXf(self.ref_array))
    #         img_flat = dr.ravel(img)
    #         mask_flat = dr.ravel(self.loss_mask_tensor)
    #         diff = (img_flat - ref_flat) * mask_flat
    #         loss_dr = dr.sum(dr.sqr(diff)) / (dr.sum(mask_flat) + 1e-8)
            
    #         # Backward pass through renderer
    #         dr.backward(loss_dr)
            
    #         # Extract gradients from vertex positions
    #         vertex_grads = dr.grad(vertex_positions)

    #         # Extract gradient for texture reflectance
    #         tex_grads_np = None
    #         if self.optimize_texture and tex_key is not None:
    #             try:
    #                 tex_grad = dr.grad(params_scene[tex_key])
    #                 tex_grads_np = np.array(tex_grad, dtype=np.float32).reshape(-1)[:3]
    #             except Exception:
    #                 tex_grads_np = None
            
    #         # Compute parameter gradients using chain rule with JAX Jacobian
    #         J = compute_geometry_jacobian(params_np, self.bitmap)
    #         vertex_grads_np = np.array(vertex_grads).reshape(-1, 3)
            
    #         # Compute gradients: sum over vertices of (dL/dv_i * dv_i/dp_j)
    #         grads_np = np.zeros(len(params_np))
    #         for i in range(len(params_np)):
    #             grads_np[i] = np.sum(vertex_grads_np * J[:, :, i])
            
    #         # Convert loss to numpy (properly extract scalar value from DrJit type)
    #         base_loss = float(dr.sum(loss_dr)[0])
    #         base_img = np.array(img)
    #     else:
    #         # Fallback: use finite differences if gradient tracking fails
    #         print("Gradient tracking unavailable, using finite differences")
    #         img = mi.render(scene, spp=32)
    #         base_loss = self._masked_mse_np(np.array(img), self.ref_array)
    #         base_img = np.array(img)
            
    #         # Finite difference gradients
    #         grads_np = np.zeros(len(params_np))
    #         for i in range(len(params_np)):
    #             p_eps = params_np.copy()
    #             p_eps[i] += epsilon
    #             verts_list_eps = compute_knitting_vertices(p_eps, self.consts)
    #             faces_list_eps = compute_knitting_faces(self.consts['segments'], verts_list_eps)
    #             mesh_data_eps = [(v, [], f, n) for (v, n), f in zip(verts_list_eps, faces_list_eps)]
                
    #             path_eps = os.path.join(OUTPUT_DIR, "meshes", f"temp_eps_{i}.obj")
    #             save_into_obj_files(mesh_data_eps, path_eps.replace(".obj", ""))
    #             scene_eps = mi.load_dict(self.get_scene_dict(path_eps.replace(".obj", "_combined.obj"), p_eps))
    #             img_eps = mi.render(scene_eps, spp=32)
    #             loss_eps = self._masked_mse_np(np.array(img_eps), self.ref_array)
    #             grads_np[i] = (loss_eps - base_loss) / epsilon
                
    #             # Clean up temp file
    #             try:
    #                 os.remove(path_eps.replace(".obj", "_combined.obj"))
    #                 os.remove(path_eps)
    #             except:
    #                 pass

    #         tex_grads_np = None
        
    #     # Update parameters using Adam optimizer
    #     if self.opt_state is None: 
    #         self.opt_state = self.optimizer.init(jnp.array(params))
    #     updates, self.opt_state = self.optimizer.update(jnp.array(grads_np), self.opt_state)
    #     new_params = optax.apply_updates(jnp.array(params), updates)

    #     # Update texture via differentiable gradients (if available)
    #     if self.optimize_texture and tex_grads_np is not None and len(tex_grads_np) == 3:
    #         if self.texture_opt_state is None:
    #             self.texture_opt_state = self.texture_optimizer.init(self.texture_rgb)

    #         tex_updates, self.texture_opt_state = self.texture_optimizer.update(
    #             jnp.array(tex_grads_np, dtype=jnp.float32),
    #             self.texture_opt_state,
    #         )
    #         self.texture_rgb = optax.apply_updates(self.texture_rgb, tex_updates)
    #         self.texture_rgb = jnp.clip(self.texture_rgb, 0.0, 1.0)
    #         self.texture_history.append(np.array(self.texture_rgb))
        
    #     # Clip to physical reality
    #     # [stitch_bulge, stitch_z, loop_height, dy, radius, curve_skew]
    #     new_params = jnp.clip(
    #         new_params,
    #         jnp.array([0.1, -0.8, 0.2, 0.05, 0.02]),
    #         jnp.array([0.6, -0.1, 2.5, 1.0, 0.3])
    #     )
        
    #     # Track history
    #     self.loss_history.append(base_loss)
    #     self.param_history.append(new_params)
    #     self.gradient_history.append(grads_np)
        
    #     # Display progress
    #     param_names = ['bulge', 'z', 'loop_h', 'dy', 'rad', 'skew']
    #     grad_str = ' | '.join([f"∂L/∂{name}={g:+.4f}" for name, g in zip(param_names, grads_np)])
    #     print(f"\nEpoch {self.iteration:02d} | Loss: {base_loss:.6f}")
    #     print(f"  Gradients: {grad_str}")
    #     print(f"  Parameters: {np.round(new_params, 4)}")
    #     if self.optimize_texture:
    #         print(f"  Texture RGB: {np.round(np.array(self.texture_rgb), 4)}")
    #     print(f"  [Using DrJit autodiff through Mitsuba renderer]")
        
    #     # Save progress render
    #     mi.util.write_bitmap(os.path.join(OUTPUT_DIR, "renders", f"iter_{self.iteration:03d}.png"), base_img)
        
    #     # Create comprehensive visualization
    #     print(f"  Creating epoch visualization...")
    #     self.visualize_epoch_summary(new_params, base_img, base_loss, param_names)
        
    #     return new_params
    def step(self, params, epsilon=0.01):
        self.iteration += 1
        params_np = np.array(params)
        
        # Generate mesh geometry from current parameters
        verts_list = compute_knitting_vertices(params_np, self.consts)
        faces_list = compute_knitting_faces(self.consts['segments'], verts_list)
        mesh_data = [(v, [], f, n) for (v, n), f in zip(verts_list, faces_list)]
        
        # Save mesh to OBJ file(s)
        path = os.path.join(OUTPUT_DIR, "meshes", f"epoch_{self.iteration:03d}.obj")
        
        # Always save combined OBJ for gradient computation
        save_into_obj_files(mesh_data, path.replace(".obj", ""))
        obj_path_combined = path.replace(".obj", "_combined.obj")
        
        # For gradient computation, use combined mesh (single vertex_positions array)
        scene = mi.load_dict(self.get_scene_dict(obj_path_combined, params_np))
        
        # Access mesh vertex positions through scene parameters
        params_scene = mi.traverse(scene)
        vertex_key = [k for k in params_scene.keys() if 'vertex_positions' in k]

        # Optional: enable texture gradient
        tex_key = None
        if self.optimize_texture:
            tex_key = self._find_scene_key(params_scene, lambda k: ('reflectance' in k and 'value' in k))
            if tex_key is None:
                tex_key = self._find_scene_key(params_scene, lambda k: 'reflectance' in k)
            if tex_key is not None:
                tex_val = params_scene[tex_key]
                dr.enable_grad(tex_val)
                params_scene[tex_key] = tex_val
        
        if len(vertex_key) > 0:
            # Enable gradient tracking on vertex positions
            vertex_positions = params_scene[vertex_key[0]]
            dr.enable_grad(vertex_positions)
            params_scene[vertex_key[0]] = vertex_positions
            params_scene.update()
            
            # Differentiable rendering (at reference resolution)
            img = mi.render(scene, params=params_scene, spp=32)
            
            # Compute blended loss with gradient tracking (clip ignores background)
            loss_dr = self._compute_loss_dr(img)
            
            # Backward pass through renderer
            dr.backward(loss_dr)
            
            # Extract gradients from vertex positions
            vertex_grads = dr.grad(vertex_positions)

            # Extract gradient for texture reflectance
            tex_grads_np = None
            if self.optimize_texture and tex_key is not None:
                try:
                    tex_grad = dr.grad(params_scene[tex_key])
                    tex_grads_np = np.array(tex_grad, dtype=np.float32).reshape(-1)[:3]
                except Exception:
                    tex_grads_np = None
            
            # Compute parameter gradients using chain rule with JAX Jacobian
            J = compute_geometry_jacobian(params_np, self.bitmap)
            vertex_grads_np = np.array(vertex_grads).reshape(-1, 3)
            
            # Compute gradients: sum over vertices of (dL/dv_i * dv_i/dp_j)
            grads_np = np.zeros(len(params_np))
            for i in range(len(params_np)):
                grads_np[i] = np.sum(vertex_grads_np * J[:, :, i])
            
            # Convert loss to numpy (properly extract scalar value from DrJit type)
            base_loss = float(dr.ravel(loss_dr)[0])
            base_img = np.array(img)
        else:
            # Fallback: use finite differences if gradient tracking fails
            print("Gradient tracking unavailable, using finite differences")
            img = mi.render(scene, spp=32)
            base_loss = self._masked_mse_np(np.array(img), self.ref_array)
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
                loss_eps = self._masked_mse_np(np.array(img_eps), self.ref_array)
                grads_np[i] = (loss_eps - base_loss) / epsilon
                
                # Clean up temp file
                try:
                    os.remove(path_eps.replace(".obj", "_combined.obj"))
                    os.remove(path_eps)
                except:
                    pass

            tex_grads_np = None
        
        # Update parameters using Adam optimizer
        if self.opt_state is None: 
            self.opt_state = self.optimizer.init(jnp.array(params))
        updates, self.opt_state = self.optimizer.update(jnp.array(grads_np), self.opt_state)
        new_params = optax.apply_updates(jnp.array(params), updates)

        # Update texture via differentiable gradients (if available)
        if self.optimize_texture and tex_grads_np is not None and len(tex_grads_np) == 3:
            if self.texture_opt_state is None:
                self.texture_opt_state = self.texture_optimizer.init(self.texture_rgb)

            tex_updates, self.texture_opt_state = self.texture_optimizer.update(
                jnp.array(tex_grads_np, dtype=jnp.float32),
                self.texture_opt_state,
            )
            self.texture_rgb = optax.apply_updates(self.texture_rgb, tex_updates)
            self.texture_rgb = jnp.clip(self.texture_rgb, 0.0, 1.0)
            self.texture_history.append(np.array(self.texture_rgb))
        
        # Clip to physical reality
        # [stitch_bulge, stitch_z, loop_height, dy, radius, curve_skew, y_sharp, x_bias, z_bias, ellipse_ratio]
        new_params = jnp.clip(
            new_params,
            jnp.array([0.1, -0.8, 0.2, 0.05, 0.02, -0.6, 0.0, -0.2, -0.2, 0.4]),
            jnp.array([0.6, -0.1, 2.5, 1.0, 0.3, 0.6, 0.8, 0.2, 0.2, 1.6])
        )
        
        # Track history
        self.loss_history.append(base_loss)
        self.param_history.append(new_params)
        self.gradient_history.append(grads_np)
        
        # Render colored version for visualization if enabled
        if self.use_colored_rows:
            # Use per-loop coloring for pattern: Brown, Blue, Yellow/Red alternating
            loop_res = self.consts['loop_res']
            segments = self.consts['segments']
            obj_info = save_per_loop_obj_files(mesh_data, path.replace(".obj", ""), loop_res, segments)
            colored_scene = mi.load_dict(self.get_per_loop_colored_scene_dict(obj_info, params_np))
            colored_img = mi.render(colored_scene, spp=32)
            display_img = np.array(colored_img)
        else:
            display_img = base_img
        
        # Display progress
        param_names = ['bulge', 'z', 'loop_h', 'dy', 'rad', 'skew', 'y_sharp', 'x_bias', 'z_bias', 'ellipse']
        grad_str = ' | '.join([f"∂L/∂{name}={g:+.4f}" for name, g in zip(param_names, grads_np)])
        print(f"\nEpoch {self.iteration:02d} | Loss: {base_loss:.6f}")
        print(f"  Gradients: {grad_str}")
        print(f"  Parameters: {np.round(new_params, 4)}")
        if self.optimize_texture:
            print(f"  Texture RGB: {np.round(np.array(self.texture_rgb), 4)}")
        print(f"  [Using DrJit autodiff through Mitsuba renderer]")
        
        # Save progress render
        mi.util.write_bitmap(os.path.join(OUTPUT_DIR, "renders", f"iter_{self.iteration:03d}.png"), display_img)
        
        # Create comprehensive visualization
        print(f"  Creating epoch visualization...")
        self.visualize_epoch_summary(new_params, display_img, base_loss, param_names)
        
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
    print(f"  loop_height:  {params[2]:.4f}")
    print(f"  dy (spacing): {params[3]:.4f}")
    print(f"  radius:       {params[4]:.4f}")
    print(f"  curve_skew:   {params[5]:.4f}\n")
    print(f"  y_sharp:      {params[6]:.4f}")
    print(f"  x_bias:       {params[7]:.4f}")
    print(f"  z_bias:       {params[8]:.4f}")
    print(f"  ellipse_ratio:{params[9]:.4f}\n")

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
            print(f"New best loss: {best_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nConverged! No improvement for {patience} iterations.")
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
        print("Render files not found, skipping comparison")
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
    param_names = [
        'stitch_bulge',
        'stitch_z',
        'loop_height',
        'dy',
        'radius',
        'curve_skew',
        'y_sharp',
        'x_bias',
        'z_bias',
        'ellipse_ratio',
    ]
    
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
    
    param_names = [
        'stitch_bulge',
        'stitch_z',
        'loop_height',
        'dy',
        'radius',
        'curve_skew',
        'y_sharp',
        'x_bias',
        'z_bias',
        'ellipse_ratio',
    ]
    save_path = os.path.join(output_dir, "parameter_effects.png")
    visualize_parameter_effects(params, bitmap, param_names, save_path)
    
    # Show interactively
    consts = {'BITMAP': bitmap, 'loop_res': 32, 'segments': 8}
    n_params = len(param_names)
    fig = plt.figure(figsize=(5 * n_params, 5))
    
    for i, name in enumerate(param_names):
        ax = fig.add_subplot(1, n_params, i + 1, projection='3d')
        delta = 0.2 * params[i]
        
        # Handle both list and JAX array
        params_low = list(params)
        params_low[i] = params[i] - delta
        params_high = list(params)
        params_high[i] = params[i] + delta
        
        test_params = [params_low, list(params), params_high]
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
    
    # Convert to JAX array if needed
    params_jax = jnp.array(params)
    J = compute_geometry_jacobian(params_jax, bitmap)
    J = np.array(J)  # Convert back to numpy for printing
    param_names = [
        'stitch_bulge',
        'stitch_z',
        'loop_height',
        'dy',
        'radius',
        'curve_skew',
        'y_sharp',
        'x_bias',
        'z_bias',
        'ellipse_ratio',
    ]
    
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
    
    param_names = [
        'stitch_bulge',
        'stitch_z',
        'loop_height',
        'dy',
        'radius',
        'curve_skew',
        'y_sharp',
        'x_bias',
        'z_bias',
        'ellipse_ratio',
    ]
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
    _, stitch_z, _, dy, _, _, _, _, _, _ = params_np
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
    _, stitch_z, _, dy, _, _, _, _, _, _ = params_np
    
    # Use same camera settings as optimization
    center = [n_loops / 2.0, ((n_rows - 1) * dy) / 2.0, stitch_z / 2.0]
    # dist = max(n_loops, n_rows * dy) * 1.05
    dist = max(n_loops, n_rows * dy) *0.7
    # CHANGE HERE

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
    MAX_ITERATIONS = 1
    SPP_OPTIMIZATION = 32
    SPP_FINAL = 128
    EPSILON = 0.01
    PATIENCE = MAX_ITERATIONS
    # TEST MODE: Set to True to test known parameters without optimization
    TEST_MODE = False

    # Extra optimization controls
    OPTIMIZE_TEXTURE = True
    TEXTURE_LEARNING_RATE = 0.05
    INITIAL_TEXTURE_RGB = (0.8, 0.4, 0.3)

    # Clip (crop) region used for LOSS only (keeps render resolution unchanged)
    # Example: (0.6, 0.6) means use only the central 60% width/height for loss.
    LOSS_CENTER_CROP = (0.55, 0.70)

    # Make a bigger model (more loops/rows) but still compare a clip
    BITMAP_ROWS = 9
    BITMAP_LOOPS = 4

    OPTIMIZE_CAMERA = True  # calibrate once before geometry optimization
    FORCE_CAMERA_PARAMS = True  # if True, skip calibration and use manual params below
    CAMERA_DIST_MULT = 0.70
    CAMERA_FOV = 45.0
    CAMERA_GRID_DIST = (0.85, 0.95, 1.05, 1.15, 1.25)
    CAMERA_GRID_FOV = (35.0, 40.0, 45.0, 50.0, 55.0)
    CAMERA_REFINE_SCIPY = False
    epsilon = 0.001
    TEST_PARAMS = [
        0.2993 + epsilon,
        -0.3505 + epsilon,
        1.0,
        0.40109998 + epsilon,
        0.1497 + epsilon,
        0.0,
        0.30,
        0.05,
        0.02,
        0.60,
    ]
    # =========================================================
    
    # Load reference image and setup
    # ref = Image.open("referenceImage.jpg").convert("RGB")
    # ref = Image.open("referenceImage.jpg").convert("RGB")
    ref = Image.open("referenceImage_cropped_new1.jpg").convert("RGB")

    BITMAP = jnp.ones((BITMAP_ROWS, BITMAP_LOOPS))
    # init_params = [0.2749, -0.375, 0.4210, 0.1251]
    epsilon = 0.001
    init_params = [
        0.2993 + epsilon,
        -0.3505 + epsilon,
        1.5,
        0.40109998 + epsilon,
        0.18 + epsilon,
        0.0,
        0.30,
        0.05,
        0.02,
        0.60,
    ]
    
    if TEST_MODE:
        print("="*80)
        print("RUNNING IN TEST MODE - VALIDATING PARAMETERS")
        print("="*80)
        test_parameters(TEST_PARAMS, ref, BITMAP, OUTPUT_DIR)
    else:
        # Display configuration
        print_hyperparameters(LEARNING_RATE, MAX_ITERATIONS, SPP_OPTIMIZATION, 
                             SPP_FINAL, EPSILON, PATIENCE)
        
        print_initial_parameters(init_params)
        
        # Run optimization
        opt = KnittingOptimizer(
            ref,
            BITMAP,
            learning_rate=LEARNING_RATE,
            optimize_texture=OPTIMIZE_TEXTURE,
            texture_learning_rate=TEXTURE_LEARNING_RATE,
            initial_texture_rgb=INITIAL_TEXTURE_RGB,
            loss_center_crop=LOSS_CENTER_CROP,
            camera_params=(CAMERA_DIST_MULT, CAMERA_FOV),
        )

        if OPTIMIZE_CAMERA and not FORCE_CAMERA_PARAMS:
            print("\n" + "="*80)
            print("CAMERA CALIBRATION (MATCH REFERENCE RESOLUTION / FRAMING)")
            print("="*80)
            opt.calibrate_camera_to_reference(
                init_params,
                output_dir=OUTPUT_DIR,
                dist_mult_grid=CAMERA_GRID_DIST,
                fov_grid=CAMERA_GRID_FOV,
                spp=16,
                refine_with_scipy=CAMERA_REFINE_SCIPY,
            )
        # ==================== INTERACTIVE EDITING LOOP ====================
        # Choose editor mode: spline (point-based) or parameter editor
        params = init_params
        best_loss = float('inf')
        
        print("\n" + "="*80)
        print("EDITOR MODE SELECTION")
        print("="*80)
        print("Choose editor mode:")
        print("  1. SPLINE EDITOR - Click & drag points, change radius (like spline.py)")
        print("  2. PARAMETER EDITOR - Adjust global parameters with arrow keys")
        print()
        mode_choice = input("Enter choice (1 or 2, default=1): ").strip()
        use_spline_editor = (mode_choice != '2')
        
        while True:
            print("\n" + "="*80)
            print("INTERACTIVE MODEL EDITOR")
            print("="*80)
            
            if use_spline_editor:
                print("SPLINE EDITOR: Click to select points, drag to move, M=radius mode")
                print("Press R to render with Mitsuba, O to optimize, F to finish.")
                print("="*80)
                
                # Open interactive spline editor (like spline.py - control point editing)
                # Returns updated params extracted from edited control points
                params, action = interactive_spline_edit(
                    params,
                    BITMAP,
                    optimizer=opt,
                    ref_img=ref
                )
                print(f"Updated params from spline editor: {[f'{p:.4f}' for p in params]}")
            else:
                print("PARAMETER EDITOR: UP/DOWN to select, LEFT/RIGHT to adjust.")
                print("Press R to render with Mitsuba, O to optimize, F to finish.")
                print("="*80)
                
                # Open interactive parameter editor (global params)
                params, action = interactive_edit_model(
                    params,
                    BITMAP,
                    optimizer=opt,
                    ref_img=ref
                )
            
            if action == 'finish':
                print("\n=== Finishing with current parameters ===")
                break
            elif action == 'optimize':
                print("\n=== Running optimization... ===")
                params, best_loss = run_optimization_loop(opt, params, MAX_ITERATIONS, 
                                                         EPSILON, PATIENCE)
                print(f"Optimization complete! Best loss: {best_loss:.6f}")
                print("Returning to interactive editor...")
                # Loop continues - user can edit again, optimize more, or finish
                # Loop continues - user can edit again, optimize more, or finish
        
        # Generate all visualizations
        #if len(opt.loss_history) > 0:
            #create_before_after_comparison(OUTPUT_DIR, ref, len(opt.loss_history))
            #create_optimization_summary(opt, init_params, params, OUTPUT_DIR)
        #visualize_parameter_effects_final(params, BITMAP, OUTPUT_DIR)
        #compute_jacobian_analysis(params, BITMAP)
        print_final_summary(opt, init_params, params, best_loss if best_loss != float('inf') else 0.0)
        
        # Save best model
        save_best_model(params, BITMAP, OUTPUT_DIR, spp_final=SPP_FINAL)
        
        print("\n" + "="*80)
        print("All visualizations complete and displayed interactively!")
        print("="*80)
# %%

