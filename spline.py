import numpy as np
from vedo import Spline, Tube, Points, Line, show, Text2D, Plotter, Sphere

# Color palette matching the target image (red, blue, yellow, brown)
YARN_COLORS = ["red", "dodgerblue", "gold", "saddlebrown"]

# Global state for interactive editing
class EditorState:
    def __init__(self):
        self.all_control_points = []
        self.all_radii = []  # Per-point radius values
        self.selected_point_idx = None
        self.selected_row = None
        self.point_spheres = []  # Clickable spheres for each control point
        self.spline_actors = []
        self.mesh_actors = []
        self.plotter = None
        self.dragging = False
        self.editing_locked = False  # True = point is locked for editing, double-click to release
        self.edit_mode = 'position'  # 'position' or 'radius'
        self.default_radius = 0.08
        self.mode_text = None  # Text display for current mode
        
state = EditorState()

def eval_curve(t, scale=1.0, stitch_bulge=0.30, stitch_z=-0.4):
    """
    Parametric stitch curve from draft_opt.py.
    t: parameter from 0 to 2*pi for one stitch
    Returns (x, y, z) where x spans [0, 1] over one stitch
    """
    x = stitch_bulge * np.sin(2*t) + t/(2*np.pi)
    y = -(np.cos(t) - 1)/2  # goes 0 -> 1 -> 0 (arch)
    z = stitch_z * (np.cos(2*t) - 1)/2
    return x, y * scale, z * scale

def generate_knit_row(row, cols=5, samples_per_stitch=5, stitch_bulge=0.30, stitch_z=-0.4, row_spacing=0.4, noise_scale=0.03):
    """
    Generates a single row of knitting as a separate loop.
    row_spacing: vertical distance between rows (default 0.4)
    noise_scale: amount of random variation to add (default 0.03)
    """
    pts = []
    # Alternate direction every row
    col_range = range(cols) if row % 2 == 0 else range(cols - 1, -1, -1)
    
    # Per-row random offset for natural variation
    row_y_offset = np.random.uniform(-0.02, 0.02)
    row_z_offset = np.random.uniform(-0.02, 0.02)
    
    for c in col_range:
        # Per-stitch random variation
        stitch_bulge_var = stitch_bulge + np.random.uniform(-0.03, 0.03)
        stitch_z_var = stitch_z + np.random.uniform(-0.05, 0.05)
        
        # Parameter t from 0 to 2*pi for each stitch
        if row % 2 == 0:
            t_vals = np.linspace(0, 2*np.pi, samples_per_stitch, endpoint=False)
        else:
            # Reverse direction for odd rows
            t_vals = np.linspace(2*np.pi, 0, samples_per_stitch, endpoint=False)
        
        for t in t_vals:
            x_local, y_local, z_local = eval_curve(t, scale=1.0, stitch_bulge=stitch_bulge_var, stitch_z=stitch_z_var)
            # Add random noise to each point
            noise_x = np.random.uniform(-noise_scale, noise_scale)
            noise_y = np.random.uniform(-noise_scale, noise_scale)
            noise_z = np.random.uniform(-noise_scale, noise_scale)
            
            # Offset by column and row
            x = c + x_local + noise_x
            y = row * row_spacing + y_local + row_y_offset + noise_y
            z = z_local + row_z_offset + noise_z
            pts.append([x, y, z])

    return np.array(pts)

def remove_duplicate_consecutive(points, tol=1e-9):
    """Remove consecutive duplicate points that cause splprep to fail."""
    if len(points) < 2:
        return points
    mask = np.ones(len(points), dtype=bool)
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - points[i-1]) < tol:
            mask[i] = False
    return points[mask]


# 1. Generate separate rows (one loop per row)
np.random.seed(42)  # For reproducible randomness
rows = 6
cols = 8
meshes = []
paths = []
all_control_points = []  # Store control points for export
all_spline_curves = []  # Store smooth spline curves for export

for r in range(rows):
    row_points = generate_knit_row(r, cols=cols)
    row_points = remove_duplicate_consecutive(row_points)
    all_control_points.append(row_points)  # Save for export
    
    # Create smooth spline for this row
    row_spline = Spline(row_points, res=800)
    all_spline_curves.append(row_spline.vertices)  # Save smooth curve
    
    # Create single tube from spline vertices
    row_mesh = Tube(row_spline.vertices, r=0.12, res=12)
    
    # Assign color based on row (cycling through palette)
    row_color = YARN_COLORS[r % len(YARN_COLORS)]
    row_mesh.color(row_color).lighting("plastic")
    meshes.append(row_mesh)
    
    # Create path visualization: smooth spline curve + control points
    spline_line = Line(row_spline.vertices).color("white").linewidth(2)  # Smooth spline path
    control_pts = Points(row_points, r=8).color("yellow")  # Original control points
    paths.extend([spline_line, control_pts])

# 3. Save control points as OBJ file (just 5 points per stitch)
def save_control_points():
    # File 1: Control points with radii
    filepath = "knitting_control_points.obj"
    total_points = sum(len(pts) for pts in state.all_control_points)
    with open(filepath, "w") as f:
        f.write(f"# Knitting control points - {total_points} total\n")
        f.write(f"# Format: v x y z (radius stored as comment after each row)\n")
        vertex_idx = 1
        for r, row_pts in enumerate(state.all_control_points):
            row_radii = state.all_radii[r]
            f.write(f"# Row {r} - {len(row_pts)} control points\n")
            for i, pt in enumerate(row_pts):
                f.write(f"v {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
            # Write radii as a comment for this row
            radii_str = " ".join(f"{rad:.4f}" for rad in row_radii)
            f.write(f"# radii: {radii_str}\n")
            # Write edges connecting consecutive points
            for i in range(len(row_pts) - 1):
                f.write(f"l {vertex_idx + i} {vertex_idx + i + 1}\n")
            vertex_idx += len(row_pts)
    print(f"*** SAVED: {filepath} ({total_points} control points) ***")
    
    # File 2: Full mesh with varying radius tubes - manually built like draft_opt
    mesh_filepath = "knitting_mesh.obj"
    segments = 8  # Number of vertices around tube circumference
    
    with open(mesh_filepath, "w") as f:
        f.write(f"# Knitting mesh with varying radius\n")
        f.write(f"# Generated from spline editor\n\n")
        
        vertex_offset = 0
        total_verts = 0
        total_faces = 0
        
        for r, row_pts in enumerate(state.all_control_points):
            row_radii = state.all_radii[r]
            
            # Create spline
            row_spline = Spline(row_pts, res=200)
            spline_pts = row_spline.vertices
            n_spline_pts = len(spline_pts)
            n_ctrl_pts = len(row_pts)
            
            # Interpolate radii along spline
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
            
            # Build tube vertices manually: for each spline point, create a ring
            # Compute tangent vectors for proper orientation
            for i, pt in enumerate(spline_pts):
                # Compute tangent
                if i == 0:
                    tangent = spline_pts[1] - spline_pts[0]
                elif i == n_spline_pts - 1:
                    tangent = spline_pts[-1] - spline_pts[-2]
                else:
                    tangent = spline_pts[i+1] - spline_pts[i-1]
                tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
                
                # Find perpendicular vectors (Frenet frame approximation)
                up = np.array([0, 0, 1])
                if abs(np.dot(tangent, up)) > 0.9:
                    up = np.array([1, 0, 0])
                normal = np.cross(tangent, up)
                normal = normal / (np.linalg.norm(normal) + 1e-8)
                binormal = np.cross(tangent, normal)
                
                # Create ring of vertices
                radius = radii_interp[i]
                for j in range(segments):
                    theta = 2 * np.pi * j / segments
                    offset = radius * (np.cos(theta) * normal + np.sin(theta) * binormal)
                    v = pt + offset
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                    total_verts += 1
            
            # Build faces: connect adjacent rings with quads
            for i in range(n_spline_pts - 1):
                for j in range(segments):
                    # Vertices of quad (1-indexed for OBJ)
                    v0 = vertex_offset + i * segments + j + 1
                    v1 = vertex_offset + i * segments + (j + 1) % segments + 1
                    v2 = vertex_offset + (i + 1) * segments + (j + 1) % segments + 1
                    v3 = vertex_offset + (i + 1) * segments + j + 1
                    f.write(f"f {v0} {v1} {v2} {v3}\n")
                    total_faces += 1
            
            vertex_offset += n_spline_pts * segments
            f.write("\n")
    
    print(f"*** SAVED: {mesh_filepath} ({total_verts} vertices, {total_faces} faces) ***")

def rebuild_visuals():
    """Rebuild splines and meshes from current control points with varying radii."""
    # Remove old actors
    for actor in state.spline_actors + state.mesh_actors:
        state.plotter.remove(actor)
    state.spline_actors.clear()
    state.mesh_actors.clear()
    
    # Rebuild splines and meshes
    for r, row_pts in enumerate(state.all_control_points):
        row_color = YARN_COLORS[r % len(YARN_COLORS)]
        row_radii = state.all_radii[r]
        
        # Create smooth spline
        row_spline = Spline(row_pts, res=800)
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
        state.spline_actors.append(spline_line)
        state.plotter.add(spline_line)
        
        # Mesh tube with varying radius
        row_mesh = Tube(row_spline.vertices, r=radii_interp, res=8)
        row_mesh.color(row_color).alpha(0.6).lighting("plastic")
        state.mesh_actors.append(row_mesh)
        state.plotter.add(row_mesh)
    
    state.plotter.render()

def update_point_colors():
    """Update point sphere colors and sizes based on selection and mode."""
    flat_idx = 0
    for r, row_pts in enumerate(state.all_control_points):
        for i in range(len(row_pts)):
            sphere = state.point_spheres[flat_idx]
            if state.selected_row == r and state.selected_point_idx == i:
                sphere.color("lime").alpha(1.0)
            elif state.edit_mode == 'radius':
                # In radius mode, color spheres by their radius (orange tint)
                sphere.color("orange").alpha(0.9)
            else:
                sphere.color("white").alpha(0.8)
            flat_idx += 1

def update_mode_display():
    """Update the mode indicator text."""
    if state.mode_text:
        state.plotter.remove(state.mode_text)
    
    if state.edit_mode == 'position':
        mode_str = "MODE: POSITION (press M to switch to RADIUS)"
        color = "cyan"
    else:
        mode_str = "MODE: RADIUS (press M to switch to POSITION)"
        color = "orange"
    
    state.mode_text = Text2D(mode_str, pos='bottom-center', c=color, s=1.0, bold=True)
    state.plotter.add(state.mode_text)
    state.plotter.render()

def on_left_click(evt):
    """Handle point selection."""
    if evt.picked3d is None:
        return
    
    click_pos = np.array(evt.picked3d)
    
    # Find closest control point sphere to click position
    min_dist = float('inf')
    best_flat_idx = None
    
    for flat_idx, sphere in enumerate(state.point_spheres):
        sphere_pos = np.array(sphere.pos())
        dist = np.linalg.norm(click_pos - sphere_pos)
        if dist < min_dist:
            min_dist = dist
            best_flat_idx = flat_idx
    
    # Only select if click is close enough to a point (within 0.15 units)
    if min_dist < 0.15 and best_flat_idx is not None:
        # Find which row and point index this is
        cumsum = 0
        for r, row_pts in enumerate(state.all_control_points):
            if best_flat_idx < cumsum + len(row_pts):
                state.selected_row = r
                state.selected_point_idx = best_flat_idx - cumsum
                print(f"Selected: Row {r}, Point {state.selected_point_idx}")
                update_point_colors()
                state.plotter.render()
                return
            cumsum += len(row_pts)

def on_key_press(evt):
    """Handle keyboard input for moving selected point or changing radius."""
    key = evt.keypress.lower()
    
    # P to save works anytime (no selection needed)
    if key == 'p':
        save_control_points()
        return
    
    # M to toggle mode between position and radius
    if key == 'm':
        if state.edit_mode == 'position':
            state.edit_mode = 'radius'
            print("=== RADIUS MODE === Click a point, then W=bigger, S=smaller")
        else:
            state.edit_mode = 'position'
            print("=== POSITION MODE === Drag points or use WASD to move")
        update_mode_display()
        update_point_colors()
        return
    
    # Other keys require a selected point
    if state.selected_row is None or state.selected_point_idx is None:
        if state.edit_mode == 'radius':
            print("*** No point selected! Click a point first, then use W/S to change radius ***")
        return
    
    r = state.selected_row
    i = state.selected_point_idx
    
    if state.edit_mode == 'radius':
        # Radius editing mode
        radius_delta = 0.01
        current_radius = state.all_radii[r][i]
        
        changed = False
        if key in ['w', 'up', 'equal', 'plus']:  # Increase radius
            state.all_radii[r][i] = min(0.3, current_radius + radius_delta)
            changed = True
            print(f">>> RADIUS INCREASED at Row {r}, Point {i}: {state.all_radii[r][i]:.3f}")
        elif key in ['s', 'down', 'minus']:  # Decrease radius
            state.all_radii[r][i] = max(0.01, current_radius - radius_delta)
            changed = True
            print(f">>> RADIUS DECREASED at Row {r}, Point {i}: {state.all_radii[r][i]:.3f}")
        else:
            print(f"(In radius mode, press W to increase or S to decrease radius)")
        
        if changed:
            rebuild_visuals()
    else:
        # Position editing mode
        delta = 0.05  # Movement amount
        pt = state.all_control_points[r][i].copy()
        
        moved = False
        if key == 'w':  # Move up (Y+)
            pt[1] += delta
            moved = True
        elif key == 's':  # Move down (Y-)
            pt[1] -= delta
            moved = True
        elif key == 'a':  # Move left (X-)
            pt[0] -= delta
            moved = True
        elif key == 'd':  # Move right (X+)
            pt[0] += delta
            moved = True
        elif key == 'r':  # Move forward (Z+)
            pt[2] += delta
            moved = True
        elif key == 'f':  # Move backward (Z-)
            pt[2] -= delta
            moved = True
        
        if moved:
            # Update control point
            state.all_control_points[r][i] = pt
            
            # Update sphere position
            flat_idx = sum(len(state.all_control_points[rr]) for rr in range(r)) + i
            state.point_spheres[flat_idx].pos(pt)
            
            # Rebuild splines and meshes
            rebuild_visuals()
            print(f"Moved point to ({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f})")

# 4. Interactive Editor
state.all_control_points = all_control_points

# Initialize radii for each control point (default value)
state.all_radii = [np.full(len(pts), state.default_radius) for pts in all_control_points]

# Create plotter
state.plotter = Plotter(bg='blackboard', axes=1, title="Knitting Editor - M to toggle Position/Radius mode")

# Create clickable spheres for each control point
for r, row_pts in enumerate(state.all_control_points):
    for i, pt in enumerate(row_pts):
        sphere = Sphere(pt, r=0.06).color("white").alpha(0.8)
        sphere.pickable(True)
        state.point_spheres.append(sphere)
        state.plotter.add(sphere)

# Initial splines and meshes
for r, row_pts in enumerate(state.all_control_points):
    row_color = YARN_COLORS[r % len(YARN_COLORS)]
    row_spline = Spline(row_pts, res=800)
    
    spline_line = Line(row_spline.vertices).color(row_color).linewidth(3)
    state.spline_actors.append(spline_line)
    state.plotter.add(spline_line)
    
    row_mesh = Tube(row_spline.vertices, r=0.08, res=8)
    row_mesh.color(row_color).alpha(0.6).lighting("plastic")
    state.mesh_actors.append(row_mesh)
    state.plotter.add(row_mesh)

# Instructions
txt = Text2D(
    "Controls:\n"
    "  Click point to select (GREEN)\n"
    "  Drag to move point\n"
    "  DOUBLE-CLICK to confirm & pick another\n"
    "  M: Toggle POSITION/RADIUS mode\n"
    "  W/S: Up/Down (or +/- radius)\n"
    "  A/D: Left/Right | R/F: Forward/Back\n"
    "  P: Save OBJ files",
    pos='top-left', c='white', s=0.8
)
state.plotter.add(txt)

# Add mode indicator
update_mode_display()

# Mouse drag handlers
def on_mouse_move(evt):
    """Handle mouse drag for moving selected point (position mode only)."""
    # Only allow dragging in position mode
    if state.edit_mode != 'position':
        return
    if not state.dragging or state.selected_row is None:
        return
    if evt.picked3d is None:
        return
    
    new_pos = np.array(evt.picked3d)
    r = state.selected_row
    i = state.selected_point_idx
    
    # Update control point position
    state.all_control_points[r][i] = new_pos
    
    # Update sphere position
    flat_idx = sum(len(state.all_control_points[rr]) for rr in range(r)) + i
    state.point_spheres[flat_idx].pos(new_pos)
    
    # Rebuild visuals
    rebuild_visuals()

def on_left_button_release(evt):
    """Stop dragging on mouse release (but keep point selected)."""
    if state.dragging:
        state.dragging = False

import time
last_click_time = [0.0]  # Use list to allow mutation in closure

def on_left_click_with_drag(evt):
    """Handle click: select point, or double-click to deselect."""
    current_time = time.time()
    double_click = (current_time - last_click_time[0]) < 0.3  # 300ms threshold
    last_click_time[0] = current_time
    
    if double_click and state.editing_locked:
        # Double-click: confirm point and allow picking another
        if state.selected_row is not None:
            r = state.selected_row
            i = state.selected_point_idx
            pt = state.all_control_points[r][i]
            print(f"Confirmed point at ({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f})")
        
        state.editing_locked = False
        state.selected_row = None
        state.selected_point_idx = None
        state.dragging = False
        update_point_colors()
        state.plotter.render()
        return
    
    if state.editing_locked:
        # Already editing a point - start dragging only in position mode
        if state.edit_mode == 'position':
            state.dragging = True
        return
    
    # Try to select a new point
    if evt.picked3d is None:
        return
    
    click_pos = np.array(evt.picked3d)
    
    # Find closest control point sphere to click position
    min_dist = float('inf')
    best_flat_idx = None
    
    for flat_idx, sphere in enumerate(state.point_spheres):
        sphere_pos = np.array(sphere.pos())
        dist = np.linalg.norm(click_pos - sphere_pos)
        if dist < min_dist:
            min_dist = dist
            best_flat_idx = flat_idx
    
    # Only select if click is close enough to a point (within 0.15 units)
    if min_dist < 0.15 and best_flat_idx is not None:
        # Find which row and point index this is
        cumsum = 0
        for r, row_pts in enumerate(state.all_control_points):
            if best_flat_idx < cumsum + len(row_pts):
                state.selected_row = r
                state.selected_point_idx = best_flat_idx - cumsum
                state.editing_locked = True
                # Only start dragging in position mode
                state.dragging = (state.edit_mode == 'position')
                if state.edit_mode == 'radius':
                    print(f"Selected: Row {r}, Point {state.selected_point_idx} - use W/S to change radius")
                else:
                    print(f"Selected: Row {r}, Point {state.selected_point_idx} (double-click to confirm)")
                update_point_colors()
                state.plotter.render()
                return
            cumsum += len(row_pts)

# Add callbacks
state.plotter.add_callback("LeftButtonPress", on_left_click_with_drag)
state.plotter.add_callback("MouseMove", on_mouse_move)
state.plotter.add_callback("LeftButtonRelease", on_left_button_release)
state.plotter.add_callback("KeyPress", on_key_press)

save_control_points()
state.plotter.show(interactive=True)