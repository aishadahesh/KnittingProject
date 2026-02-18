import os
import numpy as np
import base64
from io import BytesIO
from PIL import Image

import mitsuba as mi
import drjit as dr
import jax.numpy as jnp
import jax
import optax

# Trame imports
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vuetify3, vtk as vtk_widgets
from trame_server.controller import FunctionNotImplementedError

# VTK imports
import vtk
from vtk.util import numpy_support

# Import core reconstruction logic from the shared core module
from knitting_core import (
    CONFIG,
    compute_knitting_vertices,
    compute_knitting_vertices_jit,
    compute_knitting_faces,
    get_loop_color,
    save_combined_obj,
    KnittingOptimizer,
    run_optimization_loop
)

# %% SERVER SETUP
server = get_server()
state, ctrl = server.state, server.controller

# %% GLOBAL CONTEXT
REFERENCE_PATH = CONFIG['ui']['reference_image']
try:
    with open(REFERENCE_PATH, "rb") as f:
        ref_data = f.read()
        ref_base64 = f"data:image/png;base64,{base64.b64encode(ref_data).decode()}"
    ref_pil = Image.open(REFERENCE_PATH).convert("RGB")
except Exception as e:
    print(f"Error loading reference image: {e}")
    ref_base64 = ""
    ref_pil = Image.new('RGB', (64, 64), (255, 255, 255))

BITMAP = jnp.ones((CONFIG['geometry']['bitmap_rows'], CONFIG['geometry']['bitmap_loops']))
OPTIMIZER = KnittingOptimizer(ref_pil, BITMAP)

# %% STATE INITIALIZATION
PARAM_NAMES = CONFIG['geometry']['param_names']
init_vals = CONFIG['geometry']['initial_params']
for i, val in enumerate(init_vals):
    state[f"p{i}"] = float(val)

state.mode = 'parameter' 
state.render_base64 = ""
state.ref_base64 = ref_base64
state.optimizing = False

# %% VTK SCENE SETUP
renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

# Disable default VTK interactor bindings
render_window_interactor.RemoveObservers('CharEvent')
render_window_interactor.RemoveObservers('KeyPressEvent')

# Set interactor style for 3D rotation
style = vtk.vtkInteractorStyleTrackballCamera()
render_window_interactor.SetInteractorStyle(style)
render_window_interactor.Initialize()

# Scene storage
mesh_actors = [] # List of (actor, polydata)
handle_widgets = [] # List of vtkHandleWidgets
ctrl_pts = [] # List of numpy arrays (one per row)

# %% UTILITIES

def safe_view_update():
    """Forces VTK to render and notifies Trame to sync the view."""
    try:
        render_window.Render()
        ctrl.view_update()
    except FunctionNotImplementedError:
        pass
    except Exception as e:
        print(f"View Update Error: {e}")

def get_current_params():
    """Constructs the parameter list from individual state variables."""
    return [state[f"p{i}"] for i in range(len(PARAM_NAMES))]

def update_vtk_meshes():
    """Synchronizes VTK actors with current global parameters."""
    global mesh_actors
    res, seg = CONFIG['geometry']['loop_res'], CONFIG['geometry']['segments']
    params = get_current_params()
    
    try:
        verts_list = compute_knitting_vertices(params, BITMAP)
        
        if not mesh_actors:
            faces_list = compute_knitting_faces(seg, verts_list)
            for i, (verts, faces) in enumerate(zip(verts_list, faces_list)):
                polydata = vtk.vtkPolyData()
                points = vtk.vtkPoints()
                points.SetData(numpy_support.numpy_to_vtk(np.array(verts), deep=True))
                polydata.SetPoints(points)
                
                f_np = np.array(faces)
                tris = np.empty((len(f_np) * 2, 3), dtype=np.int32)
                tris[0::2], tris[1::2] = f_np[:, [0, 1, 2]], f_np[:, [0, 2, 3]]
                cells = np.column_stack([np.full(len(tris), 3), tris])
                vtk_cells = vtk.vtkCellArray()
                vtk_cells.ImportLegacyFormat(numpy_support.numpy_to_vtkIdTypeArray(cells, deep=True))
                polydata.SetPolys(vtk_cells)
                
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(polydata)
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(get_loop_color(i, 0))
                renderer.AddActor(actor)
                mesh_actors.append((actor, polydata))
            renderer.ResetCamera()
        else:
            for i, (verts, _) in enumerate(verts_list):
                polydata = mesh_actors[i][1]
                polydata.GetPoints().SetData(numpy_support.numpy_to_vtk(np.array(verts), deep=True))
                polydata.Modified()
        
        safe_view_update()
    except Exception as e:
        print(f"Error updating meshes: {e}")

def update_mesh_from_splines():
    """Real-time mesh update based on manual control point positions."""
    global ctrl_pts
    res, seg = CONFIG['geometry']['loop_res'], CONFIG['geometry']['segments']
    radius, ratio = state.p4, state.p9
    
    for r, row_pts in enumerate(ctrl_pts):
        spline_x, spline_y, spline_z = vtk.vtkCardinalSpline(), vtk.vtkCardinalSpline(), vtk.vtkCardinalSpline()
        for i, p in enumerate(row_pts):
            spline_x.AddPoint(i, p[0]); spline_y.AddPoint(i, p[1]); spline_z.AddPoint(i, p[2])
            
        n_points = res * BITMAP.shape[1] + 1
        interpolated_pts = []
        for i in range(n_points):
            t = (len(row_pts) - 1) * i / (n_points - 1)
            interpolated_pts.append([spline_x.Evaluate(t), spline_y.Evaluate(t), spline_z.Evaluate(t)])
        
        pts = np.array(interpolated_pts)
        T = np.gradient(pts, axis=0)
        T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-8)
        U = np.cross(T, [0,0,1])
        mask = np.linalg.norm(U, axis=1) < 1e-6
        U[mask] = np.cross(T[mask], [1,0,0])
        U /= (np.linalg.norm(U, axis=1, keepdims=True) + 1e-8)
        V = np.cross(T, U)
        angles = np.linspace(0, 2*np.pi, seg, endpoint=False)
        offsets = U[:,None,:] * np.cos(angles)[None,:,None] * radius * ratio + V[:,None,:] * np.sin(angles)[None,:,None] * radius
        
        mesh_actors[r][1].GetPoints().SetData(numpy_support.numpy_to_vtk((pts[:,None,:] + offsets).reshape(-1, 3), deep=True))
        mesh_actors[r][1].Modified()
    
    safe_view_update()

# %% INTERACTION HANDLERS

def on_handle_interaction(obj, event):
    """Called whenever a spline handle is moved."""
    global ctrl_pts
    for flat_idx, hw in enumerate(handle_widgets):
        pos = hw.GetRepresentation().GetWorldPosition()
        cumsum = 0
        for r, row in enumerate(ctrl_pts):
            if flat_idx < cumsum + len(row):
                ctrl_pts[r][flat_idx - cumsum] = pos
                break
            cumsum += len(row)
    update_mesh_from_splines()

def generate_initial_ctrl_pts():
    bulge, z, _, dy, _ = get_current_params()[:5]
    n_r, n_c = BITMAP.shape
    rows = []
    for r in range(n_r):
        pts = []
        cols = range(n_c) if r % 2 == 0 else range(n_c - 1, -1, -1)
        for c in cols:
            t_v = np.linspace(0, 2*np.pi, 5, endpoint=False)
            if r % 2 != 0: t_v = t_v[::-1]
            for t in t_v:
                pts.append([c + bulge*np.sin(2*t) + t/(2*np.pi), r*dy - (np.cos(t)-1)/2, z*(np.cos(2*t)-1)/2])
        pts.append([n_c if r % 2 == 0 else 0, r*dy, 0])
        rows.append(np.array(pts))
    return rows

# %% REACTIVE CALLBACKS

@state.change("mode")
def on_mode_change(mode, **kwargs):
    global handle_widgets, ctrl_pts
    print(f"UI Mode -> {mode}")
    if mode == 'spline':
        ctrl_pts = generate_initial_ctrl_pts()
        for row in ctrl_pts:
            for p in row:
                hw = vtk.vtkHandleWidget()
                hw.SetInteractor(render_window_interactor)
                rep = vtk.vtkPointHandleRepresentation3D()
                rep.SetWorldPosition(p)
                rep.GetProperty().SetColor(1, 1, 1)
                rep.SetHandleSize(10)
                hw.SetRepresentation(rep)
                hw.AddObserver("InteractionEvent", on_handle_interaction)
                hw.On()
                handle_widgets.append(hw)
    else:
        if ctrl_pts:
            flat_pts = np.concatenate(ctrl_pts)
            state.p3 = float(np.mean(np.diff([np.mean(row[:,1]) for row in ctrl_pts]))) if len(ctrl_pts)>1 else state.p3
            z_range = np.max(flat_pts[:,2]) - np.min(flat_pts[:,2])
            state.p1 = float(-z_range / 2) if z_range > 0 else state.p1
        for h in handle_widgets: h.Off()
        handle_widgets.clear()
    safe_view_update()

@state.change(*(f"p{i}" for i in range(len(init_vals))))
def on_params_change(**kwargs):
    if state.mode == 'parameter':
        update_vtk_meshes()

# %% ACTIONS

def trigger_render():
    print("Action: Render")
    params = get_current_params()
    try:
        res, seg = CONFIG['geometry']['loop_res'], CONFIG['geometry']['segments']
        verts_list = compute_knitting_vertices(params, BITMAP)
        faces_list = compute_knitting_faces(seg, verts_list)
        mesh_data = [(v, [], f, n) for (v, n), f in zip(verts_list, faces_list)]
        path = os.path.join(CONFIG['rendering']['output_dir'], "meshes", "trame_preview")
        save_combined_obj(mesh_data, path)
        scene_dict = OPTIMIZER.get_scene_dict(path + "_combined.obj", params)
        img = mi.render(mi.load_dict(scene_dict), spp=128)
        pil_img = Image.fromarray((np.clip(np.array(img), 0, 1) * 255).astype(np.uint8))
        buffer = BytesIO()
        pil_img.save(buffer, format="PNG")
        state.render_base64 = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
    except Exception as e: print(f"Render Error: {e}")

def trigger_optimization():
    print("Action: Optimize")
    state.optimizing = True
    try:
        new_params, _ = run_optimization_loop(OPTIMIZER, get_current_params())
        for i, val in enumerate(new_params): state[f"p{i}"] = float(val)
        update_vtk_meshes()
    except Exception as e: print(f"Optimization Error: {e}")
    finally: state.optimizing = False

# %% UI LAYOUT
with SinglePageWithDrawerLayout(server) as layout:
    layout.title.set_text("Knitting Reconstruction - Trame")
    layout.drawer.width = 300
    
    with layout.toolbar:
        vuetify3.VSpacer()
        vuetify3.VBtn(click=trigger_render, color="primary", icon="mdi-camera")
        vuetify3.VBtn(click=trigger_optimization, color="success", loading=("optimizing", False), icon="mdi-auto-fix")
        vuetify3.VBtn(click=lambda: ctrl.view_reset_camera(), icon="mdi-crop-free")
        with vuetify3.VBtnToggle(v_model=("mode", "parameter"), mandatory=True, color="indigo"):
            vuetify3.VBtn("Param", value="parameter")
            vuetify3.VBtn("Spline", value="spline")

    with layout.content:
        with vuetify3.VContainer(fluid=True, classes="pa-0 fill-height"):
            with vuetify3.VRow(no_gutters=True, classes="fill-height"):
                with vuetify3.VCol(cols=4, classes="fill-height border-right"):
                    view = vtk_widgets.VtkRemoteView(render_window, ref="view")
                    ctrl.view_update = view.update
                    ctrl.view_reset_camera = view.reset_camera
                with vuetify3.VCol(cols=4, classes="fill-height border-right"):
                    with vuetify3.VContainer(classes="fill-height d-flex align-center justify-center pa-2"):
                        vuetify3.VImg(src=("render_base64", ""), max_height="100%", classes="elevation-2")
                with vuetify3.VCol(cols=4, classes="fill-height"):
                    with vuetify3.VContainer(classes="fill-height d-flex align-center justify-center pa-2"):
                        vuetify3.VImg(src=("ref_base64", ""), max_height="100%", classes="elevation-2")

    with layout.drawer:
        with vuetify3.VList():
            for i, name in enumerate(PARAM_NAMES):
                with vuetify3.VListItem():
                    vmin, vmax = CONFIG['geometry']['param_ranges'][i]
                    vuetify3.VSlider(label=name, v_model=(f"p{i}",), min=vmin, max=vmax,
                                     step=CONFIG['geometry']['param_deltas'][i], 
                                     hide_details=True, thumb_label="always", density="compact")

if __name__ == "__main__":
    update_vtk_meshes()
    server.start()
