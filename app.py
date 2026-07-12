"""guided_workflow_app.py — Knitting Reconstruction GUI (Modular Refactored Edition)

Stack:
  - imgui_bundle  : window + Dear ImGui UI (sliders, buttons, layout)
  - moderngl      : mesh rendering via OpenGL FBO → displayed as imgui image
  - scipy         : CubicSpline replaces vtk.vtkCardinalSpline
  - knitting_core : core JAX computation and mitsuba optimizer interface
"""

# %% PYOPENGL CONFIG FOR WAYLAND/LINUX
import os, sys
if sys.platform.startswith("linux"):
    os.environ["PYOPENGL_PLATFORM"] = "egl"

# %% IMPORTS
import numpy as np
import glfw
import moderngl
from PIL import Image
from imgui_bundle import imgui, imguizmo
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
import jax.numpy as jnp


import json

project_root = os.path.dirname(os.path.abspath(__file__))
resolve_project_path = lambda p: p if os.path.isabs(p) else os.path.join(project_root, p)

with open(os.path.join(project_root, "config.json"), "r") as f:
    config = json.load(f)

from rendering import Camera, MeshRenderer, pil_to_texture
from app_state import AppState
from gui import (
     draw_menu_bar,
     draw_sidebar,
     draw_viewport,
)

# %% MAIN ─────────────────────────────────────────────────────────────────────

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    # ── Reference image ──────────────────────────────────────────────────────
    try:
        ref_pil = Image.open(resolve_project_path(config["ui"]["reference_image"])).convert("RGB")
    except Exception:
        ref_pil = Image.new("RGB", (256, 256), (60, 40, 40))



    # ── GLFW + OpenGL ─────────────────────────────────────────────────────────
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    window = glfw.create_window(1600, 900, "Knitting Guided Workflow", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # ── Dear ImGui ────────────────────────────────────────────────────────────
    imgui.create_context()
    io = imgui.get_io()
    io.config_windows_move_from_title_bar_only = True
    io.config_flags |= imgui.ConfigFlags_.docking_enable
    io.config_flags |= imgui.ConfigFlags_.viewports_enable
    io.set_ini_filename(os.path.join(project_root, "imgui_layout.ini"))
    impl = GlfwRenderer(window)

    style = imgui.get_style()
    style.window_menu_button_position = imgui.Dir_.none
    if io.config_flags & imgui.ConfigFlags_.viewports_enable:
        style.window_rounding = 0.0
        style.color_(imgui.Col_.window_bg).w = 1.0

    # ── moderngl (shares the existing GL context) ─────────────────────────────
    ctx = moderngl.create_context()

    # ── Scene objects ─────────────────────────────────────────────────────────
    camera   = Camera()
    renderer = MeshRenderer(ctx, 960, 720)
    # Create meshes/renders output directories
    for d in ("meshes", "renders"):
        os.makedirs(os.path.join(resolve_project_path(config["rendering"]["output_dir"]), d), exist_ok=True)

    # ── App state ─────────────────────────────────────────────────────────────
    state = AppState(camera, renderer)
    state.reference_image_pixels = np.asarray(ref_pil, dtype=np.float32) / 255.0

    # initial_params.json is the frozen reset baseline.
    # params.json is the working autosave and is loaded for normal resume.
    work_params_path = state.load_path
    initial_params_path = os.path.join(project_root, "initial_params.json")

    if os.path.exists(initial_params_path):
        state.load_params(initial_params_path)
        if state.status_msg.startswith('Load error:'):
            state.rebuild_spline_from_params()
    else:
        state.rebuild_spline_from_params()
    state.capture_initial_state()

    if os.path.exists(work_params_path):
        state.load_params(work_params_path)
        if state.status_msg.startswith('Load error:'):
            state.restore_snapshot(state._initial_snapshot)
    state.load_path = work_params_path
    state.save_path = work_params_path

    # ── Static textures ───────────────────────────────────────────────────────
    ref_tex = pil_to_texture(ctx, ref_pil)

    # ── Main loop ─────────────────────────────────────────────────────────────
    undo_shortcut_was_down = False
    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()
        imguizmo.im_guizmo.begin_frame()

        ctrl_down = (
            glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS
        )
        z_down = glfw.get_key(window, glfw.KEY_Z) == glfw.PRESS
        undo_shortcut_down = ctrl_down and z_down
        if undo_shortcut_down and not undo_shortcut_was_down and state.undo_stack:
            state.undo_last()
        undo_shortcut_was_down = undo_shortcut_down

        # Camera movement is disabled in the viewport; only model transforms are interactive.



        win_w, win_h = glfw.get_window_size(window)

        viewport = imgui.get_main_viewport()
        imgui.set_next_window_pos(viewport.pos)
        imgui.set_next_window_size(viewport.size)
        imgui.set_next_window_viewport(viewport.id_)
        host_flags = (
            imgui.WindowFlags_.no_docking |
            imgui.WindowFlags_.no_title_bar |
            imgui.WindowFlags_.no_collapse |
            imgui.WindowFlags_.no_resize |
            imgui.WindowFlags_.no_move |
            imgui.WindowFlags_.no_bring_to_front_on_focus |
            imgui.WindowFlags_.no_nav_focus |
            imgui.WindowFlags_.menu_bar
        )
        dockspace_flags = imgui.DockNodeFlags_.passthru_central_node
        imgui.push_style_var(imgui.StyleVar_.window_rounding, 0.0)
        imgui.push_style_var(imgui.StyleVar_.window_border_size, 0.0)
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0.0, 0.0))
        imgui.begin("MainDockSpace", flags=host_flags)
        imgui.pop_style_var(3)

        draw_menu_bar(state)
        imgui.dock_space_over_viewport(flags=dockspace_flags)
        imgui.end()

        # ── Sidebar ───────────────────────────────────────────────────────────
        draw_sidebar(state, renderer, window)

        # MuJoCo offscreen rendering can make its own GL context current.
        # Restore the app context before ModernGL allocates/draws viewport buffers.
        glfw.make_context_current(window)
        ctx.screen.use()

        # ── 3D Viewport ───────────────────────────────────────────────────────
        draw_viewport(state, renderer, ref_tex, window)



        # ── Reference Image ───────────────────────────────────────────────────

        # ── Final GL clear + imgui draw ───────────────────────────────────────
        ctx.screen.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        imgui.render()
        impl.render(imgui.get_draw_data())

        if io.config_flags & imgui.ConfigFlags_.viewports_enable:
            backup_current_context = glfw.get_current_context()
            imgui.update_platform_windows()
            imgui.render_platform_windows_default()
            glfw.make_context_current(backup_current_context)

        glfw.swap_buffers(window)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    embedded = getattr(state, 'embedded_scanner', None)
    if embedded is not None:
        try:
            embedded.close()
        except Exception:
            pass
        state.embedded_scanner = None
    try:
        glfw.make_context_current(window)
        ctx.screen.use()
    except Exception:
        pass
    impl.shutdown()
    imgui.destroy_context()
    glfw.terminate()

if __name__ == "__main__":
    main()

# %%
