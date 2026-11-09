"""Knitted and crocheted geometry editor, scanner, and reconstruction viewer."""

import os
import sys
import logging
from contextlib import ExitStack
if sys.platform.startswith("linux"):
    os.environ["PYOPENGL_PLATFORM"] = "egl"

# %% MODULE PATH
# The application modules live in src/ while this entry point stays in the
# project root. Putting src/ on the path here lets them keep importing each
# other by plain name (`from app_state import AppState`) rather than every
# module carrying a package prefix.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# %% IMPORTS
import threading

import numpy as np
import glfw
import moderngl
from PIL import Image
from imgui_bundle import imgui, imguizmo
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer


import json

import paths

from rendering import Camera, MeshRenderer, pil_to_texture
from app_state import AppState
from scanner_storage import ScannerStorage
from gui import (
     draw_menu_bar,
     draw_orbit_viewport,
     draw_sidebar,
     draw_viewport,
)

# %% MAIN ─────────────────────────────────────────────────────────────────────

def main():
    with ExitStack() as resources:
        with open(paths.CONFIG_JSON, encoding="utf-8") as handle:
            config = json.load(handle)
        # ── Reference image ──────────────────────────────────────────────────────
        try:
            ref_pil = Image.open(paths.resolve(config["ui"]["reference_image"])).convert("RGB")
        except OSError as exc:
            logging.warning("Reference image: %s", exc)
            ref_pil = Image.new("RGB", (256, 256), (60, 40, 40))



        # ── GLFW + OpenGL ─────────────────────────────────────────────────────────
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        resources.callback(glfw.terminate)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        window = glfw.create_window(1600, 900, "Knitting Guided Workflow", None, None)
        if window is None:
            raise RuntimeError("GLFW window creation failed")
        resources.callback(glfw.destroy_window, window)
        glfw.make_context_current(window)
        glfw.swap_interval(1)

        # ── Dear ImGui ────────────────────────────────────────────────────────────
        imgui.create_context()
        resources.callback(imgui.destroy_context)
        io = imgui.get_io()
        io.config_windows_move_from_title_bar_only = True
        io.config_flags |= imgui.ConfigFlags_.docking_enable
        io.config_flags |= imgui.ConfigFlags_.viewports_enable
        io.set_ini_filename(str(paths.LAYOUT_INI))
        impl = GlfwRenderer(window)
        resources.callback(impl.shutdown)

        style = imgui.get_style()
        style.window_menu_button_position = imgui.Dir_.none
        if io.config_flags & imgui.ConfigFlags_.viewports_enable:
            style.window_rounding = 0.0
            style.color_(imgui.Col_.window_bg).w = 1.0

        # ── moderngl (shares the existing GL context) ─────────────────────────────
        ctx = moderngl.create_context()
        resources.callback(ctx.release)

        # ── Scene objects ─────────────────────────────────────────────────────────
        camera   = Camera()
        renderer = MeshRenderer(ctx, 960, 720)
        resources.callback(renderer.close)
        # Puzzle owns a separate framebuffer and mesh buffers. Scan can rebuild the
        # main viewport as often as it needs without ever touching Puzzle's scene.
        puzzle_renderer = MeshRenderer(ctx, 960, 720)
        resources.callback(puzzle_renderer.close)
        # Second view with its own camera and framebuffer, so orbiting it never
        # disturbs the main viewport's fixed framing.
        orbit_camera   = Camera()
        orbit_renderer = MeshRenderer(ctx, 960, 720)
        resources.callback(orbit_renderer.close)
        # ── App state ─────────────────────────────────────────────────────────────
        state = AppState(
            camera,
            renderer,
            orbit_camera=orbit_camera,
            orbit_renderer=orbit_renderer,
            puzzle_renderer=puzzle_renderer,
        )
        resources.callback(glfw.make_context_current, window)

        def close_controllers():
            for key in ('embedded_scanner', 'ur5_controller'):
                controller = state.get(key)
                if controller is not None:
                    try:
                        controller.close()
                        state[key] = None
                    except Exception:
                        logging.exception("Closing %s", key)

        resources.callback(close_controllers)
        state.reference_image_pixels = np.asarray(ref_pil, dtype=np.float32) / 255.0

        # initial_params.json is the frozen reset baseline.
        # params.json is the working autosave and is loaded for normal resume.
        work_params_path = state.load_path
        initial_params_path = str(paths.INITIAL_PARAMS_JSON)

        if os.path.exists(initial_params_path):
            if not state.load_params(initial_params_path):
                state.rebuild_spline_from_params()
        else:
            state.rebuild_spline_from_params()
        state.capture_initial_state()

        if os.path.exists(work_params_path):
            if not state.load_params(work_params_path, restore_mode=True):
                state.restore_snapshot(state._initial_snapshot)
        state.load_path = work_params_path
        state.save_path = work_params_path
        scanner_storage = ScannerStorage(paths.PROJECT_ROOT)
        object.__setattr__(state, "scanner_storage", scanner_storage)
        if scanner_storage.restore_scanner_state(state):
            state.scanner_status = "Restored previous scanner database state"
            if str(state.get("app_mode", "edit")) == "scan":
                state.scanner_preview_grid_enabled = True
                state.scanner_preview_rows = max(1, int(state.scanner_rows))
                state.scanner_preview_cols = max(1, int(state.scanner_cols))
                state.rebuild_spline_mesh(preserve_model_placement=False)

        # ── Background simulation thread ──────────────────────────────────────────
        def sim_may_run(state):
            """The solver rewrites ctrl_rows, which Scan, Puzzle and Database all
            snapshot and replace with their own temporary geometry. Restricting it
            to Edit Mode keeps the two from writing to the same state."""
            return bool(state.sim_active) and str(state.get('app_mode', 'edit')) == 'edit'

        def start_simulation_thread(state):
            from yarn_simulation import run_simulation_step, eval_energy

            stop = threading.Event()

            def run_loop():
                try:
                    while not stop.is_set():
                        if not sim_may_run(state):
                            stop.wait(0.05)
                            continue

                        with state.sim_lock:
                            if state.sim_needs_jacobian_rebuild:
                                state.rebuild_cached_jacobian()
                            if state.J_cached is None:
                                stop.wait(0.01)
                                continue
                            ctrl_rows = [cp.copy() for cp in state.ctrl_rows]
                            period_offset_x = state.period_offset_x.copy()
                            period_offset_y = state.period_offset_y.copy()
                            config = state.config
                            J_cached = state.J_cached
                            L0_array = state.sim_L0
                            ks, kb = state.sim_k_s, state.sim_k_b
                            kc, dhat = state.sim_k_c, state.sim_dhat

                        if L0_array is None or len(L0_array) == 0:
                            stop.wait(0.01)
                            continue

                        # Solved outside the lock: a step takes ~0.15s and the UI thread
                        # must not block on it.
                        new_ctrl_rows = run_simulation_step(
                            ctrl_rows, period_offset_x, period_offset_y, config,
                            J_cached, L0_array, ks, kb, kc, dhat,
                        )

                        if stop.is_set():
                            return

                        flat_P_old = np.concatenate(ctrl_rows).astype(float) if ctrl_rows else np.empty((0, 3), float)
                        flat_P = np.concatenate(new_ctrl_rows).astype(float) if new_ctrl_rows else np.empty((0, 3), float)
                        if len(flat_P) > 0:
                            e_el, e_b, e_col = eval_energy(
                                flat_P, new_ctrl_rows, period_offset_x, period_offset_y,
                                config, L0_array, dhat,
                            )
                        else:
                            e_el, e_b, e_col = 0.0, 0.0, 0.0
                        delta_P = flat_P - flat_P_old if len(flat_P_old) == len(flat_P) and len(flat_P) else None

                        with state.sim_lock:
                            state.sim_e_el, state.sim_e_b, state.sim_e_col = float(e_el), float(e_b), float(e_col)
                            if delta_P is not None:
                                state.sim_delta_P = delta_P.copy()
                            # Re-checked under the lock: the mode may have changed while
                            # this step was solving, and writing scan-time geometry back
                            # from a stale solve would corrupt the capture.
                            if not stop.is_set() and sim_may_run(state) and not state.sim_needs_jacobian_rebuild:
                                state.ctrl_rows = new_ctrl_rows
                                state.flat_pts = (
                                    np.concatenate(new_ctrl_rows).astype(np.float32)
                                    if new_ctrl_rows else np.empty((0, 3), np.float32)
                                )
                                state.sim_mesh_dirty = True
                        stop.wait(0.01)

                except Exception as exc:
                    with state.sim_lock:
                        state.sim_active = False
                        state.status_msg = f'Simulation error: {exc}'
                    logging.exception("Yarn simulation stopped")

            thread = threading.Thread(target=run_loop, daemon=True, name="yarn-sim")
            thread.start()

            def close():
                stop.set()
                thread.join(timeout=5.0)
                if thread.is_alive():
                    logging.warning("Yarn solver still finishing after shutdown")

            return close

        resources.callback(start_simulation_thread(state))

        # ── Static textures ───────────────────────────────────────────────────────
        ref_tex = pil_to_texture(ctx, ref_pil)
        resources.callback(ref_tex.release)
        resources.callback(glfw.make_context_current, window)

        # ── Frame pacing ──────────────────────────────────────────────────────────
        # The loop used to redraw flat out whether anything changed or not. Now it
        # waits for input, up to IDLE_FRAME_SECONDS, unless something on screen
        # moves by itself every frame - Scan Mode's simulator stepping, the yarn
        # solver, or Model Mode's spinning model - and any mouse or key event still
        # wakes it at once.
        IDLE_FRAME_SECONDS = 1.0 / 15.0

        def animating(state):
            embedded = state.get('embedded_scanner')
            scanning = embedded is not None and (
                bool(getattr(embedded, 'running', False))
                or bool(getattr(embedded, 'single_target_active', False))
                or getattr(embedded, '_capture_job', None) is not None
            )
            model_view = state.get('model_view')
            spinning = model_view is not None and (bool(getattr(model_view, 'spin', False))
                                                   or bool(getattr(model_view, 'wiggle', False)))
            return scanning or spinning or sim_may_run(state)

        # ── Main loop ─────────────────────────────────────────────────────────────
        undo_shortcut_was_down = False
        while not glfw.window_should_close(window):
            if animating(state):
                glfw.poll_events()
            else:
                glfw.wait_events_timeout(IDLE_FRAME_SECONDS)
            impl.process_inputs()
            # Upload whatever the solver produced since the last frame. GL work has
            # to happen on this thread, so the solver only flags that it moved.
            with state.sim_lock:
                if state.get('sim_mesh_dirty', False):
                    state.sim_mesh_dirty = False
                    if sim_may_run(state):
                        state.rebuild_spline_mesh(preserve_model_placement=True)
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
            draw_sidebar(state, state.active_scene_renderer(), window)

            # MuJoCo offscreen rendering can make its own GL context current.
            # Restore the app context before ModernGL allocates/draws viewport buffers.
            glfw.make_context_current(window)
            ctx.screen.use()

            # ── 3D Viewport ───────────────────────────────────────────────────────
            # UR5 Robot Mode only draws the knit preview when asked: the arm and the
            # cell are TeleopApp's to show, and re-rendering the tiled scan layout
            # every frame beside it was the heaviest thing the app did there.
            mode_now = str(state.get("app_mode", "edit"))
            if mode_now not in ("database", "model") and (mode_now != "ur5" or bool(state.get("ur5_show_scene", False))):
                # Re-evaluate after draw_sidebar: its mode buttons can switch into
                # or out of Puzzle during this same frame.
                draw_viewport(state, state.active_scene_renderer(), ref_tex, window)
                # Edit Mode only: the other modes drive the shared model/camera
                # themselves, and a second live view of it would fight them.
                if str(state.get('app_mode', 'edit')) == 'edit':
                    draw_orbit_viewport(state, window)

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

if __name__ == "__main__":
    main()

# %%
