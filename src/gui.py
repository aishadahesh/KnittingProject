"""imgui panels for the knitting app: menu bar, sidebar, and 3D viewport.

Module layout (dependencies point one way, no cycles):

    gui.py             this file -- imgui panels and per-mode UI
      |- scanner_core      scanner/puzzle logic, no imgui
      |- gui_ur5           UR5 Robot Mode UI           -> scanner_core
      `- gui_model         Model Mode UI               -> strand_equations

`app.py` imports draw_menu_bar / draw_sidebar / draw_viewport /
draw_orbit_viewport from here. Names moved into scanner_core kept their original
spelling and are re-imported below, so call sites in this file are unchanged.
"""
import logging
import os
import json
import numpy as np
import glfw
import time
import threading
from types import SimpleNamespace
from pathlib import Path

import tkinter as tk
from tkinter import filedialog as _filedialog
from imgui_bundle import imgui, imguizmo

from rendering import (draw_fitted_texture, project_to_screen, transform_points,
                       upload_rgb_texture)
from fabric_scanner import clamp_capture_size
import paths
from embedded_scanner import EmbeddedMujocoScanner
from gui_database import _draw_database_summary_page, draw_database_viewport
# Moved to scanner_core (see its docstring). Re-imported under their original
# names so every call site in this file is unchanged.
from scanner_core import (
    _clamp_scanner_selected_cell, _ensure_scanner_shared_colors,
    _maybe_persist_scanner_state,
    _puzzle_apply_geometry_copies,
    _puzzle_build_seamless_tile, _puzzle_capture_geometry_image, _puzzle_capture_rect,
    _puzzle_detect_colors, _puzzle_target_copies,
    _refresh_embedded_scanner_display_colors, _scanner_batch_texture_size, _scanner_capture_image_size,
    _scanner_display_batch_colors, _scanner_estimated_cell_colors,
    _scanner_generate_tiled_layout, _scanner_material_uniforms, _scanner_pattern_dimensions, _scanner_pattern_repeats, _scanner_repeat_spacing,
    _scanner_storage, _set_puzzle_capture_rect,
    _set_scanner_estimated_batch_color, _upload_puzzle_texture, build_scanner_plan_args,
)

# UR5 Robot Mode's panel. Safe to import here: gui_ur5 imports gui only from
# inside the functions that need it, so there is no cycle at load time.
import gui_ur5
import gui_model
import crochet_core
import scan_fit

# Per-step solver displacements are tiny next to the model, so the force overlay
# exaggerates them to stay visible. Display only -- never fed back into state.
# ============================================================================
# EmbeddedMujocoScanner: the MuJoCo scan runtime
# ============================================================================

FORCE_ARROW_SCALE = 5.0


# %% FILE PICKER HELPERS ───────────────────────────────────────────────────────

def _pick_file(mode, initial_path):
    """Opens a native file dialog on the main thread; returns path string or ''."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    if mode == 'save':
        path = _filedialog.asksaveasfilename(
            parent=root,
            title='Save parameters',
            defaultextension='.json',
            filetypes=[('JSON files', '*.json'), ('All files', '*.*')],
            initialfile=os.path.basename(initial_path),
            initialdir=os.path.dirname(initial_path),
        )
    else:
        path = _filedialog.askopenfilename(
            parent=root,
            title='Load parameters',
            filetypes=[('JSON files', '*.json'), ('All files', '*.*')],
            initialdir=os.path.dirname(initial_path),
        )
    root.destroy()
    return path or ''


# ============================================================================
# Scanner UI Section: Model, Palette, and Layout Helpers
# ============================================================================


def _draw_scanner_batch_color_grid(state, title=None):
    display_colors, stage = _scanner_display_batch_colors(state)
    if not display_colors:
        return
    rows = max(1, int(state.scanner_rows))
    cols = max(1, int(state.scanner_cols))
    estimates = {
        (int(item["row"]), int(item["col"])): item
        for item in _scanner_estimated_cell_colors(state)
    }
    selected = np.asarray(state.get('scanner_selected_cell', [0, 0]), dtype=np.int32).reshape(-1)
    selected_r = int(np.clip(selected[0] if selected.size > 0 else 0, 0, rows - 1))
    selected_c = int(np.clip(selected[1] if selected.size > 1 else 0, 0, cols - 1))

    if title is None:
        title = "Actual analyzed colors" if stage == "actual" else "Estimated color before scan"
    imgui.text(title)
    if stage == "actual":
        imgui.text_disabled("Updated from captured fabric RGB analysis.")
    else:
        imgui.text_disabled("Predicted from random bitmap visibility and shared colors.")
    cell_size = max(18.0, min(34.0, (imgui.get_content_region_avail().x - max(0, cols - 1) * 3.0) / max(cols, 1)))
    imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(3, 3))
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            rgb = [float(v) for v in (display_colors[idx] if idx < len(display_colors) else [42.0, 42.0, 42.0])]
            rgba = (
                np.clip(rgb[0] / 255.0, 0.0, 1.0),
                np.clip(rgb[1] / 255.0, 0.0, 1.0),
                np.clip(rgb[2] / 255.0, 0.0, 1.0),
                1.0,
            )
            imgui.push_style_color(imgui.Col_.button, rgba)
            imgui.push_style_color(imgui.Col_.button_hovered, (
                min(float(rgba[0]) + 0.12, 1.0),
                min(float(rgba[1]) + 0.12, 1.0),
                min(float(rgba[2]) + 0.12, 1.0),
                1.0,
            ))
            selected_cell = r == selected_r and c == selected_c
            if selected_cell:
                imgui.push_style_color(imgui.Col_.border, (1.0, 0.78, 0.15, 1.0))
                imgui.push_style_var(imgui.StyleVar_.frame_border_size, 2.0)
            clicked = imgui.button(f"##estimated_color_{r}_{c}", imgui.ImVec2(cell_size, cell_size))
            if selected_cell:
                imgui.pop_style_var()
                imgui.pop_style_color()
            imgui.pop_style_color(2)
            if clicked:
                state.scanner_selected_cell = [r, c]
                state.scanner_analysis_selected_cell = [r, c]
            if c < cols - 1:
                imgui.same_line()
        if r < rows - 1:
            imgui.spacing()
    imgui.pop_style_var()

    idx = selected_r * cols + selected_c
    if idx < len(display_colors):
        rgb = [float(v) for v in display_colors[idx]]
        estimate = estimates.get((selected_r, selected_c), {})
        suffix = ""
        if stage == "estimated":
            suffix = f" | active {100.0 * float(estimate.get('active_ratio', 0.0)):.1f}%"
        imgui.text_disabled(
            f"Selected {stage} R{selected_r + 1} C{selected_c + 1}: "
            f"RGB {rgb[0]:.0f}, {rgb[1]:.0f}, {rgb[2]:.0f}{suffix}"
        )


def _draw_puzzle_capture_overlay(state, image, rect):
    if image is None or rect is None:
        return
    x, y, w, h = rect
    sx = float(w) / max(float(image.size[0]), 1.0)
    sy = float(h) / max(float(image.size[1]), 1.0)
    dl = imgui.get_window_draw_list()

    for point in state.get('puzzle_edge_landmarks', []):
        px = x + float(point.get("x", 0)) * sx
        py = y + float(point.get("y", 0)) * sy
        color = imgui.get_color_u32((0.0, 1.0, 0.55, 0.95))
        dl.add_circle_filled(imgui.ImVec2(px, py), 3.2, color)

    for point in state.get('puzzle_projected_points', []):
        px = x + float(point.get("x", 0)) * sx
        py = y + float(point.get("y", 0)) * sy
        color = imgui.get_color_u32((1.0, 1.0, 1.0, 0.90))
        dl.add_circle_filled(imgui.ImVec2(px, py), 2.4, color)

    selected = state.get('puzzle_selected_pixel', None)
    if selected is not None:
        px = x + float(selected.get("x", 0)) * sx
        py = y + float(selected.get("y", 0)) * sy
        color = imgui.get_color_u32((1.0, 0.80, 0.10, 1.0))
        dl.add_circle(imgui.ImVec2(px, py), 7.0, color, 16, 2.0)
        dl.add_line(imgui.ImVec2(px - 10.0, py), imgui.ImVec2(px + 10.0, py), color, 1.5)
        dl.add_line(imgui.ImVec2(px, py - 10.0), imgui.ImVec2(px, py + 10.0), color, 1.5)


def _draw_puzzle_glue_edge_overlay(image, rect, info):
    """Draws a line at every pasted tile boundary so individual copies (and any
    seam mismatch between them) are visible for a sanity check."""
    if image is None or rect is None or not info:
        return
    tile_w = int(info.get("tile_w", 0))
    tile_h = int(info.get("tile_h", 0))
    cols = int(info.get("cols", 1))
    rows = int(info.get("rows", 1))
    if tile_w <= 0 or tile_h <= 0:
        return
    x, y, w, h = rect
    sx = float(w) / max(float(image.size[0]), 1.0)
    sy = float(h) / max(float(image.size[1]), 1.0)
    dl = imgui.get_window_draw_list()
    color = imgui.get_color_u32((1.0, 0.15, 0.75, 0.85))

    for col_i in range(cols + 1):
        px = x + float(col_i * tile_w) * sx
        dl.add_line(imgui.ImVec2(px, y), imgui.ImVec2(px, y + h), color, 1.5)
    for row_i in range(rows + 1):
        py = y + float(row_i * tile_h) * sy
        dl.add_line(imgui.ImVec2(x, py), imgui.ImVec2(x + w, py), color, 1.5)


def _draw_puzzle_capture_widget(state, image, texture, avail_w, avail_h):
    rect = draw_fitted_texture(texture.glo, image.size[0], image.size[1], avail_w, avail_h, flip_y=True)
    if rect is None:
        return
    _draw_puzzle_capture_overlay(state, image, rect)
    x, y, w, h = rect
    io = imgui.get_io()
    mx, my = float(io.mouse_pos.x), float(io.mouse_pos.y)
    hovered = imgui.is_item_hovered() and x <= mx <= x + w and y <= my <= y + h
    if hovered and imgui.is_mouse_clicked(0):
        px = int(np.clip(round((mx - x) / max(w, 1.0) * (image.size[0] - 1)), 0, image.size[0] - 1))
        py = int(np.clip(round((my - y) / max(h, 1.0) * (image.size[1] - 1)), 0, image.size[1] - 1))
        rgb = image.getpixel((px, py))[:3]
        state.puzzle_selected_pixel = {"x": px, "y": py, "rgb": [int(v) for v in rgb]}

    selected = state.get('puzzle_selected_pixel', None)
    if selected is not None:
        rgb = selected.get("rgb", [0, 0, 0])
        imgui.color_button(
            "##puzzle_selected_pixel_color",
            (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, 1.0),
            imgui.ColorEditFlags_.no_tooltip,
            imgui.ImVec2(34, 22),
        )
        imgui.same_line()
        imgui.text(
            f"Pixel ({int(selected.get('x', 0))}, {int(selected.get('y', 0))}) "
            f"RGB {rgb[0]}, {rgb[1]}, {rgb[2]}"
        )
    else:
        imgui.text_disabled("Click the captured image to inspect a pixel RGB value.")


# %% GUI DRAWING PANELS ────────────────────────────────────────────────────────


def draw_menu_bar(state):
    current_mode = str(state.get('app_mode', 'edit'))
    if imgui.begin_menu_bar():
        if imgui.begin_menu("File"):
            clicked_save, _ = imgui.menu_item("Save Parameters...", "", False)
            if clicked_save:
                path = _pick_file('save', state.save_path)
                if path:
                    state.save_params(path)
            clicked_load, _ = imgui.menu_item("Load Parameters...", "", False)
            if clicked_load:
                path = _pick_file('load', state.load_path)
                if path:
                    state.load_params(path)
            clicked_auto, _ = imgui.menu_item("Autosave", "", bool(state.autosave_enabled))
            if clicked_auto:
                state.autosave_enabled = not bool(state.autosave_enabled)
            imgui.separator()
            clicked_init, _ = imgui.menu_item("Reset to Initial", "", False, current_mode == 'edit')
            if clicked_init:
                state.reset_to_initial()
            imgui.end_menu()
        if imgui.begin_menu("Edit"):
            has_undo = bool(state.undo_stack)
            undo_label = f"Undo {state.undo_stack[-1][0]}" if has_undo else "Undo"
            clicked_undo, _ = imgui.menu_item(undo_label, "Ctrl+Z", False, has_undo)
            if clicked_undo:
                state.undo_last()
            imgui.end_menu()
        if imgui.begin_menu("Mode"):
            for mode_id, mode_label, _ in MODE_SPECS:
                clicked_mode, _ = imgui.menu_item(mode_label, "", current_mode == mode_id)
                if clicked_mode:
                    _set_app_mode(state, mode_id)
            imgui.end_menu()
        if imgui.begin_menu("Window"):
            clicked_reset, _ = imgui.menu_item("Reset Layout", "", False)
            if clicked_reset:
                _reset_layout(state)
            light_theme = str(state.get('ui_theme', 'dark')) == 'light'
            clicked_theme, _ = imgui.menu_item("Light Theme", "", light_theme)
            if clicked_theme:
                state.ui_theme = 'dark' if light_theme else 'light'
                state._applied_ui_theme = ''
            imgui.end_menu()
        imgui.end_menu_bar()


def _reset_layout(state):
    """Restore all windows to their default positions and sizes."""
    try:
        os.remove(paths.LAYOUT_INI)
    except FileNotFoundError:
        pass
    context = imgui.get_current_context()
    imgui.internal.dock_context_clear_nodes(context, 0, True)
    imgui.internal.clear_ini_settings()
    state._layout_reset_pending = True
    state._layout_reset_viewport = True


def _mark_sim_geometry_dirty(state):
    """Invalidate the solver's cached Jacobian after the UI moves geometry.

    Taken under sim_lock so the flag cannot be set in the middle of a step's
    write-back, which would let a solve computed against the old layout land on
    top of the new one."""
    with state.sim_lock:
        state.sim_needs_jacobian_rebuild = True


def _draw_yarn_simulation_panel(state):
    """Edit-Mode yarn simulation controls. The solver itself runs on the
    background thread started in app.py; this only reads and writes state."""
    sim_allowed = str(state.get('app_mode', 'edit')) == 'edit'
    if not sim_allowed:
        imgui.text_disabled("Available in Edit Mode only.")
        return

    changed_active, active = imgui.checkbox("Run simulation##run_sim", bool(state.sim_active))
    if changed_active:
        if active:
            # Treat the shape at switch-on as the relaxed state, so stretch is
            # measured against what the user is looking at.
            state._refresh_sim_rest_lengths()
            _mark_sim_geometry_dirty(state)
        state.sim_active = bool(active)

    changed_ks, val_ks = imgui.slider_float("Stretch stiffness##sim_ks", float(state.sim_k_s), 0.0, 5000.0, "%.1f")
    if changed_ks:
        state.sim_k_s = float(val_ks)
    changed_kb, val_kb = imgui.slider_float("Bending stiffness##sim_kb", float(state.sim_k_b), 0.0, 500.0, "%.1f")
    if changed_kb:
        state.sim_k_b = float(val_kb)
    changed_kc, val_kc = imgui.slider_float("Collision stiffness##sim_kc", float(state.sim_k_c), 0.0, 100.0, "%.1f")
    if changed_kc:
        state.sim_k_c = float(val_kc)
    changed_dhat, val_dhat = imgui.slider_float("Yarn thickness##sim_dhat", float(state.sim_dhat), 0.005, 1.0, "%.3f")
    if changed_dhat:
        state.sim_dhat = float(val_dhat)

    if imgui.small_button("Reset to rest state##sim_reset"):
        state.push_undo("Simulation reset")
        state.sim_active = False
        state.rebuild_spline_from_params()
    imgui.same_line()
    if imgui.small_button("Verify derivatives##sim_fd"):
        # Uses the same implementation the solver runs, not knitting_core's
        # older copy, so the check reflects what is actually being solved.
        from yarn_simulation import check_gradients_and_hessians_fd
        with state.sim_lock:
            if state.sim_needs_jacobian_rebuild:
                state.rebuild_cached_jacobian()
            if state.J_cached is None:
                state.status_msg = "Nothing to verify: no control rows."
            else:
                res = check_gradients_and_hessians_fd(
                    state.ctrl_rows, state.period_offset_x, state.period_offset_y,
                    state.config, state.J_cached,
                    state.sim_k_s, state.sim_k_b, state.sim_k_c, state.sim_dhat,
                )
                state.status_msg = str(res).replace("\n", " | ")

    imgui.separator()
    imgui.text("Energy")
    with state.sim_lock:
        e_el, e_b, e_col = float(state.sim_e_el), float(state.sim_e_b), float(state.sim_e_col)
        delta_P = state.sim_delta_P
    imgui.text(f"Stretch:   {e_el:.6e}")
    imgui.text(f"Bending:   {e_b:.6e}")
    imgui.text(f"Collision: {e_col:.6e}")
    imgui.text(f"Total:     {e_el + e_b + e_col:.6e}")
    if delta_P is not None and len(delta_P):
        imgui.text(f"Max force: {float(np.max(np.linalg.norm(delta_P, axis=1))):.6e}")
    else:
        imgui.text("Max force: n/a")

    imgui.separator()
    changed_forces, val_forces = imgui.checkbox("Visualize forces##sim_forces", bool(state.sim_show_forces))
    if changed_forces:
        state.sim_show_forces = bool(val_forces)


def _open_model_in_edit(state, request):
    """Model Mode's repeating loops, fitted to a scan, made Edit Mode's pattern: its loop parameters
    (mm) and its tile of worked stitches and openings, shown repeated so the joins can be seen.
    One undo step, taken in Edit Mode, brings back what was there before."""
    _set_app_mode(state, 'edit')
    state.push_undo("Pattern from scanned model")
    if request.get('fabric') == 'crochet':
        # Crochet: Edit Mode's crochet fabric takes the fitted stitch numbers and grid as they are.
        state.fabric_type = 'crochet'
        state.crochet_params = dict(request['parameters'])
        state.crochet_stitches = np.asarray(request['stitches'], dtype=np.float32).copy()
        state.display_copies = np.array([1, 1], dtype=np.int32)
        state.rebuild_spline_mesh(preserve_model_placement=False)
        state.center_model_on_view()
        rows, stitches = state.crochet_stitches.shape
        state.status_msg = f"Crochet from the scan: {rows} rows x {stitches} stitches"
        return
    state.fabric_type = 'knit'
    state.apply_knit_pattern(request['parameters'], request['bitmap'])
    state.display_copies = np.array([1, 1], dtype=np.int32)
    state.rebuild_spline_mesh(preserve_model_placement=False)
    state.center_model_on_view()
    state.status_msg = f"Pattern from the scan: {state.bitmap.shape[0]} rows x {state.bitmap.shape[1]} stitches"


def _set_app_mode(state, mode):
    if mode in ('splat', 'splatting'):
        mode = 'model'
    mode = mode if mode in ('edit', 'scan', 'puzzle', 'database', 'ur5', 'model') else 'edit'
    current_mode = str(state.get('app_mode', 'edit'))
    if current_mode == mode:
        return
    # Invalidates every temporary Scan snapshot made in the mode being left.
    # A delayed cleanup must not restore that old model over the new workspace.
    state.mode_epoch = int(state.get('mode_epoch', 0)) + 1
    # Leaving UR5 Robot Mode stops the run, releases the arm and closes the
    # camera. Left connected, a background scan thread would keep moving real
    # hardware while the user is in a mode that shows none of it.
    if current_mode == 'ur5':
        controller = state.get('ur5_controller')
        if controller is not None:
            try:
                controller.close()
            except Exception as exc:
                logging.warning("gui._set_app_mode: %s", exc)
            state.ur5_controller = None
    if mode != 'edit' and bool(state.get('sim_active', False)):
        # Stop the solver before handing the model to Scan/Puzzle/Database.
        # Those snapshot ctrl_rows and swap in their own geometry, so a solver
        # still writing to it would fight them. Taken under sim_lock so a step
        # already in flight finishes and sees sim_active False on write-back.
        with state.sim_lock:
            state.sim_active = False
    scanner_idx = next((i for i, item in enumerate(state.workflow_stages) if item[0] == 'Scanner'), 0)
    embedded = state.get('embedded_scanner')
    if embedded is not None:
        try:
            embedded.close()
        except Exception as exc:
            logging.warning("gui._set_app_mode: %s", exc)
        state.embedded_scanner = None
    # Puzzle owns its model, pattern and undo history. Swap workspaces only
    # after the scanner has closed, so an in-flight temporary scan render has
    # finished restoring the non-Puzzle model before it is snapshotted.
    if current_mode == 'puzzle':
        state.leave_puzzle_workspace()
    state.app_mode = mode
    if mode == 'puzzle':
        # Select Puzzle before restoring its workspace so restore_snapshot
        # cannot infer the previous Scan renderer from the old mode.
        state.enter_puzzle_workspace()
    if mode == 'scan':
        state.workflow_step = scanner_idx
        state.scanner_preview_grid_enabled = True
        state.scanner_preview_rows = max(1, int(state.scanner_rows))
        state.scanner_preview_cols = max(1, int(state.scanner_cols))
        _clamp_scanner_selected_cell(state)
    elif mode == 'puzzle':
        state.workflow_step = scanner_idx
        state.scanner_preview_grid_enabled = False
        # enter_puzzle_workspace restored Puzzle's own placement. Its copy grid
        # is symmetric about the base model, so rebuilding it must preserve that
        # placement; recentering here made a prior Scan scene appear to transform
        # into Puzzle rather than returning to Puzzle's saved scene.
        _puzzle_apply_geometry_copies(state, preserve=True)
    elif mode == 'database':
        state.scanner_preview_grid_enabled = False
        state.display_copies = np.array([0, 0], dtype=np.int32)
        state.database_view_mode = "results"
    elif mode == 'model':
        # Model Mode draws its own pictures of a trained splat; the knit preview is not shown.
        state.scanner_preview_grid_enabled = False
        state.display_copies = np.array([0, 0], dtype=np.int32)
    elif mode == 'ur5':
        # Shows the same fabric grid preview Scan Mode does, since the real
        # robot scans that same layout -- the 3D View is the plan being sent to
        # the arm, so the two modes must not disagree about it.
        state.workflow_step = scanner_idx
        state.scanner_preview_grid_enabled = True
        state.scanner_preview_rows = max(1, int(state.scanner_rows))
        state.scanner_preview_cols = max(1, int(state.scanner_cols))
        _clamp_scanner_selected_cell(state)
    else:
        state.workflow_step = 0
        state.scanner_preview_grid_enabled = False
        state.display_copies = np.array([0, 0], dtype=np.int32)
    # Puzzle already rebuilt its copies above. A second non-preserving rebuild
    # discarded the independent transform that was just restored.
    if mode != 'puzzle':
        state.rebuild_spline_mesh(preserve_model_placement=False)


def _scan_duplicate_runs(state):
    """Previous scan runs that used the configuration currently set up.

    Returns None when there are none, or when the check itself cannot be made.
    A failure here must never block scanning: the point is to save the user a
    redundant run, which is worth nothing if it can also stop a wanted one.
    """
    storage = _scanner_storage(state)
    if storage is None:
        return None
    try:
        signature = storage.scan_config_signature_for_state(state)
        runs = storage.find_scan_runs_for_config(signature)
    except Exception:
        return None
    return runs or None


def _reference_overlay_active(state):
    """Whether the reference image should be drawn behind the model right now.

    It belongs to Edit Mode alone -- it exists to align the model against a
    photograph, which is an editing task. Puzzle Mode was showing it because the
    only gate was "not a scanner mode", so the fabric it lays out sat on top of
    an unrelated picture.

    Deliberately a display gate rather than a write to `show_ref_bg`: the
    checkbox stays the user's stored preference, so leaving Edit Mode hides the
    overlay and returning restores it, with nothing to remember or put back.
    """
    return str(state.get('app_mode', 'edit')) == 'edit' and bool(state.get('show_ref_bg', False))


def _apply_ui_theme(state):
    theme = str(state.get('ui_theme', 'dark'))
    if state.get('_applied_ui_theme') == theme:
        return
    if theme == 'light':
        imgui.style_colors_light()
    else:
        imgui.style_colors_dark()
    style = imgui.get_style()
    style.window_rounding = 6.0
    style.child_rounding = 4.0
    style.frame_rounding = 4.0
    style.grab_rounding = 3.0
    style.tab_rounding = 4.0
    style.frame_padding = imgui.ImVec2(6.0, 4.0)
    style.item_spacing = imgui.ImVec2(8.0, 6.0)
    style.window_padding = imgui.ImVec2(10.0, 10.0)
    state._applied_ui_theme = theme


def _draw_fabric_type(state):
    """Edit Mode's fabric: knitted loop rows, or crochet posts and chains. Returns the one shown."""
    fabric = str(state.get('fabric_type', 'knit'))
    imgui.text("Fabric:")
    for value, label in (('knit', "Knit (loops)"), ('crochet', "Crochet (posts & chains)")):
        imgui.same_line()
        if imgui.radio_button(f"{label}##fabric_{value}", fabric == value) and fabric != value:
            state.push_undo("Fabric type")
            if value == 'crochet' and bool(state.get('sim_active', False)):
                with state.sim_lock:
                    state.sim_active = False
            state.fabric_type = value
            fabric = value
            state.rebuild_spline_mesh(preserve_model_placement=False)
            state.center_model_on_view()
    imgui.separator()
    return fabric


# What each crochet number may be set to, and its step: (key, label, low, high, format, tooltip).
CROCHET_SLIDERS = (
    ('stitch_width', "Stitch width (mm)", 0.3, 5.0, "%.2f", "How far apart neighbouring stitches stand along the row."),
    ('row_height', "Row height (mm)", 0.8, 8.0, "%.2f", "How tall one row of posts is."),
    ('yarn_radius', "Yarn radius (mm)", 0.05, 1.0, "%.2f", "Half the yarn's thickness."),
    ('twist', "Post twist (turns)", 0.0, 2.0, "%.2f", "How many times a post's two legs turn round each other."),
    ('lean', "Post lean (mm)", -2.0, 2.0, "%.2f", "How far a post's top sits sideways of its foot."),
)


def _draw_crochet_editor(state):
    """Crochet in Edit Mode: the stitch grid (a post or a chain across, per stitch), the numbers in the
    stitch's equations, and the equations themselves (crochet_core)."""
    params = state.crochet_parameters().as_dict()
    stitches = np.asarray(state.get('crochet_stitches'), dtype=np.float32)
    changed = False
    if imgui.collapsing_header("Stitch grid##crochet_grid", imgui.TreeNodeFlags_.default_open):
        rows, cols = stitches.shape
        imgui.set_next_item_width(110)
        rows_changed, new_rows = imgui.input_int("Rows##crochet_rows", rows)
        imgui.same_line()
        imgui.set_next_item_width(110)
        cols_changed, new_cols = imgui.input_int("Stitches##crochet_cols", cols)
        if rows_changed or cols_changed:
            new_rows, new_cols = int(np.clip(new_rows, 1, 16)), int(np.clip(new_cols, 1, 32))
            if (new_rows, new_cols) != (rows, cols):
                state.push_undo("Crochet grid size")
                grown = np.ones((new_rows, new_cols), dtype=np.float32)
                grown[:min(rows, new_rows), :min(cols, new_cols)] = stitches[:min(rows, new_rows), :min(cols, new_cols)]
                stitches, changed = grown, True
        imgui.text_disabled("Click cell: green = post, dark = chain opening.")
        cell_w, cell_h = 22.0, 16.0
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (2, 2))
        for r in range(stitches.shape[0] - 1, -1, -1):
            imgui.text_disabled(f"R{r + 1}")
            imgui.same_line()
            for c in range(stitches.shape[1]):
                post = stitches[r, c] > 0.5
                imgui.push_style_color(imgui.Col_.button, (0.18, 0.62, 0.28, 1.0) if post else (0.22, 0.22, 0.24, 1.0))
                imgui.push_style_color(imgui.Col_.button_hovered, (0.28, 0.72, 0.38, 1.0) if post else (0.35, 0.35, 0.38, 1.0))
                imgui.push_style_color(imgui.Col_.button_active, (0.28, 0.72, 0.38, 1.0) if post else (0.35, 0.35, 0.38, 1.0))
                if imgui.button(f"##crochet_cell_{r}_{c}", (cell_w, cell_h)):
                    state.push_undo("Crochet stitch")
                    stitches = stitches.copy()
                    stitches[r, c] = 0.0 if post else 1.0
                    changed = True
                imgui.pop_style_color(3)
                if imgui.is_item_hovered():
                    imgui.set_tooltip(f"Row {r + 1}, Stitch {c + 1}: {'post' if post else 'chain (open)'}")
                if c < stitches.shape[1] - 1:
                    imgui.same_line()
        imgui.pop_style_var()
    if imgui.collapsing_header("Stitch shape##crochet_shape", imgui.TreeNodeFlags_.default_open):
        for key, label, low, high, fmt, tip in CROCHET_SLIDERS:
            moved, value = imgui.slider_float(f"{label}##crochet_{key}", float(params[key]), low, high, fmt)
            if imgui.is_item_activated():
                state.push_undo(label)
            if imgui.is_item_hovered():
                imgui.set_tooltip(tip)
            if moved:
                params[key], changed = float(value), True
        moved, wraps = imgui.slider_int("Yarn-overs per post##crochet_wraps", int(params['wraps']), 0, 4)
        if imgui.is_item_activated():
            state.push_undo("Yarn-overs")
        if imgui.is_item_hovered():
            imgui.set_tooltip("0 half double / single, 1 double crochet, 2 treble, 3 double treble")
        if moved:
            params['wraps'], changed = int(wraps), True
        copies = [int(c) for c in state.display_copies]
        imgui.set_next_item_width(160)
        moved, copies = imgui.input_int2("Repeat X, Y##crochet_copies", copies)
        if moved:
            state.display_copies = np.array([int(np.clip(c, 0, 6)) for c in copies], dtype=np.int32)
            changed = True
    if imgui.collapsing_header("Equations##crochet_equations"):
        name = {0: "single / half double crochet", 1: "double crochet", 2: "treble crochet", 3: "double treble"}.get(int(params['wraps']), "")
        imgui.text_disabled(f"{name}; s in [0, 1] up a post, t round a loop, u along a wrap (mm)")
        for line in crochet_core.equations(crochet_core.CrochetParams(**params)):
            imgui.text_wrapped(line)
    if changed:
        state.crochet_params = params
        state.crochet_stitches = stitches
        state.rebuild_spline_mesh(preserve_model_placement=True)


# Edit Mode's "Fit to a scan": one job at a time, its scanned model prepared once and reused by refits.
_SCAN_FIT = SimpleNamespace(models=None, index=-1, prepared=None, job=None, progress="", result=None, fabric=None,
                            read_openings=True, along_openings=True, picture=None, texture=None, message="",
                            photo_fit=None, try_more=True, grid=None)


def _start_scan_fit(state, fabric):
    view = _SCAN_FIT
    if view.models is None:
        view.models = scan_fit.find_models()
        measured = [i for i, m in enumerate(view.models) if m["mm_per_unit"] is not None]
        view.index = max(measured, key=lambda i: view.models[i]["psnr"] or 0.0) if measured else (0 if view.models else -1)
    model = view.models[view.index]
    prepared = view.prepared if view.prepared is not None and view.prepared["model"] is model else None
    holder = {}
    if fabric == 'crochet':
        grid = np.asarray(state.get('crochet_stitches'), dtype=np.float32).copy()
        wraps = int(state.crochet_parameters().wraps)
    else:
        grid, wraps = np.asarray(state.bitmap, dtype=np.float32).copy(), 0
    read_openings, along = bool(view.read_openings), bool(view.along_openings)

    def run():
        try:
            ready = prepared or scan_fit.prepare(model, progress=lambda m: setattr(view, "progress", m))
            say = lambda m: setattr(view, "progress", m)
            if fabric == 'crochet':
                result = scan_fit.fit_crochet(ready, grid, wraps, read_openings, along, progress=say)
            else:
                result = scan_fit.fit_knit(ready, grid, progress=say)
            say("Drawing the scan from above")
            holder.update(prepared=ready, result=result, picture=scan_fit.scan_picture(ready, result.angle_deg))
        except Exception as exc:          # shown in the panel; the fabric is left as it was
            holder["error"] = str(exc)

    view.progress, view.fabric = "Starting", fabric
    view.job = (threading.Thread(target=run, daemon=True, name="edit-scan-fit"), holder)
    view.job[0].start()


def _finish_scan_fit(state, renderer):
    """A finished fit made the fabric's: its numbers (and grid, when read from the scan), one undo step."""
    view = _SCAN_FIT
    if view.job is None or view.job[0].is_alive():
        return
    holder, view.job = view.job[1], None
    if "error" in holder:
        view.message = f"Could not fit: {holder['error']}"
        return
    view.prepared, view.result, view.picture = holder["prepared"], holder["result"], holder["picture"]
    view.texture = upload_rgb_texture(renderer.ctx, view.texture, np.asarray(view.picture))
    result = view.result
    state.push_undo("Fit to scan")
    if view.fabric == 'crochet':
        state.crochet_params = result.params.as_dict()
        if view.read_openings:
            state.crochet_stitches = np.asarray(result.cells, dtype=np.float32).copy()
        p = result.params
        view.message = (f"{result.stitch_name}: stitch {p.stitch_width:.2f} mm, row {p.row_height:.2f} mm, yarn r {p.yarn_radius:.2f} mm, "
                        f"twist {p.twist:.2f}, lean {p.lean:+.2f} mm")
    else:
        state.apply_knit_pattern(result.edit_parameters(), result.bitmap)
        view.message = (f"stitch {result.stitch_width:.2f} mm, rows {result.row_pitch:.2f} mm apart, loop {result.loop_height:.2f} mm, "
                        f"bulge {result.stitch_bulge:.2f}")
    state.rebuild_spline_mesh(preserve_model_placement=False)
    state.center_model_on_view()
    state.status_msg = "Fitted to the scan"


def _finish_photo_fit(state):
    """A finished photo fit made the crochet fabric's: the learned stitch, its numbers and grid, one undo step."""
    view = _SCAN_FIT
    job = view.photo_fit
    if job is None or job.running:
        return
    view.photo_fit = None
    result = job.result()
    if result is None:
        view.message = "The photo fit did not finish: " + (job.lines[-1] if job.lines else "no output")
        return
    _apply_photo_fit(state, result)


def _apply_photo_fit(state, result):
    """A photo fit's learned stitch, numbers and grid made the crochet fabric's, as one undo step."""
    view = _SCAN_FIT
    params, stitches, summary = result
    state.push_undo("Fit to the photos")
    state.fabric_type = 'crochet'
    state.crochet_params = params
    state.crochet_stitches = stitches
    state.rebuild_spline_mesh(preserve_model_placement=False)
    state.center_model_on_view()
    view.message = (f"Learned stitch: a stitch every {params['stitch_width']:.2f} mm, rows {params['row_height']:.2f} mm, "
                    f"yarn r {params['yarn_radius']:.2f} mm; {summary}")
    state.status_msg = "Fitted to the photos"


# How far the grid's sliders reach: degrees either way of where it started, and its sizes (mm).
GRID_TURN_DEG = 30.0
GRID_SIZES = {"stitch_width": (0.4, 5.0), "row_height": (0.8, 9.0)}


def _start_grid(state, renderer):
    """Read the chosen scan's photos and cameras, and start the stitch grid where the scan suggests
    (or where it was last lined up), on a background thread."""
    view = _SCAN_FIT
    model = view.models[view.index]
    grid_size = np.asarray(state.get('crochet_stitches')).shape
    wraps = int(state.crochet_parameters().wraps)
    holder = {}

    def run():
        try:
            prepared = view.prepared if view.prepared is not None and view.prepared["model"] is model else scan_fit.prepare(model)
            cameras, lens, train, _ = scan_fit.load_cameras(model["run"], model["dataset"])
            lattice = scan_fit.load_grid(model["output"])
            if lattice is None:
                start = scan_fit.fit(prepared["extraction"], stitches=int(grid_size[1]), rows=int(grid_size[0]), wraps=wraps,
                                     rows_along_holes=True, refine=False)
                lattice = scan_fit.lattice_from_fit(start)
            photos = scan_fit.read_photos(cameras)
            holder.update(prepared=prepared, cameras=cameras, lens=lens, train=train, photos=photos, lattice=lattice,
                          start_angle=float(lattice["angle_deg"]), index=scan_fit.square_on_camera(prepared, cameras),
                          table=scan_fit.table_colour(photos, prepared, cameras, lens, train))
        except Exception as exc:
            holder["error"] = str(exc)

    view.grid = SimpleNamespace(job=threading.Thread(target=run, daemon=True, name="stitch-grid"), holder=holder, lattice=None,
                                texture=None, picture_key=None, openings=[], message="Reading the scan's photos...")
    view.grid.job.start()


def _draw_grid(state, renderer):
    """The stitch grid over a photo of the scan, to line it up by eye: posts green, rows' tops yellow,
    openings read from the photos as red rings."""
    view = _SCAN_FIT
    grid = view.grid
    if grid is None:
        if imgui.button("Line up the stitch grid on a photo##grid_start", (-1, 0)):
            _start_grid(state, renderer)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Shows the scan's most square-on photo with the stitch grid drawn on it: turn, slide and\n"
                              "stretch the grid until the posts and openings sit on the real ones, then fit to the photos.")
        return
    if grid.job is not None:
        if grid.job.is_alive():
            imgui.text_colored((0.65, 0.65, 0.65, 1.0), grid.message)
            return
        holder, grid.job = grid.holder, None
        if "error" in holder:
            view.grid = None
            view.message = f"Could not show the grid: {holder['error']}"
            return
        for key, value in holder.items():
            setattr(grid, key, value)
        grid.message = ""
    lattice = grid.lattice
    changed = False
    _, angle = imgui.slider_float("Rows' direction (deg)##grid_angle", float(lattice["angle_deg"]),
                                  grid.start_angle - GRID_TURN_DEG, grid.start_angle + GRID_TURN_DEG, "%.1f")
    if angle != lattice["angle_deg"]:
        lattice["angle_deg"], changed = float(angle), True
    for key, label in (("stitch_width", "Stitch width (mm)"), ("row_height", "Row height (mm)")):
        low, high = GRID_SIZES[key]
        _, value = imgui.slider_float(f"{label}##grid_{key}", float(lattice[key]), low, high, "%.3f")
        if value != lattice[key]:
            lattice[key], changed = float(value), True
    w, h = lattice["stitch_width"], lattice["row_height"]
    _, along = imgui.slider_float("Slide along the rows (mm)##grid_su", float(lattice["shift"][0] % w), 0.0, w, "%.2f")
    _, across = imgui.slider_float("Slide across the rows (mm)##grid_sv", float(lattice["shift"][1] % h), 0.0, h, "%.2f")
    if (along, across) != (lattice["shift"][0] % w, lattice["shift"][1] % h):
        lattice["shift"], changed = [float(along), float(across)], True
    if changed:
        grid.openings = []                 # read again for the new grid
    if imgui.small_button("< Previous photo##grid_prev"):
        grid.index = (grid.index - 1) % len(grid.cameras)
    imgui.same_line()
    if imgui.small_button("Next photo >##grid_next"):
        grid.index = (grid.index + 1) % len(grid.cameras)
    imgui.same_line()
    if imgui.small_button("Read the openings##grid_open"):
        places = scan_fit.stitch_places(grid.prepared, lattice)
        share = scan_fit.photo_openings(grid.prepared, lattice, places, grid.cameras, grid.lens, grid.photos, grid.train, grid.table)
        grid.openings = [place for place, s in zip(places, share) if s >= 0.35]
        grid.message = f"{len(grid.openings)} of {len(places)} stitches read as openings (red)"
    imgui.same_line()
    if imgui.small_button("Save grid##grid_save"):
        path = scan_fit.save_grid(view.models[view.index]["output"], lattice)
        grid.message = f"Saved to {path.name}; the photo fit keeps it"
    key = (grid.index, json.dumps(lattice, sort_keys=True), len(grid.openings))
    if grid.picture_key != key:
        camera = grid.cameras[grid.index]
        picture = scan_fit.grid_picture(grid.prepared, lattice, camera, grid.lens, grid.photos[grid.index], grid.openings)
        grid.texture = upload_rgb_texture(renderer.ctx, grid.texture, np.asarray(picture))
        grid.picture_key, grid.size = key, picture.size
    imgui.text_colored((0.65, 0.65, 0.65, 1.0), f"Photo {grid.index + 1} of {len(grid.cameras)}: posts green, rows' tops yellow, openings red")
    width = max(160.0, imgui.get_content_region_avail().x)
    draw_fitted_texture(grid.texture.glo, grid.size[0], grid.size[1], width, width * grid.size[1] / grid.size[0], flip_y=True)
    if grid.message:
        imgui.text_wrapped(grid.message)


def _draw_photo_fit(state):
    """The slow, best fit: the stitch's own curves fitted to the photos (scan_fit.PhotoFit)."""
    view = _SCAN_FIT
    _finish_photo_fit(state)
    if view.photo_fit is not None:
        imgui.text_colored((0.65, 0.65, 0.65, 1.0), view.photo_fit.progress)
        if imgui.small_button("Stop the photo fit##photo_fit_stop"):
            view.photo_fit.stop()
        return
    _, view.try_more = imgui.checkbox("Also try 2x and 3x the stitches (more posts per repeat)##photo_fit_try", bool(view.try_more))
    if imgui.button("Fit to the photos (slow, best)##photo_fit_go", (-1, 0)):
        grid = np.asarray(state.get('crochet_stitches'))
        rows, stitches = int(grid.shape[0]), int(grid.shape[1])
        output = view.models[view.index]["output"]
        lined_up = view.grid is not None and view.grid.lattice is not None
        candidates = ",".join(f"{stitches * k}x{rows}" for k in (1, 2, 3)) if view.try_more and not lined_up else ""
        lattice_path = scan_fit.save_grid(output, view.grid.lattice) if lined_up else None
        view.photo_fit = scan_fit.PhotoFit(output, stitches, rows, int(state.crochet_parameters().wraps), candidates, lattice_path)
    if imgui.is_item_hovered():
        imgui.set_tooltip("Draws the stitch equations into every photo of the scan and changes them until the drawings look\n"
                          "like the photos: the lattice, the cloth's bow, and the stitch's own curves (learned, as Fourier\n"
                          "series). A few minutes on the graphics card; close the splat viewer first.")
    output = view.models[view.index].get("output")
    last = Path(output) / "stitch_fit" if output is not None else None
    if last is not None and (last / "stitch_model.json").is_file():
        if imgui.small_button("Open the last photo fit##photo_fit_open"):
            result = scan_fit.read_photo_fit(last)
            if result is not None:
                _apply_photo_fit(state, result)
        if imgui.is_item_hovered():
            imgui.set_tooltip("The stitch this scan's last photo fit learned, without fitting again")
    if state.crochet_parameters().templates:
        if imgui.small_button("Back to the hand-made stitch##photo_fit_forget"):
            state.push_undo("Hand-made stitch")
            state.crochet_params = {**state.crochet_parameters().as_dict(), "templates": None}
            state.rebuild_spline_mesh(preserve_model_placement=True)


def _draw_scan_fit(state, fabric, renderer):
    """Fit this fabric's equations to a scanned model, keeping the repeat's size (the grid's rows and
    stitches) and, if asked, the grid drawn here; the scan from above beside the result."""
    view = _SCAN_FIT
    _finish_scan_fit(state, renderer)
    if view.models is None:
        view.models = scan_fit.find_models()
        measured = [i for i, m in enumerate(view.models) if m["mm_per_unit"] is not None]
        view.index = max(measured, key=lambda i: view.models[i]["psnr"] or 0.0) if measured else (0 if view.models else -1)
    if not view.models:
        imgui.text_colored((0.95, 0.75, 0.20, 1.0), "No trained scans under robot_scans.")
        return
    imgui.set_next_item_width(-1)
    changed, index = imgui.combo("##scan_fit_model", view.index, [m["name"] for m in view.models])
    if changed:
        view.index, view.prepared, view.result, view.picture = index, None, None, None
    grid = np.asarray(state.get('crochet_stitches') if fabric == 'crochet' else state.bitmap)
    imgui.text_wrapped(f"The repeat is this grid: {grid.shape[0]} rows x {grid.shape[1]} stitches - from one opening to the next. "
                       "Set its size above to what you count on the lace; the stitch sizes, yarn and loop shape are measured.")
    if fabric == 'crochet':
        _, view.read_openings = imgui.checkbox("Read the openings from the scan (else keep my grid)##scan_fit_read", bool(view.read_openings))
        _, view.along_openings = imgui.checkbox("Rows run along the rows of openings##scan_fit_along", bool(view.along_openings))
    busy = view.job is not None
    if busy:
        imgui.begin_disabled()
    if imgui.button("Fit these equations to the scan##scan_fit_go", (-1, 0)):
        _start_scan_fit(state, fabric)
    if busy:
        imgui.end_disabled()
        imgui.text_colored((0.65, 0.65, 0.65, 1.0), view.progress + "...")
    if fabric == 'crochet':
        _draw_grid(state, renderer)
        _draw_photo_fit(state)
    if view.message:
        imgui.text_wrapped(view.message)
    result = view.result
    if result is not None and view.fabric == fabric and result.comparison:
        c = result.comparison
        imgui.text(f"Model yarn to scan: {c['rebuilt_to_scan_mean_mm']:.2f} mm mean; "
                   f"scan covered within 0.2 mm: {c['scan_within_0.2mm'] * 100:.0f}%")
    if view.texture is not None and view.picture is not None:
        imgui.text_colored((0.65, 0.65, 0.65, 1.0), "The scan from above, its rows turned to run left to right:")
        width = max(160.0, imgui.get_content_region_avail().x)
        w, h = view.picture.size
        draw_fitted_texture(view.texture.glo, w, h, width, width * h / w, flip_y=True)


def _draw_bitmap_editor(state, id_suffix=""):
    """Editable 0/1 stitch bitmap grid: resize rows/columns and toggle each cell.
    Shared by the main Pattern panel and Puzzle Mode so both edit the same
    state.bitmap that drives ctrl_rows / the knitted model geometry."""
    max_rows = int(state.config['knit_parameters']['bitmap_rows'])
    ch_r, new_rows = imgui.slider_int(f"Rows##bres{id_suffix}", int(state.bitmap_size[0]), 1, max_rows)
    ch_c, new_cols = imgui.slider_int(f"Columns##bres{id_suffix}", int(state.bitmap_size[1]), 1, 32)
    if ch_r or ch_c:
        state.push_undo("Bitmap size")
        state.on_bitmap_resize(new_rows, new_cols)
    if imgui.small_button(f"All active##bmap{id_suffix}"):
        state.push_undo("Pattern reset")
        state.bitmap[:] = 1.0
        state.on_bitmap_change()
    nr, nc = state.bitmap.shape
    cell_w, cell_h = 22, 16
    imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(2, 2))
    changed_bitmap = False
    # Drawn bottom-up so row 1 sits at the bottom of the grid, where it sits in
    # the fabric: rows are built at y = row_idx * dy, so index 0 is the lowest
    # row knitted. Iterating 0..n put index 0 at the top instead, leaving the
    # grid a mirror image of the model -- clearing the third square from the top
    # opened a gap second from the top, and the wrong stitch appeared to grow.
    # Bottom-up also matches how a knitting chart is read.
    for r in reversed(range(nr)):
        imgui.text_disabled(f"R{r + 1}")
        imgui.same_line()
        # Row 1 is the cast-on: every other cleared cell is stood in for by the
        # stitch below it, and row 0 has none, so clearing it leaves the fabric
        # hanging from nothing. Locked rather than merely repaired afterwards,
        # so the pattern on screen is always one that can be knitted.
        locked = r == 0
        for c in range(nc):
            active = float(state.bitmap[r, c]) > 0.5
            if locked:
                # Its own green, and the same colour for hover and press: a
                # square that lights up and then does nothing reads as broken.
                fill = (0.16, 0.44, 0.24, 1.0)
                imgui.push_style_color(imgui.Col_.button, fill)
                imgui.push_style_color(imgui.Col_.button_hovered, fill)
                imgui.push_style_color(imgui.Col_.button_active, fill)
            else:
                imgui.push_style_color(imgui.Col_.button, (0.18, 0.62, 0.28, 1.0) if active else (0.22, 0.22, 0.22, 1.0))
                imgui.push_style_color(imgui.Col_.button_hovered, (0.28, 0.72, 0.38, 1.0) if active else (0.35, 0.35, 0.35, 1.0))
                imgui.push_style_color(imgui.Col_.button_active, (0.28, 0.72, 0.38, 1.0) if active else (0.35, 0.35, 0.35, 1.0))
            clicked = imgui.button(f"##bm{id_suffix}_{r}_{c}", imgui.ImVec2(cell_w, cell_h))
            imgui.pop_style_color(3)
            if locked and imgui.is_item_hovered():
                imgui.set_tooltip("Bottom row - always knitted. The fabric hangs from these stitches.")
            # Turning a bottom cell on is still allowed, so a pattern that
            # arrived cleared can be fixed by clicking it. Turning one off is
            # refused before push_undo, so a refused click leaves no undo entry
            # that would appear to do nothing.
            if clicked and not (locked and active):
                if not changed_bitmap:
                    state.push_undo("Pattern")
                state.bitmap[r, c] = 0.0 if active else 1.0
                changed_bitmap = True
            if c < nc - 1:
                imgui.same_line()
    imgui.pop_style_var()
    imgui.text_disabled("Row 1 is the bottom row, as knitted - always on.")
    if changed_bitmap:
        state.on_bitmap_change()
    return changed_bitmap


MODE_SPECS = (
    ('edit', "Edit", "Edit knitted and crocheted geometry, material, and simulation"),
    ('scan', "Scan", "Simulate multi-angle UR5e fabric scanning in MuJoCo"),
    ('puzzle', "Puzzle", "Test seamless tile repeat and landmark extraction"),
    ('database', "Database", "Browse captured images, metadata, and RGB analyses"),
    ('model', "Splatting", "Inspect 3D Gaussian splats and fit yarn/stitch equations"),
    ('ur5', "UR5 Robot", "Connect to physical UR5e or TeleopApp, capture, and reconstruct"),
)

GEOMETRY_PARAM_LABELS = {
    'stitch_bulge': "Stitch bulge",
    'stitch_z': "Loop depth (Z)",
    'dy': "Row spacing (dy)",
    'radius': "Yarn radius",
    'ellipse_ratio': "Ellipse ratio",
}


def _draw_appearance_controls(state, rebuild_current_mesh, is_knit=True):
    if imgui.collapsing_header("Material", imgui.TreeNodeFlags_.default_open):
        changed_mode, use_row_colors = imgui.checkbox("Colors per row##rowcolors", bool(state.use_row_colors))
        if changed_mode:
            state.push_undo("Color mode")
            state.use_row_colors = use_row_colors
            rebuild_current_mesh()
        if not state.use_row_colors:
            changed_c, new_col = imgui.color_edit3("Model color##single_color", tuple(float(x) for x in state.single_model_color[:3]))
            if changed_c:
                state.single_model_color = np.array(new_col, dtype=np.float32)
                rebuild_current_mesh()
        else:
            colors_changed = False
            for row_idx in range(int(state.bitmap_size[0])):
                col = state.row_colors[row_idx]
                changed_c, new_col = imgui.color_edit3(f"Row {row_idx + 1}##row_color_{row_idx}", (float(col[0]), float(col[1]), float(col[2])))
                if changed_c:
                    state.row_colors[row_idx] = list(new_col)
                    colors_changed = True
            if colors_changed:
                rebuild_current_mesh()

    if imgui.collapsing_header("Display & Lighting", imgui.TreeNodeFlags_.default_open):
        changed_x, new_x = imgui.slider_int("Copy via X##display_copies_x", int(state.display_copies[0]), 0, 20)
        released_x = imgui.is_item_deactivated_after_edit()
        changed_y, new_y = imgui.slider_int("Copy via Y##display_copies_y", int(state.display_copies[1]), 0, 20)
        released_y = imgui.is_item_deactivated_after_edit()
        if changed_x or changed_y:
            if not bool(state.__dict__.get('_display_copies_dragging', False)):
                state.push_undo("Display copies")
                object.__setattr__(state, '_display_copies_dragging', True)
            state.scanner_preview_grid_enabled = False
            state.display_copies = np.array([int(new_x), int(new_y)], dtype=np.int32)
        if released_x or released_y:
            object.__setattr__(state, '_display_copies_dragging', False)
            state.rebuild_spline_mesh(preserve_model_placement=True)
        if imgui.small_button("Single model##display_single"):
            state.push_undo("Display copies")
            state.display_copies = np.array([0, 0], dtype=np.int32)
            state.rebuild_spline_mesh(preserve_model_placement=True)
        imgui.same_line()
        if imgui.small_button("Center model##display_center"):
            state.push_undo("Center model")
            state.center_model_on_view()
        changed_alpha, new_alpha = imgui.slider_float("Model opacity##mdl", float(state.model_alpha), 0.0, 1.0, "%.2f")
        if changed_alpha:
            state.model_alpha = float(new_alpha)
        changed_view_fov, new_view_fov = imgui.slider_float("View FoV##view", float(state.view_fov), 10.0, 120.0, "%.1f")
        if changed_view_fov:
            state.view_fov = float(new_view_fov)
            state.camera.fov_deg = float(new_view_fov)
        _, state.show_ref_bg = imgui.checkbox("Show reference overlay##display_ref", bool(state.show_ref_bg))
        if state.show_ref_bg:
            _, state.ref_bg_alpha = imgui.slider_float("Reference opacity##bg", float(state.ref_bg_alpha), 0.0, 1.0, "%.2f")
        imgui.separator()
        changed_light, new_light = imgui.color_edit3(
            "Light color##render_light",
            (float(state.render_light_color[0]), float(state.render_light_color[1]), float(state.render_light_color[2])),
        )
        if changed_light:
            state.render_light_color = np.array(new_light, dtype=np.float32)
        changed_intensity, new_intensity = imgui.slider_float("Light intensity##render_light", float(state.render_light_intensity), 0.05, 3.0, "%.2f")
        if changed_intensity:
            state.render_light_intensity = float(new_intensity)
        changed_ao_s, new_ao_s = imgui.slider_float("AO strength##render_ao", float(state.render_ao_strength), 0.0, 2.0, "%.2f")
        changed_ao_r, new_ao_r = imgui.slider_float("AO radius##render_ao", float(state.render_ao_radius), 0.01, 1.0, "%.2f")
        if changed_ao_s:
            state.render_ao_strength = float(new_ao_s)
        if changed_ao_r:
            state.render_ao_radius = float(new_ao_r)
        if imgui.small_button("Reset lighting##render_light"):
            state.push_undo("Lighting")
            state.render_light_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            state.render_light_intensity = 0.9
            state.render_ao_strength = 0.5
            state.render_ao_radius = 0.15

    if not is_knit:
        return

    if imgui.collapsing_header("Surface Fibers", imgui.TreeNodeFlags_.default_open):
        changed_enabled, enabled = imgui.checkbox("Use multi-fiber rows##fiber_geometry_enabled", bool(state.fiber_geometry_enabled))
        fibers_changed = False
        if changed_enabled:
            state.push_undo("Surface fibers")
            state.fiber_geometry_enabled = enabled
            fibers_changed = True
        if state.fiber_geometry_enabled:
            changed, value = imgui.slider_int("Fibers per row##fiber_geometry_count", int(state.fiber_geometry_count), 1, 64)
            if changed:
                state.fiber_geometry_count = int(value)
                fibers_changed = True
            for key, label, lo, hi in (
                ('fiber_geometry_radius_scale', 'Fiber radius scale', 0.04, 0.45),
                ('fiber_geometry_lift', 'Lift above surface', 0.0, 1.0),
                ('fiber_geometry_surface_arc', 'Surface spread', 0.05, 1.0),
                ('fiber_geometry_randomness', 'Randomness', 0.0, 1.0),
                ('fiber_geometry_twist', 'Fiber twist', -3.0, 3.0),
            ):
                changed, value = imgui.slider_float(f"{label}##{key}", float(state[key]), float(lo), float(hi), "%.2f")
                if changed:
                    state[key] = float(value)
                    fibers_changed = True
        else:
            imgui.text_disabled("Enable multi-fiber rows to separate each yarn into smaller fibers.")
        if fibers_changed:
            rebuild_current_mesh()

    if imgui.collapsing_header("Texture"):
        changed_tex, new_tex = imgui.color_edit3(
            "Texture tint##render_texture",
            (float(state.render_texture_color[0]), float(state.render_texture_color[1]), float(state.render_texture_color[2])),
        )
        if imgui.is_item_activated():
            state.push_undo("Render texture")
        if changed_tex:
            state.render_texture_color = np.array(new_tex, dtype=np.float32)
        if imgui.small_button("Neutral tint##texture"):
            state.push_undo("Render texture")
            state.render_texture_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        imgui.same_line()
        if imgui.small_button("Copy material color##texture"):
            state.push_undo("Render texture")
            src = state.row_colors[0] if state.use_row_colors and state.row_colors else state.single_model_color
            state.render_texture_color = np.array(src[:3], dtype=np.float32)
        for group in state.texture_control_groups:
            if imgui.tree_node(group['title']):
                for control in group['controls']:
                    key = control['key']
                    changed, new_val = imgui.slider_float(
                        f"{control['label']}##{key}",
                        float(state[key]),
                        float(control['min']),
                        float(control['max']),
                        control['format'],
                    )
                    if imgui.is_item_activated():
                        state.push_undo(control['label'])
                    if changed:
                        state[key] = float(new_val)
                imgui.tree_pop()
        imgui.separator()
        for preset in state.texture_preset_buttons:
            if preset.get('same_line'):
                imgui.same_line()
            if imgui.small_button(f"{preset['label']}##texture_preset_{preset['preset']}"):
                state.push_undo("Texture preset")
                state.apply_texture_preset(preset['preset'])


def draw_sidebar(state, renderer, window=None):
    _apply_ui_theme(state)
    current_mode = str(state.get('app_mode', 'edit'))
    reset = bool(state.get('_layout_reset_pending', False)) or not bool(state.get('_sidebar_sized_once', False))
    cond = imgui.Cond_.always if reset else imgui.Cond_.first_use_ever
    imgui.set_next_window_pos((20, 20), cond=cond)
    imgui.set_next_window_size((360, 820), cond=cond)
    state._sidebar_sized_once = True
    if reset:
        imgui.set_next_window_focus()
        state._layout_reset_pending = False
    imgui.begin("Knitting Control")

    def rebuild_current_mesh(preserve=True):
        state.rebuild_spline_mesh(preserve_model_placement=preserve)

    def ensure_embedded_robot_viewer(auto_start=False, single_capture=False):
        if not auto_start and not single_capture and bool(state.get('_embedded_scanner_viewer_failed', False)):
            return None
        _clamp_scanner_selected_cell(state)
        state.save_params(state.save_path, silent=True)
        embedded = state.get('embedded_scanner')
        if embedded is not None:
            if auto_start and not getattr(embedded, 'running', False):
                try:
                    embedded.close()
                except Exception as exc:
                    logging.warning("gui.ensure_embedded_robot_viewer: %s", exc)
                state.embedded_scanner = None
            else:
                if single_capture:
                    embedded.running = False
                    embedded.paused = True
                    embedded.args.capture_mode = str(state.get('scanner_capture_mode', 'natural'))
                    embedded.args.camera_zoom = float(state.get('scanner_camera_zoom', 1.0))
                return embedded
        try:
            tiled_full_image, tiled_per_cell = _scanner_generate_tiled_layout(state, renderer)
            embedded = EmbeddedMujocoScanner(
                state,
                renderer.ctx,
                window,
                preview_image=tiled_full_image,
                per_cell_images=tiled_per_cell,
                auto_start=auto_start,
            )
            state.embedded_scanner = embedded
            state._embedded_scanner_viewer_failed = False
            if single_capture:
                embedded.running = False
                embedded.paused = True
                embedded.args.capture_mode = str(state.get('scanner_capture_mode', 'natural'))
                embedded.args.camera_zoom = float(state.get('scanner_camera_zoom', 1.0))
            elif not auto_start:
                state.scanner_status = embedded.status
            return embedded
        except Exception as exc:
            state.embedded_scanner = None
            if not single_capture:
                state._embedded_scanner_viewer_failed = True
            label = "single capture preview" if single_capture else ("embedded MuJoCo scanner" if auto_start else "embedded MuJoCo Robot Viewer")
            state.scanner_status = f"Could not start {label}: {exc}"
            return None

    def start_scanner_process(force=False):
        mode = str(state.scanner_execution_mode)
        embedded = state.get('embedded_scanner')
        if embedded is not None and mode == "simulation":
            if getattr(embedded, 'paused', False) and not getattr(embedded, 'single_capture_mode', False):
                embedded.resume()
                state.scanner_status = "Embedded MuJoCo scanner continued"
                return
            if getattr(embedded, 'running', False):
                state.scanner_status = "Scanner already running"
                return

        if not force:
            existing = _scan_duplicate_runs(state)
            if existing:
                state._scan_duplicate_prompt = existing
                state.scanner_status = "This scan already exists in the database."
                return
        state._scan_duplicate_prompt = None

        if embedded is not None:
            try:
                embedded.close()
            except Exception as exc:
                logging.warning("gui.start_scanner_process: %s", exc)
            state.embedded_scanner = None
        _clamp_scanner_selected_cell(state)
        state.save_params(state.save_path, silent=True)
        if mode == "simulation":
            embedded = ensure_embedded_robot_viewer(auto_start=True)
            if embedded is not None:
                state.scanner_status = "Embedded MuJoCo scanner running"
            return

        try:
            import fabric_scanner as scanner

            args = build_scanner_plan_args(
                state,
                model_curves=None,
                robot_ip=str(state.scanner_robot_ip),
                robot_port=int(state.scanner_robot_port),
                robot_vel=0.015,
                robot_acc=0.08,
                robot_dwell=0.20,
            )
            plan = scanner.build_plan(args)
            plan = scanner.densify_plan_for_robot(plan)
            safe, issues = scanner.assess_plan_safety(plan.mapped_points, max_step=0.08)
            if not safe:
                state.scanner_status = "Real UR5 path blocked: " + "; ".join(issues)
                return
            scanner.run_robot_motion(plan, args)
            state.scanner_status = "Real UR5 command sent from selected GUI settings"
        except Exception as exc:
            state.scanner_process = None
            state.scanner_status = f"Could not start scanner: {exc}"

    def stop_scanner_process():
        embedded = state.get('embedded_scanner')
        if embedded is not None:
            embedded.pause()
            state.scanner_status = embedded.status
            return
        state.scanner_status = "Scanner idle"

    edit_active = current_mode == 'edit'
    scan_active = current_mode == 'scan'
    puzzle_active = current_mode == 'puzzle'
    database_active = current_mode == 'database'
    ur5_active = current_mode == 'ur5'
    model_active = current_mode == 'model'

    cols_per_row = 3
    spacing_x = imgui.get_style().item_spacing.x
    avail_w = imgui.get_content_region_avail().x
    mode_btn_w = max(72.0, (avail_w - spacing_x * (cols_per_row - 1)) / cols_per_row)
    is_light = str(state.get('ui_theme', 'dark')) == 'light'
    inactive_bg = (0.82, 0.84, 0.88, 1.0) if is_light else (0.22, 0.24, 0.28, 1.0)
    inactive_ur5 = (0.92, 0.82, 0.72, 1.0) if is_light else (0.30, 0.22, 0.16, 1.0)
    for idx, (mode_id, mode_label, mode_tip) in enumerate(MODE_SPECS):
        if idx % cols_per_row != 0:
            imgui.same_line()
        active = current_mode == mode_id
        if mode_id == 'ur5':
            bg = (0.78, 0.44, 0.14, 1.0) if active else inactive_ur5
            hov = (0.88, 0.52, 0.20, 1.0)
        else:
            bg = (0.22, 0.48, 0.78, 1.0) if active else inactive_bg
            hov = (0.30, 0.56, 0.86, 1.0) if active else ((0.74, 0.78, 0.84, 1.0) if is_light else (0.30, 0.33, 0.38, 1.0))
        imgui.push_style_color(imgui.Col_.button, bg)
        imgui.push_style_color(imgui.Col_.button_hovered, hov)
        if active and is_light:
            imgui.push_style_color(imgui.Col_.text, (1.0, 1.0, 1.0, 1.0))
        if imgui.button(f"{mode_label}##mode_{mode_id}", (mode_btn_w, 0)):
            _set_app_mode(state, mode_id)
        if active and is_light:
            imgui.pop_style_color()
        imgui.pop_style_color(2)
        if imgui.is_item_hovered():
            imgui.set_tooltip(mode_tip)
    imgui.separator()

    if edit_active or puzzle_active:
        quarter_w = max(64.0, (imgui.get_content_region_avail().x - spacing_x * 3.0) / 4.0)
        undo_disabled = not state.undo_stack
        if undo_disabled:
            imgui.begin_disabled()
        if imgui.button("Undo##main", (quarter_w, 0)):
            state.undo_last()
        if undo_disabled:
            imgui.end_disabled()
        imgui.same_line()
        if not edit_active:
            imgui.begin_disabled()
        if imgui.button("Reset##reset_saved_initial_global", (quarter_w, 0)):
            state.reset_to_initial()
        if not edit_active:
            imgui.end_disabled()
        imgui.same_line()
        if imgui.button("Save...##save_params", (quarter_w, 0)):
            path = _pick_file('save', state.save_path)
            if path:
                state.save_params(path)
        imgui.same_line()
        if imgui.button("Load...##load_params", (quarter_w, 0)):
            path = _pick_file('load', state.load_path)
            if path:
                state.load_params(path)
        imgui.separator()

    if database_active:
        _draw_database_summary_page(state, renderer)
        state.maybe_autosave()
        imgui.end()
        return

    if model_active:
        gui_model.draw_model_panel(state, renderer)
        request = state.get('model_to_edit')
        if request is not None:
            state.model_to_edit = None
            imgui.end()
            _open_model_in_edit(state, request)
            return
        state.maybe_autosave()
        imgui.end()
        return

    if ur5_active:
        gui_ur5.draw_ur5_panel(state, renderer, window)
        state.maybe_autosave()
        imgui.end()
        return

    if edit_active:
        _finish_scan_fit(state, renderer)
        _finish_photo_fit(state)
        fabric = _draw_fabric_type(state)
        if imgui.begin_tab_bar("##edit_tabs"):
            tab_shape, _ = imgui.begin_tab_item("Pattern & Shape")
            if tab_shape:
                if fabric == 'crochet':
                    _draw_crochet_editor(state)
                else:
                    if imgui.collapsing_header("Pattern", imgui.TreeNodeFlags_.default_open):
                        _draw_bitmap_editor(state)

                    if imgui.collapsing_header("Geometry", imgui.TreeNodeFlags_.default_open):
                        quality_changed = False
                        changed_loop_res, new_loop_res = imgui.slider_int("Path smoothness##mesh_loop_res", int(state.config['knit_parameters']['loop_res']), 8, 96)
                        changed_segments, new_segments = imgui.slider_int("Fiber roundness##mesh_segments", int(state.config['knit_parameters']['segments']), 8, 64)
                        if changed_loop_res:
                            state.config['knit_parameters']['loop_res'] = int(new_loop_res)
                            quality_changed = True
                        if changed_segments:
                            state.config['knit_parameters']['segments'] = int(new_segments)
                            quality_changed = True
                        params_changed = False
                        for i, pd in enumerate(state.config['knit_parameters']['parameters']):
                            if pd['name'] not in GEOMETRY_PARAM_LABELS:
                                continue
                            lo, hi = pd['range']
                            label = GEOMETRY_PARAM_LABELS[pd['name']]
                            changed, new_val = imgui.slider_float(f"{label}##p{i}", float(state.params[i]), float(lo), float(hi), "%.3f")
                            if imgui.is_item_activated():
                                state.push_undo(label)
                            if changed:
                                state.params[i] = float(new_val)
                                params_changed = True
                        if quality_changed:
                            rebuild_current_mesh()
                        if params_changed:
                            state.nudge_spline_from_params()

                    if imgui.collapsing_header("Loop Heights", imgui.TreeNodeFlags_.default_open):
                        params = state.config['knit_parameters']['parameters']
                        default_idx = state._lh_idx[0] if state._lh_idx else None
                        lo, hi = (0.0, 6.0)
                        if default_idx is not None:
                            lo, hi = params[default_idx]['range']
                        changed_any = False
                        active_mask = state.bitmap > 0.5
                        mean_h = float(np.mean(state.loop_heights[active_mask])) if np.any(active_mask) else float(lo)
                        ch_all, val_all = imgui.slider_float("Uniform height##loop_h_all", mean_h, float(lo), float(hi), "%.2f")
                        if imgui.is_item_activated():
                            state.push_undo("Uniform loop height")
                        if ch_all:
                            for r in range(int(state.bitmap_size[0])):
                                for c in range(int(state.bitmap_size[1])):
                                    if float(state.bitmap[r, c]) > 0.5:
                                        state.set_loop_height_cell(r, c, float(val_all))
                            changed_any = True
                        imgui.text_disabled("Drag cell values (bottom row R1 first):")
                        nr, nc = int(state.bitmap_size[0]), int(state.bitmap_size[1])
                        cell_w = max(38.0, min(58.0, (imgui.get_content_region_avail().x - 28.0 - max(0, nc - 1) * 4.0) / max(nc, 1)))
                        imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(4, 3))
                        for r in reversed(range(nr)):
                            imgui.text_disabled(f"R{r + 1}")
                            imgui.same_line()
                            for c in range(nc):
                                active = float(state.bitmap[r, c]) > 0.5
                                if not active:
                                    imgui.begin_disabled()
                                imgui.set_next_item_width(cell_w)
                                changed, val = imgui.drag_float(
                                    f"##loop_h_{r}_{c}",
                                    float(state.loop_heights[r, c]) if active else 0.0,
                                    0.02,
                                    float(lo),
                                    float(hi),
                                    "%.2f",
                                )
                                if not active:
                                    imgui.end_disabled()
                                if imgui.is_item_activated():
                                    state.push_undo("Loop height")
                                if active and changed:
                                    state.set_loop_height_cell(r, c, float(np.clip(val, lo, hi)))
                                    changed_any = True
                                if c < nc - 1:
                                    imgui.same_line()
                        imgui.pop_style_var()
                        if changed_any:
                            state.rebuild_spline_from_params()

                    if imgui.collapsing_header("Spline & Tiling Period"):
                        if imgui.small_button("Rebuild spline from params##spline_rebuild"):
                            state.push_undo("Rebuild from params")
                            state.rebuild_spline_from_params()
                        ch_spl, new_spl = imgui.slider_int("Samples/loop##spl", int(state.samples_per_loop), 2, 20)
                        if ch_spl:
                            state.push_undo("Spline resolution")
                            state.samples_per_loop = int(new_spl)
                            state.rebuild_spline_from_params()
                        changed_step, new_step = imgui.slider_float("Keyboard step##spline_keyboard_step", float(state.spline_keyboard_step), 0.001, 0.2, "%.3f")
                        if changed_step:
                            state.spline_keyboard_step = float(new_step)
                        imgui.text(f"Control points: {len(state.flat_pts)}")
                        if int(state.selected_idx) >= 0:
                            imgui.same_line()
                            imgui.text(f"| Selected: #{int(state.selected_idx)}")
                        imgui.separator()
                        imgui.text_disabled("Tiling period vectors (X: spline period, Y: simulation):")
                        changed_px = False
                        px = np.asarray(state.period_offset_x, dtype=np.float32).copy()
                        for axis, label in enumerate(("Period X.x", "Period X.y", "Period X.z")):
                            ch, val = imgui.slider_float(f"{label}##period_x_{axis}", float(px[axis]), -10.0, 10.0, "%.2f")
                            if ch:
                                px[axis] = float(val)
                                changed_px = True
                        if changed_px:
                            state.push_undo("Period X")
                            state.period_offset_x = px
                            _mark_sim_geometry_dirty(state)
                            state.rebuild_spline_mesh(preserve_model_placement=True)
                        changed_py = False
                        py = np.asarray(state.period_offset_y, dtype=np.float32).copy()
                        for axis, label in enumerate(("Period Y.x", "Period Y.y", "Period Y.z")):
                            ch, val = imgui.slider_float(f"{label}##period_y_{axis}", float(py[axis]), -10.0, 10.0, "%.2f")
                            if ch:
                                py[axis] = float(val)
                                changed_py = True
                        if changed_py:
                            state.push_undo("Period Y")
                            state.period_offset_y = py
                            _mark_sim_geometry_dirty(state)
                        if imgui.small_button("Re-derive period from model##period_resync"):
                            state.push_undo("Period resync")
                            state.sync_period_offset_to_model_width()
                            state.sync_period_offset_y_to_row_count()
                            _mark_sim_geometry_dirty(state)
                            state.rebuild_spline_mesh(preserve_model_placement=True)
                imgui.end_tab_item()

            tab_look, _ = imgui.begin_tab_item("Appearance")
            if tab_look:
                _draw_appearance_controls(state, rebuild_current_mesh, is_knit=(fabric == 'knit'))
                imgui.end_tab_item()

            tab_fit, _ = imgui.begin_tab_item("Sim & Scan Fit" if fabric == 'knit' else "Scan Fit")
            if tab_fit:
                if imgui.collapsing_header("Fit to a scan##edit_scan_fit", imgui.TreeNodeFlags_.default_open):
                    _draw_scan_fit(state, fabric, renderer)
                if fabric == 'knit' and imgui.collapsing_header("Yarn Simulation", imgui.TreeNodeFlags_.default_open):
                    _draw_yarn_simulation_panel(state)
                imgui.end_tab_item()

            imgui.end_tab_bar()
        if fabric == 'crochet':
            state.maybe_autosave()
            imgui.end()
            return

    elif scan_active:
        embedded = state.get('embedded_scanner')
        embedded_running = embedded is not None and getattr(embedded, 'running', False) and not getattr(embedded, 'paused', False)
        embedded_paused = embedded is not None and getattr(embedded, 'paused', False)
        running = embedded_running
        single_workflow = str(state.get('scanner_camera_workflow', 'path')) == 'single'
        path_workflow = not single_workflow

        if running:
            imgui.text_colored((0.15, 0.85, 0.35, 1.0), "Scanner is running")
        elif embedded_paused:
            imgui.text_colored((0.95, 0.75, 0.20, 1.0), "Scanner is paused")
        if state.scanner_status:
            imgui.text_wrapped(str(state.scanner_status))

        half_btn = max(90.0, (imgui.get_content_region_avail().x - imgui.get_style().item_spacing.x) * 0.5)
        duplicate_prompt = state.get('_scan_duplicate_prompt')
        if duplicate_prompt:
            imgui.separator()
            imgui.text_colored((0.95, 0.75, 0.20, 1.0), "This scan already exists in the database.")
            total = sum(int(run.get('capture_count', 0)) for run in duplicate_prompt)
            imgui.text_disabled(f"{len(duplicate_prompt)} previous run(s), {total} captures:")
            for run in duplicate_prompt[:4]:
                imgui.text_disabled(
                    f"   {run.get('scan_run', '?')} - {int(run.get('capture_count', 0))} captures"
                    f", {run.get('last_capture_at', '')}"
                )
            if len(duplicate_prompt) > 4:
                imgui.text_disabled(f"   ... and {len(duplicate_prompt) - 4} more")
            imgui.push_style_color(imgui.Col_.button, (0.72, 0.42, 0.16, 1.0))
            if imgui.button("Rescan##scanner_duplicate_rescan", (half_btn, 0)):
                state._scan_duplicate_prompt = None
                start_scanner_process(force=True)
            imgui.pop_style_color()
            imgui.same_line()
            if imgui.button("Cancel##scanner_duplicate_cancel", (half_btn, 0)):
                state._scan_duplicate_prompt = None
                state.scanner_status = "Scan cancelled; the existing database result was kept"
            imgui.separator()
        else:
            if not path_workflow:
                imgui.begin_disabled()
            if imgui.button("Start Scanner Now##scanner_start", (half_btn, 0)):
                start_scanner_process()
            if not path_workflow:
                imgui.end_disabled()
            imgui.same_line()
            if not (running or embedded_paused):
                imgui.begin_disabled()
            if imgui.button("Stop Scanner##scanner_stop", (half_btn, 0)):
                stop_scanner_process()
            if not (running or embedded_paused):
                imgui.end_disabled()
            if embedded_paused and path_workflow:
                if imgui.button("Continue Scanner##scanner_continue", (-1, 0)):
                    embedded.resume()
                    state.scanner_status = "Embedded MuJoCo scanner continued"

        imgui.separator()
        if imgui.collapsing_header("Scan Layout & Capture##scan_layout_section", imgui.TreeNodeFlags_.default_open):
            rows = max(1, int(state.scanner_rows))
            cols = max(1, int(state.scanner_cols))
            changed_r, new_r = imgui.slider_int("Layout rows##scanner_rows", rows, 1, 12)
            changed_c, new_c = imgui.slider_int("Layout columns##scanner_cols", cols, 1, 16)
            changed_a, new_a = imgui.slider_int("Angles per square##scanner_angles", int(state.scanner_angles), 1, 16)
            changed_layout = changed_r or changed_c or changed_a
            if changed_r:
                state.scanner_rows = int(new_r)
            if changed_c:
                state.scanner_cols = int(new_c)
            if changed_a:
                state.scanner_angles = int(new_a)
            if changed_layout:
                state.scanner_preview_rows = max(1, int(state.scanner_rows))
                state.scanner_preview_cols = max(1, int(state.scanner_cols))
                embedded = state.get('embedded_scanner')
                if embedded is not None:
                    try:
                        embedded.close()
                    except Exception as exc:
                        logging.warning("gui.draw_sidebar: %s", exc)
                    state.embedded_scanner = None
                    state.scanner_status = "Scanner layout changed; press Start Scanner to generate new random patterns"
                state.rebuild_spline_mesh(preserve_model_placement=False)
            grid_active = str(state.get('scanner_layout_pattern', 'grid')) == 'grid'
            if imgui.radio_button("Grid##scan_layout_grid", grid_active):
                state.scanner_layout_pattern = 'grid'
                state.rebuild_spline_mesh(preserve_model_placement=False)
            imgui.same_line()
            if imgui.radio_button("Staggered##scan_layout_staggered", not grid_active):
                state.scanner_layout_pattern = 'staggered'
                state.rebuild_spline_mesh(preserve_model_placement=False)
            imgui.same_line()
            if imgui.small_button("Reset Layout##scanner_reset"):
                state.scanner_rows = 3
                state.scanner_cols = 4
                state.scanner_angles = 6
                state.scanner_preview_rows = 3
                state.scanner_preview_cols = 4
                state.scanner_preview_grid_enabled = True
                state.rebuild_spline_mesh(preserve_model_placement=False)
                state.scanner_status = "Scan layout reset"

            capture_mode = str(state.get('scanner_capture_mode', 'natural'))
            focused = capture_mode == "focused"
            mode_is_robot = str(state.scanner_execution_mode) == "robot"
            if not single_workflow:
                changed_mode, new_mode_robot = imgui.checkbox("Run real UR5 robot##scanner_mode", mode_is_robot)
                if changed_mode:
                    state.scanner_execution_mode = "robot" if new_mode_robot else "simulation"
            changed_speed, new_speed = imgui.slider_float("Simulation speed##scanner_speed", float(state.scanner_speed), 0.05, 8.0, "%.2f")
            if changed_speed:
                state.scanner_speed = float(new_speed)
            if not single_workflow:
                changed_dwell, new_dwell = imgui.slider_float("Dwell per view (s)##scanner_dwell", float(state.scanner_dwell), 0.0, 2.0, "%.2f")
                if changed_dwell:
                    state.scanner_dwell = float(new_dwell)
            _, state.scanner_add_camera = imgui.checkbox("Show scanner camera##scanner_camera", bool(state.scanner_add_camera))
            imgui.text("Camera image mode")
            if imgui.radio_button("Single-batch focused##scanner_capture_focused", focused and not single_workflow):
                state.scanner_camera_workflow = 'path'
                state.scanner_capture_mode = "focused"
                embedded = state.get('embedded_scanner')
                if embedded is not None:
                    embedded.args.capture_mode = "focused"
            if imgui.radio_button("Natural robot camera##scanner_capture_natural", not focused and not single_workflow):
                state.scanner_camera_workflow = 'path'
                state.scanner_capture_mode = "natural"
                embedded = state.get('embedded_scanner')
                if embedded is not None:
                    embedded.args.capture_mode = "natural"
            if imgui.radio_button("Single image capture##scanner_capture_single", single_workflow):
                state.scanner_camera_workflow = 'single'
                state.scanner_status = "Single image capture: click a mini-fabric in the 3D view"
            single_workflow = str(state.get('scanner_camera_workflow', 'path')) == 'single'
            if not single_workflow:
                _, state.scanner_save_images = imgui.checkbox("Save scanner images##scanner_images", bool(state.scanner_save_images))
                if bool(state.scanner_save_images):
                    every_view = str(state.scanner_image_every) == "view"
                    changed_every, every_view = imgui.checkbox("Save every angle view##scanner_every", every_view)
                    if changed_every:
                        state.scanner_image_every = "view" if every_view else "station"
                    capture_width, capture_height = _scanner_capture_image_size(state, 'scanner_capture_width')
                    changed_scan_res, scan_capture_width = imgui.slider_int(
                        "Full scan capture resolution##scanner_capture_resolution",
                        capture_width,
                        320,
                        4096,
                    )
                    scan_capture_width = int(np.clip(int(round(float(scan_capture_width) / 64.0) * 64), 320, 4096))
                    scan_capture_width, scan_capture_height = clamp_capture_size(scan_capture_width)
                    if changed_scan_res:
                        state.scanner_capture_width = scan_capture_width
                        embedded = state.get('embedded_scanner')
                        if embedded is not None:
                            embedded.set_capture_resolution(scan_capture_width)
                    imgui.text_disabled(f"Full scanner output: {scan_capture_width} x {scan_capture_height}")
            if single_workflow:
                imgui.text_disabled("Click the fabric, adjust angle/zoom, then capture one image.")
                max_row = max(1, int(state.scanner_rows))
                max_col = max(1, int(state.scanner_cols))
                max_angle = max(1, int(state.scanner_angles))
                state.scanner_single_row = int(np.clip(int(state.get('scanner_single_row', 1)), 1, max_row))
                state.scanner_single_col = int(np.clip(int(state.get('scanner_single_col', 1)), 1, max_col))
                imgui.text_colored(
                    (0.95, 0.80, 0.20, 1.0),
                    f"Target: row {int(state.scanner_single_row)}, col {int(state.scanner_single_col)}",
                )
                changed_single_angle, single_angle = imgui.slider_int(
                    "Camera angle##single_capture_angle",
                    int(np.clip(int(state.get('scanner_single_angle', 1)), 1, max_angle)),
                    1,
                    max_angle,
                )
                if changed_single_angle:
                    state.scanner_single_angle = int(single_angle)
                    embedded = state.get('embedded_scanner')
                    if embedded is not None:
                        embedded.preview_single_target(
                            int(state.get('scanner_single_row', 1)) - 1,
                            int(state.get('scanner_single_col', 1)) - 1,
                            int(state.get('scanner_single_angle', 1)) - 1,
                            float(state.get('scanner_camera_zoom', 1.0)),
                        )
                changed_camera_zoom, camera_zoom = imgui.slider_float(
                    "Robot camera zoom##single_capture_zoom",
                    float(np.clip(float(state.get('scanner_camera_zoom', 1.0)), 0.25, 5.0)),
                    0.25,
                    5.0,
                    "%.2fx",
                )
                if changed_camera_zoom:
                    state.scanner_camera_zoom = float(camera_zoom)
                    embedded = state.get('embedded_scanner')
                    if embedded is not None:
                        embedded.args.camera_zoom = float(camera_zoom)
                        embedded.preview_single_target(
                            int(state.get('scanner_single_row', 1)) - 1,
                            int(state.get('scanner_single_col', 1)) - 1,
                            int(state.get('scanner_single_angle', 1)) - 1,
                            float(state.get('scanner_camera_zoom', 1.0)),
                        )
                if imgui.small_button("Zoom in##single_capture_zoom_in"):
                    state.scanner_camera_zoom = float(min(5.0, float(state.get('scanner_camera_zoom', 1.0)) * 1.15))
                    embedded = state.get('embedded_scanner')
                    if embedded is not None:
                        embedded.preview_single_target(
                            int(state.get('scanner_single_row', 1)) - 1,
                            int(state.get('scanner_single_col', 1)) - 1,
                            int(state.get('scanner_single_angle', 1)) - 1,
                            float(state.get('scanner_camera_zoom', 1.0)),
                        )
                imgui.same_line()
                if imgui.small_button("Zoom out##single_capture_zoom_out"):
                    state.scanner_camera_zoom = float(max(0.25, float(state.get('scanner_camera_zoom', 1.0)) / 1.15))
                    embedded = state.get('embedded_scanner')
                    if embedded is not None:
                        embedded.preview_single_target(
                            int(state.get('scanner_single_row', 1)) - 1,
                            int(state.get('scanner_single_col', 1)) - 1,
                            int(state.get('scanner_single_angle', 1)) - 1,
                            float(state.get('scanner_camera_zoom', 1.0)),
                        )
                current_capture_width, _current_capture_height = _scanner_capture_image_size(state, 'scanner_single_capture_width')
                current_capture_width = int(np.clip(current_capture_width, 320, 4096))
                changed_capture_res, capture_width = imgui.slider_int(
                    "Capture resolution##single_capture_resolution",
                    current_capture_width,
                    320,
                    4096,
                )
                capture_width = int(np.clip(int(round(float(capture_width) / 64.0) * 64), 320, 4096))
                capture_width, capture_height = clamp_capture_size(capture_width)
                if changed_capture_res:
                    state.scanner_single_capture_width = capture_width
                    embedded = state.get('embedded_scanner')
                    if embedded is not None:
                        embedded.set_single_capture_resolution(capture_width)
                imgui.text_disabled(f"Single capture output: {capture_width} x {capture_height}")
                if imgui.button("Preview Selected Target##single_capture_preview", (-1, 0)):
                    embedded = ensure_embedded_robot_viewer(single_capture=True)
                    if embedded is not None:
                        embedded.set_single_capture_resolution(capture_width)
                        embedded.preview_single_target(
                            int(state.get('scanner_single_row', 1)) - 1,
                            int(state.get('scanner_single_col', 1)) - 1,
                            int(state.get('scanner_single_angle', 1)) - 1,
                            float(state.get('scanner_camera_zoom', 1.0)),
                        )
                        state.scanner_status = embedded.status
                single_job = state.get('single_capture_job') or {}
                single_job_busy = str(single_job.get('status', '')) in {
                    'positioning', 'preparing', 'rendering', 'saving', 'cancelling',
                }
                capture_label = "Capture in progress...##single_capture_save" if single_job_busy else "Capture Image##single_capture_save"
                if imgui.button(capture_label, (-1, 0)) and not single_job_busy:
                    embedded = ensure_embedded_robot_viewer(single_capture=True)
                    if embedded is not None:
                        embedded.set_single_capture_resolution(capture_width)
                        embedded.start_single_capture(
                            int(state.get('scanner_single_row', 1)) - 1,
                            int(state.get('scanner_single_col', 1)) - 1,
                            int(state.get('scanner_single_angle', 1)) - 1,
                            float(state.get('scanner_camera_zoom', 1.0)),
                        )
                        state.scanner_status = embedded.status
                if single_job:
                    progress = float(np.clip(single_job.get('progress', 0.0), 0.0, 1.0))
                    imgui.text_disabled(f"Capture progress: {progress * 100.0:.0f}%")
                    message = str(single_job.get('message', '')).strip()
                    if message:
                        imgui.text_wrapped(message)
                    if single_job_busy and str(single_job.get('status')) != 'saving':
                        if imgui.button("Cancel Capture##single_capture_cancel", (-1, 0)):
                            embedded = state.get('embedded_scanner')
                            if embedded is not None:
                                embedded.cancel_single_capture()
                                state.scanner_status = embedded.status
            if not single_workflow and str(state.scanner_execution_mode) == "robot":
                changed_ip, ip = imgui.input_text("Robot IP##scanner_robot_ip", str(state.scanner_robot_ip))
                changed_port, port = imgui.input_int("Robot port##scanner_robot_port", int(state.scanner_robot_port))
                if changed_ip:
                    state.scanner_robot_ip = ip
                if changed_port:
                    state.scanner_robot_port = int(port)

        if imgui.collapsing_header("Random Pattern & Shared Colors##scan_pattern_section"):
            pattern_rows, pattern_cols = _scanner_pattern_dimensions(state)
            repeat_rows, repeat_cols = _scanner_pattern_repeats(state)
            spacing_x, spacing_y = _scanner_repeat_spacing(state)
            batch_texture_width, batch_texture_height = _scanner_batch_texture_size(state)
            imgui.text_disabled(f"Template bitmap: {pattern_rows} x {pattern_cols} (randomizes 0/1 values per sample)")
            changed_rr, new_rr = imgui.slider_int("Image repeats Y##scanner_pattern_repeat_rows", repeat_rows, 1, 64)
            changed_rc, new_rc = imgui.slider_int("Image repeats X##scanner_pattern_repeat_cols", repeat_cols, 1, 64)
            changed_tex_res, new_tex_res = imgui.slider_int(
                "Batch texture resolution##scanner_batch_texture_width",
                batch_texture_width,
                160,
                4096,
            )
            new_tex_res = int(np.clip(int(round(float(new_tex_res) / 32.0) * 32), 160, 4096))
            changed_sx, new_sx = imgui.slider_float(
                "X repeat spacing##scanner_repeat_spacing_x",
                spacing_x,
                0.55,
                1.45,
                "%.2f",
            )
            changed_sy, new_sy = imgui.slider_float(
                "Y repeat spacing##scanner_repeat_spacing_y",
                spacing_y,
                0.55,
                1.45,
                "%.2f",
            )
            imgui.text_disabled(f"Batch texture: {batch_texture_width} x {batch_texture_height} px (1.00 = edge-to-edge)")
            changed_density, new_density = imgui.slider_float(
                "Active stitch probability##scanner_density",
                float(state.get('scanner_pattern_density', 0.62)),
                0.05,
                0.95,
                "%.2f",
            )
            changed_seed, new_seed = imgui.input_int("Random seed##scanner_seed", int(state.get('scanner_random_seed', 1)))
            pattern_rows_now, pattern_cols_now = state._scanner_pattern_size()
            changed_prows, new_prows = imgui.slider_int(
                "Pattern rows##scanner_pattern_rows", pattern_rows_now,
                2, max(2, int(state.config['knit_parameters']['bitmap_rows'])),
            )
            changed_pcols, new_pcols = imgui.slider_int(
                "Pattern columns##scanner_pattern_cols", pattern_cols_now,
                2, int(state.SCANNER_PATTERN_MAX_COLS),
            )
            random_changed = (changed_rr or changed_rc or changed_tex_res or changed_sx or changed_sy
                              or changed_density or changed_seed or changed_prows or changed_pcols)
            if changed_rr:
                state.scanner_pattern_repeat_rows = int(new_rr)
            if changed_rc:
                state.scanner_pattern_repeat_cols = int(new_rc)
            if changed_tex_res:
                state.scanner_batch_texture_width = int(new_tex_res)
            if changed_sx:
                state.scanner_repeat_spacing_x = float(new_sx)
            if changed_sy:
                state.scanner_repeat_spacing_y = float(new_sy)
            if changed_density:
                state.scanner_pattern_density = float(new_density)
            if changed_seed:
                state.scanner_random_seed = int(new_seed)
            if changed_prows:
                state.scanner_pattern_rows = int(new_prows)
            if changed_pcols:
                state.scanner_pattern_cols = int(new_pcols)
            template_bitmap = np.asarray(state._scanner_template().get('bitmap', np.empty((0, 0))), dtype=np.float32)
            if template_bitmap.ndim == 2 and template_bitmap.size:
                tpl_rows, tpl_cols = int(template_bitmap.shape[0]), int(template_bitmap.shape[1])
                pattern_rows_now, pattern_cols_now = state._scanner_pattern_size()
                if (pattern_rows_now, pattern_cols_now) != (tpl_rows, tpl_cols):
                    imgui.text_colored(
                        (0.95, 0.75, 0.20, 1.0),
                        f"Pattern is {pattern_rows_now}x{pattern_cols_now}; saved template is {tpl_rows}x{tpl_cols}.",
                    )
            if random_changed:
                embedded = state.get('embedded_scanner')
                if embedded is not None:
                    try:
                        embedded.close()
                    except Exception as exc:
                        logging.warning("gui.draw_sidebar: %s", exc)
                    state.embedded_scanner = None
                state.scanner_status = "Random pattern settings changed; press Start Scanner to rebuild"
            imgui.text("Preview mode")
            if not state.get('scanner_color_mode'):
                state.scanner_color_mode = 'realistic'
            estimate_mode = str(state.get('scanner_color_mode', 'realistic')) == 'estimated'
            realistic_mode = not estimate_mode
            if imgui.radio_button("Random fabric preview##scanner_preview_realistic", realistic_mode):
                if estimate_mode:
                    state.scanner_color_mode = 'realistic'
                    state.scanner_status = "Showing actual generated fabric preview"
            imgui.same_line()
            if imgui.radio_button("Estimated color picker##scanner_preview_picker", estimate_mode):
                if realistic_mode:
                    state.scanner_color_mode = 'estimated'
                    state.scanner_status = "Click a batch and edit its estimated display color"

            imgui.separator()
            imgui.text("Shared fabric colors")
            palette = _ensure_scanner_shared_colors(state)
            changed_color_count, color_count = imgui.slider_int(
                "Number of colors##scanner_shared_color_count",
                len(palette),
                1,
                12,
            )
            palette_changed = False
            if changed_color_count:
                palette = _ensure_scanner_shared_colors(state, int(color_count))
                palette_changed = True
            for color_idx, color in enumerate(palette):
                changed_shared, new_shared = imgui.color_edit3(
                    f"Color {color_idx + 1}##scanner_shared_color_{color_idx}",
                    (float(color[0]), float(color[1]), float(color[2])),
                )
                if changed_shared:
                    palette[color_idx] = [
                        float(new_shared[0]),
                        float(new_shared[1]),
                        float(new_shared[2]),
                        1.0,
                    ]
                    state.scanner_color_variants = [list(c) for c in palette]
                    palette_changed = True
            if palette_changed:
                state.scanner_estimated_color_overrides = []
                embedded = state.get('embedded_scanner')
                if embedded is not None:
                    try:
                        embedded.close()
                    except Exception as exc:
                        logging.warning("gui.draw_sidebar: %s", exc)
                    state.embedded_scanner = None
                state.scanner_status = "Shared scanner colors changed; press Start Scanner to rebuild"
                state.rebuild_spline_mesh(preserve_model_placement=False)
            if estimate_mode:
                _draw_scanner_batch_color_grid(state)
                rows = max(1, int(state.scanner_rows))
                cols = max(1, int(state.scanner_cols))
                selected = np.asarray(state.get('scanner_selected_cell', [0, 0]), dtype=np.int32).reshape(-1)
                selected_r = int(np.clip(selected[0] if selected.size > 0 else 0, 0, rows - 1))
                selected_c = int(np.clip(selected[1] if selected.size > 1 else 0, 0, cols - 1))
                selected_index = selected_r * cols + selected_c
                estimates = _scanner_estimated_cell_colors(state)
                selected_rgb = [float(v) / 255.0 for v in estimates[selected_index].get("rgb", [128.0, 128.0, 128.0])[:3]]
                imgui.text(f"Selected estimated batch: R{selected_r + 1} C{selected_c + 1}")
                changed_batch_color, batch_color = imgui.color_edit3(
                    "Estimated batch color##scanner_selected_estimated_color",
                    (float(selected_rgb[0]), float(selected_rgb[1]), float(selected_rgb[2])),
                )
                if changed_batch_color:
                    _set_scanner_estimated_batch_color(state, selected_r, selected_c, batch_color)
                    state.scanner_status = f"Updated estimated color for R{selected_r + 1} C{selected_c + 1}"
                    state.rebuild_spline_mesh(preserve_model_placement=False)
                if imgui.small_button("Reset selected estimate##scanner_reset_selected_estimate"):
                    overrides = state.get('scanner_estimated_color_overrides', [])
                    if isinstance(overrides, list) and selected_index < len(overrides):
                        overrides[selected_index] = None
                        state.scanner_estimated_color_overrides = overrides
                        _refresh_embedded_scanner_display_colors(state)
                        state.rebuild_spline_mesh(preserve_model_placement=False)
                imgui.same_line()
                if imgui.small_button("Reset all estimates##scanner_reset_all_estimates"):
                    state.scanner_estimated_color_overrides = []
                    _refresh_embedded_scanner_display_colors(state)
                    state.rebuild_spline_mesh(preserve_model_placement=False)

        if imgui.collapsing_header("Scanner Lighting##scan_lighting_section"):
            lighting_changed = False
            lighting_enabled = bool(state.get('scanner_lighting_enabled', True))
            changed_no_lighting, no_lighting = imgui.checkbox(
                "No Lighting Effect##scanner_no_lighting",
                not lighting_enabled,
            )
            if changed_no_lighting:
                state.scanner_lighting_enabled = not bool(no_lighting)
                lighting_enabled = bool(state.scanner_lighting_enabled)
                lighting_changed = True
            if not lighting_enabled:
                imgui.text_disabled("Flat colors: shadows, highlights, and directional shading disabled.")
                imgui.begin_disabled()
            changed_az, new_az = imgui.slider_float(
                "Sun direction##scanner_light_azimuth",
                float(state.get('scanner_light_azimuth', -35.0)),
                -180.0,
                180.0,
                "%.0f deg",
            )
            changed_el, new_el = imgui.slider_float(
                "Sun height##scanner_light_elevation",
                float(state.get('scanner_light_elevation', 48.0)),
                10.0,
                80.0,
                "%.0f deg",
            )
            changed_si, new_si = imgui.slider_float(
                "Light strength##scanner_light_sun",
                float(state.get('scanner_light_sun_intensity', 0.68)),
                0.20,
                1.20,
                "%.2f",
            )
            changed_sh, new_sh = imgui.slider_float(
                "Soft shadow##scanner_light_shadow",
                float(state.get('scanner_light_shadow', 0.20)),
                0.0,
                0.45,
                "%.2f",
            )
            changed_sheen, new_sheen = imgui.slider_float(
                "Yarn sheen##scanner_light_sheen",
                float(state.get('scanner_light_sheen', 0.025)),
                0.0,
                0.10,
                "%.3f",
            )
            if changed_az:
                state.scanner_light_azimuth = float(new_az)
                lighting_changed = True
            if changed_el:
                state.scanner_light_elevation = float(new_el)
                lighting_changed = True
            if changed_si:
                state.scanner_light_sun_intensity = float(new_si)
                lighting_changed = True
            if changed_sh:
                state.scanner_light_shadow = float(new_sh)
                lighting_changed = True
            if changed_sheen:
                state.scanner_light_sheen = float(new_sheen)
                lighting_changed = True
            if imgui.small_button("Reset scanner lighting##scanner_light_reset"):
                state.scanner_lighting_enabled = True
                state.scanner_light_azimuth = -35.0
                state.scanner_light_elevation = 48.0
                state.scanner_light_sun_intensity = 0.68
                state.scanner_light_shadow = 0.20
                state.scanner_light_sheen = 0.025
                lighting_changed = True
            if not lighting_enabled:
                imgui.end_disabled()
            if lighting_changed:
                embedded = state.get('embedded_scanner')
                if embedded is not None:
                    try:
                        embedded.close()
                    except Exception as exc:
                        logging.warning("gui.draw_sidebar: %s", exc)
                state.embedded_scanner = None
                state.scanner_status = "Scanner lighting changed; press Start Scanner to rebuild"

        if imgui.collapsing_header("Per-Sample RGB Analysis##scan_rgb_section", imgui.TreeNodeFlags_.default_open):
            analysis_ready = embedded is not None and bool(getattr(embedded, "capture_records", []))
            if not analysis_ready:
                imgui.text_disabled("Capture scanner images first.")
                imgui.begin_disabled()
            if imgui.button("Analyze captured RGB##scanner_analyze_rgb", (-1, 0)):
                result = embedded.analyze_captures(save_outputs=True)
                _refresh_embedded_scanner_display_colors(state)
                if result.get("cells"):
                    state.scanner_status = f"RGB analysis saved for {len(result['cells'])} mini-squares"
                else:
                    state.scanner_status = result.get("summary", "No RGB analysis results")
            if not analysis_ready:
                imgui.end_disabled()
            result = getattr(embedded, "analysis_results", None) if embedded is not None else None
            has_analysis = bool(result and result.get("cells"))

            rows = max(1, int(state.scanner_rows))
            cols = max(1, int(state.scanner_cols))
            cells_by_pos = {
                (int(cell["row"]), int(cell["col"])): cell
                for cell in result["cells"]
            } if has_analysis else {}

            selected = np.asarray(state.get('scanner_analysis_selected_cell', state.get('scanner_selected_cell', [0, 0])), dtype=np.int32).reshape(-1)
            selected_r = int(np.clip(selected[0] if selected.size > 0 else 0, 0, rows - 1))
            selected_c = int(np.clip(selected[1] if selected.size > 1 else 0, 0, cols - 1))
            if cells_by_pos and (selected_r, selected_c) not in cells_by_pos:
                selected_r, selected_c = next(iter(cells_by_pos.keys()))
            state.scanner_analysis_selected_cell = [selected_r, selected_c]

            estimate_by_pos = {
                (int(item.get("row", 0)), int(item.get("col", 0))): item
                for item in _scanner_estimated_cell_colors(state)
            }
            for pos, cell in cells_by_pos.items():
                estimate_by_pos[pos] = {
                    **estimate_by_pos.get(pos, {}),
                    "row": int(pos[0]),
                    "col": int(pos[1]),
                    "rgb": [float(v) for v in cell.get("estimated_rgb", [26.0, 26.0, 26.0])[:3]],
                    "active_ratio": float(cell.get("estimated_active_ratio", estimate_by_pos.get(pos, {}).get("active_ratio", 0.0))),
                }

            grid_gap = imgui.get_style().item_spacing.x
            available_w = max(180.0, float(imgui.get_content_region_avail().x))
            panel_w = max(86.0, (available_w - grid_gap) * 0.5) if has_analysis else available_w
            cell_size = max(12.0, min(36.0, (panel_w - max(0, cols - 1) * 3.0) / max(cols, 1)))

            def color_rgba(rgb):
                return (
                    float(np.clip(float(rgb[0]) / 255.0, 0.0, 1.0)),
                    float(np.clip(float(rgb[1]) / 255.0, 0.0, 1.0)),
                    float(np.clip(float(rgb[2]) / 255.0, 0.0, 1.0)),
                    1.0,
                )

            def draw_analysis_grid(title, source_by_pos, rgb_key, id_prefix):
                nonlocal selected_r, selected_c
                imgui.begin_group()
                imgui.text(title)
                imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(3, 3))
                for r in range(rows):
                    for c in range(cols):
                        cell = source_by_pos.get((r, c))
                        rgb = cell.get(rgb_key, [26.0, 26.0, 26.0]) if cell is not None else [26.0, 26.0, 26.0]
                        rgba = color_rgba(rgb)
                        imgui.push_style_color(imgui.Col_.button, rgba)
                        imgui.push_style_color(imgui.Col_.button_hovered, (
                            min(float(rgba[0]) + 0.12, 1.0),
                            min(float(rgba[1]) + 0.12, 1.0),
                            min(float(rgba[2]) + 0.12, 1.0),
                            1.0,
                        ))
                        border_selected = r == selected_r and c == selected_c
                        if border_selected:
                            imgui.push_style_color(imgui.Col_.border, (1.0, 0.78, 0.15, 1.0))
                            imgui.push_style_var(imgui.StyleVar_.frame_border_size, 2.0)
                        clicked = imgui.button(f"##{id_prefix}_{r}_{c}", imgui.ImVec2(cell_size, cell_size))
                        if border_selected:
                            imgui.pop_style_var()
                            imgui.pop_style_color()
                        imgui.pop_style_color(2)
                        if clicked and cell is not None:
                            selected_r, selected_c = r, c
                            state.scanner_analysis_selected_cell = [r, c]
                        if c < cols - 1:
                            imgui.same_line()
                    if r < rows - 1:
                        imgui.spacing()
                imgui.pop_style_var()
                imgui.end_group()

            draw_analysis_grid("Estimated Color Grid", estimate_by_pos, "rgb", "estimated_rgb_result")
            if has_analysis:
                imgui.same_line()
                draw_analysis_grid("Average RGB Color Grid", cells_by_pos, "overall_rgb", "actual_rgb_result")
            else:
                imgui.text_disabled("Average RGB grid appears after scanning and analyzing.")
            state.scanner_analysis_selected_cell = [selected_r, selected_c]

            if has_analysis:
                imgui.text(f"Images analyzed: {int(result.get('image_count', 0))}")
                json_path = result.get("json_path")
                if json_path:
                    imgui.text_disabled(f"Saved: {Path(json_path).name}")
                used_dir = result.get("used_pixels_dir")
                if used_dir:
                    imgui.text_disabled(f"Used-pixel crops: {Path(used_dir).name}/")

                selected_cell = cells_by_pos.get((selected_r, selected_c))
                if selected_cell is not None:
                    rgb = [float(v) for v in selected_cell["overall_rgb"]]
                    estimated_rgb = [float(v) for v in selected_cell.get("estimated_rgb", [0.0, 0.0, 0.0])]
                    estimate_delta = float(selected_cell.get("estimate_actual_delta_rgb", 0.0))
                    imgui.separator()
                    imgui.text(f"Selected mini-square: R{selected_r + 1} C{selected_c + 1}")
                    imgui.color_button(
                        "##selected_overall_rgb",
                        (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, 1.0),
                        imgui.ColorEditFlags_.no_tooltip,
                        imgui.ImVec2(46, 28),
                    )
                    imgui.same_line()
                    imgui.text(f"All views average RGB: {rgb[0]:.0f}, {rgb[1]:.0f}, {rgb[2]:.0f}")
                    imgui.color_button(
                        "##selected_estimated_rgb",
                        (
                            estimated_rgb[0] / 255.0,
                            estimated_rgb[1] / 255.0,
                            estimated_rgb[2] / 255.0,
                            1.0,
                        ),
                        imgui.ColorEditFlags_.no_tooltip,
                        imgui.ImVec2(46, 28),
                    )
                    imgui.same_line()
                    imgui.text(
                        f"Estimated RGB: {estimated_rgb[0]:.0f}, {estimated_rgb[1]:.0f}, {estimated_rgb[2]:.0f}"
                        f" | delta {estimate_delta:.1f}"
                    )
                    fabric_px = int(selected_cell.get("fabric_pixel_count", 0))
                    total_px = int(selected_cell.get("analysis_total_pixels", 0))
                    if fabric_px > 0 and total_px > 0:
                        pct = 100.0 * float(fabric_px) / max(float(total_px), 1.0)
                        masks = ", ".join(str(v) for v in selected_cell.get("analysis_masks", []))
                        imgui.text_disabled(f"Fabric pixels used: {fabric_px} / {total_px} ({pct:.1f}%)")
                        if masks:
                            imgui.text_disabled(f"Mask: {masks}")
                    imgui.text("Per-angle colors")
                    for angle_index, angle_result in enumerate(selected_cell["angles"]):
                        angle_rgb = [float(v) for v in angle_result["rgb"]]
                        imgui.color_button(
                            f"##angle_rgb_{selected_r}_{selected_c}_{angle_index}",
                            (angle_rgb[0] / 255.0, angle_rgb[1] / 255.0, angle_rgb[2] / 255.0, 1.0),
                            imgui.ColorEditFlags_.no_tooltip,
                            imgui.ImVec2(34, 22),
                        )
                        imgui.same_line()
                        imgui.text(
                            f"{angle_result['angle']}: {angle_rgb[0]:.0f}, {angle_rgb[1]:.0f}, {angle_rgb[2]:.0f}"
                        )
        _maybe_persist_scanner_state(state)

    elif puzzle_active:
        imgui.text_disabled("Tile real geometry, capture a region, and verify seamless repeat edges.")
        if imgui.collapsing_header("1. Stitch Pattern & Copies##puzzle_pattern", imgui.TreeNodeFlags_.default_open):
            _draw_bitmap_editor(state, id_suffix="_puzzle")
            imgui.separator()
            copies_x, copies_y = _puzzle_target_copies(state)
            changed_px, new_px = imgui.slider_int("Copies X##puzzle_copies_x", copies_x, 1, 9)
            released_px = imgui.is_item_deactivated_after_edit()
            changed_py, new_py = imgui.slider_int("Copies Y##puzzle_copies_y", copies_y, 1, 9)
            released_py = imgui.is_item_deactivated_after_edit()
            if changed_px:
                if int(new_px) % 2 == 0:
                    new_px += 1
                state.puzzle_copies_x = int(np.clip(new_px, 1, 9))
            if changed_py:
                if int(new_py) % 2 == 0:
                    new_py += 1
                state.puzzle_copies_y = int(np.clip(new_py, 1, 9))
            if changed_px or changed_py:
                state.status_msg = (
                    f"Release to apply {int(state.puzzle_copies_x)} x "
                    f"{int(state.puzzle_copies_y)} geometry"
                )
            if released_px or released_py:
                _puzzle_apply_geometry_copies(state, preserve=True)
                state.status_msg = f"Puzzle geometry set to {int(state.puzzle_copies_x)} x {int(state.puzzle_copies_y)}"
            half_p = max(80.0, (imgui.get_content_region_avail().x - imgui.get_style().item_spacing.x) * 0.5)
            if imgui.button("Apply 5 x 5##puzzle_apply_5x5", (half_p, 0)):
                state.puzzle_copies_x = 5
                state.puzzle_copies_y = 5
                _puzzle_apply_geometry_copies(state, preserve=True)
                state.status_msg = "Puzzle geometry set to 5 x 5"
            imgui.same_line()
            if imgui.button("Apply Copies##puzzle_apply", (half_p, 0)):
                _puzzle_apply_geometry_copies(state, preserve=True)
                state.status_msg = "Puzzle geometry copies applied"

        if imgui.collapsing_header("2. Capture Frame & Landmarks##puzzle_capture_section", imgui.TreeNodeFlags_.default_open):
            rx, ry, rw, rh = _puzzle_capture_rect(state)
            changed_rx, rx = imgui.slider_float("Frame X##puzzle_capture_rect_x", rx, 0.0, 0.95, "%.2f")
            changed_ry, ry = imgui.slider_float("Frame Y##puzzle_capture_rect_y", ry, 0.0, 0.95, "%.2f")
            changed_rw, rw = imgui.slider_float("Frame width##puzzle_capture_rect_w", rw, 0.05, 1.0, "%.2f")
            changed_rh, rh = imgui.slider_float("Frame height##puzzle_capture_rect_h", rh, 0.05, 1.0, "%.2f")
            if changed_rx or changed_ry or changed_rw or changed_rh:
                _set_puzzle_capture_rect(state, [rx, ry, rw, rh])
            changed_count, new_count = imgui.slider_int("Detected colors##puzzle_detect_color_count", int(state.get('puzzle_detect_color_count', 6)), 1, 12)
            if changed_count:
                state.puzzle_detect_color_count = int(new_count)
            if imgui.button("Capture geometry preview##puzzle_capture", (-1, 0)):
                image, points, landmarks = _puzzle_capture_geometry_image(state, renderer)
                state.puzzle_capture_image = image
                state.puzzle_projected_points = points
                state.puzzle_edge_landmarks = landmarks
                state.puzzle_detected_colors = _puzzle_detect_colors(image, int(state.get('puzzle_detect_color_count', 6))) if image is not None else []
                state.puzzle_capture_texture_key = None
                if image is not None:
                    state.status_msg = f"Puzzle capture: {len(points)} control points, {len(landmarks)} edge landmarks"
                else:
                    state.status_msg = "Puzzle capture failed: 3D viewport has not rendered yet"
            image = state.get('puzzle_capture_image', None)
            if image is not None:
                if imgui.button("Save puzzle capture##puzzle_save_capture", (-1, 0)):
                    output_dir = Path(state.project_root) / "scanner_data" / "puzzle_mode"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"puzzle_capture_{int(time.time())}.png"
                    image.save(output_path)
                    state.status_msg = f"Saved {output_path.name}"
                imgui.text(f"Captured image: {image.size[0]} x {image.size[1]} | pts: {len(state.get('puzzle_projected_points', []))} | landmarks: {len(state.get('puzzle_edge_landmarks', []))}")
                colors = state.get('puzzle_detected_colors', [])
                if colors:
                    for idx, item in enumerate(colors):
                        rgb = item.get("rgb", [0, 0, 0])
                        imgui.color_button(
                            f"##puzzle_color_{idx}",
                            (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, 1.0),
                            imgui.ColorEditFlags_.no_tooltip,
                            imgui.ImVec2(28, 18),
                        )
                        imgui.same_line()
                        imgui.text(f"{rgb[0]}, {rgb[1]}, {rgb[2]} ({int(item.get('count', 0))} px)")

        if imgui.collapsing_header("3. Seamless Tile & Glue Preview##puzzle_glue_section", imgui.TreeNodeFlags_.default_open):
            glue_cols = int(state.get('puzzle_glue_cols', 3))
            glue_rows = int(state.get('puzzle_glue_rows', 3))
            changed_gc, new_gc = imgui.slider_int("Glue copies X##puzzle_glue_cols", glue_cols, 1, 100)
            changed_gr, new_gr = imgui.slider_int("Glue copies Y##puzzle_glue_rows", glue_rows, 1, 100)
            if changed_gc:
                state.puzzle_glue_cols = int(new_gc)
            if changed_gr:
                state.puzzle_glue_rows = int(new_gr)
            changed_hl, new_hl = imgui.checkbox("Highlight tile edges##puzzle_glue_highlight", bool(state.get('puzzle_glue_highlight_edges', False)))
            if changed_hl:
                state.puzzle_glue_highlight_edges = bool(new_hl)
            if imgui.button("Build seamless tile##puzzle_build_glue", (-1, 0)):
                tile, glued, info = _puzzle_build_seamless_tile(
                    state, renderer,
                    int(state.get('puzzle_glue_cols', 3)),
                    int(state.get('puzzle_glue_rows', 3)),
                )
                state.puzzle_tile_image = tile
                state.puzzle_glued_image = glued
                state.puzzle_glue_info = info
                state.puzzle_glued_texture_key = None
                if glued is not None:
                    state.status_msg = (
                        f"Glued {info.get('cols', 0)} x {info.get('rows', 0)} "
                        f"tiles ({info.get('tile_w', 0)} x {info.get('tile_h', 0)} px each)"
                    )
                else:
                    state.status_msg = "Seamless tile build failed: capture the 3D viewport first"
            glued_image = state.get('puzzle_glued_image', None)
            info = state.get('puzzle_glue_info', {}) or {}
            if glued_image is not None:
                skew_x = float(info.get('skew_x_px', 0.0))
                skew_y = float(info.get('skew_y_px', 0.0))
                imgui.text(f"Tile: {info.get('tile_w', 0)} x {info.get('tile_h', 0)} px | Period: {info.get('x_period_world', 0.0):.2f} / {info.get('y_period_world', 0.0):.2f}")
                skew_color = (1.0, 0.55, 0.2, 1.0) if (abs(skew_x) > 1.5 or abs(skew_y) > 1.5) else (0.6, 0.9, 0.6, 1.0)
                imgui.text_colored(skew_color, f"Seam skew: x={skew_x:.2f}px, y={skew_y:.2f}px")
                if imgui.button("Save seamless tile##puzzle_save_glue", (-1, 0)):
                    output_dir = Path(state.project_root) / "scanner_data" / "puzzle_mode"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    stamp = int(time.time())
                    tile_image = state.get('puzzle_tile_image', None)
                    if tile_image is not None:
                        tile_image.save(output_dir / f"puzzle_tile_{stamp}.png")
                    glued_image.save(output_dir / f"puzzle_glued_{stamp}.png")
                    state.status_msg = f"Saved puzzle_tile_{stamp}.png and puzzle_glued_{stamp}.png"

    if (
        str(state.get('app_mode', 'edit')) == 'scan'
        and str(state.get('scanner_execution_mode', 'simulation')) == 'simulation'
        and state.get('embedded_scanner') is None
    ):
        ensure_embedded_robot_viewer(auto_start=False)

    embedded = state.get('embedded_scanner')
    if str(state.get('app_mode', 'edit')) == 'scan' and embedded is not None:
        scan_vp = str(state.get('_scan_vp_mode', 'split'))
        embedded.preview_visible = scan_vp in ('split', 'camera')
        try:
            needs_update = (
                getattr(embedded, 'running', False)
                or getattr(embedded, 'paused', False)
                or getattr(embedded, 'single_target_active', False)
                or embedded.texture is None
            )
            if needs_update:
                embedded.update()
                state.scanner_status = embedded.status
        except Exception as exc:
            state.scanner_status = f"Embedded scanner error: {exc}"
            try:
                embedded.close()
            except Exception as cleanup_error:
                logging.warning("Scanner cleanup: %s", cleanup_error)
            state.embedded_scanner = None

    if state.status_msg:
        imgui.separator()
        imgui.text_colored((0.4, 0.9, 0.4, 1.0), str(state.status_msg))
    state.maybe_autosave()
    imgui.end()


def _draw_scan_simulator_canvas(state, embedded, window, height=0):
    if embedded is None:
        imgui.text_disabled("Simulator not initialized. Press Start Scan in Knitting Control.")
        return
    if embedded.texture is None:
        imgui.text_disabled("Waiting for simulator frame...")
        return
    view_changed = False
    imgui.text("UR5 Robot Simulator")
    imgui.same_line()
    imgui.set_next_item_width(90)
    changed_zoom, zoom = imgui.slider_float("##scanner_view_zoom", float(embedded.view_zoom), 0.45, 2.50, "%.2fx")
    if changed_zoom:
        embedded.set_zoom(zoom)
        view_changed = True
    imgui.same_line()
    if imgui.small_button("Reset##scanner_view"):
        embedded.reset_view()
        view_changed = True
    imgui.same_line()
    if imgui.small_button("Rot L##scanner_view"):
        embedded.rotate_view(delta_azimuth=-12.0)
        view_changed = True
    imgui.same_line()
    if imgui.small_button("Rot R##scanner_view"):
        embedded.rotate_view(delta_azimuth=12.0)
        view_changed = True
    imgui.same_line()
    if imgui.small_button("Tilt +##scanner_view"):
        embedded.rotate_view(delta_elevation=8.0)
        view_changed = True
    imgui.same_line()
    if imgui.small_button("Tilt -##scanner_view"):
        embedded.rotate_view(delta_elevation=-8.0)
        view_changed = True
    avail = imgui.get_content_region_avail()
    main_w = max(1, int(avail.x))
    main_h = max(1, int(height if height > 0 else avail.y - 24))
    rect = draw_fitted_texture(embedded.texture.glo, embedded.width, embedded.height, main_w, main_h, flip_y=True)
    if rect is not None:
        x, y, w, h = rect
        io = imgui.get_io()
        mx, my = float(io.mouse_pos.x), float(io.mouse_pos.y)
        image_hovered = (x <= mx <= x + w and y <= my <= y + h and imgui.is_window_hovered())
        if image_hovered:
            if float(io.mouse_wheel) != 0.0:
                embedded.set_zoom(embedded.view_zoom * float(np.exp(float(io.mouse_wheel) * 0.16)))
                view_changed = True
            if window is not None:
                lmb = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
                rmb = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
                mmb = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
                dx, dy = float(io.mouse_delta.x), float(io.mouse_delta.y)
                if (abs(dx) > 0.0 or abs(dy) > 0.0) and lmb:
                    embedded.orbit_view(dx, dy)
                    view_changed = True
                elif (abs(dx) > 0.0 or abs(dy) > 0.0) and (rmb or mmb):
                    embedded.pan_view(dx, dy)
                    view_changed = True
        imgui.text_disabled("Wheel: zoom | LMB drag: rotate | RMB/MMB drag: pan")
    if view_changed:
        embedded._render_frame()


def _draw_scan_gripper_camera_canvas(embedded, height=0):
    imgui.text("Live Gripper Camera")
    imgui.same_line()
    imgui.text_disabled("(UR5 wrist camera feed)")
    avail = imgui.get_content_region_avail()
    preview_w = max(120, int(avail.x))
    preview_h = max(90, int(height if height > 0 else avail.y - 8))
    if embedded is None:
        imgui.dummy((preview_w, preview_h))
        imgui.text_disabled("Camera preview waiting for scan. Press Start Scan.")
    elif embedded.camera_texture is None:
        imgui.dummy((preview_w, preview_h))
        imgui.text_disabled("Camera preview waiting for the first scanner frame.")
    else:
        draw_fitted_texture(
            embedded.camera_texture.glo,
            embedded.camera_preview_width,
            embedded.camera_preview_height,
            preview_w,
            preview_h,
            flip_y=True,
        )


def _draw_puzzle_inspector_canvas(state, renderer, height=0):
    imgui.text("Captured Frame & Landmarks")
    imgui.same_line()
    imgui.text_disabled("(White: loop pts | Green: edge landmarks | Click to inspect RGB)")
    image = state.get('puzzle_capture_image', None)
    if image is None:
        imgui.text_disabled("No puzzle frame captured yet. Click 'Capture geometry preview' in Knitting Control.")
        return
    texture = _upload_puzzle_texture(state, renderer, 'capture')
    if texture is None:
        return
    avail = imgui.get_content_region_avail()
    w = max(120, int(avail.x))
    h = max(120, int(height if height > 0 else avail.y - 36))
    _draw_puzzle_capture_widget(state, image, texture, w, h)


def _draw_puzzle_glue_canvas(state, renderer, height=0):
    imgui.text("Seamless Glued Tile")
    glued_image = state.get('puzzle_glued_image', None)
    info = state.get('puzzle_glue_info', {}) or {}
    if glued_image is None:
        imgui.text_disabled("No seamless tile built yet. Click 'Build seamless tile' in Knitting Control.")
        return
    glue_texture = _upload_puzzle_texture(state, renderer, 'glued')
    if glue_texture is None:
        return
    avail = imgui.get_content_region_avail()
    w = max(120, int(avail.x))
    h = max(120, int(height if height > 0 else avail.y - 8))
    glue_rect = draw_fitted_texture(glue_texture.glo, glued_image.size[0], glued_image.size[1], w, h, flip_y=True)
    if bool(state.get('puzzle_glue_highlight_edges', False)):
        _draw_puzzle_glue_edge_overlay(glued_image, glue_rect, info)


def _draw_orbit_canvas(state, window):
    orbit_renderer = getattr(state, 'orbit_renderer', None)
    orbit_camera = getattr(state, 'orbit_camera', None)
    if orbit_renderer is None or orbit_camera is None:
        imgui.text_disabled("No orbit renderer attached.")
        return
    imgui.text("3D Orbit Grid")
    imgui.same_line()
    if imgui.small_button("Reset Orbit##orbit_reset"):
        orbit_camera.az = float(np.radians(-30.0))
        orbit_camera.el = float(np.radians(28.0))
        orbit_camera.dist = 32.0
    imgui.same_line()
    imgui.text_disabled("LMB drag: orbit | Wheel: zoom")
    avail_x, avail_y = imgui.get_content_region_avail()
    disp_w = max(1, int(avail_x))
    disp_h = max(1, int(avail_y))
    orbit_renderer.resize(disp_w, disp_h)
    orbit_camera.target = (np.asarray(state.mesh_center, dtype=np.float32)
                           + np.asarray(state.model_t, dtype=np.float32)).astype(np.float32)
    model_mat = state.current_model_matrix()
    mvp = (orbit_camera.mvp(disp_w, disp_h) @ model_mat).astype(np.float32)
    mv = (orbit_camera.mv(disp_w, disp_h) @ model_mat).astype(np.float32)
    orbit_renderer.render(
        mvp, mv,
        state.get_material_uniforms(),
        camera=orbit_camera,
        visible_rows=state.row_visible,
        model_mat=model_mat,
        show_grid=True,
    )
    draw_fitted_texture(
        orbit_renderer.texture_id,
        disp_w,
        disp_h,
        avail_x,
        avail_y,
        flip_y=True,
    )
    is_hovered = imgui.is_item_hovered()
    mx, my = imgui.get_mouse_pos()
    if is_hovered:
        wheel = float(imgui.get_io().mouse_wheel)
        if wheel != 0.0:
            zoom_factor = float(np.exp(wheel * 0.12))
            orbit_camera.dist = float(np.clip(float(orbit_camera.dist) / zoom_factor, 1.0, 200.0))
    if window is not None and glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
        if state.get('orbit_dragging', False):
            prev = state.get('prev_orbit_mouse', None)
            if prev is not None:
                orbit_camera.orbit(mx - prev[0], my - prev[1])
            state.prev_orbit_mouse = (mx, my)
        elif is_hovered:
            state.orbit_dragging = True
            state.prev_orbit_mouse = (mx, my)
    else:
        state.orbit_dragging = False
        state.prev_orbit_mouse = None


def draw_viewport(state, renderer, ref_tex, window):
    reset_vp = bool(state.get('_layout_reset_viewport', False)) or not bool(state.get('_viewport_sized_once', False))
    vp_cond = imgui.Cond_.always if reset_vp else imgui.Cond_.first_use_ever
    imgui.set_next_window_pos((390, 20), cond=vp_cond)
    imgui.set_next_window_size((1190, 820), cond=vp_cond)
    state._viewport_sized_once = True
    if state.get('_layout_reset_viewport', False):
        state._layout_reset_viewport = False
    imgui.begin("3D View", flags=imgui.WindowFlags_.no_scroll_with_mouse)

    current_mode = str(state.get('app_mode', 'edit'))
    if current_mode == 'database':
        draw_database_viewport(state, renderer)
        imgui.end()
        return

    if current_mode == 'model':
        gui_model.draw_model_viewport(state, renderer)
        imgui.end()
        return

    if current_mode == 'ur5':
        show_scene = bool(state.get('ur5_show_scene', False))
        imgui.text("UR5 Workspace:")
        imgui.same_line()
        if imgui.radio_button("Camera & Capture Path##ur5_vp_cam", not show_scene):
            state.ur5_show_scene = False
            show_scene = False
        imgui.same_line()
        if imgui.radio_button("3D Knit Layout##ur5_vp_scene", show_scene):
            state.ur5_show_scene = True
            show_scene = True
        imgui.separator()
        if not show_scene:
            gui_ur5.draw_ur5_viewport(state, renderer, window)
            imgui.end()
            return

    split_right_mode = None
    if current_mode == 'edit':
        edit_vp = str(state.get('_edit_vp_mode', 'split'))
        imgui.text("Edit View:")
        for key, label in (('split', "Split: 3D + Orbit Grid"), ('3d', "3D Fabric Only"), ('orbit', "3D Orbit Grid Only")):
            imgui.same_line()
            if imgui.radio_button(f"{label}##edit_vp_{key}", edit_vp == key):
                state._edit_vp_mode = key
                edit_vp = key
        if edit_vp == 'orbit':
            imgui.separator()
            _draw_orbit_canvas(state, window)
            imgui.end()
            return
        if edit_vp == 'split':
            split_right_mode = 'edit_orbit'
    elif current_mode == 'scan':
        scan_vp = str(state.get('_scan_vp_mode', 'split'))
        imgui.text("Scan View:")
        for key, label in (
            ('split', "Split: Fabric + Robot"),
            ('fabric', "3D Fabric Grid"),
            ('simulator', "UR5 Robot Simulator"),
            ('camera', "Gripper Camera"),
        ):
            imgui.same_line()
            if imgui.radio_button(f"{label}##scan_vp_{key}", scan_vp == key):
                state._scan_vp_mode = key
                scan_vp = key
        embedded = state.get('embedded_scanner')
        if embedded is not None:
            embedded.preview_visible = scan_vp in ('split', 'camera')
        if scan_vp == 'simulator':
            imgui.separator()
            _draw_scan_simulator_canvas(state, embedded, window)
            imgui.end()
            return
        if scan_vp == 'camera':
            imgui.separator()
            _draw_scan_gripper_camera_canvas(embedded)
            imgui.end()
            return
        if scan_vp == 'split':
            split_right_mode = 'scan_robot'
    elif current_mode == 'puzzle':
        renderer = state.puzzle_renderer
        puzzle_vp = str(state.get('_puzzle_vp_mode', 'split'))
        imgui.text("Puzzle View:")
        for key, label in (
            ('split', "Split: 3D + Previews"),
            ('3d', "3D Puzzle Only"),
            ('capture', "Captured Frame & Landmarks"),
            ('glue', "Seamless Glued Tile"),
        ):
            imgui.same_line()
            if imgui.radio_button(f"{label}##puzzle_vp_{key}", puzzle_vp == key):
                state._puzzle_vp_mode = key
                puzzle_vp = key
        if puzzle_vp == 'capture':
            imgui.separator()
            _draw_puzzle_inspector_canvas(state, renderer)
            imgui.end()
            return
        if puzzle_vp == 'glue':
            imgui.separator()
            _draw_puzzle_glue_canvas(state, renderer)
            imgui.end()
            return
        if puzzle_vp == 'split':
            split_right_mode = 'puzzle_previews'

    imgui.text(
        "3D Puzzle Viewport"
        if current_mode == 'puzzle'
        else "3D Viewport"
    )
    imgui.same_line()
    if imgui.small_button("Center##vp_center"):
        state.center_model_in_viewport()
    imgui.same_line()
    if imgui.small_button("Front View##vp_front"):
        state.camera.az = 0.0
        state.camera.el = 0.0
        state.center_model_in_viewport()
    imgui.same_line()
    if imgui.small_button("3D Angle##vp_angle"):
        state.camera.az = float(np.radians(-30.0))
        state.camera.el = float(np.radians(28.0))
    imgui.same_line()
    if imgui.small_button("Reset Transform##vp_reset_transform"):
        state.model_t = np.zeros(3, dtype=np.float32)
        state.model_rot = np.zeros(3, dtype=np.float32)
        state.model_scale = np.ones(3, dtype=np.float32)
        state.center_model_in_viewport()
    imgui.same_line()
    imgui.text_disabled("LMB: move | RMB/MMB: orbit | Wheel: zoom")

    total_avail_x, avail_y = imgui.get_content_region_avail()
    avail_x = max(240.0, (total_avail_x - 8.0) * 0.52) if split_right_mode else total_avail_x
    disp_w = max(1, int(avail_x))
    disp_h = max(1, int(avail_y))
    renderer.resize(disp_w, disp_h)
    draw_pos  = imgui.get_cursor_screen_pos()
    state.vp_origin = np.array([draw_pos.x, draw_pos.y], dtype=np.float32)
    state.vp_scale  = 1.0

    # Build model matrix and render FBO
    model_mat = state.current_model_matrix()
    mvp = (state.camera.mvp(disp_w, disp_h) @ model_mat).astype(np.float32)
    mv  = (state.camera.mv(disp_w, disp_h)  @ model_mat).astype(np.float32)
    bg_zoom = state.camera.zoom_factor()
    # UR5 Robot Mode shows the same scan preview as Scan Mode, and for the same
    # reason: the viewport is the layout being handed to the robot. Sharing this
    # flag keeps the two previews identical and, just as importantly, keeps the
    # model-editing handles off -- editing the fabric mid-scan would leave the
    # plan the arm is executing describing geometry that no longer exists.
    scanner_stage_active = str(state.get('app_mode', 'edit')) in ('scan', 'ur5')
    scanner_estimate_mode = scanner_stage_active and str(state.get('scanner_color_mode', 'realistic')) == 'estimated'
    edit_controls_active = not scanner_stage_active
    if scanner_stage_active:
        state.bbox_active_handle = -1
        state.bbox_hover_handle = -1
        state.spline_grab_active = False
        state.radius_grab_active = False
        state.gizmo_edit_active = False

    render_hover_idx = state.hover_idx
    render_selected_idx = state.selected_idx
    visible_ctrl_indices = np.empty((0,), dtype=np.int32)
    visible_ctrl_index_map = {}
    if state.mode == 'spline' and edit_controls_active:
        n_real_total = len(state.flat_pts)
        n_rows_total = len(state.ctrl_rows)
        real_chunks = []
        virtual_indices = []
        for row_idx, row in enumerate(state.ctrl_rows):
            if not state.row_visible[row_idx]:
                continue
            start = state._row_starts[row_idx]
            end = start + len(row)
            real_chunks.append(np.arange(start, end, dtype=np.int32))
            # Two period handles per row, matching AppState.flat_pts_all's
            # [real | X handles | Y handles] layout.
            virtual_indices.append(n_real_total + row_idx)
            virtual_indices.append(n_real_total + n_rows_total + row_idx)
        if real_chunks:
            visible_ctrl_indices = np.concatenate(real_chunks + [np.array(virtual_indices, dtype=np.int32)])
            visible_ctrl_pts = state.flat_pts_all[visible_ctrl_indices]
        else:
            visible_ctrl_pts = np.empty((0, 3), dtype=np.float32)
        renderer.set_ctrl_pts(visible_ctrl_pts)
        visible_ctrl_index_map = {
            int(flat_idx): int(local_idx)
            for local_idx, flat_idx in enumerate(visible_ctrl_indices.tolist())
        }
        render_hover_idx = visible_ctrl_index_map.get(int(state.hover_idx), -1)
        render_selected_idx = visible_ctrl_index_map.get(int(state.selected_idx), -1)
    else:
        renderer.set_ctrl_pts(np.empty((0, 3), dtype=np.float32))
        render_hover_idx = -1
        render_selected_idx = -1

    bg_uniforms = {
        'bg_scale_x':  state.ref_bg_scale[0] * bg_zoom,
        'bg_scale_y':  state.ref_bg_scale[1] * bg_zoom,
        'bg_rotation': state.ref_bg_rotation,
        'bg_offset_x': state.ref_bg_offset[0],
        'bg_offset_y': state.ref_bg_offset[1],
        'vp_aspect':   disp_w / disp_h,
        'img_aspect':  ref_tex.width / ref_tex.height if ref_tex is not None else 1.0,
    }

    def sample_reference_color(view_x, view_y):
        ref_pixels = state.get('reference_image_pixels', None)
        if ref_pixels is None:
            return None
        pixels = np.asarray(ref_pixels, dtype=np.float32)
        if pixels.ndim != 3 or pixels.shape[0] <= 0 or pixels.shape[1] <= 0:
            return None

        vp_aspect = float(bg_uniforms.get('vp_aspect', 1.0))
        img_aspect = float(bg_uniforms.get('img_aspect', 1.0))
        scale_x = max(float(bg_uniforms.get('bg_scale_x', 1.0)), 0.01)
        scale_y = max(float(bg_uniforms.get('bg_scale_y', 1.0)), 0.01)
        rotation = float(bg_uniforms.get('bg_rotation', 0.0))
        offset_x = float(bg_uniforms.get('bg_offset_x', 0.0))
        offset_y = float(bg_uniforms.get('bg_offset_y', 0.0))

        c = np.array([
            float(view_x) / max(float(disp_w), 1.0) - 0.5,
            0.5 - float(view_y) / max(float(disp_h), 1.0),
        ], dtype=np.float32)
        iso = np.array([c[0] * vp_aspect, c[1]], dtype=np.float32)
        cr, sr = np.cos(rotation), np.sin(rotation)
        rot = np.array([cr * iso[0] - sr * iso[1], sr * iso[0] + cr * iso[1]], dtype=np.float32)
        uv = np.array([
            rot[0] / (img_aspect * scale_x) - offset_x + 0.5,
            rot[1] / scale_y - offset_y + 0.5,
        ], dtype=np.float32)
        if np.any(uv < 0.0) or np.any(uv > 1.0):
            return None

        h, w = pixels.shape[:2]
        px = int(np.clip(round(float(uv[0]) * (w - 1)), 0, w - 1))
        py = int(np.clip(round((1.0 - float(uv[1])) * (h - 1)), 0, h - 1))
        return pixels[py, px, :3].astype(np.float32)

    material_uniforms = _scanner_material_uniforms(state) if scanner_stage_active else dict(state.get_material_uniforms())

    renderer.render(
        mvp, mv,
        material_uniforms,
        render_hover_idx, render_selected_idx,
        hover_mesh_idx=state.hover_mesh_idx,
        selected_mesh_idx=state.selected_mesh_idx,
        visible_rows=np.zeros(max(1, len(state.row_visible)), dtype=bool) if scanner_estimate_mode else state.row_visible,
        bg_tex      = ref_tex if _reference_overlay_active(state) else None,
        bg_alpha    = state.ref_bg_alpha if _reference_overlay_active(state) else 0.0,
        bg_uniforms = bg_uniforms,
        camera      = state.camera,
        n_real_pts  = sum(len(row) for r_idx, row in enumerate(state.ctrl_rows) if state.row_visible[r_idx])
    )

    # Display FBO
    drawn_rect = draw_fitted_texture(
        renderer.texture_id,
        disp_w,
        disp_h,
        avail_x,
        avail_y,
        flip_y=True,
        zoom=1.0,
        pan=state.viewport_pan,
    )
    if drawn_rect is not None:
        origin_x, origin_y, draw_w, _ = drawn_rect
        state.vp_origin = np.array([origin_x, origin_y], dtype=np.float32)
        state.vp_scale = float(draw_w / max(float(disp_w), 1.0))
        if str(state.get('app_mode', 'edit')) == 'puzzle':
            x, y, w, h = drawn_rect
            rx, ry, rw, rh = _puzzle_capture_rect(state)
            fx0 = x + rx * w
            fy0 = y + ry * h
            fx1 = x + (rx + rw) * w
            fy1 = y + (ry + rh) * h
            dl = imgui.get_window_draw_list()
            shade = imgui.get_color_u32((0.0, 0.0, 0.0, 0.34))
            frame = imgui.get_color_u32((1.0, 0.78, 0.08, 1.0))
            dl.add_rect_filled(imgui.ImVec2(x, y), imgui.ImVec2(x + w, fy0), shade)
            dl.add_rect_filled(imgui.ImVec2(x, fy1), imgui.ImVec2(x + w, y + h), shade)
            dl.add_rect_filled(imgui.ImVec2(x, fy0), imgui.ImVec2(fx0, fy1), shade)
            dl.add_rect_filled(imgui.ImVec2(fx1, fy0), imgui.ImVec2(x + w, fy1), shade)
            dl.add_rect(imgui.ImVec2(fx0, fy0), imgui.ImVec2(fx1, fy1), frame, 0.0, 2.5, 0)
            label_bg = imgui.get_color_u32((0.0, 0.0, 0.0, 0.70))
            dl.add_rect_filled(imgui.ImVec2(fx0, fy0 - 22.0), imgui.ImVec2(fx0 + 148.0, fy0), label_bg)
            dl.add_text(imgui.ImVec2(fx0 + 6.0, fy0 - 18.0), frame, "Puzzle capture area")
    is_hovered = imgui.is_item_hovered()
    state.mouse_in_vp = is_hovered
    mx, my = imgui.get_mouse_pos()
    viewport_scale = max(float(state.vp_scale), 1e-6)
    lx = (mx - state.vp_origin[0]) / viewport_scale
    ly = (my - state.vp_origin[1]) / viewport_scale
    def projected_mesh_bounds(model_matrix=None):
        if model_matrix is None:
            model_matrix = model_mat

        # In spline mode, derive the bbox from visible control points for stable,
        # predictable resize behavior.
        if state.mode == 'spline' and len(visible_ctrl_indices) > 0 and not scanner_stage_active:
            ctrl_pts = state.flat_pts_all[visible_ctrl_indices]
            world_pts = transform_points(ctrl_pts, model_matrix)
            view_proj = state.camera.proj(disp_w, disp_h) @ state.camera.view()
            screen, valid, _ = project_to_screen(world_pts, view_proj, disp_w, disp_h)
            if not np.any(valid):
                return None
            all_pts = screen[valid]
            x_min, y_min = np.min(all_pts, axis=0)
            x_max, y_max = np.max(all_pts, axis=0)
            pad = float(state.config.get("ui", {}).get("bbox_padding", 20.0))
            x_min = float(x_min - pad)
            y_min = float(y_min - pad)
            x_max = float(x_max + pad)
            y_max = float(y_max + pad)
            if x_max - x_min < 12.0 or y_max - y_min < 12.0:
                return None
            return x_min, y_min, x_max, y_max

        if not renderer.mesh_pick_data:
            return None
        view_proj = state.camera.proj(disp_w, disp_h) @ state.camera.view()
        pts_2d = []
        for verts, row_idx in renderer.mesh_pick_data:
            if state.row_visible is not None and len(state.row_visible) > 0:
                base_row_idx = int(row_idx) % len(state.row_visible)
                if not bool(state.row_visible[base_row_idx]):
                    continue
            elif state.row_visible is not None:
                continue
            if len(verts) == 0:
                continue
            stride = max(1, len(verts) // 300)
            sample = verts[::stride]
            world_pts = transform_points(sample, model_matrix)
            screen, valid, _ = project_to_screen(world_pts, view_proj, disp_w, disp_h)
            if not np.any(valid):
                continue
            pts_2d.append(screen[valid])
        if not pts_2d:
            return None
        all_pts = np.concatenate(pts_2d, axis=0)
        x_min, y_min = np.min(all_pts, axis=0)
        x_max, y_max = np.max(all_pts, axis=0)
        pad = 6.0
        x_min = float(x_min - pad)
        y_min = float(y_min - pad)
        x_max = float(x_max + pad)
        y_max = float(y_max + pad)
        if x_max - x_min < 12.0 or y_max - y_min < 12.0:
            return None
        return x_min, y_min, x_max, y_max

    def projected_scanner_cell_bounds(model_matrix=None):
        if model_matrix is None:
            model_matrix = model_mat
        rows = max(1, int(getattr(state, 'scanner_rows', 1)))
        cols = max(1, int(getattr(state, 'scanner_cols', 1)))
        base_rows = max(1, int(state.bitmap_size[0]))

        if scanner_estimate_mode:
            overall = projected_mesh_bounds(model_matrix)
            if overall is None:
                fallback_w = min(float(disp_w) * 0.68, float(disp_h) * 0.68)
                fallback_h = fallback_w
                x_min = (float(disp_w) - fallback_w) * 0.5
                y_min = (float(disp_h) - fallback_h) * 0.5
                x_max = x_min + fallback_w
                y_max = y_min + fallback_h
            else:
                x_min, y_min, x_max, y_max = overall
            available_w = max(12.0, float(x_max - x_min))
            available_h = max(12.0, float(y_max - y_min))
            cell_size = max(12.0, min(available_w / cols, available_h / rows))
            grid_w = cell_size * cols
            grid_h = cell_size * rows
            left = 0.5 * (float(x_min + x_max) - grid_w)
            top = 0.5 * (float(y_min + y_max) - grid_h)
            return [
                (
                    left + c * cell_size,
                    top + r * cell_size,
                    left + (c + 1) * cell_size,
                    top + (r + 1) * cell_size,
                )
                for r in range(rows)
                for c in range(cols)
            ]

        if not renderer.mesh_pick_data:
            return None

        view_proj = state.camera.proj(disp_w, disp_h) @ state.camera.view()
        cell_pts = [[] for _ in range(rows * cols)]
        for verts, row_idx in renderer.mesh_pick_data:
            cell_index = int(row_idx) // base_rows
            if not (0 <= cell_index < rows * cols):
                continue
            base_row_idx = int(row_idx) % base_rows
            if state.row_visible is not None and len(state.row_visible) > 0 and not bool(state.row_visible[base_row_idx]):
                continue
            if len(verts) == 0:
                continue
            stride = max(1, len(verts) // 300)
            sample = verts[::stride]
            world_pts = transform_points(sample, model_matrix)
            screen, valid, _ = project_to_screen(world_pts, view_proj, disp_w, disp_h)
            if not np.any(valid):
                continue
            cell_pts[cell_index].append(screen[valid])

        bounds = []
        for chunks in cell_pts:
            if not chunks:
                bounds.append(None)
                continue
            pts = np.concatenate(chunks, axis=0)
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            pad = 8.0
            bounds.append((float(x_min - pad), float(y_min - pad), float(x_max + pad), float(y_max + pad)))
        return bounds

    def bounds_handles(bounds):
        x_min, y_min, x_max, y_max = bounds
        x_mid = 0.5 * (x_min + x_max)
        y_mid = 0.5 * (y_min + y_max)
        return [
            (x_min, y_min), (x_mid, y_min), (x_max, y_min),
            (x_max, y_mid),
            (x_max, y_max), (x_mid, y_max), (x_min, y_max),
            (x_min, y_mid),
        ]

    gizmo_bounds = projected_mesh_bounds()
    handle_radius = 6.0
    active_handle = int(state.get('bbox_active_handle', -1))
    hover_handle = -1
    scanner_cell_bounds = projected_scanner_cell_bounds() if scanner_stage_active else None
    if edit_controls_active and gizmo_bounds is not None and is_hovered:
        handles = bounds_handles(gizmo_bounds)
        d2 = [((lx - hx) ** 2 + (ly - hy) ** 2) for hx, hy in handles]
        best_idx = int(np.argmin(d2))
        if d2[best_idx] <= (handle_radius + 3.0) ** 2:
            hover_handle = best_idx
    state.bbox_hover_handle = hover_handle

    if gizmo_bounds is not None:
        dl = imgui.get_window_draw_list()
        ox, oy = float(state.vp_origin[0]), float(state.vp_origin[1])
        x_min, y_min, x_max, y_max = gizmo_bounds
        if edit_controls_active:
            rect_col = imgui.get_color_u32((0.95, 0.95, 0.95, 0.92))
            dl.add_rect((ox + x_min, oy + y_min), (ox + x_max, oy + y_max), rect_col, 0.0, 2.0, 0)
        if scanner_stage_active and scanner_cell_bounds:
            rows = max(1, int(getattr(state, 'scanner_rows', state.bitmap_size[0])))
            cols = max(1, int(getattr(state, 'scanner_cols', state.bitmap_size[1])))
            if str(state.get('scanner_camera_workflow', 'path')) == 'single':
                selected_r = int(np.clip(int(state.get('scanner_single_row', 1)) - 1, 0, rows - 1))
                selected_c = int(np.clip(int(state.get('scanner_single_col', 1)) - 1, 0, cols - 1))
            else:
                selected = np.asarray(getattr(state, 'scanner_selected_cell', [0, 0]), dtype=np.int32).reshape(-1)
                selected_r = int(np.clip(selected[0] if selected.size > 0 else 0, 0, rows - 1))
                selected_c = int(np.clip(selected[1] if selected.size > 1 else 0, 0, cols - 1))
            grid_col = imgui.get_color_u32((0.15, 0.85, 1.0, 0.34))
            fill_col = imgui.get_color_u32((0.15, 0.85, 1.0, 0.055))
            selected_fill = imgui.get_color_u32((1.0, 0.78, 0.18, 0.18))
            selected_line = imgui.get_color_u32((1.0, 0.78, 0.18, 0.92))
            display_colors, display_stage = _scanner_display_batch_colors(state)
            for r in range(rows):
                for c in range(cols):
                    cell_bounds = scanner_cell_bounds[r * cols + c]
                    if cell_bounds is None:
                        continue
                    cx0, cy0, cx1, cy1 = cell_bounds
                    selected_cell = r == selected_r and c == selected_c
                    if scanner_estimate_mode:
                        color_idx = r * cols + c
                        batch_rgb = display_colors[color_idx] if color_idx < len(display_colors) else [42.0, 42.0, 42.0]
                        batch_fill = imgui.get_color_u32((
                            float(np.clip(float(batch_rgb[0]) / 255.0, 0.0, 1.0)),
                            float(np.clip(float(batch_rgb[1]) / 255.0, 0.0, 1.0)),
                            float(np.clip(float(batch_rgb[2]) / 255.0, 0.0, 1.0)),
                            0.96,
                        ))
                        batch_line = selected_line if selected_cell else imgui.get_color_u32((0.08, 0.08, 0.08, 0.85))
                        dl.add_rect_filled(
                            (ox + cx0, oy + cy0),
                            (ox + cx1, oy + cy1),
                            batch_fill,
                        )
                        dl.add_rect(
                            (ox + cx0, oy + cy0),
                            (ox + cx1, oy + cy1),
                            batch_line,
                            0.0,
                            3.0 if selected_cell else 1.5,
                            0,
                        )
                        label = f"R{r + 1} C{c + 1}"
                        dl.add_text((ox + cx0 + 8, oy + cy0 + 8), imgui.get_color_u32((1.0, 1.0, 1.0, 0.92)), label)
                        continue
                    color_idx = r * cols + c
                    if color_idx < len(display_colors):
                        display_rgb = [float(v) / 255.0 for v in display_colors[color_idx]]
                        cell_fill = imgui.get_color_u32((
                            float(np.clip(display_rgb[0], 0.0, 1.0)),
                            float(np.clip(display_rgb[1], 0.0, 1.0)),
                            float(np.clip(display_rgb[2], 0.0, 1.0)),
                            0.24 if display_stage == "actual" else (0.18 if selected_cell else 0.12),
                        ))
                    else:
                        cell_fill = selected_fill if selected_cell else fill_col
                    dl.add_rect_filled(
                        (ox + cx0, oy + cy0),
                        (ox + cx1, oy + cy1),
                        cell_fill,
                    )
                    dl.add_rect(
                        (ox + cx0, oy + cy0),
                        (ox + cx1, oy + cy1),
                        selected_line if selected_cell else grid_col,
                        0.0,
                        2.0 if selected_cell else 1.0,
                        0,
                    )
        if edit_controls_active:
            for i, (hx, hy) in enumerate(bounds_handles(gizmo_bounds)):
                is_hot = (i == hover_handle or i == active_handle)
                fill = imgui.get_color_u32((0.95, 0.65, 0.10, 1.0) if is_hot else (0.96, 0.96, 0.96, 0.95))
                stroke = imgui.get_color_u32((0.12, 0.12, 0.12, 1.0))
                dl.add_circle_filled((ox + hx, oy + hy), handle_radius, fill, 16)
                dl.add_circle((ox + hx, oy + hy), handle_radius, stroke, 16, 1.5)

    # ImGuizmo
    if edit_controls_active and state.mode == 'spline' and state.selected_idx >= 0 and int(state.selected_idx) in visible_ctrl_index_map:
        local_pos = state.flat_pts_all[state.selected_idx].astype(np.float32)
        pos = transform_points([local_pos], model_mat)[0].astype(np.float32)
        M16 = imguizmo.im_guizmo.Matrix16

        view_m = M16(); view_m.values[:] = state.camera.view().T.flatten()
        proj_m = M16(); proj_m.values[:] = state.camera.proj(disp_w, disp_h).T.flatten()

        mat = np.eye(4, dtype=np.float32)
        mat[0, 3] = pos[0]; mat[1, 3] = pos[1]; mat[2, 3] = pos[2]
        obj_m = M16(); obj_m.values[:] = mat.T.flatten()

        imguizmo.im_guizmo.set_orthographic(True)
        imguizmo.im_guizmo.set_drawlist()
        gizmo_x = float(state.vp_origin[0])
        gizmo_y = float(state.vp_origin[1])
        gizmo_w = float(disp_w) * float(state.vp_scale)
        gizmo_h = float(disp_h) * float(state.vp_scale)
        imguizmo.im_guizmo.set_rect(gizmo_x, gizmo_y, gizmo_w, gizmo_h)
        changed = imguizmo.im_guizmo.manipulate(
            view_m, proj_m,
            imguizmo.im_guizmo.OPERATION.translate,
            imguizmo.im_guizmo.MODE.world,
            obj_m,
        )
        if changed:
            if not state.gizmo_edit_active:
                state.push_undo("Spline point")
                state.gizmo_edit_active = True
            new_world = np.array(obj_m.values[12:15], dtype=np.float32)
            new_local = transform_points([new_world], np.linalg.inv(model_mat))[0]
            state.move_ctrl_pt(state.selected_idx, new_local)
            state.rebuild_spline_mesh()
        elif state.gizmo_edit_active and not imguizmo.im_guizmo.is_using():
            state.gizmo_edit_active = False

    # Simulation force vectors: per-control-point displacement from the last
    # solver step, drawn in model space so they track the model's transform.
    if (
        edit_controls_active
        and bool(state.get('sim_show_forces', False))
        and len(visible_ctrl_indices) > 0
    ):
        with state.sim_lock:
            delta_P = state.sim_delta_P
            delta_P = None if delta_P is None else np.asarray(delta_P, dtype=np.float32).copy()
        if delta_P is not None and len(delta_P) > 0:
            dl = imgui.get_window_draw_list()
            view_proj = state.camera.proj(disp_w, disp_h) @ state.camera.view()
            force_color = imgui.get_color_u32((1.0, 0.2, 0.2, 1.0))
            # Read the viewport origin here rather than reusing ox/oy from the
            # bounding-box block above: that block only runs when gizmo_bounds
            # exists, so those names are not always bound at this point.
            origin = np.array([float(state.vp_origin[0]), float(state.vp_origin[1])], dtype=np.float32)
            for idx in visible_ctrl_indices:
                idx = int(idx)
                # Virtual (period) control points have no simulated counterpart.
                if idx >= len(delta_P):
                    continue
                p0_local = state.flat_pts_all[idx].astype(np.float32)
                p1_local = p0_local + delta_P[idx] * FORCE_ARROW_SCALE
                screen, valid, _ = project_to_screen(transform_points([p0_local, p1_local], model_mat), view_proj, disp_w, disp_h)
                if bool(np.all(valid)):
                    p0 = tuple((origin + screen[0]).tolist())
                    p1 = tuple((origin + screen[1]).tolist())
                    dl.add_line(p0, p1, force_color, 2.0)
                    dl.add_circle_filled(p1, 3.0, force_color)

    # Mouse interaction inside the viewport
    # Follows what is actually on screen, not just the checkbox: the lock exists
    # to stop the model drifting against the reference image, so it has no job
    # in a mode that does not draw one.
    alignment_locked = bool(_reference_overlay_active(state) and state.ref_bg_lock_zoom)

    def viewport_pixel_delta_to_world(dx_px, dy_px):
        aspect = max(1.0, disp_w) / max(1.0, disp_h)
        half_h = max(1e-4, float(state.camera.dist) * np.tan(np.radians(float(state.camera.fov_deg)) * 0.5))
        half_w = half_h * aspect
        wu_x = (2.0 * half_w) / max(float(disp_w), 1.0)
        wu_y = (2.0 * half_h) / max(float(disp_h), 1.0)
        view = state.camera.view()
        right = view[0, :3]
        up = view[1, :3]
        return right * (float(dx_px) * wu_x) - up * (float(dy_px) * wu_y)

    def pixel_drag_to_world_delta(dx_screen, dy_screen):
        # Convert screen-space mouse delta to viewport-pixel delta.
        dx_px = float(dx_screen) / max(float(state.vp_scale), 1e-6)
        dy_px = float(dy_screen) / max(float(state.vp_scale), 1e-6)
        return viewport_pixel_delta_to_world(dx_px, dy_px)

    suppress_mesh_click = False
    if is_hovered:
        io = imgui.get_io()
        lmb_down = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        if scanner_stage_active and scanner_cell_bounds and hover_handle < 0 and imgui.is_mouse_clicked(imgui.MouseButton_.left) and not io.key_shift and not io.key_alt:
            rows = max(1, int(getattr(state, 'scanner_rows', state.bitmap_size[0])))
            cols = max(1, int(getattr(state, 'scanner_cols', state.bitmap_size[1])))
            for cell_index, cell_bounds in enumerate(scanner_cell_bounds):
                if cell_bounds is None:
                    continue
                x_min, y_min, x_max, y_max = cell_bounds
                if x_min <= lx <= x_max and y_min <= ly <= y_max:
                    cell_r = cell_index // cols
                    cell_c = cell_index % cols
                    state.scanner_selected_cell = [cell_r, cell_c]
                    if str(state.get('scanner_camera_workflow', 'path')) == 'single':
                        state.scanner_single_row = int(cell_r + 1)
                        state.scanner_single_col = int(cell_c + 1)
                        state.scanner_status = f"Single capture target: row {cell_r + 1}, col {cell_c + 1}"
                        embedded = state.get('embedded_scanner')
                        if embedded is not None:
                            embedded.preview_single_target(
                                cell_r,
                                cell_c,
                                int(state.get('scanner_single_angle', 1)) - 1,
                                float(state.get('scanner_camera_zoom', 1.0)),
                            )
                            state.scanner_status = embedded.status
                    else:
                        state.scanner_status = f"Selected mini-fabric R{cell_r + 1} C{cell_c + 1}"
                    suppress_mesh_click = True
                    break

        # Bounding-box resize handles (window-like scaling in screen space).
        if edit_controls_active and gizmo_bounds is not None and hover_handle >= 0 and imgui.is_mouse_clicked(imgui.MouseButton_.left) and not io.key_shift and not io.key_alt:
            state.push_undo("Bounding box scale")
            state.bbox_active_handle = int(hover_handle)
            state.bbox_start_bounds = np.array(gizmo_bounds, dtype=np.float32)
            state.bbox_start_mouse = np.array([lx, ly], dtype=np.float32)
            state.bbox_start_t = np.array(state.model_t, dtype=np.float32)
            state.bbox_start_model_scale = np.array(state.model_scale, dtype=np.float32)
            suppress_mesh_click = True

        active_handle = int(state.get('bbox_active_handle', -1))
        if edit_controls_active and active_handle >= 0 and lmb_down and state.get('bbox_start_bounds') is not None:
            x0_min, y0_min, x0_max, y0_max = [float(v) for v in state.bbox_start_bounds]
            old_w = max(1e-4, x0_max - x0_min)
            old_h = max(1e-4, y0_max - y0_min)
            min_size = 20.0

            x_min, y_min, x_max, y_max = x0_min, y0_min, x0_max, y0_max
            if active_handle in (0, 7, 6):
                x_min = min(lx, x0_max - min_size)
            if active_handle in (2, 3, 4):
                x_max = max(lx, x0_min + min_size)
            if active_handle in (0, 1, 2):
                y_min = min(ly, y0_max - min_size)
            if active_handle in (4, 5, 6):
                y_max = max(ly, y0_min + min_size)

            new_w = max(1e-4, x_max - x_min)
            new_h = max(1e-4, y_max - y_min)
            sx = new_w / old_w
            sy = new_h / old_h
            if active_handle in (1, 5):
                scale_vec = np.array([1.0, sy, 1.0], dtype=np.float32)
            elif active_handle in (3, 7):
                scale_vec = np.array([sx, 1.0, 1.0], dtype=np.float32)
            else:
                scale_vec = np.array([sx, sy, 1.0], dtype=np.float32)

            start_scale = np.array(state.get('bbox_start_model_scale', state.model_scale), dtype=np.float32)
            if start_scale.size == 1:
                start_scale = np.repeat(start_scale, 3)
            state.model_scale = np.maximum(start_scale[:3] * scale_vec, 1e-4).astype(np.float32)

            start_t = np.array(state.get('bbox_start_t', state.model_t), dtype=np.float32)
            old_cx, old_cy = 0.5 * (x0_min + x0_max), 0.5 * (y0_min + y0_max)
            new_cx, new_cy = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
            correction = viewport_pixel_delta_to_world(new_cx - old_cx, new_cy - old_cy)
            state.model_t = (start_t + correction).astype(np.float32)
            suppress_mesh_click = True
        elif active_handle >= 0 and (not edit_controls_active or not lmb_down):
            state.bbox_active_handle = -1
            state.bbox_start_bounds = None
            state.bbox_start_mouse = None
            state.bbox_start_t = None
            state.bbox_start_model_scale = None

        state.hover_mesh_idx = renderer.pick_mesh_index(model_mat, state.camera, disp_w, disp_h, lx, ly, visible_rows=state.row_visible)

        color_pick_active = bool(state.reference_color_pick_active or io.key_alt)
        if color_pick_active:
            imgui.set_mouse_cursor(imgui.MouseCursor_.hand)

        if edit_controls_active and imgui.is_mouse_clicked(imgui.MouseButton_.left) and not io.key_shift and not color_pick_active and not suppress_mesh_click:
            state.selected_mesh_idx = state.hover_mesh_idx

        if edit_controls_active and color_pick_active and imgui.is_mouse_clicked(imgui.MouseButton_.left):
            mesh_idx = state.hover_mesh_idx if state.hover_mesh_idx >= 0 else state.selected_mesh_idx
            sampled = sample_reference_color(lx, ly)
            if sampled is None:
                sampled = renderer.sample_color(lx, ly)
            row_idx = renderer.get_row_for_mesh_index(mesh_idx)
            if sampled is not None and row_idx is not None and int(state.bitmap_size[0]) > 0:
                row_idx = int(row_idx) % int(state.bitmap_size[0])
                state.push_undo("Pick yarn color")
                state.use_row_colors = True
                state.row_colors[row_idx] = sampled.tolist()
                state.rebuild_spline_mesh()
                state.selected_mesh_idx = mesh_idx
                state.reference_color_pick_active = False
                state.status_msg = f"Picked row {row_idx + 1} color from reference"
    else:
        state.hover_mesh_idx = -1
        active_handle = int(state.get('bbox_active_handle', -1))
        lmb_down = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        if active_handle >= 0 and (not edit_controls_active or not lmb_down):
            state.bbox_active_handle = -1
            state.bbox_start_bounds = None
            state.bbox_start_mouse = None
            state.bbox_start_t = None
            state.bbox_start_model_scale = None

    def radius_range():
        radius_idx = state._pidx['radius']
        lo, hi = state.config["knit_parameters"]["parameters"][radius_idx]["range"]
        return radius_idx, float(lo), float(hi)


    def local_radius_edit_index():
        if state.mode != 'spline':
            return -1
        hover_idx = int(state.hover_idx)
        selected_idx = int(state.selected_idx)
        if hover_idx >= 0 and hover_idx in visible_ctrl_index_map:
            return hover_idx
        if selected_idx >= 0 and selected_idx in visible_ctrl_index_map:
            return selected_idx
        return -1

    def local_radius_value(flat_idx):
        row_idx = np.searchsorted(state._row_starts, flat_idx, side="right") - 1
        if not (0 <= row_idx < len(state.spline_radius_rows)):
            return float(state.params[state._pidx['radius']])
        local_idx = int(flat_idx - state._row_starts[row_idx])
        return float(state.spline_radius_rows[row_idx][local_idx])

    def set_local_radius_from_viewport(flat_idx, value, start_rows=None):
        state.set_local_radius(flat_idx, value, start_rows=start_rows)
        state.rebuild_spline_mesh()

    if is_hovered:
        curr = (mx, my)
        bbox_drag_active = int(state.get('bbox_active_handle', -1)) >= 0
        spline_drag_active = (
            state.mode == 'spline'
            and (
                (state.selected_idx >= 0 and (imguizmo.im_guizmo.is_using() or imguizmo.im_guizmo.is_over()))
                or state.hover_idx >= 0
            )
        )

        io = imgui.get_io()
        if io.mouse_wheel != 0 and not io.key_shift:
            # Orthographic camera zoom by changing distance and re-rendering.
            zoom_factor = float(np.exp(io.mouse_wheel * 0.12))
            state.camera.dist = float(np.clip(float(state.camera.dist) / zoom_factor, 1.0, 200.0))
            # Keep legacy image zoom neutral to avoid pixelated post-scale.
            state.viewport_zoom = 1.0

        if state.get('prev_mouse') is not None:
            prev = state.prev_mouse
            dx = mx - prev[0]
            dy = my - prev[1]
            lmb = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT)   == glfw.PRESS
            rmb = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT)  == glfw.PRESS
            mmb = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
            shift_down = imgui.get_io().key_shift

            if not edit_controls_active:
                can_transform_model = False
            elif shift_down and state.mode == 'spline' and state.selected_idx >= 0 and (
                    imguizmo.im_guizmo.is_using() or imguizmo.im_guizmo.is_over()):
                can_transform_model = False
            elif shift_down and state.mode == 'spline' and state.hover_idx >= 0:
                can_transform_model = False
            else:
                can_transform_model = shift_down and not alignment_locked

            if bool(state.spline_grab_active) or bool(state.radius_grab_active):
                can_transform_model = False

            if can_transform_model and rmb:
                if not state.model_drag_undo_active:
                    state.push_undo("Model transform")
                    state.model_drag_undo_active = True
                state.model_t += pixel_drag_to_world_delta(dx, dy)
            elif (mmb or (rmb and not shift_down)) and not bool(state.spline_grab_active) and not bool(state.radius_grab_active):
                state.camera.orbit(dx, dy)
            elif edit_controls_active and lmb and not shift_down and not bbox_drag_active and not spline_drag_active and not bool(state.spline_grab_active) and not bool(state.radius_grab_active):
                if not state.model_drag_undo_active:
                    state.push_undo("Model translate")
                    state.model_drag_undo_active = True
                state.model_t += pixel_drag_to_world_delta(dx, dy)

        r_down = glfw.get_key(window, glfw.KEY_R) == glfw.PRESS
        r_pressed = r_down and not bool(state.radius_grab_key_was_down)
        state.radius_grab_key_was_down = r_down
        if edit_controls_active and r_pressed and not bool(state.radius_grab_active):
            edit_idx = local_radius_edit_index()
            if edit_idx >= 0:
                state.push_undo("Local tube radius")
                state.selected_idx = edit_idx
                state.radius_grab_active = True
                state.radius_grab_point_idx = edit_idx
                state.radius_grab_start_mouse = np.array([mx, my], dtype=np.float32)
                state.radius_grab_start_value = local_radius_value(edit_idx)
                state.radius_grab_start_rows = [row.copy() for row in state.spline_radius_rows]

        if edit_controls_active and bool(state.radius_grab_active):
            start_mouse = np.asarray(state.radius_grab_start_mouse, dtype=np.float32)
            start_radius = float(state.radius_grab_start_value)
            _, lo, hi = radius_range()
            drag_px = mx - start_mouse[0]
            if (
                glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS
                or glfw.get_key(window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS
            ):
                drag_px *= 0.25
            if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS:
                drag_px *= 2.5
            new_radius = np.clip(start_radius + drag_px * ((hi - lo) / 500.0), lo, hi)
            set_local_radius_from_viewport(
                int(state.radius_grab_point_idx),
                new_radius,
                start_rows=state.radius_grab_start_rows,
            )
            if (
                imgui.is_mouse_clicked(imgui.MouseButton_.left)
                or glfw.get_key(window, glfw.KEY_ENTER) == glfw.PRESS
                or glfw.get_key(window, glfw.KEY_KP_ENTER) == glfw.PRESS
            ):
                state.radius_grab_active = False
                state.radius_grab_start_rows = []
            elif (
                imgui.is_mouse_clicked(imgui.MouseButton_.right)
                or glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS
            ):
                state.spline_radius_rows = [row.copy() for row in state.radius_grab_start_rows]
                state.rebuild_spline_mesh()
                state.radius_grab_active = False
                state.radius_grab_start_rows = []

        if edit_controls_active and not bool(state.radius_grab_active):
            radius_delta = 0.0
            edit_idx = local_radius_edit_index()
            current_radius = local_radius_value(edit_idx) if edit_idx >= 0 else 0.0
            radius_step = max(0.001, current_radius * 0.04)
            if (
                glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS
                or glfw.get_key(window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS
            ):
                radius_step *= 0.25
            if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS:
                radius_step *= 2.5
            if glfw.get_key(window, glfw.KEY_RIGHT_BRACKET) == glfw.PRESS or glfw.get_key(window, glfw.KEY_EQUAL) == glfw.PRESS:
                radius_delta += radius_step
            if glfw.get_key(window, glfw.KEY_LEFT_BRACKET) == glfw.PRESS or glfw.get_key(window, glfw.KEY_MINUS) == glfw.PRESS:
                radius_delta -= radius_step

            if abs(radius_delta) > 0.0 and edit_idx >= 0:
                if not bool(state.radius_keyboard_edit_active):
                    state.push_undo("Local tube radius")
                    state.radius_keyboard_edit_active = True
                state.selected_idx = edit_idx
                set_local_radius_from_viewport(edit_idx, current_radius + radius_delta)
            else:
                state.radius_keyboard_edit_active = False

        if edit_controls_active and state.mode == 'spline' and len(visible_ctrl_indices) > 0 and not bool(state.radius_grab_active):
            g_down = glfw.get_key(window, glfw.KEY_G) == glfw.PRESS
            g_pressed = g_down and not bool(state.spline_grab_key_was_down)
            state.spline_grab_key_was_down = g_down

            if g_pressed and not bool(state.spline_grab_active):
                grab_idx = int(state.hover_idx) if int(state.hover_idx) >= 0 else int(state.selected_idx)
                if grab_idx >= 0 and grab_idx in visible_ctrl_index_map:
                    state.push_undo("Spline point")
                    state.selected_idx = grab_idx
                    state.spline_grab_active = True
                    state.spline_grab_start_mouse = np.array([mx, my], dtype=np.float32)
                    state.spline_grab_start_pos = state.flat_pts_all[grab_idx].astype(np.float32).copy()

            if bool(state.spline_grab_active):
                if state.selected_idx < 0 or int(state.selected_idx) not in visible_ctrl_index_map:
                    state.spline_grab_active = False
                else:
                    start_mouse = np.asarray(state.spline_grab_start_mouse, dtype=np.float32)
                    start_pos = np.asarray(state.spline_grab_start_pos, dtype=np.float32)
                    world_delta = pixel_drag_to_world_delta(mx - start_mouse[0], my - start_mouse[1])
                    local_delta = np.linalg.inv(model_mat)[:3, :3] @ world_delta
                    state.move_ctrl_pt(
                        state.selected_idx,
                        start_pos + local_delta.astype(np.float32),
                    )
                    state.rebuild_spline_mesh()
                    if (
                        imgui.is_mouse_clicked(imgui.MouseButton_.left)
                        or glfw.get_key(window, glfw.KEY_ENTER) == glfw.PRESS
                        or glfw.get_key(window, glfw.KEY_KP_ENTER) == glfw.PRESS
                    ):
                        state.spline_grab_active = False
                    elif (
                        imgui.is_mouse_clicked(imgui.MouseButton_.right)
                        or glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS
                    ):
                        state.move_ctrl_pt(state.selected_idx, start_pos)
                        state.rebuild_spline_mesh()
                        state.spline_grab_active = False

        if edit_controls_active and state.mode == 'spline' and state.selected_idx >= 0 and int(state.selected_idx) in visible_ctrl_index_map and not bool(state.radius_grab_active):
            view = state.camera.view()
            right = view[0, :3]
            up = view[1, :3]
            forward = -view[2, :3]
            step = float(state.spline_keyboard_step)
            if (
                glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS
                or glfw.get_key(window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS
            ):
                step *= 0.2

            world_delta = np.zeros(3, dtype=np.float32)
            if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS or glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
                world_delta -= right * step
            if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS or glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
                world_delta += right * step
            if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS or glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
                world_delta += up * step
            if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS or glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
                world_delta -= up * step
            if glfw.get_key(window, glfw.KEY_PAGE_UP) == glfw.PRESS or glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
                world_delta += forward * step
            if glfw.get_key(window, glfw.KEY_PAGE_DOWN) == glfw.PRESS or glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
                world_delta -= forward * step

            if np.linalg.norm(world_delta) > 0.0:
                if not state.spline_keyboard_edit_active:
                    state.push_undo("Spline point")
                    state.spline_keyboard_edit_active = True
                local_delta = np.linalg.inv(model_mat)[:3, :3] @ world_delta
                state.move_ctrl_pt(
                    state.selected_idx,
                    state.flat_pts_all[state.selected_idx] + local_delta.astype(np.float32),
                )
                state.rebuild_spline_mesh()
            else:
                state.spline_keyboard_edit_active = False

        state.model_rot_dragging = False
        if not (
            not alignment_locked
            and imgui.get_io().key_shift
            and glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        ):
            state.model_drag_undo_active = False

        # Spline handle hover + select
        if edit_controls_active and state.mode == 'spline' and len(visible_ctrl_indices) > 0:
            gizmo_active = state.selected_idx >= 0 and (
                imguizmo.im_guizmo.is_using() or imguizmo.im_guizmo.is_over()
            )
            if not gizmo_active:
                visible_ctrl_pts = state.flat_pts_all[visible_ctrl_indices]
                world_pts = transform_points(visible_ctrl_pts, model_mat)
                view_proj = state.camera.proj(disp_w, disp_h) @ state.camera.view()
                screen, valid, ndc = project_to_screen(world_pts, view_proj, disp_w, disp_h)
                in_view = (
                    valid
                    & (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0)
                    & (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
                )
                d2 = np.sum((screen - np.array([lx, ly], dtype=np.float32)) ** 2, axis=1)
                d2[~in_view] = np.inf
                best_i = int(np.argmin(d2))
                best_i = int(visible_ctrl_indices[best_i]) if d2[best_i] <= 16.0 ** 2 else -1
                state.hover_idx = best_i
                if imgui.is_mouse_clicked(imgui.MouseButton_.left):
                    state.selected_idx = best_i

        state.prev_mouse = curr
    else:
        state.prev_mouse = None

    if split_right_mode is not None:
        right_w = max(180.0, total_avail_x - avail_x - 8.0)
        imgui.set_cursor_screen_pos(imgui.ImVec2(draw_pos.x + avail_x + 8.0, draw_pos.y))
        if imgui.begin_child("##vp_split_right", imgui.ImVec2(right_w, avail_y), imgui.ChildFlags_.borders):
            if split_right_mode == 'edit_orbit':
                _draw_orbit_canvas(state, window)
            elif split_right_mode == 'scan_robot':
                embedded = state.get('embedded_scanner')
                top_h = max(140.0, (avail_y - 90.0) * 0.56)
                _draw_scan_simulator_canvas(state, embedded, window, height=top_h)
                imgui.separator()
                bot_h = max(100.0, imgui.get_content_region_avail().y - 36.0)
                _draw_scan_gripper_camera_canvas(embedded, height=bot_h)
            elif split_right_mode == 'puzzle_previews':
                top_h = max(140.0, (avail_y - 90.0) * 0.50)
                _draw_puzzle_inspector_canvas(state, renderer, height=top_h)
                imgui.separator()
                bot_h = max(120.0, imgui.get_content_region_avail().y - 32.0)
                _draw_puzzle_glue_canvas(state, renderer, height=bot_h)
        imgui.end_child()

    imgui.end()


def draw_orbit_viewport(state, window):
    """Second, freely-orbitable view of the same model, over a ground grid."""
    imgui.set_next_window_pos((1220, 20), cond=imgui.Cond_.first_use_ever)
    imgui.set_next_window_size((360, 370), cond=imgui.Cond_.first_use_ever)
    imgui.begin("3D Orbit View", flags=imgui.WindowFlags_.no_scroll_with_mouse)
    _draw_orbit_canvas(state, window)
    imgui.end()
