import os
import numpy as np
import glfw
import tkinter as tk
from tkinter import filedialog as _filedialog
from imgui_bundle import imgui, imguizmo

from rendering import draw_fitted_texture, transform_points
from app_state import WORKFLOW_STAGES, TEXTURE_CONTROL_GROUPS, TEXTURE_PRESET_BUTTONS
from knitting_core import (
    CONFIG, geometry_param_index, geometry_param_range, geometry_parameter_names
)

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

# %% GUI DRAWING PANELS ────────────────────────────────────────────────────────

def draw_menu_bar(state):
    if imgui.begin_menu_bar():
        if imgui.begin_menu("Window"):
            clicked_reset, _ = imgui.menu_item("Reset Layout")
            if clicked_reset:
                try:
                    os.remove(os.path.join(os.path.dirname(state.save_path), "imgui_layout.ini"))
                except FileNotFoundError:
                    pass
            imgui.end_menu()
        imgui.end_menu_bar()


def draw_workflow_header(state):
    stage_idx = int(np.clip(state.workflow_step, 0, len(WORKFLOW_STAGES) - 1))
    title, subtitle = WORKFLOW_STAGES[stage_idx]
    imgui.text(f"Step {stage_idx + 1} of {len(WORKFLOW_STAGES)}")
    imgui.text_colored((0.92, 0.74, 0.34, 1.0), title)
    imgui.text_wrapped(subtitle)
    imgui.spacing()

    avail_w = imgui.get_content_region_avail().x
    dot_w = max(18.0, (avail_w - (len(WORKFLOW_STAGES) - 1) * 4.0) / len(WORKFLOW_STAGES))
    imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(4, 2))
    for i, _ in enumerate(WORKFLOW_STAGES):
        active = i == stage_idx
        color = (0.25, 0.55, 0.85, 1.0) if active else (0.22, 0.22, 0.22, 1.0)
        hover = (0.35, 0.65, 0.95, 1.0) if active else (0.34, 0.34, 0.34, 1.0)
        imgui.push_style_color(imgui.Col_.button, color)
        imgui.push_style_color(imgui.Col_.button_hovered, hover)
        if imgui.button(f"{i + 1}##stage_{i}", imgui.ImVec2(dot_w, 22)):
            state.workflow_step = i
        imgui.pop_style_color(2)
        if i < len(WORKFLOW_STAGES) - 1:
            imgui.same_line()
    imgui.pop_style_var()

    imgui.spacing()
    back_disabled = stage_idx == 0
    next_disabled = stage_idx == len(WORKFLOW_STAGES) - 1
    nav_w = max(90, (imgui.get_content_region_avail().x - imgui.get_style().item_spacing.x) * 0.5)
    if back_disabled:
        imgui.begin_disabled()
    if imgui.button("Back##workflow", (nav_w, 0)):
        state.workflow_step = max(0, stage_idx - 1)
    if back_disabled:
        imgui.end_disabled()
    imgui.same_line()
    if next_disabled:
        imgui.begin_disabled()
    if imgui.button("Next##workflow", (nav_w, 0)):
        state.workflow_step = min(len(WORKFLOW_STAGES) - 1, stage_idx + 1)
    if next_disabled:
        imgui.end_disabled()
    imgui.separator()


def draw_sidebar(state, renderer):
    imgui.set_next_window_pos((20, 20), cond=imgui.Cond_.first_use_ever)
    imgui.set_next_window_size((320, 820), cond=imgui.Cond_.first_use_ever)
    imgui.begin("Guided Workflow")

    draw_workflow_header(state)

    undo_disabled = not state.undo_stack
    if undo_disabled:
        imgui.begin_disabled()
    if imgui.button("Undo##main", (120, 0)):
        state.undo_last()
    if undo_disabled:
        imgui.end_disabled()
    imgui.same_line()
    last_undo = state.undo_stack[-1]['label'] if state.undo_stack else "No changes"
    imgui.text_disabled(last_undo)
    imgui.separator()

    stage = int(np.clip(state.workflow_step, 0, len(WORKFLOW_STAGES) - 1))

    def rebuild_current_mesh():
        if state.mode == 'spline':
            state.rebuild_spline_mesh()
        else:
            state.rebuild_param_mesh()

    def draw_surface_fiber_controls(context_label="Surface fibers"):
        changed_enabled, enabled = imgui.checkbox(
            "Use multi-fiber rows##fiber_geometry_enabled",
            state.fiber_geometry_enabled,
        )
        fibers_changed = False
        if changed_enabled:
            state.push_undo(context_label)
            state.fiber_geometry_enabled = enabled
            fibers_changed = True

        if not state.fiber_geometry_enabled:
            imgui.text_disabled("Enable this to replace each row tube with separate fiber tubes.")
        else:
            controls = (
                ('fiber_geometry_count', 'Fibers per row', 1, 12, 'int'),
                ('fiber_geometry_radius_scale', 'Fiber radius scale', 0.04, 0.45, 'float'),
                ('fiber_geometry_lift', 'Lift above surface', 0.0, 1.0, 'float'),
                ('fiber_geometry_surface_arc', 'Surface spread', 0.05, 1.0, 'float'),
                ('fiber_geometry_randomness', 'Randomness', 0.0, 1.0, 'float'),
                ('fiber_geometry_twist', 'Fiber twist', -3.0, 3.0, 'float'),
            )
            for key, label, lo, hi, kind in controls:
                if kind == 'int':
                    changed, new_val = imgui.slider_int(f"{label}##{key}", int(state[key]), int(lo), int(hi))
                else:
                    changed, new_val = imgui.slider_float(f"{label}##{key}", float(state[key]), float(lo), float(hi), "%.2f")
                if imgui.is_item_activated():
                    state.push_undo(label)
                if changed:
                    state[key] = int(new_val) if kind == 'int' else float(new_val)
                    fibers_changed = True

        if fibers_changed:
            rebuild_current_mesh()

    # Copies configuration (vectorized display_copies)
    changed_x, new_copies_x = imgui.slider_int("Copies X", int(state.display_copies[0]), 0, 5)
    changed_y, new_copies_y = imgui.slider_int("Copies Y", int(state.display_copies[1]), 0, 5)
    if changed_x or changed_y:
        state.push_undo("Display copies")
        state.display_copies[0] = new_copies_x
        state.display_copies[1] = new_copies_y
        rebuild_current_mesh()
    imgui.separator()

    if stage == 0:
        imgui.text("Viewport Alignment")
        imgui.spacing()
        _, state.show_ref_bg = imgui.checkbox("Show reference overlay", state.show_ref_bg)
        alignment_locked = bool(state.show_ref_bg and state.ref_bg_lock_zoom)
        if state.show_ref_bg:
            old_lock_zoom = state.ref_bg_lock_zoom
            changed_lock, new_lock_zoom = imgui.checkbox("Lock image/model alignment##bg", state.ref_bg_lock_zoom)
            if changed_lock:
                zoom_factor = max(float(state.camera.zoom_factor()), 1e-6)
                old_zoom_scale = zoom_factor if old_lock_zoom else 1.0
                new_zoom_scale = zoom_factor if new_lock_zoom else 1.0
                visible_scale_x = state.ref_bg_scale[0] * old_zoom_scale
                visible_scale_y = state.ref_bg_scale[1] * old_zoom_scale
                state.ref_bg_lock_zoom = old_lock_zoom
                state.push_undo("Background lock")
                state.ref_bg_lock_zoom = new_lock_zoom
                state.ref_bg_scale[0] = visible_scale_x / new_zoom_scale
                state.ref_bg_scale[1] = visible_scale_y / new_zoom_scale
            _, state.ref_bg_alpha = imgui.slider_float("Opacity##bg", state.ref_bg_alpha, 0.0, 1.0, "%.2f")
            alignment_locked = bool(state.ref_bg_lock_zoom)
            if alignment_locked:
                imgui.begin_disabled()
            changed_dims_lock, new_dims_lock = imgui.checkbox("Lock Dimensions##bg", state.ref_bg_lock_dimensions)
            if changed_dims_lock:
                state.push_undo("Image dimensions")
                state.ref_bg_lock_dimensions = new_dims_lock
                if new_dims_lock:
                    state.ref_bg_scale[1] = state.ref_bg_scale[0]
            changed_w, new_w = imgui.drag_float("Image width##bgsx", float(state.ref_bg_scale[0]), 0.01, 0.01, 50.0, "%.2f")
            if imgui.is_item_activated():
                state.push_undo("Image dimensions")
            changed_h, new_h = imgui.drag_float("Image height##bgsy", float(state.ref_bg_scale[1]), 0.01, 0.01, 50.0, "%.2f")
            if imgui.is_item_activated():
                state.push_undo("Image dimensions")
            if state.ref_bg_lock_dimensions:
                if changed_w:
                    state.ref_bg_scale[0] = new_w
                    state.ref_bg_scale[1] = new_w
                elif changed_h:
                    state.ref_bg_scale[0] = new_h
                    state.ref_bg_scale[1] = new_h
            else:
                if changed_w:
                    state.ref_bg_scale[0] = new_w
                if changed_h:
                    state.ref_bg_scale[1] = new_h
            if alignment_locked:
                imgui.end_disabled()
                imgui.text_disabled("Alignment locked. Unlock to move image or model.")
            if state.ref_bg_lock_zoom:
                zoom_factor = state.camera.zoom_factor()
                imgui.text_disabled(
                    f"Displayed: {state.ref_bg_scale[0] * zoom_factor:.2f} x "
                    f"{state.ref_bg_scale[1] * zoom_factor:.2f}"
                )
            if alignment_locked:
                imgui.begin_disabled()
            if imgui.small_button("Image 1:1##bg"):
                state.ref_bg_scale[0] = state.ref_bg_scale[1] = 1.0
            _, state.ref_bg_offset[0] = imgui.drag_float("Image X##bgox", float(state.ref_bg_offset[0]), 0.001, -2.0, 2.0, "%.3f")
            _, state.ref_bg_offset[1] = imgui.drag_float("Image Y##bgoy", float(state.ref_bg_offset[1]), 0.001, -2.0, 2.0, "%.3f")
            if imgui.small_button("Center image##bg"):
                state.ref_bg_offset[0] = state.ref_bg_offset[1] = 0.0
            _, state.ref_bg_rotation = imgui.slider_float("Image rotation##bg", state.ref_bg_rotation, -float(np.pi), float(np.pi), "%.2f rad")
            if alignment_locked:
                imgui.end_disabled()

        imgui.separator()
        changed_view_fov, new_view_fov = imgui.slider_float("View FoV##view", state.view_fov, 10.0, 120.0, "%.1f")
        if imgui.is_item_activated():
            state.push_undo("View FoV")
        if changed_view_fov:
            state.view_fov = new_view_fov
            state.camera.fov_deg = new_view_fov

        if alignment_locked:
            imgui.begin_disabled()
        imgui.text_wrapped("Viewport controls: Shift+LMB rotate, Shift+MMB scale, Shift+RMB translate.")
        imgui.text_wrapped("Image controls: Mouse wheel zoom, drag (no Shift) pan image.")
        pi = float(np.pi)
        imgui.text_colored((0.8, 0.8, 0.4, 1.0), "Rotation")
        
        # Vectorized model rotation
        labels = ["X", "Y", "Z"]
        for idx in range(3):
            changed, val = imgui.slider_float(f"{labels[idx]}##align_rot_{idx}", float(state.model_rot[idx]), -pi, pi, f"{labels[idx]}: %.2f rad")
            if imgui.is_item_activated():
                state.push_undo("Model rotation")
            if changed:
                state.model_rot[idx] = val

        imgui.text_colored((0.4, 0.8, 0.8, 1.0), "Position")
        # Vectorized model translation
        for idx in range(3):
            changed, val = imgui.drag_float(f"{labels[idx]}##align_t_{idx}", float(state.model_t[idx]), 0.01, -100.0, 100.0, f"{labels[idx]}: %.3f")
            if imgui.is_item_activated():
                state.push_undo("Model position")
            if changed:
                state.model_t[idx] = val

        changed_scale, new_scale = imgui.slider_float("Scale##align_scale", float(state.model_scale), 0.05, 20.0, "%.3f")
        if imgui.is_item_activated():
            state.push_undo("Model scale")
        if changed_scale:
            state.model_scale = new_scale

        if imgui.small_button("Center model##align"):
            state.push_undo("Center model")
            state.center_model_on_view()
        imgui.same_line()
        if imgui.small_button("Reset transform##align"):
            state.push_undo("Reset transform")
            state.model_rot[:] = 0.0
            state.model_scale = 1.0
            state.center_model_on_view()
        if alignment_locked:
            imgui.end_disabled()

    elif stage == 1:
        imgui.text("Pattern and Rows")
        max_rows = int(CONFIG['geometry']['bitmap_rows'])
        ch_r, new_rows = imgui.slider_int("Rows##bres", int(state.bitmap_size[0]), 1, max_rows)
        ch_c, new_cols = imgui.slider_int("Columns##bres", int(state.bitmap_size[1]), 1, 16)
        if ch_r or ch_c:
            state.push_undo("Bitmap size")
            state.on_bitmap_resize(new_rows, new_cols)

        imgui.separator()
        imgui.text("Row visibility")
        if imgui.small_button("Show all##rows"):
            state.push_undo("Show rows")
            state.row_visible[:] = True
        imgui.same_line()
        if imgui.small_button("Hide all##rows"):
            state.push_undo("Hide rows")
            state.row_visible[:] = False
        row_changed = False
        for r in range(int(state.bitmap_size[0])):
            changed_row, new_visible = imgui.checkbox(f"Row {r + 1}##row_vis_{r}", bool(state.row_visible[r]))
            if changed_row:
                if not row_changed:
                    state.push_undo("Row visibility")
                state.row_visible[r] = new_visible
                row_changed = True

        imgui.separator()
        imgui.text("Pattern")
        imgui.same_line()
        if imgui.small_button("Reset##bmap"):
            state.push_undo("Pattern reset")
            state.bitmap[:] = 1.0
            state.on_bitmap_change()
        nr, nc = state.bitmap.shape
        CELL_W, CELL_H = 22, 16
        grid_w = nc * CELL_W + (nc - 1) * 2
        offset_x = max(0.0, (imgui.get_content_region_avail().x - grid_w) / 2)
        bmap_changed = False
        imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(2, 2))
        for r in range(nr):
            imgui.set_cursor_pos_x(imgui.get_cursor_pos().x + offset_x)
            for c in range(nc):
                val = state.bitmap[r, c]
                if val > 0:
                    imgui.push_style_color(imgui.Col_.button, (0.18, 0.62, 0.28, 1.0))
                    imgui.push_style_color(imgui.Col_.button_hovered, (0.28, 0.72, 0.38, 1.0))
                else:
                    imgui.push_style_color(imgui.Col_.button, (0.22, 0.22, 0.22, 1.0))
                    imgui.push_style_color(imgui.Col_.button_hovered, (0.35, 0.35, 0.35, 1.0))
                if imgui.button(f"##bm_{r}_{c}", imgui.ImVec2(CELL_W, CELL_H)):
                    if not bmap_changed:
                        state.push_undo("Pattern")
                    state.bitmap[r, c] = 0.0 if val > 0 else 1.0
                    bmap_changed = True
                imgui.pop_style_color(2)
                if c < nc - 1:
                    imgui.same_line()
        imgui.pop_style_var()
        if bmap_changed:
            state.on_bitmap_change()

    elif stage == 2:
        imgui.text("Yarn Radius")
        imgui.text_wrapped("Set the main yarn tube size before adding separate surface fibers.")
        params_changed = False
        for name, label in (('radius', 'Tube radius'), ('ellipse_ratio', 'Oval width ratio')):
            idx = geometry_param_index(name)
            lo, hi = geometry_param_range(idx)
            changed, new_val = imgui.slider_float(f"{label}##{name}", state.params[idx], lo, hi, "%.4f")
            if imgui.is_item_activated():
                state.push_undo(label)
            if changed:
                state.params[idx] = new_val
                params_changed = True
        if params_changed:
            rebuild_current_mesh()

    elif stage == 3:
        imgui.text("Surface Fibers")
        draw_surface_fiber_controls("Surface fibers")

    elif stage == 4:
        imgui.text("Geometry Parameters")
        quality_changed = False
        changed_loop_res, new_loop_res = imgui.slider_int(
            "Path smoothness##mesh_loop_res",
            int(CONFIG['geometry']['loop_res']),
            8,
            96,
        )
        if changed_loop_res:
            CONFIG['geometry']['loop_res'] = int(new_loop_res)
            quality_changed = True
        changed_segments, new_segments = imgui.slider_int(
            "Fiber roundness##mesh_segments",
            int(CONFIG['geometry']['segments']),
            8,
            64,
        )
        if changed_segments:
            CONFIG['geometry']['segments'] = int(new_segments)
            quality_changed = True
        if quality_changed:
            rebuild_current_mesh()

        params_changed = False
        for i, name in enumerate(geometry_parameter_names()):
            if name in ('radius', 'ellipse_ratio'):
                continue
            span = state.loop_height_span(name)
            if span is not None and span > state.bitmap_size[0]:
                continue
            lo, hi = geometry_param_range(i)
            changed, new_val = imgui.slider_float(f"##p{i}", state.params[i], lo, hi, format=f"{name}: %.3f")
            if imgui.is_item_activated():
                state.push_undo(name)
            if changed:
                state.params[i] = new_val
                params_changed = True
        if imgui.small_button("Fit loop heights to rows##fit_loop_heights"):
            state.push_undo("Loop heights")
            state.fit_loop_heights_to_rows()
            params_changed = False
        if params_changed:
            rebuild_current_mesh()

    elif stage == 5:
        imgui.text("Material")
        _, state.model_alpha = imgui.slider_float("Opacity##mdl", state.model_alpha, 0.0, 1.0, "%.2f")
        changed_mode, use_row_colors = imgui.checkbox("Control colors per row##rowcolors", state.use_row_colors)
        if changed_mode:
            state.push_undo("Color mode")
            state.use_row_colors = use_row_colors
            rebuild_current_mesh()
        if not state.use_row_colors:
            changed_c, new_col = imgui.color_edit3("One color for all##single_color", (float(state.single_model_color[0]), float(state.single_model_color[1]), float(state.single_model_color[2])))
            if imgui.is_item_activated():
                state.push_undo("Single color")
            if changed_c:
                state.single_model_color = np.array(new_col, dtype=np.float32)
                rebuild_current_mesh()
        else:
            colors_changed = False
            for row_idx in range(int(state.bitmap_size[0])):
                col = state.row_colors[row_idx]
                changed_c, new_col = imgui.color_edit3(f"Row {row_idx + 1}##row_color_{row_idx}", (float(col[0]), float(col[1]), float(col[2])))
                if imgui.is_item_activated():
                    state.push_undo("Row color")
                if changed_c:
                    state.row_colors[row_idx] = list(new_col)
                    colors_changed = True
            if colors_changed:
                rebuild_current_mesh()

    elif stage == 6:
        imgui.text("Texture")
        imgui.text_wrapped("Tune procedural yarn texture in the live viewport.")
        changed_tex, new_tex = imgui.color_edit3(
            "Tint##render_texture",
            (
                float(state.render_texture_color[0]),
                float(state.render_texture_color[1]),
                float(state.render_texture_color[2]),
            ),
        )
        if imgui.is_item_activated():
            state.push_undo("Render texture")
        if changed_tex:
            state.render_texture_color = np.array(new_tex, dtype=np.float32)
        if imgui.small_button("Neutral tint##texture"):
            state.push_undo("Render texture")
            state.render_texture_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        imgui.same_line()
        if imgui.small_button("Copy material##texture"):
            state.push_undo("Render texture")
            state.render_texture_color = state.single_model_color.copy()
        for group in TEXTURE_CONTROL_GROUPS:
            imgui.separator()
            imgui.text(group['title'])
            for control in group['controls']:
                key = control['key']
                label = control['label']
                changed, new_val = imgui.slider_float(
                    f"{label}##{key}",
                    state[key],
                    control['min'],
                    control['max'],
                    control['format'],
                )
                if imgui.is_item_activated():
                    state.push_undo(label)
                if changed:
                    state[key] = new_val

        imgui.separator()
        for preset in TEXTURE_PRESET_BUTTONS:
            if preset.get('same_line'):
                imgui.same_line()
            if imgui.small_button(f"{preset['label']}##texture"):
                state.apply_texture_preset(preset['preset'])

    elif stage == 7:
        imgui.text("Lighting")
        imgui.text_wrapped("Lighting changes are shown immediately in the viewport and used by the final render.")
        changed_light, new_light = imgui.color_edit3(
            "Light color##render_light",
            (
                float(state.render_light_color[0]),
                float(state.render_light_color[1]),
                float(state.render_light_color[2]),
            ),
        )
        if imgui.is_item_activated():
            state.push_undo("Light color")
        if changed_light:
            state.render_light_color = np.array(new_light, dtype=np.float32)
        changed_intensity, new_intensity = imgui.slider_float(
            "Light intensity##render_light",
            state.render_light_intensity,
            0.05,
            3.0,
            "%.2f",
        )
        if imgui.is_item_activated():
            state.push_undo("Light intensity")
        if changed_intensity:
            state.render_light_intensity = new_intensity
        if imgui.small_button("Reset lighting##render_light"):
            state.push_undo("Lighting")
            state.render_light_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            state.render_light_intensity = 0.9

    elif stage == 8:
        imgui.text("Spline Refinement")
        imgui.text("Mode")
        imgui.same_line()
        if imgui.radio_button("Param", state.mode == 'parameter') and state.mode != 'parameter':
            state.push_undo("Mode")
            state.mode = 'parameter'
            renderer.set_ctrl_pts([])
            state.rebuild_param_mesh()
        imgui.same_line()
        if imgui.radio_button("Spline", state.mode == 'spline') and state.mode != 'spline':
            state.push_undo("Mode")
            state.mode = 'spline'
            state.spline.init_from_params(state.params)
            state.rebuild_spline_mesh()
        ch_spl, new_spl = imgui.slider_int("Samples/loop##spl", state.samples_per_loop, 2, 20)
        if ch_spl:
            state.push_undo("Spline resolution")
            state.samples_per_loop = new_spl
            state.spline.samples_per_loop = new_spl
            if state.mode == 'spline':
                state.spline.init_from_params(state.params)
                state.rebuild_spline_mesh()
        changed_fix, new_fix = imgui.checkbox("Auto-fix endpoints for copies##spline_endpoint_fix", state.auto_fix_spline_endpoints)
        if changed_fix:
            state.push_undo("Spline endpoint fix")
            state.auto_fix_spline_endpoints = new_fix
            if state.mode == 'spline':
                state.rebuild_spline_mesh()
        if state.mode == 'spline':
            imgui.text(f"Points: {len(state.spline.flat_pts)}")
            if state.hover_idx >= 0:
                imgui.text(f"Hover: {state.hover_idx}")
            if state.selected_idx >= 0:
                imgui.text(f"Selected: {state.selected_idx}")
            changed_step, new_step = imgui.slider_float(
                "Keyboard step##spline_keyboard_step",
                float(state.spline_keyboard_step),
                0.001,
                0.2,
                "%.3f",
            )
            if changed_step:
                state.spline_keyboard_step = float(new_step)
            imgui.text_wrapped("Select a white point, then drag it in the viewport or use the gizmo arrows.")
            imgui.separator()
            imgui.text("Fibers Follow Spline")
            draw_surface_fiber_controls("Spline fibers")
        else:
            imgui.text_wrapped("Switch to Spline mode when the parameter model is already close.")

    elif stage == 9:
        imgui.text("Review and Render")
        _, state.mi_cam_dist_mult = imgui.slider_float("Render dist mult##mi", state.mi_cam_dist_mult, 0.3, 3.0, "%.2f")
        _, state.mi_cam_fov = imgui.slider_float("Render FoV##mi", state.mi_cam_fov, 10.0, 120.0, "%.1f")
        imgui.separator()
        avail_controls_w = imgui.get_content_region_avail().x
        btn_w = max(120, (avail_controls_w - imgui.get_style().item_spacing.x) * 0.5)
        
        is_rendering = state.is_rendering
        if is_rendering:
            imgui.begin_disabled()
        if imgui.button("Render##btn" if not is_rendering else "Rendering...", (btn_w, 0)):
            state.start_background_render()
        if is_rendering:
            imgui.end_disabled()
        
        imgui.same_line()
        is_optimizing = state.is_optimizing
        import mitsuba as mi
        mitsuba_ad_available = "_ad_" in (mi.variant() or "")
        if is_optimizing or not mitsuba_ad_available:
            imgui.begin_disabled()
        if imgui.button("Optimize##btn" if not is_optimizing else "Running...", (btn_w, 0)):
            state.start_background_optimize()
        if is_optimizing or not mitsuba_ad_available:
            imgui.end_disabled()
        if not mitsuba_ad_available:
            imgui.text_disabled(f"Optimize needs CUDA/LLVM AD. Current Mitsuba: {mi.variant()}")
        
        imgui.separator()
        half_w = max(100, (imgui.get_content_region_avail().x - imgui.get_style().item_spacing.x) * 0.5)
        if imgui.button("Save params...", (half_w, 0)):
            path = _pick_file('save', state.save_path)
            if path:
                state.save_params(path)
        imgui.same_line()
        if imgui.button("Load params...", (half_w, 0)):
            path = _pick_file('load', state.load_path)
            if path:
                state.load_params(path)
        changed_auto, new_auto = imgui.checkbox("Autosave", state.autosave_enabled)
        if changed_auto:
            state.autosave_enabled = new_auto
            if new_auto:
                state.autosave_last_time = 0.0
        if state.status_msg:
            imgui.spacing()
            imgui.text_colored((0.4, 0.9, 0.4, 1.0), state.status_msg)

    state.maybe_autosave()

    imgui.end()


def draw_viewport(state, renderer, ref_tex, window):
    imgui.set_next_window_pos((360, 20), cond=imgui.Cond_.first_use_ever)
    imgui.set_next_window_size((840, 820), cond=imgui.Cond_.first_use_ever)
    imgui.begin("3D View", flags=imgui.WindowFlags_.no_scroll_with_mouse)
    imgui.text("3D Viewport")

    avail_x, avail_y = imgui.get_content_region_avail()
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
    bg_zoom = state.camera.zoom_factor() if state.ref_bg_lock_zoom else 1.0

    bg_uniforms = {
        'bg_scale_x':  state.ref_bg_scale[0] * bg_zoom,
        'bg_scale_y':  state.ref_bg_scale[1] * bg_zoom,
        'bg_rotation': state.ref_bg_rotation,
        'bg_offset_x': state.ref_bg_offset[0],
        'bg_offset_y': state.ref_bg_offset[1],
        'vp_aspect':   disp_w / disp_h,
        'img_aspect':  ref_tex.width / ref_tex.height if ref_tex is not None else 1.0,
    }

    renderer.render(
        mvp, mv,
        state.get_material_uniforms(),
        state.hover_idx, state.selected_idx,
        hover_mesh_idx=state.hover_mesh_idx,
        selected_mesh_idx=state.selected_mesh_idx,
        visible_rows=state.row_visible,
        bg_tex      = ref_tex if state.show_ref_bg else None,
        bg_alpha    = state.ref_bg_alpha,
        bg_uniforms = bg_uniforms
    )

    # Display FBO
    drawn_rect = draw_fitted_texture(
        renderer.texture_id,
        disp_w,
        disp_h,
        avail_x,
        avail_y,
        flip_y=True,
        zoom=state.viewport_zoom,
        pan=state.viewport_pan,
    )
    if drawn_rect is not None:
        origin_x, origin_y, draw_w, _ = drawn_rect
        state.vp_origin = np.array([origin_x, origin_y], dtype=np.float32)
        state.vp_scale = float(draw_w / max(float(disp_w), 1.0))
    is_hovered = imgui.is_item_hovered()
    state.mouse_in_vp = is_hovered

    # ImGuizmo
    if state.mode == 'spline' and state.selected_idx >= 0:
        local_pos = state.spline.flat_pts[state.selected_idx].astype(np.float32)
        pos = transform_points([local_pos], model_mat)[0].astype(np.float32)
        M16 = imguizmo.im_guizmo.Matrix16

        view_m = M16(); view_m.values[:] = state.camera.view().T.flatten()
        proj_m = M16(); proj_m.values[:] = state.camera.proj(disp_w, disp_h).T.flatten()

        mat = np.eye(4, dtype=np.float32)
        mat[0, 3] = pos[0]; mat[1, 3] = pos[1]; mat[2, 3] = pos[2]
        obj_m = M16(); obj_m.values[:] = mat.T.flatten()

        imguizmo.im_guizmo.set_orthographic(False)
        imguizmo.im_guizmo.set_drawlist()
        imguizmo.im_guizmo.set_rect(draw_pos.x, draw_pos.y, disp_w, disp_h)
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
            state.spline.move(state.selected_idx, new_local)
            state.rebuild_spline_mesh()
        elif state.gizmo_edit_active and not imguizmo.im_guizmo.is_using():
            state.gizmo_edit_active = False

    # Mouse interaction inside the viewport
    mx, my = imgui.get_mouse_pos()
    alignment_locked = bool(state.show_ref_bg and state.ref_bg_lock_zoom)
    viewport_scale = max(float(state.vp_scale), 1e-6)
    lx = (mx - state.vp_origin[0]) / viewport_scale
    ly = (my - state.vp_origin[1]) / viewport_scale

    if is_hovered:
        io = imgui.get_io()
        state.hover_mesh_idx = renderer.pick_mesh_index(model_mat, state.camera, disp_w, disp_h, lx, ly, visible_rows=state.row_visible)

        if state.selected_mesh_idx >= 0 and io.key_alt:
            # Closest cursor available in Dear ImGui; used as eyedropper cue.
            imgui.set_mouse_cursor(imgui.MouseCursor_.hand)

        if imgui.is_mouse_clicked(imgui.MouseButton_.left) and not io.key_shift and not io.key_alt:
            state.selected_mesh_idx = state.hover_mesh_idx

        if state.selected_mesh_idx >= 0 and io.key_alt and imgui.is_mouse_clicked(imgui.MouseButton_.left):
            sampled = renderer.sample_color(lx, ly)
            row_idx = renderer.get_row_for_mesh_index(state.selected_mesh_idx)
            if sampled is not None and row_idx is not None and int(state.bitmap_size[0]) > 0:
                row_idx = int(row_idx) % int(state.bitmap_size[0])
                state.push_undo("Pick yarn color")
                state.use_row_colors = True
                state.row_colors[row_idx] = sampled.tolist()
                if state.mode == 'spline':
                    state.rebuild_spline_mesh()
                else:
                    state.rebuild_param_mesh()
    else:
        state.hover_mesh_idx = -1

    if is_hovered:
        curr = (mx, my)

        io = imgui.get_io()
        if io.mouse_wheel != 0 and not io.key_shift:
            zoom_factor = float(np.exp(io.mouse_wheel * 0.1))
            state.viewport_zoom = float(np.clip(float(state.viewport_zoom) * zoom_factor, 0.25, 8.0))

        if state.get('prev_mouse') is not None:
            prev = state.prev_mouse
            dx = mx - prev[0]
            dy = my - prev[1]
            lmb = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT)  == glfw.PRESS
            rmb = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
            mmb = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE)== glfw.PRESS
            shift_down = imgui.get_io().key_shift
            alt_down = imgui.get_io().key_alt
            can_transform_model = False
            direct_spline_drag = (
                state.mode == 'spline'
                and state.selected_idx >= 0
                and lmb
                and not shift_down
                and not alt_down
                and (state.spline_point_drag_active or state.hover_idx == state.selected_idx)
            )

            if direct_spline_drag:
                if not state.spline_point_drag_active:
                    state.push_undo("Spline point")
                    state.spline_point_drag_active = True
                view = state.camera.view()
                right = view[0, :3]
                up = view[1, :3]
                drag_speed = state.camera.dist * 0.00045 / max(float(state.model_scale), 1e-6)
                world_delta = (right * dx - up * dy) * drag_speed
                local_delta = np.linalg.inv(model_mat)[:3, :3] @ world_delta
                old_local = state.spline.flat_pts[state.selected_idx]
                new_local = old_local + local_delta.astype(np.float32)
                if np.linalg.norm(new_local - old_local) > 1e-6:
                    state.spline.move(state.selected_idx, new_local)
                    state.rebuild_spline_mesh()
            elif shift_down and state.mode == 'spline' and state.selected_idx >= 0 and (
                    imguizmo.im_guizmo.is_using() or imguizmo.im_guizmo.is_over()):
                can_transform_model = False
            elif shift_down and state.mode == 'spline' and state.hover_idx >= 0:
                can_transform_model = False
            else:
                can_transform_model = shift_down and not alignment_locked

            if can_transform_model and (lmb or rmb or mmb):
                if not state.model_drag_undo_active:
                    state.push_undo("Model transform")
                    state.model_drag_undo_active = True
                if lmb:
                    sens = 0.005
                    state.model_rot[1] += dx * sens
                    state.model_rot[0] += dy * sens
                elif rmb:
                    view = state.camera.view()
                    right = view[0, :3]
                    up    = view[1, :3]
                    t_sens = state.camera.dist * 0.003
                    state.model_t += (right * dx - up * dy) * t_sens
                elif mmb:
                    scale_factor = float(np.exp(-dy * 0.01))
                    state.model_scale = float(np.clip(float(state.model_scale) * scale_factor, 0.05, 20.0))
            elif (
                (lmb or mmb or rmb)
                and not shift_down
                and not (state.mode == 'spline' and state.selected_idx >= 0)
            ):
                state.viewport_pan[0] += dx
                state.viewport_pan[1] += dy

            if not lmb:
                state.spline_point_drag_active = False

        if state.mode == 'spline' and state.selected_idx >= 0 and len(state.spline.flat_pts) > 0:
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
                state.spline.move(
                    state.selected_idx,
                    state.spline.flat_pts[state.selected_idx] + local_delta.astype(np.float32),
                )
                state.rebuild_spline_mesh()
            else:
                state.spline_keyboard_edit_active = False

        if (
            not alignment_locked
            and imgui.get_io().key_shift
            and (
                glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
                or glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
                or glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
            )
        ):
            state.model_rot_dragging = True
        else:
            state.model_rot_dragging = False
            state.model_drag_undo_active = False

        # Spline handle hover + select
        if state.mode == 'spline' and len(state.spline.flat_pts) > 0:
            gizmo_active = state.selected_idx >= 0 and (
                imguizmo.im_guizmo.is_using() or imguizmo.im_guizmo.is_over()
            )
            if not gizmo_active:
                world_pts = transform_points(state.spline.flat_pts, model_mat)
                homo = np.column_stack((world_pts, np.ones(len(world_pts), dtype=np.float32)))
                view_proj = state.camera.proj(disp_w, disp_h) @ state.camera.view()
                clip = homo @ view_proj.T
                valid = clip[:, 3] > 1e-6
                ndc = np.zeros((len(world_pts), 3), dtype=np.float32)
                ndc[valid] = clip[valid, :3] / clip[valid, 3:4]
                screen = np.column_stack((
                    (ndc[:, 0] * 0.5 + 0.5) * disp_w,
                    (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * disp_h,
                ))
                in_view = (
                    valid
                    & (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0)
                    & (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
                )
                d2 = np.sum((screen - np.array([lx, ly], dtype=np.float32)) ** 2, axis=1)
                d2[~in_view] = np.inf
                best_i = int(np.argmin(d2))
                best_i = best_i if d2[best_i] <= 16.0 ** 2 else -1
                state.hover_idx = best_i
                if imgui.is_mouse_clicked(imgui.MouseButton_.left):
                    state.selected_idx = best_i

        state.prev_mouse = curr
    else:
        state.prev_mouse = None

    imgui.end()


def draw_mitsuba_panel(state):
    imgui.set_next_window_pos((1220, 20), cond=imgui.Cond_.first_use_ever)
    imgui.set_next_window_size((420, 360), cond=imgui.Cond_.first_use_ever)
    imgui.begin("Mitsuba Render")
    imgui.text("Mitsuba Render")
    if state.render_tex:
        avail_x, avail_y = imgui.get_content_region_avail()
        draw_fitted_texture(
            state.render_tex.glo,
            state.render_tex.width,
            state.render_tex.height,
            avail_x,
            avail_y,
        )
    else:
        imgui.spacing()
        imgui.text_disabled("Render output will appear here.")
        imgui.text_disabled("Use the button in the left rail.")
    imgui.end()


def draw_reference_image_panel(state, ref_tex):
    imgui.set_next_window_pos((1220, 400), cond=imgui.Cond_.first_use_ever)
    imgui.set_next_window_size((420, 440), cond=imgui.Cond_.first_use_ever)
    imgui.begin("Reference Image")
    imgui.text("Reference Image")
    avail_x, avail_y = imgui.get_content_region_avail()
    draw_fitted_texture(
        ref_tex.glo,
        ref_tex.width,
        ref_tex.height,
        avail_x,
        avail_y,
    )
    imgui.end()
