import os
import numpy as np
import glfw
import tkinter as tk
from tkinter import filedialog as _filedialog
from imgui_bundle import imgui, imguizmo

from rendering import draw_fitted_texture, transform_points

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
    stage_idx = int(np.clip(state.workflow_step, 0, len(state.workflow_stages) - 1))
    title, subtitle = state.workflow_stages[stage_idx]
    imgui.text(f"Step {stage_idx + 1} of {len(state.workflow_stages)}")
    imgui.text_colored((0.92, 0.74, 0.34, 1.0), title)
    imgui.text_wrapped(subtitle)
    imgui.spacing()

    avail_w = imgui.get_content_region_avail().x
    dot_w = max(18.0, (avail_w - (len(state.workflow_stages) - 1) * 4.0) / len(state.workflow_stages))
    imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(4, 2))
    for i, _ in enumerate(state.workflow_stages):
        active = i == stage_idx
        color = (0.25, 0.55, 0.85, 1.0) if active else (0.22, 0.22, 0.22, 1.0)
        hover = (0.35, 0.65, 0.95, 1.0) if active else (0.34, 0.34, 0.34, 1.0)
        imgui.push_style_color(imgui.Col_.button, color)
        imgui.push_style_color(imgui.Col_.button_hovered, hover)
        if imgui.button(f"{i + 1}##stage_{i}", imgui.ImVec2(dot_w, 22)):
            state.workflow_step = i
        imgui.pop_style_color(2)
        if i < len(state.workflow_stages) - 1:
            imgui.same_line()
    imgui.pop_style_var()

    imgui.spacing()
    back_disabled = stage_idx == 0
    next_disabled = stage_idx == len(state.workflow_stages) - 1
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
        state.workflow_step = min(len(state.workflow_stages) - 1, stage_idx + 1)
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

    if imgui.button("Reset initial model##reset_unit_model_global", (-1, 0)):
        state.reset_to_unit_model()
    imgui.separator()

    stage = int(np.clip(state.workflow_step, 0, len(state.workflow_stages) - 1))

    def rebuild_current_mesh():
        state.rebuild_spline_mesh()

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
    changed_x, new_copies_x = imgui.slider_int("Copies X", int(state.display_copies[0]), 0, 20)
    changed_y, new_copies_y = imgui.slider_int("Copies Y", int(state.display_copies[1]), 0, 20)
    if changed_x or changed_y:
        state.push_undo("Display copies")
        state.display_copies = np.array([new_copies_x, new_copies_y], dtype=np.int32)
        state.rebuild_spline_mesh(preserve_model_placement=True)
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
        imgui.text_wrapped("Viewport controls: Shift+RMB translate.")
        imgui.text_wrapped("Camera controls: Mouse wheel zoom camera. Drag in the viewport to move the model.")
        labels = ["X", "Y", "Z"]
        imgui.text_colored((0.4, 0.8, 0.8, 1.0), "Position")
        # Vectorized model translation
        for idx in range(3):
            changed, val = imgui.drag_float(f"{labels[idx]}##align_t_{idx}", float(state.model_t[idx]), 0.01, -100.0, 100.0, f"{labels[idx]}: %.3f")
            if imgui.is_item_activated():
                state.push_undo("Model position")
            if changed:
                state.model_t[idx] = val

        imgui.text_disabled("Scale via bbox handles in the viewport.")

        if imgui.small_button("Center model##align"):
            state.push_undo("Center model")
            state.center_model_on_view()
        imgui.same_line()
        if imgui.small_button("Reset transform##align"):
            state.push_undo("Reset transform")
            state.center_model_on_view()
        if alignment_locked:
            imgui.end_disabled()

    elif stage == 1:
        imgui.text("Pattern and Rows")
        max_rows = int(state.config['knit_parameters']['bitmap_rows'])
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
            idx = state._pidx[name]
            lo, hi = state.config["knit_parameters"]["parameters"][idx]["range"]
            changed, new_val = imgui.slider_float(f"{label}##{name}", state.params[idx], lo, hi, "%.4f")
            if imgui.is_item_activated():
                state.push_undo(label)
            if changed:
                state.params[idx] = new_val
                params_changed = True
        if params_changed:
            state.nudge_spline_from_params()

    elif stage == 3:
        imgui.text("Surface Fibers")
        draw_surface_fiber_controls("Surface fibers")

    elif stage == 4:
        imgui.text("Geometry Parameters")
        quality_changed = False
        changed_loop_res, new_loop_res = imgui.slider_int(
            "Path smoothness##mesh_loop_res",
            int(state.config['knit_parameters']['loop_res']),
            8,
            96,
        )
        if changed_loop_res:
            state.config['knit_parameters']['loop_res'] = int(new_loop_res)
            quality_changed = True
        changed_segments, new_segments = imgui.slider_int(
            "Fiber roundness##mesh_segments",
            int(state.config['knit_parameters']['segments']),
            8,
            64,
        )
        if changed_segments:
            state.config['knit_parameters']['segments'] = int(new_segments)
            quality_changed = True
        if quality_changed:
            rebuild_current_mesh()

        params_changed = False
        for i, pd in enumerate(state.config["knit_parameters"]["parameters"]):
            if pd["name"] in ('radius', 'ellipse_ratio'):
                continue
            span = state.loop_height_span(pd["name"])
            if span is not None and span > state.bitmap_size[0]:
                continue
            lo, hi = pd["range"]
            changed, new_val = imgui.slider_float(f"##p{i}", state.params[i], lo, hi, format=f"{pd['name']}: %.3f")
            if imgui.is_item_activated():
                state.push_undo(pd["name"])
            if changed:
                state.params[i] = new_val
                params_changed = True
        if imgui.small_button("Rebuild from params##rebuild_params"):
            state.push_undo("Rebuild from params")
            state.rebuild_spline_from_params()
        imgui.same_line()
        if imgui.small_button("Debug compare rebuild##rebuild_params"):
            stats = state.debug_compare_to_fresh_rebuild()
            if stats.get('ok'):
                state.status_msg = (
                    f"Rebuild delta: mean={stats['mean']:.6f}, "
                    f"p95={stats['p95']:.6f}, max={stats['max']:.6f}"
                )
            else:
                state.status_msg = f"Rebuild compare failed: {stats.get('reason', 'unknown')}"
        if imgui.small_button("Fit loop heights to rows##fit_loop_heights"):
            state.push_undo("Loop heights")
            state.fit_loop_heights_to_rows()
            params_changed = False
        if params_changed:
            state.nudge_spline_from_params()

    elif stage == 5:
        imgui.text("Material")
        _, state.model_alpha = imgui.slider_float("Opacity##mdl", state.model_alpha, 0.0, 1.0, "%.2f")
        changed_mode, use_row_colors = imgui.checkbox("Control colors per row##rowcolors", state.use_row_colors)
        if changed_mode:
            state.push_undo("Color mode")
            state.use_row_colors = use_row_colors
            rebuild_current_mesh()
        if imgui.small_button("Pick row color from image##pick_ref_row_color"):
            state.reference_color_pick_active = True
            state.use_row_colors = True
            state.status_msg = "Pick mode: click a row over the reference image"
        imgui.same_line()
        imgui.text_disabled("or hold Alt and click a row")
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
        for group in state.texture_control_groups:
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
        for preset in state.texture_preset_buttons:
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

        imgui.separator()
        imgui.text("Ambient Occlusion (AO)")
        changed_ao_s, new_ao_s = imgui.slider_float(
            "AO Strength##render_ao",
            state.render_ao_strength,
            0.0,
            2.0,
            "%.2f",
        )
        if imgui.is_item_activated():
            state.push_undo("AO Strength")
        if changed_ao_s:
            state.render_ao_strength = new_ao_s

        changed_ao_r, new_ao_r = imgui.slider_float(
            "AO Radius##render_ao",
            state.render_ao_radius,
            0.01,
            1.0,
            "%.2f",
        )
        if imgui.is_item_activated():
            state.push_undo("AO Radius")
        if changed_ao_r:
            state.render_ao_radius = new_ao_r

        if imgui.small_button("Reset lighting##render_light"):
            state.push_undo("Lighting")
            state.render_light_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            state.render_light_intensity = 0.9
            state.render_ao_strength = 0.5
            state.render_ao_radius = 0.15

    elif stage == 8:
        imgui.text("Spline Refinement")
        imgui.text_disabled("Spline is the canonical editing path.")
        if imgui.small_button("Rebuild spline from params##spline_rebuild"):
            state.push_undo("Rebuild from params")
            state.rebuild_spline_from_params()
        ch_spl, new_spl = imgui.slider_int("Samples/loop##spl", state.samples_per_loop, 2, 20)
        if ch_spl:
            state.push_undo("Spline resolution")
            state.samples_per_loop = new_spl
            state.rebuild_spline_from_params()
        changed_step, new_step = imgui.slider_float(
            "Keyboard step##spline_keyboard_step",
            float(state.spline_keyboard_step),
            0.001,
            0.2,
            "%.3f",
        )
        if changed_step:
            state.spline_keyboard_step = float(new_step)
        imgui.text(f"Points: {len(state.flat_pts)}")
        if state.hover_idx >= 0:
            imgui.text(f"Hover: {state.hover_idx}")
        if state.selected_idx >= 0:
            imgui.text(f"Selected: {state.selected_idx}")
        imgui.text_wrapped("Select a white point, then drag the gizmo arrows or use Arrow/WASD/Q/E keys to refine it.")
        imgui.separator()
        imgui.text("Fibers Follow Spline")
        draw_surface_fiber_controls("Spline fibers")

    elif stage == 9:
        imgui.text("Review parameters")
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
        if imgui.button("Reset all to initial##reset_all", (half_w, 0)):
            state.reset_to_initial()
        imgui.same_line()
        if imgui.button("Reset initial model##reset_unit_model", (half_w, 0)):
            state.reset_to_unit_model()
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
    bg_zoom = state.camera.zoom_factor()

    render_hover_idx = state.hover_idx
    render_selected_idx = state.selected_idx
    visible_ctrl_indices = np.empty((0,), dtype=np.int32)
    visible_ctrl_index_map = {}
    if state.mode == 'spline':
        visible_chunks = []
        for row_idx, row in enumerate(state.ctrl_rows):
            if not state.row_visible[row_idx]:
                continue
            start = state._row_starts[row_idx]
            end = start + len(row)
            visible_chunks.append(np.arange(start, end, dtype=np.int32))
        if visible_chunks:
            visible_ctrl_indices = np.concatenate(visible_chunks)
            visible_ctrl_pts = state.flat_pts[visible_ctrl_indices]
        else:
            visible_ctrl_pts = np.empty((0, 3), dtype=np.float32)
        renderer.set_ctrl_pts(visible_ctrl_pts)
        visible_ctrl_index_map = {
            int(flat_idx): int(local_idx)
            for local_idx, flat_idx in enumerate(visible_ctrl_indices.tolist())
        }
        render_hover_idx = visible_ctrl_index_map.get(int(state.hover_idx), -1)
        render_selected_idx = visible_ctrl_index_map.get(int(state.selected_idx), -1)

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

    renderer.render(
        mvp, mv,
        state.get_material_uniforms(),
        render_hover_idx, render_selected_idx,
        hover_mesh_idx=state.hover_mesh_idx,
        selected_mesh_idx=state.selected_mesh_idx,
        visible_rows=state.row_visible,
        bg_tex      = ref_tex if state.show_ref_bg else None,
        bg_alpha    = state.ref_bg_alpha,
        bg_uniforms = bg_uniforms,
        camera      = state.camera
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
        if state.mode == 'spline' and len(visible_ctrl_indices) > 0:
            ctrl_pts = state.flat_pts[visible_ctrl_indices]
            world_pts = transform_points(ctrl_pts, model_matrix)
            view_proj = state.camera.proj(disp_w, disp_h) @ state.camera.view()
            homo = np.column_stack((world_pts, np.ones(len(world_pts), dtype=np.float32)))
            clip = homo @ view_proj.T
            valid = clip[:, 3] > 1e-6
            if not np.any(valid):
                return None
            ndc = np.zeros((len(world_pts), 3), dtype=np.float32)
            ndc[valid] = clip[valid, :3] / clip[valid, 3:4]
            in_view = (
                valid
                & (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0)
                & (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
            )
            if not np.any(in_view):
                return None
            screen = np.column_stack((
                (ndc[:, 0] * 0.5 + 0.5) * disp_w,
                (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * disp_h,
            ))
            all_pts = screen[in_view]
            x_min, y_min = np.min(all_pts, axis=0)
            x_max, y_max = np.max(all_pts, axis=0)
            w = max(1.0, float(x_max - x_min))
            h = max(1.0, float(y_max - y_min))
            pad_x = 0.10 * w
            pad_y = 0.10 * h
            x_min = float(np.clip(x_min - pad_x, 0.0, disp_w - 1.0))
            y_min = float(np.clip(y_min - pad_y, 0.0, disp_h - 1.0))
            x_max = float(np.clip(x_max + pad_x, 0.0, disp_w - 1.0))
            y_max = float(np.clip(y_max + pad_y, 0.0, disp_h - 1.0))
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
            homo = np.column_stack((world_pts, np.ones(len(world_pts), dtype=np.float32)))
            clip = homo @ view_proj.T
            valid = clip[:, 3] > 1e-6
            if not np.any(valid):
                continue
            ndc = np.zeros((len(world_pts), 3), dtype=np.float32)
            ndc[valid] = clip[valid, :3] / clip[valid, 3:4]
            in_view = (
                valid
                & (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0)
                & (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
            )
            if not np.any(in_view):
                continue
            screen = np.column_stack((
                (ndc[:, 0] * 0.5 + 0.5) * disp_w,
                (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * disp_h,
            ))
            pts_2d.append(screen[in_view])
        if not pts_2d:
            return None
        all_pts = np.concatenate(pts_2d, axis=0)
        x_min, y_min = np.min(all_pts, axis=0)
        x_max, y_max = np.max(all_pts, axis=0)
        pad = 6.0
        x_min = float(np.clip(x_min - pad, 0.0, disp_w - 1.0))
        y_min = float(np.clip(y_min - pad, 0.0, disp_h - 1.0))
        x_max = float(np.clip(x_max + pad, 0.0, disp_w - 1.0))
        y_max = float(np.clip(y_max + pad, 0.0, disp_h - 1.0))
        if x_max - x_min < 12.0 or y_max - y_min < 12.0:
            return None
        return x_min, y_min, x_max, y_max

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
    if gizmo_bounds is not None and is_hovered:
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
        rect_col = imgui.get_color_u32((0.95, 0.95, 0.95, 0.92))
        dl.add_rect((ox + x_min, oy + y_min), (ox + x_max, oy + y_max), rect_col, 0.0, 2.0, 0)
        for i, (hx, hy) in enumerate(bounds_handles(gizmo_bounds)):
            is_hot = (i == hover_handle or i == active_handle)
            fill = imgui.get_color_u32((0.95, 0.65, 0.10, 1.0) if is_hot else (0.96, 0.96, 0.96, 0.95))
            stroke = imgui.get_color_u32((0.12, 0.12, 0.12, 1.0))
            dl.add_circle_filled((ox + hx, oy + hy), handle_radius, fill, 16)
            dl.add_circle((ox + hx, oy + hy), handle_radius, stroke, 16, 1.5)

    # ImGuizmo
    if state.mode == 'spline' and state.selected_idx >= 0 and int(state.selected_idx) in visible_ctrl_index_map:
        local_pos = state.flat_pts[state.selected_idx].astype(np.float32)
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

    # Mouse interaction inside the viewport
    alignment_locked = bool(state.show_ref_bg and state.ref_bg_lock_zoom)

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

        # Bounding-box resize handles (window-like scaling in screen space).
        if gizmo_bounds is not None and hover_handle >= 0 and imgui.is_mouse_clicked(imgui.MouseButton_.left) and not io.key_shift and not io.key_alt:
            state.push_undo("Bounding box scale")
            state.bbox_active_handle = int(hover_handle)
            state.bbox_start_bounds = np.array(gizmo_bounds, dtype=np.float32)
            state.bbox_start_mouse = np.array([lx, ly], dtype=np.float32)
            state.bbox_start_t = np.array(state.model_t, dtype=np.float32)
            if state.mode == 'spline':
                state.bbox_start_ctrl_rows = [row.copy() for row in state.ctrl_rows]
            else:
                state.bbox_start_model_scale = np.array(state.model_scale, dtype=np.float32)
            suppress_mesh_click = True

        active_handle = int(state.get('bbox_active_handle', -1))
        if active_handle >= 0 and lmb_down and state.get('bbox_start_bounds') is not None:
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

            if state.mode == 'spline':
                base_rows = state.get('bbox_start_ctrl_rows')
                if base_rows:
                    visible_rows_local = [
                        row for row_idx, row in enumerate(base_rows)
                        if state.row_visible is None or len(state.row_visible) == 0 or bool(state.row_visible[row_idx % len(state.row_visible)])
                    ]
                    if visible_rows_local:
                        pts = np.concatenate(visible_rows_local, axis=0)
                        pivot = ((pts.min(axis=0) + pts.max(axis=0)) * 0.5).astype(np.float32)
                    else:
                        pivot = np.asarray(state.mesh_center, dtype=np.float32)
                    state.ctrl_rows = [pivot + (row - pivot) * scale_vec for row in base_rows]
                    state._rebuild_spline_points()
                    state.rebuild_spline_mesh()
            else:
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
        elif active_handle >= 0 and not lmb_down:
            state.bbox_active_handle = -1
            state.bbox_start_bounds = None
            state.bbox_start_mouse = None
            state.bbox_start_t = None
            state.bbox_start_model_scale = None
            state.bbox_start_ctrl_rows = None

        state.hover_mesh_idx = renderer.pick_mesh_index(model_mat, state.camera, disp_w, disp_h, lx, ly, visible_rows=state.row_visible)

        color_pick_active = bool(state.reference_color_pick_active or io.key_alt)
        if color_pick_active:
            imgui.set_mouse_cursor(imgui.MouseCursor_.hand)

        if imgui.is_mouse_clicked(imgui.MouseButton_.left) and not io.key_shift and not color_pick_active and not suppress_mesh_click:
            state.selected_mesh_idx = state.hover_mesh_idx

        if color_pick_active and imgui.is_mouse_clicked(imgui.MouseButton_.left):
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
        if active_handle >= 0 and not lmb_down:
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
        state._ensure_spline_radius_rows()
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
            lmb = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT)  == glfw.PRESS
            rmb = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
            mmb = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE)== glfw.PRESS
            shift_down = imgui.get_io().key_shift

            if shift_down and state.mode == 'spline' and state.selected_idx >= 0 and (
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
            elif lmb and not shift_down and not bbox_drag_active and not spline_drag_active and not bool(state.spline_grab_active) and not bool(state.radius_grab_active):
                if not state.model_drag_undo_active:
                    state.push_undo("Model translate")
                    state.model_drag_undo_active = True
                state.model_t += pixel_drag_to_world_delta(dx, dy)

        r_down = glfw.get_key(window, glfw.KEY_R) == glfw.PRESS
        r_pressed = r_down and not bool(state.radius_grab_key_was_down)
        state.radius_grab_key_was_down = r_down
        if r_pressed and not bool(state.radius_grab_active):
            edit_idx = local_radius_edit_index()
            if edit_idx >= 0:
                state.push_undo("Local tube radius")
                state.selected_idx = edit_idx
                state.radius_grab_active = True
                state.radius_grab_point_idx = edit_idx
                state.radius_grab_start_mouse = np.array([mx, my], dtype=np.float32)
                state.radius_grab_start_value = local_radius_value(edit_idx)
                state.radius_grab_start_rows = [row.copy() for row in state.spline_radius_rows]

        if bool(state.radius_grab_active):
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

        if not bool(state.radius_grab_active):
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

        if state.mode == 'spline' and len(visible_ctrl_indices) > 0 and not bool(state.radius_grab_active):
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
                    state.spline_grab_start_pos = state.flat_pts[grab_idx].astype(np.float32).copy()

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

        if state.mode == 'spline' and state.selected_idx >= 0 and int(state.selected_idx) in visible_ctrl_index_map and not bool(state.radius_grab_active):
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
                    state.flat_pts[state.selected_idx] + local_delta.astype(np.float32),
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
        if state.mode == 'spline' and len(visible_ctrl_indices) > 0:
            gizmo_active = state.selected_idx >= 0 and (
                imguizmo.im_guizmo.is_using() or imguizmo.im_guizmo.is_over()
            )
            if not gizmo_active:
                visible_ctrl_pts = state.flat_pts[visible_ctrl_indices]
                world_pts = transform_points(visible_ctrl_pts, model_mat)
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
                best_i = int(visible_ctrl_indices[best_i]) if d2[best_i] <= 16.0 ** 2 else -1
                state.hover_idx = best_i
                if imgui.is_mouse_clicked(imgui.MouseButton_.left):
                    state.selected_idx = best_i

        state.prev_mouse = curr
    else:
        state.prev_mouse = None

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
