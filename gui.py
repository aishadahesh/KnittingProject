"""imgui panels for the knitting app: menu bar, sidebar, and 3D viewport.

Module layout after the split (dependencies point one way, no cycles):

    gui.py             this file -- imgui panels and per-mode UI
      |- scanner_core      pure scanner/puzzle logic, no imgui
      |- gui_database      Database Mode UI            -> scanner_core
      `- embedded_scanner  EmbeddedMujocoScanner       -> scanner_core

`app.py` imports only draw_menu_bar / draw_sidebar / draw_viewport from here.
Names moved into the modules above kept their original spelling and are
re-imported below, so call sites in this file are unchanged.
"""
import os
import json
import numpy as np
import glfw
import time
from types import SimpleNamespace
from pathlib import Path

import tkinter as tk
from tkinter import filedialog as _filedialog
from imgui_bundle import imgui, imguizmo
from PIL import Image, ImageDraw

from rendering import draw_fitted_texture, pil_to_texture, transform_points

# Split out of this file; imported under their original names so call sites
# elsewhere in gui.py stay unchanged.
from scanner_core import (
    _as_rgba,
    _ensure_scanner_shared_colors,
    _generate_scanner_random_patterns,
    _puzzle_build_seamless_tile,
    _puzzle_capture_rect,
    _puzzle_project_points,
    _puzzle_tiled_control_points,
    _scan_render_tiled_pattern_image,
    _scanner_base_palette,
    _scanner_batch_colors_for_simulator,
    _scanner_batch_texture_size,
    _scanner_capture_image_size,
    _scanner_display_batch_colors,
    _scanner_estimated_cell_colors,
    _scanner_lighting_settings,
    _scanner_material_uniforms,
    _scanner_pattern_dimensions,
    _scanner_pattern_repeats,
    _scanner_repeat_spacing,
    _scanner_shared_cell_color_sets,
    _scanner_storage,
)
from gui_database import _draw_database_summary_page
from embedded_scanner import EmbeddedMujocoScanner

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


def _state_scanner_model_curves(state):
    curves = []
    period = np.asarray(getattr(state, 'period_offset_x', [1.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
    if period.size < 2:
        period = np.array([1.0, 0.0], dtype=np.float32)
    for row in getattr(state, 'ctrl_rows', []) or []:
        row = np.asarray(row, dtype=np.float32)
        if row.ndim != 2 or row.shape[0] < 2:
            continue
        cp = row[:, :2]
        cp_aug = np.vstack((cp, cp[0] + period[:2]))
        seg_lens = np.maximum(np.linalg.norm(np.diff(cp_aug, axis=0), axis=1), 1e-6)
        t = np.concatenate(([0.0], np.cumsum(seg_lens))).astype(np.float32)
        samples = max(48, min(120, int(len(cp) * 6)))
        to = np.linspace(float(t[0]), float(t[-1]), samples, dtype=np.float32)
        detrended = cp_aug - period[:2][None, :] * (t / max(float(t[-1]), 1e-6))[:, None]
        if len(cp) == 2:
            pts = np.column_stack([np.interp(to, t, detrended[:, axis]) for axis in range(2)])
        else:
            try:
                from scipy.interpolate import CubicSpline
                pts = np.column_stack([CubicSpline(t, detrended[:, axis], bc_type="periodic")(to) for axis in range(2)])
            except Exception:
                pts = np.column_stack([np.interp(to, t, detrended[:, axis]) for axis in range(2)])
        pts = pts + period[:2][None, :] * (to / max(float(t[-1]), 1e-6))[:, None]
        curves.append(pts.astype(np.float32))

    if not curves:
        return None
    all_pts = np.vstack(curves)
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    scale = 0.92 / float(max(span[0], span[1]))
    center = (min_xy + max_xy) * 0.5
    return [((curve - center) * scale).astype(np.float32) for curve in curves]


def _scanner_default_batch_colors(state):
    return _scanner_base_palette(state)


def _clamp_scanner_selected_cell(state):
    rows = max(1, int(state.scanner_rows))
    cols = max(1, int(state.scanner_cols))
    selected = np.asarray(state.get('scanner_selected_cell', [0, 0]), dtype=np.int32).reshape(-1)
    selected_r = int(selected[0]) if selected.size > 0 else 0
    selected_c = int(selected[1]) if selected.size > 1 else 0
    state.scanner_selected_cell = [
        int(np.clip(selected_r, 0, rows - 1)),
        int(np.clip(selected_c, 0, cols - 1)),
    ]
    return tuple(state.scanner_selected_cell)


def _set_scanner_estimated_batch_color(state, row: int, col: int, rgb) -> None:
    rows = max(1, int(state.scanner_rows))
    cols = max(1, int(state.scanner_cols))
    index = int(np.clip(row, 0, rows - 1)) * cols + int(np.clip(col, 0, cols - 1))
    overrides = state.get('scanner_estimated_color_overrides', [])
    if not isinstance(overrides, list):
        overrides = []
    while len(overrides) < rows * cols:
        overrides.append(None)
    rgba = _as_rgba(rgb, [0.5, 0.5, 0.5, 1.0])
    overrides[index] = [float(rgba[0]), float(rgba[1]), float(rgba[2]), 1.0]
    state.scanner_estimated_color_overrides = overrides[:rows * cols]
    _refresh_embedded_scanner_display_colors(state)


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


def _maybe_persist_scanner_state(state, force=False):
    storage = _scanner_storage(state)
    if storage is None:
        return
    now = time.monotonic()
    if not force and now - float(state.__dict__.get("_scanner_db_last_save", 0.0)) < 2.0:
        return
    try:
        snapshot = storage.scanner_state_snapshot(state)
        signature = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
        if force or signature != state.__dict__.get("_scanner_db_last_signature"):
            storage.save_scanner_state(state)
            object.__setattr__(state, "_scanner_db_last_signature", signature)
            object.__setattr__(state, "_scanner_db_last_save", now)
    except Exception:
        pass


def _refresh_embedded_scanner_display_colors(state):
    embedded = state.get('embedded_scanner')
    if embedded is None or getattr(embedded, "plan", None) is None:
        return
    embedded.plan.display_batch_colors = _scanner_batch_colors_for_simulator(state)
    if hasattr(embedded, "_render_frame"):
        embedded._render_frame()


def _puzzle_target_copies(state):
    copies_x = int(np.clip(int(state.get('puzzle_copies_x', 5)), 1, 9))
    copies_y = int(np.clip(int(state.get('puzzle_copies_y', 5)), 1, 9))
    if copies_x % 2 == 0:
        copies_x += 1
    if copies_y % 2 == 0:
        copies_y += 1
    return copies_x, copies_y


def _puzzle_apply_geometry_copies(state, preserve=True):
    copies_x, copies_y = _puzzle_target_copies(state)
    state.display_copies = np.array([(copies_x - 1) // 2, (copies_y - 1) // 2], dtype=np.int32)
    state.scanner_preview_grid_enabled = False
    state.rebuild_spline_mesh(preserve_model_placement=preserve)


def _set_puzzle_capture_rect(state, rect):
    x, y, w, h = [float(v) for v in rect]
    x = float(np.clip(x, 0.0, 0.95))
    y = float(np.clip(y, 0.0, 0.95))
    w = float(np.clip(w, 0.05, 1.0 - x))
    h = float(np.clip(h, 0.05, 1.0 - y))
    state.puzzle_capture_rect = [x, y, w, h]


def _puzzle_capture_geometry_image(state, renderer):
    """Reads back the actual rendered 3D viewport (same pixels shown behind the
    'Puzzle capture area' frame) and overlays real spline control points projected
    with the same camera/model matrices used to render that frame."""
    vp_w = int(getattr(renderer, "vp_w", 0))
    vp_h = int(getattr(renderer, "vp_h", 0))
    if vp_w < 2 or vp_h < 2 or getattr(renderer, "color_tex", None) is None:
        return None, [], []

    raw = renderer.color_tex.read()
    full_image = Image.frombytes("RGBA", (vp_w, vp_h), raw).convert("RGB")
    full_image = full_image.transpose(Image.FLIP_TOP_BOTTOM)

    model_mat = state.current_model_matrix()
    mvp = (state.camera.mvp(vp_w, vp_h) @ model_mat).astype(np.float32)

    projected_points = []
    for verts, row_idx in _puzzle_tiled_control_points(state):
        pix = _puzzle_project_points(verts, mvp, vp_w, vp_h)
        for px, py in pix:
            projected_points.append({"x": int(round(float(px))), "y": int(round(float(py))), "row": int(row_idx)})

    landmarks = []
    if projected_points:
        xs = np.array([p["x"] for p in projected_points], dtype=np.float32)
        ys = np.array([p["y"] for p in projected_points], dtype=np.float32)
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        for x in (x_min, x_max):
            for y in np.linspace(y_min, y_max, 9):
                landmarks.append({"x": int(round(x)), "y": int(round(float(y))), "kind": "x-edge"})
        for y in (y_min, y_max):
            for x in np.linspace(x_min, x_max, 9):
                landmarks.append({"x": int(round(float(x))), "y": int(round(y)), "kind": "y-edge"})

    rx, ry, rw, rh = _puzzle_capture_rect(state)
    crop_box = (
        int(round(rx * vp_w)),
        int(round(ry * vp_h)),
        int(round((rx + rw) * vp_w)),
        int(round((ry + rh) * vp_h)),
    )
    crop_box = (
        int(np.clip(crop_box[0], 0, vp_w - 1)),
        int(np.clip(crop_box[1], 0, vp_h - 1)),
        int(np.clip(crop_box[2], crop_box[0] + 1, vp_w)),
        int(np.clip(crop_box[3], crop_box[1] + 1, vp_h)),
    )
    cropped = full_image.crop(crop_box)
    x0, y0, x1, y1 = crop_box

    def adjust_points(items):
        adjusted = []
        for item in items:
            px = int(item.get("x", -1))
            py = int(item.get("y", -1))
            if x0 <= px < x1 and y0 <= py < y1:
                out = dict(item)
                out["x"] = px - x0
                out["y"] = py - y0
                adjusted.append(out)
        return adjusted

    return cropped, adjust_points(projected_points), adjust_points(landmarks)


def _puzzle_detect_colors(image, count=6):
    if image is None:
        return []
    count = max(1, int(count))
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8).reshape(-1, 3)
    bg = np.array([18, 23, 31], dtype=np.int16)
    mask = np.linalg.norm(arr.astype(np.int16) - bg[None, :], axis=1) > 24.0
    pixels = arr[mask]
    if pixels.size == 0:
        return []
    quantized = (pixels // 16) * 16
    unique, counts = np.unique(quantized, axis=0, return_counts=True)
    order = np.argsort(-counts)[:count]
    return [
        {"rgb": [int(v) for v in unique[i].tolist()], "count": int(counts[i])}
        for i in order
    ]


def _upload_puzzle_capture_texture(state, renderer):
    image = state.get('puzzle_capture_image', None)
    if image is None:
        return None
    texture = state.get('puzzle_capture_texture', None)
    key = (id(image), image.size)
    if texture is not None and state.get('puzzle_capture_texture_key') == key:
        return texture
    if texture is not None:
        try:
            texture.release()
        except Exception:
            pass
    texture = pil_to_texture(renderer.ctx, image)
    state.puzzle_capture_texture = texture
    state.puzzle_capture_texture_key = key
    return texture


def _upload_puzzle_glued_texture(state, renderer):
    image = state.get('puzzle_glued_image', None)
    if image is None:
        return None
    texture = state.get('puzzle_glued_texture', None)
    key = (id(image), image.size)
    if texture is not None and state.get('puzzle_glued_texture_key') == key:
        return texture
    if texture is not None:
        try:
            texture.release()
        except Exception:
            pass
    texture = pil_to_texture(renderer.ctx, image)
    state.puzzle_glued_texture = texture
    state.puzzle_glued_texture_key = key
    return texture


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


def _scanner_tiled_layout_cache_key(state):
    rows = max(1, int(state.scanner_rows))
    cols = max(1, int(state.scanner_cols))
    repeat_rows, repeat_cols = _scanner_pattern_repeats(state)
    pattern_rows, pattern_cols = _scanner_pattern_dimensions(state)
    lighting_settings = _scanner_lighting_settings(state)
    lighting_key = tuple((key, round(float(lighting_settings[key]), 4)) for key in sorted(lighting_settings))
    return (
        rows,
        cols,
        repeat_rows,
        repeat_cols,
        pattern_rows,
        pattern_cols,
        round(float(state.get('scanner_pattern_density', 0.62)), 4),
        int(state.get('scanner_random_seed', 1)),
        json.dumps(state.get('scanner_color_variants', []), sort_keys=True),
        lighting_key,
    )


def _scanner_generate_tiled_layout(state, renderer):
    """Builds Scan Mode's per-pattern fabric imagery automatically, using the same
    real-3D-render + exact-period capture/glue pipeline validated in Puzzle Mode
    (duplicate the real model -> auto-frame -> capture -> exact-period crop -> glue
    by Image Repeat X/Y), once per grid cell. Cached like the old preview was, so
    this only re-renders when a relevant setting actually changes."""
    try:
        # Normalizes scanner_color_variants (e.g. RGB -> RGBA) as a side effect the
        # first time it runs each session -- must happen before computing the cache
        # key, or the key would change out from under an already-stored cache entry.
        cell_sets = _scanner_shared_cell_color_sets(state)
        cache_key = _scanner_tiled_layout_cache_key(state)
        cached_key = state.__dict__.get("_scanner_tiled_layout_key")
        if cached_key == cache_key:
            cached = state.__dict__.get("_scanner_tiled_layout_result")
            if cached is not None:
                full_image, per_cell = cached
                return full_image.copy(), list(per_cell)

        rows = max(1, int(state.scanner_rows))
        cols = max(1, int(state.scanner_cols))
        repeat_rows, repeat_cols = _scanner_pattern_repeats(state)
    except Exception:
        return None, []

    per_cell_images = []
    for cell_index in range(rows * cols):
        try:
            bitmap = state._scanner_random_bitmap(cell_index)
            loop_heights = state._scanner_loop_heights_for_bitmap(bitmap)
            colors = cell_sets[cell_index % len(cell_sets)] if cell_sets else None
            tile_image = _scan_render_tiled_pattern_image(
                state, renderer, bitmap, loop_heights, colors, repeat_cols, repeat_rows,
            )
        except Exception:
            tile_image = None
        if tile_image is None:
            tile_image = Image.new("RGB", (64, 64), (18, 23, 31))
        per_cell_images.append(tile_image)

    cell_w = max((img.size[0] for img in per_cell_images), default=64)
    cell_h = max((img.size[1] for img in per_cell_images), default=64)
    full_image = Image.new("RGB", (max(1, cols * cell_w), max(1, rows * cell_h)), (18, 23, 31))
    draw = ImageDraw.Draw(full_image)
    border = (38, 124, 137)
    for row in range(rows):
        for col in range(cols):
            cell_index = row * cols + col
            cell_img = per_cell_images[cell_index]
            x0 = col * cell_w
            y0 = row * cell_h
            full_image.paste(cell_img, (x0, y0))
            draw.rectangle(
                [x0, y0, x0 + cell_img.size[0] - 1, y0 + cell_img.size[1] - 1],
                outline=border,
                width=1,
            )

    max_dim = 1400
    scale = min(1.0, max_dim / max(float(full_image.size[0]), 1.0), max_dim / max(float(full_image.size[1]), 1.0))
    if scale < 1.0:
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)
        full_image = full_image.resize(
            (max(1, int(full_image.size[0] * scale)), max(1, int(full_image.size[1] * scale))), resample,
        )

    object.__setattr__(state, "_scanner_tiled_layout_key", cache_key)
    object.__setattr__(state, "_scanner_tiled_layout_result", (full_image.copy(), list(per_cell_images)))
    return full_image, per_cell_images


def _scanner_full_layout_preview_image(state, renderer):
    full_image, _per_cell = _scanner_generate_tiled_layout(state, renderer)
    return full_image


def _scanner_per_cell_tiled_images(state, renderer):
    _full_image, per_cell = _scanner_generate_tiled_layout(state, renderer)
    return per_cell


# %% GUI DRAWING PANELS ────────────────────────────────────────────────────────


def _workflow_stage_title(state, step=None):
    index = int(np.clip(state.workflow_step if step is None else step, 0, len(state.workflow_stages) - 1))
    return str(state.workflow_stages[index][0])


def _set_workflow_step(state, step):
    step = int(np.clip(step, 0, len(state.workflow_stages) - 1))
    old_scanner = _workflow_stage_title(state) == "Scanner"
    new_scanner = _workflow_stage_title(state, step) == "Scanner"
    if step == int(state.workflow_step) and old_scanner == new_scanner:
        return

    state.workflow_step = step
    if new_scanner:
        state.scanner_preview_grid_enabled = True
        state.scanner_preview_rows = max(1, int(state.scanner_rows))
        state.scanner_preview_cols = max(1, int(state.scanner_cols))
        state.rebuild_spline_mesh(preserve_model_placement=False)
    elif old_scanner or bool(state.get('scanner_preview_grid_enabled', False)):
        state.scanner_preview_grid_enabled = False
        state.display_copies = np.array([0, 0], dtype=np.int32)
        state.rebuild_spline_mesh(preserve_model_placement=False)


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
            _set_workflow_step(state, i)
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
        _set_workflow_step(state, max(0, stage_idx - 1))
    if back_disabled:
        imgui.end_disabled()
    imgui.same_line()
    if next_disabled:
        imgui.begin_disabled()
    if imgui.button("Next##workflow", (nav_w, 0)):
        _set_workflow_step(state, min(len(state.workflow_stages) - 1, stage_idx + 1))
    if next_disabled:
        imgui.end_disabled()
    imgui.separator()


def _set_app_mode(state, mode):
    mode = mode if mode in ('edit', 'scan', 'puzzle', 'database') else 'edit'
    if str(state.get('app_mode', 'edit')) == mode:
        return
    state.app_mode = mode
    scanner_idx = next((i for i, item in enumerate(state.workflow_stages) if item[0] == 'Scanner'), 0)
    embedded = state.get('embedded_scanner')
    if embedded is not None:
        try:
            embedded.close()
        except Exception:
            pass
        state.embedded_scanner = None
    if mode == 'scan':
        state.workflow_step = scanner_idx
        state.scanner_preview_grid_enabled = True
        state.scanner_preview_rows = max(1, int(state.scanner_rows))
        state.scanner_preview_cols = max(1, int(state.scanner_cols))
        _clamp_scanner_selected_cell(state)
    elif mode == 'puzzle':
        state.workflow_step = scanner_idx
        state.scanner_preview_grid_enabled = False
        _puzzle_apply_geometry_copies(state, preserve=False)
    elif mode == 'database':
        state.scanner_preview_grid_enabled = False
        state.display_copies = np.array([0, 0], dtype=np.int32)
        state.database_view_mode = "results"
    else:
        state.workflow_step = 0
        state.scanner_preview_grid_enabled = False
        state.display_copies = np.array([0, 0], dtype=np.int32)
    state.rebuild_spline_mesh(preserve_model_placement=False)


def _apply_ui_theme(state):
    theme = str(state.get('ui_theme', 'dark'))
    if state.get('_applied_ui_theme') == theme:
        return
    if theme == 'light':
        imgui.style_colors_light()
    else:
        imgui.style_colors_dark()
    state._applied_ui_theme = theme


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
    for r in range(nr):
        for c in range(nc):
            active = float(state.bitmap[r, c]) > 0.5
            imgui.push_style_color(imgui.Col_.button, (0.18, 0.62, 0.28, 1.0) if active else (0.22, 0.22, 0.22, 1.0))
            imgui.push_style_color(imgui.Col_.button_hovered, (0.28, 0.72, 0.38, 1.0) if active else (0.35, 0.35, 0.35, 1.0))
            if imgui.button(f"##bm{id_suffix}_{r}_{c}", imgui.ImVec2(cell_w, cell_h)):
                if not changed_bitmap:
                    state.push_undo("Pattern")
                state.bitmap[r, c] = 0.0 if active else 1.0
                changed_bitmap = True
            imgui.pop_style_color(2)
            if c < nc - 1:
                imgui.same_line()
    imgui.pop_style_var()
    if changed_bitmap:
        state.on_bitmap_change()
    return changed_bitmap


def draw_sidebar(state, renderer, window=None):
    _apply_ui_theme(state)
    database_active_layout = str(state.get('app_mode', 'edit')) == 'database'
    if database_active_layout:
        imgui.set_next_window_pos((20, 20), cond=imgui.Cond_.always)
        imgui.set_next_window_size((1280, 840), cond=imgui.Cond_.always)
    else:
        imgui.set_next_window_pos((20, 20), cond=imgui.Cond_.first_use_ever)
        imgui.set_next_window_size((360, 820), cond=imgui.Cond_.first_use_ever)
    imgui.begin("Knitting Control")

    def rebuild_current_mesh(preserve=True):
        state.rebuild_spline_mesh(preserve_model_placement=preserve)

    def ensure_embedded_robot_viewer(auto_start=False):
        if not auto_start and bool(state.get('_embedded_scanner_viewer_failed', False)):
            return None
        _clamp_scanner_selected_cell(state)
        state.save_params(state.save_path, silent=True)
        embedded = state.get('embedded_scanner')
        if embedded is not None:
            if auto_start and not getattr(embedded, 'running', False):
                try:
                    embedded.close()
                except Exception:
                    pass
                state.embedded_scanner = None
            else:
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
            if not auto_start:
                state.scanner_status = embedded.status
            return embedded
        except Exception as exc:
            state.embedded_scanner = None
            state._embedded_scanner_viewer_failed = True
            label = "scanner" if auto_start else "Robot Viewer"
            state.scanner_status = f"Could not start embedded MuJoCo {label}: {exc}"
            return None

    def start_scanner_process():
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
            embedded.close()
            state.embedded_scanner = None
        elif embedded is not None and mode != "simulation":
            try:
                embedded.close()
            except Exception:
                pass
            state.embedded_scanner = None
        _clamp_scanner_selected_cell(state)
        state.save_params(state.save_path, silent=True)
        if mode == "simulation":
            embedded = ensure_embedded_robot_viewer(auto_start=True)
            if embedded is not None:
                state.scanner_status = "Embedded MuJoCo scanner running"
            return

        cell_color_sets = _scanner_shared_cell_color_sets(state)
        pattern_rows, pattern_cols = _scanner_pattern_dimensions(state)
        repeat_rows, repeat_cols = _scanner_pattern_repeats(state)
        spacing_x, spacing_y = _scanner_repeat_spacing(state)
        lighting = _scanner_lighting_settings(state)
        batch_texture_width, batch_texture_height = _scanner_batch_texture_size(state)
        try:
            import fabric_scanner as scanner

            args = SimpleNamespace(
                rows=int(state.scanner_rows),
                cols=int(state.scanner_cols),
                number_of_angles=int(state.scanner_angles),
                width=float(max(0.06, int(state.scanner_cols) * pattern_cols * 0.045)),
                length=float(max(0.06, int(state.scanner_rows) * pattern_rows * 0.040)),
                edge_margin=0.004,
                square_margin=0.006,
                surface_wave=0.003,
                view_radius=0.018,
                angle_lift=0.014,
                approach_lift=0.040,
                center=[-0.45, -0.08, 0.30],
                max_span=scanner.DEFAULT_MAX_SPAN.tolist(),
                palette=_scanner_base_palette(state),
                cell_color_sets=cell_color_sets,
                model_json=str(state.save_path),
                model_curves=None,
                cell_model_curves=_generate_scanner_random_patterns(state),
                random_patterns=True,
                pattern_rows=int(pattern_rows),
                pattern_cols=int(pattern_cols),
                pattern_repeat_rows=int(repeat_rows),
                pattern_repeat_cols=int(repeat_cols),
                pattern_repeat_spacing_x=float(spacing_x),
                pattern_repeat_spacing_y=float(spacing_y),
                batch_texture_width=int(batch_texture_width),
                batch_texture_height=int(batch_texture_height),
                pattern_density=float(state.get('scanner_pattern_density', 0.62)),
                random_seed=int(state.get('scanner_random_seed', 1)),
                scanner_lighting=lighting,
                display_batch_colors=_scanner_batch_colors_for_simulator(state),
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

    def ensure_embedded_single_capture():
        _clamp_scanner_selected_cell(state)
        state.save_params(state.save_path, silent=True)
        embedded = state.get('embedded_scanner')
        if embedded is None:
            try:
                tiled_full_image, tiled_per_cell = _scanner_generate_tiled_layout(state, renderer)
                embedded = EmbeddedMujocoScanner(
                    state,
                    renderer.ctx,
                    window,
                    preview_image=tiled_full_image,
                    per_cell_images=tiled_per_cell,
                    auto_start=False,
                )
                state.embedded_scanner = embedded
            except Exception as exc:
                state.embedded_scanner = None
                state.scanner_status = f"Could not start single capture preview: {exc}"
                return None
        embedded.running = False
        embedded.paused = True
        embedded.args.capture_mode = str(state.get('scanner_capture_mode', 'natural'))
        embedded.args.camera_zoom = float(state.get('scanner_camera_zoom', 1.0))
        return embedded

    current_mode = str(state.get('app_mode', 'edit'))
    edit_active = current_mode == 'edit'
    scan_active = current_mode == 'scan'
    puzzle_active = current_mode == 'puzzle'
    database_active = current_mode == 'database'
    button_w = max(72, (imgui.get_content_region_avail().x - imgui.get_style().item_spacing.x * 3.0) / 4.0)
    imgui.push_style_color(imgui.Col_.button, (0.22, 0.48, 0.78, 1.0) if edit_active else (0.20, 0.20, 0.20, 1.0))
    if imgui.button("Edit Mode##mode_edit", (button_w, 0)):
        _set_app_mode(state, 'edit')
    imgui.pop_style_color()
    imgui.same_line()
    imgui.push_style_color(imgui.Col_.button, (0.22, 0.48, 0.78, 1.0) if scan_active else (0.20, 0.20, 0.20, 1.0))
    if imgui.button("Scan Mode##mode_scan", (button_w, 0)):
        _set_app_mode(state, 'scan')
    imgui.pop_style_color()
    imgui.same_line()
    imgui.push_style_color(imgui.Col_.button, (0.22, 0.48, 0.78, 1.0) if puzzle_active else (0.20, 0.20, 0.20, 1.0))
    if imgui.button("Puzzle Mode##mode_puzzle", (button_w, 0)):
        _set_app_mode(state, 'puzzle')
    imgui.pop_style_color()
    imgui.same_line()
    imgui.push_style_color(imgui.Col_.button, (0.22, 0.48, 0.78, 1.0) if database_active else (0.20, 0.20, 0.20, 1.0))
    if imgui.button("Database##mode_database", (button_w, 0)):
        _set_app_mode(state, 'database')
    imgui.pop_style_color()
    imgui.separator()
    action_w = max(120, (imgui.get_content_region_avail().x - imgui.get_style().item_spacing.x) * 0.5)

    light_theme = str(state.get('ui_theme', 'dark')) == 'light'
    changed_theme, light_theme = imgui.checkbox("Light mode##ui_theme", light_theme)
    if changed_theme:
        state.ui_theme = 'light' if light_theme else 'dark'
        state._applied_ui_theme = ''
    imgui.same_line()
    imgui.text_disabled("Scanner and model controls update live")
    imgui.separator()

    undo_disabled = not state.undo_stack
    if undo_disabled:
        imgui.begin_disabled()
    if imgui.button("Undo##main", (action_w, 0)):
        state.undo_last()
    if undo_disabled:
        imgui.end_disabled()
    imgui.same_line()
    if imgui.button("Reset initial##reset_saved_initial_global", (action_w, 0)):
        state.reset_to_initial()
    imgui.separator()

    if database_active:
        _draw_database_summary_page(state, renderer)
        imgui.end()
        return

    if edit_active:
        if bool(state.get('scanner_preview_grid_enabled', False)):
            state.scanner_preview_grid_enabled = False
            state.display_copies = np.array([0, 0], dtype=np.int32)
            state.rebuild_spline_mesh(preserve_model_placement=False)

        if imgui.collapsing_header("Pattern", imgui.TreeNodeFlags_.default_open):
            _draw_bitmap_editor(state)

        if imgui.collapsing_header("Loop Heights", imgui.TreeNodeFlags_.default_open):
            state._sync_loop_heights()
            params = state.config['knit_parameters']['parameters']
            default_idx = state._lh_idx[0] if state._lh_idx else None
            lo, hi = (0.0, 6.0)
            if default_idx is not None:
                lo, hi = params[default_idx]['range']
            changed_any = False
            for r in range(int(state.bitmap_size[0])):
                imgui.text(f"Row {r + 1}")
                for c in range(int(state.bitmap_size[1])):
                    active = float(state.bitmap[r, c]) > 0.5
                    label = f"R{r + 1} C{c + 1}##loop_h_{r}_{c}"
                    if not active:
                        imgui.begin_disabled()
                    changed, val = imgui.slider_float(label, float(state.loop_heights[r, c]) if active else 0.0, float(lo), float(hi), "%.2f")
                    if not active:
                        imgui.end_disabled()
                    if imgui.is_item_activated():
                        state.push_undo("Loop height")
                    if active and changed:
                        state.set_loop_height_cell(r, c, val)
                        changed_any = True
                imgui.separator()
            if changed_any:
                state.rebuild_spline_from_params()

        if imgui.collapsing_header("Geometry", imgui.TreeNodeFlags_.default_open):
            quality_changed = False
            changed_loop_res, new_loop_res = imgui.slider_int("Path smoothness##mesh_loop_res", int(state.config['knit_parameters']['loop_res']), 8, 96)
            changed_segments, new_segments = imgui.slider_int("Fiber roundness##mesh_segments", int(state.config['knit_parameters']['segments']), 8, 64)
            if changed_loop_res:
                state.config['knit_parameters']['loop_res'] = int(new_loop_res); quality_changed = True
            if changed_segments:
                state.config['knit_parameters']['segments'] = int(new_segments); quality_changed = True
            useful_params = {'stitch_bulge', 'stitch_z', 'dy', 'radius', 'ellipse_ratio'}
            params_changed = False
            for i, pd in enumerate(state.config['knit_parameters']['parameters']):
                if pd['name'] not in useful_params:
                    continue
                lo, hi = pd['range']
                changed, new_val = imgui.slider_float(f"{pd['name']}##p{i}", float(state.params[i]), float(lo), float(hi), "%.3f")
                if imgui.is_item_activated():
                    state.push_undo(pd['name'])
                if changed:
                    state.params[i] = float(new_val)
                    params_changed = True
            if quality_changed:
                rebuild_current_mesh()
            if params_changed:
                state.nudge_spline_from_params()

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
                    state.fiber_geometry_count = int(value); fibers_changed = True
                for key, label, lo, hi in (
                    ('fiber_geometry_radius_scale', 'Fiber radius scale', 0.04, 0.45),
                    ('fiber_geometry_lift', 'Lift above surface', 0.0, 1.0),
                    ('fiber_geometry_surface_arc', 'Surface spread', 0.05, 1.0),
                    ('fiber_geometry_randomness', 'Randomness', 0.0, 1.0),
                    ('fiber_geometry_twist', 'Fiber twist', -3.0, 3.0),
                ):
                    changed, value = imgui.slider_float(f"{label}##{key}", float(state[key]), float(lo), float(hi), "%.2f")
                    if changed:
                        state[key] = float(value); fibers_changed = True
            else:
                imgui.text_disabled("Enable multi-fiber rows to separate each yarn into smaller fibers.")
            if fibers_changed:
                rebuild_current_mesh()

        if imgui.collapsing_header("Texture", imgui.TreeNodeFlags_.default_open):
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

        if imgui.collapsing_header("Display", imgui.TreeNodeFlags_.default_open):
            changed_x, new_x = imgui.slider_int("Copy via X##display_copies_x", int(state.display_copies[0]), 0, 20)
            changed_y, new_y = imgui.slider_int("Copy via Y##display_copies_y", int(state.display_copies[1]), 0, 20)
            if changed_x or changed_y:
                state.push_undo("Display copies")
                state.scanner_preview_grid_enabled = False
                state.display_copies = np.array([int(new_x), int(new_y)], dtype=np.int32)
                state.rebuild_spline_mesh(preserve_model_placement=True)
            if imgui.small_button("Single model##display_single"):
                state.push_undo("Display copies")
                state.display_copies = np.array([0, 0], dtype=np.int32)
                state.rebuild_spline_mesh(preserve_model_placement=True)
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
            if imgui.small_button("Center model##display_center"):
                state.push_undo("Center model")
                state.center_model_on_view()

        if imgui.collapsing_header("Lighting", imgui.TreeNodeFlags_.default_open):
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

        if imgui.collapsing_header("Spline", imgui.TreeNodeFlags_.default_open):
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
            imgui.text(f"Points: {len(state.flat_pts)}")
            if int(state.selected_idx) >= 0:
                imgui.text(f"Selected point: {int(state.selected_idx)}")
            imgui.text_disabled("Select a white point in the viewport, then drag or use keyboard controls.")

        if imgui.collapsing_header("Review", imgui.TreeNodeFlags_.default_open):
            imgui.text("Save / Load")
            if imgui.button("Save params...##save_params", (button_w, 0)):
                path = _pick_file('save', state.save_path)
                if path:
                    state.save_params(path)
            imgui.same_line()
            if imgui.button("Load params...##load_params", (button_w, 0)):
                path = _pick_file('load', state.load_path)
                if path:
                    state.load_params(path)
            changed_auto, new_auto = imgui.checkbox("Autosave", bool(state.autosave_enabled))
            if changed_auto:
                state.autosave_enabled = bool(new_auto)

    elif scan_active:
        imgui.text("Scanner")
        # -- Scanning section: scan layout, random patterns, and capture setup --
        rows = max(1, int(state.scanner_rows))
        cols = max(1, int(state.scanner_cols))
        preview_mismatch = (
            not bool(state.get('scanner_preview_grid_enabled', False))
            or int(state.get('scanner_preview_rows', 0)) != rows
            or int(state.get('scanner_preview_cols', 0)) != cols
        )
        if preview_mismatch:
            state.scanner_preview_grid_enabled = True
            state.scanner_preview_rows = rows
            state.scanner_preview_cols = cols
            state.rebuild_spline_mesh(preserve_model_placement=False)
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
                except Exception:
                    pass
                state.embedded_scanner = None
                state.scanner_status = "Scanner layout changed; press Start Scanner to generate new random patterns"
            state.rebuild_spline_mesh(preserve_model_placement=False)
        imgui.text("Layout pattern")
        grid_active = str(state.get('scanner_layout_pattern', 'grid')) == 'grid'
        if imgui.radio_button("Grid##scan_layout_grid", grid_active):
            state.scanner_layout_pattern = 'grid'
            state.rebuild_spline_mesh(preserve_model_placement=False)
        imgui.same_line()
        if imgui.radio_button("Staggered##scan_layout_staggered", not grid_active):
            state.scanner_layout_pattern = 'staggered'
            state.rebuild_spline_mesh(preserve_model_placement=False)
        imgui.separator()
        imgui.text("Random pattern generation")
        imgui.text_wrapped("Each mini-grid is one fabric sample. A random bitmap pattern is generated once, then repeated many times inside that sample. All samples use the same colors; only the bitmap changes.")
        pattern_rows, pattern_cols = _scanner_pattern_dimensions(state)
        repeat_rows, repeat_cols = _scanner_pattern_repeats(state)
        spacing_x, spacing_y = _scanner_repeat_spacing(state)
        batch_texture_width, batch_texture_height = _scanner_batch_texture_size(state)
        imgui.text(f"Template bitmap: {pattern_rows} rows x {pattern_cols} cols")
        imgui.text_disabled("Loaded from initial_params.json; Scan Mode randomizes only 0/1 bitmap values.")
        changed_rr, new_rr = imgui.slider_int("Image repeats Y##scanner_pattern_repeat_rows", repeat_rows, 1, 64)
        changed_rc, new_rc = imgui.slider_int("Image repeats X##scanner_pattern_repeat_cols", repeat_cols, 1, 64)
        changed_tex_res, new_tex_res = imgui.slider_int(
            "Batch texture resolution##scanner_batch_texture_width",
            batch_texture_width,
            160,
            2048,
        )
        new_tex_res = int(np.clip(int(round(float(new_tex_res) / 32.0) * 32), 160, 2048))
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
        imgui.text_disabled(
            f"Batch source image: {batch_texture_width} x {batch_texture_height}. "
            "Higher is sharper but heavier."
        )
        imgui.text_disabled("1.00 = edge-to-edge, below 1.00 overlaps copies, above 1.00 adds space.")
        changed_density, new_density = imgui.slider_float(
            "Active stitch probability##scanner_density",
            float(state.get('scanner_pattern_density', 0.62)),
            0.05,
            0.95,
            "%.2f",
        )
        changed_seed, new_seed = imgui.input_int("Random seed##scanner_seed", int(state.get('scanner_random_seed', 1)))
        random_changed = changed_rr or changed_rc or changed_tex_res or changed_sx or changed_sy or changed_density or changed_seed
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
        imgui.text_disabled("Patterns update when repeat, spacing, density, or seed changes.")
        if random_changed:
            embedded = state.get('embedded_scanner')
            if embedded is not None:
                try:
                    embedded.close()
                except Exception:
                    pass
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
        imgui.text_disabled("These colors are reused for every random fabric sample; only bitmap patterns change.")
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
                except Exception:
                    pass
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
            imgui.text_disabled("This edits only the estimated-color swatch, not the knitted model colors.")
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
        imgui.separator()
        # -- Lighting section: scanner lighting controls ----------------------
        imgui.text("Scanner lighting")
        imgui.text_disabled("Applied directly to the Scan Mode 3D view, robot camera, and saved images.")
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
            imgui.text_disabled("Flat colors: shadows, highlights, sheen, and directional shading are disabled.")
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
                except Exception:
                    pass
            state.embedded_scanner = None
            state.scanner_status = "Scanner lighting changed; press Start Scanner to rebuild"

        # -- Scanning section: robot camera modes and execution controls -------
        capture_mode = str(state.get('scanner_capture_mode', 'natural'))
        focused = capture_mode == "focused"
        single_workflow = str(state.get('scanner_camera_workflow', 'path')) == 'single'
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
                scan_capture_height = int(round(scan_capture_width * 0.75))
                if changed_scan_res:
                    state.scanner_capture_width = scan_capture_width
                    embedded = state.get('embedded_scanner')
                    if embedded is not None:
                        embedded.set_capture_resolution(scan_capture_width)
                imgui.text_disabled(f"Full scanner output: {scan_capture_width} x {scan_capture_height}")
        imgui.separator()
        single_workflow = str(state.get('scanner_camera_workflow', 'path')) == 'single'
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
            current_capture_width = int(np.clip(current_capture_width, 320, 2048))
            changed_capture_res, capture_width = imgui.slider_int(
                "Capture resolution##single_capture_resolution",
                current_capture_width,
                320,
                2048,
            )
            capture_width = int(np.clip(int(round(float(capture_width) / 64.0) * 64), 320, 2048))
            capture_height = int(round(capture_width * 0.75))
            if changed_capture_res:
                state.scanner_single_capture_width = capture_width
                embedded = state.get('embedded_scanner')
                if embedded is not None:
                    embedded.set_single_capture_resolution(capture_width)
                    embedded.preview_single_target(
                        int(state.get('scanner_single_row', 1)) - 1,
                        int(state.get('scanner_single_col', 1)) - 1,
                        int(state.get('scanner_single_angle', 1)) - 1,
                        float(state.get('scanner_camera_zoom', 1.0)),
                    )
            imgui.text_disabled(f"Single capture output: {capture_width} x {capture_height}")
            if imgui.button("Preview Selected Target##single_capture_preview", (-1, 0)):
                embedded = ensure_embedded_single_capture()
                if embedded is not None:
                    embedded.set_single_capture_resolution(capture_width)
                    embedded.preview_single_target(
                        int(state.get('scanner_single_row', 1)) - 1,
                        int(state.get('scanner_single_col', 1)) - 1,
                        int(state.get('scanner_single_angle', 1)) - 1,
                        float(state.get('scanner_camera_zoom', 1.0)),
                    )
                    state.scanner_status = embedded.status
            if imgui.button("Capture Image##single_capture_save", (-1, 0)):
                embedded = ensure_embedded_single_capture()
                if embedded is not None:
                    embedded.set_single_capture_resolution(capture_width)
                    path = embedded.capture_single_target(
                        int(state.get('scanner_single_row', 1)) - 1,
                        int(state.get('scanner_single_col', 1)) - 1,
                        int(state.get('scanner_single_angle', 1)) - 1,
                        float(state.get('scanner_camera_zoom', 1.0)),
                    )
                    state.scanner_status = f"Saved single capture: {Path(path).name}"
        if not single_workflow and str(state.scanner_execution_mode) == "robot":
            changed_ip, ip = imgui.input_text("Robot IP##scanner_robot_ip", str(state.scanner_robot_ip), 64)
            changed_port, port = imgui.input_int("Robot port##scanner_robot_port", int(state.scanner_robot_port))
            if changed_ip:
                state.scanner_robot_ip = ip
            if changed_port:
                state.scanner_robot_port = int(port)
        embedded = state.get('embedded_scanner')
        embedded_running = embedded is not None and getattr(embedded, 'running', False) and not getattr(embedded, 'paused', False)
        embedded_paused = embedded is not None and getattr(embedded, 'paused', False)
        running = embedded_running
        path_workflow = str(state.get('scanner_camera_workflow', 'path')) == 'path'
        if not path_workflow:
            imgui.begin_disabled()
        if imgui.button("Start Scanner Now##scanner_start", (-1, 0)):
            start_scanner_process()
        if not path_workflow:
            imgui.end_disabled()
            imgui.text_disabled("Switch to Full scanner path to run the full path.")
        if embedded_paused and path_workflow:
            if imgui.button("Continue Scanner##scanner_continue", (-1, 0)):
                embedded.resume()
                state.scanner_status = "Embedded MuJoCo scanner continued"
        if running:
            imgui.text_colored((0.15, 0.85, 0.35, 1.0), "Scanner is running")
        elif embedded_paused:
            imgui.text_colored((0.95, 0.75, 0.20, 1.0), "Scanner is paused")
        if not (running or embedded_paused):
            imgui.begin_disabled()
        if imgui.button("Stop Scanner##scanner_stop", (-1, 0)):
            stop_scanner_process()
        if not (running or embedded_paused):
            imgui.end_disabled()
        if imgui.button("Reset Scan Layout##scanner_reset", (-1, 0)):
            state.scanner_rows = 3
            state.scanner_cols = 4
            state.scanner_angles = 6
            state.scanner_preview_rows = 3
            state.scanner_preview_cols = 4
            state.scanner_preview_grid_enabled = True
            state.rebuild_spline_mesh(preserve_model_placement=False)
            state.scanner_status = "Scan layout reset"
        imgui.separator()
        # -- Analysis section: captured fabric color dashboard -----------------
        imgui.text("Per-sample RGB analysis")
        analysis_ready = embedded is not None and bool(getattr(embedded, "capture_records", []))
        if not analysis_ready:
            imgui.text_disabled("Capture scanner images first.")
        if not analysis_ready:
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

        # The estimated grid always reflects the current bitmap/color settings
        # (visible before any scan happens); once cells have been analyzed,
        # fold in the exact estimated value that was actually compared against
        # each analyzed cell, so both grids describe the same measurement.
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

        # Estimated grid is always visible, before and after scanning; the
        # average-RGB grid only exists once an analysis has actually run.
        draw_analysis_grid("Estimated Color Grid", estimate_by_pos, "rgb", "estimated_rgb_result")
        if has_analysis:
            imgui.same_line()
            draw_analysis_grid("Average RGB Color Grid", cells_by_pos, "overall_rgb", "actual_rgb_result")
        else:
            imgui.text_disabled("Average RGB grid appears here after you scan and click 'Analyze captured RGB'.")
        state.scanner_analysis_selected_cell = [selected_r, selected_c]

        if has_analysis:
            imgui.text(f"Images analyzed: {int(result.get('image_count', 0))}")
            if bool(result.get("background_ignored", False)):
                imgui.text_wrapped("Background ignored: RGB is calculated only from detected fabric pixels.")
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
        imgui.text_wrapped(str(state.scanner_status))

    elif puzzle_active:
        imgui.text("Puzzle Mode")
        imgui.text_wrapped(
            "Manual workflow for testing seamless repeat logic before automation: duplicate real geometry, capture it, detect colors, and inspect edge landmarks."
        )
        if imgui.collapsing_header("Stitch pattern (bitmap)##puzzle_pattern", imgui.TreeNodeFlags_.default_open):
            imgui.text_disabled(
                f"Editing the small {int(state.bitmap_size[0])} x {int(state.bitmap_size[1])} stitch bitmap "
                "used to build the model this test tiles and captures."
            )
            _draw_bitmap_editor(state, id_suffix="_puzzle")
        imgui.separator()
        copies_x, copies_y = _puzzle_target_copies(state)
        changed_px, new_px = imgui.slider_int("Real geometry copies X##puzzle_copies_x", copies_x, 1, 9)
        changed_py, new_py = imgui.slider_int("Real geometry copies Y##puzzle_copies_y", copies_y, 1, 9)
        if changed_px:
            if int(new_px) % 2 == 0:
                new_px += 1
            state.puzzle_copies_x = int(np.clip(new_px, 1, 9))
        if changed_py:
            if int(new_py) % 2 == 0:
                new_py += 1
            state.puzzle_copies_y = int(np.clip(new_py, 1, 9))
        if changed_px or changed_py:
            _puzzle_apply_geometry_copies(state, preserve=True)
            state.status_msg = f"Puzzle geometry set to {int(state.puzzle_copies_x)} x {int(state.puzzle_copies_y)}"
        if imgui.button("Apply 5 x 5 geometry##puzzle_apply_5x5", (-1, 0)):
            state.puzzle_copies_x = 5
            state.puzzle_copies_y = 5
            _puzzle_apply_geometry_copies(state, preserve=True)
            state.status_msg = "Puzzle geometry set to 5 x 5"
        if imgui.button("Apply current copy settings##puzzle_apply", (-1, 0)):
            _puzzle_apply_geometry_copies(state, preserve=True)
            state.status_msg = "Puzzle geometry copies applied"
        imgui.separator()
        imgui.text("Screenshot capture frame")
        rx, ry, rw, rh = _puzzle_capture_rect(state)
        changed_rx, rx = imgui.slider_float("Frame X##puzzle_capture_rect_x", rx, 0.0, 0.95, "%.2f")
        changed_ry, ry = imgui.slider_float("Frame Y##puzzle_capture_rect_y", ry, 0.0, 0.95, "%.2f")
        changed_rw, rw = imgui.slider_float("Frame width##puzzle_capture_rect_w", rw, 0.05, 1.0, "%.2f")
        changed_rh, rh = imgui.slider_float("Frame height##puzzle_capture_rect_h", rh, 0.05, 1.0, "%.2f")
        if changed_rx or changed_ry or changed_rw or changed_rh:
            _set_puzzle_capture_rect(state, [rx, ry, rw, rh])
        imgui.text_disabled("Yellow frame in the 3D viewport shows the region that will be captured.")
        imgui.separator()
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
            imgui.text(f"Captured image: {image.size[0]} x {image.size[1]}")
            colors = state.get('puzzle_detected_colors', [])
            if colors:
                imgui.text("Detected colors")
                for idx, item in enumerate(colors):
                    rgb = item.get("rgb", [0, 0, 0])
                    imgui.color_button(
                        f"##puzzle_color_{idx}",
                        (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, 1.0),
                        imgui.ColorEditFlags_.no_tooltip,
                        imgui.ImVec2(34, 22),
                    )
                    imgui.same_line()
                    imgui.text(f"{rgb[0]}, {rgb[1]}, {rgb[2]} | px {int(item.get('count', 0))}")
            imgui.text(f"Control points: {len(state.get('puzzle_projected_points', []))}")
            imgui.text(f"Edge landmarks: {len(state.get('puzzle_edge_landmarks', []))}")
            texture = _upload_puzzle_capture_texture(state, renderer)
            if texture is not None:
                avail_w = max(120, int(imgui.get_content_region_avail().x))
                preview_h = int(avail_w * image.size[1] / max(image.size[0], 1))
                _draw_puzzle_capture_widget(state, image, texture, avail_w, preview_h)
        else:
            imgui.text_disabled("Capture a geometry preview to inspect colors and edge landmarks.")

        imgui.separator()
        imgui.text("Seamless tile / glue preview")
        imgui.text_disabled(
            "Crops one exact repeat period from the capture area and stitches copies "
            "edge-to-edge to verify seamless tiling."
        )
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
            imgui.text(f"Tile size: {info.get('tile_w', 0)} x {info.get('tile_h', 0)} px")
            imgui.text(
                f"World period X/Y: {info.get('x_period_world', 0.0):.2f} / {info.get('y_period_world', 0.0):.2f}"
            )
            skew_color = (1.0, 0.55, 0.2, 1.0) if (abs(skew_x) > 1.5 or abs(skew_y) > 1.5) else (0.6, 0.9, 0.6, 1.0)
            imgui.text_colored(skew_color, f"Seam skew (should be ~0): x={skew_x:.2f}px, y={skew_y:.2f}px")
            if abs(skew_x) > 1.5 or abs(skew_y) > 1.5:
                imgui.text_wrapped(
                    "Skew is non-trivial: the camera view is not axis-aligned with the repeat "
                    "directions, so straight grid gluing will show a slight seam drift. "
                    "Rotate the camera to a top-down view for a perfect seam."
                )
            if imgui.button("Save seamless tile##puzzle_save_glue", (-1, 0)):
                output_dir = Path(state.project_root) / "scanner_data" / "puzzle_mode"
                output_dir.mkdir(parents=True, exist_ok=True)
                stamp = int(time.time())
                tile_image = state.get('puzzle_tile_image', None)
                if tile_image is not None:
                    tile_image.save(output_dir / f"puzzle_tile_{stamp}.png")
                glued_image.save(output_dir / f"puzzle_glued_{stamp}.png")
                state.status_msg = f"Saved puzzle_tile_{stamp}.png and puzzle_glued_{stamp}.png"
            glue_texture = _upload_puzzle_glued_texture(state, renderer)
            if glue_texture is not None:
                avail_w = max(120, int(imgui.get_content_region_avail().x))
                preview_h = int(min(avail_w * glued_image.size[1] / max(glued_image.size[0], 1), 480))
                glue_rect = draw_fitted_texture(glue_texture.glo, glued_image.size[0], glued_image.size[1], avail_w, preview_h, flip_y=True)
                if bool(state.get('puzzle_glue_highlight_edges', False)):
                    _draw_puzzle_glue_edge_overlay(glued_image, glue_rect, info)
        else:
            imgui.text_disabled("Build a seamless tile to preview the glued, repeated fabric.")

    if (
        str(state.get('app_mode', 'edit')) == 'scan'
        and str(state.get('scanner_execution_mode', 'simulation')) == 'simulation'
        and state.get('embedded_scanner') is None
    ):
        ensure_embedded_robot_viewer(auto_start=False)

    embedded = state.get('embedded_scanner')
    if str(state.get('app_mode', 'edit')) == 'scan' and embedded is not None:
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
            except Exception:
                pass
            state.embedded_scanner = None

    if str(state.get('app_mode', 'edit')) == 'scan':
        imgui.separator()
        imgui.text("Robot Camera View")
        imgui.text_disabled("Live image from the UR5 gripper camera")
        embedded = state.get('embedded_scanner')
        preview_w = max(120, int(imgui.get_content_region_avail().x))
        preview_h = int(preview_w * 0.75)
        if embedded is None:
            imgui.dummy((preview_w, preview_h))
            imgui.text_wrapped("Camera preview waiting for scan. Press Start Scanner Now.")
        elif embedded.camera_texture is None:
            imgui.dummy((preview_w, preview_h))
            imgui.text_wrapped("Camera preview waiting for the first scanner frame.")
        else:
            draw_fitted_texture(
                embedded.camera_texture.glo,
                embedded.camera_preview_width,
                embedded.camera_preview_height,
                preview_w,
                preview_h,
                flip_y=False,
            )

    if state.status_msg:
        imgui.separator()
        imgui.text_colored((0.4, 0.9, 0.4, 1.0), str(state.status_msg))
    state.maybe_autosave()
    imgui.end()

    embedded = state.get('embedded_scanner')
    if str(state.get('app_mode', 'edit')) == 'scan' and embedded is not None and embedded.texture is not None:
        imgui.set_next_window_size((760, 540), cond=imgui.Cond_.first_use_ever)
        imgui.begin("Robot Viewer")
        imgui.text(str(embedded.status))
        view_changed = False
        changed_zoom, zoom = imgui.slider_float("Zoom##scanner_view_zoom", float(embedded.view_zoom), 0.45, 2.50, "%.2fx")
        if changed_zoom:
            embedded.set_zoom(zoom)
            view_changed = True
        if imgui.small_button("Zoom in##scanner_view"):
            embedded.zoom_in()
            view_changed = True
        imgui.same_line()
        if imgui.small_button("Zoom out##scanner_view"):
            embedded.zoom_out()
            view_changed = True
        imgui.same_line()
        if imgui.small_button("Reset view##scanner_view"):
            embedded.reset_view()
            view_changed = True
        if imgui.small_button("Rotate left##scanner_view"):
            embedded.rotate_view(delta_azimuth=-12.0)
            view_changed = True
        imgui.same_line()
        if imgui.small_button("Rotate right##scanner_view"):
            embedded.rotate_view(delta_azimuth=12.0)
            view_changed = True
        imgui.same_line()
        if imgui.small_button("Tilt up##scanner_view"):
            embedded.rotate_view(delta_elevation=8.0)
            view_changed = True
        imgui.same_line()
        if imgui.small_button("Tilt down##scanner_view"):
            embedded.rotate_view(delta_elevation=-8.0)
            view_changed = True
        imgui.separator()
        imgui.text("UR5 Robot Simulator")
        avail = imgui.get_content_region_avail()
        main_w = max(1, int(avail.x))
        main_h = max(1, int(avail.y))
        rect = draw_fitted_texture(embedded.texture.glo, embedded.width, embedded.height, main_w, main_h, flip_y=False)
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
            imgui.text_disabled("Mouse over simulator: wheel zoom, left-drag rotate, right/middle-drag pan")
        if view_changed:
            embedded._render_frame()
        imgui.end()

    if str(state.get('app_mode', 'edit')) == 'puzzle' and state.get('puzzle_capture_image', None) is not None:
        image = state.get('puzzle_capture_image')
        texture = _upload_puzzle_capture_texture(state, renderer)
        if image is not None and texture is not None:
            imgui.set_next_window_size((760, 620), cond=imgui.Cond_.first_use_ever)
            imgui.begin("Puzzle Capture Inspector")
            imgui.text("Captured Image")
            imgui.text_disabled("White dots are real loop control points. Green dots are detected X/Y edge landmarks. Click the image to inspect pixel RGB.")
            avail = imgui.get_content_region_avail()
            inspector_w = max(240, int(avail.x))
            inspector_h = max(240, int(avail.y - 40))
            _draw_puzzle_capture_widget(state, image, texture, inspector_w, inspector_h)
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
    scanner_stage_active = str(state.get('app_mode', 'edit')) == 'scan'
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
        real_chunks = []
        virtual_indices = []
        for row_idx, row in enumerate(state.ctrl_rows):
            if not state.row_visible[row_idx]:
                continue
            start = state._row_starts[row_idx]
            end = start + len(row)
            real_chunks.append(np.arange(start, end, dtype=np.int32))
            virtual_indices.append(n_real_total + row_idx)
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
        bg_tex      = None if scanner_stage_active else (ref_tex if state.show_ref_bg else None),
        bg_alpha    = 0.0 if scanner_stage_active else state.ref_bg_alpha,
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
            homo = np.column_stack((world_pts, np.ones(len(world_pts), dtype=np.float32)))
            clip = homo @ view_proj.T
            valid = clip[:, 3] > 1e-6
            if not np.any(valid):
                return None
            ndc = np.zeros((len(world_pts), 3), dtype=np.float32)
            ndc[valid] = clip[valid, :3] / clip[valid, 3:4]
            screen = np.column_stack((
                (ndc[:, 0] * 0.5 + 0.5) * disp_w,
                (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * disp_h,
            ))
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
            homo = np.column_stack((world_pts, np.ones(len(world_pts), dtype=np.float32)))
            clip = homo @ view_proj.T
            valid = clip[:, 3] > 1e-6
            if not np.any(valid):
                continue
            ndc = np.zeros((len(world_pts), 3), dtype=np.float32)
            ndc[valid] = clip[valid, :3] / clip[valid, 3:4]
            screen = np.column_stack((
                (ndc[:, 0] * 0.5 + 0.5) * disp_w,
                (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * disp_h,
            ))
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
            homo = np.column_stack((world_pts, np.ones(len(world_pts), dtype=np.float32)))
            clip = homo @ view_proj.T
            valid = clip[:, 3] > 1e-6
            if not np.any(valid):
                continue
            ndc = np.zeros((len(world_pts), 3), dtype=np.float32)
            ndc[valid] = clip[valid, :3] / clip[valid, 3:4]
            screen = np.column_stack((
                (ndc[:, 0] * 0.5 + 0.5) * disp_w,
                (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * disp_h,
            ))
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
    if str(state.get('app_mode', 'edit')) == 'scan':
        return
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

