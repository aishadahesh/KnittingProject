"""imgui panels for the knitting app: menu bar, sidebar, and 3D viewport.

Module layout (dependencies point one way, no cycles):

    gui.py             this file -- imgui panels and per-mode UI
      |- scanner_core      scanner/puzzle logic, no imgui
      `- gui_ur5           UR5 Robot Mode UI           -> scanner_core

`app.py` imports draw_menu_bar / draw_sidebar / draw_viewport /
draw_orbit_viewport from here. Names moved into scanner_core kept their original
spelling and are re-imported below, so call sites in this file are unchanged.
"""
import os
import json
import numpy as np
import glfw
import time
import queue
import threading
from types import SimpleNamespace
from pathlib import Path

import tkinter as tk
from tkinter import filedialog as _filedialog
from imgui_bundle import imgui, imguizmo
from PIL import Image

from rendering import (draw_fitted_texture, project_to_screen, transform_points,
                       upload_rgb_texture, MeshRenderer)
from rgb_analysis import fabric_rgb_stats, summarize_capture_records as fabric_rgb_summary
from fabric_scanner import clamp_capture_size
import paths
# Moved to scanner_core (see its docstring). Re-imported under their original
# names so every call site in this file is unchanged.
from scanner_core import (
    _clamp_scanner_selected_cell, _ensure_scanner_shared_colors,
    _generate_scanner_random_patterns, _maybe_persist_scanner_state,
    _persist_scanner_state, _puzzle_apply_geometry_copies,
    _puzzle_build_seamless_tile, _puzzle_capture_geometry_image, _puzzle_capture_rect,
    _puzzle_detect_colors, _puzzle_target_copies,
    _refresh_embedded_scanner_display_colors, _scan_render_tiled_pattern_images,
    _scanner_base_palette, _scanner_batch_colors_for_simulator,
    _scanner_batch_texture_size, _scanner_capture_image_size,
    _scanner_display_batch_colors, _scanner_estimated_cell_colors,
    _scanner_generate_tiled_layout, _scanner_lighting_settings,
    _scanner_material_uniforms, _scanner_pattern_database_payload,
    _scanner_pattern_dimensions, _scanner_pattern_repeats, _scanner_repeat_spacing,
    _scanner_shared_cell_color_sets, _scanner_storage, _set_puzzle_capture_rect,
    _set_scanner_estimated_batch_color, _upload_puzzle_texture,
)

# UR5 Robot Mode's panel. Safe to import here: gui_ur5 imports gui only from
# inside the functions that need it, so there is no cycle at load time.
import gui_ur5

# Per-step solver displacements are tiny next to the model, so the force overlay
# exaggerates them to stay visible. Display only -- never fed back into state.
# ============================================================================
# Database Mode UI
# ============================================================================

def _database_resolve_image_path(state, image_path):
    path = Path(str(image_path or ""))
    if not path.is_absolute():
        path = Path(state.project_root) / path
    return path


def _database_rgb_label(rgb):
    if not isinstance(rgb, (list, tuple)) or len(rgb) < 3:
        return "pending"
    return f"{float(rgb[0]):.0f}, {float(rgb[1]):.0f}, {float(rgb[2]):.0f}"


def _database_prune_texture_cache(cache, current_frame, max_entries=200):
    """Releases old GL textures once the cache grows past budget.

    Only entries NOT touched during the current frame are eligible: a texture
    used this frame is already baked into ImGui's draw list, which isn't
    actually submitted to the GPU until impl.render() runs at the very end of
    the frame. Releasing (glDeleteTextures) an id that a still-pending draw
    command references crashes with GL_INVALID_OPERATION on glBindTexture --
    evicting purely by insertion order caused exactly that crash."""
    if len(cache) <= max_entries:
        return
    stale_keys = [key for key, value in cache.items() if value[3] != current_frame]
    stale_keys.sort(key=lambda key: cache[key][3])
    for key in stale_keys[: max(0, len(cache) - max_entries)]:
        old = cache.pop(key, None)
        if old is not None:
            try:
                old[0].release()
            except Exception:
                pass


def _database_image_texture(state, renderer, image_path, max_side=280):
    """Loads and caches a GL texture thumbnail for a saved scan image, so the
    database browser can show actual pictures instead of only file paths."""
    ctx = getattr(renderer, "ctx", None)
    if ctx is None:
        return None
    path = _database_resolve_image_path(state, image_path)
    if not path.exists():
        return None
    cache = state.__dict__.setdefault("_database_image_texture_cache", {})
    key = (str(path), int(max_side), int(path.stat().st_mtime_ns))
    current_frame = int(imgui.get_frame_count())
    cached = cache.get(key)
    if cached is not None:
        tex, w, h, _last_frame = cached
        cache[key] = (tex, w, h, current_frame)
        return (tex, w, h)
    try:
        image = Image.open(path).convert("RGB")
        image.thumbnail((int(max_side), int(max_side)), Image.Resampling.LANCZOS)
        rgb = np.asarray(image, dtype=np.uint8)
        h, w = rgb.shape[:2]
        tex = upload_rgb_texture(ctx, None, rgb)
        cache[key] = (tex, w, h, current_frame)
        _database_prune_texture_cache(cache, current_frame)
        return (tex, w, h)
    except Exception:
        return None


def _database_bitmap_label(bitmap):
    if bitmap is None:
        return "No bitmap"
    try:
        arr = np.asarray(bitmap)
        if arr.ndim == 0:
            return str(bitmap)
        if arr.ndim == 1:
            rows = ["".join("1" if float(v) > 0.5 else "0" for v in arr)]
        else:
            rows = [
                "".join("1" if float(v) > 0.5 else "0" for v in row)
                for row in arr
            ]
        label = "/".join(rows)
        return label if len(label) <= 64 else label[:61] + "..."
    except Exception:
        text = json.dumps(bitmap, sort_keys=True)
        return text if len(text) <= 64 else text[:61] + "..."


def _database_rgb_category(rgb):
    if not rgb:
        return "No RGB result"
    try:
        r, g, b = [float(v) for v in rgb[:3]]
    except Exception:
        return "No RGB result"
    strongest = max(r, g, b)
    weakest = min(r, g, b)
    if strongest < 45:
        return "Very dark"
    if strongest - weakest < 18:
        return "Neutral"
    if r >= g and r >= b:
        return "Red dominant" if g < r * 0.85 else "Warm"
    if g >= r and g >= b:
        return "Green dominant" if r < g * 0.85 else "Yellow/green"
    return "Blue dominant"


def _database_filter_value(item, key):
    if key == "robot_mode":
        return "Real UR5" if str(item.get("robot_mode")) == "real_ur5" else "Simulation"
    if key == "session_id":
        return str(item.get("session_id") or "No robot session")
    if key == "pattern_id":
        return str(item.get("pattern_name") or item.get("pattern_id") or "Unknown pattern")
    if key == "bitmap":
        return _database_bitmap_label(item.get("bitmap"))
    if key == "selected_colors":
        return str(item.get("selected_colors_label") or "No colors")
    if key == "lighting_mode":
        return str(item.get("lighting_mode") or "No lighting mode")
    if key == "camera_angle":
        angle = item.get("camera_angle")
        return f"{angle} deg" if angle not in (None, "") else "No angle"
    if key == "scan_station":
        station = item.get("scan_station")
        return f"Station {station}" if station not in (None, "") else "No station"
    if key == "batch_id":
        return str(item.get("batch_id") or "No batch")
    if key == "capture_mode":
        return str(item.get("capture_mode") or "No capture mode")
    if key == "scan_run":
        return str(item.get("scan_run") or "No scan run")
    if key == "average_rgb":
        return _database_rgb_category(item.get("average_rgb"))
    if key == "estimated_color":
        return _database_rgb_category(item.get("estimated_color"))
    return str(item.get(key) or "")


def _database_selected_filters(state):
    selected = state.get("database_filter_values", {})
    return selected if isinstance(selected, dict) else {}


def _database_matches_filters(item, selected_filters):
    for key, values in selected_filters.items():
        if not values:
            continue
        if _database_filter_value(item, key) not in set(values):
            return False
    return True


def _database_filter_specs():
    return [
        # First, because "was this scanned by the real arm or the simulator" is
        # the coarsest split in the dataset once both modes have contributed.
        ("Robot", "robot_mode"),
        ("Robot sessions", "session_id"),
        ("Pattern IDs", "pattern_id"),
        ("Bitmaps", "bitmap"),
        ("Selected colors", "selected_colors"),
        ("Lighting modes", "lighting_mode"),
        ("Camera angles", "camera_angle"),
        ("Scan stations", "scan_station"),
        ("Batches", "batch_id"),
        ("Capture modes", "capture_mode"),
        ("Scan runs", "scan_run"),
        ("Average RGB categories", "average_rgb"),
        ("Estimated color categories", "estimated_color"),
    ]


def _database_filter_label(key):
    labels = {key: label for label, key in _database_filter_specs()}
    return labels.get(key, key.replace("_", " ").title())


def _database_set_filter(selected_filters, key, value, *, append=False):
    selected = {name: list(values) for name, values in selected_filters.items()}
    if not append:
        selected[key] = [value]
        return selected
    values = set(selected.get(key, []))
    values.add(value)
    selected[key] = sorted(values)
    return selected


def _database_remove_filter(selected_filters, key, value):
    selected = {name: list(values) for name, values in selected_filters.items()}
    values = [item for item in selected.get(key, []) if item != value]
    if values:
        selected[key] = values
    elif key in selected:
        del selected[key]
    return selected


def _database_latest_value(captures, key):
    for item in reversed(captures):
        value = _database_filter_value(item, key)
        if value not in (None, "", "No scan run", "No pattern"):
            return value
    return None


def _database_capture_file_exists(state, item):
    return _database_resolve_image_path(state, item.get("image_path")).exists()


def _database_group_specs():
    return [
        ("No grouping", None),
        ("Pattern", "pattern_id"),
        ("Lighting", "lighting_mode"),
        ("Camera angle", "camera_angle"),
        ("Batch / mini-square", "batch_id"),
        ("Scan run", "scan_run"),
    ]


def _database_group_items(items, key):
    if key is None:
        return [(None, items)]
    groups = {}
    order = []
    for item in items:
        value = _database_filter_value(item, key)
        if value not in groups:
            groups[value] = []
            order.append(value)
        groups[value].append(item)
    return [(value, groups[value]) for value in order]


def _draw_database_image_card(state, renderer, item, *, card_w=192, card_h=246):
    """One card in the results grid: a real thumbnail (not a file path), the
    metadata that actually matters for a scan experiment, and RGB swatches.
    Clicking the thumbnail or the Open button opens the full-size preview.
    Missing-on-disk files are still shown (with a clear tag, no thumbnail, no
    Open action) rather than silently disappearing while "Show missing" is on."""
    cid = item.get("id")
    missing = not _database_capture_file_exists(state, item)
    selected = int(item.get("id", -1)) == int(state.get("database_selected_capture_id", -1))
    if missing:
        bg = (0.24, 0.14, 0.13, 1.0)
    elif selected:
        bg = (0.30, 0.24, 0.10, 1.0)
    else:
        bg = (0.15, 0.15, 0.17, 1.0)
    imgui.push_style_color(imgui.Col_.child_bg, bg)
    imgui.begin_child(f"##db_card_{cid}", imgui.ImVec2(card_w, card_h), imgui.ChildFlags_.borders)

    opened = False
    thumb_w = card_w - 16
    thumb_h = int(thumb_w * 0.72)
    if missing:
        imgui.dummy(imgui.ImVec2(thumb_w, thumb_h))
        imgui.text_colored((0.90, 0.45, 0.40, 1.0), "MISSING FILE")
    else:
        texture = _database_image_texture(state, renderer, item.get("image_path"), max_side=max(thumb_w, thumb_h))
        if texture is not None:
            tex, w, h = texture
            scale = min(thumb_w / max(w, 1), thumb_h / max(h, 1))
            draw_w, draw_h = max(1.0, w * scale), max(1.0, h * scale)
            imgui.image(
                imgui.ImTextureRef(tex.glo), imgui.ImVec2(draw_w, draw_h),
                uv0=imgui.ImVec2(0, 1), uv1=imgui.ImVec2(1, 0),
            )
            if imgui.is_item_clicked():
                opened = True
            if imgui.is_item_hovered():
                imgui.set_tooltip("Click to open full preview")
        else:
            imgui.dummy(imgui.ImVec2(thumb_w, thumb_h))
            imgui.text_disabled("No preview")

    imgui.text_wrapped(str(item.get("pattern_name", "Unknown pattern")))
    imgui.text_disabled(f"{item.get('batch_id', '')}  |  angle {item.get('camera_angle', '')}")
    imgui.text_disabled(str(item.get("lighting_mode", "")))
    imgui.text_disabled(f"Run: {item.get('scan_run', '')}")
    confidence = item.get("patch_confidence")
    if confidence is not None:
        conf = float(confidence)
        conf_color = (0.35, 0.85, 0.45, 1.0) if conf >= 0.6 else ((0.90, 0.70, 0.20, 1.0) if conf >= 0.3 else (0.90, 0.35, 0.30, 1.0))
        imgui.text_colored(conf_color, f"Patch confidence: {conf:.2f}")

    if missing:
        imgui.text_disabled("Removed from disk since last scan.")
    else:
        avg = item.get("average_rgb")
        est = item.get("estimated_color")
        if est:
            rgba = [float(v) / 255.0 for v in est[:3]]
            imgui.color_button(f"##card_est_{cid}", (rgba[0], rgba[1], rgba[2], 1.0), imgui.ColorEditFlags_.no_tooltip, imgui.ImVec2(18, 16))
            imgui.same_line()
        if avg:
            rgba = [float(v) / 255.0 for v in avg[:3]]
            imgui.color_button(f"##card_avg_{cid}", (rgba[0], rgba[1], rgba[2], 1.0), imgui.ColorEditFlags_.no_tooltip, imgui.ImVec2(18, 16))
        if imgui.small_button(f"Open##card_open_{cid}"):
            opened = True
    imgui.end_child()
    imgui.pop_style_color()
    return opened


def _draw_database_card_grid(state, renderer, items, *, card_w=192, card_h=246):
    """Lays cards out in a wrapping grid instead of a single-column list."""
    avail_w = max(float(card_w), imgui.get_content_region_avail().x)
    spacing = imgui.get_style().item_spacing.x
    cols = max(1, int((avail_w + spacing) // (card_w + spacing)))
    opened_item = None
    for idx, item in enumerate(items):
        if idx % cols != 0:
            imgui.same_line()
        if _draw_database_image_card(state, renderer, item, card_w=card_w, card_h=card_h):
            opened_item = item
    return opened_item


def _database_reset_image_view(state):
    state.database_image_zoom = 1.0
    state.database_image_pan = [0.0, 0.0]
    state.database_image_rotation = 0.0


def _draw_database_zoomable_image(state, tex_id, tex_w, tex_h, avail_w, avail_h):
    """Interactive viewer for the preview image: mouse-wheel zoom, drag to pan,
    and free rotation. Drawn as a rotatable quad (add_image_quad) rather than
    the axis-aligned draw_fitted_texture, since ImGui's plain imgui.image()
    can't be rotated."""
    avail_w = max(1.0, float(avail_w))
    avail_h = max(1.0, float(avail_h))
    zoom = max(0.1, float(state.get("database_image_zoom", 1.0)))
    pan = state.get("database_image_pan", [0.0, 0.0])
    rotation_deg = float(state.get("database_image_rotation", 0.0))

    origin = imgui.get_cursor_screen_pos()
    imgui.invisible_button("##db_image_view", imgui.ImVec2(avail_w, avail_h))
    hovered = imgui.is_item_hovered()
    active = imgui.is_item_active()

    io = imgui.get_io()
    if hovered and abs(io.mouse_wheel) > 1e-6:
        zoom = float(np.clip(zoom * (1.0 + io.mouse_wheel * 0.12), 0.1, 12.0))
        state.database_image_zoom = zoom
    if active and imgui.is_mouse_dragging(0):
        delta = imgui.get_mouse_drag_delta(0)
        pan = [float(pan[0]) + delta.x, float(pan[1]) + delta.y]
        state.database_image_pan = pan
        imgui.reset_mouse_drag_delta(0)

    base_scale = min(avail_w / max(tex_w, 1), avail_h / max(tex_h, 1))
    draw_w = tex_w * base_scale * zoom
    draw_h = tex_h * base_scale * zoom
    center_x = origin.x + avail_w * 0.5 + float(pan[0])
    center_y = origin.y + avail_h * 0.5 + float(pan[1])

    angle = np.radians(rotation_deg)
    cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
    half_w, half_h = draw_w * 0.5, draw_h * 0.5

    def rotated_corner(dx, dy):
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        return imgui.ImVec2(center_x + rx, center_y + ry)

    p1 = rotated_corner(-half_w, -half_h)
    p2 = rotated_corner(half_w, -half_h)
    p3 = rotated_corner(half_w, half_h)
    p4 = rotated_corner(-half_w, half_h)

    # Rotating and/or zooming can easily push the quad's corners outside the
    # reserved preview area (e.g. a rotated rectangle's bounding box is wider
    # than the rectangle itself). Without clipping that overflow paints over
    # whatever UI sits below/around the preview -- so scissor to exactly the
    # reserved rect regardless of angle or zoom.
    clip_min = imgui.ImVec2(origin.x, origin.y)
    clip_max = imgui.ImVec2(origin.x + avail_w, origin.y + avail_h)
    draw_list = imgui.get_window_draw_list()
    draw_list.add_rect_filled(clip_min, clip_max, imgui.get_color_u32((0.05, 0.06, 0.08, 1.0)))
    draw_list.push_clip_rect(clip_min, clip_max, True)
    # Matches the flip_y=True convention used elsewhere for textures uploaded
    # with a pre-flip (see _database_image_texture / pil_to_texture).
    draw_list.add_image_quad(
        imgui.ImTextureRef(tex_id), p1, p2, p3, p4,
        imgui.ImVec2(0, 1), imgui.ImVec2(1, 1), imgui.ImVec2(1, 0), imgui.ImVec2(0, 0),
    )
    draw_list.pop_clip_rect()
    draw_list.add_rect(clip_min, clip_max, imgui.get_color_u32((0.35, 0.35, 0.40, 1.0)))
    if hovered:
        imgui.set_tooltip("Scroll to zoom, drag to pan")


def _draw_database_image_preview(state, renderer, item):
    """Full-size preview: the image comes first, then metadata that explains
    what was scanned and how, with the file path demoted into a Details
    section since it's the least useful field for understanding the result."""
    if state.get("database_image_view_id") != item.get("id"):
        _database_reset_image_view(state)
        state.database_image_view_id = item.get("id")

    if imgui.button("Back to results##db_back_to_results", imgui.ImVec2(200, 0)):
        state.database_view_mode = "results"
    imgui.same_line()
    imgui.text(f"{item.get('pattern_name')}  |  {item.get('batch_id')}  |  angle {item.get('camera_angle')}")
    imgui.separator()

    missing = not _database_capture_file_exists(state, item)
    if missing:
        imgui.text_colored((0.90, 0.45, 0.40, 1.0), "This image file is missing from disk (it was likely deleted after the scan).")
    else:
        view_options = [("Full image", "image_path")]
        if item.get("patch_image_path"):
            view_options.append(("Detected patch", "patch_image_path"))
        if item.get("debug_image_path"):
            view_options.append(("Debug (patch boundary)", "debug_image_path"))
        view_labels = [label for label, _key in view_options]
        current_view = str(state.get("database_image_source", "image_path"))
        current_view_idx = next((i for i, (_l, k) in enumerate(view_options) if k == current_view), 0)
        if len(view_options) > 1:
            imgui.set_next_item_width(220)
            changed_view, new_view_idx = imgui.combo("##db_image_source", current_view_idx, view_labels)
            if changed_view:
                state.database_image_source = view_options[new_view_idx][1]
                current_view_idx = new_view_idx
        image_path_key = view_options[current_view_idx][1]
        texture = _database_image_texture(state, renderer, item.get(image_path_key) or item.get("image_path"), max_side=1400)
        if texture is not None:
            tex, w, h = texture
            if imgui.small_button("Zoom out##db_zoom_out"):
                state.database_image_zoom = max(0.1, float(state.get("database_image_zoom", 1.0)) / 1.25)
            imgui.same_line()
            if imgui.small_button("Zoom in##db_zoom_in"):
                state.database_image_zoom = min(12.0, float(state.get("database_image_zoom", 1.0)) * 1.25)
            imgui.same_line()
            imgui.text_disabled(f"{float(state.get('database_image_zoom', 1.0)) * 100.0:.0f}%")
            imgui.same_line()
            if imgui.small_button("Rotate left##db_rotate_left"):
                state.database_image_rotation = float(state.get("database_image_rotation", 0.0)) - 90.0
            imgui.same_line()
            if imgui.small_button("Rotate right##db_rotate_right"):
                state.database_image_rotation = float(state.get("database_image_rotation", 0.0)) + 90.0
            imgui.same_line()
            imgui.set_next_item_width(160)
            changed_rot, new_rot = imgui.slider_float("##db_rotate_free", float(state.get("database_image_rotation", 0.0)) % 360.0, 0.0, 360.0, "%.0f deg")
            if changed_rot:
                state.database_image_rotation = new_rot
            imgui.same_line()
            if imgui.small_button("Reset view##db_reset_view"):
                _database_reset_image_view(state)

            avail = imgui.get_content_region_avail()
            _draw_database_zoomable_image(state, tex.glo, w, h, max(360, int(avail.x * 0.62)), max(360, int(avail.y * 0.62)))
        else:
            imgui.text_disabled("Selected image file could not be loaded.")

    imgui.separator()
    imgui.text("Scan metadata")
    imgui.text(f"Pattern: {item.get('pattern_name')}")
    imgui.text(f"Batch / mini-square: {item.get('batch_id')}  (station {item.get('scan_station')})")
    imgui.text(f"Lighting mode: {item.get('lighting_mode')}")
    imgui.text(f"Camera angle: {item.get('camera_angle')}")
    imgui.text(f"Scan run: {item.get('scan_run')}")
    imgui.text(f"Capture mode: {item.get('capture_mode')}")
    imgui.text(f"Selected colors: {item.get('selected_colors_label')}")
    imgui.text(f"Bitmap: {_database_bitmap_label(item.get('bitmap'))}")
    if str(item.get("robot_mode")) == "real_ur5":
        imgui.separator()
        imgui.text_colored((0.92, 0.62, 0.28, 1.0), "Captured by the real UR5")
        imgui.text(f"Robot: {item.get('robot_ip') or 'unknown address'}")
        imgui.text(f"Session: {item.get('session_id')}")
        position = item.get("target_position") or []
        if len(position) >= 3:
            imgui.text(f"Target position: x {position[0]:+.3f}  y {position[1]:+.3f}  z {position[2]:+.3f} m")
        if item.get("camera_backend"):
            imgui.text(f"Camera: {item.get('camera_backend')}")
        if not item.get("camera_is_real", False):
            imgui.text_colored((0.95, 0.75, 0.20, 1.0), "Synthetic camera frames - not real imagery")
        if item.get("splat_output_path"):
            imgui.text(f"Reconstruction: {Path(str(item['splat_output_path'])).name}")

    imgui.separator()
    imgui.text("Color results")
    avg = item.get("average_rgb")
    est = item.get("estimated_color")
    if est:
        rgba = [float(v) / 255.0 for v in est[:3]]
        imgui.color_button("##db_image_est", (rgba[0], rgba[1], rgba[2], 1.0), imgui.ColorEditFlags_.no_tooltip, imgui.ImVec2(38, 24))
        imgui.same_line()
        imgui.text(f"Estimated RGB: {_database_rgb_label(est)}")
    else:
        imgui.text_disabled("Estimated RGB: pending")
    if avg:
        rgba = [float(v) / 255.0 for v in avg[:3]]
        imgui.color_button("##db_image_avg", (rgba[0], rgba[1], rgba[2], 1.0), imgui.ColorEditFlags_.no_tooltip, imgui.ImVec2(38, 24))
        imgui.same_line()
        imgui.text(f"Average RGB (analyzed): {_database_rgb_label(avg)}")
    else:
        imgui.text_disabled("Average RGB: not analyzed yet")
    if est and avg:
        delta = float(np.linalg.norm(np.asarray(avg[:3], dtype=np.float32) - np.asarray(est[:3], dtype=np.float32)))
        imgui.text(f"Estimated vs actual delta: {delta:.1f}")

    if item.get("patch_confidence") is not None or item.get("patch_bbox"):
        imgui.separator()
        imgui.text("Patch detection (debugging / evaluation)")
        confidence = item.get("patch_confidence")
        if confidence is not None:
            conf = float(confidence)
            conf_color = (0.35, 0.85, 0.45, 1.0) if conf >= 0.6 else ((0.90, 0.70, 0.20, 1.0) if conf >= 0.3 else (0.90, 0.35, 0.30, 1.0))
            imgui.text_colored(conf_color, f"Confidence: {conf:.2f}")
        bbox = item.get("patch_bbox")
        if bbox:
            imgui.text(f"Detected bbox (full image px): {bbox}")
        zoom_level = item.get("camera_zoom_level")
        angle_deg = item.get("camera_angle_deg")
        if zoom_level is not None or angle_deg is not None:
            imgui.text(f"Camera zoom: {float(zoom_level or 1.0):.2f}x   Angle used: {float(angle_deg or 0.0):.1f} deg")
        pose = item.get("camera_pose")
        if pose:
            imgui.text_disabled(f"Standoff: {float(pose.get('standoff', 0.0)):.3f} m   FOV: {float(pose.get('fov_y_deg', 0.0)):.1f} deg")

    imgui.separator()
    if imgui.tree_node("Details: file path, capture settings, raw record##db_image_details"):
        imgui.text_disabled("Image path")
        imgui.text_wrapped(str(_database_resolve_image_path(state, item.get("image_path"))))
        imgui.text_disabled("Raw metadata")
        imgui.text_wrapped(json.dumps(item, indent=2, sort_keys=True))
        imgui.tree_pop()
    imgui.separator()
    imgui.text_wrapped(str(state.scanner_status))


def _draw_database_summary_page(state, renderer):
    storage = _scanner_storage(state)
    if storage is None:
        imgui.text_disabled("Scanner database is not available.")
        return

    if imgui.button("Refresh##database_refresh", imgui.ImVec2(120, 0)):
        # Safe to release here: this runs before any thumbnail is drawn this
        # frame, so nothing in the current (or any pending) draw list can
        # reference these texture ids yet.
        stale_cache = state.__dict__.pop("_database_image_texture_cache", {})
        for cached in stale_cache.values():
            try:
                cached[0].release()
            except Exception:
                pass
        # Refresh is also where the exported JSON map is brought up to date --
        # it is the control that promises exactly that.
        try:
            storage.write_json_index()
        except Exception:
            pass
        state.scanner_status = "Dataset refreshed: re-checked saved files against the database"
    imgui.same_line()
    imgui.text_disabled("Re-checks saved image files on disk. JSON map: " + str(storage.json_index_path))

    try:
        # This panel is redrawn every frame. Rebuilding the summary and
        # re-exporting a multi-MB JSON file each time held the whole window at
        # well under one frame per second whenever the tab was open.
        summary = storage.cached_database_summary()
        storage.flush_json_index()
    except Exception as exc:
        imgui.text_wrapped(f"Could not load scanner dataset: {exc}")
        return

    all_captures = list(summary.get("captures", []))
    patterns = list(summary.get("patterns", []))
    if not all_captures and not patterns:
        imgui.separator()
        imgui.text_disabled("No saved scanner patterns or captured images yet.")
        imgui.text_disabled("Run the scanner (Scan Mode) or capture one image to populate the dataset.")
        return

    # Sync with disk: a capture row only counts as an available result if its
    # image file still exists. Deleted files are hidden by default rather than
    # shown as broken paths, and can optionally be revealed, clearly tagged.
    available_captures = []
    missing_captures = []
    for item in all_captures:
        (available_captures if _database_capture_file_exists(state, item) else missing_captures).append(item)

    show_missing = bool(state.get("database_show_missing", False))
    captures = available_captures + (missing_captures if show_missing else [])

    imgui.text("Scanner Dataset")
    imgui.text_disabled("What was scanned, under which settings, and what the analysis found -- not just file paths.")

    imgui.separator()
    imgui.text(f"Patterns: {int(summary.get('pattern_count', 0))}")
    imgui.same_line()
    imgui.text(f"Images available: {len(available_captures)}")
    imgui.same_line()
    if missing_captures:
        imgui.text_colored((0.90, 0.55, 0.35, 1.0), f"Missing: {len(missing_captures)}")
    else:
        imgui.text_disabled("Missing: 0")
    imgui.same_line()
    imgui.text(f"Analyses: {int(summary.get('analysis_count', 0))}")
    if missing_captures:
        imgui.same_line()
        changed_missing, new_show_missing = imgui.checkbox("Show missing files##db_show_missing", show_missing)
        if changed_missing:
            state.database_show_missing = new_show_missing
            show_missing = new_show_missing
            captures = available_captures + (missing_captures if show_missing else [])

    selected_filters = {key: list(values) for key, values in _database_selected_filters(state).items()}

    imgui.separator()
    imgui.text("Quick Browse")
    if imgui.small_button("All results##db_quick_all"):
        selected_filters = {}
    latest_run = _database_latest_value(captures, "scan_run")
    if latest_run:
        imgui.same_line()
        if imgui.small_button(f"Latest run: {latest_run}##db_quick_latest_run"):
            selected_filters = _database_set_filter(selected_filters, "scan_run", latest_run)
    latest_pattern = _database_latest_value(captures, "pattern_id")
    if latest_pattern:
        imgui.same_line()
        if imgui.small_button("Latest pattern##db_quick_latest_pattern"):
            selected_filters = _database_set_filter(selected_filters, "pattern_id", latest_pattern)
    if captures:
        imgui.same_line()
        if imgui.small_button("Images with RGB##db_quick_rgb"):
            rgb_options = sorted({
                _database_filter_value(item, "average_rgb")
                for item in captures
                if _database_filter_value(item, "average_rgb") != "No RGB result"
            })
            if rgb_options:
                selected_filters["average_rgb"] = rgb_options

    active_count = sum(len(values) for values in selected_filters.values())
    if active_count:
        imgui.text("Active filters")
        for key, values in list(selected_filters.items()):
            for value in list(values):
                if imgui.small_button(f"x {_database_filter_label(key)}: {value}##db_chip_{key}_{value}"):
                    selected_filters = _database_remove_filter(selected_filters, key, value)
                imgui.same_line()
        imgui.new_line()
    else:
        imgui.text_disabled("No filters selected.")
    state.database_filter_values = selected_filters

    all_selected_id = int(state.get("database_selected_capture_id", 0))
    all_selected_item = next((item for item in all_captures if int(item.get("id", -1)) == all_selected_id), None)
    if str(state.get("database_view_mode", "results")) == "image" and all_selected_item is not None:
        _draw_database_image_preview(state, renderer, all_selected_item)
        return

    imgui.separator()
    imgui.text("Group by")
    group_specs = _database_group_specs()
    group_labels = [label for label, _key in group_specs]
    current_group_key = state.get("database_group_by", None)
    current_group_index = next((i for i, (_label, key) in enumerate(group_specs) if key == current_group_key), 0)
    imgui.set_next_item_width(220)
    changed_group, new_group_index = imgui.combo("##db_group_by", current_group_index, group_labels)
    if changed_group:
        current_group_key = group_specs[new_group_index][1]
        state.database_group_by = current_group_key

    if imgui.collapsing_header("Detailed checkbox filters"):
        if imgui.small_button("Clear filters##database_clear_filters"):
            selected_filters = {}
        imgui.same_line()
        imgui.text_disabled("Choose one or more options. Results update immediately.")
        for label, key in _database_filter_specs():
            options = sorted({_database_filter_value(item, key) for item in captures}, key=lambda x: str(x).lower())
            options = [value for value in options if value not in (None, "")]
            active = set(selected_filters.get(key, []))
            title = f"{label} ({len(active)} selected)##db_filter_{key}"
            if imgui.tree_node(title):
                if len(options) > 80:
                    imgui.text_disabled(f"Showing first 80 of {len(options)} options. Use other filters to narrow the list.")
                for idx, option in enumerate(options[:80]):
                    checked = option in active
                    changed, new_checked = imgui.checkbox(f"{option}##db_filter_{key}_{idx}", checked)
                    if changed:
                        if new_checked:
                            active.add(option)
                        else:
                            active.discard(option)
                        if active:
                            selected_filters[key] = sorted(active)
                        elif key in selected_filters:
                            del selected_filters[key]
                imgui.tree_pop()
        state.database_filter_values = selected_filters

    selected_filters = _database_selected_filters(state)
    filtered = [item for item in captures if _database_matches_filters(item, selected_filters)]

    imgui.separator()
    imgui.text(f"Results: {len(filtered)} / {len(captures)} shown")

    if imgui.collapsing_header("Image results", imgui.TreeNodeFlags_.default_open):
        imgui.text_disabled("Click a thumbnail (or its Open button) for the full-size preview and metadata.")
        max_shown = 150
        shown = filtered[:max_shown]
        for group_value, group_items in _database_group_items(shown, current_group_key):
            if current_group_key is not None:
                header_label = group_value if group_value else "(none)"
                if not imgui.collapsing_header(f"{header_label} ({len(group_items)})##db_group_{header_label}", imgui.TreeNodeFlags_.default_open):
                    continue
            opened = _draw_database_card_grid(state, renderer, group_items)
            if opened is not None:
                state.database_selected_capture_id = int(opened.get("id", 0))
                state.database_view_mode = "image"
        if len(filtered) > len(shown):
            imgui.text_disabled(f"Showing first {len(shown)} of {len(filtered)} images. Add filters to narrow the rest.")

    if imgui.collapsing_header("Analysis results"):
        imgui.text_disabled("Saved RGB analysis summaries. Use filters above to compare related experiments.")
        analyses = list(summary.get("analyses", []))
        visible_analysis_count = 0
        for analysis in reversed(analyses[-20:]):
            result = analysis.get("result", {})
            cells = list(result.get("cells", []))
            matched_cells = []
            for cell in cells:
                pseudo = {
                    "pattern_name": next((p.get("name") for p in patterns if p.get("signature") == analysis.get("pattern_signature")), "Unknown pattern"),
                    "lighting_mode": "",
                    "camera_angle": "",
                    "batch_id": f"row{int(cell.get('row', 0)) + 1:02d}_col{int(cell.get('col', 0)) + 1:02d}",
                    "average_rgb": cell.get("overall_rgb"),
                    "estimated_color": cell.get("estimated_rgb"),
                    "capture_mode": "",
                    "scan_run": "",
                    "bitmap": None,
                    "selected_colors_label": "",
                    "scan_station": "",
                }
                if _database_matches_filters(pseudo, selected_filters):
                    matched_cells.append(cell)
            if not matched_cells:
                continue
            visible_analysis_count += 1
            title = f"Analysis {analysis.get('id')} | {analysis.get('created_at')} | {len(matched_cells)} matching batches"
            if imgui.tree_node(f"{title}##db_analysis_{analysis.get('id')}"):
                for cell in matched_cells[:80]:
                    row = int(cell.get("row", 0)) + 1
                    col = int(cell.get("col", 0)) + 1
                    avg = cell.get("overall_rgb", [0, 0, 0])
                    est = cell.get("estimated_rgb", [0, 0, 0])
                    delta = float(cell.get("estimate_actual_delta_rgb", 0.0))
                    rgba = [float(v) / 255.0 for v in avg[:3]]
                    imgui.color_button(f"##db_analysis_avg_{analysis.get('id')}_{row}_{col}", (rgba[0], rgba[1], rgba[2], 1.0), imgui.ColorEditFlags_.no_tooltip, imgui.ImVec2(28, 18))
                    imgui.same_line()
                    imgui.text(f"R{row} C{col} avg {_database_rgb_label(avg)} | est {_database_rgb_label(est)} | delta {delta:.1f}")
                    for angle_result in cell.get("angles", [])[:12]:
                        angle_rgb = angle_result.get("rgb", [0, 0, 0])
                        rgba = [float(v) / 255.0 for v in angle_rgb[:3]]
                        imgui.color_button(f"##db_angle_swatch_{analysis.get('id')}_{row}_{col}_{angle_result.get('angle')}", (rgba[0], rgba[1], rgba[2], 1.0), imgui.ColorEditFlags_.no_tooltip, imgui.ImVec2(20, 14))
                        imgui.same_line()
                        imgui.text_disabled(f"{angle_result.get('angle')}: {_database_rgb_label(angle_rgb)}")
                if len(matched_cells) > 80:
                    imgui.text_disabled(f"... {len(matched_cells) - 80} more batches")
                imgui.tree_pop()
        if visible_analysis_count == 0:
            imgui.text_disabled("No analysis rows match the current filters.")

    imgui.separator()
    imgui.text_wrapped(str(state.scanner_status))


# ============================================================================
# EmbeddedMujocoScanner: the MuJoCo scan runtime
# ============================================================================

class EmbeddedMujocoScanner:
    CAMERA_PREVIEW_INTERVAL = 0.18
    # Robot-viewport redraw rate while a scan is running. A redraw is ~130 ms
    # (86 ms of it MuJoCo's own offscreen render and readPixels) and it only
    # shows progress, so it is skipped on most passes.
    #
    # Both limits are required. A wall-clock interval alone does nothing here:
    # a loop pass already takes far longer than any sensible interval, so the
    # interval has always elapsed and the throttle never fires -- the same trap
    # the camera-preview throttle falls into. The frame count is what actually
    # limits it when passes are slow; the interval takes over if they get fast.
    VIEWPORT_INTERVAL = 0.10
    VIEWPORT_EVERY_N_UPDATES = 3
    # How much work update() starts before handing the frame back. Stages are
    # not interruptible once begun, so this is a "do not start another stage
    # past here" line rather than a hard cap on the pass.
    FRAME_BUDGET = 0.015
    MAX_EXECUTED_TRAIL_POINTS = 300

    def __init__(self, state, gl_ctx, window=None, width=512, height=384, preview_image=None, per_cell_images=None, auto_start=True):
        import fabric_scanner as scanner

        self.scanner = scanner
        self.app_state = state
        self.width = int(width)
        self.height = int(height)
        self.gl_ctx = gl_ctx
        self.window = window
        # A private renderer dedicated to per-capture pattern re-renders (see
        # _render_fresh_focused_capture), so repeatedly resizing/rendering it
        # for scan captures never fights over renderer.resize() with the main
        # "3D View" viewport, which is resized to that panel's size every frame.
        self._pattern_renderer = MeshRenderer(gl_ctx, 480, 360)
        self.texture = None
        self.camera_texture = None
        # Set by the sidebar each frame. The live camera view is produced inside
        # update() rather than when the widget draws, so without this the render
        # happens whether or not anyone can see it -- and it costs more per
        # target than the saved capture does (measured 0.48 s against 0.37 s).
        self.preview_visible = True
        # Robot-viewport rate limiting; see _render_frame.
        self._last_viewport_time = 0.0
        self._viewport_dirty = True
        # The capture currently spread across frames; see _step_capture.
        self._capture_job = None
        self.camera_preview_width = int(scanner.CAMERA_IMAGE_SIZE[0])
        self.camera_preview_height = int(scanner.CAMERA_IMAGE_SIZE[1])
        palette = _scanner_base_palette(state)
        cell_sets = _scanner_shared_cell_color_sets(state)
        self.estimated_cell_colors = _scanner_estimated_cell_colors(state)
        cell_model_curves = _generate_scanner_random_patterns(state)
        pattern_rows, pattern_cols = _scanner_pattern_dimensions(state)
        repeat_rows, repeat_cols = _scanner_pattern_repeats(state)
        spacing_x, spacing_y = _scanner_repeat_spacing(state)
        lighting_settings = _scanner_lighting_settings(state)
        batch_texture_width, batch_texture_height = _scanner_batch_texture_size(state)
        capture_width, capture_height = _scanner_capture_image_size(state, 'scanner_capture_width')
        single_capture_width, single_capture_height = _scanner_capture_image_size(state, 'scanner_single_capture_width')
        self.args = SimpleNamespace(
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
            # The simulated UR5 base frame is rotated exactly 180 degrees,
            # so place the embedded visual fabric on the matching scanning side.
            center=[-0.45, -0.08, 0.30],
            max_span=scanner.DEFAULT_MAX_SPAN.tolist(),
            speed=float(state.scanner_speed),
            dwell=float(state.scanner_dwell),
            add_camera=bool(state.scanner_add_camera),
            save_images=bool(state.scanner_save_images),
            image_dir=str((Path(state.project_root) / 'scanner_images').resolve()),
            image_every=str(state.scanner_image_every),
            capture_mode=str(state.get('scanner_capture_mode', 'natural')),
            camera_zoom=float(state.get('scanner_camera_zoom', 1.0)),
            capture_image_size=(int(capture_width), int(capture_height)),
            single_capture_image_size=(single_capture_width, single_capture_height),
            palette=palette,
            cell_color_sets=cell_sets,
            model_json=str(state.save_path),
            model_curves=cell_model_curves[0] if cell_model_curves else None,
            cell_model_curves=cell_model_curves,
            pattern_repeat_rows=int(repeat_rows),
            pattern_repeat_cols=int(repeat_cols),
            pattern_repeat_spacing_x=float(spacing_x),
            pattern_repeat_spacing_y=float(spacing_y),
            batch_texture_width=int(batch_texture_width),
            batch_texture_height=int(batch_texture_height),
            scanner_lighting=lighting_settings,
            display_batch_colors=_scanner_batch_colors_for_simulator(state),
        )
        # The embedded viewer only needs the true scan targets. Keeping the
        # intermediate densified path for the GUI makes scanning much slower
        # without improving camera-capture accuracy.
        self.plan = scanner.build_plan(self.args)
        if preview_image is not None:
            self.plan.rendered_fabric_image = preview_image.convert("RGB")
            self.plan.rendered_fabric_image_is_full_layout = True
            self.plan.rendered_fabric_image_lit = True
        if per_cell_images:
            # These are TiledFabricTextures (or plain images for a failed cell).
            # Passed through untouched: converting would materialise every glued
            # image, which is what put the working set past 10 GB. fabric_scanner
            # stamps each one out at the resolution its camera quad needs.
            self.plan.scan_tiled_pattern_images = [
                img if getattr(img, "mode", None) == "RGB" else img.convert("RGB")
                for img in per_cell_images
            ]
        # A plain callable, not a gui.py import, so fabric_scanner.py can stay
        # free of any gui.py dependency: it just calls whatever is attached
        # here (falling back to the static per_cell_images/2D-curve paths when
        # absent, e.g. standalone/CLI use with no live renderer).
        self.plan.scan_render_capture_fn = self._render_fresh_focused_capture
        self.pattern_signature = _persist_scanner_state(
            state,
            patterns=_scanner_pattern_database_payload(state),
            estimates=self.estimated_cell_colors,
        )
        self.mujoco, self.model, self.data, self.site_id = scanner.load_ur5e_model_data()
        self.mj_context = None
        self.renderer = None
        try:
            # MuJoCo's offscreen renderer on Windows expects the framebuffer
            # extension to be visible through a compatibility-style context.
            # The main app uses a core-profile context, so reset GLFW hints
            # before MuJoCo creates its hidden render window.
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_ANY_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, False)
            glfw.window_hint(glfw.VISIBLE, False)
            self.mj_context = self.mujoco.GLContext(self.width, self.height)
            self.mj_context.make_current()
            self.renderer = self.mujoco.Renderer(self.model, height=self.height, width=self.width)
            self.camera = self.mujoco.MjvCamera()
            self.mujoco.mjv_defaultFreeCamera(self.model, self.camera)
            robot_min = self.data.xpos[:, :3].min(axis=0)
            robot_max = self.data.xpos[:, :3].max(axis=0)
            fabric_min = self.plan.fabric_origin.copy()
            fabric_max = self.plan.fabric_origin + np.array([
                self.plan.fabric_size[0],
                self.plan.fabric_size[1],
                0.04,
            ])
            scene_min = np.minimum(robot_min, fabric_min)
            scene_max = np.maximum(robot_max, fabric_max)
            scene_center = (scene_min + scene_max) * 0.5
            scene_center[2] = max(scene_center[2], 0.26)
            scene_span = float(np.linalg.norm(scene_max - scene_min))
            self.camera.lookat[:] = scene_center
            self.base_camera_distance = max(1.18, scene_span * 1.55)
            self.view_zoom = 1.0
            self.camera.distance = self.base_camera_distance
            self.camera.azimuth = 235.0
            self.camera.elevation = -20.0
            self.scene_handle = SimpleNamespace(user_scn=self.renderer.scene, sync=lambda: None)
        except Exception:
            # Some MuJoCo/OpenGL backends leave a partially initialized Renderer
            # whose __del__ expects _mjr_context. Avoid a noisy ignored exception.
            try:
                import mujoco.renderer as _mj_renderer
                if hasattr(_mj_renderer, "Renderer") and not hasattr(_mj_renderer.Renderer, "_codex_safe_del"):
                    def _safe_del(obj):
                        try:
                            obj.close()
                        except Exception:
                            pass
                    _mj_renderer.Renderer.__del__ = _safe_del
                    _mj_renderer.Renderer._codex_safe_del = True
            except Exception:
                pass
            self.close()
            raise
        finally:
            if self.window is not None:
                glfw.make_context_current(self.window)
                self.gl_ctx.screen.use()
        self.target_index = 0
        self.dwell_until = 0.0
        self.executed = []
        self.saved_targets = set()
        self.saved_stations = set()
        self.saved_count = 0
        # A new scanner session always starts with no captures/analysis, even
        # if a previous session already analyzed this same pattern -- showing
        # old results before anything has been scanned this session would
        # make the "Average RGB" grid appear before an actual scan happened,
        # contradicting the estimated-before / actual-after workflow. Past
        # results remain fully available in the Database section.
        self.capture_records = []
        self.analysis_results = None
        self.scan_run_id = time.strftime("run_%Y%m%d_%H%M%S")
        self.scan_output_dir = Path(self.args.image_dir) / self.scan_run_id
        self.running = bool(auto_start)
        self.paused = False
        self.latest_camera_image = None
        self.single_capture_mode = False
        self.single_target_active = False
        self.single_target_station = 0
        self._last_camera_preview_time = 0.0
        self._camera_preview_dirty = True
        self._last_camera_preview_key = None
        self._saved_preview_hold_until = 0.0
        self._save_queue = queue.Queue()
        self._save_stop = threading.Event()
        self._save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self._save_thread.start()
        # Composing a capture -- warping the fabric texture into the camera
        # frame and locating the patch -- is ~280 ms of pure PIL/NumPy with no
        # OpenGL in it, and it ran on the frame thread, which is what kept the
        # window at ~3 fps during a scan. It runs here instead. The GL half
        # (rendering this cell's tiles) stays on the frame thread and always
        # runs first, so this worker only ever reads an already-rendered tile.
        self._compose_queue = queue.Queue()
        self._compose_stop = threading.Event()
        self._compose_thread = threading.Thread(target=self._compose_worker, daemon=True)
        self._compose_thread.start()
        self._gl_thread = threading.current_thread()
        self.status = (
            f"Running 1/{len(self.plan.poses)} | saved 0"
            if self.running
            else f"Robot Viewer ready: {len(self.plan.poses)} scan poses"
        )
        self._render_frame()

    def pause(self):
        if self.running:
            self.paused = True
            self.status = f"Paused at {min(self.target_index + 1, len(self.plan.poses))}/{len(self.plan.poses)} | saved {self.saved_count}"

    def resume(self):
        if self.target_index < len(self.plan.poses):
            self.running = True
            self.paused = False
            self.status = f"Running {self.target_index + 1}/{len(self.plan.poses)} | saved {self.saved_count}"

    def _mark_viewport_dirty(self):
        """Forces the next _render_frame to redraw rather than wait its turn.

        The viewport is rate limited while a scan runs, but a camera change is a
        direct response to the user, so it must not be held back.
        """
        self._viewport_dirty = True

    def set_zoom(self, zoom):
        self.view_zoom = float(np.clip(zoom, 0.45, 2.50))
        self.camera.distance = self.base_camera_distance / self.view_zoom
        self._mark_viewport_dirty()

    def zoom_in(self):
        self.set_zoom(self.view_zoom * 1.15)

    def zoom_out(self):
        self.set_zoom(self.view_zoom / 1.15)

    def reset_view(self):
        self.set_zoom(1.0)
        self.camera.azimuth = 235.0
        self.camera.elevation = -20.0
        self._mark_viewport_dirty()

    def orbit_view(self, dx, dy):
        self.camera.azimuth = float((self.camera.azimuth - dx * 0.35) % 360.0)
        self.camera.elevation = float(np.clip(self.camera.elevation + dy * 0.25, -80.0, -5.0))
        self._mark_viewport_dirty()

    def rotate_view(self, delta_azimuth=0.0, delta_elevation=0.0):
        self.camera.azimuth = float((self.camera.azimuth + float(delta_azimuth)) % 360.0)
        self.camera.elevation = float(np.clip(self.camera.elevation + float(delta_elevation), -80.0, -5.0))
        self._mark_viewport_dirty()

    def pan_view(self, dx, dy):
        scale = 0.0014 * float(self.camera.distance)
        az = np.deg2rad(float(self.camera.azimuth))
        right = np.array([np.cos(az), -np.sin(az), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        self.camera.lookat[:] = self.camera.lookat + right * (-dx * scale) + up * (dy * scale)
        self._mark_viewport_dirty()

    # -- Robot camera preview and image capture -------------------------------

    def _capture_image_size(self):
        width, height = getattr(self.args, "capture_image_size", self.scanner.CAMERA_IMAGE_SIZE)
        return clamp_capture_size(width, height)

    def set_capture_resolution(self, width):
        self.args.capture_image_size = clamp_capture_size(width)
        self._camera_preview_dirty = True

    def _single_capture_image_size(self):
        width, height = getattr(self.args, "single_capture_image_size", self.scanner.CAMERA_IMAGE_SIZE)
        return clamp_capture_size(width, height)

    def _active_camera_image_size(self):
        return self._single_capture_image_size() if self.single_capture_mode else self._capture_image_size()

    def set_single_capture_resolution(self, width):
        self.args.single_capture_image_size = clamp_capture_size(width)
        self._camera_preview_dirty = True

    def _camera_render_lighting_key(self):
        lighting = self.scanner._normalize_scanner_lighting(getattr(self.plan, "scanner_lighting", None))
        return tuple((key, round(float(lighting[key]), 4)) for key in sorted(lighting))

    @staticmethod
    def _focused_capture_azimuth(angle_deg):
        """Maps a scan view angle onto the azimuth the pattern is rendered from.

        Compresses the full requested sweep into a modest +-45deg range: enough
        to reveal genuinely different yarn facets under the fixed light (fixing
        the "identical across angles" bug), while keeping the Y-repeat-tiling
        skew tradeoff (see _scan_render_tiled_pattern_images' docstring) mild
        rather than severe. Normalized first so e.g. "angle 300" doesn't
        collapse onto the same azimuth as "angle 60".
        """
        normalized = ((float(angle_deg) + 180.0) % 360.0) - 180.0
        return normalized * (45.0 / 180.0)

    def _cell_capture_azimuths(self, row, col):
        """Every azimuth this scan will ask (row, col) for, in plan order."""
        wanted = []
        for target_index, station_id in enumerate(self.plan.station_ids):
            if tuple(self.plan.station_cells[int(station_id)]) != (int(row), int(col)):
                continue
            view = self.plan.view_names[target_index] if self.plan.view_names else "angle 0"
            az = self._focused_capture_azimuth(self.scanner._view_angle_degrees(view))
            if az not in wanted:
                wanted.append(az)
        return wanted

    def _render_fresh_focused_capture(self, row, col, angle_deg, zoom):
        """Returns a fresh, angle-specific render of one scan pattern's tile.

        The scan visits every angle of a cell consecutively and the pattern's
        geometry is the same for all of them, so the first angle asked for
        builds the mesh once and renders every azimuth this cell will need; the
        rest are served from that batch. Rebuilding per angle was 30% of the
        whole scan loop. Uses the dedicated _pattern_renderer so this never
        disturbs the live "3D View" panel's own renderer/resize.
        """
        state = self.app_state
        zoom = float(zoom)
        az_deg = self._focused_capture_azimuth(angle_deg)
        cache = getattr(self, "_cell_azimuth_tiles", None)
        if cache is None:
            cache = {}
            self._cell_azimuth_tiles = cache
        key = (int(row), int(col), round(az_deg, 6), round(zoom, 4))
        if key in cache:
            return cache[key]

        if threading.current_thread() is not getattr(self, "_gl_thread", threading.current_thread()):
            # Rendering needs the OpenGL context, which belongs to the frame
            # thread. The compose worker must never get here: _capture_stage_tiles
            # warms this cache on the frame thread first, and _capture_dispatch_image
            # refuses to hand off unless the tile it needs is present. Returning
            # None rather than rendering keeps a mistake from crashing the driver.
            return None

        cols = max(1, int(state.get("scanner_cols", 1)))
        cell_index = int(row) * cols + int(col)
        try:
            bitmap = state._scanner_random_bitmap(cell_index)
            loop_heights = state._scanner_loop_heights_for_bitmap(bitmap)
            cell_sets = _scanner_shared_cell_color_sets(state)
            colors = cell_sets[cell_index % len(cell_sets)] if cell_sets else None
            repeat_rows, repeat_cols = _scanner_pattern_repeats(state)

            azimuths = self._cell_capture_azimuths(row, col) or [az_deg]
            if az_deg not in azimuths:
                azimuths = list(azimuths) + [az_deg]

            target_w = max(160, int(round(480 * float(np.clip(zoom, 0.3, 3.0)))))
            rendered = _scan_render_tiled_pattern_images(
                state, self._pattern_renderer, bitmap, loop_heights, colors,
                repeat_cols, repeat_rows, camera_az_degs=azimuths, copies=1,
                target_w=target_w, camera_el_deg=0.0, zoom=zoom,
            )
        except Exception:
            return None

        # Only this cell's tiles are worth keeping: the scan finishes a cell
        # before moving on, and each entry holds a period tile, not a glued image.
        cache.clear()
        for az, texture in rendered.items():
            cache[(int(row), int(col), round(float(az), 6), round(zoom, 4))] = texture
        return cache.get(key)

    def _render_robot_camera_image(self, tcp_pos, target_index, station_id=None, target_pose=None, image_size=None, detect_patch=False):
        target_index = min(int(target_index), len(self.plan.poses) - 1)
        if station_id is None:
            station_id = self.plan.station_ids[target_index]
        station_id = int(station_id)
        if target_pose is None:
            target_pose = self.plan.poses[target_index]
        return self.scanner.render_camera_image(
            self.plan,
            np.asarray(tcp_pos, dtype=float)[:3],
            target_index,
            station_id,
            self.plan.view_names[target_index],
            target_pose=target_pose,
            capture_mode=str(getattr(self.args, "capture_mode", "natural")),
            camera_zoom=float(getattr(self.args, "camera_zoom", 1.0)),
            image_size=image_size if image_size is not None else self._active_camera_image_size(),
            detect_patch=detect_patch,
        )

    def _show_robot_camera_image(self, image, *, hold_seconds=0.0):
        self._upload_camera_preview(image)
        self.latest_camera_image = image.copy()
        now = time.monotonic()
        self._last_camera_preview_time = now
        self._camera_preview_dirty = False
        if hold_seconds > 0.0:
            self._saved_preview_hold_until = max(self._saved_preview_hold_until, now + float(hold_seconds))

    # -- Staged capture ------------------------------------------------------
    #
    # A capture is ~400 ms of work and used to run start-to-finish inside one
    # update(), so the frame loop could not draw or read input for its whole
    # duration. It is split here into stages that update() runs one at a time,
    # returning in between, so the window keeps redrawing while a scan runs.
    # The stages do exactly the same work in the same order; only when they run
    # differs, so the saved images are unaffected.

    def _begin_capture(self, tcp_pos, target_index, station_id):
        self._capture_job = {
            "stage": 0,
            "tcp": np.asarray(tcp_pos, dtype=float).copy(),
            "target_index": int(target_index),
            "station_id": int(station_id),
        }

    def _capture_in_progress(self):
        return getattr(self, "_capture_job", None) is not None

    def _step_capture(self, deadline):
        """Runs capture stages until `deadline` passes. True when the capture is done."""
        job = getattr(self, "_capture_job", None)
        while job is not None:
            stage = job["stage"]
            if stage == 0:
                # GL work, frame thread only: make sure this cell's tiles exist.
                # Only actually renders when the scan reaches a new cell; the
                # other angles of that cell are already in the per-cell cache.
                self._capture_stage_tiles(job)
                job["stage"] = 1
            elif stage == 1:
                # Hand the pure-CPU composition to the worker and yield at once,
                # so the frame thread is free while it runs.
                if self._capture_dispatch_image(job):
                    job["stage"] = 2
                    return False
                # Could not hand off safely; do it here instead.
                self._capture_compose(job)
                job["stage"] = 3
            elif stage == 2:
                if not job["done"].is_set():
                    return False        # still composing; give the frame back
                job["stage"] = 3
            else:
                self._capture_stage_finish(job)
                self._capture_job = None
                return True
            if time.monotonic() >= deadline:
                return False
        return True

    def _capture_stage_tiles(self, job):
        try:
            row, col = self.plan.station_cells[job["station_id"]]
            view = self.plan.view_names[job["target_index"]] if self.plan.view_names else "angle 0"
            zoom = float(getattr(self.args, "camera_zoom", 1.0))
            angle_deg = self.scanner._view_angle_degrees(view)
            job["tile"] = self._render_fresh_focused_capture(int(row), int(col), angle_deg, zoom)
        except Exception:
            job["tile"] = None

    def _capture_dispatch_image(self, job):
        """Queues the composition for the worker. False if it must run inline.

        The worker cannot render, so it is only safe to hand over once the tile
        this capture needs is already in the per-cell cache. If it is not there
        -- a failed render, or a fallback path -- the composition stays on this
        thread, where reaching the renderer is legal.
        """
        if job.get("tile") is None:
            return False
        job["done"] = threading.Event()
        self._compose_queue.put(job)
        return True

    def _compose_worker(self):
        while not self._compose_stop.is_set():
            try:
                job = self._compose_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._capture_compose(job)
            except Exception as exc:
                job["result"] = None
                job["error"] = exc
            finally:
                job["done"].set()
                self._compose_queue.task_done()

    def _capture_compose(self, job):
        """Projects the fabric texture into the camera frame and finds the patch.

        Pure PIL/NumPy: safe on the worker. Everything that mutates scanner
        state stays in _capture_stage_finish on the frame thread, so capture
        records keep their order.
        """
        target_pose = self.plan.poses[min(job["target_index"], len(self.plan.poses) - 1)]
        job["result"] = self._render_robot_camera_image(
            job["tcp"],
            job["target_index"],
            station_id=job["station_id"],
            target_pose=target_pose,
            image_size=self._capture_image_size(),
            detect_patch=True,
        )

    def _capture_stage_finish(self, job):
        if job.get("error") is not None:
            self.status = f"Capture error: {job['error']}"
        self._write_capture_result(
            job.get("result"), job["target_index"], job["station_id"],
        )

    def _save_gripper_camera_image(self, tcp_pos, target_index, station_id):
        """Renders and saves one capture synchronously (single-target path)."""
        target_pose = self.plan.poses[min(target_index, len(self.plan.poses) - 1)]
        result = self._render_robot_camera_image(
            tcp_pos,
            target_index,
            station_id=station_id,
            target_pose=target_pose,
            image_size=self._capture_image_size(),
            detect_patch=True,
        )
        return self._write_capture_result(result, target_index, station_id)

    def _write_capture_result(self, result, target_index, station_id):
        output_dir = Path(getattr(self, "scan_output_dir", Path(self.args.image_dir)))
        output_dir.mkdir(parents=True, exist_ok=True)
        capture_mode = str(getattr(self.args, "capture_mode", "natural"))
        image, patch_image, debug_image, detection = self._unpack_capture_result(result)
        if image is None:
            return None
        self._show_robot_camera_image(image, hold_seconds=0.45)
        active_row, active_col = self.plan.station_cells[station_id]
        clean_view = self.plan.view_names[target_index].replace(" ", "_")
        clean_mode = "focused_batch" if capture_mode == self.scanner.CAMERA_CAPTURE_FOCUSED else "natural"
        stem = (
            f"scan_{target_index + 1:04d}_{clean_mode}_row_{active_row + 1:02d}_col_{active_col + 1:02d}_"
            f"station_{station_id + 1:03d}_{clean_view}"
        )
        # The full image keeps its original name/contract (existing DB records
        # and analyses reference it); the patch and debug-boundary images are
        # new, additional files saved alongside it.
        path = output_dir / f"{stem}.png"
        patch_path = output_dir / f"{stem}_patch.png" if patch_image is not None else None
        debug_path = output_dir / f"{stem}_debug.png" if debug_image is not None else None
        self._record_capture_analysis(
            image, path, target_index, station_id,
            detection=detection, patch_image=patch_image, patch_path=patch_path, debug_path=debug_path,
        )
        self._queue_image_save(image, path)
        if patch_image is not None:
            self._queue_image_save(patch_image, patch_path)
        if debug_image is not None:
            self._queue_image_save(debug_image, debug_path)
        return path

    @staticmethod
    def _unpack_capture_result(result):
        """render_camera_image returns a plain Image normally, or a dict with
        full/patch/debug images + detection metadata when detect_patch=True."""
        if isinstance(result, dict):
            return (
                result.get("full_image"),
                result.get("patch_image"),
                result.get("debug_image"),
                result.get("detection"),
            )
        return result, None, None, None

    def _save_worker(self):
        while not self._save_stop.is_set() or not self._save_queue.empty():
            try:
                image, path = self._save_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
            except Exception as exc:
                self.status = f"Could not save scan image: {exc}"
            finally:
                self._save_queue.task_done()

    def _queue_image_save(self, image, path):
        self._save_queue.put((image.copy(), Path(path)))

    # -- Analysis section: fabric-only color measurements ----------------------

    @staticmethod
    def _fabric_rgb_stats(image, debug_path=None):
        """Measured by rgb_analysis so real UR5 captures share this exact
        definition of "average fabric colour" -- see rgb_analysis.fabric_rgb_stats."""
        return fabric_rgb_stats(image, debug_path=debug_path)

    def _record_capture_analysis(self, image, path, target_index, station_id, detection=None, patch_image=None, patch_path=None, debug_path=None):
        row, col = self.plan.station_cells[station_id]
        view_name = str(self.plan.view_names[target_index])
        # Prefer averaging the already-detected, already-tightly-cropped patch
        # (dynamically located from this specific capture's full image) over
        # re-guessing the fabric region from scratch on the wide frame -- the
        # patch still gets its own brightness/saturation mask applied so any
        # residual background at the crop's small padding border is excluded.
        stats = self._fabric_rgb_stats(patch_image if patch_image is not None else image)
        avg = np.asarray(stats["rgb"], dtype=np.float32)
        record = {
            "row": int(row),
            "col": int(col),
            "station": int(station_id),
            "target_index": int(target_index),
            "angle": view_name,
            "path": str(path),
            "rgb": [float(v) for v in avg],
            "fabric_pixel_count": int(stats["pixel_count"]),
            "analysis_total_pixels": int(stats["total_pixels"]),
            "analysis_mask": str(stats["method"]),
        }
        if patch_path is not None:
            record["patch_image_path"] = str(patch_path)
        if debug_path is not None:
            record["debug_image_path"] = str(debug_path)
        if detection:
            record["patch_bbox"] = detection.get("bbox")
            record["patch_confidence"] = detection.get("confidence")
            record["camera_zoom_level"] = detection.get("zoom")
            record["camera_angle_deg"] = detection.get("angle_deg")
            record["camera_pose"] = detection.get("camera_pose")
        self.capture_records.append(record)
        try:
            storage = _scanner_storage(self.app_state)
            if storage is not None:
                storage.record_capture(self.app_state, record)
        except Exception:
            pass
        self.analysis_results = None

    def analyze_captures(self, save_outputs=True):
        if not self.capture_records:
            self.analysis_results = {"cells": [], "summary": "No captured images to analyze."}
            return self.analysis_results
        output_dir = Path(getattr(self, "scan_output_dir", Path(self.args.image_dir)))
        debug_dir = output_dir / "analysis_used_pixels"
        for record in self.capture_records:
            path = Path(str(record.get("path", "")))
            if not path.exists():
                continue
            try:
                debug_path = None
                if save_outputs:
                    safe_stem = path.stem.replace(" ", "_")
                    debug_path = debug_dir / f"{safe_stem}_used_pixels.png"
                with Image.open(path) as saved_image:
                    stats = self._fabric_rgb_stats(saved_image, debug_path=debug_path)
            except Exception:
                continue
            rgb = np.asarray(stats["rgb"], dtype=np.float32)
            record["rgb"] = [float(v) for v in rgb]
            record["fabric_pixel_count"] = int(stats["pixel_count"])
            record["analysis_total_pixels"] = int(stats["total_pixels"])
            record["analysis_mask"] = str(stats["method"])
            if stats.get("debug_path"):
                record["analysis_used_image"] = str(stats["debug_path"])

        # Grouped by the same helper the real UR5 scan uses, so a simulated and
        # a real analysis of the same fabric are directly comparable rather than
        # merely similar. This block used to be a hand-rolled copy of it.
        result = fabric_rgb_summary(
            self.capture_records,
            getattr(self, "estimated_cell_colors", []),
            pattern_signature=str(getattr(self, "pattern_signature", "")),
        )
        cells = result["cells"]
        if save_outputs:
            output_dir.mkdir(parents=True, exist_ok=True)
            json_path = output_dir / "per_sample_rgb_analysis.json"
            with json_path.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
            swatch_w, swatch_h = 72, 54
            for cell in cells:
                row = int(cell["row"])
                col = int(cell["col"])
                estimate_color = tuple(int(np.clip(v, 0, 255)) for v in cell.get("estimated_rgb", [0, 0, 0]))
                Image.new("RGB", (swatch_w, swatch_h), estimate_color).save(output_dir / f"estimated_rgb_row_{row + 1:02d}_col_{col + 1:02d}.png")
                color = tuple(int(np.clip(v, 0, 255)) for v in cell["overall_rgb"])
                Image.new("RGB", (swatch_w, swatch_h), color).save(output_dir / f"avg_rgb_row_{row + 1:02d}_col_{col + 1:02d}.png")
                for angle_result in cell["angles"]:
                    angle_name = str(angle_result["angle"]).replace(" ", "_").replace("/", "_")
                    angle_color = tuple(int(np.clip(v, 0, 255)) for v in angle_result["rgb"])
                    Image.new("RGB", (swatch_w, swatch_h), angle_color).save(
                        output_dir / f"avg_rgb_row_{row + 1:02d}_col_{col + 1:02d}_{angle_name}.png"
                    )
            result["json_path"] = str(json_path)
            result["used_pixels_dir"] = str(debug_dir)
        self.analysis_results = result
        try:
            storage = _scanner_storage(self.app_state)
            if storage is not None:
                storage.save_analysis(self.app_state, result)
        except Exception:
            pass
        return result

    # -- Scanning section: robot motion and target selection -------------------

    def _append_executed_point(self, point):
        point = np.asarray(point, dtype=float)
        if not self.executed or float(np.linalg.norm(np.asarray(self.executed[-1]) - point)) > 0.004:
            self.executed.append(point.copy())
            if len(self.executed) > self.MAX_EXECUTED_TRAIL_POINTS:
                self.executed = self.executed[-self.MAX_EXECUTED_TRAIL_POINTS:]

    def _adaptive_ik_substeps(self, target_pose, single_target=False):
        tcp = self.scanner.get_tcp(self.mujoco, self.model, self.data, self.site_id)
        err_pos, err_rot = self.scanner.pose_errors(tcp, target_pose)
        speed = max(float(getattr(self.args, "speed", 1.0)), 0.05)
        if single_target:
            speed = max(speed, 1.25)
        if err_pos > 0.08 or err_rot > 0.55:
            return min(36, max(8, int(self.scanner.IK_SUBSTEPS * speed * 5.0)))
        if err_pos > 0.025 or err_rot > 0.22:
            return min(24, max(5, int(self.scanner.IK_SUBSTEPS * speed * 3.2)))
        return min(12, max(2, int(self.scanner.IK_SUBSTEPS * speed * 1.8)))

    def _step_toward_pose(self, target_pose, single_target=False):
        for _ in range(self._adaptive_ik_substeps(target_pose, single_target=single_target)):
            self.scanner.step_ik(self.mujoco, self.model, self.data, self.site_id, target_pose)
        tcp = self.scanner.get_tcp(self.mujoco, self.model, self.data, self.site_id)
        self._append_executed_point(tcp[:3])
        return tcp

    def _scan_indices_for_station(self, station_id):
        non_scan = getattr(self.scanner, "NON_SCAN_VIEW_NAMES", set())
        return [
            idx for idx, sid in enumerate(self.plan.station_ids)
            if int(sid) == int(station_id) and str(self.plan.view_names[idx]).lower() not in non_scan
        ]

    def _single_target_index(self, row, col, angle_index):
        row = int(np.clip(row, 0, self.plan.grid_rows - 1))
        col = int(np.clip(col, 0, self.plan.grid_cols - 1))
        station_id = row * self.plan.grid_cols + col
        indices = self._scan_indices_for_station(station_id)
        if not indices:
            indices = [idx for idx, sid in enumerate(self.plan.station_ids) if int(sid) == int(station_id)]
        if not indices:
            return min(self.target_index, len(self.plan.poses) - 1), station_id
        angle_index = int(np.clip(angle_index, 0, len(indices) - 1))
        return indices[angle_index], station_id

    def preview_single_target(self, row, col, angle_index, camera_zoom):
        target_index, station_id = self._single_target_index(row, col, angle_index)
        self.target_index = target_index
        self.args.camera_zoom = float(camera_zoom)
        self.running = False
        self.paused = True
        self.single_capture_mode = True
        self.single_target_active = True
        self.single_target_station = station_id
        self._render_frame()
        active_row, active_col = self.plan.station_cells[station_id]
        self.status = f"Moving to single target: row {active_row + 1}, col {active_col + 1}, {self.plan.view_names[target_index]}"

    def capture_single_target(self, row, col, angle_index, camera_zoom):
        self.preview_single_target(row, col, angle_index, camera_zoom)
        target_index, station_id = self._single_target_index(row, col, angle_index)
        pose = self.plan.poses[target_index]
        for _ in range(max(32, int(self.scanner.IK_SUBSTEPS) * 8)):
            self.scanner.step_ik(self.mujoco, self.model, self.data, self.site_id, pose)
        self.single_target_active = False
        tcp = self.scanner.get_tcp(self.mujoco, self.model, self.data, self.site_id)
        target_pose = self.plan.poses[target_index]
        self.args.camera_zoom = float(camera_zoom)
        result = self._render_robot_camera_image(
            tcp,
            target_index,
            station_id=station_id,
            target_pose=target_pose,
            image_size=self._single_capture_image_size(),
            detect_patch=True,
        )
        image, patch_image, debug_image, detection = self._unpack_capture_result(result)
        self._show_robot_camera_image(image, hold_seconds=0.0)
        output_dir = Path(getattr(self, "scan_output_dir", Path(self.args.image_dir)))
        output_dir.mkdir(parents=True, exist_ok=True)
        active_row, active_col = self.plan.station_cells[station_id]
        clean_view = self.plan.view_names[target_index].replace(" ", "_")
        clean_mode = "focused_batch" if str(getattr(self.args, "capture_mode", "natural")) == self.scanner.CAMERA_CAPTURE_FOCUSED else "natural"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        stem = (
            f"single_capture_{stamp}_{clean_mode}_row_{active_row + 1:02d}_col_{active_col + 1:02d}_"
            f"station_{station_id + 1:03d}_{clean_view}"
        )
        path = output_dir / f"{stem}.png"
        patch_path = output_dir / f"{stem}_patch.png" if patch_image is not None else None
        debug_path = output_dir / f"{stem}_debug.png" if debug_image is not None else None
        image.save(path)
        if patch_image is not None:
            patch_image.save(patch_path)
        if debug_image is not None:
            debug_image.save(debug_path)
        self._record_capture_analysis(
            image, path, target_index, station_id,
            detection=detection, patch_image=patch_image, patch_path=patch_path, debug_path=debug_path,
        )
        self.saved_count += 1
        self.status = f"Captured single target: {path.name}"
        return path

    # -- Rendering section: simulator and live camera textures -----------------

    def _flush_capture_index(self):
        """Writes the deferred captures_index.json export if one is pending.

        record_capture() now only commits to SQLite; the multi-MB JSON export is
        batched to here, so it costs one write per scan rather than one per
        captured angle. A no-op when nothing is pending, so the finished-scan
        branch below can call it on every frame for free.
        """
        try:
            storage = _scanner_storage(self.app_state)
            if storage is not None:
                storage.flush_json_index()
        except Exception:
            pass

    def close(self):
        self.running = False
        self.paused = False
        self._flush_capture_index()
        try:
            if hasattr(self, '_compose_stop'):
                self._compose_stop.set()
            if hasattr(self, '_compose_thread') and self._compose_thread.is_alive():
                self._compose_thread.join(timeout=1.5)
            if hasattr(self, '_save_stop'):
                self._save_stop.set()
            if hasattr(self, '_save_thread') and self._save_thread.is_alive():
                self._save_thread.join(timeout=1.5)
        except Exception:
            pass
        try:
            if self.window is not None:
                glfw.make_context_current(self.window)
                self.gl_ctx.screen.use()
        except Exception:
            pass
        try:
            if self.texture is not None:
                self.texture.release()
                self.texture = None
        except Exception:
            pass
        try:
            if self.camera_texture is not None:
                self.camera_texture.release()
                self.camera_texture = None
        except Exception:
            pass
        try:
            if getattr(self, "_pattern_renderer", None) is not None:
                if self._pattern_renderer.fbo is not None:
                    self._pattern_renderer.fbo.release()
                    self._pattern_renderer.color_tex.release()
                    if self._pattern_renderer.depth_tex is not None:
                        self._pattern_renderer.depth_tex.release()
                self._pattern_renderer = None
                # Any cached fresh renders belong to the renderer just released.
                if getattr(self, "plan", None) is not None:
                    self.plan.scan_render_capture_fn = None
        except Exception:
            pass
        try:
            if self.mj_context is not None:
                self.mj_context.make_current()
        except Exception:
            pass
        try:
            if self.renderer is not None:
                self.renderer.close()
        except Exception:
            pass
        try:
            if self.mj_context is not None:
                if hasattr(self.mj_context, 'free'):
                    self.mj_context.free()
                elif hasattr(self.mj_context, 'close'):
                    self.mj_context.close()
                if hasattr(self.mj_context, '_context'):
                    self.mj_context._context = None
        except Exception:
            pass
        self.renderer = None
        self.mj_context = None
        try:
            if self.window is not None:
                glfw.make_context_current(self.window)
                self.gl_ctx.screen.use()
        except Exception:
            pass

    def _upload_frame(self, frame):
        if self.window is not None:
            glfw.make_context_current(self.window)
            self.gl_ctx.screen.use()
        # Reallocates when the viewport is resized. The previous version created
        # the texture once at the initial size and only ever wrote to it after,
        # so a resize wrote the wrong number of bytes into it.
        self.texture = upload_rgb_texture(self.gl_ctx, self.texture, frame)

    def _upload_camera_preview(self, image):
        if self.window is not None:
            glfw.make_context_current(self.window)
            self.gl_ctx.screen.use()
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        self.camera_preview_width = int(rgb.shape[1])
        self.camera_preview_height = int(rgb.shape[0])
        self.camera_texture = upload_rgb_texture(self.gl_ctx, self.camera_texture, rgb)

    def _render_camera_preview(self, tcp_pose, target_index, target_pose):
        # Nothing on screen is showing this, so do not spend a full camera
        # render producing it. Saved captures go through
        # _save_gripper_camera_image and are unaffected.
        if not getattr(self, "preview_visible", True):
            return
        if time.monotonic() < self._saved_preview_hold_until and self.latest_camera_image is not None:
            return
        preview_key = (
            int(target_index),
            str(getattr(self.args, "capture_mode", "natural")),
            round(float(getattr(self.args, "camera_zoom", 1.0)), 3),
            self._active_camera_image_size(),
            self._camera_render_lighting_key(),
        )
        now = time.monotonic()
        force = self._camera_preview_dirty or preview_key != self._last_camera_preview_key or self.single_capture_mode
        if not force and self.latest_camera_image is not None and now - self._last_camera_preview_time < self.CAMERA_PREVIEW_INTERVAL:
            return
        station = self.plan.station_ids[min(target_index, len(self.plan.station_ids) - 1)]
        image = self._render_robot_camera_image(
            tcp_pose,
            target_index,
            station_id=station,
            target_pose=target_pose,
            image_size=self._active_camera_image_size(),
        )
        self._show_robot_camera_image(image, hold_seconds=0.0)
        self._last_camera_preview_key = preview_key

    def update(self):
        if self.target_index >= len(self.plan.poses):
            self.running = False
            self.paused = False
            self.status = f"Finished | saved images: {self.saved_count}"
            self._flush_capture_index()
            self._render_frame()
            return

        pose = self.plan.poses[self.target_index]
        if self.single_capture_mode and self.paused and self.single_target_active:
            tcp = self._step_toward_pose(pose, single_target=True)
            err_pos, err_rot = self.scanner.pose_errors(tcp, pose)
            station = int(getattr(self, "single_target_station", self.plan.station_ids[self.target_index]))
            active_row, active_col = self.plan.station_cells[station]
            if err_pos < self.scanner.TARGET_TOL and err_rot < self.scanner.TARGET_ROT_TOL:
                self.single_target_active = False
                self.status = f"Ready to capture: row {active_row + 1}, col {active_col + 1}, {self.plan.view_names[self.target_index]}"
            else:
                self.status = f"Moving to target row {active_row + 1}, col {active_col + 1} | pos err {err_pos:.3f} m"
        elif self.running and not self.paused:
            deadline = time.monotonic() + self.FRAME_BUDGET

            # A capture already under way owns this pass: finish what fits in
            # the budget, then hand the frame back so the window can redraw.
            if self._capture_in_progress():
                if self._step_capture(deadline):
                    self.saved_count += 1
                    self._camera_preview_dirty = True
                    self.target_index += 1
                    self._camera_preview_dirty = True
                    self.dwell_until = 0.0
                self._render_frame()
                return

            tcp = self._step_toward_pose(pose)

            err_pos, err_rot = self.scanner.pose_errors(tcp, pose)
            now = time.monotonic()
            if err_pos < self.scanner.TARGET_TOL and err_rot < self.scanner.TARGET_ROT_TOL:
                dwell = max(float(self.args.dwell), 0.0)
                if dwell <= 0.0 or self.dwell_until == 0.0:
                    self.dwell_until = now + dwell
                if dwell <= 0.0 or now >= self.dwell_until:
                    station = self.plan.station_ids[self.target_index]
                    if self.scanner.should_save_scan_image(
                        self.args,
                        self.plan,
                        self.target_index,
                        self.saved_targets,
                        self.saved_stations,
                        save_images=bool(self.args.save_images),
                    ):
                        self.saved_targets.add(self.target_index)
                        self.saved_stations.add(station)
                        self._begin_capture(tcp, self.target_index, station)
                        if not self._step_capture(deadline):
                            # Stages left to run; the rest of this capture, and
                            # advancing to the next target, happen on following
                            # frames.
                            self._render_frame()
                            return
                        self.saved_count += 1
                        self._camera_preview_dirty = True
                    self.target_index += 1
                    self._camera_preview_dirty = True
                    self.dwell_until = 0.0

        self._render_frame()
        pct = 100.0 * min(self.target_index, len(self.plan.poses)) / max(1, len(self.plan.poses))
        if self.single_capture_mode:
            pass
        elif self.paused:
            self.status = f"Paused {self.target_index + 1}/{len(self.plan.poses)} ({pct:.0f}%) | saved {self.saved_count}"
        elif self.running:
            self.status = f"Running {self.target_index + 1}/{len(self.plan.poses)} ({pct:.0f}%) | saved {self.saved_count}"

    def _render_frame(self, force=False):
        """Advances MuJoCo and, when it is due, redraws the robot viewport.

        Rebuilding and rendering the MuJoCo scene costs ~140 ms, and it was run
        on every pass of the scan loop -- about a quarter of the loop's time
        spent redrawing a progress view far faster than anyone can read it. The
        robot pose itself is still advanced every call; only the redraw is rate
        limited, and a camera change or a paused/finished scan forces it
        immediately so interaction never feels held back.
        """
        self.mujoco.mj_forward(self.model, self.data)
        current_pose = self.scanner.get_tcp(self.mujoco, self.model, self.data, self.site_id)
        target_pose = self.plan.poses[min(self.target_index, len(self.plan.poses) - 1)]

        now = time.monotonic()
        self._viewport_skipped = int(getattr(self, "_viewport_skipped", 0)) + 1
        due = (
            force
            or self.texture is None
            or getattr(self, "_viewport_dirty", False)
            or not (self.running and not self.paused)
            or (
                self._viewport_skipped >= self.VIEWPORT_EVERY_N_UPDATES
                and (now - float(getattr(self, "_last_viewport_time", 0.0))) >= self.VIEWPORT_INTERVAL
            )
        )
        if due:
            self._viewport_dirty = False
            self._viewport_skipped = 0
            self._last_viewport_time = now
            if self.mj_context is not None:
                self.mj_context.make_current()
            self.renderer.update_scene(self.data, self.camera)
            self.scanner.draw_scene(
                self.mujoco,
                self.scene_handle,
                self.plan,
                min(self.target_index, len(self.plan.poses) - 1),
                self.executed,
                camera_enabled=bool(self.args.add_camera),
                current_pose=current_pose,
                target_pose=target_pose,
                clear_scene=False,
                simplified=True,
            )
            frame = self.renderer.render()
            self._upload_frame(frame)
        self._render_camera_preview(
            current_pose,
            min(self.target_index, len(self.plan.poses) - 1),
            target_pose,
        )


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
    if imgui.begin_menu_bar():
        if imgui.begin_menu("Window"):
            clicked_reset, _ = imgui.menu_item("Reset Layout")
            if clicked_reset:
                try:
                    # Located via paths, not from save_path's folder: the saved
                    # parameters live in config/ while the layout file sits in
                    # the project root, so deriving one from the other deletes
                    # nothing and silently does not reset the layout.
                    os.remove(paths.LAYOUT_INI)
                except FileNotFoundError:
                    pass
            imgui.end_menu()
        imgui.end_menu_bar()


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


def _set_app_mode(state, mode):
    mode = mode if mode in ('edit', 'scan', 'puzzle', 'database', 'ur5') else 'edit'
    if str(state.get('app_mode', 'edit')) == mode:
        return
    # Leaving UR5 Robot Mode stops the run, releases the arm and closes the
    # camera. Left connected, a background scan thread would keep moving real
    # hardware while the user is in a mode that shows none of it.
    if str(state.get('app_mode', 'edit')) == 'ur5':
        controller = state.get('ur5_controller')
        if controller is not None:
            try:
                controller.close()
            except Exception:
                pass
            state.ur5_controller = None
    if mode != 'edit' and bool(state.get('sim_active', False)):
        # Stop the solver before handing the model to Scan/Puzzle/Database.
        # Those snapshot ctrl_rows and swap in their own geometry, so a solver
        # still writing to it would fight them. Taken under sim_lock so a step
        # already in flight finishes and sees sim_active False on write-back.
        with state.sim_lock:
            state.sim_active = False
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
    ur5_active = current_mode == 'ur5'
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
    # On its own row, and in its own colour: the four above are simulation, this
    # one drives the physical arm. Crowding it into the same row as the others
    # would present the two as interchangeable, which they are not.
    imgui.push_style_color(imgui.Col_.button, (0.72, 0.42, 0.16, 1.0) if ur5_active else (0.24, 0.18, 0.14, 1.0))
    imgui.push_style_color(imgui.Col_.button_hovered, (0.86, 0.52, 0.20, 1.0))
    if imgui.button("UR5 Robot Mode (real hardware)##mode_ur5", (-1, 0)):
        _set_app_mode(state, 'ur5')
    imgui.pop_style_color(2)
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

    if ur5_active:
        gui_ur5.draw_ur5_panel(state, renderer, window)
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
            # Rebuilt on release, not on every intermediate value. The mesh is
            # (2n+1)^2 copies of the model, so a rebuild runs from 0.08 s at 1x1
            # to 7.4 s at 9x9 (16.3 M verts) -- dragging across the range used to
            # pay every step, ~13 s of rebuilds with the frame loop blocked
            # throughout, which is what made the window stop responding.
            changed_x, new_x = imgui.slider_int("Copy via X##display_copies_x", int(state.display_copies[0]), 0, 20)
            released_x = imgui.is_item_deactivated_after_edit()
            changed_y, new_y = imgui.slider_int("Copy via Y##display_copies_y", int(state.display_copies[1]), 0, 20)
            released_y = imgui.is_item_deactivated_after_edit()
            if changed_x or changed_y:
                # One undo entry per drag, captured before the first change.
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

        if imgui.collapsing_header("Tiling Period"):
            imgui.text_disabled("Vectors a tiled copy is offset by. X drives the")
            imgui.text_disabled("spline period; Y is used by the yarn simulation.")
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
                # Y only feeds the simulation's periodic collision topology, so
                # this needs no mesh rebuild -- just a fresh Jacobian.
                _mark_sim_geometry_dirty(state)
            if imgui.small_button("Re-derive from model##period_resync"):
                state.push_undo("Period resync")
                state.sync_period_offset_to_model_width()
                state.sync_period_offset_y_to_row_count()
                _mark_sim_geometry_dirty(state)
                state.rebuild_spline_mesh(preserve_model_placement=True)

        if imgui.collapsing_header("Yarn Simulation"):
            _draw_yarn_simulation_panel(state)

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
                scan_capture_width, scan_capture_height = clamp_capture_size(scan_capture_width)
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
            capture_width, capture_height = clamp_capture_size(capture_width)
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
        # Geometry is rebuilt when the slider is released rather than on every
        # value it passes through: 9x9 is 81 copies of the model (16.3 M verts,
        # 7.4 s per rebuild), so rebuilding per step meant a drag across the
        # range spent ~13 s with the frame loop blocked.
        changed_px, new_px = imgui.slider_int("Real geometry copies X##puzzle_copies_x", copies_x, 1, 9)
        released_px = imgui.is_item_deactivated_after_edit()
        changed_py, new_py = imgui.slider_int("Real geometry copies Y##puzzle_copies_y", copies_y, 1, 9)
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
            texture = _upload_puzzle_texture(state, renderer, 'capture')
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
            glue_texture = _upload_puzzle_texture(state, renderer, 'glued')
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
        embedded = state.get('embedded_scanner')
        # Folding this away genuinely stops the work, not just the drawing: the
        # live view costs a full camera render per target -- more than the saved
        # capture -- and the scanner only produces it while this is open.
        shown = imgui.collapsing_header("Robot Camera View", imgui.TreeNodeFlags_.default_open)
        if embedded is not None:
            embedded.preview_visible = bool(shown)
        if shown:
            imgui.text_disabled("Live image from the UR5 gripper camera")
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
        else:
            imgui.text_disabled("Live view paused -- scanning runs faster while this is closed.")

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
        texture = _upload_puzzle_texture(state, renderer, 'capture')
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
            origin_x, origin_y = float(state.vp_origin[0]), float(state.vp_origin[1])

            def force_point_to_screen(pt_world):
                """One point to viewport-relative pixels, for the force arrows.

                Distinct from rendering.project_to_screen: it takes a single
                point and offsets by the viewport origin. It was called
                project_to_screen, which shadowed the shared helper throughout
                draw_viewport and broke the closures defined above it.
                """
                h = np.array([pt_world[0], pt_world[1], pt_world[2], 1.0], dtype=np.float32) @ view_proj.T
                if h[3] < 1e-6:
                    return None
                ndc = h[:3] / h[3]
                return (
                    origin_x + (float(ndc[0]) * 0.5 + 0.5) * disp_w,
                    origin_y + (1.0 - (float(ndc[1]) * 0.5 + 0.5)) * disp_h,
                )

            for idx in visible_ctrl_indices:
                idx = int(idx)
                # Virtual (period) control points have no simulated counterpart.
                if idx >= len(delta_P):
                    continue
                p0_local = state.flat_pts_all[idx].astype(np.float32)
                p1_local = p0_local + delta_P[idx] * FORCE_ARROW_SCALE
                p0 = force_point_to_screen(transform_points([p0_local], model_mat)[0])
                p1 = force_point_to_screen(transform_points([p1_local], model_mat)[0])
                if p0 and p1:
                    dl.add_line(p0, p1, force_color, 2.0)
                    dl.add_circle_filled(p1, 3.0, force_color)

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

    imgui.end()


def draw_orbit_viewport(state, window):
    """Second, freely-orbitable view of the same model, over a ground grid.

    The main viewport locks the camera and only lets the model be transformed;
    this one is the opposite, so the fabric can be inspected from any angle
    without disturbing the edit view's framing.
    """
    imgui.set_next_window_pos((1220, 20), cond=imgui.Cond_.first_use_ever)
    imgui.set_next_window_size((360, 370), cond=imgui.Cond_.first_use_ever)
    imgui.begin("3D Orbit View", flags=imgui.WindowFlags_.no_scroll_with_mouse)

    # getattr, not state.get(): orbit_camera/orbit_renderer are live objects
    # held on the instance and deliberately kept out of state._data, so the
    # dict-style accessor would always report them as missing.
    orbit_renderer = getattr(state, 'orbit_renderer', None)
    orbit_camera = getattr(state, 'orbit_camera', None)
    if orbit_renderer is None or orbit_camera is None:
        imgui.text_disabled("No orbit renderer attached.")
        imgui.end()
        return

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

    # Drag continues once started even if the cursor leaves the window, so a
    # fast orbit does not stop dead at the edge.
    if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
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

    imgui.end()
