"""Database Mode: browse the captures, their metadata and colour analysis.

The whole mode lives here -- filtering, grouping, the card grid, the zoomable
image view and the summary page. `gui.py` calls one function,
_draw_database_summary_page, and this module reaches back for exactly one thing,
scanner_core._scanner_storage, to open the database.

It shows both simulated and real UR5 captures; the Robot and Robot session
filters are what separate them.
"""

import json
import time
from pathlib import Path

import numpy as np
from imgui_bundle import imgui
from PIL import Image

from rendering import draw_fitted_texture, pil_to_texture, upload_rgb_texture
from scanner_core import _scanner_storage
import paths


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
