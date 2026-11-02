"""UR5 Robot Mode's captured images: browse a session's pictures with what each one shows, preview
what a splat trainer will make of them, and look at a trained splat.

Kept apart from gui_ur5 because none of it touches the arm: it reads what a run left on disk -
the images, their sidecar JSON, the session's contact sheet - so it works on any past session as
well as the one that just finished. Textures live on the controller, which owns the GL context
and is the one place released when the mode closes.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from imgui_bundle import imgui
from PIL import Image

import gaussian_splatting
import scan_geometry
from rendering import draw_fitted_texture, upload_rgb_texture

COLOR_OK = (0.25, 0.80, 0.40, 1.0)
COLOR_WARN = (0.95, 0.75, 0.20, 1.0)
COLOR_MUTED = (0.65, 0.65, 0.65, 1.0)

THUMB_WIDTH = 120
THUMBS_SHOWN = 8
VIEW_MAX_WIDTH = 1280
TEXTURE_CACHE = 48
# Never part of a capture's own pictures - what a session folder holds besides them.
NOT_CAPTURES = ("_patch", "_debug", "_fabric_mask", "_detected", "contact_sheet", "used_pixels", "avg_rgb", "splat")

# What a capture is judged by, in the order it is listed under the picture: label, key, format.
DETAILS = (
    ("view",              "capture_view",                  "{}"),
    ("camera height",     "camera_height_above_fabric_m",  "{:.3f} m"),
    ("distance vs first", "relative_distance",             "{:.2f}"),
    ("tilt from vertical", "tilt_from_vertical_deg",       "{:.1f} deg"),
    ("detail",            "fabric_mm_per_pixel",           "{:.3f} mm/px"),
    ("frame covers",      "fabric_across_frame_mm",        "{:.0f} mm across"),
    ("sharpness",         "sharpness",                     "{:.0f}"),
    ("retakes",           "retakes",                       "{}"),
    ("matches picture",   "linked_to",                     "{}"),
    ("fabric in frame",   "fabric_coverage",               "{:.0%}"),
    ("fabric middle off", "fabric_centre_offset",          "{}"),
    ("moved to safety",   "moved_to_nearest_safe",         "{}"),
)


def gallery(controller) -> SimpleNamespace:
    if not hasattr(controller, "gallery"):
        controller.gallery = SimpleNamespace(session="", records=[], index=0, textures={}, show_detected=True,
                                             preview=None, preview_covered=None, splat=None, splat_path="",
                                             splat_view=(0.0, 60.0, 1.0), splat_texture=None, splat_size=(0, 0), splat_object_only=False,
                                             splat_pan=(0.0, 0.0), splat_dragging=False, splat_shown=None, splat_fine=None,
                                             splat_worker=None, splat_result=None, splat_rested_at=0.0, splat_loader=None, splat_cleaner=None,
                                             preview_job=None, preview_ready=None)
    return controller.gallery


# Listings of folders on disk change only when a run writes one, yet the panel draws ~60 times a
# second; they are read again at most this often.
LISTING_REFRESH = 2.0
_listings: dict = {}


def every(key, compute, ttl: float = LISTING_REFRESH):
    """`compute()`, remembered under `key` for `ttl` seconds."""
    now = time.monotonic()
    stamp, value = _listings.get(key, (-1e9, None))
    if now - stamp > ttl:
        value = compute()
        _listings[key] = (now, value)
    return value


def sessions(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted((path.name for path in root.iterdir() if path.is_dir()), reverse=True)


def load_session(folder: Path) -> list[dict]:
    """The session's captures, in order: their sidecar details where written, else just their images."""
    records = []
    for sidecar in sorted(folder.glob("*.json")):
        try:
            record = json.loads(sidecar.read_text())
        except Exception:
            continue
        if isinstance(record, dict) and record.get("path"):
            records.append(record)
    if records:
        return records
    combined = folder / "combined.json"
    if combined.is_file():
        # A combined session stands for several: its pictures are theirs, filtered as its solve will
        # filter them (no video frames, a least sharpness), each marked with the session it came from.
        try:
            settings = json.loads(combined.read_text())
        except Exception:
            settings = {}
        skip_video = bool(settings.get("skip_video", False))
        least = float(settings.get("min_sharpness", 0.0))
        for other in settings.get("sessions", []):
            for record in load_session(Path(other)):
                if skip_video and record.get("camera_position") is None:
                    continue
                if least and record.get("sharpness") is not None and float(record["sharpness"]) < least:
                    continue
                records.append({**record, "combined_from": Path(other).name})
        return records
    return [{"path": str(path)} for path in sorted(folder.glob("*.png")) if not any(tag in path.name for tag in NOT_CAPTURES)]


def release(controller) -> None:
    state = gallery(controller)
    for texture, _ in state.textures.values():
        try:
            texture.release()
        except Exception:
            pass
    state.textures.clear()


def file_texture(controller, path: str, width: int):
    """`path` uploaded at `width`, cached - the gallery redraws every frame and must not re-read files."""
    state = gallery(controller)
    key = (path, width)
    if key not in state.textures:
        if len(state.textures) >= TEXTURE_CACHE:
            stale = next(iter(state.textures))
            state.textures.pop(stale)[0].release()
        with Image.open(path) as image:
            image = image.convert("RGB")
        if image.size[0] > width:
            image = image.resize((width, max(1, int(image.size[1] * width / image.size[0]))), Image.BILINEAR)
        state.textures[key] = (upload_rgb_texture(controller.gl_ctx, None, np.asarray(image)), image.size)
    return state.textures[key]


def _select_session(controller, root: Path, name: str) -> None:
    state = gallery(controller)
    release(controller)
    state.session, state.records, state.index = name, load_session(root / name), 0
    state.preview, state.preview_covered = None, None


SKETCH_HEIGHT = 170.0


def _camera_axis(record):
    """Where the camera looked, from the pose the arm was sent to: the tool's +z, as a unit vector."""
    pose = record.get("target_position") or record.get("actual_tcp_pose")
    if pose is None or len(pose) < 6:
        return None
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(np.asarray(pose[3:6], dtype=float)).as_matrix() @ np.array([0.0, 0.0, 1.0])


def draw_viewpoint_sketch(records, index: int) -> None:
    """Where the arm held the camera for every picture of the session, the current one picked out:
    a map from above (x, y) and an elevation from the side (x, z), with the camera's line of sight
    and the spot it was aimed at. Numbers under it for the current picture. Drawn straight onto
    the window, so it costs nothing to keep up with the gallery."""
    posed = [(i, r) for i, r in enumerate(records) if r.get("camera_position") is not None]
    if not posed:
        imgui.text_colored(COLOR_MUTED, "Point of view: these pictures carry no arm pose (video frames or hand-held).")
        return
    positions = np.array([r["camera_position"][:3] for _, r in posed], dtype=float)
    subjects = np.array([r["capture_subject"][:3] for _, r in posed if r.get("capture_subject") is not None], dtype=float)
    heights = np.array([float(r.get("camera_height_above_fabric_m") or 0.0) for _, r in posed])
    fabric_z = float(np.median(positions[:, 2] - heights)) if heights.any() else float(positions[:, 2].min())
    current = records[index]
    here = np.asarray(current["camera_position"][:3], dtype=float) if current.get("camera_position") is not None else None

    avail = max(240.0, imgui.get_content_region_avail().x)
    gap = imgui.get_style().item_spacing.x
    panel_w = (avail - gap) * 0.5
    origin = imgui.get_cursor_screen_pos()
    draw = imgui.get_window_draw_list()
    col_dot = imgui.get_color_u32(imgui.ImVec4(0.55, 0.6, 0.68, 1.0))
    col_now = imgui.get_color_u32(imgui.ImVec4(0.25, 0.80, 0.40, 1.0))
    col_aim = imgui.get_color_u32(imgui.ImVec4(0.95, 0.75, 0.20, 1.0))
    col_bg = imgui.get_color_u32(imgui.ImVec4(0.08, 0.09, 0.11, 1.0))
    col_text = imgui.get_color_u32(imgui.ImVec4(0.65, 0.65, 0.65, 1.0))
    col_fab = imgui.get_color_u32(imgui.ImVec4(0.35, 0.3, 0.2, 1.0))

    def panel(px, title, xs, ys, extra):
        """One panel: points (xs, ys in metres) fitted into a box, y up."""
        x0, y0 = origin.x + px, origin.y
        draw.add_rect_filled(imgui.ImVec2(x0, y0), imgui.ImVec2(x0 + panel_w, y0 + SKETCH_HEIGHT), col_bg, 4.0)
        draw.add_text(imgui.ImVec2(x0 + 6, y0 + 4), col_text, title)
        lo = np.array([xs.min(), ys.min()]) - 0.02
        hi = np.array([xs.max(), ys.max()]) + 0.02
        span = max(float((hi - lo).max()), 0.05)
        scale = (min(panel_w, SKETCH_HEIGHT) - 36.0) / span
        cx, cy = (lo + hi) / 2.0

        def at(x, y):
            return imgui.ImVec2(x0 + panel_w / 2.0 + (x - cx) * scale, y0 + SKETCH_HEIGHT / 2.0 - (y - cy) * scale + 8.0)

        extra(at)
        for x, y in zip(xs, ys):
            draw.add_circle_filled(at(x, y), 2.5, col_dot)
        return at

    # Map from above.
    def map_extra(at):
        if len(subjects):
            s = np.median(subjects, axis=0)
            draw.add_circle(at(s[0], s[1]), 5.0, col_aim, 12, 1.5)
        if here is not None:
            axis = _camera_axis(current)
            draw.add_circle_filled(at(here[0], here[1]), 5.0, col_now)
            if axis is not None and abs(axis[2]) > 1e-6:
                hit = here - axis * (here[2] - fabric_z) / axis[2]      # where the line of sight meets the cloth
                draw.add_line(at(here[0], here[1]), at(hit[0], hit[1]), col_now, 1.5)
                draw.add_circle(at(hit[0], hit[1]), 3.0, col_now, 12, 1.0)
    panel(0.0, "From above (x right, y up) - dots: every picture; green: this one; yellow ring: the scan's aim", positions[:, 0], positions[:, 1], map_extra)

    # Elevation from the side.
    def side_extra(at):
        left, right = at(positions[:, 0].min() - 0.02, fabric_z), at(positions[:, 0].max() + 0.02, fabric_z)
        draw.add_line(left, right, col_fab, 2.0)
        if here is not None:
            axis = _camera_axis(current)
            draw.add_circle_filled(at(here[0], here[2]), 5.0, col_now)
            if axis is not None and abs(axis[2]) > 1e-6:
                hit = here - axis * (here[2] - fabric_z) / axis[2]
                draw.add_line(at(here[0], here[2]), at(hit[0], hit[2]), col_now, 1.5)
    panel(panel_w + gap, "From the side (x right, height up) - brown line: the cloth", positions[:, 0], positions[:, 2], side_extra)
    imgui.dummy(imgui.ImVec2(avail, SKETCH_HEIGHT + 4.0))

    if here is not None:
        axis = _camera_axis(current)
        tilt = float(np.degrees(np.arccos(np.clip(-axis[2] if axis is not None else 1.0, -1.0, 1.0)))) if axis is not None else 0.0
        heading = float(np.degrees(np.arctan2(axis[1], axis[0]))) if axis is not None and tilt > 0.5 else None
        imgui.text(f"Camera at x {here[0] * 1000:+.0f}  y {here[1] * 1000:+.0f}  z {here[2] * 1000:+.0f} mm (robot base), "
                   f"{(here[2] - fabric_z) * 1000:.0f} mm over the cloth, tilted {tilt:.0f} deg"
                   + (f" towards {heading:+.0f} deg" if heading is not None else " (straight down)"))
    else:
        imgui.text_colored(COLOR_MUTED, "This picture carries no arm pose.")


def draw_captures_section(state, controller, header: str) -> None:
    if not imgui.collapsing_header(header, imgui.TreeNodeFlags_.default_open):
        return
    view = gallery(controller)
    root = Path(state.project_root) / "robot_scans"
    names = every(("sessions", str(root)), lambda: sessions(root))
    runner = controller.runner
    if runner is not None and runner.session_id and not runner.running and runner.session_id in names and view.session != runner.session_id:
        _select_session(controller, root, runner.session_id)
    if not names:
        imgui.text_disabled("No scan sessions yet.")
        return
    if view.session not in names:
        _select_session(controller, root, names[0])

    changed, chosen = imgui.combo("Session##gallery_session", names.index(view.session), names)
    if changed:
        _select_session(controller, root, names[chosen])
    imgui.same_line()
    if imgui.small_button("Reload##gallery_reload"):
        _select_session(controller, root, view.session)
    if runner is not None and runner.running and runner.session_id == view.session and len(runner.capture_records) != len(view.records):
        view.records = load_session(root / view.session)

    if not view.records:
        imgui.text_disabled("This session has no captures.")
        return
    view.index = int(np.clip(view.index, 0, len(view.records) - 1))
    record = view.records[view.index]

    _, view.show_detected = imgui.checkbox("Show the fabric detection##gallery_detected", view.show_detected)
    shown = record.get("detected_image") if view.show_detected and record.get("detected_image") and Path(record["detected_image"]).exists() else record["path"]
    if Path(shown).exists():
        texture, (width, height) = file_texture(controller, shown, VIEW_MAX_WIDTH)
        avail = max(160.0, imgui.get_content_region_avail().x)
        draw_fitted_texture(texture.glo, width, height, avail, min(avail * height / width, 520.0), flip_y=True)

    third = max(70.0, (imgui.get_content_region_avail().x - 2.0 * imgui.get_style().item_spacing.x) / 3.0)
    if imgui.button("< Previous##gallery_prev", (third, 0)):
        view.index = max(0, view.index - 1)
    imgui.same_line()
    imgui.text(f"  {view.index + 1} / {len(view.records)}  ")
    imgui.same_line()
    if imgui.button("Next >##gallery_next", (third, 0)):
        view.index = min(len(view.records) - 1, view.index + 1)

    imgui.text_colored(COLOR_MUTED, Path(record["path"]).name)
    for label, key, fmt in DETAILS:
        value = record.get(key)
        if value is None or value == "":
            continue
        shown_value = fmt.format(value) if not isinstance(value, list) else "[" + ", ".join(f"{v:+.2f}" for v in value) + "]"
        imgui.text(f"{label:<20} {shown_value}")
    if record.get("mostly_background"):
        imgui.text_colored(COLOR_WARN, "Mostly background - left out of the colour analysis")
    if record.get("blurred"):
        imgui.text_colored(COLOR_WARN, "Blurred - below the sharpness threshold even after retakes")
    draw_viewpoint_sketch(view.records, view.index)

    first = int(np.clip(view.index - THUMBS_SHOWN // 2, 0, max(0, len(view.records) - THUMBS_SHOWN)))
    for position, index in enumerate(range(first, min(len(view.records), first + THUMBS_SHOWN))):
        path = view.records[index]["path"]
        if not Path(path).exists():
            continue
        texture, (width, height) = file_texture(controller, path, THUMB_WIDTH)
        if position:
            imgui.same_line()
        tint = (1.0, 1.0, 1.0, 1.0) if index == view.index else (0.6, 0.6, 0.6, 1.0)
        if imgui.image_button(f"##gallery_thumb_{index}", imgui.ImTextureRef(texture.glo), imgui.ImVec2(THUMB_WIDTH * 0.6, height * 0.6),
                              imgui.ImVec2(0, 1), imgui.ImVec2(1, 0), imgui.ImVec4(0, 0, 0, 0), imgui.ImVec4(*tint)):
            view.index = index

    if every(("sheet", view.session), lambda: (root / view.session / "contact_sheet.jpg").exists()):
        imgui.text_colored(COLOR_MUTED, f"Contact sheet and captures.csv are in {root.name}/{view.session}")

    imgui.separator()
    imgui.text("Splatting preview - what a trainer will believe about these poses")
    laying = view.preview_job is not None and view.preview_job.is_alive()
    if laying:
        imgui.begin_disabled()
    if imgui.button("Laying the captures..." if laying else "Lay the captures onto the fabric##gallery_mosaic", (-1, 0)):
        _make_preview(state, controller)
    if laying:
        imgui.end_disabled()
    if view.preview_ready is not None:
        image, covered = view.preview_ready
        view.preview_ready = None
        if view.preview is not None:
            view.preview[0].release()
        view.preview = (upload_rgb_texture(controller.gl_ctx, None, np.asarray(image)), image.size)
        view.preview_covered = covered
    if view.preview is not None:
        texture, (width, height) = view.preview
        avail = max(160.0, imgui.get_content_region_avail().x)
        draw_fitted_texture(texture.glo, width, height, avail, min(avail * height / width, 520.0), flip_y=True)
        imgui.text_colored(COLOR_MUTED, "Stitches that line up across overlapping frames mean the poses are right; doubled or "
                                        "turned ones mean the camera's mount or roll is off - fix that before training.")
        if view.preview_covered is not None:
            imgui.text_colored(COLOR_OK if view.preview_covered > 0.95 else COLOR_WARN,
                               f"{view.preview_covered * 100:.0f}% of the fabric outline is covered by some capture")


def _make_preview(state, controller) -> None:
    view = gallery(controller)
    corners = np.asarray(state.get("teleop_fabric_corners", []), dtype=float).reshape(-1, 3)
    heights = [record.get("camera_position", [0, 0, 0])[2] - record["camera_height_above_fabric_m"]
               for record in view.records if record.get("camera_height_above_fabric_m") is not None and record.get("camera_position")]
    if heights:
        fabric_z = float(np.median(heights))
    elif len(corners):
        fabric_z = scan_geometry.FabricRegion(corners).z
    else:
        controller.status = "Splatting preview needs captures that recorded their height over the fabric"
        return
    outline = scan_geometry.FabricRegion(corners).outline if len(corners) >= 3 else None
    records, fov, turn = list(view.records), float(state.get("ur5_splat_fov", 55.0)), state.get("ur5_camera_image_turn") or None
    saved_to = Path(state.project_root) / "robot_scans" / view.session / "splat_preview.jpg"

    def lay():
        """Off the panel's thread: projecting every capture onto the plane takes seconds."""
        try:
            image, covered = gaussian_splatting.preview_mosaic(records, fabric_z, fov, outline, turn)
            image.save(saved_to, quality=90)
        except Exception as exc:
            controller.status = f"Splatting preview failed: {exc}"
            return
        view.preview_ready = (image, covered)
        controller.status = f"Splatting preview: {len(records)} captures laid onto the fabric (saved as splat_preview.jpg)"

    controller.status = "Laying the captures onto the fabric..."
    view.preview_job = threading.Thread(target=lay, daemon=True, name="splat-preview")
    view.preview_job.start()


def held_out_pairs(output_dir: Path, dataset_root: Path) -> list[tuple[Path, Path]]:
    """Each picture the trainer rendered of a held-out view, beside the real picture it is judged against."""
    pictures = {path.stem: path for path in (dataset_root / "images").glob("*.png")} if (dataset_root / "images").is_dir() else {}
    # ns-render writes JPEGs under renders/<split>/rgb; only the held-out (test) split is judged.
    folder = output_dir / "renders" / "test" / "rgb"
    if not folder.is_dir():
        folder = output_dir / "renders"
    renders = sorted(list(folder.rglob("*.jpg")) + list(folder.rglob("*.png")))
    return [(render, pictures[render.stem]) for render in renders if render.stem in pictures and "/train/" not in render.as_posix()]


def draw_splat_judgement(controller, output_dir: Path, dataset_root: Path) -> None:
    """What the trainer measured on the pictures it never trained on: the PSNR, and a rendered picture beside the real one.

    The number is the honest verdict on a splat; the dot viewer below is only a shape. Under 20 dB the
    splat does not reproduce the pictures; over 28 it does.
    """
    view = gallery(controller)
    evaluation = output_dir / "eval.json"
    if evaluation.is_file():
        try:
            psnr = float(json.loads(evaluation.read_text())["results"]["psnr"])
            imgui.text_colored(COLOR_OK if psnr >= 24.0 else COLOR_WARN,
                               f"Held-out pictures reproduced at {psnr:.1f} dB PSNR"
                               + (" - the splat matches the pictures" if psnr >= 28.0 else
                                  " - rough; the poses or the pictures are off" if psnr < 20.0 else " - fair"))
        except Exception as exc:
            imgui.text_colored(COLOR_WARN, f"Could not read {evaluation.name}: {exc}")
    pairs = every(("renders", str(output_dir)), lambda: held_out_pairs(output_dir, dataset_root))
    if not pairs:
        return
    view.render_index = int(np.clip(getattr(view, "render_index", 0), 0, len(pairs) - 1))
    if imgui.button("< Previous##render_prev"):
        view.render_index = (view.render_index - 1) % len(pairs)
    imgui.same_line()
    imgui.text(f"held-out picture {view.render_index + 1} / {len(pairs)}: the splat's rendering (left) and the real picture (right)")
    imgui.same_line()
    if imgui.button("Next >##render_next"):
        view.render_index = (view.render_index + 1) % len(pairs)
    render, picture = pairs[view.render_index]
    avail = max(160.0, imgui.get_content_region_avail().x)
    half = (avail - imgui.get_style().item_spacing.x) * 0.5
    for index, path in enumerate((render, picture)):
        texture, (width, height) = file_texture(controller, str(path), VIEW_MAX_WIDTH)
        draw_fitted_texture(texture.glo, width, height, half, min(half * height / width, 360.0), flip_y=True)
        if index == 0:
            imgui.same_line()


def _trained_object_only(artifact: Path) -> bool:
    """Whether the dataset a splat was trained on was an object-only export (black background)."""
    for parent in Path(artifact).parents:
        if parent.name.startswith("splat_output"):
            transforms = parent.parent / parent.name.replace("output", "dataset") / "transforms.json"
            try:
                return bool(json.loads(transforms.read_text(encoding="utf-8")).get("object_only"))
            except (OSError, ValueError):
                return False
    return False


def draw_splat_viewer(controller, artifacts) -> None:
    """A trained splat, turned with sliders - the reconstruction, seen without the trainer's GPU viewer."""
    view = gallery(controller)
    plys = [path for path in artifacts if path.suffix.lower() == ".ply"]
    if not plys:
        return
    if view.splat_path != str(plys[0]):
        # A splat file can run to hundreds of megabytes: it is read on a thread, the panel keeps going.
        loader = getattr(view, "splat_loader", None)
        if loader is None or loader[0] != str(plys[0]):
            holder = {}

            def load(path=plys[0]):
                try:
                    holder["splat"] = gaussian_splatting.load_splat_ply(path)
                except Exception as exc:
                    holder["error"] = str(exc)

            thread = threading.Thread(target=load, daemon=True, name="splat-load")
            thread.start()
            view.splat_loader = (str(plys[0]), thread, holder)
            loader = view.splat_loader
        _, thread, holder = loader
        if thread.is_alive():
            imgui.text_colored(COLOR_MUTED, f"Reading {plys[0].name}...")
            return
        view.splat_loader = None
        if "error" in holder:
            imgui.text_colored(COLOR_WARN, f"Could not read {plys[0].name}: {holder['error']}")
            return
        view.splat, view.splat_path = holder["splat"], str(plys[0])
        view.splat_view, view.splat_texture, view.splat_pan = (0.0, 60.0, 1.0), None, (0.0, 0.0)
        view.splat_shown, view.splat_fine, view.splat_worker, view.splat_result = None, None, None, None
        view.splat_object_only = _trained_object_only(plys[0])
    imgui.text_colored(COLOR_OK, f"Splat: {plys[0].name}, {view.splat['count']:,} Gaussians"
                                 + (f" (drawing {len(view.splat['positions']):,})" if view.splat["count"] > len(view.splat["positions"]) else ""))
    object_only = bool(getattr(view, "splat_object_only", False))
    if not plys[0].name.endswith("_clean.ply"):
        imgui.same_line()
        cleaning = getattr(view, "splat_cleaner", None)
        if cleaning is not None and cleaning.is_alive():
            imgui.text_colored(COLOR_MUTED, "Removing floaters...")
        elif imgui.small_button("Remove floaters##splat_clean"):
            source, cleaned = plys[0], plys[0].with_name(plys[0].stem + "_clean.ply")

            def clean():
                try:
                    dropped = gaussian_splatting.clean_splat(source, cleaned, object_only)
                    controller.status = (f"Kept {dropped['kept']:,} of {dropped['total']:,} Gaussians: dropped {dropped['detached']:,} detached, "
                                         f"{dropped['dark']:,} black, {dropped['invisible']:,} transparent -> {cleaned.name}")
                except Exception as exc:
                    controller.status = f"Could not clean the splat: {exc}"

            view.splat_cleaner = threading.Thread(target=clean, daemon=True, name="splat-clean")
            view.splat_cleaner.start()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Drops Gaussians not attached to the main body, and ones that cannot be seen. Writes splat_clean.ply beside it.")
    azimuth, elevation, zoom = view.splat_view
    pan = tuple(getattr(view, "splat_pan", (0.0, 0.0)))
    imgui.text_colored(COLOR_MUTED, "Drag to turn, right-drag to move, wheel to zoom")
    imgui.same_line()
    if imgui.small_button("Reset view##splat_reset"):
        azimuth, elevation, zoom, pan = 0.0, 60.0, 1.0, (0.0, 0.0)

    # Every input is answered at once with a dots-only picture (a few milliseconds); the proper
    # render (a second or two) runs on a thread once the view has rested for a moment, and is
    # swapped in when it is done - the panel never waits for it.
    background = (0, 0, 0) if object_only else (18, 18, 24)
    now = time.monotonic()
    wanted = (azimuth, elevation, zoom, pan)
    if view.splat_texture is None or getattr(view, "splat_shown", None) != wanted:
        image = gaussian_splatting.render_splat_quick(view.splat, azimuth, elevation, zoom, size=(480, 360), background=background, pan=pan)
        view.splat_texture = upload_rgb_texture(controller.gl_ctx, view.splat_texture, np.asarray(image))
        view.splat_size = image.size
        view.splat_shown, view.splat_rested_at, view.splat_fine = wanted, now, None
    worker = getattr(view, "splat_worker", None)
    if worker is not None and not worker.is_alive():
        result = getattr(view, "splat_result", None)
        view.splat_worker, view.splat_result = None, None
        if result is not None and result[0] == view.splat_shown:
            view.splat_texture = upload_rgb_texture(controller.gl_ctx, view.splat_texture, np.asarray(result[1]))
            view.splat_size = result[1].size
            view.splat_fine = result[0]
    if (worker is None and view.splat_fine != wanted and now - getattr(view, "splat_rested_at", now) > 0.25):
        splat = view.splat

        def render(wanted=wanted):
            view.splat_result = (wanted, gaussian_splatting.render_splat(splat, wanted[0], wanted[1], wanted[2], size=(480, 360),
                                                                         background=background, pan=wanted[3]))

        view.splat_worker = threading.Thread(target=render, daemon=True, name="splat-render")
        view.splat_worker.start()
    width, height = view.splat_size
    avail = max(160.0, imgui.get_content_region_avail().x)
    rect = draw_fitted_texture(view.splat_texture.glo, width, height, avail, min(avail * height / width, 480.0), flip_y=True)

    io = imgui.get_io()
    over = False
    if rect is not None:
        x, y, w, h = rect
        over = x <= float(io.mouse_pos.x) <= x + w and y <= float(io.mouse_pos.y) <= y + h and imgui.is_window_hovered()
    dragging = bool(getattr(view, "splat_dragging", False))
    left, right_button = imgui.is_mouse_down(imgui.MouseButton_.left), imgui.is_mouse_down(imgui.MouseButton_.right)
    if (over or dragging) and (left or right_button):
        dx, dy = float(io.mouse_delta.x), float(io.mouse_delta.y)
        if left:
            azimuth = ((azimuth - dx * 0.4 + 180.0) % 360.0) - 180.0
            elevation = float(np.clip(elevation + dy * 0.4, -10.0, 89.0))
        else:
            pan = (pan[0] - dx / max(w if rect else 480.0, 1.0) * 2.0, pan[1] + dy / max(h if rect else 360.0, 1.0) * 2.0)
        view.splat_dragging = True
    elif dragging:
        view.splat_dragging = False
    if over and float(io.mouse_wheel) != 0.0:
        zoom = float(np.clip(zoom * np.exp(float(io.mouse_wheel) * 0.2), 0.2, 8.0))
    view.splat_view, view.splat_pan = (azimuth, elevation, zoom), pan
