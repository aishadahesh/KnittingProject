"""Embedded MuJoCo scanner: drives the simulated UR5 scan and owns its GL resources.

Extracted verbatim from gui.py. Imports fabric_scanner lazily inside __init__ (as
before), and takes its scanner helpers from scanner_core, so this module never
imports gui.py -- keeping the dependency edge one-way: gui -> embedded_scanner.
"""
import json
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import glfw
import moderngl
import numpy as np
from PIL import Image

from rendering import MeshRenderer
from scanner_core import (
    _generate_scanner_random_patterns,
    _persist_scanner_state,
    _scan_render_tiled_pattern_image,
    _scanner_base_palette,
    _scanner_batch_colors_for_simulator,
    _scanner_batch_texture_size,
    _scanner_capture_image_size,
    _scanner_estimated_cell_colors,
    _scanner_lighting_settings,
    _scanner_pattern_database_payload,
    _scanner_pattern_dimensions,
    _scanner_pattern_repeats,
    _scanner_repeat_spacing,
    _scanner_shared_cell_color_sets,
    _scanner_storage,
)


class EmbeddedMujocoScanner:
    CAMERA_PREVIEW_INTERVAL = 0.18
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
            self.plan.scan_tiled_pattern_images = [img.convert("RGB") for img in per_cell_images]
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

    def start_path(self):
        if self.target_index >= len(self.plan.poses) or (self.target_index == 0 and self.capture_records):
            self.target_index = 0
            self.dwell_until = 0.0
            self.executed = []
            self.saved_targets = set()
            self.saved_stations = set()
            self.saved_count = 0
            self.capture_records = []
            self.analysis_results = None
            self.scan_run_id = time.strftime("run_%Y%m%d_%H%M%S")
            self.scan_output_dir = Path(self.args.image_dir) / self.scan_run_id
        self.single_capture_mode = False
        self.single_target_active = False
        self.running = True
        self.paused = False
        self.status = f"Running {self.target_index + 1}/{len(self.plan.poses)} | saved {self.saved_count}"

    def set_zoom(self, zoom):
        self.view_zoom = float(np.clip(zoom, 0.45, 2.50))
        self.camera.distance = self.base_camera_distance / self.view_zoom

    def zoom_in(self):
        self.set_zoom(self.view_zoom * 1.15)

    def zoom_out(self):
        self.set_zoom(self.view_zoom / 1.15)

    def reset_view(self):
        self.set_zoom(1.0)
        self.camera.azimuth = 235.0
        self.camera.elevation = -20.0

    def orbit_view(self, dx, dy):
        self.camera.azimuth = float((self.camera.azimuth - dx * 0.35) % 360.0)
        self.camera.elevation = float(np.clip(self.camera.elevation + dy * 0.25, -80.0, -5.0))

    def rotate_view(self, delta_azimuth=0.0, delta_elevation=0.0):
        self.camera.azimuth = float((self.camera.azimuth + float(delta_azimuth)) % 360.0)
        self.camera.elevation = float(np.clip(self.camera.elevation + float(delta_elevation), -80.0, -5.0))

    def pan_view(self, dx, dy):
        scale = 0.0014 * float(self.camera.distance)
        az = np.deg2rad(float(self.camera.azimuth))
        right = np.array([np.cos(az), -np.sin(az), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        self.camera.lookat[:] = self.camera.lookat + right * (-dx * scale) + up * (dy * scale)

    # -- Robot camera preview and image capture -------------------------------

    def _capture_image_size(self):
        width, height = getattr(self.args, "capture_image_size", self.scanner.CAMERA_IMAGE_SIZE)
        width = int(np.clip(int(width), 320, 4096))
        height = int(np.clip(int(height), 240, 3072))
        return width, height

    def set_capture_resolution(self, width):
        width = int(np.clip(int(width), 320, 4096))
        height = int(round(width * 0.75))
        self.args.capture_image_size = (width, height)
        self._camera_preview_dirty = True

    def _single_capture_image_size(self):
        width, height = getattr(self.args, "single_capture_image_size", self.scanner.CAMERA_IMAGE_SIZE)
        width = int(np.clip(int(width), 320, 4096))
        height = int(np.clip(int(height), 240, 3072))
        return width, height

    def _active_camera_image_size(self):
        return self._single_capture_image_size() if self.single_capture_mode else self._capture_image_size()

    def set_single_capture_resolution(self, width):
        width = int(np.clip(int(width), 320, 4096))
        height = int(round(width * 0.75))
        self.args.single_capture_image_size = (width, height)
        self._camera_preview_dirty = True

    def _camera_from_gripper(self, tcp_pos, look_at):
        cam = self.mujoco.MjvCamera()
        self.mujoco.mjv_defaultFreeCamera(self.model, cam)
        look_at = np.asarray(look_at, dtype=float)
        tcp_pos = np.asarray(tcp_pos, dtype=float)
        rel = tcp_pos - look_at
        dist = max(float(np.linalg.norm(rel)), 0.025)
        cam.lookat[:] = look_at
        cam.distance = dist
        cam.azimuth = float(np.degrees(np.arctan2(rel[1], rel[0])))
        cam.elevation = float(np.clip(np.degrees(np.arcsin(rel[2] / dist)), -85.0, 85.0))
        return cam

    def _camera_render_lighting_key(self):
        lighting = self.scanner._normalize_scanner_lighting(getattr(self.plan, "scanner_lighting", None))
        return tuple((key, round(float(lighting[key]), 4)) for key in sorted(lighting))

    def _render_fresh_focused_capture(self, row, col, angle_deg, zoom):
        """Renders a genuinely fresh, angle-specific view of one scan pattern's
        stitch tile for a focused capture. Previously every angle reused one
        static front-on render (identical pixels, only the 2D projection quad
        differed), which is why different angles produced near-identical
        average colors. Uses the dedicated _pattern_renderer so this never
        disturbs the live "3D View" panel's own renderer/resize."""
        state = self.app_state
        cols = max(1, int(state.get("scanner_cols", 1)))
        cell_index = int(row) * cols + int(col)
        try:
            bitmap = state._scanner_random_bitmap(cell_index)
            loop_heights = state._scanner_loop_heights_for_bitmap(bitmap)
            cell_sets = _scanner_shared_cell_color_sets(state)
            colors = cell_sets[cell_index % len(cell_sets)] if cell_sets else None
            repeat_rows, repeat_cols = _scanner_pattern_repeats(state)

            # Compress the full requested sweep into a modest +-45deg azimuth
            # range: enough to reveal genuinely different yarn facets under
            # the fixed light (fixing the "identical across angles" bug),
            # while keeping the Y-repeat-tiling skew tradeoff (see
            # _scan_render_tiled_pattern_image's docstring) mild rather than
            # severe. Normalize first so e.g. "angle 300" doesn't collapse
            # onto the same azimuth as "angle 60".
            normalized = ((float(angle_deg) + 180.0) % 360.0) - 180.0
            az_deg = normalized * (45.0 / 180.0)

            target_w = max(160, int(round(480 * float(np.clip(zoom, 0.3, 3.0)))))
            return _scan_render_tiled_pattern_image(
                state, self._pattern_renderer, bitmap, loop_heights, colors,
                repeat_cols, repeat_rows, copies=1, target_w=target_w,
                camera_az_deg=az_deg, camera_el_deg=0.0, zoom=float(zoom),
            )
        except Exception:
            return None

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

    def _save_gripper_camera_image(self, tcp_pos, target_index, station_id):
        output_dir = Path(getattr(self, "scan_output_dir", Path(self.args.image_dir)))
        output_dir.mkdir(parents=True, exist_ok=True)
        target_pose = self.plan.poses[min(target_index, len(self.plan.poses) - 1)]
        capture_mode = str(getattr(self.args, "capture_mode", "natural"))
        result = self._render_robot_camera_image(
            tcp_pos,
            target_index,
            station_id=station_id,
            target_pose=target_pose,
            image_size=self._capture_image_size(),
            detect_patch=True,
        )
        image, patch_image, debug_image, detection = self._unpack_capture_result(result)
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
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[0] <= 0 or rgb.shape[1] <= 0:
            return {
                "rgb": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                "pixel_count": 0,
                "total_pixels": 0,
                "method": "empty",
            }

        # Ignore the black label strip and scanner annotations. The analysis is
        # the perceived fabric color, not UI text/background color.
        y_offset = 58 if rgb.shape[0] > 70 else 0
        work = rgb[y_offset:, :, :]
        h, w = work.shape[:2]
        total_pixels = int(h * w)

        def finish(mask, method):
            mask = np.asarray(mask, dtype=bool)
            if int(mask.sum()) < 1:
                fabric = work.reshape(-1, 3)
                mask = np.ones((h, w), dtype=bool)
                method = "full frame fallback"
            else:
                fabric = work[mask]

            saved_debug_path = None
            if debug_path is not None and int(mask.sum()) > 0:
                try:
                    ys_mask, xs_mask = np.where(mask)
                    x0, x1 = int(xs_mask.min()), int(xs_mask.max()) + 1
                    y0, y1 = int(ys_mask.min()), int(ys_mask.max()) + 1
                    crop_rgb = np.clip(work[y0:y1, x0:x1], 0, 255).astype(np.uint8)
                    crop_mask = mask[y0:y1, x0:x1]
                    rgba = np.zeros((crop_rgb.shape[0], crop_rgb.shape[1], 4), dtype=np.uint8)
                    rgba[:, :, :3] = crop_rgb
                    rgba[:, :, 3] = np.where(crop_mask, 255, 0).astype(np.uint8)
                    debug_out = Path(debug_path)
                    debug_out.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(rgba, "RGBA").save(debug_out)
                    saved_debug_path = str(debug_out)
                except Exception:
                    saved_debug_path = None

            return {
                "rgb": fabric.mean(axis=0),
                "pixel_count": int(len(fabric)),
                "total_pixels": total_pixels,
                "method": method,
                "debug_path": saved_debug_path,
            }

        # Preferred path: saved scan images draw a bright green rectangle around
        # the target fabric. Use that outline to build an oriented rectangle mask
        # and average only the pixels inside the fabric area.
        green = (
            (work[:, :, 1] > 165.0)
            & (work[:, :, 0] < 90.0)
            & (work[:, :, 2] < 135.0)
        )
        ys, xs = np.where(green)
        if xs.size >= 24:
            pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
            center = pts.mean(axis=0)
            centered = pts - center
            cov = centered.T @ centered / max(float(len(pts) - 1), 1.0)
            try:
                _, vecs = np.linalg.eigh(cov)
                axes = vecs[:, ::-1].astype(np.float32)
                outline_proj = centered @ axes
                lo = outline_proj.min(axis=0)
                hi = outline_proj.max(axis=0)
                margin = 4.0
                if np.all((hi - lo) > margin * 3.0):
                    yy, xx = np.mgrid[0:h, 0:w]
                    grid = np.column_stack((xx.reshape(-1), yy.reshape(-1))).astype(np.float32)
                    proj = (grid - center) @ axes
                    mask = (
                        (proj[:, 0] >= lo[0] + margin)
                        & (proj[:, 0] <= hi[0] - margin)
                        & (proj[:, 1] >= lo[1] + margin)
                        & (proj[:, 1] <= hi[1] - margin)
                    ).reshape(h, w)
                    mask &= ~green
                    if int(mask.sum()) >= 32:
                        brightness = work.max(axis=2)
                        saturation = work.max(axis=2) - work.min(axis=2)
                        color_mask = mask & (brightness > 42.0) & (saturation > 16.0)
                        if int(color_mask.sum()) >= 32:
                            return finish(color_mask, "fabric-outline color mask")
                        else:
                            return finish(mask, "fabric-outline mask")
            except Exception:
                pass

        # Fallback: estimate the scanner background from image edges and keep
        # pixels that differ from that background enough to be fabric.
        edge = np.concatenate([
            work[:8, :, :].reshape(-1, 3),
            work[-8:, :, :].reshape(-1, 3),
            work[:, :8, :].reshape(-1, 3),
            work[:, -8:, :].reshape(-1, 3),
        ])
        bg = np.median(edge, axis=0)
        diff = np.linalg.norm(work - bg[None, None, :], axis=2)
        brightness = work.max(axis=2)
        saturation = work.max(axis=2) - work.min(axis=2)
        mask = (diff > 30.0) & (brightness > 42.0) & (saturation > 16.0)
        mask &= ~green
        if int(mask.sum()) < 32:
            mask = (diff > 22.0) & (brightness > 28.0) & (saturation > 10.0)
        if int(mask.sum()) < 1:
            return finish(np.ones((h, w), dtype=bool), "full frame fallback")
        return finish(mask, "background mask")

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

        grouped = {}
        for record in self.capture_records:
            key = (int(record["row"]), int(record["col"]))
            grouped.setdefault(key, []).append(record)

        cells = []
        estimates_by_pos = {
            (int(item.get("row", 0)), int(item.get("col", 0))): item
            for item in getattr(self, "estimated_cell_colors", [])
        }
        for (row, col), records in sorted(grouped.items()):
            all_rgb = np.asarray([record["rgb"] for record in records], dtype=np.float32)
            overall = all_rgb.mean(axis=0)
            estimate = estimates_by_pos.get((int(row), int(col)), {})
            estimated_rgb = [float(v) for v in estimate.get("rgb", [0.0, 0.0, 0.0])]
            estimate_delta = float(np.linalg.norm(overall - np.asarray(estimated_rgb, dtype=np.float32)))
            angle_results = []
            for angle in sorted({str(record["angle"]) for record in records}):
                angle_rgb = np.asarray([record["rgb"] for record in records if str(record["angle"]) == angle], dtype=np.float32)
                angle_results.append({
                    "angle": angle,
                    "rgb": [float(v) for v in angle_rgb.mean(axis=0)],
                    "count": int(len(angle_rgb)),
                })
            cells.append({
                "row": int(row),
                "col": int(col),
                "estimated_rgb": estimated_rgb,
                "estimated_active_ratio": float(estimate.get("active_ratio", 0.0)),
                "estimate_actual_delta_rgb": estimate_delta,
                "overall_rgb": [float(v) for v in overall],
                "count": int(len(records)),
                "angles": angle_results,
                "fabric_pixel_count": int(sum(int(record.get("fabric_pixel_count", 0)) for record in records)),
                "analysis_total_pixels": int(sum(int(record.get("analysis_total_pixels", 0)) for record in records)),
                "analysis_masks": sorted({str(record.get("analysis_mask", "unknown")) for record in records}),
            })

        result = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image_count": int(len(self.capture_records)),
            "pattern_signature": str(getattr(self, "pattern_signature", "")),
            "cells": cells,
            "background_ignored": True,
            "analysis_note": "Average RGB is computed from detected fabric pixels only; scanner background and label areas are ignored.",
        }
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

    def close(self):
        self.running = False
        self.paused = False
        try:
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
        rgba = np.dstack((frame, np.full(frame.shape[:2], 255, dtype=np.uint8)))
        rgba = np.ascontiguousarray(np.flipud(rgba))
        if self.texture is None:
            self.texture = self.gl_ctx.texture((self.width, self.height), 4, rgba.tobytes())
            self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        else:
            self.texture.write(rgba.tobytes())

    def _upload_camera_preview(self, image):
        if self.window is not None:
            glfw.make_context_current(self.window)
            self.gl_ctx.screen.use()
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        self.camera_preview_width = int(rgb.shape[1])
        self.camera_preview_height = int(rgb.shape[0])
        rgba = np.dstack((rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)))
        rgba = np.ascontiguousarray(np.flipud(rgba))
        if (
            self.camera_texture is None
            or self.camera_texture.size != (self.camera_preview_width, self.camera_preview_height)
        ):
            if self.camera_texture is not None:
                self.camera_texture.release()
            self.camera_texture = self.gl_ctx.texture((self.camera_preview_width, self.camera_preview_height), 4, rgba.tobytes())
            self.camera_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        else:
            self.camera_texture.write(rgba.tobytes())

    def _render_camera_preview(self, tcp_pose, target_index, target_pose):
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
                        self._save_gripper_camera_image(
                            tcp,
                            self.target_index,
                            station,
                        )
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

    def _render_frame(self):
        self.mujoco.mj_forward(self.model, self.data)
        if self.mj_context is not None:
            self.mj_context.make_current()
        self.renderer.update_scene(self.data, self.camera)
        current_pose = self.scanner.get_tcp(self.mujoco, self.model, self.data, self.site_id)
        target_pose = self.plan.poses[min(self.target_index, len(self.plan.poses) - 1)]
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
