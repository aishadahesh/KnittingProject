"""The MuJoCo scan runtime: a simulated UR5e driving the fabric scan.

Scan Mode's robot. It walks the same fabric_scanner plan the real arm does,
steps IK toward each target, renders what the gripper camera would see, saves
the capture and records it against the database.

It draws no UI -- it produces textures and status strings that Scan Mode's panel
displays, which is what lets the panel be tested with this class stubbed out.
Its dependencies are scanner_core (the pattern and tile logic it renders
through) and fabric_scanner (the plan and the camera model); a test enforces
that it never imports imgui.
"""

import json
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import glfw
import numpy as np
from PIL import Image

from fabric_scanner import clamp_capture_size
from rendering import MeshRenderer, upload_rgb_texture
from rgb_analysis import fabric_rgb_stats, summarize_capture_records as fabric_rgb_summary
from scanner_core import (
    _persist_scanner_state, _scan_render_tiled_pattern_images,
    _scanner_base_palette, _scanner_batch_colors_for_simulator,
    _scanner_batch_texture_size, _scanner_capture_image_size,
    _scanner_estimated_cell_colors, _scanner_lighting_settings,
    _scanner_pattern_database_payload, _scanner_pattern_dimensions,
    _scanner_pattern_repeats, _scanner_repeat_spacing, _scanner_shared_cell_color_sets,
    _scanner_storage, _generate_scanner_random_patterns,
)


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
