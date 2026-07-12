import os
import json
import numpy as np
import glfw
import subprocess
import sys
import time
from types import SimpleNamespace
from pathlib import Path

import moderngl
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



def _state_scanner_model_curves(state):
    curves = []
    period = np.asarray(getattr(state, 'period_offset', [1.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
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


def _scanner_base_palette(state):
    if bool(state.use_row_colors):
        source = state.row_colors
    else:
        source = [state.single_model_color]
    count = max(1, int(state.bitmap_size[0]))
    palette = []
    for i in range(count):
        color = source[i % len(source)]
        palette.append([float(color[0]), float(color[1]), float(color[2]), 1.0])
    return palette


def _as_rgba(color, fallback):
    if isinstance(color, np.ndarray):
        color = color.tolist()
    if not isinstance(color, (list, tuple)) or len(color) < 3:
        return list(fallback)
    return [
        float(color[0]),
        float(color[1]),
        float(color[2]),
        float(color[3]) if len(color) > 3 else 1.0,
    ]


def _scanner_default_batch_colors(state):
    variants = state.get('scanner_color_variants', [])
    base = _scanner_base_palette(state)
    fallback = base[0]
    colors = []
    if isinstance(variants, list):
        for color in variants:
            colors.append(_as_rgba(color, fallback))
    return colors or base


def _batch_color_from_saved_cell(saved_cell, fallback):
    if isinstance(saved_cell, np.ndarray):
        saved_cell = saved_cell.tolist()
    if not isinstance(saved_cell, (list, tuple)) or not saved_cell:
        return list(fallback)
    if (
        len(saved_cell) >= 3
        and all(isinstance(v, (int, float, np.integer, np.floating)) for v in saved_cell[:3])
    ):
        return _as_rgba(saved_cell, fallback)
    return _as_rgba(saved_cell[0], fallback)


def _ensure_scanner_batch_colors(state):
    rows = max(1, int(state.scanner_rows))
    cols = max(1, int(state.scanner_cols))
    total = rows * cols
    defaults = _scanner_default_batch_colors(state)
    raw = state.get('scanner_cell_color_sets', [])
    colors = []
    for cell in range(total):
        fallback = defaults[cell % len(defaults)]
        if isinstance(raw, list) and cell < len(raw) and raw[cell]:
            src = _batch_color_from_saved_cell(raw[cell], fallback)
        else:
            src = fallback
        colors.append(_as_rgba(src, fallback))
    return colors


def _ensure_scanner_cell_color_sets(state):
    rows = max(1, int(state.scanner_rows))
    cols = max(1, int(state.scanner_cols))
    total = rows * cols
    base = _scanner_base_palette(state)
    batch_colors = _ensure_scanner_batch_colors(state)
    sets = []
    for cell in range(total):
        batch_color = batch_colors[cell]
        cell_palette = []
        for i in range(len(base)):
            cell_palette.append([
                float(batch_color[0]),
                float(batch_color[1]),
                float(batch_color[2]),
                float(batch_color[3]) if len(batch_color) > 3 else 1.0,
            ])
        sets.append(cell_palette)
    state.scanner_cell_color_sets = sets
    selected = np.asarray(state.get('scanner_selected_cell', [0, 0]), dtype=np.int32).reshape(-1)
    selected_r = int(selected[0]) if selected.size > 0 else 0
    selected_c = int(selected[1]) if selected.size > 1 else 0
    state.scanner_selected_cell = [
        int(np.clip(selected_r, 0, rows - 1)),
        int(np.clip(selected_c, 0, cols - 1)),
    ]
    return sets

# %% GUI DRAWING PANELS ────────────────────────────────────────────────────────



class EmbeddedMujocoScanner:
    def __init__(self, state, gl_ctx, window=None, width=512, height=384):
        import mujoco_fabric_scanner as scanner

        self.scanner = scanner
        self.width = int(width)
        self.height = int(height)
        self.gl_ctx = gl_ctx
        self.window = window
        self.texture = None
        self.camera_texture = None
        self.camera_preview_width = int(scanner.CAMERA_IMAGE_SIZE[0])
        self.camera_preview_height = int(scanner.CAMERA_IMAGE_SIZE[1])
        self.color_picker_mode = str(state.get('scanner_color_mode', 'realistic')) == 'picker'
        palette = _scanner_base_palette(state)
        cell_sets = _ensure_scanner_cell_color_sets(state)
        self.args = SimpleNamespace(
            rows=int(state.scanner_rows),
            cols=int(state.scanner_cols),
            number_of_angles=int(state.scanner_angles),
            width=float(max(0.06, int(state.scanner_cols) * int(state.bitmap_size[1]) * 0.045)),
            length=float(max(0.06, int(state.scanner_rows) * int(state.bitmap_size[0]) * 0.040)),
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
            palette=palette,
            cell_color_sets=cell_sets,
            model_json=str(state.save_path),
            model_curves=_state_scanner_model_curves(state),
            color_picker_mode=self.color_picker_mode,
        )
        self.plan = scanner.densify_plan_for_robot(scanner.build_plan(self.args))
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
            self.camera.azimuth =90
            self.camera.elevation = 270 ## HERE: change robot view to top view
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
        self.running = True
        self.paused = False
        self.status = f"Ready: {len(self.plan.poses)} scan poses"

    def pause(self):
        if self.running:
            self.paused = True
            self.status = f"Paused at {min(self.target_index + 1, len(self.plan.poses))}/{len(self.plan.poses)} | saved {self.saved_count}"

    def resume(self):
        if self.target_index < len(self.plan.poses):
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

    def pan_view(self, dx, dy):
        scale = 0.0014 * float(self.camera.distance)
        az = np.deg2rad(float(self.camera.azimuth))
        right = np.array([np.cos(az), -np.sin(az), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        self.camera.lookat[:] = self.camera.lookat + right * (-dx * scale) + up * (dy * scale)

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

    def _save_gripper_camera_image(self, tcp_pos, target_index, station_id):
        output_dir = Path(self.args.image_dir)
        target_pose = self.plan.poses[min(target_index, len(self.plan.poses) - 1)]
        return self.scanner.save_camera_image(
            self.plan,
            tcp_pos[:3],
            output_dir,
            target_index,
            station_id,
            self.plan.view_names[target_index],
            target_pose=target_pose,
            color_picker_mode=self.color_picker_mode,
        )

    def close(self):
        self.running = False
        self.paused = False
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
        station = self.plan.station_ids[min(target_index, len(self.plan.station_ids) - 1)]
        image = self.scanner.render_camera_image(
            self.plan,
            tcp_pose[:3],
            min(target_index, len(self.plan.poses) - 1),
            station,
            self.plan.view_names[min(target_index, len(self.plan.view_names) - 1)],
            target_pose=target_pose,
            color_picker_mode=self.color_picker_mode,
        )
        self._upload_camera_preview(image)

    def update(self):
        if self.target_index >= len(self.plan.poses):
            self.running = False
            self.paused = False
            self.status = f"Finished | saved images: {self.saved_count}"
            self._render_frame()
            return

        pose = self.plan.poses[self.target_index]
        if self.running and not self.paused:
            substeps = min(6, max(1, int(self.scanner.IK_SUBSTEPS * max(float(self.args.speed), 0.05))))
            for _ in range(substeps):
                self.scanner.step_ik(self.mujoco, self.model, self.data, self.site_id, pose)
            tcp = self.scanner.get_tcp(self.mujoco, self.model, self.data, self.site_id)
            self.executed.append(tcp[:3].copy())

            err_pos, err_rot = self.scanner.pose_errors(tcp, pose)
            now = time.monotonic()
            if err_pos < self.scanner.TARGET_TOL and err_rot < self.scanner.TARGET_ROT_TOL:
                if self.dwell_until == 0.0:
                    self.dwell_until = now + max(float(self.args.dwell), 0.0)
                elif now >= self.dwell_until:
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
                    self.target_index += 1
                    self.dwell_until = 0.0

        self._render_frame()
        pct = 100.0 * min(self.target_index, len(self.plan.poses)) / max(1, len(self.plan.poses))
        if self.paused:
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
            color_picker_mode=self.color_picker_mode,
        )
        frame = self.renderer.render()
        self._upload_frame(frame)
        self._render_camera_preview(
            current_pose,
            min(self.target_index, len(self.plan.poses) - 1),
            target_pose,
        )


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
    mode = 'scan' if mode == 'scan' else 'edit'
    if str(state.get('app_mode', 'edit')) == mode:
        return
    state.app_mode = mode
    scanner_idx = next((i for i, item in enumerate(state.workflow_stages) if item[0] == 'Scanner'), 0)
    if mode == 'scan':
        state.workflow_step = scanner_idx
        state.scanner_preview_grid_enabled = True
        state.scanner_preview_rows = max(1, int(state.scanner_rows))
        state.scanner_preview_cols = max(1, int(state.scanner_cols))
        _ensure_scanner_cell_color_sets(state)
    else:
        state.workflow_step = 0
        state.scanner_preview_grid_enabled = False
        state.display_copies = np.array([0, 0], dtype=np.int32)
        embedded = state.get('embedded_scanner')
        if embedded is not None:
            try:
                embedded.close()
            except Exception:
                pass
            state.embedded_scanner = None
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


def draw_sidebar(state, renderer, window=None):
    _apply_ui_theme(state)
    imgui.set_next_window_pos((20, 20), cond=imgui.Cond_.first_use_ever)
    imgui.set_next_window_size((360, 820), cond=imgui.Cond_.first_use_ever)
    imgui.begin("Knitting Control")

    def rebuild_current_mesh(preserve=True):
        state.rebuild_spline_mesh(preserve_model_placement=preserve)

    def scanner_process_running():
        proc = getattr(state, 'scanner_process', None)
        return proc is not None and proc.poll() is None

    def update_scanner_process_status():
        proc = getattr(state, 'scanner_process', None)
        if proc is None:
            return
        code = proc.poll()
        if code is None:
            elapsed = max(0.0, time.time() - float(getattr(state, 'scanner_started_at', 0.0)))
            state.scanner_status = f"Scanner running ({elapsed:.0f}s)"
        else:
            state.scanner_status = "Scanner finished" if code == 0 else f"Scanner stopped/error ({code})"
            state.scanner_process = None

    def start_scanner_process():
        embedded = state.get('embedded_scanner')
        if embedded is not None:
            if getattr(embedded, 'paused', False):
                embedded.resume()
                state.scanner_status = "Embedded MuJoCo scanner continued"
                return
            if getattr(embedded, 'running', False):
                state.scanner_status = "Scanner already running"
                return
            embedded.close()
            state.embedded_scanner = None
        if scanner_process_running():
            state.scanner_status = "Scanner already running"
            return
        mode = str(state.scanner_execution_mode)
        _ensure_scanner_cell_color_sets(state)
        state.save_params(state.save_path, silent=True)
        if mode == "simulation":
            try:
                previous = state.get('embedded_scanner')
                if previous is not None:
                    previous.close()
                state.embedded_scanner = EmbeddedMujocoScanner(state, renderer.ctx, window)
                state.scanner_status = "Embedded MuJoCo scanner running"
            except Exception as exc:
                state.embedded_scanner = None
                state.scanner_status = f"Could not start embedded MuJoCo scanner: {exc}"
            return

        cmd = [
            sys.executable,
            os.path.join(state.project_root, "mujoco_fabric_scanner.py"),
            "--execution-mode", mode,
            "--no-setup-gui",
            "--no-run-gui",
            "--model-json", str(state.save_path),
            "--rows", str(int(state.scanner_rows)),
            "--cols", str(int(state.scanner_cols)),
            "--number-of-angles", str(int(state.scanner_angles)),
            "--speed", f"{float(state.scanner_speed):.3f}",
            "--dwell", f"{float(state.scanner_dwell):.3f}",
        ]
        if bool(state.scanner_add_camera):
            cmd.append("--add-camera")
        if bool(state.scanner_save_images):
            cmd.extend(["--save-images", "--image-every", str(state.scanner_image_every)])
        cell_color_sets = _ensure_scanner_cell_color_sets(state)
        cmd.extend(["--cell-colors-json", json.dumps(cell_color_sets)])
        cmd.extend(["--robot-ip", str(state.scanner_robot_ip), "--robot-port", str(int(state.scanner_robot_port))])
        try:
            state.scanner_process = subprocess.Popen(cmd, cwd=state.project_root)
            state.scanner_started_at = time.time()
            state.scanner_status = "Real UR5 command running from selected GUI settings"
        except Exception as exc:
            state.scanner_process = None
            state.scanner_status = f"Could not start scanner: {exc}"

    def stop_scanner_process():
        embedded = state.get('embedded_scanner')
        if embedded is not None:
            embedded.pause()
            state.scanner_status = embedded.status
            return
        proc = getattr(state, 'scanner_process', None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            state.scanner_status = "Stopping scanner"
        else:
            state.scanner_process = None
            state.scanner_status = "Scanner idle"

    current_mode = str(state.get('app_mode', 'edit'))
    edit_active = current_mode != 'scan'
    button_w = max(120, (imgui.get_content_region_avail().x - imgui.get_style().item_spacing.x) * 0.5)
    imgui.push_style_color(imgui.Col_.button, (0.22, 0.48, 0.78, 1.0) if edit_active else (0.20, 0.20, 0.20, 1.0))
    if imgui.button("Edit Mode##mode_edit", (button_w, 0)):
        _set_app_mode(state, 'edit')
    imgui.pop_style_color()
    imgui.same_line()
    imgui.push_style_color(imgui.Col_.button, (0.22, 0.48, 0.78, 1.0) if not edit_active else (0.20, 0.20, 0.20, 1.0))
    if imgui.button("Scan Mode##mode_scan", (button_w, 0)):
        _set_app_mode(state, 'scan')
    imgui.pop_style_color()
    imgui.separator()

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
    if imgui.button("Undo##main", (button_w, 0)):
        state.undo_last()
    if undo_disabled:
        imgui.end_disabled()
    imgui.same_line()
    if imgui.button("Reset initial##reset_saved_initial_global", (button_w, 0)):
        state.reset_to_initial()
    imgui.separator()

    if edit_active:
        if bool(state.get('scanner_preview_grid_enabled', False)):
            state.scanner_preview_grid_enabled = False
            state.display_copies = np.array([0, 0], dtype=np.int32)
            state.rebuild_spline_mesh(preserve_model_placement=False)

        if imgui.collapsing_header("Pattern", imgui.TreeNodeFlags_.default_open):
            max_rows = int(state.config['knit_parameters']['bitmap_rows'])
            ch_r, new_rows = imgui.slider_int("Rows##bres", int(state.bitmap_size[0]), 1, max_rows)
            ch_c, new_cols = imgui.slider_int("Columns##bres", int(state.bitmap_size[1]), 1, 32)
            if ch_r or ch_c:
                state.push_undo("Bitmap size")
                state.on_bitmap_resize(new_rows, new_cols)
            if imgui.small_button("All active##bmap"):
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
                    if imgui.button(f"##bm_{r}_{c}", imgui.ImVec2(cell_w, cell_h)):
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

    else:
        imgui.text("Scanner")
        update_scanner_process_status()
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
            _ensure_scanner_cell_color_sets(state)
            embedded = state.get('embedded_scanner')
            if embedded is not None:
                try:
                    embedded.close()
                except Exception:
                    pass
                state.embedded_scanner = None
                state.scanner_status = "Scanner layout changed; press Start Scanner to rebuild from latest model"
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
        imgui.text("Batch colors")
        cell_sets = _ensure_scanner_cell_color_sets(state)
        selected = np.asarray(state.scanner_selected_cell, dtype=np.int32).reshape(-1)
        selected_r = int(selected[0]) if selected.size > 0 else 0
        selected_c = int(selected[1]) if selected.size > 1 else 0
        selected_r = int(np.clip(selected_r, 0, max(0, int(state.scanner_rows) - 1)))
        selected_c = int(np.clip(selected_c, 0, max(0, int(state.scanner_cols) - 1)))
        selected_index = selected_r * max(1, int(state.scanner_cols)) + selected_c
        if not state.get('scanner_color_mode'):
            state.scanner_color_mode = 'realistic'
        realistic_mode = str(state.get('scanner_color_mode', 'realistic')) == 'realistic'
        if imgui.radio_button("Realistic model preview##scanner_color_realistic", realistic_mode):
            if str(state.get('scanner_color_mode', 'realistic')) != 'realistic':
                state.scanner_color_mode = 'realistic'
                embedded = state.get('embedded_scanner')
                if embedded is not None:
                    try:
                        embedded.close()
                    except Exception:
                        pass
                    state.embedded_scanner = None
                state.scanner_status = "Scanner view changed; press Start Scanner to rebuild"
        imgui.same_line()
        if imgui.radio_button("Color picker mode##scanner_color_picker", not realistic_mode):
            if str(state.get('scanner_color_mode', 'realistic')) != 'picker':
                state.scanner_color_mode = 'picker'
                embedded = state.get('embedded_scanner')
                if embedded is not None:
                    try:
                        embedded.close()
                    except Exception:
                        pass
                    state.embedded_scanner = None
                state.scanner_status = "Scanner view changed; press Start Scanner to rebuild"
        imgui.text_colored((1.0, 0.78, 0.18, 1.0), f"Selected batch: R{selected_r + 1} C{selected_c + 1}")
        imgui.text_disabled("Click a mini-square in the 3D view, or use the batch grid below.")

        cell_changed = False
        if not realistic_mode:
            imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(3, 3))
            for r in range(max(1, int(state.scanner_rows))):
                for c in range(max(1, int(state.scanner_cols))):
                    idx = r * max(1, int(state.scanner_cols)) + c
                    palette = cell_sets[idx]
                    first = palette[0]
                    is_selected = idx == selected_index
                    btn_col = (float(first[0]), float(first[1]), float(first[2]), 1.0)
                    hover_col = tuple(min(1.0, float(ch) + 0.15) for ch in btn_col[:3]) + (1.0,)
                    active_col = (1.0, 0.78, 0.18, 1.0) if is_selected else btn_col
                    imgui.push_style_color(imgui.Col_.button, active_col)
                    imgui.push_style_color(imgui.Col_.button_hovered, hover_col)
                    imgui.push_style_color(imgui.Col_.button_active, active_col)
                    if imgui.button(f"{r + 1},{c + 1}##scanner_batch_{r}_{c}", imgui.ImVec2(42, 24)):
                        state.scanner_selected_cell = [r, c]
                        selected_r, selected_c, selected_index = r, c, idx
                    imgui.pop_style_color(3)
                    if c < int(state.scanner_cols) - 1:
                        imgui.same_line()
            imgui.pop_style_var()
        else:
            imgui.text_wrapped("Realistic preview is shown in the main 3D view. The selected mini-square is highlighted there; this color is a batch tint, not an edit to the original model internals.")

        selected_color = cell_sets[selected_index][0]
        changed_color, new_color = imgui.color_edit3(
            f"Batch color##scanner_batch_color_{selected_index}",
            (float(selected_color[0]), float(selected_color[1]), float(selected_color[2])),
        )
        if changed_color:
            batch_color = [float(new_color[0]), float(new_color[1]), float(new_color[2]), 1.0]
            state.scanner_cell_color_sets[selected_index] = [
                list(batch_color)
                for _ in range(max(1, int(state.bitmap_size[0])))
            ]
            cell_changed = True
        if imgui.small_button("Use Edit Mode base color##scanner_copy_selected"):
            base = _scanner_base_palette(state)[0]
            state.scanner_cell_color_sets[selected_index] = [
                list(base)
                for _ in range(max(1, int(state.bitmap_size[0])))
            ]
            cell_changed = True
        imgui.same_line()
        if imgui.small_button("Apply selected to all##scanner_apply_all"):
            selected_batch = list(state.scanner_cell_color_sets[selected_index][0])
            batch_palette = [
                list(selected_batch)
                for _ in range(max(1, int(state.bitmap_size[0])))
            ]
            state.scanner_cell_color_sets = [
                [list(c) for c in batch_palette]
                for _ in range(max(1, int(state.scanner_rows)) * max(1, int(state.scanner_cols)))
            ]
            cell_changed = True
        if cell_changed:
            state.rebuild_spline_mesh(preserve_model_placement=False)
            embedded = state.get('embedded_scanner')
            if embedded is not None:
                try:
                    embedded.close()
                except Exception:
                    pass
                state.embedded_scanner = None
                state.scanner_status = "Batch colors changed; press Start Scanner to rebuild scanner view"
        imgui.text_wrapped("Scan Mode duplicates the edited fabric square. The picker assigns one color to each mini-square/batch while keeping the original model structure, pattern, and texture unchanged.")
        mode_is_robot = str(state.scanner_execution_mode) == "robot"
        changed_mode, new_mode_robot = imgui.checkbox("Run real UR5 robot##scanner_mode", mode_is_robot)
        if changed_mode:
            state.scanner_execution_mode = "robot" if new_mode_robot else "simulation"
        changed_speed, new_speed = imgui.slider_float("Simulation speed##scanner_speed", float(state.scanner_speed), 0.05, 2.0, "%.2f")
        changed_dwell, new_dwell = imgui.slider_float("Dwell per view (s)##scanner_dwell", float(state.scanner_dwell), 0.0, 2.0, "%.2f")
        if changed_speed:
            state.scanner_speed = float(new_speed)
        if changed_dwell:
            state.scanner_dwell = float(new_dwell)
        _, state.scanner_add_camera = imgui.checkbox("Show scanner camera##scanner_camera", bool(state.scanner_add_camera))
        _, state.scanner_save_images = imgui.checkbox("Save scanner images##scanner_images", bool(state.scanner_save_images))
        if bool(state.scanner_save_images):
            every_view = str(state.scanner_image_every) == "view"
            changed_every, every_view = imgui.checkbox("Save every angle view##scanner_every", every_view)
            if changed_every:
                state.scanner_image_every = "view" if every_view else "station"
        if str(state.scanner_execution_mode) == "robot":
            changed_ip, ip = imgui.input_text("Robot IP##scanner_robot_ip", str(state.scanner_robot_ip), 64)
            changed_port, port = imgui.input_int("Robot port##scanner_robot_port", int(state.scanner_robot_port))
            if changed_ip:
                state.scanner_robot_ip = ip
            if changed_port:
                state.scanner_robot_port = int(port)
        embedded = state.get('embedded_scanner')
        embedded_running = embedded is not None and getattr(embedded, 'running', False) and not getattr(embedded, 'paused', False)
        embedded_paused = embedded is not None and getattr(embedded, 'paused', False)
        running = scanner_process_running() or embedded_running
        if imgui.button("Start Scanner Now##scanner_start", (-1, 0)):
            start_scanner_process()
        if embedded_paused:
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
        imgui.text_wrapped(str(state.scanner_status))

    embedded = state.get('embedded_scanner')
    if str(state.get('app_mode', 'edit')) == 'scan' and embedded is not None:
        try:
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
        imgui.begin("UR5 Scanner")
        imgui.text(str(embedded.status))
        changed_zoom, zoom = imgui.slider_float("Zoom##scanner_view_zoom", float(embedded.view_zoom), 0.45, 2.50, "%.2fx")
        if changed_zoom:
            embedded.set_zoom(zoom)
        if imgui.small_button("Zoom in##scanner_view"):
            embedded.zoom_in()
        imgui.same_line()
        if imgui.small_button("Zoom out##scanner_view"):
            embedded.zoom_out()
        imgui.same_line()
        if imgui.small_button("Reset view##scanner_view"):
            embedded.reset_view()
        imgui.separator()
        imgui.text("UR5 Simulator View")
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
                if window is not None:
                    lmb = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
                    rmb = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
                    mmb = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
                    dx, dy = float(io.mouse_delta.x), float(io.mouse_delta.y)
                    if (abs(dx) > 0.0 or abs(dy) > 0.0) and lmb:
                        embedded.orbit_view(dx, dy)
                    elif (abs(dx) > 0.0 or abs(dy) > 0.0) and (rmb or mmb):
                        embedded.pan_view(dx, dy)
            imgui.text_disabled("Mouse: wheel zoom, left-drag rotate, right/middle-drag pan")
        imgui.end()

    embedded = state.get('embedded_scanner')
    if str(state.get('app_mode', 'edit')) == 'scan':
        imgui.set_next_window_pos((980, 80), cond=imgui.Cond_.first_use_ever)
        imgui.set_next_window_size((380, 320), cond=imgui.Cond_.first_use_ever)
        imgui.begin("Robot Camera View", flags=imgui.WindowFlags_.no_collapse)
        imgui.text("Live gripper camera")
        imgui.text_disabled("Matches saved scanner images")
        if embedded is None:
            imgui.separator()
            imgui.text_wrapped("Press Start Scanner to create the live robot-camera view.")
        elif embedded.camera_texture is None:
            imgui.separator()
            imgui.text_wrapped("Waiting for the first robot-camera frame...")
            imgui.text_disabled(str(getattr(embedded, 'status', 'Scanner initializing')))
        else:
            avail = imgui.get_content_region_avail()
            preview_w = max(1, int(avail.x))
            preview_h = max(1, int(min(avail.y, preview_w * 0.75)))
            draw_fitted_texture(
                embedded.camera_texture.glo,
                embedded.camera_preview_width,
                embedded.camera_preview_height,
                preview_w,
                preview_h,
                flip_y=False,
            )
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
    scanner_picker_mode = scanner_stage_active and str(state.get('scanner_color_mode', 'realistic')) == 'picker'

    render_hover_idx = state.hover_idx
    render_selected_idx = state.selected_idx
    visible_ctrl_indices = np.empty((0,), dtype=np.int32)
    visible_ctrl_index_map = {}
    if state.mode == 'spline':
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
    if scanner_picker_mode:
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

    renderer.render(
        mvp, mv,
        state.get_material_uniforms(),
        render_hover_idx, render_selected_idx,
        hover_mesh_idx=state.hover_mesh_idx,
        selected_mesh_idx=state.selected_mesh_idx,
        visible_rows=np.zeros(max(1, len(state.row_visible)), dtype=bool) if scanner_picker_mode else state.row_visible,
        bg_tex      = None if scanner_picker_mode else (ref_tex if state.show_ref_bg else None),
        bg_alpha    = 0.0 if scanner_picker_mode else state.ref_bg_alpha,
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
        if not scanner_picker_mode:
            rect_col = imgui.get_color_u32((0.95, 0.95, 0.95, 0.92))
            dl.add_rect((ox + x_min, oy + y_min), (ox + x_max, oy + y_max), rect_col, 0.0, 2.0, 0)
        if scanner_stage_active and scanner_cell_bounds:
            rows = max(1, int(getattr(state, 'scanner_rows', state.bitmap_size[0])))
            cols = max(1, int(getattr(state, 'scanner_cols', state.bitmap_size[1])))
            selected = np.asarray(getattr(state, 'scanner_selected_cell', [0, 0]), dtype=np.int32).reshape(-1)
            selected_r = int(np.clip(selected[0] if selected.size > 0 else 0, 0, rows - 1))
            selected_c = int(np.clip(selected[1] if selected.size > 1 else 0, 0, cols - 1))
            grid_col = imgui.get_color_u32((0.15, 0.85, 1.0, 0.34))
            fill_col = imgui.get_color_u32((0.15, 0.85, 1.0, 0.055))
            selected_fill = imgui.get_color_u32((1.0, 0.78, 0.18, 0.18))
            selected_line = imgui.get_color_u32((1.0, 0.78, 0.18, 0.92))
            cell_sets = _ensure_scanner_cell_color_sets(state) if scanner_picker_mode else []
            for r in range(rows):
                for c in range(cols):
                    cell_bounds = scanner_cell_bounds[r * cols + c]
                    if cell_bounds is None:
                        continue
                    cx0, cy0, cx1, cy1 = cell_bounds
                    selected_cell = r == selected_r and c == selected_c
                    if scanner_picker_mode:
                        batch_color = cell_sets[r * cols + c][0]
                        batch_fill = imgui.get_color_u32((
                            float(batch_color[0]),
                            float(batch_color[1]),
                            float(batch_color[2]),
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
                    dl.add_rect_filled(
                        (ox + cx0, oy + cy0),
                        (ox + cx1, oy + cy1),
                        selected_fill if selected_cell else fill_col,
                    )
                    dl.add_rect(
                        (ox + cx0, oy + cy0),
                        (ox + cx1, oy + cy1),
                        selected_line if selected_cell else grid_col,
                        0.0,
                        2.0 if selected_cell else 1.0,
                        0,
                    )
        if not scanner_picker_mode:
            for i, (hx, hy) in enumerate(bounds_handles(gizmo_bounds)):
                is_hot = (i == hover_handle or i == active_handle)
                fill = imgui.get_color_u32((0.95, 0.65, 0.10, 1.0) if is_hot else (0.96, 0.96, 0.96, 0.95))
                stroke = imgui.get_color_u32((0.12, 0.12, 0.12, 1.0))
                dl.add_circle_filled((ox + hx, oy + hy), handle_radius, fill, 16)
                dl.add_circle((ox + hx, oy + hy), handle_radius, stroke, 16, 1.5)

    # ImGuizmo
    if state.mode == 'spline' and state.selected_idx >= 0 and int(state.selected_idx) in visible_ctrl_index_map:
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
                    state.scanner_status = f"Selected mini-fabric R{cell_r + 1} C{cell_c + 1}"
                    suppress_mesh_click = True
                    break

        # Bounding-box resize handles (window-like scaling in screen space).
        if gizmo_bounds is not None and hover_handle >= 0 and imgui.is_mouse_clicked(imgui.MouseButton_.left) and not io.key_shift and not io.key_alt:
            state.push_undo("Bounding box scale")
            state.bbox_active_handle = int(hover_handle)
            state.bbox_start_bounds = np.array(gizmo_bounds, dtype=np.float32)
            state.bbox_start_mouse = np.array([lx, ly], dtype=np.float32)
            state.bbox_start_t = np.array(state.model_t, dtype=np.float32)
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
        if state.mode == 'spline' and len(visible_ctrl_indices) > 0:
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
