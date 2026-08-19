"""UR5 Robot Mode: the panel that drives the real arm.

Scan Mode and this mode run the same scan plan; the difference is that here the
consequences are physical, and the UI is arranged around that. Connection comes
first and everything else stays disabled until it succeeds, the safety controls
sit above the run controls rather than below them, and starting motion takes a
deliberate second confirmation.

The panel itself only reads and draws. All robot and camera work happens on the
threads inside ``UR5ModeController``, so a move that takes several seconds never
blocks the frame loop -- which matters most for the stop button, since a stop
the user cannot click during a move is not a stop at all.

This module deliberately does not import gui at module scope: gui imports it, so
the plan-building helpers it needs are imported inside the function that needs
them, once gui has finished loading.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from imgui_bundle import imgui
from PIL import Image

import gaussian_splatting
import robot_camera
import ur5_robot
import ur5_scan
from rendering import draw_fitted_texture, upload_rgb_texture


# Colours reused for status text, so "safe/attention/danger" reads the same
# everywhere in the panel.
COLOR_OK = (0.25, 0.80, 0.40, 1.0)
COLOR_WARN = (0.95, 0.75, 0.20, 1.0)
COLOR_DANGER = (0.95, 0.35, 0.30, 1.0)
COLOR_MUTED = (0.65, 0.65, 0.65, 1.0)

PREVIEW_INTERVAL = 0.08


class UR5ModeController:
    """Owns the robot, the camera, the run, and the splatting job."""

    def __init__(self, state, gl_ctx, window=None):
        self.state = state
        self.gl_ctx = gl_ctx
        self.window = window
        self.robot = ur5_robot.UR5Robot(ur5_robot.DryRunTransport())
        self.camera: robot_camera.RobotCamera | None = None
        self.runner: ur5_scan.RealScanRunner | None = None
        self.dataset: gaussian_splatting.SplatDataset | None = None
        self.trainer: gaussian_splatting.SplatTrainer | None = None
        self.status = "UR5 Robot Mode ready"
        self.devices: list[dict] = []
        self.devices_scanned = False
        # Set when Start is pressed; cleared by confirming or cancelling. The
        # arm does not move while this is pending.
        self.pending_start = False
        self.preview_texture = None
        self._preview_size = (0, 0)
        self._last_preview_time = 0.0
        self.last_analysis = None

    # -- camera preview ----------------------------------------------------

    def update_preview(self) -> None:
        """Refreshes the live camera texture, rate limited to the UI's needs."""
        if self.camera is None or not self.camera.running:
            return
        now = time.monotonic()
        if now - self._last_preview_time < PREVIEW_INTERVAL:
            return
        self._last_preview_time = now
        image = self.camera.preview_image()
        if image is None:
            return
        self._upload_preview(image)

    def _upload_preview(self, image: Image.Image) -> None:
        # Shared helper, which also sets the texture filter -- this uploader
        # never did, leaving the preview sampling at moderngl's default.
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        height, width = rgb.shape[:2]
        self.preview_texture = upload_rgb_texture(self.gl_ctx, self.preview_texture, rgb)
        self._preview_size = (width, height)

    # -- camera ------------------------------------------------------------

    def scan_devices(self) -> None:
        self.devices = robot_camera.list_devices()
        self.devices_scanned = True
        real = [d for d in self.devices if d.get("is_real")]
        self.status = (
            f"Found {len(real)} camera(s)" if real
            else "No camera detected; only the synthetic source is available"
        )

    def start_camera(self, spec: dict, size) -> bool:
        self.stop_camera()
        backend = robot_camera.make_backend(spec, size=size)
        self.camera = robot_camera.RobotCamera(backend)
        if not self.camera.start():
            self.status = self.camera.status
            return False
        self.status = self.camera.status
        return True

    def stop_camera(self) -> None:
        if self.camera is not None:
            self.camera.stop()
            self.camera = None
        if self.preview_texture is not None:
            try:
                self.preview_texture.release()
            except Exception:
                pass
            self.preview_texture = None
            self._preview_size = (0, 0)

    # -- robot -------------------------------------------------------------

    def connect(self, ip: str, transport_kind: str) -> bool:
        self.robot = ur5_robot.UR5Robot(ur5_robot.make_transport(transport_kind))
        ok = self.robot.connect(ip)
        self.status = self.robot.status
        return ok

    def disconnect(self) -> None:
        if self.runner is not None and self.runner.running:
            self.runner.stop()
        self.robot.disconnect()
        self.status = self.robot.status

    def close(self) -> None:
        """Leaves nothing running behind when the mode is left or the app quits."""
        if self.runner is not None and self.runner.running:
            self.runner.stop()
            self.runner.wait(timeout=2.0)
        if self.trainer is not None and self.trainer.running:
            self.trainer.stop()
        self.stop_camera()
        self.robot.disconnect()


# ============================================================================
# Plan construction
# ============================================================================

def build_scan_plan(state):
    """Builds the same fabric scan plan Scan Mode uses, for the real arm.

    Both modes go through ``fabric_scanner.build_plan`` with the same settings,
    which is what makes a real dataset comparable to a simulated one: same grid,
    same stations, same camera angles.
    """
    import fabric_scanner as scanner
    import gui  # Late: gui imports this module, so it is only resolvable now.

    pattern_rows, pattern_cols = gui._scanner_pattern_dimensions(state)
    repeat_rows, repeat_cols = gui._scanner_pattern_repeats(state)
    spacing_x, spacing_y = gui._scanner_repeat_spacing(state)
    batch_texture_width, batch_texture_height = gui._scanner_batch_texture_size(state)

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
        center=[float(v) for v in state.get("ur5_workspace_center", [-0.45, -0.08, 0.30])],
        max_span=scanner.DEFAULT_MAX_SPAN.tolist(),
        palette=gui._scanner_base_palette(state),
        cell_color_sets=gui._scanner_shared_cell_color_sets(state),
        model_json=str(state.save_path),
        model_curves=None,
        cell_model_curves=gui._generate_scanner_random_patterns(state),
        random_patterns=True,
        pattern_rows=int(pattern_rows),
        pattern_cols=int(pattern_cols),
        pattern_repeat_rows=int(repeat_rows),
        pattern_repeat_cols=int(repeat_cols),
        pattern_repeat_spacing_x=float(spacing_x),
        pattern_repeat_spacing_y=float(spacing_y),
        batch_texture_width=int(batch_texture_width),
        batch_texture_height=int(batch_texture_height),
        pattern_density=float(state.get("scanner_pattern_density", 0.62)),
        random_seed=int(state.get("scanner_random_seed", 1)),
        scanner_lighting=gui._scanner_lighting_settings(state),
        display_batch_colors=gui._scanner_batch_colors_for_simulator(state),
    )
    return scanner.build_plan(args)


def _make_runner(state, controller):
    """Assembles a run, snapshotting everything it must not read live."""
    import gui

    plan = build_scan_plan(state)
    storage = gui._scanner_storage(state)
    signature = ""
    settings = {}
    if storage is not None:
        # Taken here on the UI thread. The run thread never touches AppState.
        signature = storage.pattern_signature(state)
        settings = storage.scanner_state_snapshot(state)
    estimates = gui._scanner_estimated_cell_colors(state)

    capture_width = int(state.get("ur5_capture_width", 1280))
    capture_height = int(round(capture_width * 3 / 4))
    return ur5_scan.RealScanRunner(
        controller.robot,
        controller.camera,
        plan,
        storage,
        output_root=Path(state.project_root) / "robot_scans",
        pattern_signature=signature,
        settings_snapshot={
            **settings,
            "ur5_robot_ip": str(state.get("ur5_robot_ip", "")),
            "ur5_speed_limit": float(state.get("ur5_speed_limit", 0.25)),
            "ur5_capture_size": [capture_width, capture_height],
        },
        estimated_cell_colors=estimates,
        lighting_condition=str(state.get("ur5_lighting_condition", "")),
        dwell=float(state.get("ur5_dwell", 0.2)),
        save_images=bool(state.get("ur5_save_images", True)),
    )


# ============================================================================
# Panel
# ============================================================================

def draw_ur5_panel(state, renderer, window=None):
    """The whole UR5 Robot Mode sidebar."""
    controller = state.get("ur5_controller")
    if controller is None:
        controller = UR5ModeController(state, renderer.ctx, window)
        state.ur5_controller = controller

    controller.update_preview()

    imgui.text_colored((0.92, 0.74, 0.34, 1.0), "UR5 Robot Mode - real hardware")
    imgui.text_wrapped(
        "Runs the scan workflow on the physical UR5. Simulation Scan Mode is "
        "unaffected and still available."
    )
    imgui.separator()

    _draw_connection_section(state, controller)
    _draw_safety_section(state, controller)
    _draw_camera_section(state, controller)
    _draw_scan_section(state, controller)
    _draw_analysis_section(state, controller)
    _draw_splatting_section(state, controller)

    imgui.separator()
    imgui.text_wrapped(controller.status)


def _draw_connection_section(state, controller):
    if not imgui.collapsing_header("1. Robot connection", imgui.TreeNodeFlags_.default_open):
        return
    robot = controller.robot
    connected = robot.connected

    changed_ip, ip = imgui.input_text("Robot IP##ur5_ip", str(state.get("ur5_robot_ip", "")), 64)
    if changed_ip:
        state.ur5_robot_ip = ip

    use_real = str(state.get("ur5_transport", "rtde")) == "rtde"
    rtde_ok = ur5_robot.rtde_available()
    if not rtde_ok:
        imgui.text_colored(COLOR_WARN, "ur_rtde is not installed; only dry run is available")
        use_real = False
        state.ur5_transport = "dry_run"
    if rtde_ok:
        changed_transport, use_real = imgui.checkbox("Connect to the real robot (RTDE)##ur5_transport", use_real)
        if changed_transport:
            state.ur5_transport = "rtde" if use_real else "dry_run"
    if not use_real:
        imgui.text_colored(COLOR_WARN, "Dry run: no hardware is commanded")
        imgui.text_wrapped(
            "Captures made in dry run are stored, and marked as not real, so "
            "they stay distinguishable from a genuine scan."
        )

    if connected:
        if imgui.button("Disconnect##ur5_disconnect", (-1, 0)):
            controller.disconnect()
    else:
        if imgui.button("Connect##ur5_connect", (-1, 0)):
            controller.connect(str(state.get("ur5_robot_ip", "")), str(state.get("ur5_transport", "rtde")))

    state_now = robot.refresh()
    imgui.text("Status:")
    imgui.same_line()
    if connected:
        imgui.text_colored(COLOR_OK if robot.ready() else COLOR_WARN, state_now.summary())
    else:
        imgui.text_colored(COLOR_MUTED, "Disconnected")
    if state_now.error:
        imgui.text_colored(COLOR_DANGER, f"Error: {state_now.error}")

    if connected:
        pose = state_now.tcp_pose
        if len(pose) >= 6:
            imgui.text(f"TCP  x {pose[0]:+.3f}  y {pose[1]:+.3f}  z {pose[2]:+.3f} m")
        imgui.text(f"Transport: {robot.describe()}")

    ready, issues = robot.readiness()
    if imgui.button("Verify robot is ready##ur5_verify", (-1, 0)):
        controller.status = (
            "Robot is ready to move" if ready else "Robot is not ready: " + "; ".join(issues)
        )
    if connected and not ready:
        for issue in issues:
            imgui.text_colored(COLOR_WARN, f"- {issue}")
    elif connected:
        imgui.text_colored(COLOR_OK, "Ready to move")


def _draw_safety_section(state, controller):
    if not imgui.collapsing_header("2. Safety", imgui.TreeNodeFlags_.default_open):
        return
    robot = controller.robot
    runner = controller.runner

    # The stop is always enabled, even when disconnected. A control that is
    # sometimes greyed out is one the user has to think about before pressing,
    # and this is the one control that must never require thought.
    imgui.push_style_color(imgui.Col_.button, (0.70, 0.12, 0.12, 1.0))
    imgui.push_style_color(imgui.Col_.button_hovered, (0.85, 0.18, 0.18, 1.0))
    if imgui.button("EMERGENCY STOP##ur5_estop", (-1, 42)):
        if runner is not None:
            runner.emergency_stop()
        else:
            robot.emergency_stop()
        controller.pending_start = False
        controller.status = robot.status
    imgui.pop_style_color(2)
    imgui.text_colored(COLOR_MUTED, "Software protective stop - not a substitute")
    imgui.text_colored(COLOR_MUTED, "for the hardware emergency stop button.")

    half = max(90.0, (imgui.get_content_region_avail().x - imgui.get_style().item_spacing.x) * 0.5)
    paused = bool(robot.paused)
    if imgui.button(("Continue" if paused else "Pause") + "##ur5_pause", (half, 0)):
        if paused:
            controller.status = robot.status if not robot.resume() else "Continuing"
        else:
            if runner is not None and runner.running:
                runner.pause()
            else:
                robot.pause()
            controller.status = robot.status
    imgui.same_line()
    if imgui.button("Reset##ur5_reset", (half, 0)):
        robot.reset()
        controller.status = robot.status

    if robot.stop_requested:
        imgui.text_colored(COLOR_DANGER, "A stop is latched. Press Reset to clear it.")

    changed_speed, speed = imgui.slider_float(
        "Speed limit##ur5_speed", float(state.get("ur5_speed_limit", 0.25)), 0.01, 1.0, "%.2f"
    )
    if changed_speed:
        state.ur5_speed_limit = float(speed)
        robot.apply_speed_limit(float(speed))
    imgui.text_colored(
        COLOR_MUTED,
        f"Commanded {robot.limited_speed():.3f} m/s, {robot.limited_acceleration():.2f} m/s^2",
    )

    if imgui.button("Move to safe home position##ur5_home", (-1, 0)):
        controller.status = robot.status if not robot.move_home() else "Moving to safe home position"


def _draw_camera_section(state, controller):
    if not imgui.collapsing_header("3. Camera", imgui.TreeNodeFlags_.default_open):
        return
    if not controller.devices_scanned:
        controller.scan_devices()

    if imgui.button("Rescan cameras##ur5_camera_scan", (-1, 0)):
        controller.scan_devices()
    if not robot_camera.opencv_available():
        imgui.text_colored(COLOR_WARN, "OpenCV is not installed; real cameras cannot be opened")
        imgui.text_wrapped("Install opencv-python to use the gripper camera.")

    labels = [str(device["label"]) for device in controller.devices] or ["(none)"]
    selected = int(np.clip(int(state.get("ur5_camera_device", 0)), 0, len(labels) - 1))
    changed_device, selected = imgui.combo("Camera##ur5_camera_device", selected, labels)
    if changed_device:
        state.ur5_camera_device = int(selected)

    changed_width, width = imgui.slider_int(
        "Capture width##ur5_capture_width", int(state.get("ur5_capture_width", 1280)), 320, 3840
    )
    if changed_width:
        state.ur5_capture_width = int(np.clip(int(round(width / 32.0) * 32), 320, 3840))

    running = controller.camera is not None and controller.camera.running
    if running:
        if imgui.button("Stop camera##ur5_camera_stop", (-1, 0)):
            controller.stop_camera()
    else:
        if imgui.button("Start live preview##ur5_camera_start", (-1, 0)):
            spec = controller.devices[selected] if selected < len(controller.devices) else {}
            capture_width = int(state.get("ur5_capture_width", 1280))
            controller.start_camera(spec, (capture_width, int(round(capture_width * 3 / 4))))

    if running:
        camera = controller.camera
        imgui.text_colored(COLOR_OK if camera.is_real else COLOR_WARN, camera.describe())
        if not camera.is_real:
            imgui.text_colored(COLOR_WARN, "Synthetic frames - not a real camera")
        if controller.preview_texture is not None:
            avail_w = max(120, int(imgui.get_content_region_avail().x))
            width, height = controller._preview_size
            preview_h = int(min(avail_w * height / max(width, 1), 320))
            draw_fitted_texture(
                controller.preview_texture.glo, width, height, avail_w, preview_h, flip_y=True
            )
        imgui.text_colored(COLOR_MUTED, "Saved captures are this exact frame at full resolution.")

    changed_light, lighting = imgui.input_text(
        "Lighting condition##ur5_lighting", str(state.get("ur5_lighting_condition", "")), 128
    )
    if changed_light:
        state.ur5_lighting_condition = lighting


def _draw_scan_section(state, controller):
    if not imgui.collapsing_header("4. Scan workflow", imgui.TreeNodeFlags_.default_open):
        return
    robot = controller.robot
    runner = controller.runner

    imgui.text(f"Grid: {int(state.scanner_rows)} x {int(state.scanner_cols)} cells, "
               f"{int(state.scanner_angles)} angles")
    imgui.text_colored(COLOR_MUTED, "Set the grid and angles in Scan Mode; both modes share them.")

    changed_dwell, dwell = imgui.slider_float(
        "Dwell per view (s)##ur5_dwell", float(state.get("ur5_dwell", 0.2)), 0.0, 3.0, "%.2f"
    )
    if changed_dwell:
        state.ur5_dwell = float(dwell)
    changed_save, save_images = imgui.checkbox(
        "Save captured images##ur5_save", bool(state.get("ur5_save_images", True))
    )
    if changed_save:
        state.ur5_save_images = bool(save_images)

    scanning = runner is not None and runner.running
    camera_running = controller.camera is not None and controller.camera.running

    if scanning:
        progress = runner.progress
        imgui.text_colored(COLOR_OK, f"Scanning {progress.target_index + 1}/{progress.target_count} "
                                     f"({progress.percent:.0f}%)")
        imgui.progress_bar(progress.percent / 100.0, (-1, 0))
        imgui.text(f"Captured {progress.captured} images")
        imgui.text_wrapped(progress.status)
        if imgui.button("Stop scan##ur5_scan_stop", (-1, 0)):
            runner.stop()
        return

    if controller.pending_start:
        # The deliberate second step before any motion. Phrased as what is about
        # to physically happen, not as a generic "are you sure".
        imgui.separator()
        imgui.text_colored(COLOR_DANGER, "The robot arm is about to move.")
        imgui.text_wrapped(
            "Check that the workspace is clear, the fabric is mounted, and you "
            "can reach the hardware emergency stop."
        )
        half = max(90.0, (imgui.get_content_region_avail().x - imgui.get_style().item_spacing.x) * 0.5)
        imgui.push_style_color(imgui.Col_.button, (0.20, 0.55, 0.25, 1.0))
        if imgui.button("Confirm and start##ur5_scan_confirm", (half, 0)):
            controller.pending_start = False
            _start_scan(state, controller)
        imgui.pop_style_color()
        imgui.same_line()
        if imgui.button("Cancel##ur5_scan_cancel", (half, 0)):
            controller.pending_start = False
            controller.status = "Scan cancelled"
        imgui.separator()
        return

    can_start = robot.ready() and camera_running
    if not can_start:
        imgui.begin_disabled()
    if imgui.button("Start real scan...##ur5_scan_start", (-1, 0)):
        controller.pending_start = True
    if not can_start:
        imgui.end_disabled()
        if not robot.connected:
            imgui.text_colored(COLOR_MUTED, "Connect to the robot first.")
        elif not robot.ready():
            imgui.text_colored(COLOR_MUTED, "The robot is not ready; see the connection section.")
        elif not camera_running:
            imgui.text_colored(COLOR_MUTED, "Start the camera preview first.")

    if runner is not None:
        progress = runner.progress
        if progress.finished or progress.captured:
            imgui.text_wrapped(progress.status)
            if progress.error:
                imgui.text_colored(COLOR_DANGER, progress.error)
            if progress.output_dir:
                imgui.text_colored(COLOR_MUTED, f"Saved to {Path(progress.output_dir).name}")


def _start_scan(state, controller):
    try:
        controller.runner = _make_runner(state, controller)
    except Exception as exc:
        controller.status = f"Could not build the scan plan: {exc}"
        return
    if controller.runner.start():
        controller.status = "Real UR5 scan running"
    else:
        controller.status = controller.runner.progress.status


def _draw_analysis_section(state, controller):
    if not imgui.collapsing_header("5. RGB analysis", imgui.TreeNodeFlags_.default_open):
        return
    runner = controller.runner
    if runner is None or not runner.capture_records:
        imgui.text_disabled("Run a scan to measure the captured fabric colours.")
        return

    imgui.text(f"{len(runner.capture_records)} captures available")
    if runner.running:
        imgui.text_disabled("Analysis runs when the scan finishes.")
        return
    if imgui.button("Analyze captured images##ur5_analyze", (-1, 0)):
        try:
            controller.last_analysis = runner.analyze()
            controller.status = f"Analyzed {controller.last_analysis['image_count']} images"
        except Exception as exc:
            controller.status = f"Analysis failed: {exc}"

    result = controller.last_analysis or runner.analysis_results
    if not result or not result.get("cells"):
        return
    imgui.text(f"{len(result['cells'])} fabric cells measured")
    for cell in result["cells"][:12]:
        rgb = [float(v) for v in cell["overall_rgb"]]
        imgui.color_button(
            f"##ur5_swatch_{cell['row']}_{cell['col']}",
            (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, 1.0),
            0,
            (18, 18),
        )
        imgui.same_line()
        imgui.text(
            f"row {int(cell['row']) + 1}, col {int(cell['col']) + 1}: "
            f"rgb({rgb[0]:.0f}, {rgb[1]:.0f}, {rgb[2]:.0f}) - {int(cell['count'])} views"
        )
    comparisons = result.get("comparisons", [])
    if comparisons and imgui.tree_node("Colour differences##ur5_compare"):
        for item in comparisons[:10]:
            imgui.text(f"{item['a']} vs {item['b']}: delta {item['delta_rgb']:.1f}")
        imgui.tree_pop()
    if result.get("json_path"):
        imgui.text_colored(COLOR_MUTED, f"Saved {Path(result['json_path']).name}")


def _draw_splatting_section(state, controller):
    if not imgui.collapsing_header("6. Gaussian Splatting (optional)"):
        return
    runner = controller.runner
    imgui.text_wrapped(
        "Exports the captured images with the camera pose the robot recorded "
        "for each one, so no structure-from-motion step is needed."
    )

    changed_fov, fov = imgui.slider_float(
        "Camera field of view##ur5_splat_fov",
        float(state.get("ur5_splat_fov", gaussian_splatting.DEFAULT_FOV_DEGREES)), 10.0, 140.0, "%.1f",
    )
    if changed_fov:
        state.ur5_splat_fov = float(fov)
    changed_offset, offset = imgui.slider_float(
        "Camera offset from tool (m)##ur5_splat_offset",
        float(state.get("ur5_splat_camera_offset", gaussian_splatting.DEFAULT_CAMERA_OFFSET_M)),
        0.0, 0.4, "%.3f",
    )
    if changed_offset:
        state.ur5_splat_camera_offset = float(offset)
    imgui.text_colored(COLOR_WARN, "Both are estimates unless the camera is calibrated.")

    has_captures = runner is not None and bool(runner.capture_records)
    if not has_captures:
        imgui.begin_disabled()
    if imgui.button("Export splatting dataset##ur5_splat_export", (-1, 0)):
        try:
            controller.dataset = gaussian_splatting.export_dataset(
                runner.capture_records,
                Path(state.project_root) / "robot_scans" / runner.session_id / "splat_dataset",
                session_id=runner.session_id,
                hand_eye=gaussian_splatting.hand_eye_matrix(
                    float(state.get("ur5_splat_camera_offset", gaussian_splatting.DEFAULT_CAMERA_OFFSET_M))
                ),
                fov_degrees=float(state.get("ur5_splat_fov", gaussian_splatting.DEFAULT_FOV_DEGREES)),
            )
            controller.status = f"Exported {controller.dataset.image_count} images for splatting"
        except Exception as exc:
            controller.status = f"Dataset export failed: {exc}"
    if not has_captures:
        imgui.end_disabled()
        imgui.text_colored(COLOR_MUTED, "Run a scan first.")

    dataset = controller.dataset
    if dataset is not None:
        imgui.text(f"Dataset: {dataset.image_count} images")
        imgui.text_colored(COLOR_MUTED, str(dataset.root))
        for warning in dataset.warnings[:4]:
            imgui.text_colored(COLOR_WARN, f"- {warning}")

    changed_cmd, command = imgui.input_text(
        "Trainer command##ur5_splat_cmd", str(state.get("ur5_splat_command", "")), 512
    )
    if changed_cmd:
        state.ur5_splat_command = command
    imgui.text_colored(COLOR_MUTED, "Use {dataset} and {output} as placeholders.")

    trainer = controller.trainer
    if trainer is not None and trainer.running:
        imgui.text_colored(COLOR_OK, "Training running")
        if imgui.button("Stop training##ur5_splat_stop", (-1, 0)):
            trainer.stop()
    else:
        can_train = dataset is not None and dataset.image_count > 0 and str(state.get("ur5_splat_command", "")).strip()
        if not can_train:
            imgui.begin_disabled()
        if imgui.button("Run reconstruction##ur5_splat_run", (-1, 0)):
            output_dir = Path(dataset.root).parent / "splat_output"
            controller.trainer = gaussian_splatting.SplatTrainer(
                str(state.get("ur5_splat_command", "")), dataset, output_dir
            )
            if controller.trainer.start():
                controller.status = "Gaussian Splatting training started"
            else:
                controller.status = controller.trainer.status
        if not can_train:
            imgui.end_disabled()

    if trainer is not None:
        imgui.text_wrapped(trainer.status)
        for line in trainer.tail(6):
            imgui.text_colored(COLOR_MUTED, line[:110])
        artifacts = trainer.output_artifacts()
        if artifacts:
            imgui.text_colored(COLOR_OK, f"Reconstruction: {artifacts[0].name}")
            if imgui.button("Record result in the database##ur5_splat_record", (-1, 0)):
                _record_splat_output(state, controller, artifacts[0])


def _record_splat_output(state, controller, artifact: Path):
    import gui

    storage = gui._scanner_storage(state)
    runner = controller.runner
    if storage is None or runner is None or not runner.session_id:
        controller.status = "No scan session to attach the reconstruction to"
        return
    try:
        storage.set_session_splat_output(runner.session_id, str(artifact))
        storage.flush_json_index()
        controller.status = f"Recorded {artifact.name} against session {runner.session_id}"
    except Exception as exc:
        controller.status = f"Could not record the reconstruction: {exc}"
