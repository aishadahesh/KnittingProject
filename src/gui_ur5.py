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

The scan helpers this panel needs come from scanner_core. They used to live in
gui.py, which imports this module, so reaching them meant importing gui from
inside three functions to dodge the cycle -- that is gone.
"""

from __future__ import annotations

import logging
import json
import re
import threading
import time
from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np
from imgui_bundle import imgui
from PIL import Image

import charuco
import fabric_detector
import free_capture
import gaussian_splatting
import gui_captures
import hand_eye
import paths
from persistence import write_json
import robot_camera
import scan_geometry
import scanner_core
import teleop_transport
import ur5_robot
import ur5_scan
from scan_geometry import clamp_capture_size
from rendering import draw_fitted_texture, upload_rgb_texture


# Colours reused for status text, so "safe/attention/danger" reads the same
# everywhere in the panel.
COLOR_OK = (0.25, 0.80, 0.40, 1.0)
COLOR_WARN = (0.95, 0.75, 0.20, 1.0)
COLOR_DANGER = (0.95, 0.35, 0.30, 1.0)
COLOR_MUTED = (0.65, 0.65, 0.65, 1.0)

PREVIEW_INTERVAL = 0.08
# The preview is shown at up to this many pixels across - the frame's own resolution for a 720p
# camera, so focus can be judged from it - while the fabric detection drawn over it runs on a
# small copy, which is all a texture test needs and keeps the refresh cheap.
PREVIEW_MAX_WIDTH = 1280
DETECTION_WIDTH = 480
# The live sharpness readout, for setting focus by hand, is refreshed this often.
SHARPNESS_INTERVAL = 0.5
# Find best focus: the camera's own autofocus hunts or gives up on a close, repeating knit, so the
# panel sweeps the lens itself - a coarse pass over the driver's whole range, then a fine pass
# around the sharpest - judging each position by the sharpness a run is judged by. A lens needs a
# moment to travel and a frame grabbed after it settles; a sweep whose sharpness barely changes
# means the driver did not move the lens at all.
FOCUS_RANGE = (0.0, 255.0)
FOCUS_COARSE_STEP = 15.0
FOCUS_FINE_STEP = 3.0
FOCUS_SETTLE = 0.4
FOCUS_FLAT = 0.05

# The one-click trainer: nerfstudio's splatfacto in its own venv (scripts/setup_splatting.ps1),
# run through scripts/train_splat.ps1, which also exports the result as a PLY the viewer reads.
SPLAT_VENV = paths.PROJECT_ROOT / ".venv-splat"
EXTRA_SESSIONS: set[str] = set()
SPLATFACTO_COMMAND = ('powershell -ExecutionPolicy Bypass -File "' + str(paths.SCRIPTS_DIR / "train_splat.ps1")
                      + '" -Dataset "{dataset}" -Output "{output}"')

# Image-turn calibration: probes this far along x and y, from this high over the fabric. Small
# enough that a repeating knit cannot alias the slide, large enough to measure to a fraction of a
# pixel; a result further than CALIBRATION_MAX_ERROR from a clean rotation is refused.
CALIBRATION_STEP = 0.0015
CALIBRATION_HEIGHT = 0.10
CALIBRATION_MAX_ERROR = 0.15
# Measuring the fabric height by parallax: the slide, well under half a knit stitch so the repeating
# pattern cannot alias it, and how far apart the x and y probes may read before the measure is refused.
MEASURE_STEP = 0.003
MEASURE_AGREEMENT = 0.2

# Camera-guided centring. It looks from CENTRING_HEIGHT, high enough for a frame to take in a good
# part of the cloth, probes PROBE_STEP along each axis to learn which way the image moves - which is
# what lets it work on any camera mount - then steps the fabric's middle to the frame's centre until
# it is within CENTRE_TOLERANCE of it, or the frame is all cloth, which is as centred as it can see.
CENTRING_HEIGHT = 0.25
PROBE_STEP = 0.03
MAX_CENTRING_STEP = 0.08
CENTRE_TOLERANCE = 0.03
FULL_FRAME_SHARE = 0.97
MAX_CENTRING_ITERATIONS = 4

# Home looks down at the fabric from well above it, so a fabric "measured" closer to Home's
# camera height than this was measured with the lens in the air - most likely at Home itself.
HOME_TO_FABRIC_MIN = 0.05

# The capture height (Z) the panel offers, and the fine nudge of the setting.
# Every move is still collision checked; the low end sits just above the clearance the check
# demands, so the lowest height offered is one the check can grant.
CAPTURE_HEIGHT_MIN = scan_geometry.FABRIC_CLEARANCE + 0.005
CAPTURE_HEIGHT_MAX = 0.50
OVERVIEW_HEIGHT_MAX = 1.00
CAPTURE_HEIGHT_NUDGE = 0.005

# The Position tab's jog pad: the steps one press may move the camera, and the pad itself as rows
# of (label, right, forward, up) in units of the chosen step, right and forward as seen in the
# preview. An empty label is a gap that keeps the arrows in their places.
JOG_STEPS = (0.001, 0.005, 0.01, 0.02, 0.05)
JOG_PAD = ((("",         0, 0, 0), ("Forward",  0,  1, 0), ("",      0, 0, 0), ("Raise", 0, 0,  1)),
           (("Left",    -1, 0, 0), ("",         0,  0, 0), ("Right", 1, 0, 0), ("",      0, 0,  0)),
           (("",         0, 0, 0), ("Back",     0, -1, 0), ("",      0, 0, 0), ("Lower", 0, 0, -1)))

# What the capture height means to each pattern; a pattern absent here does not read it.
HEIGHT_USE = {
    "grid":    "Every grid frame is taken straight down from this height.",
    "sweep":   "The whole sweep is flown straight down at this height.",
    "centred": "The straight-down view is taken from this height; the tilted rings keep this distance to the fabric's middle.",
    "splat":   "The dome's nearest views keep this distance to the object under the camera; its first view is the camera's own position.",
}

# The three ways this panel can reach an arm. Only one may be connected: a UR controller
# grants a single RTDE control session, and TeleopApp holds it while it runs.
#
# The table is here rather than in ur5_robot because that module cannot see TeleopApp's
# transport -- the transport is built on it -- and a factory that reached back for it made
# the two modules import each other.
Transport = namedtuple("Transport", "label explain build available")

#
# There is no dry run. Simulating is TeleopApp's: its 3D UR5e moves through the checked path
# while the arm stays still, which shows the motion a run will make - a stand-in that only
# reported arriving showed nothing, and planned from a pose the arm was never in.
TRANSPORTS = {
    "sim":     Transport("Simulated UR5e", "In-process 3D UR5e in the shared world. Moves smoothly at the configured joint speed.",
                         ur5_robot.SimUR5Transport, lambda: True),
    "teleop":  Transport("Via TeleopApp", "TeleopApp holds the controller and this app asks it to move. "
                                          "To simulate, leave TeleopApp not following the arm (or on its emulator), "
                                          "then Check and Rehearse: its 3D arm flies the path and the real one stays still.",
                         teleop_transport.TeleopAppTransport, teleop_transport.teleop_available),
    "rtde":    Transport("Direct RTDE",   "This app holds the controller and every move is the arm's. TeleopApp must not be running.",
                         ur5_robot.RTDETransport, ur5_robot.rtde_available),
}

# Which capture-path fields mean anything for which pattern. The pattern combo used to
# leave every field on screen whether it applied or not, which is how a subject distance
# came to be edited for a motion that never reads it.
PATTERN_FIELDS = {
    "centred": ("teleop_angles", "teleop_rings", "teleop_max_tilt"),
    "grid":   ("teleop_grid_rows", "teleop_grid_cols", "teleop_grid_overlap", "teleop_grid_views", "teleop_max_tilt", "teleop_grid_passes"),
    "sweep":  ("teleop_grid_rows", "teleop_grid_cols", "teleop_sweep_overlap"),
    "taught": (),
    "orbit":  ("teleop_orbit_radius", "teleop_angles", "teleop_arc", "teleop_elevation", "teleop_rings", "teleop_max_rise"),
    "splat":  ("teleop_splat_layers", "teleop_splat_views", "teleop_rings", "teleop_max_tilt"),
    "spin":   ("teleop_angles",),
    "tilt":   ("teleop_angles", "teleop_view_radius"),
}

# The grid's hand-set block, which an outline replaces: with the fabric marked, the rows and
# columns are whatever fills it. The spacing is never hand-set - it is one frame less the overlap.
GRID_BLOCK_FIELDS = ("teleop_grid_rows", "teleop_grid_cols")

PATTERN_BLURBS = {
    "centred": "Stays over the fabric's middle and tilts around it, at the capture height: straight down first, then rings tilted "
               "further, each direction only as far as the whole frame stays on the cloth. Needs the fabric "
               "corners (2 Position > Fabric) to know where the cloth ends.",
    "grid":   "Automatic close-up: at each station one view straight down from the capture height, then the rest of "
              "Views per position tilted by Max tilt, spread around and all aimed at the station's centre. "
              "Height passes repeats the whole grid higher each time (1.6x the capture height per pass). "
              "With 3+ fabric corners marked it "
              "fills the outline and keeps every frame on the cloth; without, it is a rows x cols block "
              "centred where the camera looks now.",
    "sweep":  "For Gaussian Splatting: the grid, straight down from the capture height, with stations only Overlap of a frame "
              "apart in both directions - a still at every one, each with the camera pose the arm was planned to, so the "
              "splat can be trained from the arm's own poses (4 Results) without matching pictures, which a plain knit "
              "defeats. Hundreds of stations: allow the time.",
    "taught": "Flies the views recorded below, in the recorded joints, in order.",
    "orbit":  "Circles the spot the camera looks at on the fabric, at Orbit radius, aiming at it from each "
              "viewpoint. For the closest images use grid instead.",
    "splat":  "For Gaussian Splatting: a dome of views all aimed at the spot under the camera now - put the object under the "
              "crosshair first. Straight down from the capture height (Use the camera's height now keeps the camera where it "
              "is), then full rings tilted up to Max tilt, repeated at each distance - the nearest at the capture height and "
              "each further one half of it more; Distances 1 stays at that height. Views may take in the table, which a "
              "splat needs. Train it under 4 Results > Gaussian Splatting.",
    "spin":   "Holds the camera still and turns it about its own axis.",
    "tilt":   "Holds one position and takes the scan angles from within a narrow cone.",
}

# label, low, high, format. The low value's type decides whether it is an int or a float
# control, so a row here is the whole definition of a field.
FIELD_SPECS = {
    "teleop_angles":            ("Angles",                 1,     24,   "%d"),
    "teleop_arc":               ("Arc (deg)",             30.0,  360.0, "%.0f"),
    "teleop_elevation":         ("Elevation (deg)",       20.0,   80.0, "%.0f"),
    "teleop_rings":             ("Rings",                  1,      4,   "%d"),
    "teleop_max_rise":          ("Max camera rise (m)",    0.02,   0.40, "%.3f"),
    "teleop_view_radius":       ("View radius (m)",        0.01,   0.20, "%.3f"),
    "teleop_grid_rows":         ("Grid rows",              1,      8,   "%d"),
    "teleop_grid_cols":         ("Grid columns",           1,      8,   "%d"),
    "teleop_grid_overlap":      ("Frame overlap",          0.0,    0.8,  "%.2f"),
    "teleop_sweep_overlap":     ("Overlap between frames", 0.4,    0.9,  "%.2f"),
    "teleop_grid_views":        ("Views per position",     1,      13,   "%d"),
    "teleop_grid_passes":       ("Height passes",          1,      3,    "%d"),
    "ur5_splat_fov":            ("Lens horizontal FOV (deg)", 20.0, 120.0, "%.0f"),
    "free_auto_interval":       ("Seconds between pictures", 0.5,   5.0,  "%.1f"),
    "ur5_min_sharpness":        ("Retake below sharpness", 0.0, 1000.0, "%.0f"),
    "ur5_min_coverage":         ("Fabric at least",        0.3,    1.0,  "%.2f"),
    "teleop_subject_distance":  ("Home to fabric, if not set (m)", 0.08, 0.40, "%.3f"),
    "teleop_orbit_radius":      ("Orbit radius (m)",       0.05,   0.30, "%.3f"),
    "teleop_splat_views":       ("Views per ring",         4,     24,   "%d"),
    "teleop_splat_layers":      ("Distances",              1,      3,   "%d"),
    "teleop_max_tilt":          ("Max tilt from vertical (deg)", 5.0, 60.0, "%.0f"),
    "teleop_tool_offset":       ("Tool offset (m)",        0.0,    0.30, "%.4f"),
    "teleop_move_seconds":      ("Seconds per move, min",  1.0,   30.0, "%.1f"),
    "teleop_speed_deg_s":       ("Joint speed (deg/s)",    2.0,   45.0, "%.0f"),
}


class UR5ModeController:
    """Owns the robot, the camera, the run, and the splatting job."""

    def __init__(self, state, gl_ctx, window=None):
        self.state = state
        self.gl_ctx = gl_ctx
        self.window = window
        self.robot = ur5_robot.UR5Robot(teleop_transport.TeleopAppTransport())
        self.camera: robot_camera.RobotCamera | None = None
        self.runner: ur5_scan.RealScanRunner | None = None
        self.dataset: gaussian_splatting.SplatDataset | None = None
        self.trainer: gaussian_splatting.SplatTrainer | None = None
        self.free: free_capture.FreeCaptureSession | None = None
        self.robot_solver: gaussian_splatting.SplatTrainer | None = None
        self.robot_solved: dict | None = None
        # The camera-mount (hand-eye) calibration fitted from a scan, running or done, and its outcome.
        self.hand_eye_fit: threading.Thread | None = None
        self.hand_eye_result = ""
        # A dataset made under Advanced (COLMAP or the planner's poses), trained apart from the guided flow's.
        self.colmap_dataset: gaussian_splatting.SplatDataset | None = None
        self.video_extract: threading.Thread | None = None
        self.video_extract_result = ""
        self.status = "UR5 Robot Mode ready"
        self.devices: list[dict] = [{
            "backend": "sim",
            "index": -1,
            "label": "Simulated wrist camera (3D world)",
            "is_real": False,
        }]
        # Names which run was asked for and is awaiting confirmation - "scan", "capture",
        # or "" for none. The arm does not move while one is pending.
        self.pending_start = ""
        self.plan: scan_geometry.CapturePlan | None = None
        self.plan_error = ""
        self.rehearse_index = -1
        self.rehearse_thread: threading.Thread | None = None
        self.plan_thread: threading.Thread | None = None
        self._check_lock = threading.Lock()
        self._check_again = False
        self._rehearse_stop = threading.Event()
        self.preview_texture = None
        self._preview_size = (0, 0)
        self._last_preview_time = 0.0
        self._last_sharpness_time = 0.0
        self._previewed_index = -1
        self.centring_thread: threading.Thread | None = None
        self._centring_stop = threading.Event()
        self.live_sharpness: float | None = None
        # The ruler reading being typed into the panel, lens to cloth - not a setting until Set.
        self.fabric_gap_cm: float | None = None
        self.live_fabric_share: float | None = None
        self.last_analysis = None
        # The program found running at the last connect, which the panel offers to stop.
        self.blocking_program: ur5_robot.ProgramAlreadyRunning | None = None

    # -- camera preview ----------------------------------------------------

    def _sim_camera_pose_provider(self):
        joints = self.robot.refresh().joints
        tool_offset = float(self.state.get("teleop_tool_offset", 0.0306))
        pos, rotvec = scan_geometry.camera_pose(joints, tool_offset)
        pose = [*pos, *rotvec]
        fov = float(self.state.get("ur5_splat_fov", gaussian_splatting.DEFAULT_FOV_DEGREES))
        fz = _settings_fabric_z(self.state, None)
        return pose, fov, (-0.45, -0.08, float(fz))

    def _sim_camera_scene_provider(self) -> dict[str, Any]:
        robot_state = self.robot.state() if callable(self.robot.state) else self.robot.refresh()
        raw_joints = getattr(robot_state, "joints", None)
        if raw_joints and (getattr(robot_state, "connected", False) or any(float(v) != 0.0 for v in raw_joints)):
            joints = [float(v) for v in raw_joints]
        else:
            joints = [float(v) for v in (self.state.get("ur5_home_joints") or ur5_robot.DEFAULT_HOME_JOINTS)]
        grid_span = (
            float(self.state.canonical_grid_span())
            if hasattr(self.state, "canonical_grid_span")
            else 80.0
        )
        fabric_center = (-0.45, -0.08, 0.0)
        fabric_size = (
            float(self.state.get("scanner_fabric_width", 0.28)),
            float(self.state.get("scanner_fabric_length", 0.19)),
        )
        mesh_epoch = int(getattr(self.state, "mesh_epoch", 0))
        cached = getattr(self.state, "last_prepared_meshes", (None, None))
        prepared_meshes, colors = (
            cached if isinstance(cached, tuple) and len(cached) == 2 else (None, None)
        )
        material_uniforms = (
            scanner_core._scanner_material_uniforms(self.state)
            if hasattr(self.state, "get_material_uniforms")
            else {}
        )
        sim_renderer = (
            getattr(self.state, "renderer", None)
            if threading.current_thread() is threading.main_thread()
            else None
        )
        tile_images, tiles_epoch = (
            scanner_core.ensure_swatch_unit_tile_images(self.state, sim_renderer)
            if hasattr(scanner_core, "ensure_swatch_unit_tile_images")
            else ([], 0)
        )
        swatch_slots = (
            self.state.all_swatch_slot_rects()
            if hasattr(self.state, "all_swatch_slot_rects")
            else []
        )
        swatch_repeats = (
            scanner_core._scanner_pattern_repeats(self.state)
            if hasattr(scanner_core, "_scanner_pattern_repeats")
            else (
                self.state.swatch_repeat_counts()
                if hasattr(self.state, "swatch_repeat_counts")
                else (24, 24)
            )
        )
        macro_keys = {
            "swatch_tile_images": tile_images,
            "swatch_tiles_epoch": tiles_epoch,
            "swatch_slots": swatch_slots,
            "swatch_repeats": swatch_repeats,
            "hide_unit_cell_mesh": bool(tile_images and swatch_slots),
        }
        material_uniforms = {**material_uniforms, **macro_keys}
        return {
            "joints": joints,
            "grid_span": grid_span,
            "fabric_center": fabric_center,
            "fabric_size": fabric_size,
            "mesh_epoch": mesh_epoch,
            "prepared_meshes": prepared_meshes,
            "colors": colors,
            "material_uniforms": material_uniforms,
            **macro_keys,
        }

    def connect_sim(self, connect_robot: bool = True, start_cam: bool = True) -> None:
        if connect_robot:
            self.state.ur5_transport = "sim"
            self.connect("sim", "sim")
        if start_cam:
            spec = {"backend": "sim", "index": -1, "label": "Simulated wrist camera (3D world)", "is_real": False}
            self.start_camera(spec, clamp_capture_size(self.state.get("ur5_capture_width", 1280)))

    def update_preview(self, renderer=None) -> None:
        """Refreshes the live camera texture, rate limited to the UI's needs."""
        if (self.camera is None or not self.camera.running) and self.robot.connected and getattr(self.robot.transport, "name", "") == "sim":
            self.connect_sim(connect_robot=False, start_cam=True)
        if self.camera is None or not self.camera.running:
            return
        now = time.monotonic()
        if now - self._last_preview_time < PREVIEW_INTERVAL:
            return
        self._last_preview_time = now
        latest = self.camera.latest_frame()
        if latest is None or latest.index == self._previewed_index:
            return
        self._previewed_index = latest.index
        full = latest.image
        if bool(self.state.get("ur5_preview_detail", False)):
            width, height = full.size
            image = full.crop((width // 4, height // 4, width - width // 4, height - height // 4))
        else:
            image = self.camera.preview_image(PREVIEW_MAX_WIDTH)
        if now - self._last_sharpness_time >= SHARPNESS_INTERVAL:
            self._last_sharpness_time = now
            self.live_sharpness = robot_camera.sharpness(full)
        if bool(self.state.get("ur5_show_detection", True)):
            mask = fabric_detector.texture_mask(image.resize((DETECTION_WIDTH, max(1, image.size[1] * DETECTION_WIDTH // image.size[0]))))
            self.live_fabric_share = float(mask.mean())
            image = fabric_detector.annotate(image, mask, [f"fabric {self.live_fabric_share:.0%}"
                                                           + ("" if self.live_sharpness is None else f"   sharpness {self.live_sharpness:.0f}")])
        self._upload_preview(image)

    def _upload_preview(self, image: Image.Image) -> None:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        height, width = rgb.shape[:2]
        if self.gl_ctx is not None:
            self.preview_texture = upload_rgb_texture(self.gl_ctx, self.preview_texture, rgb)
        self._preview_size = (width, height)

    # -- camera ------------------------------------------------------------

    def scan_devices(self) -> None:
        self.devices = self._simulated_cameras() + robot_camera.list_devices()
        real = [d for d in self.devices if d.get("is_real")]
        self.status = (
            f"Found {len(real)} camera(s)" if real
            else "No camera detected; only the synthetic source is available"
        )

    def _simulated_cameras(self) -> list[dict]:
        """The cameras TeleopApp mounts on the arm, listed first: with it on its emulator they are the simulator's."""
        transport = self.robot.transport
        if not hasattr(transport, "cameras") or not transport.linked():
            return []
        try:
            names = transport.cameras()
        except Exception:
            return []
        return [{"backend": "teleop", "camera": name, "index": -2, "is_real": False,
                 "label": f"TeleopApp: {name}" + ("" if transport.on_hardware else " (simulated)")} for name in names]

    def start_camera(self, spec: dict, size) -> bool:
        self.stop_camera()
        backend = (
            teleop_transport.TeleopCameraBackend(self.robot.transport, spec["camera"], size)
            if spec.get("backend") == "teleop"
            else robot_camera.make_backend(
                spec,
                size=size,
                pose_provider=self._sim_camera_pose_provider,
                scene_provider=self._sim_camera_scene_provider,
            )
        )
        self.camera = robot_camera.RobotCamera(backend)
        self._previewed_index = -1
        if not self.camera.start():
            self.status = self.camera.status
            return False
        self.apply_focus()
        self.status = self.camera.status
        return True

    def test_fabric_check(self) -> None:
        """Run the fabric check on the live frame, so it is known to work before a run relies on it."""
        frame = None if self.camera is None else self.camera.latest_frame()
        if frame is None:
            self.status = "Fabric check: start the camera first"
            return
        try:
            share, _ = fabric_detector.coverage(frame.image)
        except Exception as exc:
            self.status = f"Fabric check failed: {exc}"
            return
        self.status = f"Fabric check: {share * 100:.0f}% of this frame is fabric"

    def apply_focus(self) -> None:
        if self.camera is not None:
            self.camera.request_focus(bool(self.state.get("ur5_camera_autofocus", True)), float(self.state.get("ur5_camera_focus", 0.0)))

    def find_focus(self) -> None:
        """Sweep the lens and lock it where the live frame is sharpest - autofocus off, so a run keeps it."""
        if self.camera is None or not self.camera.running:
            self.status = "Find best focus needs the camera: start the live preview first"
            return
        self._on_worker(self._focus_loop)

    def _focus_loop(self) -> None:
        low, high = FOCUS_RANGE
        scores = {}

        def judge(positions):
            for position in positions:
                if self._centring_stop.is_set():
                    raise RuntimeError("stopped")
                self.camera.request_focus(False, float(position))
                frame = self.camera.capture(newer_than=time.monotonic() + FOCUS_SETTLE)
                if frame is None:
                    raise RuntimeError("the camera delivered no frame")
                scores[float(position)] = robot_camera.sharpness(frame.image)
                self.status = f"Finding focus: lens at {position:.0f}, sharpness {scores[float(position)]:.0f}"

        try:
            judge(np.arange(low, high + 1e-6, FOCUS_COARSE_STEP))
            coarse = max(scores, key=scores.get)
            judge(np.arange(max(low, coarse - FOCUS_COARSE_STEP), min(high, coarse + FOCUS_COARSE_STEP) + 1e-6, FOCUS_FINE_STEP))
        except Exception as exc:
            self.apply_focus()
            self.status = f"Focus not found: {exc}"
            return
        best = max(scores, key=scores.get)
        worst = min(scores.values())
        if scores[best] - worst < FOCUS_FLAT * max(scores[best], 1e-9):
            self.apply_focus()
            self.status = (f"Focus not found: the picture did not change across the sweep ({self.camera.status}). "
                           "Open the camera's own settings below to see if it has a focus control at all")
            return
        self.state.ur5_camera_autofocus = False
        self.state.ur5_camera_focus = float(best)
        self.apply_focus()
        self.status = f"Focus locked at {best:.0f} (sharpness {scores[best]:.0f}, autofocus off so a run keeps it)"

    def stop_camera(self) -> None:
        if self.camera is not None:
            self.camera.stop()
            self.camera = None
        if self.preview_texture is not None:
            try:
                self.preview_texture.release()
            except Exception as exc:
                logging.warning("gui_ur5.stop_camera: %s", exc)
            self.preview_texture = None
            self._preview_size = (0, 0)

    # -- the capture path --------------------------------------------------

    def build_plan(self) -> bool:
        """Solve and check the whole path from where the arm stands, before it can be run.

        Every viewpoint is reached from the previous one and every transition is collision
        checked, so a refusal here is a refusal to move at all rather than a stall partway
        through - which is what the arm did while this check lived nowhere.
        """
        self.apply_capture_settings()
        if not self.robot.transport.linked():
            self.plan = None
            reason = self.robot.state.error or self.robot.status
            self.plan_error = f"No live link to the arm to read where it stands ({reason}) - Connect first"
            self.status = f"No capture path: {self.plan_error}"
            return False
        try:
            self.plan = scan_geometry.build_capture_plan(self.robot.refresh().joints, **_capture_settings(self.state))
            self.plan_error = ""
        except Exception as exc:
            self.plan = None
            self.plan_error = str(exc)
            self.status = f"No capture path: {exc}"
            return False

        plan = self.plan
        moved = sum(1 for point in plan.viewpoints if point.adjustment)
        self.status = (f"{len(plan.viewpoints)} viewpoints ready"
                       + (f", {moved} moved to the nearest safe spot" if moved else "")
                       + (f", {len(plan.skipped)} skipped - nothing safe near them" if plan.skipped else "")
                       + f". Worst move {max(point.transition_deg for point in plan.viewpoints):.0f} deg, fabric plane z {plan.fabric_z:+.3f} m"
                       + ("" if self.robot.connected else f". Planned from the last joints read - {self.robot.state.error or 'the arm is not connected'}; Run waits for it"))
        return plan.clear

    @property
    def planning(self) -> bool:
        return self.plan_thread is not None and self.plan_thread.is_alive()

    def check_in_background(self) -> None:
        """Check on a worker: solving a whole path - nearest-safe search included - takes seconds,
        and on the panel's own thread that froze the window for all of them.

        A Check asked for while one is running is not dropped: that one is followed by another, so
        the path left on screen is always planned from the latest settings - which is what lets a
        new capture height re-plan on its own however fast it is changed."""
        with self._check_lock:
            self.plan = None
            if self.planning:
                self._check_again = True
                return
            self.status = "Checking the path..."
            self.plan_thread = threading.Thread(target=self._check_loop, daemon=True, name="ur5-check")
            self.plan_thread.start()

    def replan(self) -> None:
        """Check again after a setting changed - when there is an arm to plan from, and no run is flying the path."""
        if self.robot.transport.linked() and not (self.runner is not None and self.runner.running):
            self.check_in_background()

    def _check_loop(self) -> None:
        while True:
            self.build_plan()
            with self._check_lock:
                if not self._check_again:
                    self.plan_thread = None
                    return
                self._check_again = False
                self.plan = None

    def apply_capture_settings(self) -> None:
        """Push the panel's pacing and calibration onto the transport that will fly it."""
        transport = self.robot.transport
        for attribute, key in (("tool_offset",         "teleop_tool_offset"),
                               ("max_joint_speed_deg", "teleop_speed_deg_s"),
                               ("floor_seconds",       "teleop_move_seconds")):
            if hasattr(transport, attribute):
                setattr(transport, attribute, float(self.state.get(key, getattr(transport, attribute))))

    @property
    def rehearsing(self) -> bool:
        return self.rehearse_thread is not None and self.rehearse_thread.is_alive()

    def rehearse(self) -> bool:
        """Walk the solved path on a virtual robot, with nothing driving the arm.

        This is the look-before-you-move step: the same configurations, in the same order,
        at the same pace, with the arm standing still. It refuses on a transport that would
        execute the moves itself, because then it would not be a rehearsal.
        """
        if self.plan is None or self.rehearsing:
            return False
        if not self.robot.transport.rehearsable():
            self.status = ("This transport moves the arm itself, so there is nothing to rehearse on - "
                           "hold the arm in TeleopApp, or drive it through TeleopApp instead")
            return False

        self.apply_capture_settings()
        self._rehearse_stop.clear()
        self.rehearse_thread = threading.Thread(target=self._rehearse_loop, daemon=True, name="ur5-rehearse")
        self.rehearse_thread.start()
        return True

    def stop_rehearsal(self) -> None:
        self._rehearse_stop.set()

    def _rehearse_loop(self) -> None:
        transport = self.robot.transport
        try:
            for index, point in enumerate(self.plan.viewpoints):
                if self._rehearse_stop.is_set():
                    break
                self.rehearse_index = index
                ramp = transport.ramp_for(point.config)
                self.status = f"Rehearsing {point.name} ({index + 1}/{len(self.plan.viewpoints)}), {ramp:.1f} s"
                transport.move_to_joints(point.config, 0.0, 0.0)
                self._rehearse_stop.wait(ramp + 0.5)
            self.status = "Rehearsal stopped" if self._rehearse_stop.is_set() else "Rehearsal finished"
        except Exception as exc:
            self.status = f"Rehearsal failed: {exc}"
        finally:
            self.rehearse_index = -1

    # -- teacher mode ------------------------------------------------------

    def free_drive(self) -> None:
        self.stop_rehearsal()
        try:
            self.robot.free_drive()
        except Exception as exc:
            self.robot.status = f"Free drive refused: {exc}"
        self.status = self.robot.status

    def hold(self) -> None:
        try:
            self.robot.hold()
        except Exception as exc:
            self.robot.status = f"Hold failed - the arm may still be loose: {exc}"
        self.status = self.robot.status

    def record_view(self) -> None:
        """Append where the arm stands now to the taught views, as its joints."""
        try:
            joints = self.robot.joints_now()
        except Exception as exc:
            self.status = f"Not recorded: {exc}"
            return
        views = list(self.state.get("teleop_taught_views", []))
        views.append({"name": f"view {len(views) + 1}", "joints": joints})
        self._set_taught_views(views)
        self.status = f"Recorded view {len(views)}"

    def forget_view(self, index: int) -> None:
        views = list(self.state.get("teleop_taught_views", []))
        del views[index]
        self._set_taught_views([{**view, "name": f"view {number + 1}"} for number, view in enumerate(views)])

    def _set_taught_views(self, views) -> None:
        self.state.teleop_taught_views = views
        # The checked path described the old views; keeping it on screen invites running it.
        if self.plan is not None and self.plan.pattern == "taught":
            self.plan = None

    # -- over the fabric ---------------------------------------------------

    def _fly_over(self, centre_xy, height, stop=None):
        """Solve, check and fly one straight-down view over `centre_xy` (or the fabric's middle), and wait for it."""
        settings = {**_capture_settings(self.state), "pattern": "centre", "camera_height": float(height), "centre_xy": centre_xy}
        estimated = "" if _measured_fabric_z(self.state) is not None else (
            " - the fabric's height is only a guess from Home; set it under 2 Position > Fabric so heights are true")
        try:
            plan = scan_geometry.build_capture_plan(self.robot.refresh().joints, **settings)
        except RuntimeError as exc:
            raise RuntimeError(f"{exc}{estimated}") from exc
        if not plan.clear:
            raise RuntimeError("the move there is refused: " + "; ".join(plan.viewpoints[0].problems) + estimated)
        joints = plan.configs[0]
        self.apply_capture_settings()
        self.robot.move_to_joints(joints)
        deadline = time.monotonic() + scan_geometry.timeout_for(self.robot.transport.ramp_for(joints))
        while not self.robot.at_joints(joints):
            if stop is not None and stop.is_set():
                raise RuntimeError("stopped")
            if time.monotonic() > deadline:
                raise RuntimeError(f"the arm did not arrive ({np.degrees(self.robot.joint_error(joints)):.1f} deg off)")
            time.sleep(0.05)
        return plan.viewpoints[0]

    def jog(self, right: float, forward: float, up: float) -> None:
        """Move the camera from the panel: `right` and `forward` as seen in the live preview, `up` away from the cloth.

        Flown on the worker through _fly_over, so a jog is solved and checked exactly as a capture
        is - it cannot take the camera into the cloth or the arm into itself - and it leaves the
        camera looking straight down, which is how every capture looks."""
        self._on_worker(lambda: self._jog_loop(np.array([right, forward, up], dtype=float)))

    def _jog_loop(self, step) -> None:
        try:
            position, rotvec = scan_geometry.camera_pose(self.robot.refresh().joints, float(self.state.get("teleop_tool_offset", 0.0306)))
            axes = scan_geometry.camera_rotation(rotvec, _image_turn(self.state))
            right, down = (np.r_[axes[:2, column], 0.0] / max(float(np.linalg.norm(axes[:2, column])), 1e-9) for column in (0, 1))
            target = position + step[0] * right - step[1] * down
            now = float(position[2] - _settings_fabric_z(self.state, None))
            if step[2] < 0.0 and now <= CAPTURE_HEIGHT_MIN + 0.001:
                raise RuntimeError(f"already at the lowest allowed, {CAPTURE_HEIGHT_MIN * 100:.1f} cm over the fabric"
                                   + ("" if _measured_fabric_z(self.state) is not None else
                                      " - but that is over a guessed fabric height; set it under Fabric if there is more room"))
            height = max(now + step[2], CAPTURE_HEIGHT_MIN)
            self.status = "Moving the camera"
            self._fly_over(target[:2], height, self._centring_stop)
        except Exception as exc:
            self.status = f"Camera not moved: {exc}"
            return
        self.status = (f"Camera moved - now {height * 100:.1f} cm over the fabric"
                       + (" (stopped at the lowest allowed)" if height > now + step[2] + 1e-9 else ""))

    def mark_corner_under_crosshair(self) -> None:
        """Record the fabric point under the preview's crosshair as a corner - no hand on the arm.

        The point is where the camera's axis meets the fabric plane, so it is only as good as that
        plane: with the height still a guess it is refused - measure or set the height first."""
        if self.state.get("teleop_fabric_z") is None:
            self.status = "Corner not marked: set the fabric height first (Measure with the camera, or a ruler gap)"
            return
        try:
            position, rotvec = scan_geometry.camera_pose(self.robot.joints_now(), float(self.state.get("teleop_tool_offset", 0.0306)))
            corner = scan_geometry.fabric_point_below(position, rotvec, _settings_fabric_z(self.state, None),
                                                      float(self.state.get("teleop_subject_distance", 0.20)))
        except Exception as exc:
            self.status = f"Corner not marked: {exc}"
            return
        corners = list(self.state.get("teleop_fabric_corners", [])) + [[float(value) for value in corner]]
        self.state.teleop_fabric_corners = corners
        self.plan = None
        self.status = (f"Corner {len(corners)} marked under the crosshair"
                       + (f" - jog to the next corner ({3 - len(corners)} more for an outline)" if len(corners) < 3 else " - outline set"))

    def outline_from_preview(self) -> None:
        """Set the cloth's outline from what the detector calls cloth in the live preview, in place of marking corners.

        Wants the camera straight down over the whole cloth - the overview height - and the fabric
        height set, since the outline is the detected edge followed down to that plane."""
        frame = None if self.camera is None else self.camera.latest_frame()
        if frame is None:
            self.status = "Outline not set: start the live preview first"
            return
        if self.state.get("teleop_fabric_z") is None:
            self.status = "Outline not set: set the fabric height first (Measure with the camera, or a ruler gap)"
            return
        try:
            position, rotvec = scan_geometry.camera_pose(self.robot.joints_now(), float(self.state.get("teleop_tool_offset", 0.0306)))
            _, mask = fabric_detector.coverage(frame.image)
            corners = scan_geometry.outline_from_mask(mask, position, rotvec, _image_turn(self.state), _settings_fabric_z(self.state, None),
                                                      float(self.state.get("ur5_splat_fov", gaussian_splatting.DEFAULT_FOV_DEGREES)))
        except Exception as exc:
            self.status = f"Outline not set: {exc}"
            return
        self.state.teleop_fabric_corners = [[float(value) for value in corner] for corner in corners]
        self.plan = None
        hull = scan_geometry.FabricRegion(corners).outline
        edges = np.roll(hull, -1, axis=0) - hull
        along = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))] / np.linalg.norm(edges, axis=1).max()
        sides = sorted(float(np.ptp(hull @ axis)) * 100.0 for axis in (along, np.array([-along[1], along[0]])))
        self.status = f"Cloth outlined from the preview: {len(corners)} corners, about {sides[1]:.0f} x {sides[0]:.0f} cm - check the outline, then Check the path"

    def calibrate_image_turn(self) -> None:
        """Learn how the camera's image is turned on its mount: probe 4 mm along x and y over the
        fabric, see which way the picture slides, and keep the turn. Every straight view, Home, the
        splat preview and the export then use the camera's real axes instead of the tool's."""
        if self.camera is None or not self.camera.running:
            self.status = "Calibration needs the camera: start the live preview first"
            return
        self._on_worker(self._calibrate_loop)

    def _calibrate_loop(self) -> None:
        stop = self._centring_stop
        try:
            region = _fabric_region(self.state)
            here = region.centre[:2] if region is not None and region.has_outline else scan_geometry.camera_position(
                self.robot.refresh().joints, float(self.state.get("teleop_tool_offset", 0.0306)))[:2]
            here = np.asarray(here, dtype=float)
            self.status = "Calibrating the camera: looking straight down at the fabric"
            view = self._fly_over(here, CALIBRATION_HEIGHT, stop)
            before = self._frame()
            slides = []
            for axis in (np.array([CALIBRATION_STEP, 0.0]), np.array([0.0, CALIBRATION_STEP])):
                self.status = "Calibrating the camera: probing which way the picture slides"
                self._fly_over(here + axis, CALIBRATION_HEIGHT, stop)
                slides.append(robot_camera.image_shift(before, self._frame()) / CALIBRATION_STEP)
            self._fly_over(here, CALIBRATION_HEIGHT, stop)

            width = before.size[0]
            fov = float(self.state.get("ur5_splat_fov", gaussian_splatting.DEFAULT_FOV_DEGREES))
            focal = gaussian_splatting.intrinsics_from_fov(width, before.size[1], fov)["fl_x"]
            turn, error, scale = scan_geometry.calibrate_image_turn(scan_geometry._heading_of(view.rotvec), slides[0], slides[1],
                                                                   view.position[2] - _settings_fabric_z(self.state, view), focal)
            if error > CALIBRATION_MAX_ERROR:
                raise RuntimeError(f"the two probes disagree ({error * 100:.0f}% from a clean turn) - is the fabric "
                                   "textured under the camera, and did the arm really move?")
        except Exception as exc:
            self.status = f"Camera not calibrated: {exc}"
            return
        self.state.ur5_camera_image_turn = [[float(v) for v in row] for row in turn]
        self.plan = None
        angle = np.degrees(np.arctan2(turn[1, 0], turn[0, 0]))
        suggested = np.degrees(2.0 * np.arctan(width / (2.0 * focal * scale)))
        self.status = (f"Camera calibrated: its image is turned {angle:+.1f} deg on the tool"
                       + (" and mirrored" if np.linalg.det(turn) < 0 else "")
                       + (f"; the lens looks {suggested:.0f} deg wide, not {fov:.0f} - set that under 1 Setup > Calibration"
                          if abs(suggested - fov) > 2.0 else "")
                       + ". Now press Straighten Home")

    def _frame(self):
        frame = self.camera.capture(newer_than=time.monotonic() + ur5_scan.SETTLE_TIME)
        if frame is None:
            raise RuntimeError("the camera delivered no frame")
        return frame.image

    def straighten_home(self) -> None:
        """Home's camera position, lens exactly down, image squared to the cloth - flown there, checked,
        and kept as Home here and as TeleopApp's taught `Home`, so the two agree and a first capture
        from Home is straight."""
        self._on_worker(self._straighten_home_loop)

    def move_to_overview(self) -> None:
        """Fly back to the pose the arm stood in when the overview was set - the one that sees the whole cloth."""
        if self.state.get("teleop_overview_joints") is None:
            self.status = "No overview yet: jog until the whole cloth is in view, then Use the camera's height now as the overview"
            return
        self._on_worker(self._move_to_overview_loop)

    def _move_to_overview_loop(self) -> None:
        try:
            joints = np.asarray(self.state.teleop_overview_joints, dtype=float)
            settings = _capture_settings(self.state)
            check = scan_geometry.check_home_move(self.robot.refresh().joints, subject_distance=settings["subject_distance"],
                                                  tool_offset=settings["tool_offset"], fabric_z=settings["fabric_z"], home_joints=joints, fixed=True)
            if not check.clear:
                raise RuntimeError("the move there is refused: " + "; ".join(check.problems))
            self.status = f"Moving to the overview, {check.transition_deg:.0f} deg on the furthest joint"
            self.apply_capture_settings()
            self.robot.move_to_joints(joints)
            deadline = time.monotonic() + scan_geometry.timeout_for(self.robot.transport.ramp_for(joints))
            while not self.robot.at_joints(joints):
                if self._centring_stop.is_set():
                    raise RuntimeError("stopped")
                if time.monotonic() > deadline:
                    raise RuntimeError("the arm did not arrive")
                time.sleep(0.05)
        except Exception as exc:
            self.status = f"Not moved to the overview: {exc}"
            return
        self.status = "At the overview: the whole cloth should be in the preview"

    def _straighten_home_loop(self) -> None:
        try:
            settings = _capture_settings(self.state)
            joints = scan_geometry.straight_home(self.robot.home_joints, settings["tool_offset"], _fabric_region(self.state),
                                                 settings["image_turn"])
            check = scan_geometry.check_home_move(self.robot.refresh().joints, subject_distance=settings["subject_distance"],
                                                  tool_offset=settings["tool_offset"], fabric_z=settings["fabric_z"], home_joints=joints)
            if not check.clear:
                raise RuntimeError("the move there is refused: " + "; ".join(check.problems))
            self.status = "Straightening Home: moving there"
            self.apply_capture_settings()
            self.robot.move_to_joints(joints)
            deadline = time.monotonic() + scan_geometry.timeout_for(self.robot.transport.ramp_for(joints))
            while not self.robot.at_joints(joints):
                if self._centring_stop.is_set():
                    raise RuntimeError("stopped")
                if time.monotonic() > deadline:
                    raise RuntimeError("the arm did not arrive")
                time.sleep(0.05)
            if hasattr(self.robot.transport, "client") and self.robot.transport.client is not None:
                self.robot.transport.client.call("savePose", self.robot.transport.target, "Home")
            self.state.ur5_home_joints = [float(value) for value in joints]
            self.robot.home_joints = list(self.state.ur5_home_joints)
            self.plan = None
        except Exception as exc:
            self.status = f"Home not straightened: {exc}"
            return
        turned = np.degrees(np.max(np.abs(np.asarray(joints) - np.asarray(ur5_robot.DEFAULT_HOME_JOINTS))))
        self.status = f"Home straightened - lens straight down, image squared to the cloth ({turned:.0f} deg on the furthest joint); saved as Home"

    def move_over_fabric(self) -> None:
        """Straight down over the fabric's middle, at the capture height - where a scan starts."""
        self._on_worker(self._move_over_loop)

    def set_capture_height(self, height: float) -> None:
        """Take a new capture height and re-plan the capture points for it. Without a link to the
        arm there is nothing to plan from, so the path is only dropped until Check."""
        self.state.teleop_camera_height = float(np.clip(height, CAPTURE_HEIGHT_MIN, CAPTURE_HEIGHT_MAX))
        self.plan = None
        self.replan()

    def camera_height_over_fabric(self, joints) -> float:
        """How far the camera stands over the fabric plane with the arm in `joints`."""
        camera = scan_geometry.camera_position(joints, float(self.state.get("teleop_tool_offset", 0.0306)))
        return float(camera[2] - _settings_fabric_z(self.state, None))

    def adopt_camera_height(self, joints=None, overview=False) -> None:
        """Capture from here: the height the camera stands at now - or in a taught view's `joints` -
        becomes the capture height, exactly.

        Marked, that height is measured from the fabric. Unmarked, the plane is a guess from Home that
        can sit far off the real cloth, so the lens-to-cloth gap typed in the panel places it first -
        which is what makes the capture height, the grid's spacing and every clearance true at this
        pose. Focus is left as it is: the camera was already looking from here."""
        try:
            joints = self.robot.joints_now() if joints is None else [float(value) for value in joints]
            if _measured_fabric_z(self.state) is None:
                if not self.fabric_gap_cm:
                    raise RuntimeError("the fabric's height is not set - type how far the lens is over the cloth (cm) first")
                self._place_fabric(joints, self.fabric_gap_cm / 100.0)
            height = self.camera_height_over_fabric(joints)
        except Exception as exc:
            self.status = f"Height not taken: {exc}"
            return
        limit = OVERVIEW_HEIGHT_MAX if overview else CAPTURE_HEIGHT_MAX
        if not CAPTURE_HEIGHT_MIN <= height <= limit:
            self.status = (f"Height not taken: the camera is {height * 100:.1f} cm over the fabric, outside "
                           f"{CAPTURE_HEIGHT_MIN * 100:.1f}-{limit * 100:.0f} cm")
            return
        if overview:
            self.state.teleop_overview_height = float(height)
            self.state.teleop_overview_joints = [float(value) for value in joints]
            self.plan = None
            self.replan()
            self.status = f"Overview from here: {height * 100:.1f} cm over the fabric - the last pass takes the whole view from this height"
            return
        self.set_capture_height(height)
        self.status = f"Capturing from here: {height * 100:.1f} cm over the fabric - the capture points are re-planned at this height"

    def set_fabric_gap(self, gap: float) -> None:
        """Set the fabric's height from a ruler: the lens is `gap` over the cloth now.

        Unset, every height and every clearance is measured from a plane guessed from Home, and a
        guess above the real cloth refuses moves into room that is there. The plane is kept, and
        saved with the other settings, until it is set again or forgotten."""
        try:
            self._place_fabric(self.robot.joints_now(), gap)
        except Exception as exc:
            self.status = f"Gap not used: {exc}"
            return
        self.plan = None
        self.replan()
        self.status = f"Fabric height set {gap * 100:.1f} cm under the lens - the camera can now go down to it"

    def _place_fabric(self, joints, gap: float) -> None:
        """Put the fabric plane `gap` under the camera in `joints`."""
        if gap <= 0.0:
            raise RuntimeError("the gap must be more than 0 cm")
        camera = scan_geometry.camera_position(joints, float(self.state.get("teleop_tool_offset", 0.0306)))
        self._set_fabric_plane(float(camera[2] - float(gap)))

    def _set_fabric_plane(self, z: float) -> None:
        """The fabric plane is at height `z`; the marked corners lie on it, so they are moved onto it too -
        otherwise a corner marked under the crosshair over the old plane would keep holding it there."""
        self.state.teleop_fabric_z = float(z)
        self.state.teleop_fabric_corners = [[float(c[0]), float(c[1]), float(z)] for c in self.state.get("teleop_fabric_corners", [])]
        self.plan = None

    def measure_fabric_height(self) -> None:
        """Measure how far the cloth is under the lens with the camera alone - no ruler, no hand on the arm."""
        if self.camera is None or not self.camera.running:
            self.status = "Measuring needs the camera: start the live preview first"
            return
        self._on_worker(self._measure_loop)

    def _measure_loop(self) -> None:
        """Parallax: slide the lens-down camera a few mm along x, then y, and see how far the picture moves.

        A slide of d moves the picture by f * d / h pixels, so h = f * d / shift. The step is kept well
        under half a stitch, so a repeating knit cannot alias the shift; the two probes must agree, or
        the picture slipped. As true as the lens field of view under 1 Setup > Calibration."""
        stop = self._centring_stop
        try:
            position, _ = scan_geometry.camera_pose(self.robot.refresh().joints, float(self.state.get("teleop_tool_offset", 0.0306)))
            here, height = position[:2], float(position[2] - _settings_fabric_z(self.state, None))
            self.status = "Measuring the fabric height: looking straight down"
            view = self._fly_over(here, height, stop)
            before = self._frame()
            slides = []
            for axis in (np.array([MEASURE_STEP, 0.0]), np.array([0.0, MEASURE_STEP])):
                self.status = "Measuring the fabric height: sliding the camera a few mm"
                self._fly_over(here + axis, height, stop)
                slides.append(float(np.linalg.norm(robot_camera.image_shift(before, self._frame()))) / MEASURE_STEP)
            self._fly_over(here, height, stop)
            if min(slides) <= 0.0 or abs(slides[0] - slides[1]) > MEASURE_AGREEMENT * max(slides):
                raise RuntimeError("the two probes disagree - is there textured fabric under the camera, and did the arm move?")
            fov = float(self.state.get("ur5_splat_fov", gaussian_splatting.DEFAULT_FOV_DEGREES))
            focal = gaussian_splatting.intrinsics_from_fov(before.size[0], before.size[1], fov)["fl_x"]
            gap = focal / float(np.mean(slides))
        except Exception as exc:
            self.status = f"Fabric height not measured: {exc}"
            return
        self._set_fabric_plane(float(view.position[2] - gap))
        self.fabric_gap_cm = round(gap * 100.0, 1)
        self.replan()
        self.status = f"Fabric measured {gap * 100:.1f} cm under the lens - the camera can now go down to it"

    def forget_fabric_height(self) -> None:
        self.state.teleop_fabric_z = None
        self.plan = None
        self.status = "Fabric height forgotten - it is a guess from Home again"

    def _move_over_loop(self) -> None:
        height = float(self.state.get("teleop_camera_height", 0.07))
        try:
            self.status = "Moving over the fabric's middle"
            self._fly_over(None, height, self._centring_stop)
        except Exception as exc:
            self.status = f"Not moved over the fabric: {exc}"
            return
        self.replan()
        self.status = f"Over the fabric's middle, {height * 100:.1f} cm up - re-planning the capture points for this height"

    @property
    def centring(self) -> bool:
        return self.centring_thread is not None and self.centring_thread.is_alive()

    def centre_on_fabric(self) -> None:
        """Find the fabric with the camera and centre over it."""
        if self.camera is None or not self.camera.running:
            self.status = "Centring needs the camera: start the live preview first"
            return
        self._on_worker(self._centre_loop)

    def _on_worker(self, loop) -> None:
        """Moves that wait for the arm run here, so the panel - and its stop button - stay live."""
        if self.centring:
            return
        self._centring_stop.clear()
        self.centring_thread = threading.Thread(target=loop, daemon=True, name="ur5-over-fabric")
        self.centring_thread.start()

    def stop_centring(self) -> None:
        self._centring_stop.set()

    def _look(self):
        frame = self.camera.capture(newer_than=time.monotonic() + ur5_scan.SETTLE_TIME)
        if frame is None:
            raise RuntimeError("the camera delivered no frame")
        return fabric_detector.fabric_offset(frame.image)

    def _centre_loop(self) -> None:
        """Probe, then step the fabric's middle to the frame's centre; adopt the centre found.

        The centre found moves the fabric outline with it, so every plan after this is centred on
        the cloth the camera saw - which is the point on a real cell, where marked corners are only
        as good as the hand that marked them.
        """
        stop = self._centring_stop
        try:
            region = _fabric_region(self.state)
            start = region.centre[:2] if region is not None and region.has_outline else scan_geometry.camera_position(
                self.robot.refresh().joints, float(self.state.get("teleop_tool_offset", 0.0306)))[:2]
            here = np.asarray(start, dtype=float)

            self.status = "Centring: looking at the fabric from above"
            self._fly_over(here, CENTRING_HEIGHT, stop)
            offset, share = self._look()
            if share < FULL_FRAME_SHARE:
                columns = []
                for axis in (np.array([PROBE_STEP, 0.0]), np.array([0.0, PROBE_STEP])):
                    self.status = "Centring: probing which way the image moves"
                    self._fly_over(here + axis, CENTRING_HEIGHT, stop)
                    moved, _ = self._look()
                    columns.append((moved - offset) / PROBE_STEP)
                image_per_metre = np.column_stack(columns)
                if abs(np.linalg.det(image_per_metre)) < 1e-3:
                    raise RuntimeError("the image did not move with the arm - is the camera the one on the arm?")

                for iteration in range(MAX_CENTRING_ITERATIONS):
                    if np.linalg.norm(offset) < CENTRE_TOLERANCE or share >= FULL_FRAME_SHARE:
                        break
                    step = -np.linalg.solve(image_per_metre, offset)
                    step *= min(1.0, MAX_CENTRING_STEP / max(float(np.linalg.norm(step)), 1e-9))
                    here = here + step
                    self.status = f"Centring: step {iteration + 1}, {np.linalg.norm(step) * 100:.1f} cm"
                    self._fly_over(here, CENTRING_HEIGHT, stop)
                    offset, share = self._look()

            if region is not None and region.has_outline:
                shift = here - region.centre[:2]
                self.state.teleop_fabric_corners = [[c[0] + shift[0], c[1] + shift[1], c[2]] for c in self.state.get("teleop_fabric_corners", [])]
                self.plan = None
            self.status = (f"Centred on the fabric: {share * 100:.0f}% of the frame is knit, its middle "
                           f"{np.linalg.norm(offset) * 100:.1f}% of a frame off centre"
                           + (f"; outline moved {np.linalg.norm(shift) * 100:.1f} cm to match" if region is not None and region.has_outline else ""))
        except Exception as exc:
            self.status = f"Centring stopped: {exc}"

    def mark_corner(self) -> None:
        """Record where the camera is, which the user has rested on a corner of the fabric by hand.

        A touched corner measures the fabric's height, so it sets the plane (and moves the other
        corners onto it); three or more give the outline that keeps every automatic frame on the cloth.
        """
        tool_offset = float(self.state.get("teleop_tool_offset", 0.0306))
        try:
            corner = scan_geometry.camera_position(self.robot.joints_now(), tool_offset)
            at_home = float(scan_geometry.camera_position(ur5_robot.DEFAULT_HOME_JOINTS, tool_offset)[2])
        except Exception as exc:
            self.status = f"Corner not marked: {exc}"
            return
        if corner[2] > at_home - HOME_TO_FABRIC_MIN:
            self.status = (f"Corner not marked: the camera is at z = {corner[2]:+.3f} m, within "
                           f"{HOME_TO_FABRIC_MIN * 100:.0f} cm of where it is at Home ({at_home:+.3f} m) - Home looks down at "
                           "the fabric from above, so the lens is not on it. Free drive, rest the lens on the fabric corner, then mark")
            return
        corners = list(self.state.get("teleop_fabric_corners", [])) + [[float(value) for value in corner]]
        self.state.teleop_fabric_corners = corners
        self._set_fabric_plane(float(corner[2]))
        self.status = (f"Corner {len(corners)} marked at [{corner[0]:+.3f}, {corner[1]:+.3f}, {corner[2]:+.3f}] m"
                       + (" - mark at least 3 for an outline" if len(corners) < 3 else " - outline set"))

    def clear_corners(self) -> None:
        self.state.teleop_fabric_corners = []
        self.plan = None
        self.status = "Fabric corners cleared - the fabric height is unchanged"

    def save_taught_set(self, name: str) -> None:
        """Write the taught views to their own file, so the same shots can be taken of the next fabric."""
        views = list(self.state.get("teleop_taught_views", []))
        if not views:
            self.status = "Nothing to save - record at least one view first"
            return
        try:
            path = taught_set_path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, {"name": path.stem, "tool_offset": float(self.state.get("teleop_tool_offset", 0.0306)),
                                        "views": views}, indent=2)
        except Exception as exc:
            self.status = f"Could not save taught views: {exc}"
            return
        self.state.teleop_taught_set = path.stem
        self.status = f"Saved {len(views)} taught views as `{path.stem}`"

    def load_taught_set(self, name: str) -> None:
        try:
            saved = json.loads(taught_set_path(name).read_text())
            views = [{"name": str(view["name"]), "joints": [float(value) for value in view["joints"]]} for view in saved["views"]]
        except Exception as exc:
            self.status = f"Could not load taught set `{name}`: {exc}"
            return
        self._set_taught_views(views)
        self.state.teleop_taught_set = str(name)
        self.status = f"Loaded {len(views)} taught views from `{name}` - Check before running"

    # -- robot -------------------------------------------------------------

    def connect(self, ip: str, transport_kind: str) -> bool:
        # A camera read through TeleopApp belongs to the connection being replaced; left running it
        # would keep asking the old TeleopApp for frames, and a real-arm run would photograph the
        # simulator.
        if self.camera is not None and isinstance(self.camera.backend, teleop_transport.TeleopCameraBackend):
            self.stop_camera()
        self.stop_centring()
        self.robot = ur5_robot.UR5Robot(TRANSPORTS[transport_kind].build())
        self.robot.home_joints = list(self.state.get("ur5_home_joints") or ur5_robot.DEFAULT_HOME_JOINTS)
        ok = self.robot.connect(ip)
        failure = self.robot.last_exception
        self.blocking_program = failure if isinstance(failure, ur5_robot.ProgramAlreadyRunning) else None
        if ok:
            self.scan_devices()
        self.status = self.robot.status
        return ok

    def build_sim_cell(self, renderer) -> None:
        """Set the table and the knitted fabric into TeleopApp's emulator scene, and take the fabric as marked.

        Only on the emulator: on the real arm the fabric is the real one, its corners are touched by
        hand, and a simulated outline adopted there would steer real frames by a cloth that is not there.
        """
        transport = self.robot.transport
        if not hasattr(transport, "cameras") or not transport.linked():
            self.status = "Simulated cell: connect Via TeleopApp first"
            return
        if transport.on_hardware:
            self.status = "Simulated cell refused: TeleopApp is driving the real arm - this is for its emulator"
            return
        try:
            if renderer is not None and hasattr(renderer, "render_wrist_camera"):
                rgb = renderer.render_wrist_camera(
                    [-0.45, -0.08, 0.25, float(np.pi), 0.0, 0.0],
                    float(self.state.get("ur5_splat_fov", gaussian_splatting.DEFAULT_FOV_DEGREES)),
                    (-0.45, -0.08, 0.0),
                    0.18,
                    512,
                    512,
                    scanner_core._scanner_material_uniforms(self.state)
                    if hasattr(self.state, "get_material_uniforms")
                    else {},
                    visible_rows=getattr(self.state, "row_visible", None),
                    model_mat=(
                        self.state.current_model_matrix()
                        if hasattr(self.state, "current_model_matrix")
                        else None
                    ),
                )
                texture = Image.fromarray(rgb, "RGB")
            else:
                colors, _ = scanner_core._scanner_display_batch_colors(self.state)
                rows = max(1, int(self.state.get("scanner_rows", 1)))
                cols = max(1, int(self.state.get("scanner_cols", 1)))
                grid = np.full((rows, cols, 3), 128, dtype=np.uint8)
                for idx, rgb_val in enumerate(colors[: rows * cols]):
                    grid[idx // cols, idx % cols] = np.clip(rgb_val[:3], 0, 255).astype(np.uint8)
                texture = Image.fromarray(grid, "RGB").resize((512, 512), Image.Resampling.NEAREST)
            corners = teleop_transport.build(transport, texture)
        except Exception as exc:
            self.status = f"Simulated cell not built: {exc}"
            return
        self.state.teleop_fabric_corners = corners
        self._set_fabric_plane(float(np.mean([corner[2] for corner in corners])))
        self.scan_devices()
        self.status = (f"Simulated cell built: table and {teleop_transport.FABRIC_WIDTH * 100:.0f} cm knitted fabric in TeleopApp, "
                       "its corners taken as the fabric outline. Start the camera 'TeleopApp: Gripper camera', then Check")

    def remove_sim_cell(self) -> None:
        try:
            removed = teleop_transport.remove(self.robot.transport)
        except Exception as exc:
            self.status = f"Simulated cell not removed: {exc}"
            return
        self.status = f"Removed {removed} simulated cell prop(s) from TeleopApp"

    def stop_program_and_connect(self, ip: str, transport_kind: str) -> bool:
        """Stop the program the controller was found running, then connect. Only on the user's click.

        Stopping is the safe direction - the arm comes to rest under its own program's stop -
        but it ends whatever that program was for, so the panel names it before offering this.
        """
        try:
            ur5_robot.stop_program(ip)
        except Exception as exc:
            self.status = f"Could not stop the program on the arm: {exc} - stop it on the pendant"
            return False
        return self.connect(ip, transport_kind)

    def disconnect(self) -> None:
        if self.runner is not None and self.runner.running:
            self.runner.stop()
        self.robot.disconnect()
        self.status = self.robot.status

    def move_home(self) -> bool:
        """Home, after the move there passes the same check as every capture move.

        Home is reached from wherever the arm was left, which is the one move nobody planned,
        so it is the one most in need of the check.
        """
        try:
            settings = _capture_settings(self.state)
            home = scan_geometry.check_home_move(self.robot.refresh().joints, subject_distance=settings["subject_distance"],
                                                 tool_offset=settings["tool_offset"], fabric_z=settings["fabric_z"],
                                                 home_joints=self.robot.home_joints)
        except Exception as exc:
            self.status = f"Home move not checked, so not sent: {exc}"
            return False
        if not home.clear:
            if settings["fabric_z"] is None and home.clearance < 0.0:
                hint = (" The arm cannot really be under the fabric, so the fabric plane is wrong: "
                        "rest the lens on a fabric corner in free drive and press Mark corner.")
            elif home.floor < scan_geometry.FABRIC_CLEARANCE:
                hint = (f" The camera starts only {home.floor * 1000 + 2:.0f} mm over the fabric, and the joint move to Home "
                        "would take it lower than that: lift the camera straight up by hand in free drive, "
                        f"at least {scan_geometry.FABRIC_CLEARANCE * 100:.0f} cm, then Hold and try again.")
            else:
                hint = " Bring the arm nearer Home by hand in free drive first."
            self.status = "Home move refused: " + "; ".join(home.problems) + "." + hint
            return False
        if not self.robot.move_home():
            self.status = self.robot.status
            return False
        self.status = f"Moving to Home, {home.transition_deg:.0f} deg on the furthest joint, {home.clearance * 1000:.0f} mm over the fabric"
        return True

    def close(self) -> None:
        """Leaves nothing running behind when the mode is left or the app quits."""
        self._rehearse_stop.set()
        self._centring_stop.set()
        if self.runner is not None and self.runner.running:
            self.runner.stop()
            self.runner.wait(timeout=2.0)
        for job in (self.trainer, self.robot_solver):
            if job is not None and job.running:
                job.stop()
        if self.free is not None:
            self.free.close()
        self.stop_camera()
        gui_captures.release(self)
        self.robot.disconnect()


# ============================================================================
# Plan construction
# ============================================================================

def taught_set_path(name: str) -> Path:
    """Where a named set of taught views lives. The name is reduced to a safe file stem."""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name).strip()).strip("_")
    if not stem:
        raise RuntimeError("give the taught set a name")
    return paths.TAUGHT_VIEWS_DIR / f"{stem}.json"


def taught_set_names() -> list[str]:
    return sorted(path.stem for path in paths.TAUGHT_VIEWS_DIR.glob("*.json")) if paths.TAUGHT_VIEWS_DIR.is_dir() else []


def build_scan_plan(state):
    """Builds the grid CapturePlan from the scanner grid settings."""
    controller = state.get("ur5_controller")
    joints = (
        controller.robot.refresh().joints
        if controller is not None and controller.robot.connected
        else list(state.get("ur5_home_joints") or ur5_robot.DEFAULT_HOME_JOINTS)
    )
    settings = {
        **_capture_settings(state),
        "pattern": "grid",
        "grid_rows": max(1, int(state.get("scanner_rows", 1))),
        "grid_cols": max(1, int(state.get("scanner_cols", 1))),
        "grid_views_per_station": max(1, int(state.get("scanner_angles", 3))),
    }
    return scan_geometry.build_capture_plan(joints, **settings)


def _make_runner(state, controller, plan):
    """Assembles a run around one plan, snapshotting everything it must not read live."""
    storage = scanner_core._scanner_storage(state)
    signature = ""
    settings = {}
    if storage is not None:
        # Taken here on the UI thread. The run thread never touches AppState.
        signature = storage.pattern_signature(state)
        settings = storage.scanner_state_snapshot(state)
    estimates = scanner_core._scanner_estimated_cell_colors(state)

    # Clamped like every other capture path. This one derived its height with
    # no bounds at all, so an out-of-range width reached the camera unchecked.
    capture_width, capture_height = clamp_capture_size(state.get("ur5_capture_width", 1280))
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
            "ur5_transport": str(state.get("ur5_transport", "")),
            **({"teleop_capture": _capture_settings(state)} if getattr(plan, "pattern", "") else {}),
        },
        estimated_cell_colors=estimates,
        lighting_condition=str(state.get("ur5_lighting_condition", "")),
        dwell=float(state.get("ur5_dwell", 0.2)),
        save_images=bool(state.get("ur5_save_images", True)),
        min_sharpness=float(state.get("ur5_min_sharpness", 150.0)),
        fabric_check=_fabric_check(state),
        min_coverage=float(state.get("ur5_min_coverage", 0.8)),
        video=bool(state.get("ur5_video", True)),
    )


# ============================================================================
# Panel
# ============================================================================

def _draw_ur5_controls_column(state, controller, renderer):
    _draw_safety_bar(state, controller)
    expert_view = bool(state.get("_ur5_expert_view", False))
    half = max(90.0, (imgui.get_content_region_avail().x - imgui.get_style().item_spacing.x) * 0.5)
    is_light = str(state.get("ui_theme", "dark")) == "light"
    active_col = (0.18, 0.50, 0.82, 1.0) if is_light else (0.22, 0.52, 0.82, 1.0)
    if not expert_view:
        imgui.push_style_color(imgui.Col_.button, active_col)
    if imgui.button("Guided Workflow##ur5_mode_guided", (half, 26)):
        state._ur5_expert_view = False
        expert_view = False
    if not expert_view:
        imgui.pop_style_color()
    imgui.same_line()
    if expert_view:
        imgui.push_style_color(imgui.Col_.button, active_col)
    if imgui.button("All Settings & Tools##ur5_mode_expert", (half, 26)):
        state._ur5_expert_view = True
        expert_view = True
    if expert_view:
        imgui.pop_style_color()
    imgui.separator()

    if not expert_view:
        _draw_simple_page(state, controller, renderer)
        return

    imgui.push_id("ur5_everything")
    if imgui.begin_tab_bar("##ur5_tabs"):
        for label, draw in UR5_TABS:
            selected, _ = imgui.begin_tab_item(label)
            if selected:
                draw(state, controller, renderer)
                imgui.end_tab_item()
        imgui.end_tab_bar()
    imgui.pop_id()


def ensure_controller(state, gl_ctx=None, window=None, auto_connect_sim: bool = False) -> UR5ModeController:
    controller = state.get("ur5_controller")
    if controller is None:
        ctx = getattr(gl_ctx, "ctx", gl_ctx)
        controller = UR5ModeController(state, ctx, window)
        state.ur5_controller = controller
    if auto_connect_sim and not controller.robot.connected:
        controller.connect_sim()
    return controller


def draw_ur5_panel(state, renderer, window=None):
    """UR5 Robot Mode sidebar controls inside 'Knitting Control'."""
    controller = ensure_controller(state, renderer, window)
    controller.update_preview(renderer)
    _draw_ur5_controls_column(state, controller, renderer)


def draw_splatting_workflow_section(state, renderer, window=None):
    """3DGS dataset export, pose solver, and training workflow for Splatting mode."""
    controller = ensure_controller(state, renderer, window)
    imgui.push_id("splat_workflow_panel")
    if imgui.collapsing_header("Dataset, Poses & 3DGS Training##splat_workflow_hdr", imgui.TreeNodeFlags_.default_open):
        gui_captures.draw_captures_section(state, controller, "Captured Scan Sessions")
        _draw_splatting_body(state, controller)
    imgui.pop_id()


def draw_ur5_viewport(state, renderer, window=None):
    """UR5 Robot Mode visual workspace inside '3D View'."""
    controller = ensure_controller(state, renderer, window)

    avail_w = max(240.0, float(imgui.get_content_region_avail().x))
    left_w = max(240.0, (avail_w - imgui.get_style().item_spacing.x) * 0.5)

    imgui.push_id("ur5_workspace")
    imgui.begin_child("ur5_vp_camera", imgui.ImVec2(left_w, 0), imgui.ChildFlags_.borders)
    imgui.separator_text("Live Camera & Jog")
    _draw_live_preview(state, controller)
    if controller.robot.connected:
        _draw_jog_pad(state, controller)
    imgui.end_child()

    imgui.same_line()
    imgui.begin_child("ur5_vp_path", imgui.ImVec2(0, 0), imgui.ChildFlags_.borders)
    imgui.separator_text("Checked Capture Path")
    plan = controller.plan
    if plan is not None:
        runner = controller.runner
        active = runner.progress.target_index if runner is not None and runner.running else controller.rehearse_index
        _draw_viewpoint_preview(plan, active, float(state.get("teleop_max_rise", 0.19)) if plan.pattern == "orbit" else None)
        _draw_viewpoint_rows(plan, active)
    else:
        imgui.text_disabled("Press 'Check path' in Step 3 / Capture to preview viewpoints.")
    imgui.end_child()
    imgui.pop_id()


def _draw_simple_page(state, controller, renderer):
    """The whole job on one page: connect, put the camera over the object, scan, reconstruct. Each
    step is a header that opens when it is the one to do; everything else keeps its default."""
    robot = controller.robot
    camera_on = controller.camera is not None and controller.camera.running
    runner = controller.runner
    shown = gui_captures.gallery(controller)
    have_pictures = bool(runner is not None and runner.capture_records) or bool(shown.records)

    def header(number, title, done, open_now):
        tick = "[x]" if done else "[ ]"
        return imgui.collapsing_header(f"{tick} {number}  {title}##simple_{number}",
                                       imgui.TreeNodeFlags_.default_open if open_now else imgui.TreeNodeFlags_.none)

    timings = {}
    clock = time.perf_counter()

    def lap(name):
        nonlocal clock
        now = time.perf_counter()
        timings[name] = (now - clock) * 1000.0
        clock = now

    connected = robot.connected
    if header(1, "Connect the arm and the camera", connected and camera_on, not (connected and camera_on)):
        _draw_connection(state, controller)
        labels = [str(device["label"]) for device in controller.devices] or ["(none)"]
        if imgui.button("Find cameras##simple_camera_scan"):
            controller.scan_devices()
        selected = int(np.clip(int(state.get("ur5_camera_device", 0)), 0, len(labels) - 1))
        changed_device, selected = imgui.combo("Camera##simple_camera_device", selected, labels)
        if changed_device:
            state.ur5_camera_device = int(selected)
        if camera_on:
            imgui.text_colored(COLOR_OK, controller.camera.describe())
            if imgui.button("Stop camera##simple_camera_stop", (-1, 0)):
                controller.stop_camera()
        else:
            _draw_camera_start(state, controller, "simple")

    lap("connect")
    positioned = connected and camera_on
    if header(2, "Put the camera over the object", have_pictures, positioned and not have_pictures):
        if not positioned:
            imgui.text_colored(COLOR_MUTED, "Connect the arm and start the camera first.")
        else:
            _draw_live_preview(state, controller)
            _help("Jog until the object is under the crosshair and sharp. The scan is aimed at the spot under the crosshair, "
                  "at the height the camera is at now.")
            _draw_jog_pad(state, controller)
            _draw_capture_height(state, controller, str(state.get("teleop_pattern", "splat")))

    lap("position")
    if header(3, "Scan", have_pictures, positioned and not have_pictures):
        patterns = list(PATTERN_FIELDS)
        pattern = str(state.get("teleop_pattern", "splat"))
        if pattern not in patterns:
            pattern = "splat"
        changed_pattern, chosen = imgui.combo("Pattern##simple_pattern", patterns.index(pattern), patterns)
        if changed_pattern:
            state.teleop_pattern = patterns[chosen]
            pattern = patterns[chosen]
            controller.plan = None
        _help("splat: a dome of views round one spot, for Gaussian Splatting. grid / sweep: every part of a cloth from straight above. "
              "Their settings are under Expert; the defaults suit a small object at 5-8 cm.")
        if pattern == "splat":
            outlined = len(state.get("teleop_fabric_corners", [])) >= 3
            if any(_draw_field(state, key) for key in ("teleop_max_tilt", "teleop_splat_views") if not (outlined and key in GRID_BLOCK_FIELDS)):
                controller.replan()
        _draw_capture_controls(state, controller)
        if controller.plan_error:
            imgui.text_colored(COLOR_DANGER, controller.plan_error)
        if runner is not None and runner.running:
            imgui.text_colored(COLOR_OK, "Running - progress and Stop are in the bar above.")
        elif controller.plan is not None:
            imgui.text_colored(COLOR_MUTED, f"Path checked: {len(controller.plan.viewpoints)} views. Rehearse flies it without taking pictures; Run takes them.")

    lap("scan")
    if header(4, "Review pictures & open in Splatting Mode", False, have_pictures):
        gui_captures.draw_captures_section(state, controller, "Pictures")
        lap("pictures")
        imgui.push_style_color(imgui.Col_.button, (0.18, 0.40, 0.65, 1.0))
        imgui.push_style_color(imgui.Col_.button_hovered, (0.24, 0.50, 0.78, 1.0))
        if imgui.button("Open in Splatting Mode (Export, Pose Solve & 3DGS)##simple_to_splat", (-1, 36)):
            state.request_app_mode = "model"
        imgui.pop_style_color(2)
    lap("reconstruct")
    # What the frame costs, so slowness has a name: the app's frame rate and this panel's share.
    heavy = ", ".join(f"{name} {ms:.0f}" for name, ms in sorted(timings.items(), key=lambda item: -item[1]) if ms >= 1.0)
    imgui.text_colored(COLOR_MUTED, f"{imgui.get_io().framerate:.0f} fps - this panel {sum(timings.values()):.0f} ms"
                                    + (f" ({heavy} ms)" if heavy else ""))


def _draw_safety_bar(state, controller):
    """Above every tab: the stop, pause and reset, where the arm stands, the last message, and a run's progress.

    The stop is always enabled, even when disconnected. A control that is sometimes greyed out
    is one the user has to think about before pressing, and this is the one control that must
    never require thought."""
    robot = controller.robot
    runner = controller.runner
    imgui.push_style_color(imgui.Col_.button, (0.70, 0.12, 0.12, 1.0))
    imgui.push_style_color(imgui.Col_.button_hovered, (0.85, 0.18, 0.18, 1.0))
    if imgui.button("EMERGENCY STOP##ur5_estop", (-1, 42)):
        if runner is not None:
            runner.emergency_stop()
        else:
            robot.emergency_stop()
        controller.pending_start = ""
        controller.stop_rehearsal()
        controller.stop_centring()
        controller.status = robot.status
    imgui.pop_style_color(2)
    imgui.text_colored(COLOR_MUTED, "Software stop - not a substitute for the hardware emergency stop button.")

    third = max(70.0, (imgui.get_content_region_avail().x - 2.0 * imgui.get_style().item_spacing.x) / 3.0)
    paused = bool(robot.paused)
    if imgui.button(("Continue" if paused else "Pause") + "##ur5_pause", (third, 0)):
        if paused:
            controller.status = robot.status if not robot.resume() else "Continuing"
        else:
            if runner is not None and runner.running:
                runner.pause()
            else:
                robot.pause()
            controller.status = robot.status
    imgui.same_line()
    if imgui.button("Reset##ur5_reset", (third, 0)):
        robot.reset()
        controller.status = robot.status
    imgui.same_line()
    if imgui.button("Move to Home##ur5_home", (third, 0)):
        controller.move_home()
    if robot.stop_requested:
        imgui.text_colored(COLOR_DANGER, "A stop is latched. Press Reset to clear it.")

    ready, issues = robot.readiness()
    if not robot.connected:
        imgui.text_colored(COLOR_MUTED, "Arm: not connected - see 1 Setup")
    elif ready:
        imgui.text_colored(COLOR_OK, "Arm: ready to move")
    else:
        imgui.text_colored(COLOR_WARN, "Arm: not ready - " + "; ".join(issues))
    imgui.text_wrapped(controller.status)
    if runner is not None and runner.running:
        _draw_run_progress(runner, "bar")
    imgui.separator()


def _draw_setup_tab(state, controller, renderer):
    """Everything set once per session: how the arm is reached, the camera, how fast, and what was measured."""
    imgui.separator_text("Robot connection")
    _draw_connection(state, controller)
    imgui.separator_text("Camera")
    _draw_camera_setup(state, controller)
    imgui.separator_text("Speed")
    changed_speed, speed = imgui.slider_float("Speed limit##ur5_speed", float(state.get("ur5_speed_limit", 0.25)), 0.01, 1.0, "%.2f")
    if changed_speed:
        state.ur5_speed_limit = float(speed)
        controller.robot.apply_speed_limit(float(speed))
    imgui.text_colored(COLOR_MUTED, f"Commanded {controller.robot.limited_speed():.3f} m/s, {controller.robot.limited_acceleration():.2f} m/s^2")
    _draw_field(state, "teleop_speed_deg_s")
    _draw_field(state, "teleop_move_seconds")
    imgui.separator_text("Calibration")
    imgui.text_colored(COLOR_MUTED, "Lens and tool")
    _help("Tool offset: flange to camera along the tool axis. Lens FOV: horizontal, degrees (the mount calibration under 4 Results measures it too). Measured, not guessed, so a planned view and the camera are the same place.")
    for key in ("teleop_tool_offset", "ur5_splat_fov", "teleop_subject_distance"):
        _draw_field(state, key)
    busy = controller.centring
    if busy:
        imgui.begin_disabled()
    if imgui.button("Straighten Home (lens straight down, image square to the cloth)##ur5_home_straight", (-1, 0)):
        controller.straighten_home()
    if busy:
        imgui.end_disabled()
    _draw_sim_cell(controller, renderer, controller.runner is not None and controller.runner.running)
    imgui.separator_text("Display")
    changed_scene, show_scene = imgui.checkbox("Show the knit preview (3D View)##ur5_show_scene", bool(state.get("ur5_show_scene", False)))
    if changed_scene:
        state.ur5_show_scene = bool(show_scene)
    imgui.text_colored(COLOR_MUTED, "Off keeps the app light: the arm and the cell are TeleopApp's to show.")


def _draw_connection(state, controller):
    robot = controller.robot
    connected = robot.connected

    kind = _draw_transport_choice(state, connected)
    address_key = "teleop_address" if kind == "teleop" else "ur5_robot_ip"
    if kind != "sim":
        label = "TeleopApp address" if kind == "teleop" else "Robot IP"
        changed_address, address = imgui.input_text(f"{label}##ur5_address_{kind}", str(state.get(address_key, "")))
        if changed_address:
            state[address_key] = address

    if connected:
        if imgui.button("Disconnect##ur5_disconnect", (-1, 0)):
            controller.disconnect()
    else:
        target_addr = "sim" if kind == "sim" else str(state.get(address_key, ""))
        if imgui.button("Connect##ur5_connect", (-1, 0)):
            controller.connect(target_addr, kind)
        blocking = controller.blocking_program
        if blocking is not None:
            imgui.text_colored(COLOR_WARN, blocking.describe()[0].upper() + blocking.describe()[1:])
            imgui.text_wrapped("Make sure TeleopApp is really closed first (check Task Manager for TeleopApp.exe) - "
                               "a hidden one would lose the arm mid-command. Stopping brings the arm to rest.")
            if imgui.button("Stop that program, then connect##ur5_stop_and_connect", (-1, 0)):
                controller.stop_program_and_connect(target_addr, kind)

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
        _draw_teleop_status(robot.transport)

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


def _draw_transport_choice(state, connected):
    """The one radio that decides which arm, if any, this panel commands."""
    reachable = {name: transport.available() for name, transport in TRANSPORTS.items()}
    kind = str(state.get("ur5_transport", "teleop"))
    if not reachable.get(kind, False):
        kind = next((name for name in TRANSPORTS if reachable[name]), "teleop")
        state.ur5_transport = kind

    if connected:
        imgui.begin_disabled()
    for name, transport in TRANSPORTS.items():
        if imgui.radio_button(f"{transport.label}##ur5_transport_{name}", kind == name) and reachable[name]:
            state.ur5_transport = name
            kind = name
        if not reachable[name]:
            imgui.same_line()
            imgui.text_colored(COLOR_MUTED, "(unavailable)")
        if kind == name:
            imgui.text_wrapped(transport.explain)
    if connected:
        imgui.end_disabled()
        imgui.text_colored(COLOR_MUTED, "Disconnect to change how this panel reaches the arm.")
    return kind


def _draw_teleop_status(transport):
    """The flags that decide whether a command will do anything, for the one transport that has them.

    `following` is read from TeleopApp's `commandSending`, not from its tracking mode: when
    its deviation guard trips it stops commanding but still reports the mode as tracking,
    and a panel that reads the mode claims to be armed while the arm stands still.
    """
    if not hasattr(transport, "following"):
        return

    imgui.text("Following:")
    imgui.same_line()
    if transport.following:
        imgui.text_colored(COLOR_OK, "yes - commands reach the arm")
    else:
        imgui.text_colored(COLOR_WARN, "no - TeleopApp is not commanding the arm")
    gap = transport.gap_deg
    imgui.text_colored(COLOR_MUTED, f"Tracking gap {gap:.1f} deg" if gap is not None else "Tracking gap not evaluable")
    if transport.pose_names:
        imgui.text_colored(COLOR_MUTED, "Taught poses: " + ", ".join(transport.pose_names))


def _draw_camera_setup(state, controller):
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
        _draw_camera_start(state, controller, "setup")

    camera = controller.camera
    if camera is not None and camera.running:
        imgui.text_colored(COLOR_OK if camera.is_real else COLOR_WARN, camera.describe())
        if not camera.is_real:
            imgui.text_colored(COLOR_WARN, "Synthetic frames - not a real camera")

    changed_light, lighting = imgui.input_text(
        "Lighting condition##ur5_lighting", str(state.get("ur5_lighting_condition", ""))
    )
    if changed_light:
        state.ur5_lighting_condition = lighting


def _draw_camera_start(state, controller, where):
    if imgui.button(f"Start live preview##ur5_camera_start_{where}", (-1, 0)):
        selected = int(state.get("ur5_camera_device", 0))
        spec = controller.devices[selected] if 0 <= selected < len(controller.devices) else {}
        controller.start_camera(spec, clamp_capture_size(state.get("ur5_capture_width", 1280)))


def _draw_focus_controls(state, controller):
    """Focus by hand for close-ups, with the frame's sharpness to judge it by, and what a run does about blur."""
    changed_show, show = imgui.checkbox("Show fabric detection on the preview##ur5_show_detection", bool(state.get("ur5_show_detection", True)))
    if changed_show:
        state.ur5_show_detection = bool(show)
    if controller.live_fabric_share is not None and bool(state.get("ur5_show_detection", True)):
        imgui.text_colored(COLOR_OK if controller.live_fabric_share >= float(state.get("ur5_min_coverage", 0.8)) else COLOR_WARN,
                           f"Fabric in view: {controller.live_fabric_share * 100:.0f}% (bright = knit, dimmed = background, + = its middle)")

    sharp = controller.live_sharpness
    threshold = float(state.get("ur5_min_sharpness", 150.0))
    if sharp is not None:
        imgui.text_colored(COLOR_OK if sharp >= threshold else COLOR_WARN, f"Sharpness now: {sharp:.0f}"
                           + ("" if sharp >= threshold else f" - below the retake threshold {threshold:.0f}"))

    busy = controller.centring or (controller.runner is not None and controller.runner.running)
    if busy:
        imgui.begin_disabled()
    if imgui.button("Find best focus (sweeps the lens, then locks it)##ur5_find_focus", (-1, 0)):
        controller.find_focus()
    if busy:
        imgui.end_disabled()
    imgui.text_colored(COLOR_MUTED, "Jog to the capture height first: focus is found for the distance the camera is at.")
    if imgui.button("Camera's own settings (focus, exposure, sharpness - as the driver offers them)##ur5_camera_settings", (-1, 0)):
        controller.camera.request_settings()
    imgui.text_colored(COLOR_MUTED, "No Focus slider in that window (Camera Control tab) means a fixed-focus camera.")
    changed_auto, auto = imgui.checkbox("Autofocus##ur5_autofocus", bool(state.get("ur5_camera_autofocus", True)))
    if changed_auto:
        state.ur5_camera_autofocus = bool(auto)
        controller.apply_focus()
    if not bool(state.get("ur5_camera_autofocus", True)):
        changed_focus, focus = imgui.slider_float("Focus##ur5_focus", float(state.get("ur5_camera_focus", 0.0)), 0.0, 255.0, "%.0f")
        if changed_focus:
            state.ur5_camera_focus = float(focus)
            controller.apply_focus()
        imgui.text_colored(COLOR_MUTED, "Jog the camera to the capture height, then turn this until Sharpness peaks.")


def _draw_quality_controls(state, controller):
    """What a run does about a blurred frame or one that is not all fabric."""
    _draw_field(state, "ur5_min_sharpness")
    imgui.text_colored(COLOR_MUTED, "A capture below this is retaken up to twice; the sharpest is kept. 0 turns it off.")

    changed_check, check = imgui.checkbox("Fabric check (MobileSAM)##ur5_fabric_check", bool(state.get("ur5_fabric_check", True)))
    if changed_check:
        state.ur5_fabric_check = bool(check)
    ready = fabric_detector.available()
    imgui.text_colored(COLOR_OK if ready else COLOR_WARN, fabric_detector.describe())
    if not ready:
        imgui.begin_disabled()
    if imgui.button("Test fabric check on this frame##ur5_fabric_test", (-1, 0)):
        controller.test_fabric_check()
    if not ready:
        imgui.end_disabled()
    if bool(state.get("ur5_fabric_check", True)):
        _draw_field(state, "ur5_min_coverage")
        imgui.text_colored(COLOR_MUTED, "A capture with less fabric than this is flagged and left out of the colour analysis.")


def _image_turn(state):
    return state.get("ur5_camera_image_turn") or [[1.0, 0.0], [0.0, 1.0]]


def _measured_fabric_z(state):
    """The fabric plane as set - measured by the camera, a ruler or a touched corner; None while it is only a guess.

    One value, not the corners' mean: a corner marked under the crosshair lies on whatever plane was
    believed then, so setting the plane moves the corners onto it rather than the other way round."""
    known = state.get("teleop_fabric_z")
    return None if known is None else float(known)


def _settings_fabric_z(state, view):
    known = _measured_fabric_z(state)
    if known is not None:
        return known
    return scan_geometry.fabric_height(float(state.get("teleop_subject_distance", 0.20)))


def _fabric_check(state):
    """MobileSAM when it is fetched, else the texture test - so every capture is judged either way."""
    if not bool(state.get("ur5_fabric_check", True)):
        return None
    if fabric_detector.available():
        return fabric_detector.coverage

    def by_texture(image):
        mask = fabric_detector.texture_mask(image)
        return float(mask.mean()), mask
    return by_texture


def _fabric_region(state):
    corners = state.get("teleop_fabric_corners", [])
    return scan_geometry.FabricRegion(np.asarray(corners, dtype=float)) if len(corners) else None


def _capture_settings(state) -> dict:
    """The panel's fields as build_capture_plan's arguments, which is where the two meet."""
    corners = [list(corner) for corner in state.get("teleop_fabric_corners", [])]
    return dict(
        pattern           = str(state.get("teleop_pattern", "orbit")),
        count             = int(state.get("teleop_angles", 6)),
        subject_distance  = float(state.get("teleop_subject_distance", 0.20)),
        orbit_radius      = float(state.get("teleop_orbit_radius", 0.10)),
        max_tilt_deg      = float(state.get("teleop_max_tilt", 35.0)),
        image_turn        = _image_turn(state),
        tool_offset       = float(state.get("teleop_tool_offset", 0.0306)),
        arc_degrees       = float(state.get("teleop_arc", 240.0)),
        elevation_degrees = float(state.get("teleop_elevation", 75.0)),
        rings             = int(state.get("teleop_rings", 3)),
        max_height        = float(state.get("teleop_max_rise", 0.19)),
        view_radius       = float(state.get("teleop_view_radius", 0.06)),
        grid_rows         = int(state.get("teleop_grid_rows", 3)),
        grid_cols         = int(state.get("teleop_grid_cols", 3)),
        camera_height     = float(state.get("teleop_camera_height", 0.07)),
        taught            = list(state.get("teleop_taught_views", [])),
        fabric_z          = _measured_fabric_z(state),
        fabric_outline    = corners,
        lens_hfov_deg     = float(state.get("ur5_splat_fov", gaussian_splatting.DEFAULT_FOV_DEGREES)),
        grid_overlap      = float(state.get("teleop_grid_overlap", 0.3)),
        sweep_overlap     = float(state.get("teleop_sweep_overlap", 0.65)),
        grid_views_per_station = int(state.get("teleop_grid_views", 1)),
        grid_height_passes = int(state.get("teleop_grid_passes", 3)),
        grid_overview_height = state.get("teleop_overview_height"),
        grid_overview_joints = state.get("teleop_overview_joints"),
        splat_views_per_ring = int(state.get("teleop_splat_views", 12)),
        splat_layers      = int(state.get("teleop_splat_layers", 2)))


def _draw_field(state, key):
    """One row of FIELD_SPECS, as an int or a float control according to its bounds; True once an edit to it is finished."""
    label, low, high, fmt = FIELD_SPECS[key]
    if isinstance(low, int):
        changed, value = imgui.slider_int(f"{label}##{key}", int(state.get(key, low)), low, high, fmt)
    else:
        changed, value = imgui.slider_float(f"{label}##{key}", float(state.get(key, low)), low, high, fmt)
    if changed:
        state[key] = type(low)(value)
    return imgui.is_item_deactivated_after_edit()


def _draw_capture_tab(state, controller, renderer):
    """What to photograph and how: the height, the pattern and its few settings, then Check, Rehearse, Run."""
    runner = controller.runner
    pattern = str(state.get("teleop_pattern", "orbit"))

    patterns = list(PATTERN_FIELDS)
    if pattern not in patterns:
        pattern = patterns[0]
    changed_pattern, chosen = imgui.combo("Pattern##teleop_pattern", patterns.index(pattern), patterns)
    if changed_pattern:
        state.teleop_pattern = patterns[chosen]
        pattern = patterns[chosen]
        # The old plan described the old pattern; keeping it on screen invites running it.
        controller.plan = None
    imgui.text_colored(COLOR_MUTED, PATTERN_BLURBS[pattern])
    _draw_capture_height(state, controller, pattern)
    imgui.separator_text("Pattern settings")

    outlined = len(state.get("teleop_fabric_corners", [])) >= 3
    edited = [_draw_field(state, key) for key in PATTERN_FIELDS[pattern] if not (outlined and key in GRID_BLOCK_FIELDS)]
    if any(edited):
        controller.replan()
    if pattern in ("grid", "sweep"):
        imgui.text_colored(COLOR_MUTED, ("Rows and columns follow from the fabric outline; " if outlined else "")
                           + "frames are spaced one frame less the overlap, so they spread with the capture height.")
    if pattern == "taught":
        _draw_teacher(state, controller)
    if pattern == "splat":
        layers, per_ring, rings = (int(state.get(key, default)) for key, default in
                                   (("teleop_splat_layers", 2), ("teleop_splat_views", 12), ("teleop_rings", 3)))
        radius = float(state.get("teleop_camera_height", 0.07))
        imgui.text_colored(COLOR_MUTED, f"{layers * (1 + rings * per_ring)} views asked for, from "
                           + ", ".join(f"{radius * (1.0 + scan_geometry.SPLAT_LAYER_STEP * layer) * 100:.1f}" for layer in range(layers))
                           + " cm. Any the arm cannot reach safely is moved to the nearest safe one, or skipped - Check lists them.")
        imgui.text_colored(COLOR_MUTED, "Soft-focus camera?")
        _help("Jog to the distance it is sharpest at and press Use the camera's height now; keep the retake threshold low.")
    imgui.separator_text("Images")
    _draw_dwell_controls(state, "capture")
    _draw_quality_controls(state, controller)

    imgui.separator_text("Check, rehearse, run")
    _draw_capture_controls(state, controller)

    if controller.plan_error:
        imgui.text_colored(COLOR_DANGER, controller.plan_error)
    plan = controller.plan
    if plan is None:
        imgui.text_disabled("Check the path to solve it and see it here.")
        return

    active = runner.progress.target_index if runner is not None and runner.running else controller.rehearse_index
    _draw_viewpoint_preview(plan, active, float(state.get("teleop_max_rise", 0.19)) if plan.pattern == "orbit" else None)
    _draw_viewpoint_rows(plan, active)


def _draw_capture_height(state, controller, pattern):
    """The capture height (Z): set it, see what a frame takes in from it, fly to it, or teach it by hand."""
    imgui.separator_text("Capture height")
    height = float(state.get("teleop_camera_height", 0.07))
    robot = controller.robot
    busy = controller.centring or (controller.runner is not None and controller.runner.running)
    width = imgui.get_content_region_avail().x
    nudge = max(30.0, imgui.get_frame_height() * 1.6)

    if imgui.button("-##capture_height_less", (nudge, 0)):
        controller.set_capture_height(height - CAPTURE_HEIGHT_NUDGE)
    imgui.same_line()
    imgui.set_next_item_width(max(80.0, width - 2.0 * (nudge + imgui.get_style().item_spacing.x)))
    changed, chosen_cm = imgui.slider_float("##capture_height_cm", height * 100.0, CAPTURE_HEIGHT_MIN * 100.0,
                                            CAPTURE_HEIGHT_MAX * 100.0, "%.1f cm over the fabric")
    if changed:
        controller.set_capture_height(round(chosen_cm * 2.0) / 200.0)
    imgui.same_line()
    if imgui.button("+##capture_height_more", (nudge, 0)):
        controller.set_capture_height(height + CAPTURE_HEIGHT_NUDGE)
    height = float(state.get("teleop_camera_height", 0.07))

    frame = None if controller.camera is None else controller.camera.latest_frame()
    pixels_x, pixels_y = frame.size if frame is not None else clamp_capture_size(state.get("ur5_capture_width", 1280))
    across = scan_geometry.frame_side(height, float(state.get("ur5_splat_fov", gaussian_splatting.DEFAULT_FOV_DEGREES)))
    imgui.text(f"One frame covers {across * 100:.1f} x {across * pixels_y / pixels_x * 100:.1f} cm of fabric, "
               f"{across * 1000.0 / pixels_x:.2f} mm per pixel")
    imgui.text_colored(COLOR_OK if pattern in HEIGHT_USE else COLOR_MUTED,
                       HEIGHT_USE.get(pattern, f"The {pattern} pattern does not use it; it still sets Move over fabric."))

    if robot.connected:
        try:
            now = controller.camera_height_over_fabric(robot.state.joints)
            imgui.text_colored(COLOR_OK if abs(now - height) < 0.003 else COLOR_MUTED, f"Camera now: {now * 100:.1f} cm over the fabric")
        except Exception as exc:
            imgui.text_colored(COLOR_WARN, f"Camera height unknown: {exc}")
    if _measured_fabric_z(state) is None and not controller.fabric_gap_cm:
        imgui.text_colored(COLOR_WARN, "Fabric height not set - type the lens-to-cloth gap under 2 Position > Fabric first.")

    half = max(90.0, (width - imgui.get_style().item_spacing.x) * 0.5)
    if not robot.connected or busy:
        imgui.begin_disabled()
    if imgui.button("Use the camera's height now##capture_height_teach", (half, 0)):
        controller.adopt_camera_height()
    imgui.same_line()
    if robot.free_driving:
        imgui.begin_disabled()
    if imgui.button("Go to this height##capture_height_go", (half, 0)):
        controller.move_over_fabric()
    if robot.free_driving:
        imgui.end_disabled()
    if imgui.button("Use the camera's height now as the overview##capture_height_overview", (half, 0)):
        controller.adopt_camera_height(overview=True)
    imgui.same_line()
    if robot.free_driving:
        imgui.begin_disabled()
    if imgui.button("Go to the overview##capture_height_overview_go", (half, 0)):
        controller.move_to_overview()
    if robot.free_driving:
        imgui.end_disabled()
    if not robot.connected or busy:
        imgui.end_disabled()
    imgui.text_colored(COLOR_MUTED, "Jog the camera (2 Position) until the preview is sharp, then Use the camera's height now: "
                                    "every capture is taken at exactly that height, so the focus holds.")
    overview = state.get("teleop_overview_height")
    passes = int(state.get("teleop_grid_passes", 3))
    imgui.text_colored(COLOR_MUTED, "Overview height not set: each pass is 1.6x higher than the last." if not overview else
                       f"Overview {float(overview) * 100:.0f} cm: {passes} passes from {height * 100:.1f} cm up to it, "
                       "the last one a single view of the whole cloth." if passes > 1 else
                       "Overview set, but it needs 2 or more Height passes to be used.")


def _draw_fabric_gap(state, controller):
    """With no corners, the fabric's height is a ruler reading, lens to cloth - or, until one is given, a guess."""
    ruler = state.get("teleop_fabric_z")
    if ruler is None:
        imgui.text_colored(COLOR_WARN, "Fabric height: NOT SET - a guess from Home, which stops the camera going lower.")
    else:
        imgui.text_colored(COLOR_OK, f"Fabric height: set (z = {float(ruler):+.3f} m) - kept until you set it again")
        imgui.same_line()
        if imgui.small_button("Forget it##fabric_forget"):
            controller.forget_fabric_height()
    busy = controller.centring or (controller.runner is not None and controller.runner.running)
    camera_on = controller.camera is not None and controller.camera.running
    if not controller.robot.connected or busy or not camera_on:
        imgui.begin_disabled()
    if imgui.button(("Measure fabric height with the camera" if ruler is None else "Measure it again")
                    + " (slides 3 mm and back)##fabric_measure", (-1, 0)):
        controller.measure_fabric_height()
    if not controller.robot.connected or busy or not camera_on:
        imgui.end_disabled()
    imgui.text_colored(COLOR_MUTED, "Or measure lens to cloth with a ruler, type it here and Set:")
    blocked = not controller.robot.connected
    if blocked:
        imgui.begin_disabled()
    imgui.set_next_item_width(max(80.0, imgui.get_content_region_avail().x * 0.4))
    changed, gap_cm = imgui.input_float("cm lens to cloth##fabric_gap_cm", float(controller.fabric_gap_cm or 0.0), 0.5, 5.0, "%.1f")
    if changed:
        controller.fabric_gap_cm = max(0.0, gap_cm)
    imgui.same_line()
    if imgui.button("Set fabric height##fabric_gap_set") and controller.fabric_gap_cm:
        controller.set_fabric_gap(controller.fabric_gap_cm / 100.0)
    if blocked:
        imgui.end_disabled()


def _draw_position_tab(state, controller, renderer):
    """Put the camera where it should be from the panel alone: watch the preview, focus, jog, mark the cloth."""
    _draw_live_preview(state, controller)
    if controller.camera is not None and controller.camera.running:
        imgui.separator_text("Focus")
        _draw_focus_controls(state, controller)
    imgui.separator_text("Move the camera")
    _draw_jog_pad(state, controller)
    imgui.separator_text("Fabric")
    _draw_fabric(state, controller)
    if imgui.tree_node("Advanced: guide the arm by hand (free drive)##ur5_hand"):
        _draw_hand_guiding(state, controller)
        imgui.tree_pop()


def _draw_live_preview(state, controller):
    """The camera's view with a crosshair on its centre - the point the jog pad and Mark corner work from.
    Returns where it was drawn, (x, y, w, h), for a caller to mark on it; None while there is no picture."""
    if controller.camera is None or not controller.camera.running:
        imgui.text_colored(COLOR_MUTED, "The camera is off.")
        _draw_camera_start(state, controller, "position")
        return None
    if controller.preview_texture is None:
        imgui.text_disabled("Waiting for the first frame...")
        return None
    avail_w = max(120, int(imgui.get_content_region_avail().x))
    width, height = controller._preview_size
    shown = draw_fitted_texture(controller.preview_texture.glo, width, height, avail_w, int(min(avail_w * height / max(width, 1), 480)), flip_y=True)
    if shown is not None:
        x, y, w, h = shown
        mark = imgui.get_color_u32((1.0, 0.85, 0.2, 0.9))
        draw_list = imgui.get_window_draw_list()
        draw_list.add_line(imgui.ImVec2(x + w * 0.5 - 14.0, y + h * 0.5), imgui.ImVec2(x + w * 0.5 + 14.0, y + h * 0.5), mark, 1.5)
        draw_list.add_line(imgui.ImVec2(x + w * 0.5, y + h * 0.5 - 14.0), imgui.ImVec2(x + w * 0.5, y + h * 0.5 + 14.0), mark, 1.5)
    changed, detail = imgui.checkbox("Zoom 2x: the centre at full resolution, to judge focus##ur5_preview_detail",
                                     bool(state.get("ur5_preview_detail", False)))
    if changed:
        state.ur5_preview_detail = bool(detail)
    robot = controller.robot
    if robot.connected:
        try:
            imgui.text(f"Camera now: {controller.camera_height_over_fabric(robot.state.joints) * 100:.1f} cm over the fabric")
        except Exception as exc:
            imgui.text_colored(COLOR_WARN, f"Camera height unknown: {exc}")
    return shown


def _draw_jog_pad(state, controller):
    """Step the camera as seen in the preview, and up or down. Every step is solved and collision checked."""
    robot = controller.robot
    busy = controller.centring or (controller.runner is not None and controller.runner.running)
    step = float(state.get("ur5_jog_step", 0.01))
    labels = [f"{value * 1000:.0f} mm" if value < 0.01 else f"{value * 100:.0f} cm" for value in JOG_STEPS]
    index = min(range(len(JOG_STEPS)), key=lambda i: abs(JOG_STEPS[i] - step))
    changed, index = imgui.combo("Step##ur5_jog_step", index, labels)
    if changed:
        state.ur5_jog_step = JOG_STEPS[index]
    step = JOG_STEPS[index]

    blocked = not robot.connected or busy or robot.free_driving
    if blocked:
        imgui.begin_disabled()
    cell = max(70.0, (imgui.get_content_region_avail().x - 3.0 * imgui.get_style().item_spacing.x) / 4.0)
    for row in JOG_PAD:
        for column, (label, right, forward, up) in enumerate(row):
            if column:
                imgui.same_line()
            if not label:
                imgui.dummy(imgui.ImVec2(cell, 0))
            elif imgui.button(f"{label}##ur5_jog_{label}", (cell, 0)):
                controller.jog(right * step, forward * step, up * step)
    if blocked:
        imgui.end_disabled()
    if controller.centring:
        imgui.text_colored(COLOR_WARN, "Moving...")
        if imgui.button("Stop moving##ur5_jog_stop", (-1, 0)):
            controller.stop_centring()
    elif not robot.connected:
        imgui.text_colored(COLOR_MUTED, "Connect under 1 Setup to move the camera.")
    else:
        imgui.text_colored(COLOR_MUTED, "Jog")
        _help("Left/right/forward/back are as seen in the preview (after Calibrate camera turn). Each move keeps the lens straight down.")


def _draw_fabric(state, controller):
    """Where the cloth is: its height (a ruler gap, or measured corners) and its outline (corners under the crosshair)."""
    robot = controller.robot
    busy = controller.centring or (controller.runner is not None and controller.runner.running)
    corners = list(state.get("teleop_fabric_corners", []))
    _draw_fabric_gap(state, controller)
    if corners:
        region = scan_geometry.FabricRegion(np.asarray(corners, dtype=float))
        imgui.text_colored(COLOR_OK, f"Outline: {len(corners)} corner(s) marked")
        if region.has_outline:
            hull = region.outline
            area = 0.5 * abs(float(np.dot(hull[:, 0], np.roll(hull[:, 1], -1)) - np.dot(hull[:, 1], np.roll(hull[:, 0], -1))))
            imgui.text_colored(COLOR_OK, f"Outline set, {area * 1e4:.0f} cm^2 - automatic frames stay on the cloth")
        else:
            imgui.text_colored(COLOR_MUTED, f"Mark {3 - len(corners)} more for an outline.")

    half = max(90.0, (imgui.get_content_region_avail().x - imgui.get_style().item_spacing.x) * 0.5)
    imgui.text_colored(COLOR_MUTED, "Outline (optional)")
    _help("Jog until a fabric corner is under the crosshair, then Mark; repeat round the cloth.")
    if not robot.connected or busy:
        imgui.begin_disabled()
    if imgui.button("Mark corner under crosshair##fabric_corner_cross", (half, 0)):
        controller.mark_corner_under_crosshair()
    if not robot.connected or busy:
        imgui.end_disabled()
    imgui.same_line()
    if not corners:
        imgui.begin_disabled()
    if imgui.button("Clear corners##fabric_clear", (half, 0)):
        controller.clear_corners()
    if not corners:
        imgui.end_disabled()

    held = not robot.connected or busy or robot.free_driving
    if held:
        imgui.begin_disabled()
    if imgui.button("Move over fabric##fabric_over", (half, 0)):
        controller.move_over_fabric()
    imgui.same_line()
    if imgui.button("Find its centre (camera)##fabric_centre", (half, 0)):
        controller.centre_on_fabric()
    if imgui.button("Outline the cloth from the preview (camera straight down, whole cloth in view)##fabric_outline_auto", (-1, 0)):
        controller.outline_from_preview()
    if imgui.button("Calibrate camera turn (makes the jog and straight views match the preview)##camera_turn", (-1, 0)):
        controller.calibrate_image_turn()
    if held:
        imgui.end_disabled()


def _draw_hand_guiding(state, controller):
    """Free drive, for whoever prefers to pose the arm by hand - and Mark corner with the lens resting on the cloth."""
    robot = controller.robot
    blocked = not robot.connected or (controller.runner is not None and controller.runner.running)
    if blocked:
        imgui.begin_disabled()
    if robot.free_driving:
        imgui.push_style_color(imgui.Col_.button, (0.20, 0.55, 0.25, 1.0))
        if imgui.button("Hold (end free drive)##teach_hold", (-1, 0)):
            controller.hold()
        imgui.pop_style_color()
    elif imgui.button("Free drive##teach_free", (-1, 0)):
        controller.free_drive()
    if imgui.button("Mark corner with the lens resting on it##fabric_corner", (-1, 0)):
        controller.mark_corner()
    if blocked:
        imgui.end_disabled()
    if robot.free_driving:
        imgui.text_colored(COLOR_WARN, "Free drive: the arm is loose. Hold it while you pose it, then Hold, then Reset to arm it.")
    imgui.text_colored(COLOR_MUTED, "A corner marked by touch measures the fabric's height too, so no ruler gap is needed.")


def _draw_sim_cell(controller, renderer, running):
    """TeleopApp's emulator as the simulator: a table and the knitted fabric in its scene."""
    transport = controller.robot.transport
    if not hasattr(transport, "cameras") or not transport.linked() or transport.on_hardware:
        return
    imgui.text_colored(COLOR_OK, "Simulator: TeleopApp is on its emulator")
    _help("Builds a table and the knitted fabric under Home in TeleopApp's 3D cell, and marks its corners.")
    half = max(90.0, (imgui.get_content_region_avail().x - imgui.get_style().item_spacing.x) * 0.5)
    if running:
        imgui.begin_disabled()
    if imgui.button("Build table + knitted fabric##sim_build", (half, 0)):
        controller.build_sim_cell(renderer)
    imgui.same_line()
    if imgui.button("Remove##sim_remove", (half, 0)):
        controller.remove_sim_cell()
    if running:
        imgui.end_disabled()
    imgui.separator()


def _draw_teacher(state, controller):
    """Record the camera where the hand left it, the list of what has been taught, and the sets on disk."""
    robot = controller.robot
    running = controller.runner is not None and controller.runner.running

    imgui.separator_text("Taught views")
    imgui.text_colored(COLOR_MUTED, "Jog the camera to each view you want (2 Position), then Record view here.")
    if not robot.connected or running:
        imgui.begin_disabled()
    if imgui.button("Record view##teach_record", (-1, 0)):
        controller.record_view()
    if not robot.connected or running:
        imgui.end_disabled()

    views = list(state.get("teleop_taught_views", []))
    if not views:
        imgui.text_disabled("No views taught yet.")
    for index, view in enumerate(views):
        if imgui.small_button(f"x##teach_forget_{index}"):
            controller.forget_view(index)
            break
        imgui.same_line()
        if imgui.small_button(f"use Z##teach_height_{index}"):
            controller.adopt_camera_height(view["joints"])
        if imgui.is_item_hovered():
            imgui.set_tooltip("Capture every frame at this view's height (unmarked fabric: type the lens-to-cloth gap at this view first)")
        imgui.same_line()
        try:
            over = f"   {controller.camera_height_over_fabric(view['joints']) * 100:+.1f} cm over the fabric"
        except Exception:
            over = ""
        imgui.text(f"{index + 1:>2}. " + "  ".join(f"{np.degrees(value):+6.1f}" for value in view["joints"]) + over)
    if views and imgui.button("Clear all taught views##teach_clear", (-1, 0)):
        controller._set_taught_views([])

    changed_name, name = imgui.input_text("Set name##teach_set_name", str(state.get("teleop_taught_set", "default")))
    if changed_name:
        state.teleop_taught_set = name
    if imgui.button("Save set##teach_save", (-1, 0)):
        controller.save_taught_set(str(state.get("teleop_taught_set", "")))
        gui_captures.every("taught sets", taught_set_names, ttl=0.0)

    names = gui_captures.every("taught sets", taught_set_names)
    if not names:
        return
    current = str(state.get("teleop_taught_set", ""))
    changed_set, chosen = imgui.combo("Saved sets##teach_saved", names.index(current) if current in names else 0, names)
    if changed_set:
        state.teleop_taught_set = names[chosen]
    if imgui.button("Load set##teach_load", (-1, 0)):
        controller.load_taught_set(names[chosen])


def _draw_capture_controls(state, controller):
    """Check, rehearse, run - in that order, because each one earns the next."""
    robot = controller.robot
    third = max(70.0, (imgui.get_content_region_avail().x - 2.0 * imgui.get_style().item_spacing.x) / 3.0)

    planning = controller.planning
    if planning:
        imgui.begin_disabled()
    if imgui.button("Checking...##teleop_check" if planning else "Check##teleop_check", (third, 0)):
        controller.check_in_background()
    if planning:
        imgui.end_disabled()
    imgui.same_line()

    can_rehearse = (controller.plan is not None and robot.connected and not robot.free_driving
                    and robot.transport.rehearsable() and not controller.rehearsing)
    if not can_rehearse:
        imgui.begin_disabled()
    if imgui.button("Rehearse##teleop_rehearse", (third, 0)):
        controller.rehearse()
    if not can_rehearse:
        imgui.end_disabled()
    imgui.same_line()

    camera_ready = _camera_ready(state, controller)
    can_run = controller.plan is not None and controller.plan.clear and robot.ready() and camera_ready and not controller.centring
    if not can_run:
        imgui.begin_disabled()
    if imgui.button("Run##teleop_run", (third, 0)):
        controller.pending_start = "capture"
    if not can_run:
        imgui.end_disabled()

    if controller.rehearsing:
        imgui.text_colored(COLOR_WARN, "Rehearsing on the virtual robot - the arm is not following")
        if imgui.button("Stop rehearsal##teleop_rehearse_stop", (-1, 0)):
            controller.stop_rehearsal()
    elif not can_run:
        for reason in _capture_blockers(controller, camera_ready):
            imgui.text_colored(COLOR_MUTED, f"- {reason}")
    elif not bool(state.get("ur5_save_images", True)):
        imgui.text_colored(COLOR_WARN, "Motion only - the path runs and waits, and nothing is photographed")

    if controller.pending_start == "capture":
        _draw_start_confirmation(state, controller, "capture")



def _camera_ready(state, controller):
    """Whether the camera side is satisfied - which a run that saves nothing already is."""
    if not bool(state.get("ur5_save_images", True)):
        return True
    return controller.camera is not None and controller.camera.running


def _capture_blockers(controller, camera_ready):
    """Why Run is disabled, as the whole list of what to fix rather than the first item."""
    plan = controller.plan
    _, issues = controller.robot.readiness()
    return [*([] if plan is not None else ["check the path first"]),
            *([] if plan is None or plan.clear else [f"{len(plan.refused)} viewpoint(s) collide"]),
            *([] if camera_ready else ["start the camera preview, or turn off Save captured images to run the motion alone"]),
            *issues]


def _draw_viewpoint_preview(plan, active_index, max_rise):
    """The solved path as two flat views, drawn straight onto the window's draw list.

    Numbered, not just dotted: the order is most of what a capture path is, and a ring of
    identical circles says nothing about which way the arm will travel through it.
    """
    points = np.array([point.position for point in plan.viewpoints], dtype=float)
    subject = np.asarray(plan.subject, dtype=float)
    radial = np.linalg.norm(points[:, :2] - subject[:2], axis=1)

    panels = (("Top down (x, y)",    points[:, 0], points[:, 1], float(subject[0]), float(subject[1]), 150.0, None),
              ("Side (distance, z)", radial,       points[:, 2], 0.0,               float(subject[2]), 118.0, max_rise))
    for title, horizontal, vertical, anchor_h, anchor_v, height, rise in panels:
        cap = None if rise is None else anchor_v + float(rise)
        imgui.text_colored(COLOR_MUTED, title)
        width = max(150.0, imgui.get_content_region_avail().x)
        origin = imgui.get_cursor_screen_pos()
        imgui.invisible_button(f"##ur5_plan_{title}", imgui.ImVec2(width, height))
        draw_list = imgui.get_window_draw_list()
        draw_list.add_rect_filled(origin, imgui.ImVec2(origin.x + width, origin.y + height),
                                  imgui.get_color_u32((0.09, 0.10, 0.12, 1.0)))

        spread_h = np.append(horizontal, anchor_h)
        spread_v = np.append(vertical, [anchor_v] if cap is None else [anchor_v, cap])
        span_h = max(float(spread_h.max() - spread_h.min()), 0.02)
        span_v = max(float(spread_v.max() - spread_v.min()), 0.02)
        scale = min((width - 36.0) / span_h, (height - 36.0) / span_v)
        mid_h = 0.5 * float(spread_h.max() + spread_h.min())
        mid_v = 0.5 * float(spread_v.max() + spread_v.min())

        def place(across, up, origin=origin, width=width, height=height, scale=scale, mid_h=mid_h, mid_v=mid_v):
            return imgui.ImVec2(origin.x + width * 0.5 + (float(across) - mid_h) * scale,
                                origin.y + height * 0.5 - (float(up) - mid_v) * scale)

        if cap is not None:
            edge = place(anchor_h, cap).y
            warn = imgui.get_color_u32(COLOR_WARN)
            draw_list.add_line(imgui.ImVec2(origin.x, edge), imgui.ImVec2(origin.x + width, edge), warn, 1.0)
            draw_list.add_text(imgui.ImVec2(origin.x + 5.0, edge + 2.0), warn, "camera rise cap")

        subject_at = place(anchor_h, anchor_v)
        pale = imgui.get_color_u32((0.85, 0.85, 0.88, 0.90))
        draw_list.add_line(imgui.ImVec2(subject_at.x - 6.0, subject_at.y), imgui.ImVec2(subject_at.x + 6.0, subject_at.y), pale, 1.4)
        draw_list.add_line(imgui.ImVec2(subject_at.x, subject_at.y - 6.0), imgui.ImVec2(subject_at.x, subject_at.y + 6.0), pale, 1.4)

        travel = imgui.get_color_u32((0.55, 0.60, 0.70, 0.75))
        for index in range(1, len(plan.viewpoints)):
            draw_list.add_line(place(horizontal[index - 1], vertical[index - 1]),
                               place(horizontal[index], vertical[index]), travel, 1.2)

        for index, point in enumerate(plan.viewpoints):
            at = place(horizontal[index], vertical[index])
            fill = COLOR_DANGER if not point.clear else (COLOR_WARN if point.transition_deg > 90.0 else COLOR_OK)
            draw_list.add_circle_filled(at, 7.5, imgui.get_color_u32(fill))
            if index == active_index:
                draw_list.add_circle(at, 11.5, imgui.get_color_u32((1.0, 1.0, 1.0, 0.95)), 20, 2.0)
            draw_list.add_text(imgui.ImVec2(at.x - 3.0, at.y - 7.0), imgui.get_color_u32((0.06, 0.06, 0.08, 1.0)), str(index + 1))


def _draw_viewpoint_rows(plan, active_index):
    """One line per viewpoint: where it looks from, how far the arm swings, and the verdict."""
    for index, point in enumerate(plan.viewpoints):
        colour = COLOR_DANGER if not point.clear else (COLOR_WARN if point.transition_deg > 90.0 else COLOR_MUTED)
        imgui.text_colored(COLOR_OK if index == active_index else colour,
                           f"{'>' if index == active_index else ' '} {index + 1:>2}. {point.name:<14}"
                           f" {point.transition_deg:5.0f} deg  {point.err_pos * 1000:4.1f} mm"
                           f"  {point.clearance * 1000:4.0f} mm over fabric"
                           + (f"  (moved: {point.adjustment})" if point.adjustment else "")
                           + (f"  (frame {'above the horizon' if np.isinf(point.overshoot) else f'{point.overshoot * 1000:.0f} mm'} past the fabric edge)"
                              if point.overshoot > scan_geometry.FRAME_TOLERANCE and not point.frame_enforced else "")
                           + ("" if point.clear else "  " + "; ".join(point.problems)))
    for name, reason in plan.skipped:
        imgui.text_colored(COLOR_WARN, f"  skipped {name}: {reason}")


def _draw_dwell_controls(state, scope):
    """Dwell and image saving, drawn wherever a run starts - one pair of values either way.

    Both runs wait at a viewpoint for the same reason and save through the same storage, so
    they read the same two keys and only the widget id differs by where it is drawn.
    """
    changed_dwell, dwell = imgui.slider_float(
        f"Wait at each view (s)##ur5_dwell_{scope}", float(state.get("ur5_dwell", 0.2)), 0.0, 6.0, "%.2f"
    )
    if changed_dwell:
        state.ur5_dwell = float(dwell)
    changed_save, save_images = imgui.checkbox(
        f"Save captured images##ur5_save_{scope}", bool(state.get("ur5_save_images", True))
    )
    if changed_save:
        state.ur5_save_images = bool(save_images)
    changed_video, video = imgui.checkbox(f"Also keep frames from the video while moving##ur5_video_{scope}", bool(state.get("ur5_video", True)))
    if changed_video:
        state.ur5_video = bool(video)


def _draw_run_progress(runner, scope):
    """The running block. Both runs report through the same ScanProgress."""
    progress = runner.progress
    imgui.text_colored(COLOR_OK, f"Running {progress.target_index + 1}/{progress.target_count} ({progress.percent:.0f}%)")
    imgui.progress_bar(progress.percent / 100.0, (-1, 0))
    imgui.text(f"Captured {progress.captured} images" + (f", {progress.video_frames} video frames" if runner.video_feed is not None else ""))
    imgui.text_wrapped(progress.status)
    if imgui.button(f"Stop##ur5_stop_{scope}", (-1, 0)):
        runner.stop()


def _draw_start_confirmation(state, controller, which):
    """The deliberate second step before any motion, phrased as what is about to happen."""
    imgui.separator()
    imgui.text_colored(COLOR_DANGER, "The robot arm is about to move.")
    imgui.text_wrapped(
        "Check that the workspace is clear, the fabric is mounted, and you "
        "can reach the hardware emergency stop."
    )
    half = max(90.0, (imgui.get_content_region_avail().x - imgui.get_style().item_spacing.x) * 0.5)
    imgui.push_style_color(imgui.Col_.button, (0.20, 0.55, 0.25, 1.0))
    if imgui.button(f"Confirm and start##ur5_confirm_{which}", (half, 0)):
        controller.pending_start = ""
        _start_run(state, controller, which)
    imgui.pop_style_color()
    imgui.same_line()
    if imgui.button(f"Cancel##ur5_cancel_{which}", (half, 0)):
        controller.pending_start = ""
        controller.status = "Cancelled"
    imgui.separator()


def _draw_results_tab(state, controller, renderer):
    """What the runs produced, and what is made from it."""
    gui_captures.draw_captures_section(state, controller, "Captured images")
    _draw_analysis_section(state, controller)
    imgui.push_style_color(imgui.Col_.button, (0.18, 0.40, 0.65, 1.0))
    imgui.push_style_color(imgui.Col_.button_hovered, (0.24, 0.50, 0.78, 1.0))
    if imgui.button("Open in Splatting Mode (Export, Pose Solve & 3DGS)##ur5_to_splat", (-1, 36)):
        state.request_app_mode = "model"
    imgui.pop_style_color(2)
    _draw_scan_section(state, controller)


def _draw_scan_section(state, controller):
    if not imgui.collapsing_header("Scan Mode grid on the real arm"):
        return
    robot = controller.robot
    runner = controller.runner

    imgui.text(f"Grid: {int(state.scanner_rows)} x {int(state.scanner_cols)} cells, "
               f"{int(state.scanner_angles)} angles")
    imgui.text_colored(COLOR_MUTED, "Set the grid and angles in Scan Mode; both modes share them.")

    _draw_dwell_controls(state, "scan")

    scanning = runner is not None and runner.running

    if scanning:
        imgui.text_colored(COLOR_OK, "Running - progress is shown at the top")
        return

    if controller.pending_start == "scan":
        _draw_start_confirmation(state, controller, "scan")
        return

    can_start = robot.ready() and _camera_ready(state, controller)
    if not can_start:
        imgui.begin_disabled()
    if imgui.button("Start real scan...##ur5_scan_start", (-1, 0)):
        controller.pending_start = "scan"
    if not can_start:
        imgui.end_disabled()
        if not robot.connected:
            imgui.text_colored(COLOR_MUTED, "Connect to the robot first.")
        elif not robot.ready():
            imgui.text_colored(COLOR_MUTED, "The robot is not ready; see the line at the top.")
        else:
            imgui.text_colored(COLOR_MUTED, "Start the camera preview, or turn off Save captured images.")

    if runner is not None:
        progress = runner.progress
        if progress.finished or progress.captured:
            imgui.text_wrapped(progress.status)
            if progress.error:
                imgui.text_colored(COLOR_DANGER, progress.error)
            if progress.output_dir:
                imgui.text_colored(COLOR_MUTED, f"Saved to {Path(progress.output_dir).name}")


def _start_run(state, controller, which):
    """Start one of the two runs. They differ only in which plan the runner is given.

    A capture run is started against the plan the user already checked and saw, not one
    solved again here: solving from a pose the arm has since left is how a run comes to
    execute a path nobody looked at.
    """
    try:
        plan = controller.plan if which == "capture" else build_scan_plan(state)
        if plan is None:
            raise RuntimeError("no checked capture path")
        controller.apply_capture_settings()
        controller.runner = _make_runner(state, controller, plan)
    except Exception as exc:
        controller.status = f"Could not start: {exc}"
        return
    if controller.runner.start():
        controller.status = f"Real UR5 {which} running"
    else:
        controller.status = controller.runner.progress.status


def _draw_video_extraction(state, controller, folder, shown):
    """Choose the session's video frames again from its recorded video, so a change to the rules costs no arm time."""
    extracting = controller.video_extract is not None and controller.video_extract.is_alive()
    if extracting:
        imgui.text_colored(COLOR_OK, f"Picking pictures from {ur5_scan.VIDEO_FILE}... {controller.video_extract_result}")
        return
    if controller.video_extract is not None:
        controller.video_extract = None
        shown.records = gui_captures.load_session(folder)
        controller.robot_solved, controller.dataset = None, None
    if controller.video_extract_result:
        imgui.text_colored(COLOR_MUTED, controller.video_extract_result)
    if imgui.button(f"Pick the pictures again from the recorded {ur5_scan.VIDEO_FILE}##video_extract", (-1, 0)):
        def extract():
            try:
                sampler = ur5_scan.extract_from_video(folder, _fabric_check(state), float(state.get("ur5_min_coverage", 0.8)),
                                                      lambda kept: setattr(controller, "video_extract_result", f"{kept} kept so far"))
                controller.video_extract_result = f"Picked {sampler.kept} pictures from the video, rejected {dict(sampler.rejected)}"
            except Exception as exc:
                controller.video_extract_result = f"Could not pick pictures from the video: {exc}"
        controller.video_extract_result = ""
        controller.video_extract = threading.Thread(target=extract, daemon=True, name="video-extract")
        controller.video_extract.start()


def _draw_analysis_section(state, controller):
    if not imgui.collapsing_header("RGB analysis", imgui.TreeNodeFlags_.default_open):
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


def _draw_splatting_body(state, controller):
    """Gaussian Splatting as one guided flow: five steps, a Next button that does the next one, and
    everything else behind Advanced. Each step's state is read from disk and the controller every
    frame, so the list is right after a restart too."""
    runner = controller.runner
    shown = gui_captures.gallery(controller)
    session_id, records = ((runner.session_id, runner.capture_records) if runner is not None and runner.capture_records
                           else (shown.session, shown.records))
    scans = Path(state.project_root) / "robot_scans"
    folder = scans / session_id if session_id else None
    flow = _splat_flow(state, controller, session_id, folder, records)

    for index, step in enumerate(flow["steps"], start=1):
        mark, colour = {"done": ("[x]", COLOR_OK), "busy": ("[~]", COLOR_WARN), "next": ("[ ]", (1.0, 1.0, 1.0, 1.0))}.get(step["state"], ("[ ]", COLOR_MUTED))
        imgui.text_colored(colour, f"{mark} {index} {step['name']}")
        imgui.same_line(170)
        imgui.text_colored(COLOR_MUTED if step["state"] != "next" else (1.0, 1.0, 1.0, 1.0), step["detail"])
        _help(step["help"])

    nxt = flow["next"]
    if nxt is not None and nxt["state"] == "next" and nxt.get("key") == "export":
        changed_only, only = imgui.checkbox("Object only (paint the background black)##splat_object_only", _object_only_choice(controller, session_id, records))
        if changed_only:
            _set_object_only_choice(controller, session_id, only)
        _help("On: the splat is one dark object alone - no paper, no floaters round it. Off: the whole frame is reconstructed; "
              "right for a cloth. Guessed from the pictures; tick or untick to override.")
    if nxt is None:
        imgui.text_colored(COLOR_OK, "All done - the result is below.")
        imgui.same_line()
        if imgui.small_button("Train again##splat_retrain"):
            flow["train"]()
        _help("Trains the same dataset afresh. To change Object only, re-export under Advanced; the Train step then reopens by itself.")
    elif nxt["state"] == "busy":
        imgui.progress_bar(-1.0 * time.time(), imgui.ImVec2(-1, 0), nxt["busy_text"])
        if nxt.get("stop"):
            if imgui.button(f"Stop##splat_stop_{nxt['key']}", (-1, 0)):
                nxt["stop"]()
    else:
        if nxt.get("blocked"):
            imgui.text_colored(COLOR_WARN, nxt["blocked"])
            imgui.begin_disabled()
        imgui.push_style_color(imgui.Col_.button, (0.18, 0.40, 0.65, 1.0))
        imgui.push_style_color(imgui.Col_.button_hovered, (0.24, 0.50, 0.78, 1.0))
        if imgui.button(f"Next: {nxt['action']}##splat_next", (-1, 40)):
            nxt["run"]()
        imgui.pop_style_color(2)
        if nxt.get("blocked"):
            imgui.end_disabled()
    if controller.hand_eye_result:
        imgui.text_wrapped(controller.hand_eye_result)
    trainer = controller.trainer
    if trainer is not None and (trainer.running or trainer.returncode not in (None, 0)):
        if not trainer.running:
            imgui.text_colored(COLOR_DANGER, f"Training did not finish: {trainer.status} (full log in splat_output\\train.log)")
        for line in trainer.tail(3):
            imgui.text_colored(COLOR_MUTED, line[:110])
    solver = controller.robot_solver
    if solver is not None and (solver.running or solver.returncode not in (None, 0)):
        if not solver.running:
            imgui.text_colored(COLOR_DANGER, f"The pose solve did not finish: {solver.status}")
        for line in solver.tail(3):
            imgui.text_colored(COLOR_MUTED, line[:110])

    if imgui.tree_node("Advanced##splat_advanced"):
        _draw_splat_advanced(state, controller, session_id, folder, records, shown, scans)
        imgui.tree_pop()

    dataset, output_dir = flow["dataset"], flow["output_dir"]
    artifacts = _artifact_paths(output_dir)
    if dataset is not None and artifacts:
        imgui.separator_text("Result")
        imgui.text_colored(COLOR_OK, f"Reconstruction: {artifacts[0].name}")
        if runner is not None and runner.session_id and imgui.button("Record result in the database##ur5_splat_record", (-1, 0)):
            _record_splat_output(state, controller, artifacts[0])
        gui_captures.draw_splat_judgement(controller, output_dir, Path(dataset.root))
        gui_captures.draw_splat_viewer(controller, artifacts)


def _help(text: str) -> None:
    """A (?) after the last item, explaining it on hover - instead of a paragraph in the panel."""
    imgui.same_line()
    imgui.text_disabled("(?)")
    if imgui.is_item_hovered():
        imgui.set_tooltip(text)


_json_cache: dict = {}


def _read_json(path) -> dict | None:
    """A JSON file the panel asks about every frame, read again only when it changes on disk."""
    path = Path(path)
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return None
    cached = _json_cache.get(str(path))
    if cached is not None and cached[0] == stamp:
        return cached[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = None
    _json_cache[str(path)] = (stamp, data)
    return data


def _splat_artifacts(output_dir: Path | None) -> list[Path]:
    """The splats under `output_dir`, newest first - from a run in this session or an earlier one.
    Walking the folder costs milliseconds and the panel draws sixty times a second, so the walk is
    repeated at most every couple of seconds."""
    if output_dir is None:
        return []

    def walk():
        if not output_dir.exists():
            return []
        # The trainer deletes old checkpoints as it writes new ones, so a file found by the walk may
        # be gone a moment later: its time is taken here, and a vanished one is left out.
        found = []
        for pattern in ("*.ply", "*.splat", "*.ckpt"):
            for path in output_dir.rglob(pattern):
                try:
                    found.append((path.stat().st_mtime, path))
                except OSError:
                    continue
        found.sort(key=lambda item: item[0], reverse=True)
        return found

    return gui_captures.every(("splat_artifacts", str(output_dir)), walk)


def _artifact_paths(output_dir: Path | None) -> list[Path]:
    return [path for _, path in _splat_artifacts(output_dir)]


def _session_dataset(folder: Path | None, session_id: str, wanted: str = "calibration"):
    """The export already on disk for this session, if there is one with the `wanted` kind of poses
    - so a restart does not lose the step. None when the folder holds no export or one with other poses."""
    if folder is None:
        return None
    root = folder / "splat_dataset"
    transforms = _read_json(root / "transforms.json")
    if transforms is None:
        return None
    if wanted not in str(transforms.get("pose_source", "")):
        return None
    return gaussian_splatting.SplatDataset(root, len(transforms.get("frames", [])), root / "transforms.json", root / "sparse" / "0", session_id)


def _splat_flow(state, controller, session_id, folder, records) -> dict:
    """The five steps' states and the next action, from what is on disk and running.

    Pictures with an arm pose go the calibrated way (mount calibration, export). A session of
    pictures without one - video frames, a hand-held capture - has nothing to calibrate, so its
    poses are solved from the pictures with COLMAP instead, and the steps say so."""
    calibration = hand_eye.load_calibration()
    fitting = controller.hand_eye_fit is not None and controller.hand_eye_fit.is_alive()
    posed = [record for record in records if (record.get("actual_tcp_pose") or record.get("target_position")) is not None]
    # A combined session (several scans solved together) has no single mount or placement to trust:
    # its poses come from the pictures.
    combined = folder is not None and (folder / "combined.json").is_file()
    from_pictures = bool(records) and (combined or not posed)
    # A scan over the printed ChArUco board is placed by the board (charuco.py): no mount calibration
    # and no COLMAP, and whatever tool-centre setting the pendant had that day cancels out.
    board = _board_survey(session_id, records) if records and not combined else None
    on_board = bool(board and board["seen"])
    if on_board:
        from_pictures = False
    # A calibration describes the mount as it was when its scan was taken. A scan from before that
    # (the session names carry the date) may have had the camera mounted differently - here it did,
    # turned 49 degrees - so for it the planner's own camera model is used instead, and the step says so.
    older_than_calibration = bool(calibration and calibration.get("from_sessions") and session_id
                                  and session_id < min(calibration["from_sessions"]))
    trainer = controller.trainer
    training = trainer is not None and trainer.running
    command = str(state.get("ur5_splat_command", "")).strip() or SPLATFACTO_COMMAND

    if from_pictures:
        solver = controller.robot_solver
        solving = solver is not None and solver.running
        dataset = controller.colmap_dataset
        if (dataset is None or Path(dataset.root).parent != folder) and folder is not None and not solving \
                and (folder / COLMAP_DATASET_NAME / "pose_report.json").is_file():
            try:
                _, dataset = _solved_dataset(folder / COLMAP_DATASET_NAME, session_id)
                controller.colmap_dataset = dataset
            except Exception:
                dataset = None
    else:
        dataset = controller.dataset
        wanted = "ChArUco" if on_board else "forward kinematics" if older_than_calibration else "calibration"
        if dataset is None or wanted not in _dataset_pose_source(dataset) or Path(dataset.root).parent != folder:
            dataset = _session_dataset(folder, session_id, wanted)
    if dataset is not None and dataset.image_count <= 0:
        dataset = None
    output_dir = Path(dataset.root).parent / Path(dataset.root).name.replace("dataset", "output") if dataset is not None else None
    exported_at = Path(dataset.transforms_path).stat().st_mtime if dataset is not None and Path(dataset.transforms_path).exists() else 0.0
    artifacts = [path for stamp, path in _splat_artifacts(output_dir)
                 if stamp >= exported_at and path.suffix == ".ply" and path.name != "visual_hull.ply"]
    trained = bool(artifacts) and not training

    if from_pictures:
        pose_steps = [
            {"name": "Calibrate mount", "state": "done",
             "detail": ("not needed: several scans solved together, from the pictures" if combined
                        else "not needed: these pictures carry no arm pose (video frames or hand-held)"),
             "help": "The mount calibration applies to pictures the arm took at a known pose. These have none, so their poses "
                     "are solved from the pictures themselves instead."},
            {"name": "Solve poses", "state": "busy" if (solver is not None and solver.running) else ("done" if dataset is not None else "next"),
             "detail": (f"{dataset.image_count} of {dataset.offered or len(records)} pictures placed by COLMAP" if dataset is not None
                        else "COLMAP, from how the pictures overlap - minutes"),
             "help": "COLMAP matches features between pictures and works out where each was taken. It needs texture - a knit is "
                     "fine, a plain sheet with one shiny object is not - and pictures that overlap their neighbours."},
        ]
    elif on_board:
        pose_steps = [
            {"name": "Calibrate mount", "state": "done",
             "detail": f"not needed: the ChArUco board places the cameras ({board['seen']} of {board['looked']} sampled pictures see it)",
             "help": "Pictures that see the board are placed by it to a fraction of a pixel. The rest are placed through the arm, "
                     "with the mount fitted to this scan's own board views - so the pendant's tool-centre setting does not matter. "
                     "The lens is calibrated from the board views too when there are 12 or more."},
            {"name": "Export dataset", "state": "done" if dataset is not None else "next",
             "detail": (f"{dataset.image_count} pictures placed by the ChArUco board" if dataset is not None
                        else f"{len(records)} pictures over the board"),
             "help": "Writes the pictures and their board-measured poses to robot_scans\\<session>\\splat_dataset, with the lens "
                     "and its distortion, and seed points coloured from the pictures. The board is the world: z up out of the paper."},
        ]
    else:
        pose_steps = [
            {"name": "Calibrate mount", "state": "busy" if fitting else ("done" if calibration is not None else "next"),
             "detail": (f"done {calibration.get('fitted_at', '')[:10]}, {calibration.get('rms_px', 0):.0f} px over {calibration.get('pictures_used', 0)} pictures"
                        + (", measured lens" if str(calibration.get("lens", "")).startswith("measured") else ", lens guessed")
                        if calibration is not None else "once per camera mount"),
             "help": "Where the camera sits on the tool, measured once from a scan of one small dark object on white paper. "
                     "Kept in config/hand_eye.json and used for every scan after. Redo it only if the camera is remounted - or after "
                     "measuring the lens with scripts/calibrate_lens.py (a checkerboard), which sharpens it from ~10 px to ~2 px."},
            {"name": "Export dataset", "state": "done" if dataset is not None else "next",
             "detail": (f"{dataset.image_count} pictures with " + ("the planner's poses" if older_than_calibration else "calibrated poses")
                        + (", object only" if _dataset_object_only(dataset) else ", with background")
                        if dataset is not None else f"{len(posed)} pictures with an arm pose"
                        + (" - scanned before the mount was calibrated, so the planner's own camera model is used" if older_than_calibration else "")
                        + (f" ({len(records) - len(posed)} video frames without one are left out)" if len(records) > len(posed) else "")),
             "help": "Writes the pictures and their camera poses (the arm's recorded pose with the mount calibration applied) "
                     "to robot_scans\\<session>\\splat_dataset. No feature matching: every picture is placed. "
                     "Object only: everything but the one dark object is painted black, so the splat is the object alone."},
        ]
    steps = [
        {"name": "Scan", "state": "done" if records else "next",
         "detail": ((f"{len(records)} pictures from {len({r.get('combined_from') for r in records})} scans" if combined
                     else f"{len(records)} pictures from {session_id}") if records else "no pictures yet"),
         "help": "Run the splat pattern under 3 Capture, or pick an earlier session under Captured images."},
        *pose_steps,
        {"name": "Train", "state": "busy" if training else ("done" if trained else "next"),
         "detail": (trainer.status if training else (f"{artifacts[0].name}" if trained else "20-30 minutes on the GPU")),
         "help": "nerfstudio splatfacto: 30,000 steps for one object on black, 15,000 for a cloth (texture edge to edge "
                 "would otherwise split into a million Gaussians at half a second a step), the cameras free to shift a little. "
                 "The full log is in splat_output\\train.log. The command is under Advanced."},
        {"name": "Judge", "state": "done" if trained else "pending",
         "detail": "held-out pictures, PSNR and the viewer below" if trained else "",
         "help": "The honest measure is how well held-out pictures are reproduced: over 28 dB is good. "
                 "The left/right comparison shows the quality; the viewer below shows the shape from any side."},
    ]
    done_through = 0
    for step in steps:
        if step["state"] == "done":
            done_through += 1
        else:
            break

    def calibrate():
        captures = [dict(record) for record in records]

        def fit_mount():
            try:
                result = hand_eye.fit(captures)
                hand_eye.save_calibration(result)
                skipped = len(result.get("skipped", []))
                controller.hand_eye_result = (f"Calibrated: {hand_eye.describe(result)}"
                                              + (f"; {skipped} pictures left out (object missing or cut by the frame)." if skipped else "."))
            except Exception as exc:
                controller.hand_eye_result = f"Calibration failed: {exc}"

        controller.hand_eye_result = ""
        controller.hand_eye_fit = threading.Thread(target=fit_mount, daemon=True, name="hand-eye-fit")
        controller.hand_eye_fit.start()

    def export():
        try:
            controller.robot_solved = None
            if on_board:
                controller.dataset = charuco.export_dataset(records, folder / "splat_dataset", session_id=session_id)
                notes = controller.dataset.warnings
                controller.status = (f"Exported {controller.dataset.image_count} of {len(records)} pictures, placed by the ChArUco board"
                                     + (f" - {notes[0]}" + (f" (+{len(notes) - 1} more notes)" if len(notes) > 1 else "") if notes else ""))
                return
            if older_than_calibration:
                controller.dataset = gaussian_splatting.export_dataset(
                    records, folder / "splat_dataset", session_id=session_id,
                    fov_degrees=float(state.get("ur5_splat_fov", gaussian_splatting.DEFAULT_FOV_DEGREES)), image_turn=_image_turn(state))
            else:
                controller.dataset = gaussian_splatting.export_dataset(records, folder / "splat_dataset", session_id=session_id,
                                                                       calibration=hand_eye.load_calibration(),
                                                                       object_only=_object_only_choice(controller, session_id, records))
            controller.status = (f"Exported {controller.dataset.image_count} pictures"
                                 if controller.dataset.image_count else "Nothing exported: no picture has an arm pose")
        except Exception as exc:
            controller.status = f"Dataset export failed: {exc}"

    def solve():
        controller.robot_solved, controller.colmap_dataset = None, None
        controller.robot_solver = _start_pose_solve(controller, folder, len(records),
                                                    float(state.get("ur5_splat_fov", gaussian_splatting.DEFAULT_FOV_DEGREES)), COLMAP_DATASET_NAME)

    def train():
        state.ur5_splat_command = command
        controller.trainer = gaussian_splatting.SplatTrainer(command, dataset, output_dir)
        controller.status = "Gaussian Splatting training started" if controller.trainer.start() else controller.trainer.status

    nxt = None
    if done_through == 0:
        nxt = {"state": "next", "action": "Scan first (3 Capture, pattern 'splat')", "run": lambda: None, "blocked": "No pictures in this session yet."}
    elif done_through == 1:
        nxt = ({"state": "busy", "busy_text": "Fitting the camera mount to the pictures...", "key": "fit"} if fitting else
               {"state": "next", "action": f"Calibrate the camera mount from these {len(records)} pictures", "run": calibrate})
    elif done_through == 2:
        if from_pictures:
            solver = controller.robot_solver
            if solver is not None and solver.running:
                nxt = {"state": "busy", "busy_text": "Solving the camera poses with COLMAP...", "key": "solve", "stop": solver.stop}
            else:
                missing = [reason for reason, absent in _solver_missing() if absent]
                nxt = {"state": "next", "action": f"Solve the camera poses with COLMAP ({len(records)} pictures)", "run": solve,
                       "blocked": missing[0] if missing else ""}
        else:
            nxt = ({"state": "next", "action": "Export the dataset with poses from the ChArUco board", "run": export, "key": "export",
                    "blocked": ""} if on_board else
                   {"state": "next", "action": "Export the dataset with calibrated poses", "run": export, "key": "export",
                    "blocked": "" if posed else "No picture in this session has an arm pose."})
    elif done_through == 3:
        if training:
            nxt = {"state": "busy", "busy_text": trainer.status, "key": "train", "stop": trainer.stop}
        else:
            blocked = ("set the trainer up once: powershell -ExecutionPolicy Bypass -File scripts\\setup_splatting.ps1"
                       if "train_splat.ps1" in command and not SPLAT_VENV.exists() else "")
            if dataset is not None and dataset.offered and (dataset.image_count < MIN_TRAINABLE_PICTURES or 2 * dataset.image_count <= dataset.offered):
                blocked = (f"COLMAP placed only {dataset.image_count} of {dataset.offered} pictures in one piece; a splat needs at least "
                           f"{MIN_TRAINABLE_PICTURES} and most of them. Capture again with more overlap.")
            nxt = {"state": "next", "action": "Train the splat (20-30 minutes)", "run": train, "blocked": blocked}
    return {"steps": steps, "next": nxt, "dataset": dataset, "output_dir": output_dir, "train": train}


def _draw_splat_advanced(state, controller, session_id, folder, records, shown, scans):
    """What the guided flow does not need: the trainer command, redoing the calibration, and the
    other two pose sources - COLMAP and the planner's uncalibrated poses."""
    changed_cmd, command = imgui.input_text("Trainer command##ur5_splat_cmd", str(state.get("ur5_splat_command", "")) or SPLATFACTO_COMMAND)
    if changed_cmd:
        state.ur5_splat_command = command
    _help("Runs with {dataset} and {output} filled in. The default is scripts/train_splat.ps1 (nerfstudio splatfacto).")
    if imgui.button("Reset to nerfstudio splatfacto##ur5_splat_preset", (-1, 0)):
        state.ur5_splat_command = SPLATFACTO_COMMAND

    calibration = hand_eye.load_calibration()
    if calibration is not None and records:
        changed_only, only = imgui.checkbox("Object only (paint the background black)##splat_object_only_adv", _object_only_choice(controller, session_id, records))
        if changed_only:
            _set_object_only_choice(controller, session_id, only)
        if imgui.button("Export the dataset again with calibrated poses##splat_reexport", (-1, 0)):
            try:
                controller.dataset = gaussian_splatting.export_dataset(records, folder / "splat_dataset", session_id=session_id,
                                                                       calibration=calibration, object_only=_object_only_choice(controller, session_id, records))
                controller.status = f"Exported {controller.dataset.image_count} pictures with calibrated poses"
            except Exception as exc:
                controller.status = f"Dataset export failed: {exc}"
        _help("Re-export after changing Object only, then train again. Training again on the same dataset is under the Train step.")
    if calibration is not None:
        imgui.text_colored(COLOR_MUTED, hand_eye.describe(calibration))
        fitting = controller.hand_eye_fit is not None and controller.hand_eye_fit.is_alive()
        if not fitting and records and imgui.button(f"Redo the mount calibration from these {len(records)} pictures##hand_eye_refit", (-1, 0)):
            captures = [dict(record) for record in records]

            def refit():
                try:
                    result = hand_eye.fit(captures)
                    hand_eye.save_calibration(result)
                    controller.hand_eye_result = f"Calibrated: {hand_eye.describe(result)}."
                except Exception as exc:
                    controller.hand_eye_result = f"Calibration failed: {exc}"

            controller.hand_eye_fit = threading.Thread(target=refit, daemon=True, name="hand-eye-fit")
            controller.hand_eye_fit.start()
        _help("Only needed after the camera is remounted. Needs a scan of one small dark object on white paper.")

    runner = controller.runner
    solver = controller.robot_solver
    if imgui.tree_node("Poses solved from the pictures with COLMAP##robot_colmap"):
        _help(f"For textured cloth with no calibration. Fails on a plain sheet with one shiny object. Its dataset goes to {COLMAP_DATASET_NAME} "
              "and trains into splat_output_colmap; it never touches the calibrated export.")
        if folder is not None and (folder / ur5_scan.VIDEO_FILE).is_file() and not (runner is not None and runner.running):
            _draw_video_extraction(state, controller, folder, shown)
        if imgui.tree_node("Also use pictures from other scans##robot_extra_sessions"):
            for other in gui_captures.sessions(scans):
                if other == session_id or other.startswith("combined_"):
                    continue
                if imgui.checkbox(f"{other}##extra_{other}", other in EXTRA_SESSIONS)[0]:
                    EXTRA_SESSIONS.symmetric_difference_update({other})
            imgui.tree_pop()
        extras = sorted(EXTRA_SESSIONS - {session_id})
        count = len(records) + sum(len(gui_captures.every(("session_count", other), lambda other=other: gui_captures.load_session(scans / other), 30.0))
                                   for other in extras)
        if not _draw_solving(solver, "robot"):
            why_not = [reason for reason, missing in (("run a capture first", not records), *_solver_missing()) if missing]
            for reason in why_not:
                imgui.text_colored(COLOR_WARN, f"- {reason}")
            if why_not:
                imgui.begin_disabled()
            label = f"{count} pictures from {len(extras) + 1} scans" if extras else f"the {count} pictures"
            if imgui.button(f"Solve camera poses with COLMAP from {label}##robot_solve", (-1, 0)):
                controller.robot_solved = None
                solve_folder = folder
                if extras:
                    solve_folder = scans / f"combined_{time.strftime('%Y%m%d_%H%M%S')}"
                    solve_folder.mkdir(parents=True)
                    write_json(solve_folder / "combined.json", {"sessions": [folder.name, *extras]})
                controller.robot_solver = _start_pose_solve(controller, solve_folder, count,
                                                            float(state.get("ur5_splat_fov", gaussian_splatting.DEFAULT_FOV_DEGREES)),
                                                            COLMAP_DATASET_NAME)
            if why_not:
                imgui.end_disabled()
        if _solve_finished(solver, "robot"):
            if controller.robot_solved is None:
                try:
                    controller.robot_solved, controller.colmap_dataset = _solved_dataset(solver.output_dir, Path(solver.dataset.root).name)
                except Exception as exc:
                    _banner(f"The pose solve left no usable report: {exc}", COLOR_DANGER, "robot_solve_report")
            report = controller.robot_solved
            if report is not None:
                _draw_checks(free_capture.solve_checks(report), "Poses solved - train it below.", "robot_solved_checks")
                _draw_unplaced(report)
        if controller.colmap_dataset is not None:
            _draw_trainer(state, controller, controller.colmap_dataset, None, "solve the poses first")
        imgui.tree_pop()

    if imgui.tree_node("Poses from the planner with a guessed mount offset (uncalibrated)##robot_arm_poses"):
        _help("The camera assumed at the tool point plus this offset along the tool axis. Only right if it really sits there; "
              "the calibration above replaces this.")
        changed_offset, offset = imgui.slider_float("Camera offset from tool (m)##ur5_splat_offset",
                                                    float(state.get("ur5_splat_camera_offset", gaussian_splatting.DEFAULT_CAMERA_OFFSET_M)),
                                                    0.0, 0.4, "%.3f")
        if changed_offset:
            state.ur5_splat_camera_offset = float(offset)
        if not records:
            imgui.begin_disabled()
        if imgui.button("Export with the planner's poses##ur5_splat_export", (-1, 0)):
            try:
                controller.colmap_dataset = gaussian_splatting.export_dataset(
                    records, folder / "splat_dataset_planner", session_id=session_id,
                    hand_eye=gaussian_splatting.hand_eye_matrix(float(state.get("ur5_splat_camera_offset", gaussian_splatting.DEFAULT_CAMERA_OFFSET_M))),
                    fov_degrees=float(state.get("ur5_splat_fov", gaussian_splatting.DEFAULT_FOV_DEGREES)),
                    image_turn=_image_turn(state))
                controller.status = f"Exported {controller.colmap_dataset.image_count} images with the planner's poses"
            except Exception as exc:
                controller.status = f"Dataset export failed: {exc}"
        if not records:
            imgui.end_disabled()
        if controller.colmap_dataset is not None and "planner" in str(controller.colmap_dataset.root):
            _draw_trainer(state, controller, controller.colmap_dataset, None, "export first")
        imgui.tree_pop()


COLMAP_DATASET_NAME = "splat_dataset_colmap"


def _object_only_choice(controller, session_id: str, records) -> bool:
    """Whether to export object-only: what was ticked for this session, else a guess from the
    pictures - a small dark thing on paper (little fabric in frame) wants it, a cloth does not."""
    chosen = getattr(controller, "object_only_choice", {})
    if session_id in chosen:
        return bool(chosen[session_id])

    def looks_like_one_object():
        from PIL import Image

        spreads = []
        for record in records[:: max(1, len(records) // 5)][:5]:
            path = Path(str(record.get("path") or ""))
            if not path.is_file():
                continue
            with Image.open(path) as image:
                grey = np.asarray(image.convert("L"))
            ys, xs = np.nonzero(grey < gaussian_splatting.OBJECT_MAX_GREY)
            spreads.append((xs.max() - xs.min()) * (ys.max() - ys.min()) / grey.size if len(xs) > 100 else 0.0)
        # One nut's dark pixels sit in a patch a fifth of the frame; a knit's dark holes span all of it.
        return bool(spreads) and float(np.median(spreads)) < 0.5

    return bool(gui_captures.every(("object_only_guess", session_id), looks_like_one_object, 600.0))


def _board_survey(session_id: str, records) -> dict | None:
    """Whether this session was scanned over the printed ChArUco board - a look at a few of its
    pictures, remembered for ten minutes. None when no board has been printed."""
    if charuco.load_board_spec() is None:
        return None
    return gui_captures.every(("charuco_survey", session_id), lambda: charuco.survey(records), 600.0)


def _set_object_only_choice(controller, session_id: str, value: bool) -> None:
    if not hasattr(controller, "object_only_choice"):
        controller.object_only_choice = {}
    controller.object_only_choice[session_id] = bool(value)


def _dataset_object_only(dataset) -> bool:
    transforms = _read_json(dataset.transforms_path)
    return bool(transforms.get("object_only")) if transforms else False


def _dataset_pose_source(dataset) -> str:
    """Where a dataset's camera poses came from, as its transforms.json records it."""
    transforms = _read_json(dataset.transforms_path)
    if transforms is None:
        return "unknown (no transforms.json yet)"
    source = transforms.get("pose_source")
    if source:
        return str(source)
    return "solved from the pictures by COLMAP" if "applied_transform" in transforms else "unknown"


def _draw_trainer(state, controller, dataset, record, needs):
    """`dataset`, the trainer command, Run reconstruction, and the result - for a robot export and a
    hand-held capture alike. `record` keeps a result against its scan session, and is None where there
    is no session in the database to keep it against; `needs` says what makes a dataset, while there is none."""
    if dataset is not None:
        imgui.text(f"Dataset: {dataset.image_count} images")
        imgui.text_colored(COLOR_MUTED, str(dataset.root))
        source = _dataset_pose_source(dataset)
        imgui.text_colored(COLOR_OK if "calibration" in source else COLOR_WARN, f"Poses: {source}")
        for warning in dataset.warnings[:4]:
            imgui.text_colored(COLOR_WARN, f"- {warning}")

    changed_cmd, command = imgui.input_text(
        "Trainer command##ur5_splat_cmd", str(state.get("ur5_splat_command", ""))
    )
    if changed_cmd:
        state.ur5_splat_command = command
    imgui.text_colored(COLOR_MUTED, "Use {dataset} and {output} as placeholders.")
    if imgui.button("Use nerfstudio splatfacto (scripts/train_splat.ps1)##ur5_splat_preset", (-1, 0)):
        state.ur5_splat_command = SPLATFACTO_COMMAND
    if not SPLAT_VENV.exists():
        imgui.text_colored(COLOR_WARN, "The trainer is not set up yet: run scripts\\setup_splatting.ps1 once (several GB).")

    trainer = controller.trainer
    if trainer is not None and trainer.running:
        imgui.text_colored(COLOR_OK, "Training running")
        if imgui.button("Stop training##ur5_splat_stop", (-1, 0)):
            trainer.stop()
    else:
        command = str(state.get("ur5_splat_command", "")).strip()
        why_not = [reason for reason, missing in (
            (needs, dataset is None or dataset.image_count <= 0),
            (f"the pose solve placed only {dataset.image_count if dataset else 0} of {dataset.offered if dataset else 0} pictures in one piece - "
             f"a splat needs at least {MIN_TRAINABLE_PICTURES} and most of them; capture again (sharper, or with more overlap) before training",
             dataset is not None and dataset.offered and (dataset.image_count < MIN_TRAINABLE_PICTURES or 2 * dataset.image_count <= dataset.offered)),
            ("set a trainer command - e.g. press 'Use nerfstudio splatfacto'", not command),
            ("set the trainer up once: powershell -ExecutionPolicy Bypass -File scripts\\setup_splatting.ps1",
             "train_splat.ps1" in command and not SPLAT_VENV.exists())) if missing]
        can_train = not why_not
        for reason in why_not:
            imgui.text_colored(COLOR_WARN, f"- {reason}")
        if not can_train:
            imgui.begin_disabled()
        if imgui.button("Run reconstruction##ur5_splat_run", (-1, 0)):
            # splat_dataset trains into splat_output, splat_dataset_colmap into splat_output_colmap:
            # the two never overwrite each other's splat.
            output_dir = Path(dataset.root).parent / Path(dataset.root).name.replace("dataset", "output")
            controller.trainer = gaussian_splatting.SplatTrainer(
                str(state.get("ur5_splat_command", "")), dataset, output_dir
            )
            if controller.trainer.start():
                controller.status = "Gaussian Splatting training started"
            else:
                controller.status = controller.trainer.status
        if not can_train:
            imgui.end_disabled()

    if trainer is not None and (trainer.running or trainer.dataset is dataset):
        imgui.text_wrapped(trainer.status)
        for line in trainer.tail(6):
            imgui.text_colored(COLOR_MUTED, line[:110])
        artifacts = trainer.output_artifacts()
        if artifacts:
            imgui.text_colored(COLOR_OK, f"Reconstruction: {artifacts[0].name}")
            if record is not None and imgui.button("Record result in the database##ur5_splat_record", (-1, 0)):
                record(artifacts[0])
            gui_captures.draw_splat_judgement(controller, Path(trainer.output_dir), Path(dataset.root))
            gui_captures.draw_splat_viewer(controller, artifacts)


def _record_splat_output(state, controller, artifact: Path):
    storage = scanner_core._scanner_storage(state)
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


FREE_CAPTURE_STEPS = (
    "1. First picture: step back until the whole fabric fits in the frame, and hold the camera straight down over it.",
    "2. Walk round the fabric at that distance, tilting the camera towards its middle. Then do it again closer, then close up.",
    "3. Move a little between pictures: every picture must overlap the one before, or it cannot be placed. The overlap line under the preview shows it live.",
    "4. A few non-repeating things beside the cloth (a ruler, a coin, printed text) stop the repeating stitches from confusing the matching.",
)
POSE_SOLVER_PYTHON = SPLAT_VENV / "Scripts" / "python.exe"
POSE_SOLVER_COMMAND = (f'"{POSE_SOLVER_PYTHON}" "{paths.SCRIPTS_DIR / "solve_poses.py"}"'
                       ' --session "{dataset}" --output "{output}" --fov {fov}')
POSE_SOLVER_STEPS = 4
MIN_TRAINABLE_PICTURES = 40
# A picture's outcome flashes the preview's frame this long, in its colour; the strip shows this
# many of the latest pictures, this wide.
FLASH_SECONDS = 0.9
FREE_THUMBS = 8
FREE_THUMB_WIDTH = 96
EVENT_COLOURS = {"kept": COLOR_OK, "unlinked": COLOR_DANGER, "skipped": COLOR_WARN}


def _draw_free_capture_tab(state, controller, renderer):
    """Hand-held capture: camera and session, shooting, readiness checklist, and handoff to Splatting Mode."""
    imgui.text_wrapped("Photograph the fabric with the camera in your hand - the arm is not used. The camera poses are then "
                       "solved from the pictures (COLMAP), and the splat is trained on them in Splatting Mode.")
    session = controller.free
    running = controller.camera is not None and controller.camera.running

    imgui.separator_text("Step 1 - Camera and session")
    _draw_free_session(state, controller, session, running)
    session = controller.free
    imgui.separator_text("Step 2 - Take pictures")
    _draw_free_shooting(state, controller, session, running)
    imgui.separator_text("Step 3 - Is it enough? (estimated while you shoot)")
    if session is None:
        imgui.text_colored(COLOR_MUTED, "The checklist fills in as you take pictures.")
    else:
        changed_sides, sides = imgui.checkbox("Require tilted views from the sides (deeper, more accurate splat)##free_sides",
                                              bool(state.get("free_require_sides", True)))
        if changed_sides:
            state.free_require_sides = bool(sides)
        _draw_checks(free_capture.readiness(session.rows(), bool(state.get("free_require_sides", True))),
                     "Enough pictures - open in Splatting Mode below.", "free_live_checks")
    imgui.separator_text("Step 4 - Solve poses & reconstruct in Splatting Mode")
    if session is None or not session.records:
        imgui.begin_disabled()
    imgui.push_style_color(imgui.Col_.button, (0.18, 0.40, 0.65, 1.0))
    imgui.push_style_color(imgui.Col_.button_hovered, (0.24, 0.50, 0.78, 1.0))
    if imgui.button("Open in Splatting Mode (Pose Solve & 3DGS)##free_to_splat", (-1, 36)):
        if session is not None:
            state.swatch_session_id = session.name
            if hasattr(state, "save_active_swatch"):
                state.save_active_swatch()
        state.request_app_mode = "model"
    imgui.pop_style_color(2)
    if session is None or not session.records:
        imgui.end_disabled()


def _draw_free_session(state, controller, session, running):
    device = int(state.get("ur5_camera_device", 0))
    label = controller.devices[device]["label"] if 0 <= device < len(controller.devices) else "none chosen"
    imgui.text(f"Camera: {label}")
    imgui.same_line()
    imgui.text_colored(COLOR_MUTED, "(choose it under 1 Setup)")
    if running:
        imgui.text_colored(COLOR_OK, f"Live: {controller.camera.describe()}")
    else:
        _draw_camera_start(state, controller, "free")
    if imgui.button("Start a new hand-held session##free_new", (-1, 0)):
        _start_free_session(state, controller)
    if session is None:
        imgui.text_colored(COLOR_WARN, "No session yet - start one to take pictures.")
    else:
        imgui.text_colored(COLOR_MUTED, f"Session {session.name}, saved in robot_scans (also under 4 Results > Captured images)")


def _start_free_session(state, controller):
    if controller.free is not None:
        controller.free.close()
    folder = Path(state.project_root) / "robot_scans" / time.strftime("free_%Y%m%d_%H%M%S")
    controller.free = free_capture.FreeCaptureSession(folder, float(state.get("ur5_splat_fov", gaussian_splatting.DEFAULT_FOV_DEGREES)))
    state.swatch_session_id = folder.name
    if hasattr(state, "save_active_swatch"):
        state.save_active_swatch()
    if not str(state.get("ur5_splat_command", "")).strip():
        state.ur5_splat_command = SPLATFACTO_COMMAND
    controller.status = f"Hand-held session {folder.name} started"


def _draw_free_shooting(state, controller, session, running):
    """The preview, framed in the colour of the last picture's outcome; the shutter; and what came of each picture."""
    if session is None or not running:
        imgui.text_colored(COLOR_MUTED, "Needs a running camera and a session (step 1).")
        return
    session.watch(controller.camera.latest_frame())
    if imgui.tree_node_ex("How to shoot it##free_steps", imgui.TreeNodeFlags_.default_open if not session.records else imgui.TreeNodeFlags_.none):
        for step in FREE_CAPTURE_STEPS:
            imgui.text_wrapped(step)
        imgui.tree_pop()

    shown = _draw_live_preview(state, controller)
    if shown is not None:
        _mark_preview(session, shown)
    colour, overlap = _overlap_reading(session)
    imgui.text_colored(colour, overlap)

    imgui.push_style_color(imgui.Col_.button, (0.18, 0.45, 0.25, 1.0))
    imgui.push_style_color(imgui.Col_.button_hovered, (0.22, 0.58, 0.32, 1.0))
    pressed = imgui.button(("Checking the picture..." if session.busy else "Take picture   (or press Space)") + "##free_take", (-1, 44))
    imgui.pop_style_color(2)
    if pressed or (not imgui.get_io().want_text_input and imgui.is_key_pressed(imgui.Key.space, False)):
        session.take()

    changed_auto, auto = imgui.checkbox("Take pictures automatically as I move##free_auto", bool(state.get("free_auto", False)))
    if changed_auto:
        state.free_auto = bool(auto)
    if bool(state.get("free_auto", False)):
        ready, why = session.auto_ready(float(state.get("free_auto_interval", 1.5)))
        if ready:
            session.take()
        imgui.text_colored(COLOR_OK if ready else COLOR_MUTED, f"Auto: {why}")
        _draw_field(state, "free_auto_interval")
        _help("A picture is taken each time the view has moved on enough while still overlapping the last.")

    event = session.last_event
    if event is not None:
        _banner(event.text, EVENT_COLOURS[event.kind], "free_last")
    imgui.text(f"{len(session.records)} pictures kept, {session.skipped} skipped")
    if session.records:
        imgui.same_line()
        if imgui.small_button("Undo last picture##free_undo"):
            session.discard_last()
    _draw_free_strip(controller, session)
    if session.events and imgui.tree_node("What happened to each try##free_events"):
        for past in reversed(session.events):
            imgui.text_colored(EVENT_COLOURS[past.kind], past.text)
        imgui.tree_pop()


def _mark_preview(session, shown):
    """The preview framed in the last outcome's colour while it is fresh, and the picture count on it."""
    x, y, w, h = shown
    draw_list = imgui.get_window_draw_list()
    event = session.last_event
    if event is not None and time.monotonic() - event.at < FLASH_SECONDS:
        draw_list.add_rect(imgui.ImVec2(x, y), imgui.ImVec2(x + w, y + h), imgui.get_color_u32(EVENT_COLOURS[event.kind]), rounding=0.0, thickness=6.0)
    label = f"{len(session.records)} pictures"
    draw_list.add_rect_filled(imgui.ImVec2(x + 6.0, y + 6.0), imgui.ImVec2(x + 14.0 + 8.0 * len(label), y + 28.0), imgui.get_color_u32((0.0, 0.0, 0.0, 0.6)))
    draw_list.add_text(imgui.ImVec2(x + 10.0, y + 10.0), imgui.get_color_u32((1.0, 1.0, 1.0, 1.0)), label)


def _overlap_reading(session):
    """How the live view overlaps the last kept picture, in words and a colour."""
    if not session.records:
        return COLOR_MUTED, "No picture yet: the first is the whole fabric, straight down."
    if session.overlap is None:
        return COLOR_MUTED, "Comparing the view with the last picture..."
    inliers, shift = session.overlap
    last = len(session.records)
    if inliers == 0:
        return COLOR_DANGER, f"No overlap with picture {last} - move back towards it."
    if inliers >= free_capture.OVERLAP_STRONG:
        return COLOR_OK, f"Overlap with picture {last}: strong ({inliers} points), moved {shift:.0%} of the frame."
    if inliers >= free_capture.OVERLAP_GOOD:
        return COLOR_OK, f"Overlap with picture {last}: good ({inliers} points), moved {shift:.0%} of the frame - a good time for a picture."
    return COLOR_WARN, f"Overlap with picture {last}: weak ({inliers} points) - take a picture now or move back a little."


def _draw_free_strip(controller, session):
    """The latest pictures, framed green when matched to an earlier one and red when not."""
    rows = session.rows()[-FREE_THUMBS:]
    for position, record in enumerate(rows):
        if not Path(record["path"]).exists():
            continue
        texture, (width, height) = gui_captures.file_texture(controller, record["path"], FREE_THUMB_WIDTH)
        if position:
            imgui.same_line()
        imgui.image(imgui.ImTextureRef(texture.glo), imgui.ImVec2(width, height), uv0=imgui.ImVec2(0, 1), uv1=imgui.ImVec2(1, 0))
        placed = record.get("relative_distance") is not None
        imgui.get_window_draw_list().add_rect(imgui.get_item_rect_min(), imgui.get_item_rect_max(),
                                              imgui.get_color_u32(COLOR_OK if placed else COLOR_DANGER), rounding=0.0, thickness=3.0)
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"{record['capture_view']}: "
                              + (f"{record['distance_band'] or 'outside the bands'}, {record['relative_distance']:.2f} of the first distance, "
                                 f"tilted {record['tilt_from_vertical_deg']:.0f} deg" if placed else "matches no earlier picture"))


def _banner(text, colour, key):
    """`text` on a band of `colour` - an outcome that must not be missed."""
    imgui.push_style_color(imgui.Col_.child_bg, (colour[0] * 0.35, colour[1] * 0.35, colour[2] * 0.35, 1.0))
    imgui.begin_child(f"##{key}", imgui.ImVec2(-1, 0), imgui.ChildFlags_.auto_resize_y | imgui.ChildFlags_.borders)
    imgui.push_style_color(imgui.Col_.text, colour)
    imgui.text_wrapped(text)
    imgui.pop_style_color()
    imgui.end_child()
    imgui.pop_style_color()


def _draw_checks(checks, enough_text, key):
    """One bar per target, then the verdict: enough, or the first target not met - which is what to shoot next.
    Returns whether every target is met."""
    for index, check in enumerate(checks):
        colour = COLOR_OK if check.done else COLOR_WARN
        imgui.push_style_color(imgui.Col_.plot_histogram, colour)
        imgui.progress_bar(check.fraction, imgui.ImVec2(150, 0), check.progress)
        imgui.pop_style_color()
        imgui.same_line()
        imgui.text_colored(colour, ("done - " if check.done else "") + check.label)
    missing = next((check for check in checks if not check.done), None)
    if missing is None:
        _banner(enough_text, COLOR_OK, key)
    else:
        _banner(f"Next: {missing.label}", COLOR_WARN, key)
    return missing is None


def _solver_missing():
    return (("install pycolmap once: uv pip install --python .venv-splat\\Scripts\\python.exe pycolmap",
             not (SPLAT_VENV / "Lib" / "site-packages" / "pycolmap").is_dir()),)


def _draw_solving(solver, key) -> bool:
    """While `solver` runs: its step as a bar, and a stop. Whether it is running."""
    if solver is None or not solver.running:
        return False
    lines = [line for line in solver.tail(40) if line[:1].isdigit() and line[1:3] == f"/{POSE_SOLVER_STEPS}"]
    imgui.progress_bar((int(lines[-1][0]) if lines else 0) / POSE_SOLVER_STEPS, imgui.ImVec2(-1, 0), lines[-1] if lines else "Starting...")
    imgui.text_colored(COLOR_MUTED, "This takes minutes - longer with more pictures. The app stays usable meanwhile.")
    if imgui.button(f"Stop solving##{key}_solve_stop", (-1, 0)):
        solver.stop()
    return True


def _solve_finished(solver, key) -> bool:
    """Whether `solver` has finished well; a failure is shown with the end of its log."""
    if solver is None or solver.running:
        return False
    if solver.returncode != 0:
        _banner(f"The pose solve did not finish: {solver.status}", COLOR_DANGER, f"{key}_solve_failed")
        for line in solver.tail(6):
            imgui.text_colored(COLOR_MUTED, line[:110])
        return False
    return True


def _draw_unplaced(report):
    if report["unregistered"]:
        imgui.text_colored(COLOR_WARN, "Not placed - more pictures between these and their neighbours would place them: "
                           + ", ".join(Path(name).stem for name in report["unregistered"][:10])
                           + (" ..." if len(report["unregistered"]) > 10 else ""))
    if int(report.get("models", 1)) > 1:
        imgui.text_colored(COLOR_WARN, f"The pictures fell into {report['models']} groups that share no view; only the largest is used.")


def _start_pose_solve(controller, folder: Path, count: int, fov_degrees: float, dataset_name: str = "splat_dataset"):
    """The pose solve over the pictures in `folder`, started, writing its dataset to folder/`dataset_name`."""
    output = Path(folder) / dataset_name
    solver = gaussian_splatting.SplatTrainer(
        POSE_SOLVER_COMMAND.replace("{fov}", f"{float(fov_degrees):.2f}"),
        gaussian_splatting.SplatDataset(Path(folder), int(count), output / "transforms.json", output / "colmap", Path(folder).name), output)
    solver.start()
    controller.status = solver.status
    return solver


def _solved_dataset(output: Path, name: str):
    """What a finished solve measured, and its dataset - which trains as a robot export would."""
    report = json.loads((output / "pose_report.json").read_text())
    pieces = int(report.get("models", 1))
    warnings = ([f"the pictures fell into {pieces} pieces that share no view; the dataset is the largest piece only"] if pieces > 1 else []) \
             + [f"not placed: {name}" for name in report["unregistered"]]
    return report, gaussian_splatting.SplatDataset(output, int(report["registered"]), output / "transforms.json", output / "colmap", name,
                                                   warnings, pieces, int(report["total"]))


# The panel's tabs, in the order a session goes through them; the hand-held capture last, as it
# needs none of the others but the camera. Defined after the functions they name; draw_ur5_panel
# reads it only when called.
UR5_TABS = (("1 Setup",        _draw_setup_tab),
            ("2 Position",     _draw_position_tab),
            ("3 Capture",      _draw_capture_tab),
            ("4 Results",      _draw_results_tab),
            ("5 Free capture", _draw_free_capture_tab))
