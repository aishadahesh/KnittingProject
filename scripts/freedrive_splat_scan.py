"""A splat scan from wherever the arm is put by hand: free drive, then a dome of views flown and photographed.

    uv run python scripts\\freedrive_splat_scan.py freedrive    arm goes loose; create robot_scans\\_freedrive\\HOLD to take it back
    uv run python scripts\\freedrive_splat_scan.py plan         the dome round where the camera looks, checked view by view (no motion)
    uv run python scripts\\freedrive_splat_scan.py run          lock focus, fly the plan, photograph every view

Same pieces as the app's UR5 panel - RTDE transport, scan_geometry's "splat" dome (every move
solved and collision-checked, unreachable views nudged to the nearest safe one or skipped) and
RealScanRunner (pictures plus a JSON record each, in robot_scans\\<session>) - without the panel,
so a scan can be set up by hand and then run unattended.

The dome is centred on the fabric point the camera looks at when the arm is held, at the distance it
was held at: that distance, then 1.5x and 2x it, each with a straight-down view and rings tilted up to
--max-tilt. `run` refuses if the arm was moved after `plan`, and creating robot_scans\\_freedrive\\STOP
halts the arm and ends the run (the pictures so far are kept).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import charuco  # noqa: E402
import hand_eye  # noqa: E402
import robot_camera  # noqa: E402
import scan_geometry  # noqa: E402
import ur5_robot  # noqa: E402
import ur5_scan  # noqa: E402
from persistence import write_json

WORK = ROOT / "robot_scans" / "_freedrive"
HOLD_FILE, STOP_FILE = WORK / "HOLD", WORK / "STOP"
HELD_JSON, PLAN_PICKLE, PLAN_JSON = WORK / "held.json", WORK / "plan.pkl", WORK / "plan.json"
FREE_DRIVE_LIMIT_S = 30 * 60
# The arm counts as unmoved since `plan` within this much on every joint.
UNMOVED_DEG = 0.5
FOCUS_RANGE, FOCUS_COARSE, FOCUS_FINE, FOCUS_SETTLE, FOCUS_FLAT = (0.0, 255.0), 15.0, 3.0, 0.4, 0.05


def settings() -> dict:
    """The UR5 panel's saved settings (config/params.json), which the plan must share with the app."""
    saved = json.loads((ROOT / "config" / "params.json").read_text(encoding="utf-8"))
    for key in ("gui_state", "params"):
        if isinstance(saved.get(key), dict) and "ur5_robot_ip" in saved[key]:
            return saved[key]
    return saved


def log(text: str) -> None:
    line = f"{time.strftime('%H:%M:%S')}  {text}"
    print(line, flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / "log.txt").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def connect(cfg: dict) -> ur5_robot.UR5Robot:
    robot = ur5_robot.UR5Robot(ur5_robot.RTDETransport())
    robot.speed_limit = float(cfg.get("ur5_speed_limit", 0.25))
    transport = robot.transport
    transport.max_joint_speed_deg = float(cfg.get("teleop_speed_deg_s", ur5_robot.DEFAULT_JOINT_SPEED_DEG))
    if hasattr(transport, "tool_offset"):
        transport.tool_offset = float(cfg.get("teleop_tool_offset", 0.0306))
    if not robot.connect(str(cfg.get("ur5_robot_ip", "192.168.0.101"))):
        raise SystemExit(f"could not connect: {robot.status}")
    # The runner waits for a move as long as it takes at the full joint speed; the speed slider slows
    # every move down to this share, so the wait is stretched to match (a longer move timed out).
    full_speed_ramp = transport.ramp_for
    transport.ramp_for = lambda joints: full_speed_ramp(joints) / max(robot.speed_limit, 0.05)
    time.sleep(0.5)
    return robot


def bgr(image) -> np.ndarray:
    """A camera frame (a PIL image) as the BGR array OpenCV draws on."""
    return np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])


def start_camera(cfg: dict) -> robot_camera.RobotCamera:
    camera = robot_camera.RobotCamera(robot_camera.make_backend({"backend": "opencv", "index": int(cfg.get("ur5_camera_device", 1))},
                                                                size=(640, 480)))
    if not camera.start():
        raise SystemExit(f"camera did not start: {camera.status}")
    time.sleep(1.5)
    return camera


def freedrive(cfg: dict) -> None:
    """The arm loose, with the camera's live picture in a window: pose it so the cloth is in the
    middle of the picture, then press H in the window (or create the HOLD file)."""
    import cv2

    for stale in (HOLD_FILE, STOP_FILE):
        stale.unlink(missing_ok=True)
    robot = connect(cfg)
    camera = None
    try:
        ok, issues = robot.readiness()
        if not ok:
            raise SystemExit("robot not ready: " + "; ".join(issues))
        try:
            camera = start_camera(cfg)
        except SystemExit as exc:
            log(f"no live preview ({exc}); hold with the HOLD file")
        board = charuco.load_board_spec()
        detector = charuco.Detector(board) if board is not None else None
        last_check, corners = 0.0, None
        robot.free_drive()
        log(f"FREE DRIVE ON - move the arm by hand; press H in the preview window or create {HOLD_FILE} to hold it "
            f"(auto-hold after {FREE_DRIVE_LIMIT_S // 60} min)")
        started = time.monotonic()
        while not HOLD_FILE.exists() and time.monotonic() - started < FREE_DRIVE_LIMIT_S:
            state = robot.state
            if not state.connected or state.protective_stopped or state.emergency_stopped:
                log(f"robot state changed during free drive: connected={state.connected} protective={state.protective_stopped} "
                    f"emergency={state.emergency_stopped}")
                break
            frame = None if camera is None else camera.latest_frame()
            if frame is not None:
                view = bgr(frame.image)
                h, w = view.shape[:2]
                cv2.line(view, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (0, 255, 255), 1)
                cv2.line(view, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (0, 255, 255), 1)
                cv2.putText(view, f"FREE DRIVE - cloth in the middle, then press H   sharpness {robot_camera.sharpness(frame.image):.0f}",
                            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                # The board is looked for twice a second (it takes ~20 ms), and how much of it shows is written in.
                if detector is not None and time.monotonic() - last_check > 0.5:
                    found = detector.detect(bgr(frame.image), min_corners=1)
                    corners = 0 if found is None else len(found["image"])
                    last_check = time.monotonic()
                if corners is not None:
                    cv2.putText(view, f"board corners seen: {corners}", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0) if corners >= charuco.MIN_CORNERS else (0, 0, 255), 1)
                cv2.imshow("Free drive - camera", view)
            if camera is not None and (cv2.waitKey(30) & 0xFF) in (ord("h"), ord("H")):
                break
            if camera is None:
                time.sleep(0.2)
        robot.hold()
        if camera is not None:
            frame = camera.latest_frame()
            if frame is not None:
                cv2.imwrite(str(WORK / "held_view.png"), bgr(frame.image))
            cv2.destroyAllWindows()
        HOLD_FILE.unlink(missing_ok=True)
        time.sleep(0.5)
        state = robot.refresh()
        held = {"joints": [float(v) for v in robot.joints_now()], "tcp_pose": [float(v) for v in state.tcp_pose],
                "held_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        write_json(HELD_JSON, held, indent=2)
        log(f"HELD - joints (deg) {np.round(np.degrees(held['joints']), 1).tolist()}, TCP {np.round(held['tcp_pose'][:3], 4).tolist()} m")
    finally:
        if camera is not None:
            camera.stop()
        robot.disconnect()


LIFT_SPEED = 0.02       # m/s: a slow, straight rise away from the cloth


def lift(cfg: dict, height: float) -> None:
    """Raise the tool straight up (base +z), attitude kept, so the run's first move starts clear of
    a cloth the arm was left a centimetre over - not a swing sideways out of that."""
    robot = connect(cfg)
    try:
        robot._guard("Lift")
        held = json.loads(HELD_JSON.read_text(encoding="utf-8"))
        moved = np.degrees(np.max(np.abs(np.asarray(robot.joints_now()) - np.asarray(held["joints"]))))
        if moved > UNMOVED_DEG:
            raise SystemExit(f"the arm is {moved:.1f} deg from where it was held - free drive and hold it again")
        start = [float(v) for v in robot.refresh().tcp_pose]
        target = start.copy()
        target[2] += float(height)
        control = robot.transport.control
        if not control.isPoseWithinSafetyLimits(target):
            raise SystemExit("the lifted pose is outside the controller's safety limits")
        log(f"LIFT {height * 100:.0f} cm straight up at {LIFT_SPEED * 100:.0f} cm/s from TCP z {start[2]:.4f}")
        control.moveL(target, LIFT_SPEED, 0.1, False)
        state = robot.refresh()
        if state.protective_stopped or state.emergency_stopped:
            raise SystemExit("the arm stopped during the lift")
        reached = np.linalg.norm(np.asarray(state.tcp_pose[:3]) - np.asarray(target[:3]))
        held = {"joints": [float(v) for v in robot.joints_now()], "tcp_pose": [float(v) for v in state.tcp_pose],
                "held_at": time.strftime("%Y-%m-%d %H:%M:%S"), "lifted_m": float(height), "subject_from": held.get("subject_from", held)}
        write_json(HELD_JSON, held, indent=2)
        log(f"LIFTED - TCP z {state.tcp_pose[2]:.4f} ({reached * 1000:.1f} mm from target)")
    finally:
        robot.disconnect()


def plan(cfg: dict, per_ring: int, rings: int, layers: int, max_tilt: float, radius: float | None, use_mount: bool,
         layer_step: float | None = None, layers_at: list[float] | None = None, mount_file: str = "",
         subject_at: list[float] | None = None) -> None:
    held = json.loads(HELD_JSON.read_text(encoding="utf-8"))
    joints = np.asarray(held["joints"], dtype=float)
    tool_offset = float(cfg.get("teleop_tool_offset", 0.0306))
    fabric_z = cfg.get("teleop_fabric_z")
    if fabric_z is None:
        raise SystemExit("the fabric height is not set: measure it once in the app (3 Capture) before planning")
    fabric_z = float(fabric_z)
    calibration = hand_eye.load_calibration() if use_mount else None
    mount = hand_eye.calibration_matrix(calibration) if calibration is not None else None
    if mount_file:
        # A mount measured today (from board views) outranks the saved one.
        mount = np.load(mount_file)
    mj, model, data, site_id = scan_geometry.load_model()
    subject_distance = float(cfg.get("teleop_subject_distance", 0.20))
    # The centre is what the camera looked at where it was held by hand - also after a lift away from it.
    aimed = np.asarray(held.get("subject_from", held)["joints"], dtype=float)
    aimed_pose = scan_geometry.seed_from(mj, model, data, site_id, aimed)
    camera = np.r_[scan_geometry.camera_point(aimed_pose, tool_offset), aimed_pose[3:6]]
    lens = scan_geometry._lens_pose(camera, mount) if mount is not None else camera
    subject = scan_geometry.fabric_point_below(lens[:3], lens[3:6], fabric_z, subject_distance)
    if subject_at:
        subject = np.asarray(subject_at, dtype=float)
    held_distance = float(np.linalg.norm(np.asarray(lens[:3]) - np.asarray(subject)))
    scan_geometry.seed_from(mj, model, data, site_id, joints)
    radius = float(min(layers_at)) if layers_at else float(radius) if radius else float(np.clip(held_distance, 0.07, 0.15))
    distances = ", ".join(f"{d * 100:.1f}" for d in layers_at) + " cm" if layers_at else f"{radius * 100:.1f} cm x {layers} distances"
    log(f"{'real lens (mount calibration)' if mount is not None else 'planner camera (no mount calibration)'} "
        f"{held_distance * 100:.1f} cm from the fabric point it looks at ({np.round(subject, 3).tolist()}); "
        f"dome at {distances}, {rings} rings of {per_ring}, tilt to {max_tilt:.0f} deg")
    started = time.monotonic()
    result = scan_geometry.build_capture_plan(
        joints, pattern="splat", subject_distance=subject_distance, tool_offset=tool_offset, camera_height=radius,
        rings=rings, splat_views_per_ring=per_ring, splat_layers=layers, max_tilt_deg=max_tilt, fabric_z=fabric_z,
        lens_hfov_deg=float(cfg.get("ur5_splat_fov", 38.0)), image_turn=cfg.get("ur5_camera_image_turn") or [[1.0, 0.0], [0.0, 1.0]],
        camera_mount=mount, splat_subject=subject if mount is not None else None, splat_layer_step=layer_step,
        splat_layer_radii=layers_at)
    safe, issues = result.assess()
    clearances = [min(v.clearance, v.arm_clearance) for v in result.viewpoints]
    lenses = [scan_geometry._lens_pose(np.r_[v.position, v.rotvec], mount) if mount is not None else np.r_[v.position, v.rotvec]
              for v in result.viewpoints]
    heights = [float(p[2] - fabric_z) for p in lenses]
    # How far off the dome's centre each view's lens axis meets the cloth: the dome works when this is small.
    misses = []
    for p in lenses:
        axis = scan_geometry._pose_matrix(p)[:3, 2]
        if axis[2] < -1e-6:
            hit = p[:3] + (fabric_z - p[2]) / axis[2] * axis
            misses.append(float(np.linalg.norm(hit[:2] - np.asarray(subject)[:2])))
    summary = {"views": len(result.viewpoints), "skipped": len(result.skipped), "safe": safe, "issues": issues[:5],
               "nudged": sum(1 for v in result.viewpoints if v.adjustment),
               "camera_height_cm": [round(min(heights) * 100, 1), round(max(heights) * 100, 1)],
               "lowest_clearance_mm": round(min(clearances) * 1000, 1),
               "lens_aim_off_centre_mm": [round(float(np.median(misses)) * 1000, 1), round(float(np.max(misses)) * 1000, 1)] if misses else None,
               "mount_calibration": bool(mount is not None),
               "subject": [float(v) for v in result.subject], "radius_m": radius, "planned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "from_joints": held["joints"], "skipped_views": [f"{name}: {reason}" for name, reason in result.skipped[:20]]}
    with PLAN_PICKLE.open("wb") as handle:
        pickle.dump(result, handle)
    write_json(PLAN_JSON, summary, indent=2)
    log(f"PLAN ({time.monotonic() - started:.0f} s to solve): {summary['views']} views, {summary['skipped']} skipped, "
        f"{summary['nudged']} nudged to a safe neighbour; camera {summary['camera_height_cm'][0]}-{summary['camera_height_cm'][1]} cm "
        f"over the fabric; lowest clearance {summary['lowest_clearance_mm']} mm; lens aim off centre "
        f"{summary['lens_aim_off_centre_mm']} mm (median, worst); safe={safe}")
    for line in summary["skipped_views"][:5]:
        log(f"  skipped {line}")


def lock_focus(camera: robot_camera.RobotCamera) -> str:
    """The app's Find best focus: sweep the lens, keep the sharpest, autofocus off."""
    scores = {}

    def judge(positions):
        for position in positions:
            camera.request_focus(False, float(position))
            frame = camera.capture(newer_than=time.monotonic() + FOCUS_SETTLE)
            if frame is None:
                raise RuntimeError("the camera delivered no frame")
            scores[float(position)] = robot_camera.sharpness(frame.image)

    try:
        judge(np.arange(FOCUS_RANGE[0], FOCUS_RANGE[1] + 1e-6, FOCUS_COARSE))
        coarse = max(scores, key=scores.get)
        judge(np.arange(max(FOCUS_RANGE[0], coarse - FOCUS_COARSE), min(FOCUS_RANGE[1], coarse + FOCUS_COARSE) + 1e-6, FOCUS_FINE))
    except Exception as exc:
        return f"focus not set ({exc}); the camera keeps its own"
    best = max(scores, key=scores.get)
    if scores[best] - min(scores.values()) < FOCUS_FLAT * max(scores[best], 1e-9):
        camera.request_focus(False, float(best))
        return f"focus control has no effect (fixed-focus camera?); sharpness {scores[best]:.0f}"
    camera.request_focus(False, float(best))
    camera.capture(newer_than=time.monotonic() + FOCUS_SETTLE)
    return f"focus locked at {best:.0f} (sharpness {scores[best]:.0f}, range {min(scores.values()):.0f}-{scores[best]:.0f})"


def run(cfg: dict, dwell: float, min_sharpness: float) -> None:
    STOP_FILE.unlink(missing_ok=True)
    with PLAN_PICKLE.open("rb") as handle:
        result = pickle.load(handle)
    summary = json.loads(PLAN_JSON.read_text(encoding="utf-8"))
    robot = connect(cfg)
    camera = None
    try:
        moved = np.degrees(np.max(np.abs(np.asarray(robot.joints_now()) - np.asarray(summary["from_joints"]))))
        if moved > UNMOVED_DEG:
            raise SystemExit(f"the arm moved {moved:.1f} deg since the plan was made from it - plan again")
        size = (640, 480)
        camera = start_camera(cfg)
        log(lock_focus(camera))
        runner = ur5_scan.RealScanRunner(robot, camera, result, None, output_root=ROOT / "robot_scans",
                                         settings_snapshot={"ur5_robot_ip": robot.ip, "ur5_capture_size": list(size),
                                                            "freedrive_splat_scan": summary},
                                         dwell=dwell, save_images=True, min_sharpness=min_sharpness, fabric_check=None,
                                         min_coverage=0.0, video=False)
        if not runner.start():
            raise SystemExit(f"the run did not start: {runner.progress.status}")
        log(f"RUN {runner.session_id}: {len(runner.scan_targets)} views - create {STOP_FILE} to stop")
        last = ""
        while runner.running:
            if STOP_FILE.exists():
                log("STOP requested")
                runner.stop()
                STOP_FILE.unlink(missing_ok=True)
            status = runner.progress.status
            if status != last:
                log(status)
                last = status
            time.sleep(0.5)
        progress = runner.progress
        log(f"DONE {runner.session_id}: {progress.captured} pictures; {progress.status}" + (f"; error: {progress.error}" if progress.error else ""))
    finally:
        if camera is not None:
            camera.stop()
        robot.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phase", choices=("freedrive", "lift", "plan", "run"))
    parser.add_argument("--lift", type=float, default=0.06, help="lift phase: how far straight up, m")
    parser.add_argument("--per-ring", type=int, default=20)
    parser.add_argument("--rings", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--max-tilt", type=float, default=40.0)
    parser.add_argument("--radius", type=float, default=None, help="dome radius in m (default: the distance the arm was held at)")
    parser.add_argument("--layer-step", type=float, default=None,
                        help="each further layer is this share of the radius further (default 0.5: 9, 13.5, 18 cm). "
                             "This fixed-focus camera is sharp only near 7-9 cm: --radius 0.07 --layer-step 0.14 gives 7, 8, 9 cm")
    parser.add_argument("--mount-file", default="", help="a 4x4 camera mount (.npy) measured today, instead of config/hand_eye.json")
    parser.add_argument("--subject", default="", help="the dome's centre, x,y,z in the arm's base frame (m), instead of where the held camera looked")
    parser.add_argument("--layers-at", default="",
                        help="the dome's distances outright, m, comma-separated - e.g. 0.085,0.09,0.095,0.15: three in the "
                             "sharp band and one far enough to see the board round the cloth")
    parser.add_argument("--no-mount", action="store_true", help="aim the planner's camera model instead of the calibrated real lens")
    parser.add_argument("--dwell", type=float, default=0.2)
    parser.add_argument("--min-sharpness", type=float, default=150.0)
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    cfg = settings()
    if args.phase == "freedrive":
        freedrive(cfg)
    elif args.phase == "lift":
        lift(cfg, args.lift)
    elif args.phase == "plan":
        layers_at = [float(v) for v in args.layers_at.split(",") if v.strip()] or None
        subject_at = [float(v) for v in args.subject.split(",") if v.strip()] or None
        plan(cfg, args.per_ring, args.rings, args.layers, args.max_tilt, args.radius, not args.no_mount, args.layer_step, layers_at,
             args.mount_file, subject_at)
    else:
        run(cfg, args.dwell, args.min_sharpness)


if __name__ == "__main__":
    main()
