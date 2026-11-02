"""TeleopApp as a UR5 transport, so the existing arm layer can drive it.

`ur5_robot` already abstracts the arm behind a transport - `RTDETransport` for the real
controller, `DryRunTransport` for tests. This adds a third: CraftLabXRFleet's TeleopApp,
reached over gRPC. Behind it `UR5Robot`, its `_guard`, `RealScanRunner`, the camera and the
storage all work unchanged.

**Only one transport may be connected at a time.** A UR controller grants a single RTDE
control session and TeleopApp holds it, so `RTDETransport` and this cannot both be live.

Two things about TeleopApp shape this class, both learned on the arm rather than from its
source:

- **`commandSending` is the only honest "the arm is following" flag.** TeleopApp stops
  commanding when a commanded pose runs more than 30 degrees ahead of the measurement; that
  clears `sendCommands` but leaves `trackingMode` still reading `"tracking"`. Reading the
  mode is what makes a panel claim it is armed while nothing moves.
- **Arming across a standing gap trips that same guard**, so `reset_stop` seeds the virtual
  robot onto the arm first and then verifies that commanding actually began.

TeleopApp has no Cartesian verb, so a pose is converted to joint angles by this project's
MuJoCo inverse kinematics (`scan_geometry`) and sent with `gotoJointAngles`.
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np
from PIL import Image

import paths
import robot_camera
import scan_geometry as geo
import ur5_robot

DEFAULT_ADDRESS = "localhost:50052"
DEFAULT_TARGET = "ur5"

# TeleopApp's gRPC stubs are generated code, regenerated on import by that project's own
# protostubs.py, so they are reached through its tree rather than vendored here. The
# checkout sits beside this one; both front ends take the default from here, so a panel and
# the command line cannot look in different places.
DEFAULT_FLEET_PYTHON = str(paths.PROJECT_ROOT.parent / "CraftLabXRFleet" / "python")

# TeleopApp serialises every RPC into its 90 Hz control loop, so each call costs a control
# tick. Poll state at this interval rather than per UI frame.
STATE_INTERVAL = 0.5

# Its deviation guard stops commanding when the command leads the measurement by this much.
GUARD_DEG = 30.0

# How often the simulated gripper camera is read. Each frame is a render TeleopApp writes to disk
# and this app reads back - about two of its frames plus a PNG round trip - so this is the preview's
# rate, and a capture waits for the next one after the arm settles like any camera.
CAMERA_INTERVAL = 0.5


def teleop_available() -> bool:
    """Whether TeleopApp could be reached from here at all: gRPC, and the stubs to speak it.

    Checking only for gRPC advertises the option and then fails on Connect with a missing
    `protostubs`, which reads as a broken app rather than a missing checkout.
    """
    try:
        import grpc  # noqa: F401
    except Exception:
        return False
    return os.path.isdir(DEFAULT_FLEET_PYTHON)


class TeleopClient:
    """One JSON-in, JSON-out call into TeleopApp's DynamicRpcServer."""

    def __init__(self, address, fleet_python):
        import sys

        if fleet_python and fleet_python not in sys.path:
            sys.path.insert(0, fleet_python)

        from protostubs import ensure_stubs

        self._pb, stubs = ensure_stubs()
        import grpc

        self._stub = stubs.DynamicRpcServiceStub(grpc.insecure_channel(address))

    def call(self, method, *args):
        request = self._pb.CommandRequest(method_name=method, payload_json=json.dumps(list(args)))
        response = self._stub.Execute(request, timeout=10.0)
        if not response.success:
            raise RuntimeError(f"{method}: {response.error_message}")
        return json.loads(response.payload_json) if response.payload_json else None


class TeleopAppTransport(ur5_robot.UR5Transport):
    """Drive the arm through a running TeleopApp."""

    name = "teleop"

    @property
    def is_real(self) -> bool:
        """Whether TeleopApp's UR5e is the arm itself rather than its emulator.

        TeleopApp picks one at startup, by whether the control box answers. On the emulator
        every move, rehearsed or run, happens only on screen, so its captures are not real.
        """
        return self.on_hardware

    def __init__(self, target=DEFAULT_TARGET, fleet_python=DEFAULT_FLEET_PYTHON, client_factory=TeleopClient, tool_offset=0.0306,
                 max_joint_speed_deg=12.0, floor_seconds=6.0):
        self.target = target
        self.tool_offset = float(tool_offset)

        # How a move is paced. TeleopApp takes a duration, not a speed, so the duration is
        # derived from the largest joint's travel - the same rule for a 5 degree nudge and a
        # 120 degree swing, which is what keeps the arm off its speed limits.
        self.max_joint_speed_deg = float(max_joint_speed_deg)
        self.floor_seconds = float(floor_seconds)
        self.speed_fraction = 1.0
        self.fleet_python = fleet_python
        self._client_factory = client_factory
        self.client = None
        self.address = ""

        # Read from getStatus, and the reason this class exists: the panel needs them and
        # RobotState has nowhere to put them.
        self.following = False
        self.gap_deg = None
        self.trajectory_running = False
        self.pose_names: list[str] = []
        self.on_hardware = False
        self.last_error = ""

        self._model = None
        self._status = {}
        self._read_at = 0.0

    # -- connection --------------------------------------------------------

    def connect(self, ip: str) -> None:
        """Open the channel and load the inverse kinematics model.

        `ip` is TeleopApp's address, not the robot's - the app owns the controller. A bare
        host or empty string falls back to the default port.
        """
        self.address = str(ip) if ":" in str(ip) else (f"{ip}:50052" if ip else DEFAULT_ADDRESS)
        self.client = self._client_factory(self.address, self.fleet_python)

        described = self.client.call("describe")
        targets = (described or {}).get("targets", {})
        if self.target not in targets:
            raise RuntimeError(f"TeleopApp has no target `{self.target}` - it offers {sorted(targets)}")

        self._model = geo.load_model()
        self._refresh(force=True)

    def cameras(self) -> list[str]:
        """The cameras TeleopApp mounts on this target - what the camera section can open."""
        if self.client is None:
            return []
        return list(((self.client.call("describe") or {}).get("targets", {}).get(self.target, {})).get("cameras", []))

    def linked(self) -> bool:
        """Answering over gRPC. Its UR5e may still be down; `connected` in its status says that."""
        return self.client is not None and bool(self._status)

    def disconnect(self) -> None:
        self.client = None
        self._model = None
        self._status = {}
        self.following = False

    # -- state -------------------------------------------------------------

    def _refresh(self, force=False):
        """One getStatus, rate limited because each call costs TeleopApp a control tick."""
        if self.client is None:
            return {}
        if not force and time.monotonic() - self._read_at < STATE_INTERVAL:
            return self._status

        self._status = self.client.call("getStatus", self.target) or {}
        self._read_at = time.monotonic()

        self.following = bool(self._status.get("commandSending"))
        self.trajectory_running = bool(self._status.get("trajectoryExecuting"))
        self.on_hardware = bool(self._status.get("realHardware"))
        self.pose_names = list(self._status.get("poses", []))

        # -1 is TeleopApp's "not evaluable" sentinel, not a gap of minus one degree.
        reported = self._status.get("trackingGapDeg", -1.0)
        self.gap_deg = None if reported is None or reported < 0.0 else float(reported)
        return self._status

    def read_state(self) -> ur5_robot.RobotState:
        status = self._refresh()
        if not status:
            return ur5_robot.RobotState(connected=False, error=self.last_error, updated_at=time.monotonic())

        connected = bool(status.get("connected"))
        return ur5_robot.RobotState(
            connected=connected,
            # TeleopApp does not pass the controller's own mode through, so these are
            # synthesised from what it does report: connected and torqued reads as running.
            robot_mode=7 if connected and status.get("motorsOn") else (4 if connected else 0),
            safety_mode=1 if connected else 0,
            # The TOOL pose in the frame the plan is written in, not TeleopApp's world.
            # UR5Robot.at_pose compares a commanded 6-vector against this one, and the
            # commanded poses come from the MuJoCo model; TeleopApp's own currentPose is a
            # different frame (its world mirrors X) and carries no rotation here, so using
            # it would leave the rotation error at ~3 rad against a 0.05 tolerance and the
            # runner would time out at every viewpoint.
            tcp_pose=list(self.tool_pose()),
            joints=[float(status.get("q", {}).get(joint, 0.0)) for joint in geo.TELEOP_JOINTS],
            protective_stopped=False,
            emergency_stopped=False,
            program_running=self.following,
            speed_scale=1.0,
            updated_at=time.monotonic(),
            error=self.last_error or ("" if connected else
                                      "TeleopApp answers, but its UR5e is not connected - it may be reconnecting, "
                                      "or the arm is off; see TeleopApp's own window"))

    def tool_pose(self):
        """Forward kinematics of the measured joints, at the tool point, in the plan's frame.

        The model's attachment site is the flange, and every planned position is a tool
        point, so the offset forward is what makes the two comparable.
        """
        if self._model is None:
            return np.zeros(6)

        from scipy.spatial.transform import Rotation

        mj, model, data, site_id = self._model
        flange = geo.seed_from(mj, model, data, site_id, self.measured_joints())
        rotvec = np.asarray(flange[3:6], dtype=float)
        position = np.asarray(flange[:3], dtype=float) + Rotation.from_rotvec(rotvec).apply([0.0, 0.0, self.tool_offset])
        return np.r_[position, rotvec]

    def app_tool_point(self):
        """Where TeleopApp thinks the tool is, in ITS world - for display only."""
        pose = self._refresh().get("currentPose", {})
        return [pose.get(key, 0.0) for key in ("px", "py", "pz")]

    def measured_joints(self):
        return [float(self._refresh().get("q", {}).get(joint, 0.0)) for joint in geo.TELEOP_JOINTS]

    def planned_joints(self):
        return [float(self._refresh().get("plannedQ", {}).get(joint, 0.0)) for joint in geo.TELEOP_JOINTS]

    # -- motion ------------------------------------------------------------

    def ramp_for(self, joints) -> float:
        """Seconds this move will take, from how far the furthest joint has to travel."""
        return geo.move_seconds(self.measured_joints(), joints,
                                self.max_joint_speed_deg * self.speed_fraction, self.floor_seconds)

    def move_to_joints(self, joints, speed: float, acceleration: float) -> None:
        """Send a configuration, paced by its own size.

        `speed` and `acceleration` are ignored, and have to be: they arrive in metres per
        second from a Cartesian interface, and a joint move has no Cartesian speed. Handing
        TeleopApp a number derived from them puts a 120 degree swing on a half-second ramp.
        """
        ramp = self.ramp_for(joints)
        self._await_idle(geo.timeout_for(ramp))
        self.client.call("setParams", self.target, {"rampTimeS": ramp})
        self.client.call("gotoJointAngles", self.target,
                         {joint: float(value) for joint, value in zip(geo.TELEOP_JOINTS, joints)})

    def move_to_pose(self, pose, speed: float, acceleration: float) -> None:
        """Solve the pose against the arm's current configuration, then send the joints.

        `pose` names where the TOOL should be, matching what tool_pose() reports and what the
        planner produces, so it is converted to a flange target before solving. Solving it as
        a flange target instead parks the tool one tool-length beyond the pose asked for, and
        arrival then never registers.
        """
        mj, model, data, site_id = self._model
        asked = np.asarray(pose, dtype=float).reshape(-1)[:6]
        flange = geo.flange_for_grasp(asked[:3], asked[3:6], self.tool_offset)

        geo.seed_from(mj, model, data, site_id, self.measured_joints())
        config, _, _ = geo.solve_ik(mj, model, data, site_id, np.r_[flange, asked[3:6]], "pose")
        self.move_to_joints(config, speed, acceleration)

    def rehearsable(self) -> bool:
        """TeleopApp always drives its virtual robot; it drives the arm only while following one.

        This is the simulator: a rehearsal moves the 3D UR5e in TeleopApp through the checked
        path at the pace a run would take, and the arm stays where it is. Following its
        emulator moves nothing physical either.
        """
        return not (self.following and self.on_hardware)

    def blockers(self) -> list[str]:
        """Not following is a blocker, and the one this transport exists to report.

        A `gotoJointAngles` sent while TeleopApp is not commanding is accepted, moves the
        virtual robot, and leaves the arm exactly where it was. Reporting it here is what
        turns that silence into a refusal.
        """
        if self.client is None:
            return []
        if not self._refresh().get("connected"):
            return ["TeleopApp is not connected to the arm"]
        if not self.following:
            return ["TeleopApp is not commanding the arm; arm it with Reset"]
        return []

    def move_home(self, joints, speed: float, acceleration: float) -> None:
        """The taught `Home` pose when the arm has one, which is the point of teaching it.

        Home is where the simulated and the real arm are defined to agree, so sending a
        configuration from this side instead would put the arm somewhere near Home and call
        it Home. Only an arm with no taught Home falls back to the joints.
        """
        if "Home" in self.pose_names:
            self.go_to_named_pose("Home")
            return
        self.move_to_joints(joints, speed, acceleration)

    def stop(self, deceleration: float = 2.0) -> None:
        """Stop following. A software stop, not the hardware emergency stop."""
        try:
            self.client.call("setTrackingMode", self.target, "holding", True)
        except Exception as exc:
            # Refused mid-trajectory; the move it is finishing is the one already commanded.
            self.last_error = str(exc)

    def protective_stop(self) -> None:
        self.stop()

    def free_drive(self) -> None:
        """TeleopApp's `free`, which on the UR5e is ur_rtde teach mode: loose, but holding against gravity."""
        self.client.call("setTrackingMode", self.target, "free", False)

    def hold(self) -> None:
        """Back to `holding`: motors on, where the hand left it, following nothing.

        Refused when teach mode will not end - a long hand-guiding session can outlive the
        control script - and that refusal is TeleopApp's to raise, so it reaches the panel.
        """
        self.client.call("setTrackingMode", self.target, "holding", True)
        self._refresh(force=True)

    def joints_now(self) -> list[float]:
        """A fresh read, past the rate limit: a taught view must be where the arm is, not was."""
        self._refresh(force=True)
        return self.measured_joints()

    def reset_stop(self) -> None:
        """Arm the arm: seed, energize, track, then check that commanding actually began."""
        status = self._refresh(force=True)
        measured = {joint: float(status.get("q", {}).get(joint, 0.0)) for joint in geo.TELEOP_JOINTS}

        gap = self.gap_deg
        if gap is None or gap > 1.0:
            self.client.call("gotoJointAngles", self.target, measured)
            self._await_idle(15.0)

        self.client.call("setMotors", self.target, True)
        self.client.call("setTrackingMode", self.target, "tracking", True)

        time.sleep(0.5)
        self._refresh(force=True)
        if not self.following:
            raise RuntimeError(
                "TeleopApp is not commanding the arm after arming. Its deviation guard stops commands when the "
                f"commanded pose runs more than {GUARD_DEG:.0f} deg ahead of the measurement - which happens when "
                "the arm stands where the planner will not hold it, near a singularity. Hand-guide it closer to "
                "Home in free drive, then arm again.")

    def set_speed_scale(self, fraction: float) -> None:
        """The panel's speed limit, applied to the joint speed a move is paced by."""
        self.speed_fraction = float(np.clip(float(fraction), 0.01, 1.0))

    def pose_within_limits(self, pose) -> bool:
        """Reachable and clear of self-collision, checked against the model rather than tried."""
        if self._model is None:
            return True
        mj, model, data, site_id = self._model
        asked = np.asarray(pose, dtype=float).reshape(-1)[:6]
        flange = geo.flange_for_grasp(asked[:3], asked[3:6], self.tool_offset)

        geo.seed_from(mj, model, data, site_id, self.measured_joints())
        try:
            config, _, _ = geo.solve_ik(mj, model, data, site_id, np.r_[flange, asked[3:6]], "pose")
        except geo.Unreachable:
            return False
        return geo.contacts_at(mj, model, data, config) == 0

    # -- helpers the panel and the runner use ------------------------------

    def go_to_named_pose(self, name) -> None:
        self._await_idle(30.0)
        self.client.call("gotoPose", self.target, name)

    def arrived(self, joints, tolerance=0.01) -> bool:
        measured = np.asarray(self.measured_joints(), dtype=float)
        return bool(np.max(np.abs(measured - np.asarray(joints, dtype=float))) < tolerance
                    and not self.trajectory_running)

    def _await_idle(self, timeout) -> bool:
        """Wait out a ramp: gotoJointAngles is refused while a trajectory is running."""
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            status = self._refresh(force=True)
            if not status.get("trajectoryExecuting") and not status.get("engaging"):
                return True
            time.sleep(0.1)
        return False


class TeleopCameraBackend(robot_camera.CameraBackend):
    """A camera TeleopApp mounts on the arm, read as frames it renders from its own 3D cell.

    This is the simulator's camera: with TeleopApp on its emulator, what it sees is the virtual
    arm's view of the simulated table and fabric (sim_cell), so a whole run - preview, captures,
    their checks and their storage - runs without hardware. Frames are marked is_real=False, like
    the synthetic camera's, so a simulated capture is never taken for a real one.

    TeleopApp renders the view at its window's size with the camera's own field of view and aspect,
    and reports that aspect with each capture. Each frame is resized to `size`'s width at that
    aspect - never to `size`'s height, which squashed a 16:9 lens into 4:3 frames that no single
    pinhole explains, and blurred every splat trained on them.
    """

    name = "teleop"
    is_real = False

    def __init__(self, transport, camera_name: str, size=robot_camera.DEFAULT_CAPTURE_SIZE, interval: float = CAMERA_INTERVAL):
        self.transport = transport
        self.camera_name = str(camera_name)
        self.size = (int(size[0]), int(size[1]))
        self.interval = float(interval)
        self.frame_path = paths.SIM_CELL_DIR / "frames" / "gripper_camera.png"
        self._read_at = 0.0

    def open(self) -> None:
        if self.transport.client is None:
            raise RuntimeError("connect Via TeleopApp first - this camera is TeleopApp's")
        if self.camera_name not in self.transport.cameras():
            raise RuntimeError(f"TeleopApp has no camera `{self.camera_name}` on `{self.transport.target}` - "
                               "build TeleopApp with the UR5e camera tool, which mounts one")
        self.frame_path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> np.ndarray | None:
        wait = self._read_at + self.interval - time.monotonic()
        if wait > 0.0:
            time.sleep(wait)
        self._read_at = time.monotonic()
        drawn = self.transport.client.call("captureCameraImage", self.transport.target, self.camera_name, str(self.frame_path.resolve()))
        size = (self.size[0], int(round(self.size[0] / float(drawn["aspect"]))))
        with Image.open(self.frame_path) as frame:
            return np.asarray(frame.convert("RGB").resize(size, Image.BILINEAR), dtype=np.uint8)

    def close(self) -> None:
        pass

    def describe(self) -> str:
        return f"TeleopApp {self.camera_name} ({self.size[0]}x{self.size[1]}, simulated)"
