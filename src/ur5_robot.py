"""Connection and safety layer for the real UR5.

Scan Mode's simulated UR5e can be commanded freely: a bad pose costs a redraw.
This module drives the physical arm, so its shape is different in two ways.

Feedback, not fire-and-forget. The existing ``fabric_scanner.run_robot_motion``
sends a whole URScript program down a socket and hopes; that is fine for
replaying a path, but a scan has to photograph the fabric at each stop, which
means knowing when the arm actually arrived. Motion here goes over RTDE, which
reports live pose and status back.

Every motion passes a safety gate. ``_guard`` is the single door all commands go
through, so a stop, a pause, a protective stop or a lost connection halts
everything -- there is no path to the arm that skips it.

One honest limit, surfaced in the UI as well: the software stop triggers the
controller's protective stop, which is a commanded decelerate-and-hold. It is
not the hardware emergency stop and does not replace it. The physical e-stop
button remains the actual safety device.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

import scan_geometry


# UR controller enumerations, so status text reads in words rather than codes.
ROBOT_MODES = {
    -1: "no controller",
    0: "disconnected",
    1: "confirm safety",
    2: "booting",
    3: "power off",
    4: "power on",
    5: "idle",
    6: "backdrive",
    7: "running",
    8: "updating firmware",
}

SAFETY_MODES = {
    1: "normal",
    2: "reduced",
    3: "protective stop",
    4: "recovery",
    5: "safeguard stop",
    6: "system emergency stop",
    7: "robot emergency stop",
    8: "violation",
    9: "fault",
}

# Safety modes in which no motion may be commanded at all.
BLOCKING_SAFETY_MODES = {3, 4, 5, 6, 7, 8, 9}
# The controller must be in "running" mode (7) for RTDE motion to execute.
READY_ROBOT_MODE = 7
# RTDE runtime state while any program runs on the controller: TeleopApp's
# control script, a pendant program, or a script another client left behind.
RUNTIME_STATE_PLAYING = 2

DEFAULT_RTDE_PORT = 30004
# How often the controller streams state to this app. ur_rtde's default on an e-Series is
# 500 Hz, and the controller drops an RTDE client that falls behind reading it - which a laptop
# busy with anything else (the fabric check's model, say) does. Nothing here needs more than
# this: moves are joint moves the controller interpolates, not a servo stream. TeleopApp reads
# the same arm at the same rate.
RTDE_FREQUENCY_HZ = 125.0
# The controller's dashboard server: plain text, one command and one reply per line. It is
# how a program left running is named, and stopped, without opening a control session.
DASHBOARD_PORT = 29999
DASHBOARD_TIMEOUT = 3.0
# How long a stopped program is given to actually leave PLAYING before connecting again.
STOP_SETTLE_TIMEOUT = 5.0
# The primary interface, where a URScript program sent replaces the one running.
SCRIPT_PORT = 30001
# What replaces a stray script: bring the arm to rest, then end, leaving nothing running.
STOP_SCRIPT = "def knitting_stop():\n  stopj(2.0)\nend\n"
# Conservative defaults for scanning near fabric. Deliberately far below the
# arm's capability: this moves a camera between close-set targets.
DEFAULT_SPEED = 0.06
DEFAULT_ACCELERATION = 0.30
MAX_SPEED = 0.25
MAX_ACCELERATION = 0.80
# The `Home` taught on this cell's arm in TeleopApp (config/ur5_saved_poses.json), in UR
# joint order - TeleopApp stores the joints alphabetically, so shoulder_pan and elbow
# trade places. The camera looks straight down at the fabric from here, 20 cm above it,
# which is why the capture check reads the fabric's height off this pose. It replaced the
# upright [0, -90, 0, -90, 0, 0] candle, which looks sideways at nothing.
DEFAULT_HOME_JOINTS = list(scan_geometry.DEFAULT_HOME_JOINTS)
POSE_REACHED_TOL = 0.003
POSE_REACHED_ROT_TOL = 0.05
# Largest joint error, in radians, at which a joint move counts as arrived.
JOINTS_REACHED_TOL = 0.01
# Default peak joint speed for a Direct RTDE joint move, until the panel's own
# setting is pushed onto the transport.
DEFAULT_JOINT_SPEED_DEG = 12.0


@dataclass
class RobotState:
    """Snapshot of what the controller last reported."""

    connected: bool = False
    robot_mode: int = -1
    safety_mode: int = 0
    tcp_pose: list[float] = field(default_factory=lambda: [0.0] * 6)
    joints: list[float] = field(default_factory=lambda: [0.0] * 6)
    protective_stopped: bool = False
    emergency_stopped: bool = False
    program_running: bool = False
    speed_scale: float = 1.0
    updated_at: float = 0.0
    error: str = ""

    @property
    def robot_mode_label(self) -> str:
        return ROBOT_MODES.get(int(self.robot_mode), f"mode {self.robot_mode}")

    @property
    def safety_mode_label(self) -> str:
        return SAFETY_MODES.get(int(self.safety_mode), f"safety {self.safety_mode}")

    def summary(self) -> str:
        if not self.connected:
            return "Disconnected"
        return f"{self.robot_mode_label} / {self.safety_mode_label}"


# ============================================================================
# Transports
# ============================================================================

class ProgramAlreadyRunning(RuntimeError):
    """The controller is running a program, so a control session would replace it.

    Carries what the dashboard says is loaded, so the panel can name it and offer to stop
    it - the one way past this that does not mean walking to the pendant.
    """

    def __init__(self, ip, program, state):
        self.ip, self.program, self.state = str(ip), str(program), str(state)
        super().__init__(
            f"{self.describe()}. Connecting would replace it. Stop it here, or on the pendant - "
            "or, if TeleopApp is still running, use Via TeleopApp")

    @property
    def sent_over_network(self) -> bool:
        """RTDE says a program runs while the dashboard says none plays: a script sent to the
        controller's socket, which the dashboard never lists - most often ur_rtde's control
        script, left by a client that closed without stopping it."""
        return not self.state.upper().startswith("PLAYING")

    def describe(self) -> str:
        if self.sent_over_network:
            return ("a script sent over the network is running on the arm - no pendant program is playing; "
                    "most likely the control script TeleopApp or an earlier session left behind")
        return f"the pendant program {self.program or '(unnamed)'} is playing on the arm"


def dashboard(ip: str, command: str) -> str:
    """One dashboard command, and the controller's one-line reply."""
    import socket

    with socket.create_connection((str(ip), DASHBOARD_PORT), timeout=DASHBOARD_TIMEOUT) as link:
        reader = link.makefile("r", encoding="utf-8", newline="\n")
        reader.readline()
        link.sendall((command + "\n").encode("utf-8"))
        return reader.readline().strip()


def runtime_state(ip: str) -> int:
    """The controller's RTDE runtime state - the one signal that sees scripts as well as programs."""
    import rtde_receive

    receive = rtde_receive.RTDEReceiveInterface(str(ip), RTDE_FREQUENCY_HZ)
    try:
        return int(receive.getRuntimeState())
    finally:
        receive.disconnect()


def send_script(ip: str, script: str) -> None:
    """Send a URScript program to the controller's primary interface, which replaces whatever runs."""
    import socket

    with socket.create_connection((str(ip), SCRIPT_PORT), timeout=DASHBOARD_TIMEOUT) as link:
        link.sendall(script.encode("utf-8"))


def _settles(ip: str, runtime) -> bool:
    deadline = time.monotonic() + STOP_SETTLE_TIMEOUT
    while time.monotonic() < deadline:
        if runtime(ip) != RUNTIME_STATE_PLAYING:
            return True
        time.sleep(0.2)
    return False


def stop_program(ip: str, runtime=runtime_state) -> str:
    """Stop whatever the controller is running - pendant program or stray script - and prove it.

    First the dashboard's `stop`, which ends a pendant program. A script sent over the network
    may outlive it, so if RTDE still says PLAYING it is replaced by one that only brings the arm
    to rest and ends. Judged by RTDE's runtime state throughout: the dashboard does not see
    scripts, which is how a stop that did nothing once read as done.
    """
    reply = dashboard(ip, "stop")
    if _settles(ip, runtime):
        return reply
    send_script(ip, STOP_SCRIPT)
    if _settles(ip, runtime):
        return "the stray script was replaced by a stop and has ended"
    raise RuntimeError(f"it is still running after the dashboard stop ({reply}) and after being replaced - "
                       "stop it on the pendant, or power-cycle the controller")


class UR5Transport:
    """What the robot layer needs from whatever is on the other end."""

    name = "base"
    is_real = False

    def connect(self, ip: str) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    def read_state(self) -> RobotState:
        raise NotImplementedError

    def move_to_pose(self, pose, speed: float, acceleration: float) -> None:
        """Starts a linear move. Returns without waiting for arrival."""
        raise NotImplementedError

    def move_to_joints(self, joints, speed: float, acceleration: float) -> None:
        raise NotImplementedError

    def ramp_for(self, joints) -> float:
        """Seconds a move to `joints` will take, for a caller that has to wait it out."""
        return 0.0

    def linked(self) -> bool:
        """Whether this app has a live session to read the arm through - whether or not the arm
        behind it is up. A path can be planned from a live link; only a live arm can run it."""
        return False

    def rehearsable(self) -> bool:
        """Whether a move can be sent without the arm moving, which is what a rehearsal is.

        True only where something else executes the motion: a dry run, or a transport whose
        far side holds a virtual robot that is not currently driving the arm. On a transport
        wired straight to a controller a "rehearsal" is an ordinary move, so it is not one.
        """
        return not self.is_real

    def blockers(self) -> list[str]:
        """What stops this transport's commands from reaching the arm, beyond RobotState.

        RobotState carries what every arm has - connected, powered, stopped. A transport
        with a condition of its own says so here, and readiness() folds it in, so that the
        guard refuses a move that would be accepted and then quietly do nothing.
        """
        return []

    def move_home(self, joints, speed: float, acceleration: float) -> None:
        """Where "start from home" sends the arm. Joints, unless a transport knows better."""
        self.move_to_joints(joints, speed, acceleration)

    def free_drive(self) -> None:
        """Make the arm hand-guidable: gravity-compensated, holding wherever it is left."""
        raise NotImplementedError(f"the {self.name} transport has no free drive")

    def hold(self) -> None:
        """End free drive: motors take the arm back and hold it where the hand left it."""
        raise NotImplementedError(f"the {self.name} transport has no free drive")

    def joints_now(self) -> list[float]:
        """The measured joints as of this call, not as of the last poll - what a taught view records."""
        return list(self.read_state().joints)

    def stop(self, deceleration: float = 2.0) -> None:
        raise NotImplementedError

    def protective_stop(self) -> None:
        raise NotImplementedError

    def reset_stop(self) -> None:
        raise NotImplementedError

    def set_speed_scale(self, fraction: float) -> None:
        raise NotImplementedError

    def pose_within_limits(self, pose) -> bool:
        return True


class RTDETransport(UR5Transport):
    """The real arm, over ur_rtde."""

    name = "rtde"
    is_real = True

    def __init__(self):
        self.control = None
        self.receive = None
        self.io = None
        self.ip = ""
        self.max_joint_speed_deg = DEFAULT_JOINT_SPEED_DEG

    def connect(self, ip: str) -> None:
        import rtde_control
        import rtde_receive

        self.ip = str(ip)
        # Receive first: it is read-only, so a wrong address or an unreachable
        # controller fails here without ever having opened a control channel
        # that could command motion.
        self.receive = rtde_receive.RTDEReceiveInterface(self.ip, RTDE_FREQUENCY_HZ)
        try:
            # Uploading the control script replaces whatever program is running,
            # and a control interface whose constructor fails because another
            # client holds the arm takes the whole process down inside ur_rtde,
            # past any except. So a busy controller is refused before either.
            if self.receive.getRuntimeState() == RUNTIME_STATE_PLAYING:
                try:
                    program, state = dashboard(self.ip, "get loaded program"), dashboard(self.ip, "programState")
                except Exception:
                    program, state = "", ""
                raise ProgramAlreadyRunning(self.ip, program.split(":", 1)[-1].strip(), state)
            self.control = rtde_control.RTDEControlInterface(self.ip, RTDE_FREQUENCY_HZ)
        except Exception:
            self.disconnect()
            raise
        try:
            import rtde_io

            self.io = rtde_io.RTDEIOInterface(self.ip)
        except Exception:
            # Speed-slider control is a convenience; losing it must not block
            # a connection that is otherwise good.
            self.io = None

    def disconnect(self) -> None:
        # ur_rtde's disconnect leaves its control script running, which reads as
        # a busy arm to the next client, this app included.
        if self.control is not None:
            try:
                self.control.stopScript()
            except Exception as exc:
                logging.warning("ur5_robot.disconnect: %s", exc)
        for handle in (self.control, self.receive, self.io):
            if handle is None:
                continue
            try:
                handle.disconnect()
            except Exception as exc:
                logging.warning("ur5_robot.disconnect: %s", exc)
        self.control = None
        self.receive = None
        self.io = None

    def read_state(self) -> RobotState:
        if self.receive is None:
            return RobotState(connected=False)
        # The receive stream only reads, so a dropped one is simply reopened. The control
        # session is not: re-establishing it re-uploads the control script, which is a decision
        # for the user after a stop, not something to do under a running scan.
        try:
            if not self.receive.isConnected():
                self.receive.reconnect()
        except Exception as exc:
            return RobotState(connected=False, error=f"the RTDE state stream dropped and would not reopen: {exc}",
                              updated_at=time.monotonic())
        try:
            state = RobotState(
                connected=bool(self.receive.isConnected()),
                robot_mode=int(self.receive.getRobotMode()),
                safety_mode=int(self.receive.getSafetyMode()),
                tcp_pose=[float(v) for v in self.receive.getActualTCPPose()],
                joints=[float(v) for v in self.receive.getActualQ()],
                protective_stopped=bool(self.receive.isProtectiveStopped()),
                emergency_stopped=bool(self.receive.isEmergencyStopped()),
                updated_at=time.monotonic(),
            )
        except Exception as exc:
            return RobotState(connected=False, error=str(exc), updated_at=time.monotonic())
        try:
            state.speed_scale = float(self.receive.getSpeedScaling())
        except Exception as exc:
            logging.warning("ur5_robot.read_state: %s", exc)
        try:
            state.program_running = bool(self.control.isProgramRunning()) if self.control else False
        except Exception as exc:
            logging.warning("ur5_robot.read_state: %s", exc)
        return state

    def move_to_pose(self, pose, speed: float, acceleration: float) -> None:
        if self.control is None:
            raise RuntimeError("Not connected to the robot")
        # asynchronous=True so the caller keeps polling state (and the UI keeps
        # redrawing) while the arm moves, instead of blocking inside the driver.
        self.control.moveL([float(v) for v in pose], float(speed), float(acceleration), True)

    def move_to_joints(self, joints, speed: float, acceleration: float) -> None:
        """A joint move at the panel's joint speed.

        `speed` and `acceleration` are ignored, as in TeleopAppTransport: they are the
        Cartesian m/s every caller passes, and moveJ reads its arguments as rad/s, so
        passing them through crawls a scan at under a degree a second. The controller's
        speed slider, which apply_speed_limit sets, still scales the move.
        """
        if self.control is None:
            raise RuntimeError("Not connected to the robot")
        joint_speed = math.radians(self.max_joint_speed_deg)
        self.control.moveJ([float(v) for v in joints], joint_speed, joint_speed, True)

    def linked(self) -> bool:
        try:
            return self.receive is not None and bool(self.receive.isConnected())
        except Exception:
            return False

    def blockers(self) -> list[str]:
        """The state stream can be up while the control session is gone, and then every move fails."""
        try:
            alive = self.control is not None and bool(self.control.isConnected())
        except Exception:
            alive = False
        return [] if alive else ["the RTDE control session dropped - Disconnect, then Connect again"]

    def free_drive(self) -> None:
        if self.control is None:
            raise RuntimeError("Not connected to the robot")
        if not self.control.teachMode():
            raise RuntimeError("the controller refused teach mode - check the pendant for a popup")

    def hold(self) -> None:
        if self.control is None:
            raise RuntimeError("Not connected to the robot")
        if not self.control.endTeachMode():
            raise RuntimeError("teach mode would not end, so the arm is still loose - the control script has "
                               "probably stopped; press Reset, then Hold again")

    def ramp_for(self, joints) -> float:
        """moveJ's trapezoid: cruise the furthest joint's travel, plus one second to ramp."""
        if self.receive is None:
            return 0.0
        largest = float(np.max(np.abs(np.asarray(joints, dtype=float) - np.asarray(self.receive.getActualQ(), dtype=float))))
        return largest / math.radians(self.max_joint_speed_deg) + 1.0

    def stop(self, deceleration: float = 2.0) -> None:
        if self.control is None:
            return
        self.control.stopL(float(deceleration))

    def protective_stop(self) -> None:
        if self.control is None:
            return
        # Decelerate first, then latch the controller into protective stop.
        # Order matters: the stop command is what actually arrests motion, and
        # it must be issued before anything that can throw.
        failures = []
        for name, action in (('decelerate', lambda: self.control.stopL(4.0)),
                             ('protective stop', self.control.triggerProtectiveStop)):
            try:
                action()
            except Exception as exc:
                failures.append(f'{name}: {exc}')
        if failures:
            raise RuntimeError('; '.join(failures))

    def reset_stop(self) -> None:
        if self.control is None:
            return
        self.control.reuploadScript()

    def set_speed_scale(self, fraction: float) -> None:
        fraction = float(np.clip(float(fraction), 0.01, 1.0))
        if self.io is not None:
            self.io.setSpeedSlider(fraction)

    def pose_within_limits(self, pose) -> bool:
        if self.control is None:
            return True
        return bool(self.control.isPoseWithinSafetyLimits([float(v) for v in pose]))


class DryRunTransport(UR5Transport):
    """A stand-in arm that reports arriving where it was told to go - the tests' arm.

    Lets the whole UR5 workflow -- connect, plan, move, capture, store, analyse
    -- be exercised with no hardware. The panel no longer offers it: it shows no
    motion, so simulating is done on TeleopApp's 3D arm instead. It reports
    ``is_real=False``, which travels into the database alongside the captures so
    a test run is never mistaken for a real scan.
    """

    name = "dry-run"
    is_real = False

    def __init__(self, travel_time: float = 0.25):
        self.travel_time = float(travel_time)
        self._pose = np.array([-0.45, -0.08, 0.30, math.pi, 0.0, 0.0], dtype=float)
        self._target = self._pose.copy()
        self._move_started = 0.0
        self._joints = np.array(DEFAULT_HOME_JOINTS, dtype=float)
        self._connected = False
        self._stopped = False
        self._speed_scale = 1.0

    def connect(self, ip: str) -> None:
        self._connected = True
        self._stopped = False

    def disconnect(self) -> None:
        self._connected = False

    def _advance(self) -> None:
        if self._move_started <= 0.0:
            return
        elapsed = time.monotonic() - self._move_started
        t = 1.0 if self.travel_time <= 0 else min(1.0, elapsed / self.travel_time)
        self._pose = self._pose + (self._target - self._pose) * t
        if t >= 1.0:
            self._pose = self._target.copy()
            self._move_started = 0.0

    def read_state(self) -> RobotState:
        self._advance()
        return RobotState(
            connected=self._connected,
            robot_mode=READY_ROBOT_MODE if self._connected else 0,
            safety_mode=3 if self._stopped else 1,
            tcp_pose=[float(v) for v in self._pose],
            joints=[float(v) for v in self._joints],
            protective_stopped=self._stopped,
            emergency_stopped=False,
            program_running=self._move_started > 0.0,
            speed_scale=self._speed_scale,
            updated_at=time.monotonic(),
        )

    def move_to_pose(self, pose, speed: float, acceleration: float) -> None:
        if not self._connected:
            raise RuntimeError("Not connected to the robot")
        if self._stopped:
            raise RuntimeError("Robot is in protective stop")
        self._target = np.asarray(pose, dtype=float).copy()
        self._move_started = time.monotonic()

    def move_to_joints(self, joints, speed: float, acceleration: float) -> None:
        if not self._connected:
            raise RuntimeError("Not connected to the robot")
        self._joints = np.asarray(joints, dtype=float).copy()

    def linked(self) -> bool:
        return self._connected

    def free_drive(self) -> None:
        if not self._connected:
            raise RuntimeError("Not connected to the robot")

    def hold(self) -> None:
        if not self._connected:
            raise RuntimeError("Not connected to the robot")

    def stop(self, deceleration: float = 2.0) -> None:
        self._advance()
        self._target = self._pose.copy()
        self._move_started = 0.0

    def protective_stop(self) -> None:
        self.stop()
        self._stopped = True

    def reset_stop(self) -> None:
        self._stopped = False

    def set_speed_scale(self, fraction: float) -> None:
        self._speed_scale = float(np.clip(float(fraction), 0.01, 1.0))


class SimUR5Transport(UR5Transport):
    """In-process simulated UR5e with URDF kinematics and smooth joint interpolation."""

    name = "sim"
    is_real = False

    def __init__(self, max_joint_speed_deg: float = 45.0, floor_seconds: float = 0.08, tool_offset: float = scan_geometry.ENDOSCOPE_TIP_OFFSET):
        self.max_joint_speed_deg = float(max_joint_speed_deg)
        self.floor_seconds = float(floor_seconds)
        self.tool_offset = float(tool_offset)
        self._joints = np.array(DEFAULT_HOME_JOINTS, dtype=float)
        self._start_joints = self._joints.copy()
        self._target_joints = self._joints.copy()
        self._move_started = 0.0
        self._move_duration = 0.0
        self._connected = False
        self._stopped = False
        self._speed_scale = 1.0
        self._pose_override: np.ndarray | None = None

    def connect(self, ip: str = "sim") -> None:
        self._connected = True
        self._stopped = False

    def disconnect(self) -> None:
        self._connected = False
        self._move_started = 0.0

    def _advance(self) -> None:
        if self._move_started <= 0.0:
            return
        elapsed = time.monotonic() - self._move_started
        t = 1.0 if self._move_duration <= 0.0 else min(1.0, elapsed / self._move_duration)
        smooth = t * t * (3.0 - 2.0 * t)
        self._joints = self._start_joints + (self._target_joints - self._start_joints) * smooth
        if t >= 1.0:
            self._joints = self._target_joints.copy()
            self._move_started = 0.0
            self._move_duration = 0.0

    def read_state(self) -> RobotState:
        self._advance()
        if self._pose_override is not None:
            pose = self._pose_override
        else:
            cam_pos, cam_rot = scan_geometry.camera_pose(self._joints, self.tool_offset)
            pose = np.r_[cam_pos, cam_rot]
        return RobotState(
            connected=self._connected,
            robot_mode=READY_ROBOT_MODE if self._connected else 0,
            safety_mode=3 if self._stopped else 1,
            tcp_pose=[float(v) for v in pose],
            joints=[float(v) for v in self._joints],
            protective_stopped=self._stopped,
            emergency_stopped=False,
            program_running=self._move_started > 0.0,
            speed_scale=self._speed_scale,
            updated_at=time.monotonic(),
        )

    def set_joints_immediate(self, joints) -> None:
        self._joints = np.asarray(joints, dtype=float).reshape(-1)[:6].copy()
        self._start_joints = self._joints.copy()
        self._target_joints = self._joints.copy()
        self._move_started = 0.0
        self._move_duration = 0.0
        self._pose_override = None

    def move_to_pose(self, pose, speed: float, acceleration: float) -> None:
        if not self._connected:
            raise RuntimeError("Not connected to the robot")
        if self._stopped:
            raise RuntimeError("Robot is in protective stop")
        pose_arr = np.asarray(pose, dtype=float).reshape(-1)[:6]
        flange_target = np.r_[
            scan_geometry.flange_for_grasp(pose_arr[:3], pose_arr[3:6], self.tool_offset),
            pose_arr[3:6],
        ]
        try:
            joints, _, _ = scan_geometry.solve_ik(self._joints, flange_target, "sim")
            self.move_to_joints(joints, speed, acceleration)
            return
        except Exception:
            pass
        self._pose_override = pose_arr.copy()
        self._move_started = 0.0

    def move_to_joints(self, joints, speed: float, acceleration: float) -> None:
        if not self._connected:
            raise RuntimeError("Not connected to the robot")
        if self._stopped:
            raise RuntimeError("Robot is in protective stop")
        self._advance()
        target = np.asarray(joints, dtype=float).reshape(-1)[:6].copy()
        self._pose_override = None
        self._start_joints = self._joints.copy()
        self._target_joints = target
        eff_speed = max(1.0, self.max_joint_speed_deg * self._speed_scale)
        self._move_duration = scan_geometry.move_seconds(
            self._start_joints, target, eff_speed, self.floor_seconds
        )
        if self._move_duration <= 1e-4 or float(np.max(np.abs(target - self._start_joints))) <= 1e-5:
            self._joints = target
            self._move_started = 0.0
            self._move_duration = 0.0
        else:
            self._move_started = time.monotonic()

    def ramp_for(self, joints) -> float:
        eff_speed = max(1.0, self.max_joint_speed_deg * self._speed_scale)
        return scan_geometry.move_seconds(self._joints, joints, eff_speed, self.floor_seconds)

    def linked(self) -> bool:
        return self._connected

    def free_drive(self) -> None:
        if not self._connected:
            raise RuntimeError("Not connected to the robot")

    def hold(self) -> None:
        if not self._connected:
            raise RuntimeError("Not connected to the robot")

    def stop(self, deceleration: float = 2.0) -> None:
        self._advance()
        self._start_joints = self._joints.copy()
        self._target_joints = self._joints.copy()
        self._move_started = 0.0
        self._move_duration = 0.0

    def protective_stop(self) -> None:
        self.stop()
        self._stopped = True

    def reset_stop(self) -> None:
        self._stopped = False

    def set_speed_scale(self, fraction: float) -> None:
        self._speed_scale = float(np.clip(float(fraction), 0.01, 1.0))


TRANSPORT_LABELS = {
    "sim":     "Simulated UR5e",
    "rtde":    "UR5 over RTDE",
    "teleop":  "UR5 via TeleopApp",
    "dry-run": "Dry run (no hardware)",
}


def rtde_available() -> bool:
    try:
        import rtde_control  # noqa: F401
        import rtde_receive  # noqa: F401
    except Exception:
        return False
    return True


# ============================================================================
# Robot
# ============================================================================

class UR5Robot:
    """Connection, live status, and the safety gate every command passes."""

    POLL_INTERVAL = 0.08

    def __init__(self, transport: UR5Transport | None = None):
        self.transport = transport or DryRunTransport()
        self.ip = ""
        self.state = RobotState()
        self.status = "Not connected"
        self.last_error = ""
        # Why the last connect failed, kept as the exception so a caller can tell a busy
        # controller - which it can offer to stop - from one that did not answer.
        self.last_exception: Exception | None = None
        # Set by the stop button. Latched: it stays set until reset() clears it,
        # so nothing can quietly resume after a stop.
        self.stop_requested = False
        self.paused = False
        # Set while the arm is hand-guidable. Nothing may be commanded then: a move sent to
        # an arm someone is holding is a move into their hands.
        self.free_driving = False
        self.speed_limit = 1.0
        self.speed = DEFAULT_SPEED
        self.acceleration = DEFAULT_ACCELERATION
        self.home_joints = list(DEFAULT_HOME_JOINTS)
        self._lock = threading.Lock()
        self._poll_stop = threading.Event()
        self._poll_thread: threading.Thread | None = None

    # -- connection --------------------------------------------------------

    @property
    def connected(self) -> bool:
        return bool(self.state.connected)

    @property
    def is_real(self) -> bool:
        return bool(getattr(self.transport, "is_real", False))

    def connect(self, ip: str) -> bool:
        self.disconnect()
        self.ip = str(ip).strip()
        if not self.ip:
            self.status = "Enter the robot IP address first"
            return False
        self.last_exception = None
        try:
            self.transport.connect(self.ip)
        except Exception as exc:
            self.last_exception = exc
            self.last_error = str(exc)
            self.status = f"Could not connect to {self.ip}: {exc}"
            self.state = RobotState(connected=False, error=str(exc))
            return False
        self.stop_requested = False
        self.paused = False
        self.state = self.transport.read_state()
        self._poll_stop.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True, name="ur5-poll")
        self._poll_thread.start()
        kind = "robot" if self.is_real else "simulated robot"
        self.status = f"Connected to {kind} at {self.ip}"
        self.apply_speed_limit(self.speed_limit)
        return True

    def disconnect(self) -> None:
        if self.free_driving:
            try:
                self.hold()
            except Exception as exc:
                self.last_error = f"left in free drive: {exc}"
        self._poll_stop.set()
        thread = self._poll_thread
        self._poll_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        try:
            self.transport.disconnect()
        except Exception as exc:
            logging.warning("ur5_robot.disconnect: %s", exc)
        self.state = RobotState(connected=False)
        self.status = "Not connected"

    def _poll_loop(self) -> None:
        while not self._poll_stop.is_set():
            try:
                state = self.transport.read_state()
            except Exception as exc:
                state = RobotState(connected=False, error=str(exc), updated_at=time.monotonic())
            with self._lock:
                self.state = state
            # A protective or emergency stop raised at the pendant must stop the
            # scan too, not just be displayed. Latching it here means any motion
            # already queued is refused by the gate on its next check.
            if state.connected and (state.protective_stopped or state.emergency_stopped):
                self.stop_requested = True
            self._poll_stop.wait(self.POLL_INTERVAL)

    def refresh(self) -> RobotState:
        """Reads state directly; used when no poll thread is running."""
        if self._poll_thread is None or not self._poll_thread.is_alive():
            try:
                with self._lock:
                    self.state = self.transport.read_state()
            except Exception as exc:
                self.state = RobotState(connected=False, error=str(exc))
        return self.state

    # -- readiness ---------------------------------------------------------

    def readiness(self) -> tuple[bool, list[str]]:
        """Whether the arm may be commanded, and everything blocking it.

        Reports all problems rather than the first, so a user with several
        things to fix at the pendant sees the whole list at once.
        """
        issues: list[str] = []
        state = self.state
        if not state.connected:
            issues.append("not connected to the robot" + (f" - {state.error}" if state.error else ""))
            return False, issues
        if state.emergency_stopped:
            issues.append("robot is in emergency stop; release it at the pendant")
        if state.protective_stopped:
            issues.append("robot is in protective stop; clear it before moving")
        if int(state.safety_mode) in BLOCKING_SAFETY_MODES:
            issues.append(f"safety mode is {state.safety_mode_label}")
        if int(state.robot_mode) != READY_ROBOT_MODE:
            issues.append(
                f"robot mode is {state.robot_mode_label}; it must be running "
                "(powered on, brakes released, remote control enabled)"
            )
        if self.stop_requested:
            issues.append("a stop is latched in the app; press Reset to clear it")
        if self.free_driving:
            issues.append("the arm is in free drive; press Hold to take it back")
        issues.extend(self.transport.blockers())
        return len(issues) == 0, issues

    def ready(self) -> bool:
        ok, _ = self.readiness()
        return ok

    def _guard(self, action: str) -> None:
        """The single door to the arm. Raises unless motion is permitted."""
        if self.stop_requested:
            raise RuntimeError(f"{action} refused: a stop is latched; press Reset first")
        if self.paused:
            raise RuntimeError(f"{action} refused: the run is paused")
        ok, issues = self.readiness()
        if not ok:
            raise RuntimeError(f"{action} refused: " + "; ".join(issues))

    # -- free drive ----------------------------------------------------------

    def free_drive(self) -> None:
        """Let the arm be posed by hand. Refused while a run or a stop could be in play."""
        if not self.connected:
            raise RuntimeError("Free drive refused: not connected to the robot")
        if self.stop_requested:
            raise RuntimeError("Free drive refused: a stop is latched; press Reset first")
        self.transport.free_drive()
        self.free_driving = True
        self.status = "Free drive - pose the camera by hand, then Record view"

    def hold(self) -> None:
        """Take the arm back from the hand. The flag clears only once the transport says it held."""
        self.transport.hold()
        self.free_driving = False
        self.status = "Holding - the arm holds where it was left"

    def joints_now(self) -> list[float]:
        if not self.connected:
            raise RuntimeError("Not connected to the robot")
        return [float(value) for value in self.transport.joints_now()]

    # -- safety controls ---------------------------------------------------

    def emergency_stop(self) -> None:
        """Software stop: decelerate now and latch the controller stopped.

        This is the controller's protective stop, not the hardware emergency
        stop -- the physical button is still the real safety device.
        """
        self.stop_requested = True
        self.paused = False
        try:
            self.transport.protective_stop()
            self.status = "STOPPED - protective stop triggered"
        except Exception as exc:
            self.last_error = str(exc)
            self.status = f"Stop command failed: {exc} - use the hardware emergency stop"

    def pause(self) -> None:
        self.paused = True
        try:
            self.transport.stop(2.0)
        except Exception as exc:
            self.last_error = str(exc)
            self.stop_requested = True
            self.status = f'Pause failed: {exc}'
            return
        self.status = "Paused"

    def resume(self) -> bool:
        if self.stop_requested:
            self.status = "Cannot continue while a stop is latched; press Reset first"
            return False
        self.paused = False
        self.status = "Continuing"
        return True

    def reset(self) -> bool:
        """Clears the latched stop, after asking the controller to recover."""
        try:
            self.transport.reset_stop()
        except Exception as exc:
            self.last_error = str(exc)
            self.status = f'Reset failed: {exc}'
            return False
        self.stop_requested = False
        self.paused = False
        self.refresh()
        ok, issues = self.readiness()
        self.status = "Reset; robot ready" if ok else "Reset; still blocked: " + "; ".join(issues)
        return ok

    def apply_speed_limit(self, fraction: float) -> float:
        """Caps commanded speed, and the controller's own speed slider with it.

        Both halves matter: the slider bounds anything the controller runs, and
        the local cap bounds what this app asks for in the first place.
        """
        self.speed_limit = float(np.clip(float(fraction), 0.01, 1.0))
        try:
            self.transport.set_speed_scale(self.speed_limit)
        except Exception as exc:
            self.last_error = str(exc)
        return self.speed_limit

    def limited_speed(self, speed: float | None = None) -> float:
        base = float(self.speed if speed is None else speed)
        return float(np.clip(base * self.speed_limit, 0.001, MAX_SPEED))

    def limited_acceleration(self, acceleration: float | None = None) -> float:
        base = float(self.acceleration if acceleration is None else acceleration)
        return float(np.clip(base * self.speed_limit, 0.01, MAX_ACCELERATION))

    def move_home(self) -> bool:
        """Sends the arm to the retracted safe pose."""
        try:
            self._guard("Move to home")
            self.transport.move_home(
                self.home_joints,
                self.limited_speed(0.35),
                self.limited_acceleration(0.8),
            )
        except Exception as exc:
            self.last_error = str(exc)
            self.status = f"Home move failed: {exc}"
            return False
        self.status = "Moving to safe home position"
        return True

    # -- motion ------------------------------------------------------------

    def move_to_pose(self, pose, speed: float | None = None, acceleration: float | None = None) -> bool:
        self._guard("Move")
        pose = [float(v) for v in np.asarray(pose, dtype=float).reshape(-1)[:6]]
        if not self.transport.pose_within_limits(pose):
            raise RuntimeError("Target pose is outside the robot's safety limits")
        self.transport.move_to_pose(pose, self.limited_speed(speed), self.limited_acceleration(acceleration))
        return True

    def pose_error(self, pose) -> tuple[float, float]:
        """Position (m) and rotation (rad) distance from the current TCP pose."""
        target = np.asarray(pose, dtype=float).reshape(-1)[:6]
        current = np.asarray(self.state.tcp_pose, dtype=float).reshape(-1)[:6]
        if current.size < 6:
            return float("inf"), float("inf")
        position = float(np.linalg.norm(target[:3] - current[:3]))
        rotation = float(np.linalg.norm(target[3:6] - current[3:6]))
        return position, rotation

    def at_pose(self, pose, tol: float = POSE_REACHED_TOL, rot_tol: float = POSE_REACHED_ROT_TOL) -> bool:
        position, rotation = self.pose_error(pose)
        return position <= float(tol) and rotation <= float(rot_tol)

    def move_to_joints(self, joints) -> bool:
        self._guard("Move")
        self.transport.move_to_joints([float(v) for v in np.asarray(joints, dtype=float).reshape(-1)[:6]],
                                      self.limited_speed(), self.limited_acceleration())
        return True

    def joint_error(self, joints) -> float:
        """Largest single-joint distance, in radians, from the measured configuration."""
        target = np.asarray(joints, dtype=float).reshape(-1)[:6]
        current = np.asarray(self.state.joints, dtype=float).reshape(-1)[:6]
        if current.size < 6:
            return float("inf")
        return float(np.max(np.abs(target - current)))

    def at_joints(self, joints, tol: float = JOINTS_REACHED_TOL) -> bool:
        return self.joint_error(joints) <= float(tol)

    def describe(self) -> str:
        # Keyed on the transport rather than on is_real, which cannot tell two real ones apart.
        transport = TRANSPORT_LABELS.get(str(self.transport.name), str(self.transport.name))
        return f"{transport} @ {self.ip}" if self.ip else transport

