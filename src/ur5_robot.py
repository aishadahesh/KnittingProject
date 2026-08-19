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

import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np


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

DEFAULT_RTDE_PORT = 30004
# Conservative defaults for scanning near fabric. Deliberately far below the
# arm's capability: this moves a camera between close-set targets.
DEFAULT_SPEED = 0.06
DEFAULT_ACCELERATION = 0.30
MAX_SPEED = 0.25
MAX_ACCELERATION = 0.80
# Joint angles for a safe, retracted pose over the table.
DEFAULT_HOME_JOINTS = [0.0, -math.pi / 2.0, 0.0, -math.pi / 2.0, 0.0, 0.0]
POSE_REACHED_TOL = 0.003
POSE_REACHED_ROT_TOL = 0.05


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

    def connect(self, ip: str) -> None:
        import rtde_control
        import rtde_receive

        self.ip = str(ip)
        # Receive first: it is read-only, so a wrong address or an unreachable
        # controller fails here without ever having opened a control channel
        # that could command motion.
        self.receive = rtde_receive.RTDEReceiveInterface(self.ip)
        self.control = rtde_control.RTDEControlInterface(self.ip)
        try:
            import rtde_io

            self.io = rtde_io.RTDEIOInterface(self.ip)
        except Exception:
            # Speed-slider control is a convenience; losing it must not block
            # a connection that is otherwise good.
            self.io = None

    def disconnect(self) -> None:
        for handle in (self.control, self.receive, self.io):
            if handle is None:
                continue
            try:
                handle.disconnect()
            except Exception:
                pass
        self.control = None
        self.receive = None
        self.io = None

    def read_state(self) -> RobotState:
        if self.receive is None:
            return RobotState(connected=False)
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
        except Exception:
            pass
        try:
            state.program_running = bool(self.control.isProgramRunning()) if self.control else False
        except Exception:
            pass
        return state

    def move_to_pose(self, pose, speed: float, acceleration: float) -> None:
        if self.control is None:
            raise RuntimeError("Not connected to the robot")
        # asynchronous=True so the caller keeps polling state (and the UI keeps
        # redrawing) while the arm moves, instead of blocking inside the driver.
        self.control.moveL([float(v) for v in pose], float(speed), float(acceleration), True)

    def move_to_joints(self, joints, speed: float, acceleration: float) -> None:
        if self.control is None:
            raise RuntimeError("Not connected to the robot")
        self.control.moveJ([float(v) for v in joints], float(speed), float(acceleration), True)

    def stop(self, deceleration: float = 2.0) -> None:
        if self.control is None:
            return
        try:
            self.control.stopL(float(deceleration))
        except Exception:
            pass

    def protective_stop(self) -> None:
        if self.control is None:
            return
        # Decelerate first, then latch the controller into protective stop.
        # Order matters: the stop command is what actually arrests motion, and
        # it must be issued before anything that can throw.
        try:
            self.control.stopL(4.0)
        except Exception:
            pass
        try:
            self.control.triggerProtectiveStop()
        except Exception:
            pass

    def reset_stop(self) -> None:
        if self.control is None:
            return
        try:
            self.control.reuploadScript()
        except Exception:
            pass

    def set_speed_scale(self, fraction: float) -> None:
        fraction = float(np.clip(float(fraction), 0.01, 1.0))
        if self.io is not None:
            try:
                self.io.setSpeedSlider(fraction)
            except Exception:
                pass

    def pose_within_limits(self, pose) -> bool:
        if self.control is None:
            return True
        try:
            return bool(self.control.isPoseWithinSafetyLimits([float(v) for v in pose]))
        except Exception:
            return True


class DryRunTransport(UR5Transport):
    """A stand-in arm that reports arriving where it was told to go.

    Lets the whole UR5 workflow -- connect, plan, move, capture, store, analyse
    -- be exercised with no hardware, and gives the tests something to drive.
    It reports ``is_real=False``, which travels into the database alongside the
    captures so a dry run is never mistaken for a real scan.
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


def make_transport(kind: str) -> UR5Transport:
    if str(kind) == "rtde":
        return RTDETransport()
    return DryRunTransport()


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
        # Set by the stop button. Latched: it stays set until reset() clears it,
        # so nothing can quietly resume after a stop.
        self.stop_requested = False
        self.paused = False
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
        try:
            self.transport.connect(self.ip)
        except Exception as exc:
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
        kind = "robot" if self.is_real else "dry-run robot"
        self.status = f"Connected to {kind} at {self.ip}"
        self.apply_speed_limit(self.speed_limit)
        return True

    def disconnect(self) -> None:
        self._poll_stop.set()
        thread = self._poll_thread
        self._poll_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        try:
            self.transport.disconnect()
        except Exception:
            pass
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
            issues.append("not connected to the robot")
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
            self.transport.move_to_joints(
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

    def describe(self) -> str:
        transport = "UR5 over RTDE" if self.is_real else "Dry run (no hardware)"
        return f"{transport} @ {self.ip}" if self.ip else transport

