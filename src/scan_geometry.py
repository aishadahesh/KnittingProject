"""Where the camera goes to photograph a subject, and the joint configurations that put it there.

Five patterns, each answering a different question about one subject:

- ``spin`` rolls the camera about its own axis. The position is held exactly - both the
  flange and the tool point lie on the axis being rotated about - so N=3 gives the pose the
  arm is in now plus 120 and 240 degrees. Use it for in-plane rotation of the frame.
- ``orbit`` moves the camera. The subject is taken to sit ``subject_distance`` in front of
  the lens along the axis it points down; the camera is then placed on a sphere of that
  radius about the subject and aimed back at it, so the standoff never changes and only the
  direction of view does. ``arc`` sweeps around, ``elevation`` says how high, and ``rings``
  repeats the arc at several elevations so one pass sees both the top and the sides.
- ``tilt`` is this project's original scan geometry, ``fabric_scanner.expand_stations_to_angle_viewpoints``,
  kept because it is what the simulated Scan Mode uses. It holds one position and nudges the
  attitude within a narrow cone, so its extremes differ by only ~33 degrees.
- ``grid`` covers the fabric close up: a rows x cols grid of straight-down views at a fixed,
  low ``camera_height`` over the fabric plane, centred where the camera looks now. Every frame
  is square-on at the same scale, which is what the orbit's 20 cm standoff cannot give.
- ``centred`` keeps the camera aimed at the fabric's middle and tilts around it: straight down
  from ``camera_height`` first, then rings tilted a step further each at that same distance, and in
  every direction only as far as the whole frame stays on the taught outline - so a view never
  swings high or low enough to lose the cloth. Grid and centred share ``camera_height``, so the one
  capture height the panel shows is how close either pattern gets.
- ``taught`` is whatever the user posed the arm at by hand in free drive, flown in exactly
  the joints recorded. Nothing is solved; every move between the views is still checked.

Two limits are the arm's, not the geometry's, and both are enforced here rather than
discovered on the hardware:

- A pose the inverse kinematics cannot reach is refused, rather than commanded as the
  solver's best effort - which is a configuration nobody asked for.
- The camera rides on the tool, so ``max_height`` caps how far it may rise above the
  subject. Lift it and the picture is of the table around the cloth.

The inverse kinematics is this project's own damped-least-squares step from
``fabric_scanner``, iterated to a tolerance, against the MuJoCo UR5e.
"""

from __future__ import annotations

import math
import threading
import dataclasses
from dataclasses import dataclass, field

import numpy as np

import fabric_scanner as scanner
import ur5_robot

# The UR5e's driven joints, in the order MuJoCo reports them in qpos. TeleopApp names them
# identically, which load_model asserts rather than assumes.
TELEOP_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                 "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

# How close the solve has to get before a pose is accepted, and how far off counts as not
# having reached it at all.
POSE_TOL = 0.002
ROT_TOL = 0.02
UNREACHED_POS = 0.004
UNREACHED_ROT = 0.04
MAX_IK_ITERS = 4000
# A solve that has not cut its error by IK_PROGRESS in IK_STALL_STEPS steps is not going to arrive.
IK_STALL_STEPS = 150
IK_PROGRESS = 0.002
# Two solve starts closer than this on every joint would slide to the same answer; only one is tried.
SEED_SEPARATION_DEG = 2.0

# TeleopApp eases every move with a quintic profile, whose peak joint speed is 1.875x its
# average. A duration asked for in terms of peak speed has to allow for that.
PEAK_OVER_MEAN = 1.875

# The floor elevation of a multi-ring sweep: shallow enough to see the subject's side. Not
# 20: from Home the arm reaches a 20 degree ring only by folding its wrist down to 30 mm
# over the fabric and through 110 degree flips between neighbouring views.
LOWEST_RING_DEG = 30.0

# Every move is checked at this joint resolution, and no coarser: a contact or a dip that
# happens between two samples is a contact or a dip the check never saw.
SWEEP_STEP_DEG = 2.0
# How close the camera and the wrist may come to the fabric at any sample of a move. The
# model has no table, fabric or camera housing, so this plane stands in for all three.
FABRIC_CLEARANCE = 0.03
# The elbow, the wrist joints and the middles of the upper arm and forearm are tubes about 9 cm
# across, and their origins are on the axes: each must stay this far over the fabric plane, or the
# arm lies on the table - which the model, having no table, would never say.
ARM_CLEARANCE = 0.10
ARM_BODIES = ("upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link")
# The first move out of where the arm stands may start lower than FABRIC_CLEARANCE - the
# camera resting on the fabric after it was measured, say - and it may then lift away, but
# not go lower than it started by more than this. Only that move: every move after it starts
# at a planned viewpoint, and a planned viewpoint is never stood at below FABRIC_CLEARANCE.
DESCENT_TOLERANCE = 0.002
# How far past the taught fabric outline a frame may reach before it counts as showing
# background. A few millimetres of the fabric's own border is not background.
FRAME_TOLERANCE = 0.005
# The bodies that pass over the fabric: the wrist, whose origins are its joint centres.
SWEPT_BODIES = ("wrist_1_link", "wrist_2_link", "wrist_3_link")
# The real arm will not fly where it is near a singularity, which the model happily solves: it stops
# protectively. Three margins keep every sample of a move away from the three of them. The wrist
# centre must stay this far from the base's vertical axis (the shoulder singularity: a tilted view
# of cloth near the base pulls the wrist in over it), and the elbow and the second wrist joint
# must stay this far from straight and from lying along the forearm.
MIN_WRIST_RADIUS = 0.20
MIN_ELBOW_BEND_DEG = 25.0
MIN_WRIST_BEND_DEG = 20.0
# A move may start inside those margins - the overview height has the elbow nearly straight - and
# then only has to stop getting closer to a singularity than it began, give or take this much
# (metres, degrees, degrees). Where it ends is held to the full margins, except for a fixed view:
# a pose the arm was taught or jogged to, which it is known to stand in, is taken as given, and
# the move to it is only held to no worse than the worse of its two ends.
SINGULARITY_SLACK = (0.005, 1.0, 1.0)



class Unreachable(RuntimeError):
    """A pose the arm cannot take. Carries the residual so a panel can say how far off."""

    def __init__(self, label, err_pos, err_rot, target_pose):
        self.label = label
        self.err_pos = float(err_pos)
        self.err_rot = float(err_rot)
        self.target_pose = np.asarray(target_pose, dtype=float)
        super().__init__(
            f"`{label}` is not reachable: the solve stalled {self.err_pos * 1000:.0f} mm and "
            f"{self.err_rot:.3f} rad short, against {UNREACHED_POS * 1000:.0f} mm and {UNREACHED_ROT:.2f} rad "
            f"allowed. Target [{self.target_pose[0]:+.3f}, {self.target_pose[1]:+.3f}, {self.target_pose[2]:+.3f}] "
            f"is outside the arm's reach or blocked by a joint limit")


@dataclass
class Viewpoint:
    """One place to photograph from, and what the solve found on the way to it."""

    name: str
    position: np.ndarray
    rotvec: np.ndarray
    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0
    cell: tuple[int, int] = (0, 0)
    config: np.ndarray = field(default_factory=lambda: np.zeros(6))
    err_pos: float = 0.0
    err_rot: float = 0.0
    contacts: int = 0
    transition_deg: float = 0.0
    path_contacts: int = 0
    clearance: float = math.inf
    floor: float = FABRIC_CLEARANCE
    standing: float = math.inf
    adjustment: str = ""
    overshoot: float = 0.0
    frame_enforced: bool = False
    aim: np.ndarray | None = None
    wrist_radius: float = math.inf
    elbow_bend_deg: float = math.inf
    wrist_bend_deg: float = math.inf
    overview: bool = False
    fixed: bool = False
    arm_clearance: float = math.inf

    @property
    def problems(self) -> list[str]:
        """Why the move to this viewpoint must not be flown; empty when it may."""
        found = []
        if self.contacts:
            found.append(f"{self.contacts} self-contacts at the pose")
        if self.path_contacts:
            found.append(f"self-contact at {self.path_contacts} points on the way in")
        if self.clearance < self.floor:
            where = (f"passes {self.clearance * 1000:.0f} mm above the fabric plane" if self.clearance >= 0.0
                     else f"reaches {-self.clearance * 1000:.0f} mm BELOW the fabric plane")
            found.append(f"{where} on the way in, against {self.floor * 1000:.0f} mm above it required")
        if self.arm_clearance < ARM_CLEARANCE:
            found.append(f"its elbow or arm passes {self.arm_clearance * 1000:.0f} mm above the fabric plane, against "
                         f"{ARM_CLEARANCE * 1000:.0f} mm required - the arm would lie on the table")
        if self.frame_enforced and self.overshoot > FRAME_TOLERANCE:
            found.append("its frame looks above the fabric's horizon" if math.isinf(self.overshoot) else
                         f"its frame reaches {self.overshoot * 1000:.0f} mm past the fabric's edge")
        if self.wrist_radius < MIN_WRIST_RADIUS:
            found.append(f"its wrist passes {self.wrist_radius * 1000:.0f} mm from the base's axis, against "
                         f"{MIN_WRIST_RADIUS * 1000:.0f} mm required - too near the base for the arm to fly")
        if self.elbow_bend_deg < MIN_ELBOW_BEND_DEG:
            found.append(f"its elbow is {self.elbow_bend_deg:.0f} deg from straight, against {MIN_ELBOW_BEND_DEG:.0f} required")
        if self.wrist_bend_deg < MIN_WRIST_BEND_DEG:
            found.append(f"its second wrist joint is {self.wrist_bend_deg:.0f} deg from singular, against {MIN_WRIST_BEND_DEG:.0f} required")
        if self.standing < FABRIC_CLEARANCE:
            found.append(f"stands {self.standing * 1000:.0f} mm over the fabric plane, against "
                         f"{FABRIC_CLEARANCE * 1000:.0f} mm required")
        return found

    @property
    def clear(self) -> bool:
        return not self.problems


@dataclass
class CapturePlan:
    """The four fields RealScanRunner reads, plus what the panel needs to draw and report.

    The runner touches only `configs`, `view_names`, `station_ids` and `station_cells`;
    `poses` is the tool pose each configuration puts the camera at, kept for the capture
    records, and the rest is for the preview and the status rows.
    """

    poses: np.ndarray
    view_names: list[str]
    station_ids: list[int]
    station_cells: list[tuple[int, int]]
    viewpoints: list[Viewpoint]
    subject: np.ndarray
    pattern: str
    fabric_z: float = math.nan
    skipped: list = field(default_factory=list)
    lens_hfov_deg: float = 55.0
    image_turn: list = field(default_factory=lambda: [[1.0, 0.0], [0.0, 1.0]])

    @property
    def configs(self) -> np.ndarray:
        """The joint configurations Check solved and collision-checked, in visiting order.

        The runner commands these rather than `poses`, so the motion flown is the motion
        checked. Handing the controller a pose instead lets it pick its own joints and its
        own path to them, neither of which anything here has looked at.
        """
        return np.array([point.config for point in self.viewpoints], dtype=float)

    @property
    def clear(self) -> bool:
        return all(point.clear for point in self.viewpoints)

    @property
    def refused(self) -> list[Viewpoint]:
        return [point for point in self.viewpoints if not point.clear]

    def assess(self) -> tuple[bool, list[str]]:
        """RealScanRunner's pre-motion safety check, answered from the solve.

        FabricPlan answers this by densifying the Cartesian path and bounding its step,
        because it has no joint solution to inspect. Here every pose is already solved and
        the joint path between consecutive poses already collision-checked, which is the
        stronger statement -- and the Cartesian step bound would reject this motion for no
        reason, since a capture path deliberately swings far between viewpoints.
        """
        return not self.refused, [f"`{point.name}`: " + "; ".join(point.problems) for point in self.refused]

    def capture_metadata(self, index: int, image_width: int | None = None) -> dict:
        """The geometry behind one capture - where the camera was over the cloth, how tilted, what
        patch of fabric the frame covers and at what detail - so every stored image says what it shows."""
        from scipy.spatial.transform import Rotation

        point = self.viewpoints[index]
        lens = Rotation.from_rotvec(point.rotvec).apply([0.0, 0.0, 1.0])
        described = {"capture_pattern": self.pattern,
                     "capture_view": point.name,
                     "capture_azimuth_deg": float(point.azimuth_deg),
                     "capture_elevation_deg": float(point.elevation_deg),
                     "capture_subject": [float(v) for v in self.subject],
                     "camera_position": [float(v) for v in point.position],
                     "camera_height_above_fabric_m": float(point.position[2] - self.fabric_z),
                     "tilt_from_vertical_deg": float(math.degrees(math.acos(float(np.clip(-lens[2], -1.0, 1.0))))),
                     "moved_to_nearest_safe": point.adjustment,
                     "camera_image_turn": self.image_turn,
                     "frame_past_fabric_edge_mm": None if math.isinf(point.overshoot) else float(point.overshoot * 1000.0)}
        corners = footprint(point.position, point.rotvec, self.fabric_z, self.lens_hfov_deg)
        if corners is not None:
            across = float(np.linalg.norm(corners[1] - corners[0]))
            described.update({"fabric_footprint": [[float(v) for v in corner[:2]] for corner in corners],
                              "fabric_across_frame_mm": across * 1000.0})
            if image_width:
                described["fabric_mm_per_pixel"] = across * 1000.0 / float(image_width)
        return described


# ============================================================================
# The model, and the two checks every pose passes
# ============================================================================

_loaded = threading.local()


def load_model():
    """The MuJoCo UR5e used for the solve, with its joint order checked against TeleopApp's.

    Parsed once per thread and kept: parsing the MJCF costs a noticeable fraction of a second, and a
    Check, a Home move and every probe used to pay it again. One per thread, because a solve writes
    the model's state - the panel's worker and the panel itself must never share one. Every entry
    point seeds the state it starts from, so a reused model carries nothing over.
    """
    if getattr(_loaded, "model", None) is None:
        mj, model, data, site_id = scanner.load_ur5e_model_data()
        named = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, index) for index in range(6)]
        if named != TELEOP_JOINTS:
            raise RuntimeError(f"MuJoCo joint order {named} is not TeleopApp's {TELEOP_JOINTS}")
        _loaded.model = (mj, model, data, site_id)
    return _loaded.model


def solve_ik(mj, model, data, site_id, target_pose, label):
    """Iterate this project's damped-least-squares step until the pose is met, or refuse.

    Gives up as soon as the error stops falling: an unreachable pose used to spend all MAX_IK_ITERS
    steps proving it, and the nearest-safe search tries many of them - which is what froze Check.
    """
    best, since_better = math.inf, 0
    for _ in range(MAX_IK_ITERS):
        err_pos, err_rot = scanner.pose_errors(scanner.get_tcp(mj, model, data, site_id), target_pose)
        if err_pos < POSE_TOL and err_rot < ROT_TOL:
            break
        score = err_pos + 0.1 * err_rot
        if score < best * (1.0 - IK_PROGRESS):
            best, since_better = score, 0
        else:
            since_better += 1
            if since_better >= IK_STALL_STEPS:
                break
        scanner.step_ik(mj, model, data, site_id, target_pose)

    err_pos, err_rot = scanner.pose_errors(scanner.get_tcp(mj, model, data, site_id), target_pose)
    if err_pos > UNREACHED_POS or err_rot > UNREACHED_ROT:
        raise Unreachable(label, err_pos, err_rot, target_pose)
    return np.array(data.qpos[:6], dtype=float), err_pos, err_rot


def contacts_at(mj, model, data, q):
    """How many self-collisions the model reports in configuration `q`.

    The UR5e MJCF carries 9 collision geoms, so this is a real check: sane poses report 0
    and a folded-back arm reports several. It sees the ARM ONLY - the fabric, the table and
    the camera mount are not in this model, so 0 means "the arm does not hit itself", never
    "the cell is clear".
    """
    held = np.array(data.qpos[:6], dtype=float)
    data.qpos[:6] = q
    mj.mj_forward(model, data)
    found = int(data.ncon)

    data.qpos[:6] = held
    mj.mj_forward(model, data)
    return found


def seed_from(mj, model, data, site_id, joints):
    """Put the model where the arm is and return its tool pose.

    Every plan is built from the arm's measured configuration, so the solve starts on the
    branch the arm is already on and the geometry is centred where the camera actually looks.
    """
    data.qpos[:6] = [float(value) for value in joints]
    mj.mj_forward(model, data)
    return scanner.get_tcp(mj, model, data, site_id)


def fabric_height(mj, model, data, site_id, subject_distance, measured=None):
    """The fabric plane every clearance is measured from.

    `measured` is the plane taken by touching the camera to the fabric, and is the one to
    trust. Without it the plane is estimated as `subject_distance` down the flange's axis at
    Home - which is only as good as that number, and a wrong one refuses every move.
    """
    if measured is not None:
        return float(measured)
    return float(subject_in_front_of(seed_from(mj, model, data, site_id, ur5_robot.DEFAULT_HOME_JOINTS),
                                     subject_distance)[2])


def camera_pose(joints, tool_offset):
    """Where the camera is in configuration `joints`, and its attitude as a rotation vector."""
    mj, model, data, site_id = load_model()
    flange = seed_from(mj, model, data, site_id, joints)
    return np.asarray(camera_point(flange, tool_offset), dtype=float), np.asarray(flange[3:6], dtype=float)


def camera_position(joints, tool_offset):
    """Where the camera is in configuration `joints` - what Mark corner records."""
    return camera_pose(joints, tool_offset)[0]


def camera_point(flange_pose, tool_offset):
    """Where the camera is, one tool offset along the flange's +Z."""
    from scipy.spatial.transform import Rotation

    return np.asarray(flange_pose[:3], dtype=float) + Rotation.from_rotvec(flange_pose[3:6]).apply([0.0, 0.0, float(tool_offset)])


def lowest_point(mj, model, data, site_id, q, tool_offset):
    """The lowest of the wrist's joint centres and the camera point in configuration `q`."""
    camera = camera_point(seed_from(mj, model, data, site_id, q), tool_offset)
    wrist = [float(data.xpos[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)][2]) for name in SWEPT_BODIES]
    return min(wrist + [float(camera[2])])


def lowest_arm_point(mj, model, data, site_id, q):
    """The lowest of the elbow, the wrist joints and the middles of the upper arm and forearm in configuration `q`."""
    seed_from(mj, model, data, site_id, q)
    ids = {name: mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name) for name in ARM_BODIES}
    if min(ids.values()) < 0:
        raise RuntimeError(f"The UR5e model has no body named {[name for name, index in ids.items() if index < 0]}")
    upper, elbow, wrist_1, wrist_2 = (np.array(data.xpos[ids[name]], dtype=float) for name in ARM_BODIES)
    return float(min(elbow[2], wrist_1[2], wrist_2[2], 0.5 * (upper[2] + elbow[2]), 0.5 * (elbow[2] + wrist_1[2])))


def singularity_margins(mj, model, data, site_id, q):
    """How far configuration `q` is from the shoulder, elbow and wrist singularities: (metres, degrees, degrees)."""
    seed_from(mj, model, data, site_id, q)
    bodies = {name: mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name) for name in ("shoulder_link", "wrist_2_link")}
    if min(bodies.values()) < 0:
        raise RuntimeError(f"The UR5e model has no body named {[name for name, index in bodies.items() if index < 0]}")
    radius = float(np.linalg.norm((data.xpos[bodies["wrist_2_link"]] - data.xpos[bodies["shoulder_link"]])[:2]))
    bend = lambda angle: math.degrees(math.asin(min(1.0, abs(math.sin(float(angle))))))
    return radius, bend(q[2]), bend(q[4])


def check_move(mj, model, data, site_id, point, previous_q, fabric_z, tool_offset, leaving=False):
    """Fill in what flying from `previous_q` to `point.config` would do, sampled every SWEEP_STEP_DEG.

    The endpoint is sampled with the path, and must also stand clear on its own. `leaving`
    marks the move out of where the arm actually is, the one move allowed to start low.
    Leaves the model where it found it, since the caller solves from there next.
    """
    held = np.array(data.qpos[:6], dtype=float)
    pieces = interpolate_steps(previous_q, point.config, SWEEP_STEP_DEG)
    point.contacts = contacts_at(mj, model, data, point.config)
    point.transition_deg = math.degrees(float(np.max(np.abs(np.asarray(point.config) - np.asarray(previous_q)))))
    point.path_contacts = sum(1 for piece in pieces if contacts_at(mj, model, data, piece))
    point.clearance = min(lowest_point(mj, model, data, site_id, piece, tool_offset) for piece in pieces) - fabric_z
    arm = [lowest_arm_point(mj, model, data, site_id, piece) - fabric_z for piece in pieces]
    arm_start = lowest_arm_point(mj, model, data, site_id, np.asarray(previous_q, dtype=float)) - fabric_z
    arm_floor = min(ARM_CLEARANCE, arm_start - DESCENT_TOLERANCE, arm[-1] - DESCENT_TOLERANCE if point.fixed else math.inf)
    arm_end_floor = -math.inf if point.fixed else ARM_CLEARANCE
    point.arm_clearance = min([value for value in arm[:-1] if value < arm_floor] + [arm[-1]] * (arm[-1] < arm_end_floor), default=math.inf)
    required = (MIN_WRIST_RADIUS, MIN_ELBOW_BEND_DEG, MIN_WRIST_BEND_DEG)
    samples =[singularity_margins(mj, model, data, site_id, piece) for piece in pieces]
    start_margins = singularity_margins(mj, model, data, site_id, np.asarray(previous_q, dtype=float))
    path_floors = [min(need, begin - slack, finish - slack if point.fixed else math.inf)
                   for need, begin, slack, finish in zip(required, start_margins, SINGULARITY_SLACK, samples[-1])]
    end_floors = [-math.inf] * 3 if point.fixed else required
    violations = lambda index: [sample[index] for sample in samples[:-1] if sample[index] < path_floors[index]] + [
        samples[-1][index]] * (samples[-1][index] < end_floors[index])
    point.wrist_radius, point.elbow_bend_deg, point.wrist_bend_deg = (min(violations(index), default=math.inf) for index in range(3))
    point.standing = lowest_point(mj, model, data, site_id, point.config, tool_offset) - fabric_z
    start = lowest_point(mj, model, data, site_id, np.asarray(previous_q, dtype=float), tool_offset) - fabric_z
    point.floor = min(FABRIC_CLEARANCE, start - DESCENT_TOLERANCE) if leaving else FABRIC_CLEARANCE
    seed_from(mj, model, data, site_id, held)


def check_home_move(joints, *, subject_distance, tool_offset, fabric_z=None, home_joints=None, fixed=False) -> Viewpoint:
    """The move from `joints` to Home - or, with `fixed`, to a pose the arm was taught - checked exactly as a capture path's moves are."""
    mj, model, data, site_id = load_model()
    fabric_z = fabric_height(mj, model, data, site_id, subject_distance, fabric_z)
    home = Viewpoint("Home", np.zeros(3), np.zeros(3), fixed=fixed,
                     config=np.asarray(ur5_robot.DEFAULT_HOME_JOINTS if home_joints is None else home_joints, dtype=float))
    check_move(mj, model, data, site_id, home, np.asarray(joints, dtype=float), fabric_z, tool_offset, leaving=True)
    return home


# ============================================================================
# Pacing
# ============================================================================

def interpolate_steps(previous_q, q, max_step_deg):
    """Split a transition into commands no larger than `max_step_deg` on any joint.

    Used to check the PATH between two configurations for self-collision, and as the
    fallback for a move that will not track in one piece.
    """
    if previous_q is None:
        return [np.asarray(q, dtype=float)]

    delta = np.asarray(q, dtype=float) - np.asarray(previous_q, dtype=float)
    largest = math.degrees(float(np.max(np.abs(delta))))
    pieces = max(1, int(math.ceil(largest / max(float(max_step_deg), 1e-6))))
    return [np.asarray(previous_q, dtype=float) + delta * (index / pieces) for index in range(1, pieces + 1)]


def move_seconds(previous_q, q, max_joint_speed_deg, floor_seconds):
    """How long to give a move so its fastest joint stays under `max_joint_speed_deg`.

    A fixed duration makes a small nudge and a large reconfiguration equally quick, and the
    big one then runs the commanded pose far ahead of where the arm has got to. TeleopApp
    watches for exactly that and stops commanding.

    `max_joint_speed_deg` is the PEAK the arm reaches, not its average. TeleopApp eases each
    move in and out - it starts and ends at a standstill, which is what makes the travel
    smooth - so the middle is faster than the average by PEAK_OVER_MEAN. Dividing by the
    average instead would quietly overshoot the speed a low setting was chosen to guarantee.
    """
    if previous_q is None:
        return float(floor_seconds)

    largest = float(np.max(np.abs(np.asarray(q, dtype=float) - np.asarray(previous_q, dtype=float))))
    average_speed = max(float(max_joint_speed_deg), 1e-6) / PEAK_OVER_MEAN
    return max(float(floor_seconds), math.degrees(largest) / average_speed)


def timeout_for(ramp_seconds):
    """How long to wait for a move of that duration before calling it stalled."""
    return 2.0 * float(ramp_seconds) + 5.0


# ============================================================================
# Where the camera goes
# ============================================================================

def flange_for_grasp(grasp_point, rotvec, tool_offset):
    """Where the flange goes so the tool point lands on `grasp_point`.

    The tool extends along the flange's +Z, so a rotation that keeps the tool point still
    has to swing the flange around it. Without this the rotations pivot about the flange and
    drag the camera through a tool-length arc.
    """
    from scipy.spatial.transform import Rotation

    return np.asarray(grasp_point, dtype=float) - Rotation.from_rotvec(rotvec).apply([0.0, 0.0, float(tool_offset)])


def subject_in_front_of(tool_pose, subject_distance):
    """The point the camera is looking at, taken to be `subject_distance` down its own axis."""
    from scipy.spatial.transform import Rotation

    axis = Rotation.from_rotvec(tool_pose[3:6]).apply([0.0, 0.0, 1.0])
    return np.asarray(tool_pose[:3], dtype=float) + axis * float(subject_distance)


def spin_views(tool_pose, count, tool_offset):
    """`count` orientations over a full turn about the camera's own axis, position held.

    Every pattern here reports where the TOOL POINT goes, never the flange, so the caller
    can convert once. `tool_pose` comes from the model's attachment site, which is the
    flange, hence the offset forward onto the camera.
    """
    from scipy.spatial.transform import Rotation

    start = Rotation.from_rotvec(tool_pose[3:6])
    held = np.asarray(tool_pose[:3], dtype=float) + start.apply([0.0, 0.0, float(tool_offset)])
    slices = max(1, int(count))

    views = []
    for index in range(slices):
        degrees = 360.0 * index / slices
        turned = start * Rotation.from_rotvec([0.0, 0.0, math.radians(degrees)])
        views.append(Viewpoint(f"spin {degrees:.0f}", held.copy(), turned.as_rotvec(), azimuth_deg=degrees))
    return views


@dataclass
class Sphere:
    """The orbit's sphere about the subject: where a view on it sits, and how it is aimed.

    Kept past orbit_views so the plan can place views on it as it solves them - each rolled
    to follow the one before, and raised clear of the fabric when it has to be.
    """

    subject: np.ndarray
    standoff: float
    facing: float
    highest_deg: float

    def direction(self, azimuth_deg, elevation_deg) -> np.ndarray:
        around = self.facing + math.radians(azimuth_deg)
        up = math.radians(elevation_deg)
        return np.array([math.cos(up) * math.cos(around), math.cos(up) * math.sin(around), math.sin(up)])

    def view(self, name, azimuth_deg, elevation_deg, follow_rotvec) -> Viewpoint:
        """A view aimed at the subject, rolled about its lens axis as near as it gets to `follow_rotvec`.

        Aiming fixes the lens axis but not the roll about it, and an orbit has no roll it
        needs: every capture records its own pose. So each view takes the roll nearest the
        camera it follows, and wrist 3 turns only as much as the path does. Holding one roll
        against the horizon instead winds wrist 3 through the whole arc - 300 degrees of it
        over a 240 degree sweep.
        """
        from scipy.spatial.transform import Rotation

        position = self.subject + self.standoff * self.direction(azimuth_deg, elevation_deg)
        toward = (self.subject - position) / self.standoff
        sideways = np.cross([0.0, 0.0, 1.0], toward)
        if np.linalg.norm(sideways) < 1e-6:
            sideways = np.array([1.0, 0.0, 0.0])
        sideways = sideways / np.linalg.norm(sideways)
        frame = np.column_stack([sideways, np.cross(toward, sideways), toward])

        x_before = Rotation.from_rotvec(follow_rotvec).as_matrix()[:, 0]
        roll = math.atan2(float(np.dot(np.cross(frame[:, 0], x_before), toward)), float(np.dot(frame[:, 0], x_before)))
        frame = frame @ Rotation.from_rotvec([0.0, 0.0, roll]).as_matrix()
        return Viewpoint(name, position, Rotation.from_matrix(frame).as_rotvec(),
                         azimuth_deg=float(azimuth_deg), elevation_deg=float(elevation_deg))


def orbit_views(tool_pose, subject, count, radius, arc_degrees=120.0, elevation_degrees=55.0,
                rings=1, max_height=None):
    """Viewpoints on a sphere about the subject, each aimed at it, and the sphere itself.

    Rings sweep alternately, so the move from the end of one ring to the start of the next is
    one azimuth step rather than the width of the whole arc. The views are rolled to follow
    the camera as it is now, which is only a first guess: the plan re-rolls each one to
    follow the view before it once it knows the order.

    `subject` is on the fabric, where the camera looks now: an orbit about a point down the
    lens axis instead is about a point inside the table whenever the camera is near the cloth.
    """
    subject = np.asarray(subject, dtype=float)
    standoff = float(radius)

    highest = float(elevation_degrees)
    if max_height is not None and standoff > 1e-6:
        highest = min(highest, math.degrees(math.asin(min(1.0, float(max_height) / standoff))))

    # The arc is centred on the direction the camera already looks from, so a sweep starts
    # near where the arm is rather than on the far side of the workspace.
    sphere = Sphere(subject, standoff, math.atan2(tool_pose[1] - subject[1], tool_pose[0] - subject[0]), highest)

    slices = max(1, int(count))
    elevations = np.linspace(highest, LOWEST_RING_DEG, int(rings)) if int(rings) > 1 else [highest]

    views = []
    for ring, high in enumerate(elevations):
        order = range(slices) if ring % 2 == 0 else reversed(range(slices))
        # A full turn has no ends, so its views divide it evenly rather than putting the last
        # one back on top of the first - and each ring is turned a fraction of a step against
        # the one above, so no two rings look from the same bearing.
        full = arc_degrees >= 360.0
        spacing = slices if full else max(1, slices - 1)
        stagger = ring / len(elevations) if full else 0.0
        for index in order:
            degrees = -0.5 * arc_degrees + arc_degrees * ((index + stagger) / spacing if slices > 1 else 0.5)
            views.append(sphere.view(f"az{degrees:+.0f} el{high:.0f}", degrees, high, tool_pose[3:6]))
    return views, sphere


def tilt_views(tool_pose, count, view_radius):
    """This project's own scan angles: one position, attitudes within a narrow cone."""
    points, _, names, rotvecs = scanner.expand_stations_to_angle_viewpoints(
        np.array([np.asarray(tool_pose[:3], dtype=float)]), int(count), float(view_radius), 0.0)
    return [Viewpoint(name, point, rotvec) for name, point, rotvec in zip(names, points, rotvecs)]


def looking_down(follow_rotvec):
    """A camera attitude with the lens straight down, rolled to keep the current camera's heading.

    Keeping the heading is what holds wrist 3 still across a whole grid: every station has
    the same attitude, so the arm only translates between them.
    """
    return looking_down_along(_heading_of(follow_rotvec))


def looking_down_along(heading):
    """The lens straight down with the image's x edge along `heading` (a direction in the plane)."""
    from scipy.spatial.transform import Rotation

    down = np.array([0.0, 0.0, -1.0])
    heading = np.array([float(heading[0]), float(heading[1]), 0.0])
    heading = heading / max(float(np.linalg.norm(heading)), 1e-9)
    return Rotation.from_matrix(np.column_stack([heading, np.cross(down, heading), down])).as_rotvec()


def _heading_of(rotvec):
    from scipy.spatial.transform import Rotation

    heading = Rotation.from_rotvec(rotvec).as_matrix()[:, 0]
    heading = np.array([heading[0], heading[1], 0.0])
    return heading / np.linalg.norm(heading) if np.linalg.norm(heading) > 1e-6 else np.array([1.0, 0.0, 0.0])


IDENTITY_TURN = ((1.0, 0.0), (0.0, 1.0))


def _image_from_tool(turn) -> np.ndarray:
    """With the lens straight down: which way the image's x edge points, per way the tool's x points.

    `turn` is the camera's image turn - its x and y axes in the tool's (calibrate_image_turn) - and
    a lens-down tool with x along h has y along (h_y, -h_x), so this is a plain 2x2 in the plane.
    """
    e = np.asarray(turn, dtype=float)
    return np.array([[e[0, 0], e[1, 0]], [-e[1, 0], e[0, 0]]])


def straight_heading(region, follow_rotvec, turn=IDENTITY_TURN):
    """The TOOL heading that squares the IMAGE to the cloth - its long edge, or the table's axes
    without an outline - whichever of the four does it nearest the image's heading now, so the wrist
    turns as little as possible. The image's own turn on its mount is undone here, which is what
    makes a straight view look straight rather than merely be straight at the flange."""
    to_image = _image_from_tool(turn)
    if region is not None and region.has_outline:
        hull = region.outline
        edges = np.roll(hull, -1, axis=0) - hull
        along = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
    else:
        along = np.array([1.0, 0.0])
    along = along / np.linalg.norm(along)
    across = np.array([-along[1], along[0]])
    now = to_image @ _heading_of(follow_rotvec)[:2]
    wanted = max((along, -along, across, -across), key=lambda candidate: float(np.dot(candidate, now)))
    heading = np.linalg.solve(to_image, wanted)
    return heading / np.linalg.norm(heading)


def calibrate_image_turn(tool_heading, shift_along_x, shift_along_y, height, focal_px):
    """The camera's image turn, from how far the picture slid when the arm moved +x and +y in the plane.

    Over a flat cloth, moving a lens-down camera by d slides the picture by -(f / h) times d seen in
    the camera's own x and y axes. The axes assumed so far are the tool's; comparing the slide
    measured with the slide those axes predict gives the turn between them - and its determinant
    says whether the image is also mirrored. Returned as the nearest orthogonal 2x2, with how far
    the measurement was from being one (0 is a clean rotation; large means the probes slipped).
    """
    h = np.asarray(tool_heading, dtype=float)[:2]
    h = h / np.linalg.norm(h)
    assumed = np.array([[h[0], h[1]], [h[1], -h[0]]])
    measured = -(float(height) / float(focal_px)) * np.column_stack([shift_along_x, shift_along_y])
    turn_t = measured @ np.linalg.inv(assumed)
    u, singular, vt = np.linalg.svd(turn_t.T)
    return u @ vt, float(abs(singular[0] - singular[1]) / max(singular.max(), 1e-9)), float(singular.mean())


def camera_rotation(rotvec, turn=IDENTITY_TURN) -> np.ndarray:
    """The camera's axes in the world: the tool's, turned in its own image plane by `turn`."""
    from scipy.spatial.transform import Rotation

    e = np.asarray(turn, dtype=float)
    block = np.eye(3)
    block[:2, :2] = e
    return Rotation.from_rotvec(rotvec).as_matrix() @ block


def straight_home(home_joints, tool_offset, region=None, turn=IDENTITY_TURN):
    """Home's camera position, with the lens exactly down and the image squared to the cloth - solved
    from Home itself, so the arm stays on Home's branch and only the wrist moves."""
    mj, model, data, site_id = load_model()
    flange = seed_from(mj, model, data, site_id, home_joints)
    camera = camera_point(flange, tool_offset)
    rotvec = looking_down_along(straight_heading(region, flange[3:6], turn))
    config, _, _ = solve_ik(mj, model, data, site_id, np.r_[flange_for_grasp(camera, rotvec, tool_offset), rotvec], "straight Home")
    return config


def fabric_point_below(camera, rotvec, fabric_z, subject_distance):
    """Where the camera's axis meets the fabric plane - the centre a grid is laid out around.

    A camera aimed nearly level never meets the plane in front of it, so it falls back to
    the point `subject_distance` down its axis, dropped onto the plane.
    """
    from scipy.spatial.transform import Rotation

    axis = Rotation.from_rotvec(rotvec).apply([0.0, 0.0, 1.0])
    reach = (fabric_z - camera[2]) / axis[2] if axis[2] < -0.2 else float(subject_distance)
    hit = np.asarray(camera, dtype=float) + axis * max(0.0, float(reach))
    return np.array([hit[0], hit[1], fabric_z])


def grid_views(tool_pose, rows, cols, camera_height, fabric_z, subject_distance, tool_offset,
               region=None, hfov_deg=55.0, overlap=0.3, turn=IDENTITY_TURN, per_station=1, tilt_deg=35.0, on_cloth=True):
    """Views at `camera_height` over the fabric plane: `per_station` of them at each station, all aimed at its centre.

    With `on_cloth` False these are overview views: a frame may take in more than the cloth, since the
    point is to see the whole of it at once, so no station is dropped for reaching past its edge.

    The first is straight down. The rest sit on a sphere of radius `camera_height` about the cloth
    point under the station, tilted `tilt_deg` from vertical and spread evenly around it, so one
    spot of cloth is seen from several directions.

    Close and square-on is what makes each frame sharp and comparable: the stitch scale is
    the same in every image. With a taught outline the grid fills it (fitted_grid_centres),
    keeping only stations whose whole frame lands on the cloth; without one it is a rows x
    cols block centred where the camera looks now. Either way neighbouring stations are one
    frame less the overlap apart, so a new height spreads the stations to match what a frame
    now takes in, and the grid is walked row by row, alternating direction, so every move is
    one step long.
    """
    from scipy.spatial.transform import Rotation

    camera = camera_point(tool_pose, tool_offset)
    rotvec = looking_down_along(straight_heading(region, tool_pose[3:6], turn))
    lift = np.array([0.0, 0.0, float(camera_height)])

    if region is not None and region.has_outline:
        centre = region.centre
        placed, _ = fitted_grid_centres(region, camera_height, hfov_deg, overlap)
        stations = {(row, col): np.r_[xy, fabric_z] for row, col, xy in placed
                    if not on_cloth or region.overshoot(footprint(np.r_[xy, fabric_z] + lift, rotvec, fabric_z, hfov_deg)) <= FRAME_TOLERANCE}
    else:
        centre = fabric_point_below(camera, tool_pose[3:6], fabric_z, subject_distance)
        frame = Rotation.from_rotvec(rotvec).as_matrix()
        across, along = frame[:, 0], frame[:, 1]
        step = max(frame_side(camera_height, hfov_deg) * (1.0 - float(overlap)), 1e-3)
        stations = {(row, col): centre + step * ((col - 0.5 * (int(cols) - 1)) * across + (row - 0.5 * (int(rows) - 1)) * along)
                    for row in range(int(rows)) for col in range(int(cols))}

    views = []
    tilted = max(1, int(per_station)) - 1
    for row in sorted({row for row, _ in stations}):
        cols_here = sorted(col for r, col in stations if r == row)
        for col in (cols_here if row % 2 == 0 else reversed(cols_here)):
            name = f"r{row + 1}c{col + 1}"
            views.append(Viewpoint(name + (" down" if tilted else ""), stations[(row, col)] + lift, rotvec.copy(), elevation_deg=90.0, cell=(row, col), overview=not on_cloth))
            sphere =Sphere(stations[(row, col)], float(camera_height) / math.cos(math.radians(tilt_deg)), 0.0, 90.0)
            for index in range(tilted):
                azimuth = -180.0 + 360.0 * index / tilted
                view = sphere.view(f"{name} az{azimuth:+.0f} tilt{tilt_deg:.0f}", azimuth, 90.0 - float(tilt_deg), rotvec)
                views.append(dataclasses.replace(view, cell=(row, col), aim=stations[(row, col)]))
    return views, centre


def frame_stays_on(region, fabric_z, hfov_deg):
    """Whether a view's whole frame lands on the cloth - on the outline when there is one, else anywhere below."""
    def test(view):
        corners = footprint(view.position, view.rotvec, fabric_z, hfov_deg)
        if corners is None:
            return False
        return region is None or region.overshoot(corners) <= FRAME_TOLERANCE
    return test


def centred_views(tool_pose, subject, count, radius, rings, max_tilt_deg, region, fabric_z, hfov_deg, turn=IDENTITY_TURN):
    """Views aimed at `subject` from `radius` away: straight down, then `rings` rings each tilted further.

    Every direction tilts only as far as its frame stays on the cloth, searched a degree at a time
    and never past `max_tilt_deg`, and the rings share that out evenly - so an elongated cloth is
    viewed steeply across its narrow side and more obliquely along its long one, and no view loses
    the fabric. Rings alternate direction and are staggered, like the orbit's.
    """
    sphere = Sphere(np.asarray(subject, dtype=float), float(radius),
                    math.atan2(tool_pose[1] - subject[1], tool_pose[0] - subject[0]), 90.0)
    keeps_frame = frame_stays_on(region, fabric_z, hfov_deg)
    follow = looking_down_along(straight_heading(region, tool_pose[3:6], turn))

    def steepest_tilt(azimuth):
        allowed = 0.0
        for tilt in np.arange(1.0, float(max_tilt_deg) + 1e-6, 1.0):
            if not keeps_frame(sphere.view("probe", azimuth, 90.0 - tilt, follow)):
                break
            allowed = float(tilt)
        return allowed

    views = [sphere.view("straight down", 0.0, 90.0, follow)]
    slices, ring_count = max(1, int(count)), max(1, int(rings))
    for ring in range(1, ring_count + 1):
        order = range(slices) if ring % 2 else reversed(range(slices))
        for index in order:
            azimuth = -180.0 + 360.0 * (index + (ring - 1) / ring_count) / slices
            tilt = steepest_tilt(azimuth) * ring / ring_count
            if tilt < 0.5:
                continue
            views.append(sphere.view(f"az{azimuth:+.0f} tilt{tilt:.0f}", azimuth, 90.0 - tilt, follow))
    return views, sphere


# The splat dome's distances: each further one is this much more of the nearest, so two give the
# close detail and the context a trainer needs to place it, and three add an overview.
SPLAT_LAYER_STEP = 0.5


def splat_views(tool_pose, subject, per_ring, rings, layers, radius, max_tilt_deg, region=None, turn=IDENTITY_TURN,
                layer_step=None, layer_radii=None):
    """A dome of views for Gaussian Splatting: every one aimed at `subject`, from `layers` distances.

    A splat learns depth only from seeing each spot from clearly different directions, and its
    detail from seeing it close: so at each distance - `radius`, then SPLAT_LAYER_STEP more of it
    per layer - there is one view straight down and `rings` full rings of `per_ring` views, tilted
    evenly up to `max_tilt_deg`. Rings alternate direction and are staggered against each other,
    and every other layer is flown outside-in, so each move is one step of the dome. The views are
    left free to frame past the cloth: an edge and some table are what anchor a splat in space.
    Returns the views and the nearest layer's sphere, which the plan nudges views about.
    `layer_step` overrides SPLAT_LAYER_STEP: a fixed-focus lens is sharp only in a narrow band, and
    layers 50 % apart put two of three out of it. `layer_radii` gives the distances outright (`layers`
    and `radius` then follow it): the close layers in the sharp band plus one far one that sees the board.
    """
    step = SPLAT_LAYER_STEP if layer_step is None else float(layer_step)
    if layer_radii:
        distances = [float(r) for r in layer_radii]
        radius = min(distances)
    else:
        distances = [float(radius) * (1.0 + step * layer) for layer in range(max(1, int(layers)))]
    facing = math.atan2(tool_pose[1] - subject[1], tool_pose[0] - subject[0])
    follow = looking_down_along(straight_heading(region, tool_pose[3:6], turn))
    slices, ring_count = max(3, int(per_ring)), max(1, int(rings))
    views, ring_index = [], 0
    for layer, distance in enumerate(distances):
        sphere = Sphere(np.asarray(subject, dtype=float), distance, facing, 90.0)
        top = [sphere.view(f"d{layer + 1} straight down", 0.0, 90.0, follow)]
        tilted = []
        for ring in range(1, ring_count + 1):
            tilt = float(max_tilt_deg) * ring / ring_count
            order = range(slices) if ring_index % 2 == 0 else reversed(range(slices))
            ring_index += 1
            tilted += [sphere.view(f"d{layer + 1} az{azimuth:+.0f} tilt{tilt:.0f}", azimuth, 90.0 - tilt, follow)
                       for index in order for azimuth in [-180.0 + 360.0 * (index + (ring - 1) / ring_count) / slices]]
        views += top + tilted if layer % 2 == 0 else tilted[::-1] + top
    return views, Sphere(np.asarray(subject, dtype=float), float(radius), facing, 90.0)


def centre_view(tool_pose, camera_height, fabric_z, region, subject_distance, tool_offset, centre_xy=None, turn=IDENTITY_TURN):
    """One straight-down view at `camera_height` over a point of the fabric.

    The point is `centre_xy` when given - what the camera-guided centring steps to - else the
    outline's middle, else where the camera looks now. Built as a one-view plan so the move there
    is solved and checked exactly as a capture path's are.
    """
    if centre_xy is not None:
        centre = np.r_[np.asarray(centre_xy, dtype=float)[:2], fabric_z]
    elif region is not None and region.has_outline:
        centre = region.centre
    else:
        centre = fabric_point_below(camera_point(tool_pose, tool_offset), tool_pose[3:6], fabric_z, subject_distance)
    view = Viewpoint("fabric centre", centre + [0.0, 0.0, float(camera_height)],
                     looking_down_along(straight_heading(region, tool_pose[3:6], turn)), elevation_deg=90.0)
    return [view], centre


def taught_views(mj, model, data, site_id, taught, tool_offset):
    """The views the user posed the arm at by hand, as recorded - joints first, geometry derived.

    A taught view is flown in exactly the joints it was recorded in, so nothing is solved;
    the camera pose is forward kinematics of those joints, kept for the preview and the
    capture records.
    """
    views = []
    for index, entry in enumerate(taught):
        config = np.asarray(entry["joints"], dtype=float)
        flange = seed_from(mj, model, data, site_id, config)
        views.append(Viewpoint(str(entry.get("name") or f"taught {index + 1}"), camera_point(flange, tool_offset),
                               np.asarray(flange[3:6], dtype=float), config=config, fixed=True))
    return views


# ============================================================================
# The fabric's outline, and what a camera frame sees of it
# ============================================================================

@dataclass
class FabricRegion:
    """The fabric as taught: its plane's height, and - from three corners on - its outline.

    Corners are camera points, touched to the cloth in free drive, so the outline is in the
    same frame every plan is written in.
    """

    corners: np.ndarray

    @property
    def z(self) -> float:
        return float(np.mean(self.corners[:, 2]))

    @property
    def has_outline(self) -> bool:
        return len(self.corners) >= 3

    @property
    def outline(self) -> np.ndarray:
        """The corners' convex hull in the plane, counter-clockwise - the order the edge tests assume."""
        points = sorted({(float(x), float(y)) for x, y in self.corners[:, :2]})
        if len(points) < 3:
            return np.array(points, dtype=float)

        def half(chain):
            hull = []
            for point in chain:
                while len(hull) >= 2 and _cross(hull[-2], hull[-1], point) <= 0.0:
                    hull.pop()
                hull.append(point)
            return hull

        lower, upper = half(points), half(reversed(points))
        return np.array(lower[:-1] + upper[:-1], dtype=float)

    @property
    def centre(self) -> np.ndarray:
        return np.r_[np.mean(self.outline, axis=0), self.z]

    def overshoot(self, footprint) -> float:
        """How far the furthest point of `footprint` lies outside the outline; 0 when all inside."""
        if footprint is None:
            return math.inf
        hull = self.outline
        worst = 0.0
        for a, b in zip(hull, np.roll(hull, -1, axis=0)):
            edge = b - a
            length = float(np.linalg.norm(edge))
            if length < 1e-9:
                continue
            for point in np.asarray(footprint, dtype=float)[:, :2]:
                worst = max(worst, -_cross(a, b, point) / length)
        return worst


def _cross(a, b, point):
    return float((b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]))


def footprint(position, rotvec, fabric_z, hfov_deg):
    """Where the corners of the camera's frame land on the fabric plane, or None if one misses it.

    Taken as a square as wide as the lens's horizontal field in both directions, because how
    the camera is rolled on its mount is not known here. That overstates the short side, so
    a frame judged inside the fabric is inside whichever way the camera is turned.
    """
    from scipy.spatial.transform import Rotation

    spread = math.tan(math.radians(float(hfov_deg)) / 2.0)
    turn = Rotation.from_rotvec(rotvec)
    corners = []
    for across, along in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        ray = turn.apply([across * spread, along * spread, 1.0])
        if ray[2] > -1e-3:
            return None
        reach = (float(fabric_z) - float(position[2])) / ray[2]
        if reach <= 0.0:
            return None
        corners.append(np.asarray(position, dtype=float) + reach * ray)
    return np.array(corners)


def outline_from_mask(mask, position, rotvec, turn, fabric_z, hfov_deg):
    """The cloth's outline on the fabric plane, from the pixels a detector called cloth: [x, y, z] corners.

    Every blob of the mask that is more than a speck - a striped cloth is found as several, one per
    light stripe - goes into one convex hull, which is simplified, and each hull vertex is
    followed along the camera's own ray - the lens axes turned by `turn` - down to the plane. Meant for
    a straight-down view that takes in the whole cloth; a cloth cut off by the frame comes out as
    cut off, since the frame's edge is then all that is known of it.
    """
    import cv2

    contours, _ = cv2.findContours(np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = [contour for contour in contours if cv2.contourArea(contour) >= 0.002 * np.asarray(mask).size]
    if not blobs:
        raise RuntimeError("no cloth was found in the picture")
    hull = cv2.convexHull(np.vstack(blobs))
    vertices = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True).reshape(-1, 2).astype(float)
    if len(vertices) < 3:
        raise RuntimeError("the cloth found in the picture is too small to outline")
    height, width = np.asarray(mask).shape[:2]
    focal = (width / 2.0) / math.tan(math.radians(float(hfov_deg)) / 2.0)
    axes = camera_rotation(rotvec, turn)
    corners = []
    for x, y in vertices:
        ray = axes @ np.array([(x - width / 2.0) / focal, (y - height / 2.0) / focal, 1.0])
        reach = (float(fabric_z) - float(position[2])) / ray[2] if ray[2] < -1e-3 else -1.0
        if reach <= 0.0:
            raise RuntimeError("the camera is not looking down at the cloth")
        corners.append(np.asarray(position, dtype=float) + reach * ray)
    return np.array(corners)


def frame_side(camera_height, hfov_deg):
    """How wide a strip of fabric one straight-down frame takes in from `camera_height`."""
    return 2.0 * float(camera_height) * math.tan(math.radians(float(hfov_deg)) / 2.0)


def fitted_grid_centres(region, camera_height, hfov_deg, overlap):
    """Straight-down stations filling the outline, each frame wholly inside it.

    Laid along the outline's longest edge, so a rectangular cloth is covered in rows rather
    than a staircase, stepping by one frame less the overlap. A cloth narrower than one frame
    gets a single row down its middle, which the frame check then keeps or refuses honestly.
    """
    hull = region.outline
    edges = np.roll(hull, -1, axis=0) - hull
    across = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
    across = across / np.linalg.norm(across)
    along = np.array([-across[1], across[0]])

    side = frame_side(camera_height, hfov_deg)
    step = max(side * (1.0 - float(overlap)), 1e-3)
    u, v = hull @ across, hull @ along

    def stations(low, high):
        usable = (high - low) - side
        count = int(math.floor(usable / step)) + 1 if usable > 0.0 else 1
        return (low + high) / 2.0 + (np.arange(count) - 0.5 * (count - 1)) * step

    rows, cols = stations(v.min(), v.max()), stations(u.min(), u.max())
    return [(row, col, across * cols[col] + along * rows[row]) for row in range(len(rows)) for col in range(len(cols))], across


# ============================================================================
# The plan
# ============================================================================

def build_capture_plan(joints, *, pattern="orbit", count=6, subject_distance=0.20, tool_offset=0.0306,
                       arc_degrees=360.0, elevation_degrees=75.0, rings=3, max_height=0.19,
                       view_radius=0.06, grid_rows=3, grid_cols=3, camera_height=0.07, orbit_radius=0.10,
                       taught=(), fabric_z=None, fabric_outline=(), lens_hfov_deg=55.0, grid_overlap=0.3,
                       centre_xy=None, max_tilt_deg=35.0, image_turn=IDENTITY_TURN, splat_views_per_ring=12,
                       splat_layers=2, grid_views_per_station=1, grid_height_passes=1, grid_overview_height=None, grid_overview_joints=None,
                       sweep_overlap=0.65, camera_mount=None, splat_subject=None, splat_layer_step=None,
                       splat_layer_radii=None) -> CapturePlan:
    """Solve and check a whole capture path before any of it can be commanded.

    Every viewpoint is solved from the previous one, so consecutive configurations stay on
    one branch, and every move - the first one, out of wherever the arm stands, included -
    is checked along its whole joint path for self-collision and for the camera or wrist
    coming down onto the fabric. The endpoints being clear says nothing about the swing
    between them. A view that cannot be flown safely is replaced by the nearest one that can
    (see _alternatives), or skipped with the reason, so one bad view no longer sinks the path;
    only a path with nothing left to fly is refused outright.

    `joints` is the arm's measured configuration, which decides where the subject is.

    `camera_mount` (the splat dome only) is the mount calibration as a 4x4 from a planned camera
    pose to the real lens (hand_eye.calibration_matrix). The real lens sits ~5 cm off and 17 deg
    tilted from the planner's camera, so a dome aimed with the planner's looked 4-7 cm beside its
    subject - round it, never at it. With the mount, the dome is centred where the real lens looks
    and laid out for the real lens; each view is turned back into the planner's camera before it
    is solved and checked, so every safety check is the usual one. `splat_subject` fixes the
    dome's centre instead of taking the fabric point the camera looks at from `joints`.
    """
    mj, model, data, site_id = load_model()
    fabric_z = fabric_height(mj, model, data, site_id, subject_distance, fabric_z)
    region = FabricRegion(np.asarray(fabric_outline, dtype=float).reshape(-1, 3)) if len(fabric_outline) else None
    here = np.asarray(joints, dtype=float)
    tool_pose = seed_from(mj, model, data, site_id, here)

    sphere = None
    subject = None
    if pattern == "spin":
        views = spin_views(tool_pose, count, tool_offset)
    elif pattern == "tilt":
        views = tilt_views(tool_pose, count, view_radius)
    elif pattern == "grid":
        views = []
        passes = max(1, int(grid_height_passes))
        top = float(grid_overview_height) if grid_overview_height else 0.0
        heights = np.geomspace(camera_height, top, passes) if passes > 1 and top > camera_height else camera_height * GRID_PASS_RISE ** np.arange(passes)
        has_overview = bool(top > camera_height and passes > 1)
        order = list(enumerate(heights))[::-1] if has_overview else list(enumerate(heights))
        for position, (index, height) in enumerate(order):
            overview = bool(has_overview and index == passes - 1)
            found, subject = grid_views(tool_pose, grid_rows, grid_cols, float(height), fabric_z, subject_distance, tool_offset, region,
                                        lens_hfov_deg, grid_overlap, image_turn, 1 if overview else grid_views_per_station, max_tilt_deg, not overview)
            if not found and index == 0:
                raise RuntimeError(f"No grid station fits inside the fabric outline at {camera_height * 100:.0f} cm - "
                                   "a frame there is wider than the cloth; lower the camera, or mark the corners again")
            if overview and grid_overview_joints is not None:
                position, rotvec = camera_pose(grid_overview_joints, tool_offset)
                found = [Viewpoint("r1c1", position, rotvec, elevation_deg=90.0, cell=(0, 0), overview=True, fixed=True,
                                   config=np.asarray(grid_overview_joints, dtype=float))]
            found = [dataclasses.replace(view, name=f"h{index + 1} {view.name}") for view in found] if passes > 1 else found
            views += found[::-1] if position % 2 else found
    elif pattern == "sweep":
        views, subject = grid_views(tool_pose, grid_rows, grid_cols, camera_height, fabric_z, subject_distance, tool_offset, region,
                                    lens_hfov_deg, sweep_overlap, image_turn)
        if not views:
            raise RuntimeError(f"No sweep station fits inside the fabric outline at {camera_height * 100:.0f} cm - "
                               "a frame there is wider than the cloth; lower the camera, or mark the corners again")
    elif pattern == "centred":
        subject = (region.centre if region is not None and region.has_outline
                   else fabric_point_below(camera_point(tool_pose, tool_offset), tool_pose[3:6], fabric_z, subject_distance))
        views, sphere = centred_views(tool_pose, subject, count, camera_height, rings, max_tilt_deg,
                                      region if region is not None and region.has_outline else None, fabric_z, lens_hfov_deg, image_turn)
    elif pattern == "splat":
        if camera_mount is not None:
            lens = _lens_pose(np.r_[camera_point(tool_pose, tool_offset), tool_pose[3:6]], camera_mount)
            subject = (np.asarray(splat_subject, dtype=float) if splat_subject is not None
                       else fabric_point_below(lens[:3], lens[3:6], fabric_z, subject_distance))
            views, sphere = splat_views(lens, subject, splat_views_per_ring, rings, splat_layers, camera_height, max_tilt_deg,
                                        region, image_turn, splat_layer_step, splat_layer_radii)
        else:
            subject = fabric_point_below(camera_point(tool_pose, tool_offset), tool_pose[3:6], fabric_z, subject_distance)
            views, sphere = splat_views(tool_pose, subject, splat_views_per_ring, rings, splat_layers, camera_height, max_tilt_deg,
                                        region, image_turn, splat_layer_step, splat_layer_radii)
    elif pattern == "centre":
        views, subject = centre_view(tool_pose, camera_height, fabric_z, region, subject_distance, tool_offset, centre_xy, image_turn)
    elif pattern == "taught":
        if not taught:
            raise RuntimeError("No taught views yet - put the arm in free drive, pose the camera by hand, and Record view")
        views = taught_views(mj, model, data, site_id, taught, tool_offset)
        subject = np.mean([subject_in_front_of(np.r_[view.position, view.rotvec], subject_distance) for view in views], axis=0)
    else:
        # With the fabric outlined, the orbit circles its middle wherever the arm happens to stand;
        # without, it circles the spot the camera looks at now, the only place known to be cloth.
        looked_at = (region.centre if region is not None and region.has_outline
                     else fabric_point_below(camera_point(tool_pose, tool_offset), tool_pose[3:6], fabric_z, subject_distance))
        views, sphere = orbit_views(tool_pose, looked_at, count, orbit_radius, arc_degrees, elevation_degrees, rings, max_height)
        subject = sphere.subject

    # Enter the sweep at whichever end is nearer. Starting from the far end spends a large
    # reconfiguration just getting there, for nothing. A taught path is flown in the order
    # it was taught, because that order is part of what was taught.
    if len(views) > 1 and pattern != "taught":
        first, last = [_probe(mj, model, data, site_id, view, tool_offset, here) for view in (views[0], views[-1])]
        if last is not None and first is not None and last < first:
            views.reverse()

    # With an outline, every automatic view is held to keeping its frame on the cloth. A taught
    # view reports where it strays but is not refused for it: it is where the user put it. The
    # splat dome is not held to it - its far, tilted views are meant to take in the cloth's edge.
    frame = None if region is None or not region.has_outline or pattern == "splat" else (region, float(lens_hfov_deg))
    solver = (mj, model, data, site_id, fabric_z, tool_offset, frame)
    mount = camera_mount if pattern == "splat" else None
    kept, skipped = [], []
    previous = here
    previous_rotvec = np.asarray(tool_pose[3:6], dtype=float) if mount is None else _lens_pose(
        np.r_[camera_point(tool_pose, tool_offset), tool_pose[3:6]], mount)[3:6]
    for view in views:
        # Only the first move leaves from where the arm actually stands; every later one
        # leaves from a planned viewpoint.
        leaving = not kept
        if pattern == "taught" or view.fixed:
            chosen, reason = _as_taught(solver, view, previous, leaving)
        else:
            candidates = _alternatives(pattern, view, sphere, subject, previous_rotvec)
            chosen, reason = _nearest_safe(solver, candidates if mount is None else _for_planner(candidates, mount), previous, leaving)
        if chosen is None:
            skipped.append((view.name, reason))
            continue
        kept.append(chosen)
        previous = chosen.config
        previous_rotvec = chosen.rotvec if mount is None else _lens_pose(np.r_[chosen.position, chosen.rotvec], mount)[3:6]

    if not kept:
        raise RuntimeError(f"No viewpoint of this {pattern} can be reached safely from here - "
                           + "; ".join(f"`{name}`: {reason}" for name, reason in skipped[:3]))

    # One station per fabric cell a view photographs, so the runner files each capture
    # under the cell it shows. Every pattern but the grid looks at a single cell.
    cells = list(dict.fromkeys(view.cell for view in kept))
    return CapturePlan(
        poses=np.array([np.r_[view.position, view.rotvec] for view in kept], dtype=float),
        view_names=[view.name for view in kept],
        station_ids=[cells.index(view.cell) for view in kept],
        station_cells=cells,
        viewpoints=kept,
        subject=np.asarray(kept[0].position if subject is None else subject, dtype=float),
        pattern=pattern,
        fabric_z=fabric_z,
        skipped=skipped,
        lens_hfov_deg=float(lens_hfov_deg),
        image_turn=[[float(v) for v in row] for row in np.asarray(image_turn, dtype=float)])


# How far a view may be moved to find a safe one near it, in the order they are tried: the
# nearest first, so the photograph taken is the closest safe one to the one asked for.
# Each row is (offsets, unit): a candidate's cost is its offsets measured in those units.
ORBIT_NUDGES = ((0.0, 5.0, 10.0, 15.0, -5.0, -10.0), 5.0), ((0.0, 10.0, -10.0, 20.0, -20.0), 10.0), ((1.0, 0.85, 0.7), 0.15)
GRID_PASS_RISE = 1.6
GRID_NUDGES =((0.0, 0.01, 0.02, 0.03, 0.05), 0.01), ((0.0, 0.25, 0.5), 0.25), ((1.0, 0.75, 0.5, 0.25, 0.0), 0.25)
MAX_ALTERNATIVES = 16
LOWEST_NUDGED_ELEVATION_DEG = 10.0


def _alternatives(pattern, view, sphere, centre, follow_rotvec):
    """`view` and the views near it, nearest first, each with a note of how it was moved."""
    import dataclasses
    import itertools

    if sphere is not None:
        (elevations, el_unit), (azimuths, az_unit), (scales, scale_unit) = ORBIT_NUDGES
        nudges = sorted((n for n in itertools.product(elevations, azimuths, scales)
                         if LOWEST_NUDGED_ELEVATION_DEG <= view.elevation_deg + n[0] <= sphere.highest_deg + 1e-6),
                        key=lambda n: abs(n[0]) / el_unit + abs(n[1]) / az_unit + (1.0 - n[2]) / scale_unit)
        # Each view is nudged about its own distance: a splat dome's views sit on several spheres.
        standoff = float(np.linalg.norm(np.asarray(view.position, dtype=float) - sphere.subject))
        for d_el, d_az, scale in nudges[:MAX_ALTERNATIVES]:
            elevation = view.elevation_deg + d_el
            moved = dataclasses.replace(sphere, standoff=standoff * scale).view(
                view.name, view.azimuth_deg + d_az, elevation, follow_rotvec)
            yield moved, ", ".join(note for note in (f"{d_el:+.0f} deg up" if d_el else "",
                                                     f"turned {d_az:+.0f} deg" if d_az else "",
                                                     f"{(1.0 - scale) * 100:.0f}% closer" if scale < 1.0 else "") if note)
    elif pattern in ("grid", "sweep"):
        (heights, height_unit), (shifts, shift_unit), (gentler, gentle_unit) = GRID_NUDGES
        tilted = view.aim is not None
        nudges = sorted(itertools.product(heights, shifts, gentler if tilted else (1.0,)),
                        key=lambda n: n[0] / height_unit + n[1] / shift_unit + (1.0 - n[2]) / gentle_unit)
        for d_z, shift, keep in nudges[:MAX_ALTERNATIVES]:
            position = np.asarray(view.position, dtype=float).copy()
            position[:2] = centre[:2] + (position[:2] - centre[:2]) * (1.0 - shift)
            position[2] += d_z
            moved = dataclasses.replace(view, position=position)
            if tilted:
                aim = np.asarray(view.aim, dtype=float).copy()
                aim[:2] = centre[:2] + (aim[:2] - centre[:2]) * (1.0 - shift)
                tilt = (90.0 - view.elevation_deg) * keep
                height = view.position[2] - aim[2] + d_z
                moved = dataclasses.replace(Sphere(aim, height / math.cos(math.radians(tilt)), 0.0, 90.0).view(
                    view.name, view.azimuth_deg, 90.0 - tilt, follow_rotvec), cell=view.cell, aim=aim)
            yield moved, ", ".join(note for note in (f"{d_z * 100:.0f} cm higher" if d_z else "",
                                                     f"{shift * 100:.0f}% nearer the centre" if shift else "",
                                                     f"tilt cut to {(90.0 - view.elevation_deg) * keep:.0f} deg" if keep < 1.0 else "") if note)
    else:
        yield view, ""


def _pose_matrix(pose) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = Rotation.from_rotvec(np.asarray(pose[3:6], dtype=float)).as_matrix(), np.asarray(pose[:3], dtype=float)
    return matrix


def _lens_pose(planned_pose, mount) -> np.ndarray:
    """Where the real lens is, for a planned camera pose: [x, y, z, rotvec], z along its view."""
    from scipy.spatial.transform import Rotation

    lens = _pose_matrix(planned_pose) @ np.asarray(mount, dtype=float)
    return np.r_[lens[:3, 3], Rotation.from_matrix(lens[:3, :3]).as_rotvec()]


def _for_planner(candidates, mount):
    """Views laid out for the real lens, as the planned camera poses that put the lens there."""
    import dataclasses

    from scipy.spatial.transform import Rotation

    back = np.linalg.inv(np.asarray(mount, dtype=float))
    for view, adjustment in candidates:
        planned = _pose_matrix(np.r_[view.position, view.rotvec]) @ back
        yield dataclasses.replace(view, position=planned[:3, 3].copy(),
                                  rotvec=Rotation.from_matrix(planned[:3, :3]).as_rotvec()), adjustment


def _nearest_safe(solver, candidates, previous_q, leaving):
    """The first of `candidates` the arm can reach and fly to safely, or None and why the last one could not."""
    reason = "nothing to try"
    for candidate, adjustment in candidates:
        try:
            chosen = _solve(solver, candidate, previous_q, leaving)
        except Unreachable as exc:
            reason = f"out of reach ({exc.err_pos * 1000:.0f} mm short)"
            continue
        if chosen.clear:
            chosen.adjustment = adjustment
            return chosen, ""
        reason = "; ".join(chosen.problems)
    return None, reason


def _as_taught(solver, view, previous_q, leaving):
    """A taught view is flown as taught or not at all - moving it would be un-teaching it."""
    mj, model, data, site_id, fabric_z, tool_offset, frame = solver
    check_move(mj, model, data, site_id, view, previous_q, fabric_z, tool_offset, leaving)
    _judge_frame(view, frame, fabric_z, enforced=False)
    return (view, "") if view.clear else (None, "; ".join(view.problems))


def _seeds(mj, model, data, site_id, previous_q, target_xy):
    """Whole-arm configurations to start a solve from, nearest first.

    The damped step only ever slides downhill from where it starts, so from the arm's own pose it
    can stall against a joint limit or a wrist singularity short of a pose the arm reaches easily
    with its shoulder and elbow set differently. Home is a known good branch for looking down at
    the cloth, and each start is also offered with the base already turned to face the target, so
    the shoulder, elbow and wrist only have to solve the reach and the height.
    """
    starts = []
    for start in (np.asarray(previous_q, dtype=float), np.asarray(ur5_robot.DEFAULT_HOME_JOINTS, dtype=float)):
        flange = seed_from(mj, model, data, site_id, start)[:2]
        base = np.asarray(data.xanchor[0][:2], dtype=float)
        turn = math.atan2(target_xy[1] - base[1], target_xy[0] - base[0]) - math.atan2(flange[1] - base[1], flange[0] - base[0])
        turned = start.copy()
        turned[0] = np.clip(start[0] + (turn + math.pi) % (2.0 * math.pi) - math.pi, model.jnt_range[0, 0], model.jnt_range[0, 1])
        starts += [start, turned]
    seeds = []
    for start in starts:
        if not any(np.max(np.abs(start - seed)) < math.radians(SEED_SEPARATION_DEG) for seed in seeds):
            seeds.append(start)
    return seeds


def _solve(solver, view, previous_q, leaving=False) -> Viewpoint:
    """`view` solved and its move checked, from the first start in _seeds that gets there safely.

    A start that reaches the pose but only through an unsafe move is kept as the answer to report
    if no other start does better; a pose no start reaches is refused with the closest miss.
    """
    import dataclasses

    mj, model, data, site_id, fabric_z, tool_offset, frame = solver
    target = np.r_[flange_for_grasp(view.position, view.rotvec, tool_offset), view.rotvec]
    refused, missed = None, None
    for seed in _seeds(mj, model, data, site_id, previous_q, target[:2]):
        attempt = dataclasses.replace(view)
        seed_from(mj, model, data, site_id, seed)
        try:
            attempt.config, attempt.err_pos, attempt.err_rot = solve_ik(mj, model, data, site_id, target, view.name)
        except Unreachable as exc:
            missed = exc if missed is None or exc.err_pos < missed.err_pos else missed
            continue
        check_move(mj, model, data, site_id, attempt, previous_q, fabric_z, tool_offset, leaving)
        _judge_frame(attempt, frame, fabric_z, enforced=True)
        if attempt.clear:
            return attempt
        refused = refused or attempt
    if refused is not None:
        return refused
    raise missed


def _judge_frame(view, frame, fabric_z, enforced):
    """How far the view's frame strays past the outline - nothing to judge without one."""
    if frame is None or view.overview:
        return
    region, hfov_deg = frame
    view.overshoot = region.overshoot(footprint(view.position, view.rotvec, fabric_z, hfov_deg))
    view.frame_enforced = enforced


def _probe(mj, model, data, site_id, view, tool_offset, here):
    """How far `view` is from `here` in joint space, or None if it cannot be reached.

    Used only to decide which end of a sweep to enter from, so a refusal here is not fatal -
    the real solve reports it properly.
    """
    seed_from(mj, model, data, site_id, here)
    try:
        config, _, _ = solve_ik(mj, model, data, site_id,
                                np.r_[flange_for_grasp(view.position, view.rotvec, tool_offset), view.rotvec], view.name)
    except Unreachable:
        return None
    return float(np.max(np.abs(config - here)))
