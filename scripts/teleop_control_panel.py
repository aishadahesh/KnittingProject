"""Control panel for CraftLabXRFleet's TeleopApp UR5e, driving this project's scan geometry.

Every motion is planned before any of it is sent. A plan is a list of steps, each
a named joint configuration solved by this project's MuJoCo IK, and each carries
what the solve found: how far it landed from the pose asked for, and how many
self-collisions the model reports in that configuration. `Check` builds a plan and
reports it while moving nothing; `Preview` sends it to the app's virtual robot;
`Run` sends it with the arm following. Three motion kinds share that one path --
jog, go-to-target and the rotation sweep -- so a plan cannot be executed by a
route that skipped its checks.

**Simulator or real arm is decided when TeleopApp starts, not here.** The app binds
`ur5` to the controller if it answers and falls back to its emulator if it does not,
and that arm's row is not swappable at runtime. `Run` therefore drives whichever the
app is bound to, and the header says which.

Home: with "start from Home" on, every Run first sends the arm to the taught `Home`
pose and plans from there, so the simulator and the arm begin each motion from the
same configuration instead of from wherever the last one finished.

The collision check sees the ARM ONLY. The fabric, the table, the camera mount and
anything else in the cell are not in the MuJoCo model, so a clear check means "the
arm does not fold into itself", never "the cell is clear". The stop button commands
holding, which is a software stop and is not the hardware emergency stop.

Run with:  python scripts/teleop_control_panel.py
"""

from __future__ import annotations

import os
import queue
import sys
import math
import time
import threading
import tkinter as tk
from tkinter import messagebox, ttk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import send_scan_path_to_teleop as driver

STATUS_PERIOD_MS = 700
HOME_POSE = "Home"

# Right/left/up/down in the robot's own base frame, which today's jog measured to be
# the frame the MuJoCo model solves in: commanded +0.20 on X moved the controller's
# TCP +0.1986 on X. Axis names are kept alongside the words because which way is
# "right" depends on where you stand, and the numbers do not.
JOG_DIRECTIONS = [
    ("Right  +X", 0, 1.0),
    ("Left   -X", 0, -1.0),
    ("Fwd    +Y", 1, 1.0),
    ("Back   -Y", 1, -1.0),
    ("Up     +Z", 2, 1.0),
    ("Down   -Z", 2, -1.0),
]


class Step:
    """One planned configuration, with what the solve found on the way to it."""

    def __init__(self, name, q, err_pos, err_rot, contacts):
        self.name = name
        self.q = q
        self.err_pos = err_pos
        self.err_rot = err_rot
        self.contacts = contacts
        # Largest joint change from the pose before this one. Filled in once the plan
        # is complete, because how far a move reaches is a property of the pair.
        self.jump = 0.0

    @property
    def safe(self):
        return self.contacts == 0

    def report(self, seconds):
        return (f"  {self.name:<12} ik {self.err_pos * 1000:6.1f} mm / {self.err_rot:.4f} rad   "
                f"self-collision: {'CLEAR' if self.safe else str(self.contacts) + ' CONTACTS'}"
                f"   move {math.degrees(self.jump):5.0f} deg in {seconds:.1f} s")


class ControlPanel:
    def __init__(self, root, teleop, target):
        self.root = root
        self.teleop = teleop
        self.target = target
        self.messages = queue.Queue()
        self.worker = None

        root.title(f"UR5e control - TeleopApp `{target}`")

        self.binding = tk.StringVar(value="reading...")
        self.state = tk.StringVar(value="")
        self.pose = tk.StringVar(value="")
        self.sync = tk.StringVar(value="")
        self.pace = tk.StringVar(value="")

        self.from_home = tk.BooleanVar(value=True)
        self.ramp = tk.DoubleVar(value=6.0)
        self.jog_step = tk.DoubleVar(value=0.10)
        self.goal = [tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0)]
        self.rotations = tk.IntVar(value=5)
        self.view_radius = tk.DoubleVar(value=0.06)
        self.dwell = tk.DoubleVar(value=2.0)
        self.max_joint_speed = tk.DoubleVar(value=12.0)
        self.tool_offset = tk.DoubleVar(value=0.0306)
        self.elevation = tk.DoubleVar(value=55.0)
        self.rings = tk.IntVar(value=2)
        self.subject_distance = tk.DoubleVar(value=0.20)
        self.arc = tk.DoubleVar(value=120.0)
        self.max_step = tk.DoubleVar(value=15.0)
        self.split = tk.BooleanVar(value=False)
        self.max_height = tk.DoubleVar(value=0.12)
        self.pattern = tk.StringVar(value="orbit: N viewpoints around the subject, over the arc")

        self._build(root)
        self._poll_status()
        self._drain_messages()

    # ---------------------------------------------------------------- layout

    def _build(self, root):
        root.columnconfigure(0, weight=1)

        status = ttk.LabelFrame(root, text="Arm", padding=8)
        status.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        ttk.Label(status, textvariable=self.binding, font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(status, textvariable=self.state).grid(row=1, column=0, sticky="w")
        ttk.Label(status, textvariable=self.pose).grid(row=2, column=0, sticky="w")
        ttk.Label(status, textvariable=self.sync).grid(row=3, column=0, sticky="w")

        safety = ttk.LabelFrame(root, text="Arm control", padding=8)
        safety.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        for column, (label, action) in enumerate([
            ("Arm (motors + track)", self._arm),
            ("Stop (holding)", lambda: self._call("setTrackingMode", self.target, "holding", True)),
            ("Free drive", lambda: self._call("setTrackingMode", self.target, "free", False)),
            ("Park", lambda: self._call("park", self.target)),
        ]):
            ttk.Button(safety, text=label, command=action).grid(row=0, column=column, sticky="ew", padx=2)
        # The two timing knobs every motion obeys, so they live together rather than on one tab.
        # The ramp is the move: the arm is told to reach each pose in exactly this long, so a
        # smaller number is a faster arm whatever the distance.
        ttk.Label(safety, text="seconds per move  (lower = faster)").grid(row=1, column=0, columnspan=2, sticky="e", pady=(8, 0))
        ttk.Spinbox(safety, from_=0.5, to=120.0, increment=0.5, textvariable=self.ramp, width=6).grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Label(safety, text="wait at each pose [s]  (capture time)").grid(row=2, column=0, columnspan=2, sticky="e", pady=(4, 0))
        ttk.Spinbox(safety, from_=0.0, to=60.0, increment=0.5, textvariable=self.dwell, width=6).grid(row=2, column=2, sticky="w", pady=(4, 0))
        ttk.Label(safety, text="max joint speed [deg/s]  (paces big moves)").grid(row=3, column=0, columnspan=2, sticky="e", pady=(4, 0))
        ttk.Spinbox(safety, from_=1.0, to=90.0, increment=1.0, textvariable=self.max_joint_speed, width=6).grid(row=3, column=2, sticky="w", pady=(4, 0))
        ttk.Checkbutton(safety, text="split large moves into steps (rougher, only if a move will not track)",
                        variable=self.split).grid(row=4, column=0, columnspan=3, sticky="e", pady=(4, 0))
        ttk.Spinbox(safety, from_=2.0, to=30.0, increment=1.0, textvariable=self.max_step, width=6).grid(row=4, column=3, sticky="w", pady=(4, 0))
        ttk.Label(safety, textvariable=self.pace).grid(row=5, column=0, columnspan=4, sticky="e", pady=(4, 0))

        home = ttk.LabelFrame(root, text="Home", padding=8)
        home.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        ttk.Button(home, text="Go Home", command=lambda: self._start(self._plan_home_only, "go home")).grid(row=0, column=0, padx=2)
        ttk.Button(home, text="Teach Home", command=lambda: self._teach(HOME_POSE)).grid(row=0, column=1, padx=2)
        ttk.Button(home, text="Teach Rest", command=lambda: self._teach("Rest")).grid(row=0, column=2, padx=2)
        ttk.Checkbutton(home, text="start every motion from Home", variable=self.from_home).grid(row=0, column=3, padx=(12, 0))

        notebook = ttk.Notebook(root)
        notebook.grid(row=3, column=0, sticky="ew", padx=8, pady=4)

        jog = ttk.Frame(notebook, padding=8)
        notebook.add(jog, text="Jog")
        ttk.Label(jog, text="step [m]").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(jog, from_=0.005, to=0.5, increment=0.005, textvariable=self.jog_step, width=6).grid(row=0, column=1, sticky="w")
        ttk.Label(jog, text="robot base frame - one press is one step").grid(row=0, column=2, columnspan=4, sticky="w", padx=(12, 0))
        for column, (label, axis, sign) in enumerate(JOG_DIRECTIONS):
            ttk.Button(jog, text=label, width=10,
                       command=lambda a=axis, s=sign: self._start(lambda: self._plan_jog(a, s), "jog")
                       ).grid(row=1, column=column, padx=2, pady=(8, 0))

        goto = ttk.Frame(notebook, padding=8)
        notebook.add(goto, text="Go to target")
        for column, axis in enumerate("XYZ"):
            ttk.Label(goto, text=f"{axis} [m]").grid(row=0, column=column * 2, sticky="e", padx=(8 if column else 0, 2))
            ttk.Entry(goto, textvariable=self.goal[column], width=9).grid(row=0, column=column * 2 + 1, sticky="w")
        ttk.Button(goto, text="Read current", command=self._read_goal_from_arm).grid(row=0, column=6, padx=(12, 2))
        ttk.Label(goto, text="tool point, robot base frame; orientation is held").grid(row=1, column=0, columnspan=7, sticky="w", pady=(6, 0))
        self._motion_buttons(goto, self._plan_target, "go to target", row=2)

        turn = ttk.Frame(notebook, padding=8)
        notebook.add(turn, text="Rotations")
        for column, (label, var, lo, hi, inc) in enumerate([
            ("angles", self.rotations, 1, 36, 1),
            ("elevation [deg]", self.elevation, 5.0, 90.0, 5.0),
            ("rings", self.rings, 1, 4, 1),
            ("max camera rise [m]", self.max_height, 0.02, 0.5, 0.01),
            ("subject distance [m]", self.subject_distance, 0.02, 1.0, 0.01),
            ("arc [deg]", self.arc, 20.0, 360.0, 10.0),
            ("tilt radius (tilt only)", self.view_radius, 0.01, 0.2, 0.01),
            ("tool offset [m]", self.tool_offset, 0.0, 0.5, 0.001),
        ]):
            ttk.Label(turn, text=label).grid(row=0, column=column * 2, sticky="e", padx=(8 if column else 0, 2))
            ttk.Spinbox(turn, from_=lo, to=hi, increment=inc, textvariable=var, width=7).grid(row=0, column=column * 2 + 1, sticky="w")
        ttk.Label(turn, text="pattern").grid(row=1, column=0, sticky="e", pady=(6, 0))
        ttk.Combobox(turn, textvariable=self.pattern, width=44, state="readonly",
                     values=["orbit: N viewpoints around the subject, over the arc",
                             "spin: a full 360 about the tool axis, sliced N ways",
                             "tilt: the scan generator's cone of viewpoints"]
                     ).grid(row=1, column=1, columnspan=5, sticky="w", pady=(6, 0))
        ttk.Label(turn, text="one position, N orientations, pivoting about the tool point above").grid(row=2, column=0, columnspan=8, sticky="w", pady=(6, 0))
        ttk.Label(turn, text="the position is where the tool stands when you press: Home first if that box is ticked, else where it is now"
                  ).grid(row=3, column=0, columnspan=8, sticky="w")
        self._motion_buttons(turn, self._plan_rotations, "rotations", row=4)

        self.log = tk.Text(root, height=16, width=104, wrap="none")
        self.log.grid(row=4, column=0, sticky="nsew", padx=8, pady=(4, 8))
        root.rowconfigure(4, weight=1)

    def _motion_buttons(self, parent, planner, label, row):
        """Check, Preview and Run always appear together and always share one plan."""
        for column, (text, mode) in enumerate([("Check only", "check"), ("Preview in simulator", "preview"), ("Run on robot", "run")]):
            ttk.Button(parent, text=text, width=22,
                       command=lambda m=mode: self._start(planner, label, mode=m)
                       ).grid(row=row, column=column * 3, columnspan=3, sticky="ew", pady=(10, 0), padx=2)

    # ---------------------------------------------------------------- plumbing

    def _say(self, line):
        self.messages.put(line)

    def _drain_messages(self):
        while True:
            try:
                self.log.insert("end", self.messages.get_nowait() + "\n")
            except queue.Empty:
                break
            self.log.see("end")
        self.root.after(120, self._drain_messages)

    def _call(self, method, *args):
        try:
            self.teleop.call(method, *args)
            self._say(f"{method} ok")
        except Exception as exc:
            self._say(f"{method} failed: {exc}")

    def _status(self):
        return self.teleop.call("getStatus", self.target)

    def _poll_status(self):
        try:
            status = self._status()
            where = "REAL HARDWARE" if status["realHardware"] else "emulator"
            self.binding.set(f"{self.target}: {where}" + ("" if status["connected"] else "   NOT CONNECTED"))
            self.state.set(f"motors {'ON' if status['motorsOn'] else 'off'} | tracking {status['trackingMode']}"
                           f" | trajectory {'running' if status['trajectoryExecuting'] else 'idle'}"
                           f" | Home {'taught' if HOME_POSE in status['poses'] else 'MISSING'}"
                           f" | Rest {'taught' if 'Rest' in status['poses'] else 'MISSING'}")
            p = status["currentPose"]
            self.pose.set(f"tool point [{p['px']:+.3f}, {p['py']:+.3f}, {p['pz']:+.3f}] m   (as the app models it)")
            drift = max(abs(status["plannedQ"][j] - status["q"][j]) for j in driver.TELEOP_JOINTS)
            self.sync.set(f"simulator vs arm: {drift:.4f} rad")

            # What the two timing boxes mean in the units the arm moves in: a jog of the
            # current step size, and how long a rotation sweep of the current size will take.
            seconds = max(float(self.ramp.get()), 1e-3)
            poses = int(self.rotations.get()) + 2
            self.pace.set(f"a {self.jog_step.get() * 100:.0f} cm jog travels at {self.jog_step.get() / seconds * 100:.1f} cm/s"
                          f"   |   a move is given the longer of {seconds:.1f} s and what {self.max_joint_speed.get():.0f} deg/s needs"
                          f"   |   {poses} poses x ({seconds:.1f} s + {self.dwell.get():.1f} s wait)"
                          f" = {poses * (seconds + self.dwell.get()):.0f} s for the sweep")
        except Exception as exc:
            self.binding.set("TeleopApp not answering")
            self.state.set(str(exc)[:120])
        self.root.after(STATUS_PERIOD_MS, self._poll_status)

    def _read_goal_from_arm(self):
        try:
            mj, model, data, site_id = driver.scanner.load_ur5e_model_data()
            here = driver.station_from_arm(mj, model, data, site_id, self.teleop, self.target)
            for var, value in zip(self.goal, here):
                var.set(round(float(value), 4))
            self._say(f"target set to where the tool is now: {np.round(here, 4).tolist()}")
        except Exception as exc:
            self._say(f"could not read the current pose: {exc}")

    def _arm(self):
        status = self._status()
        if status["realHardware"] and not messagebox.askyesno(
                "Arm the real robot",
                f"`{self.target}` is on REAL HARDWARE. Arming energizes it and it follows every"
                " command. Clear space, hand on the emergency stop. Go?"):
            return

        # Put the virtual robot on the arm before arming. The app's deviation guard trips at
        # 30 degrees per joint, so arming across a standing gap wider than that stops the
        # commands on the very first tick - which then reads as "tracking" everywhere while
        # nothing actually follows. Seeding first leaves no gap to trip on, and no lurch.
        gap = math.degrees(max(abs(status["plannedQ"][j] - status["q"][j]) for j in driver.TELEOP_JOINTS))
        if gap > 1.0:
            self._say(f"the virtual robot is {gap:.0f} deg from the arm - seeding it onto the arm first")
            self._call("gotoJointAngles", self.target, {j: status["q"][j] for j in driver.TELEOP_JOINTS})
            driver.await_idle(self.teleop, self.target, driver.timeout_for(self.ramp.get()))

        self._call("setParams", self.target, {"rampTimeS": float(self.ramp.get())})
        self._call("setMotors", self.target, True)
        self._call("setTrackingMode", self.target, "tracking", True)

        time.sleep(0.5)
        if self._status()["commandSending"]:
            self._say("armed: the arm is following")
        else:
            self._say("NOT armed - the app is not commanding the arm. Its deviation guard stops commands"
                      " when the commanded pose runs more than 30 deg ahead of the measurement; check the app log.")

    def _teach(self, name):
        if not messagebox.askyesno(f"Teach {name}", f"`{self.target}` goes into free drive and becomes"
                                                    f" hand-guidable. Move it where you want `{name}`, then confirm."):
            return
        self._call("setTrackingMode", self.target, "free", False)
        if messagebox.askokcancel(f"Teach {name}", f"Arm positioned? This saves where it stands as `{name}`."):
            self._call("savePose", self.target, name)
            self._call("setTrackingMode", self.target, "holding", True)

    # ---------------------------------------------------------------- planners

    def _solve(self, mj, model, data, site_id, pose, name):
        q, err_pos, err_rot = driver.solve_ik(mj, model, data, site_id, pose, name)
        return Step(name, q, err_pos, err_rot, driver.contacts_at(mj, model, data, q))

    def _plan_home_only(self):
        return []

    def _plan_jog(self, axis, sign):
        mj, model, data, site_id = driver.scanner.load_ur5e_model_data()
        pose, name, _ = driver.jog_target(mj, model, data, site_id, self.teleop, self.target,
                                          axis, float(self.jog_step.get()) * sign)
        return [self._solve(mj, model, data, site_id, pose, name)]

    def _plan_target(self):
        mj, model, data, site_id = driver.scanner.load_ur5e_model_data()
        driver.station_from_arm(mj, model, data, site_id, self.teleop, self.target)

        pose = driver.scanner.get_tcp(mj, model, data, site_id).copy()
        wanted = np.array([var.get() for var in self.goal], dtype=float)
        travel = wanted - pose[:3]
        pose[:3] = wanted
        return [self._solve(mj, model, data, site_id, pose, f"target ({np.linalg.norm(travel) * 100:.1f} cm away)")]

    def _plan_rotations(self):
        mj, model, data, site_id = driver.scanner.load_ur5e_model_data()
        count = int(self.rotations.get())

        if self.pattern.get().startswith("orbit"):
            # The camera travels: N viewpoints around the subject, each seeing a different
            # side of it, rather than N frames from one place.
            steps = [self._solve(mj, model, data, site_id,
                                 np.r_[driver.flange_for_grasp(position, rotvec, float(self.tool_offset.get())), rotvec], name)
                     for name, position, rotvec in driver.orbit_targets(
                         mj, model, data, site_id, self.teleop, self.target, count,
                         float(self.subject_distance.get()), float(self.arc.get()),
                         float(self.elevation.get()), int(self.rings.get()), float(self.max_height.get()))]

            # Enter the arc at whichever end is nearer. Sweeping from the far end first
            # spends a large reconfiguration just getting there, for nothing.
            here = np.array([self._status()["q"][joint] for joint in driver.TELEOP_JOINTS], dtype=float)
            if len(steps) > 1 and np.max(np.abs(steps[-1].q - here)) < np.max(np.abs(steps[0].q - here)):
                steps.reverse()
            return steps

        if self.pattern.get().startswith("spin"):
            # A full turn about the tool axis: the position is held exactly, and N=3 means
            # where it stands now, plus 120 and 240 degrees.
            return [self._solve(mj, model, data, site_id, np.r_[flange, rotvec], name)
                    for name, flange, rotvec in driver.spin_targets(mj, model, data, site_id, self.teleop, self.target, count)]

        station = driver.station_from_arm(mj, model, data, site_id, self.teleop, self.target)
        points, _, names, rotvecs = driver.scanner.expand_stations_to_angle_viewpoints(
            np.array([station]), count, float(self.view_radius.get()), 0.0)

        steps = []
        for name, point, rotvec in zip(names, points, rotvecs):
            flange = driver.flange_for_grasp(point, rotvec, float(self.tool_offset.get()))
            steps.append(self._solve(mj, model, data, site_id, np.r_[flange, rotvec], name))
        return steps

    # ---------------------------------------------------------------- execution

    def _start(self, planner, label, mode="run"):
        if self.worker and self.worker.is_alive():
            self._say("busy - the last motion has not finished")
            return

        if mode == "run":
            status = self._status()
            if status["realHardware"] and not messagebox.askyesno(
                    "Run on the real robot",
                    f"`{self.target}` is on REAL HARDWARE. This moves the arm through the whole"
                    f" {label}, at {float(self.ramp.get()):.0f} s per move. Clear space, hand on the"
                    " emergency stop. Go?"):
                self._say("cancelled")
                return

        self.worker = threading.Thread(target=self._run, args=(planner, label, mode), daemon=True)
        self.worker.start()

    def _run(self, planner, label, mode):
        try:
            self.teleop.call("setParams", self.target, {"rampTimeS": float(self.ramp.get())})
            patience = driver.timeout_for(self.ramp.get())

            status = self._status()
            armed = status["motorsOn"] and status["commandSending"]
            if mode == "preview" and armed:
                self._say("the arm is armed, so nothing can be previewed without moving it - press Stop (holding) first")
                return
            if mode == "run" and not armed:
                self._say("not armed - press `Arm (motors + track)` first")
                return

            if mode != "check" and self.from_home.get():
                if HOME_POSE not in status["poses"]:
                    self._say(f"no `{HOME_POSE}` pose taught - press Teach Home, or clear the checkbox")
                    return
                self._say(f"going to `{HOME_POSE}` first, so this motion starts where the simulator does")
                driver.await_idle(self.teleop, self.target, patience)
                self.teleop.call("gotoPose", self.target, HOME_POSE)
                driver.await_idle(self.teleop, self.target, patience)

            self._say(f"planning {label}...")
            plan = planner()

            previous = np.array([status["q"][joint] for joint in driver.TELEOP_JOINTS], dtype=float)
            for step in plan:
                step.jump = float(np.max(np.abs(step.q - previous)))
                self._say(step.report(driver.move_seconds(previous, step.q, self.max_joint_speed.get(), self.ramp.get())))
                previous = step.q

            unsafe = [step for step in plan if not step.safe]
            if unsafe:
                self._say(f"REFUSED: {len(unsafe)} pose(s) fold the arm into itself - nothing was sent")
                return
            if mode == "check":
                self._say(f"{len(plan)} pose(s) reachable and clear of self-collision. Nothing was moved.")
                return

            self._say(f"executing {len(plan)} pose(s) - {'the arm follows' if armed else 'virtual robot only'}")
            previous = np.array([status["q"][joint] for joint in driver.TELEOP_JOINTS], dtype=float)
            mj, model, data, site_id = driver.scanner.load_ur5e_model_data()
            for step in plan:
                # A large reconfiguration is walked in pieces. One command that far ahead of
                # the arm is what trips the app's 30-degree deviation guard and strands the
                # sweep; each piece stays well inside it, and each is checked for
                # self-collision before it is sent, so the PATH is cleared and not just its
                # endpoints.
                # One ramp per viewpoint by default. Walking a move in pieces keeps every
                # command close to the arm, but each piece accelerates and decelerates on its
                # own - stop-start motion that is audibly harder on the arm than one continuous
                # ramp of the same duration. Pacing by joint speed is what keeps a single ramp
                # trackable; splitting is the fallback for when it is not.
                pieces = (driver.interpolate_steps(previous, step.q, self.max_step.get())
                          if self.split.get() else [np.asarray(step.q, dtype=float)])
                folded = sum(1 for piece in pieces if driver.contacts_at(mj, model, data, piece))
                if folded:
                    self._say(f"  {step.name:<12} REFUSED: the way there folds the arm into itself at"
                              f" {folded} of {len(pieces)} intermediate poses - nothing more was sent")
                    return

                whole = driver.move_seconds(previous, step.q, self.max_joint_speed.get(), self.ramp.get())
                seconds = max(whole / len(pieces), 0.5)
                patience = driver.timeout_for(seconds)
                if len(pieces) > 1:
                    self._say(f"  {step.name:<12} {math.degrees(step.jump):.0f} deg walked as {len(pieces)}"
                              f" steps of <={self.max_step.get():.0f} deg, {seconds:.1f} s each")

                stalled = False
                for piece in pieces:
                    self.teleop.call("setParams", self.target, {"rampTimeS": seconds})
                    driver.await_idle(self.teleop, self.target, patience)
                    self.teleop.call("gotoJointAngles", self.target,
                                     dict(zip(driver.TELEOP_JOINTS, (float(v) for v in piece))))
                    if armed and driver.await_arrival(self.teleop, self.target, piece, patience) is None:
                        stalled = True
                        break

                if not armed:
                    self._say(f"  {step.name:<12} sent as {len(pieces)} step(s)")
                    previous = step.q
                    time.sleep(float(self.dwell.get()))
                    continue

                if stalled:
                    after = self._status()
                    self._say(f"  {step.name:<12} DID NOT ARRIVE - stopping the sequence")
                    if not after["commandSending"]:
                        self._say("  the app has stopped commanding the arm - its 30 deg deviation guard tripped."
                                  " Press `Arm (motors + track)` to resume, and lower max step per command.")
                    return

                gap = driver.await_arrival(self.teleop, self.target, step.q, patience)
                self._say(f"  {step.name:<12} arrived {gap:.4f} rad ({len(pieces)} step(s), {seconds:.1f} s each)")
                previous = step.q
                time.sleep(float(self.dwell.get()))

            self._say("done")
        except Exception as exc:
            self._say(f"{label} failed: {exc}")


def main():
    args = driver.parse_args()
    teleop = driver.Teleop(args.address, driver.load_teleop_stubs(args.fleet_python))

    root = tk.Tk()
    ControlPanel(root, teleop, args.target)
    root.mainloop()


if __name__ == "__main__":
    main()
