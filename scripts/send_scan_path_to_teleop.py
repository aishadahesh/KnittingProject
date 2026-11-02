"""Drive CraftLabXRFleet's TeleopApp UR5e from the terminal, using this project's own geometry.

The capture path, the inverse kinematics and the gRPC calls all live in ``src``:
``scan_geometry`` plans and solves the path, ``teleop_transport`` speaks to the app. This
script is a command line over them, so the terminal and the UR5 Robot Mode panel cannot
disagree about what a pattern means or how a move is paced.

Joint angles cross the boundary rather than Cartesian poses. The two models disagree about
where the arm stands in the world, but they agree exactly about the six joints, so a
configuration transfers where a pose would not.

TeleopApp must be running; it serves gRPC on 50052. Whether the arm follows is the app's
state rather than this script's: without ``--arm`` the poses move only its virtual robot,
which is the way to watch a path before committing an arm to it.

On the real arm, in order::

    python scripts/send_scan_path_to_teleop.py --preflight
    python scripts/send_scan_path_to_teleop.py --teach-rest
    python scripts/send_scan_path_to_teleop.py --pattern spin --angles 3
    python scripts/send_scan_path_to_teleop.py --pattern spin --angles 3 --run --arm
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np

import scan_geometry as geo
import teleop_transport


def connect(args) -> teleop_transport.TeleopAppTransport:
    transport = teleop_transport.TeleopAppTransport(
        target=args.target,
        fleet_python=args.fleet_python,
        tool_offset=args.tool_offset,
        max_joint_speed_deg=args.speed,
        floor_seconds=args.min_seconds)
    try:
        transport.connect(args.address)
    except Exception as exc:
        raise SystemExit(f"Cannot reach TeleopApp at {args.address}: {exc}\n\n"
                         "Start TeleopApp first - it serves gRPC on 50052, and nothing here can move an arm without it.")
    return transport


def report_preflight(transport) -> bool:
    """Everything that has to be true before a path drives a real arm.

    Each row is (what it checks, whether it holds, what to do about it), so the report and
    the verdict come from one list rather than a sequence of prints.
    """
    status = transport._refresh(force=True)
    checks = [
        ("on real hardware", transport.on_hardware,
         "the app is on the emulator - restart it with the arm reachable, since ur5 is not swappable at runtime"),
        ("connected", bool(status.get("connected")),
         "the controller is not answering - check the link, and that the arm is powered and in Remote Control"),
        ("`Rest` pose taught", "Rest" in transport.pose_names,
         "park has nowhere to send the arm, so the app cannot walk it home on exit - run --teach-rest"),
        ("`Home` pose taught", "Home" in transport.pose_names,
         "Home is where the simulated and the real arm are defined to agree - teach it, then Home means something"),
        ("following", transport.following,
         "the app is not commanding the arm, so a move would only shift its virtual robot - run with --arm"),
    ]

    print(f"{'check':<24} {'':<6} note")
    for name, ok, remedy in checks:
        print(f"{name:<24} {'PASS' if ok else 'FAIL':<6} {'' if ok else remedy}")

    gap = transport.gap_deg
    print(f"\nmotors {'ON' if status.get('motorsOn') else 'off'}, mode `{status.get('trackingMode')}`, "
          f"tracking gap {'not evaluable' if gap is None else f'{gap:.1f} deg'}")

    failed = [name for name, ok, _ in checks if not ok]
    print("NOT READY" if failed else "READY")
    return not failed


def teach_rest(transport) -> None:
    """Hand-guide the arm, then save where it stands as `Rest`.

    The pose is taught rather than chosen here on purpose: parkOnShutdown drives the arm to
    `Rest` on every exit, so inventing one would be inventing a trajectory the arm takes
    every time the app closes.
    """
    print(f"`{transport.target}` is going into free drive - it becomes hand-guidable.")
    transport.client.call("setTrackingMode", transport.target, "free", False)

    input("Move the arm to its rest pose, then press Enter (Ctrl+C to abandon): ")

    transport.client.call("savePose", transport.target, "Rest")
    print("saved `Rest` at " + ", ".join(f"{name}={math.degrees(value):+.1f}deg"
                                        for name, value in zip(geo.TELEOP_JOINTS, transport.measured_joints())))

    transport.client.call("setTrackingMode", transport.target, "holding", True)
    print("back to holding - motors on, following nothing")


def jog(transport, axis, metres) -> None:
    """One step along a base-frame axis from where the arm is now, orientation held."""
    before = np.asarray(transport.tool_pose(), dtype=float)
    asked = before.copy()
    asked["XYZ".index(axis)] += float(metres)

    print(f"jog {axis} {metres:+.3f} m from [{before[0]:+.3f}, {before[1]:+.3f}, {before[2]:+.3f}]"
          f" ({'the arm follows' if transport.following else 'virtual robot only'})")
    transport.move_to_pose(asked, 0.0, 0.0)
    transport._await_idle(60.0)

    after = np.asarray(transport.tool_pose(), dtype=float)
    print(f"tool point moved [{after[0] - before[0]:+.3f}, {after[1] - before[1]:+.3f}, {after[2] - before[2]:+.3f}] m")


def print_plan(plan) -> None:
    print(f"\npattern {plan.pattern}, subject [{plan.subject[0]:+.3f}, {plan.subject[1]:+.3f}, {plan.subject[2]:+.3f}]")
    print(f"\n{'view':<16} {'ik pos err':>12} {'transition':>12} {'self-collision':>16}")
    for point in plan.viewpoints:
        verdict = "CLEAR" if point.clear else f"{point.contacts + point.path_contacts} CONTACTS"
        print(f"{point.name:<16} {point.err_pos * 1000:>9.1f} mm {point.transition_deg:>8.0f} deg {verdict:>16}")


def run_plan(transport, plan, dwell) -> None:
    """Send the solved path, one smooth ramp per viewpoint, stopping on a failed arrival.

    A viewpoint that does not arrive ends the run: the app stops commanding when a
    commanded pose runs more than 30 degrees ahead of the measurement, and a loop that
    keeps sending past that point walks a virtual robot away from a standing arm.
    """
    print(f"\n{'view':<16} {'ramp':>7} {'sent (deg)':>46} {'arrival':>10}")
    for point in plan.viewpoints:
        ramp = transport.ramp_for(point.config)
        transport.move_to_joints(point.config, 0.0, 0.0)

        deadline = time.monotonic() + geo.timeout_for(ramp)
        arrived = False
        while transport.following and time.monotonic() < deadline:
            arrived = transport.arrived(point.config)
            if arrived:
                break
            time.sleep(0.2)

        degrees = " ".join(f"{math.degrees(value):+7.1f}" for value in point.config)
        reached = "arrived" if arrived else ("virtual" if not transport.following else "TIMED OUT")
        print(f"{point.name:<16} {ramp:>6.1f}s {degrees:>46} {reached:>10}", flush=True)

        if transport.following and not arrived:
            raise SystemExit(f"STOPPED: `{point.name}` never arrived - the arm is not where the path thinks it is")
        time.sleep(dwell)
    print("\ndone")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--address", default=teleop_transport.DEFAULT_ADDRESS, help="TeleopApp's gRPC address")
    parser.add_argument("--target", default=teleop_transport.DEFAULT_TARGET, help="which TeleopApp target to drive")
    parser.add_argument("--fleet-python", default=teleop_transport.DEFAULT_FLEET_PYTHON,
                        help="the CraftLabXRFleet python directory holding protostubs.py")

    parser.add_argument("--pattern", choices=["orbit", "spin", "tilt"], default="orbit", help="which capture path to fly")
    parser.add_argument("--angles", type=int, default=5, help="viewpoints per ring")
    parser.add_argument("--arc", type=float, default=120.0, help="orbit: degrees swept around the subject")
    parser.add_argument("--elevation", type=float, default=55.0, help="orbit: degrees above the subject")
    parser.add_argument("--rings", type=int, default=1, help="orbit: how many elevations the arc repeats at")
    parser.add_argument("--max-rise", type=float, default=0.12, help="orbit: cap on the camera's rise above the subject [m]")
    parser.add_argument("--view-radius", type=float, default=0.06, help="tilt: sets the tool tilt, as in Scan Mode")
    parser.add_argument("--subject-distance", type=float, default=0.20, help="how far in front of the camera the subject sits [m]")
    parser.add_argument("--tool-offset", type=float, default=0.0306,
                        help="flange-to-camera distance along the flange's +Z [m]; 0.0306 is the camera mount read off the controller")

    parser.add_argument("--speed", type=float, default=12.0, help="joint speed a move is paced by [deg/s]")
    parser.add_argument("--min-seconds", type=float, default=6.0, help="floor on how long one move takes [s]")
    parser.add_argument("--dwell", type=float, default=0.5, help="pause at each pose [s]")

    parser.add_argument("--run", action="store_true", help="send the path; without it the path is planned, reported, and nothing moves")
    parser.add_argument("--arm", action="store_true", help="energize and track, so the arm follows rather than only the virtual robot")
    parser.add_argument("--preflight", action="store_true", help="report whether the arm is ready for a real run, and move nothing")
    parser.add_argument("--teach-rest", action="store_true", help="free drive, then save where the arm stands as the `Rest` pose park needs")
    parser.add_argument("--jog", choices=["X", "Y", "Z"], default=None, help="move one step along this base-frame axis and stop")
    parser.add_argument("--jog-metres", type=float, default=0.10, help="how far --jog steps, signed [m]")
    return parser.parse_args()


def main():
    args = parse_args()
    transport = connect(args)

    if args.preflight:
        raise SystemExit(0 if report_preflight(transport) else 1)
    if args.teach_rest:
        teach_rest(transport)
        return

    if transport.on_hardware:
        print(f"!! `{args.target}` is on REAL hardware - give the arm clear space before arming", flush=True)
    if args.arm:
        transport.reset_stop()
        print("armed: motors on, tracking, and commanding confirmed")
    print(f"arm is {'FOLLOWING' if transport.following else 'not being commanded - virtual robot only'}")

    if args.jog:
        jog(transport, args.jog, args.jog_metres)
        return

    plan = geo.build_capture_plan(
        transport.measured_joints(),
        pattern=args.pattern, count=args.angles, subject_distance=args.subject_distance,
        tool_offset=args.tool_offset, arc_degrees=args.arc, elevation_degrees=args.elevation,
        rings=args.rings, max_height=args.max_rise, view_radius=args.view_radius)
    print_plan(plan)

    clear, issues = plan.assess()
    if not clear:
        raise SystemExit("\nREFUSED: " + "; ".join(issues) + " - nothing was sent")
    if not args.run:
        print(f"\n{len(plan.viewpoints)} pose(s) reachable and clear. Nothing moved; pass --run to send it.")
        return

    run_plan(transport, plan, args.dwell)


if __name__ == "__main__":
    main()
