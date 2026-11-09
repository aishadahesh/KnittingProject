"""Fit one scan's camera mount and lens to its own pictures, and export it as a splatting dataset.

    uv run python scripts\\fit_session.py ur5_20261026_162103
    uv run python scripts\\fit_session.py ur5_20261018_152616 --output splat_dataset_fit

Writes robot_scans\\<session>\\<output> (default splat_dataset_fit), ready for scripts\\train_splat.ps1.
See src/session_fit.py for how. Only the pictures the arm took (with a pose) are used.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import gui_captures  # noqa: E402
import hand_eye  # noqa: E402
import lens  # noqa: E402
import session_fit  # noqa: E402
from persistence import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session")
    parser.add_argument("--output", default="splat_dataset_fit")
    parser.add_argument("--mirrored", choices=("auto", "yes", "no"), default="auto")
    args = parser.parse_args()

    folder = ROOT / "robot_scans" / args.session
    records = [r for r in gui_captures.load_session(folder)
               if (r.get("actual_tcp_pose") or r.get("target_position")) and Path(str(r.get("path", ""))).is_file()]
    if not records:
        raise SystemExit(f"no pictures with an arm pose in {folder}")
    calibration = hand_eye.load_calibration()
    if calibration is None:
        raise SystemExit("no mount calibration (config/hand_eye.json) to start from")
    planes = [float(r["camera_position"][2]) - float(r["camera_height_above_fabric_m"]) for r in records
              if r.get("camera_position") is not None and r.get("camera_height_above_fabric_m") is not None]
    if not planes:
        raise SystemExit("the captures do not record the fabric height")
    plane_z = float(np.median(planes))
    print(f"{args.session}: {len(records)} pictures, fabric plane z {plane_z:.4f} m")
    started = time.monotonic()
    fitted = session_fit.fit(records, calibration, plane_z, mirrored={"auto": None, "yes": True, "no": False}[args.mirrored],
                             measured_lens=lens.load_intrinsics())
    lens_fit = fitted["lens"]
    print(f"fitted in {time.monotonic() - started:.0f} s: lens f {lens_fit['f']:.1f} px "
          f"({np.degrees(2 * np.arctan(fitted['size'][0] / (2 * lens_fit['f']))):.1f} deg across), centre ({lens_fit['cx']:.1f}, {lens_fit['cy']:.1f}), "
          f"k1 {lens_fit['k1']:+.3f} k2 {lens_fit['k2']:+.3f}; mirrored={fitted['mirrored']}; "
          f"{fitted['pictures_with_points']}/{fitted['pictures']} pictures matched; {len(fitted['points'])} seed points")
    moved = np.linalg.norm(fitted["mount"][:3, 3] - hand_eye.calibration_matrix(calibration)[:3, 3]) * 1000
    print(f"mount moved {moved:.1f} mm from the saved calibration (start: {fitted['start']})")
    dataset = session_fit.export_dataset(records, fitted, folder / args.output, session_id=args.session)
    for note in dataset.warnings[:10]:
        print("  " + note)
    if len(dataset.warnings) > 10:
        print(f"  ... {len(dataset.warnings) - 10} more")
    print(f"exported {dataset.image_count} of {len(records)} pictures to {dataset.root}")
    report = {"session": args.session, "rounds": fitted["rounds"], "lens": {k: float(v) for k, v in lens_fit.items()},
              "mirrored": fitted["mirrored"], "start": fitted["start"], "exported": dataset.image_count, "offered": len(records)}
    write_json(folder / args.output / 'fit_report.json', report, indent=2)


if __name__ == "__main__":
    main()
