"""Export a scan taken over the printed ChArUco board as a splatting dataset placed by the board.

    uv run python scripts\\export_board_dataset.py ur5_20261029_...
    uv run python scripts\\export_board_dataset.py ur5_20261029_... --output splat_dataset_board

Every picture that sees the board is placed by it; the rest of the scan through the arm, with the
mount fitted to this scan's own board views; the lens is calibrated from the board views. See
src/charuco.py. Prints the numbers that say whether the map is good: the board fit (want < 0.5 px),
the lens (< 1 px) and, for any picture placed through the arm, the mount fit (< 2 px).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import charuco  # noqa: E402
import gui_captures  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session")
    parser.add_argument("--output", default="splat_dataset_board")
    parser.add_argument("--several-sessions", action="store_true",
                        help="a combined session of scans of the same, untouched cloth (a run paused and resumed)")
    args = parser.parse_args()

    folder = ROOT / "robot_scans" / args.session
    records = [r for r in gui_captures.load_session(folder) if Path(str(r.get("path", ""))).is_file()]
    if not records:
        raise SystemExit(f"no pictures in {folder}")
    dataset = charuco.export_dataset(records, folder / args.output, session_id=args.session,
                                     allow_several_sessions=args.several_sessions)
    transforms = json.loads(dataset.transforms_path.read_text(encoding="utf-8"))
    from_board = sum(1 for f in transforms["frames"] if f.get("pose_from") == "board")
    print(f"{dataset.image_count} of {len(records)} pictures placed: {from_board} by the board, "
          f"{dataset.image_count - from_board} through the arm")
    print(f"board fit (median over pictures): {transforms.get('board_pose_rms_px')} px")
    print(f"lens: {transforms['intrinsics_source']}; f {transforms['fl_x']:.1f} px, "
          f"k1 {transforms['k1']:+.4f}, error {transforms['lens'].get('reprojection_error_px', float('nan')):.2f} px")
    if "mount_fit" in transforms:
        fit = transforms["mount_fit"]
        print(f"mount fit for the arm-placed pictures: {fit['rms_px']:.2f} px (worst picture {fit['max_view_px']:.2f} px)")
    for note in dataset.warnings:
        print("  note: " + note)
    print(f"written to {dataset.root}")


if __name__ == "__main__":
    main()
