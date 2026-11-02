"""The lens, measured: focal length, image centre and distortion from pictures of a checkerboard.

    uv run python scripts\\calibrate_lens.py --pictures <folder of checkerboard photos>
    uv run python scripts\\calibrate_lens.py --camera 0 --count 20        (take them now, from the gripper camera)
    uv run python scripts\\calibrate_lens.py --camera 0 --charuco         (the splat scan's ChArUco board instead)

Everything that places a pixel in space - the mount calibration, the exported camera poses, the
splat - depends on knowing where a pixel's ray goes. Until now that came from an assumed field of
view (and the mount fit's rough guess at it), with no lens distortion at all. This measures it the
standard way: a checkerboard (print one - 9 x 6 inner corners, any square size, flat on cardboard)
photographed 15-25 times at different places in the frame, near and far, tilted every way. OpenCV
finds the corners and solves fx, fy, cx, cy and the distortion k1, k2, p1, p2, k3.

Written to config/camera_intrinsics.json, which hand_eye.fit and gaussian_splatting.export_dataset
then use instead of the field-of-view guess. Delete that file to go back to the guess. The
reprojection error printed at the end should be under 0.5 px; over 1 px means the board was not
flat, the pictures were blurred, or the camera was refocused between them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from paths import CONFIG_DIR  # noqa: E402
from lens import INTRINSICS_JSON  # noqa: E402


def take_pictures(camera: int, count: int, folder: Path, columns: int, rows: int, detector=None) -> list[Path]:
    """Photograph the board with the camera: a picture is kept each time the board is found in a
    new place, with a short pause between, until `count` are kept. With a ChArUco `detector` the
    board need not be wholly in view."""
    import cv2

    folder.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        raise SystemExit(f"camera {camera} did not open")
    kept, last_centre, last_time = [], None, 0.0
    print(f"Show the board to the camera; move it between shots. Taking {count} pictures (Ctrl+C to stop early).")
    try:
        while len(kept) < count:
            ok, frame = capture.read()
            if not ok:
                continue
            if detector is not None:
                detection = detector.detect(frame, min_corners=MIN_CHARUCO_CORNERS)
                if detection is None:
                    continue
                centre = detection["image"].mean(axis=0)
            else:
                found, corners = cv2.findChessboardCorners(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (columns, rows),
                                                           cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK)
                if not found:
                    continue
                centre = corners.reshape(-1, 2).mean(axis=0)
            moved = last_centre is None or np.linalg.norm(centre - last_centre) > 40.0
            if moved and time.time() - last_time > 1.0:
                path = folder / f"board_{len(kept) + 1:02d}.png"
                cv2.imwrite(str(path), frame)
                kept.append(path)
                last_centre, last_time = centre, time.time()
                print(f"  {len(kept)}/{count} kept")
    except KeyboardInterrupt:
        pass
    capture.release()
    return kept


def calibrate(pictures: list[Path], columns: int, rows: int) -> dict:
    import cv2

    board = np.zeros((rows * columns, 3), dtype=np.float32)
    board[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    object_points, image_points, size, used = [], [], None, []
    for path in pictures:
        image = cv2.imread(str(path))
        if image is None:
            continue
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        size = grey.shape[::-1]
        found, corners = cv2.findChessboardCorners(grey, (columns, rows), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            print(f"  no board in {path.name}")
            continue
        corners = cv2.cornerSubPix(grey, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        object_points.append(board)
        image_points.append(corners)
        used.append(path.name)
    if len(used) < 8:
        raise SystemExit(f"the board was found in only {len(used)} pictures; the calibration needs at least 8 (15-25 is good)")
    error, matrix, distortion, _, _ = cv2.calibrateCamera(object_points, image_points, size, None, None)
    k1, k2, p1, p2, k3 = (float(v) for v in distortion.ravel()[:5])
    fx, fy, cx, cy = float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[0, 2]), float(matrix[1, 2])
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "k1": k1, "k2": k2, "p1": p1, "p2": p2, "k3": k3,
            "width": int(size[0]), "height": int(size[1]),
            "fov_degrees": float(np.degrees(2.0 * np.arctan(size[0] / (2.0 * fx)))),
            "reprojection_error_px": float(error), "pictures_used": used, "board": [columns, rows],
            "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S")}


MIN_CHARUCO_CORNERS = 20


def calibrate_charuco(pictures: list[Path]) -> dict:
    """The lens from pictures of the printed ChArUco board (scripts/make_charuco_board.py)."""
    import cv2

    import charuco

    spec = charuco.load_board_spec()
    if spec is None:
        raise SystemExit("no ChArUco board set up: run scripts/make_charuco_board.py first")
    detector = charuco.Detector(spec)
    detections, used, size = [], [], None
    for path in pictures:
        image = cv2.imread(str(path))
        detection = detector.detect(image, min_corners=MIN_CHARUCO_CORNERS) if image is not None else None
        if detection is None:
            print(f"  too little of the board in {path.name}")
            continue
        if detection["mirrored"]:
            print(f"  {path.name} is mirrored left to right - the camera driver flips it; calibrating the flipped picture")
        detections.append(detection)
        used.append(path.name)
        size = detection["size"]
    if len(used) < 8:
        raise SystemExit(f"the board was found in only {len(used)} pictures; the calibration needs at least 8 (15-25 is good)")
    return {**charuco.calibrate_lens(detections, size), "pictures_used": used}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pictures", default="", help="folder of checkerboard photos")
    parser.add_argument("--camera", type=int, default=None, help="take the photos now from this camera index")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--columns", type=int, default=9, help="inner corners across")
    parser.add_argument("--rows", type=int, default=6, help="inner corners down")
    parser.add_argument("--charuco", action="store_true",
                        help="photograph the printed ChArUco board (scripts/make_charuco_board.py) instead of a checkerboard")
    parser.add_argument("--output", default=str(INTRINSICS_JSON))
    args = parser.parse_args()
    if args.camera is not None:
        folder = Path(args.pictures) if args.pictures else CONFIG_DIR.parent / "robot_scans" / time.strftime("lens_%Y%m%d_%H%M%S")
        detector = None
        if args.charuco:
            import charuco

            spec = charuco.load_board_spec()
            if spec is None:
                raise SystemExit("no ChArUco board set up: run scripts/make_charuco_board.py first")
            detector = charuco.Detector(spec)
        pictures = take_pictures(args.camera, args.count, folder, args.columns, args.rows, detector)
    elif args.pictures:
        pictures = sorted(p for p in Path(args.pictures).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    else:
        raise SystemExit("give --pictures <folder> or --camera <index>")
    result = calibrate_charuco(pictures) if args.charuco else calibrate(pictures, args.columns, args.rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Lens: {result['fov_degrees']:.1f} deg across, fx {result['fx']:.1f} fy {result['fy']:.1f} px, centre ({result['cx']:.1f}, {result['cy']:.1f}), "
          f"k1 {result['k1']:+.4f} k2 {result['k2']:+.4f}; reprojection error {result['reprojection_error_px']:.2f} px over {len(result['pictures_used'])} pictures")
    print(f"Written to {args.output}. " + ("A board scan needs no mount calibration; keep this focus for the scan." if args.charuco
                                            else "Now redo the mount calibration (4 Results > Advanced) so it uses this lens."))


if __name__ == "__main__":
    main()
