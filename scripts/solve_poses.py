"""Camera poses for a capture, solved from the pictures themselves.

    .venv-splat\\Scripts\\python.exe scripts\\solve_poses.py --session robot_scans\\free_... --output robot_scans\\free_...\\splat_dataset

UR5 Robot Mode runs exactly this, for a hand-held capture (5 Free capture) and for a robot run
(4 Results > Gaussian Splatting). A hand-held capture has no poses but these; a robot run has the
arm's, but they are only as good as the camera mount is known, and a guessed mount spreads one
cloth into a fan of sheets - the pictures' own agreement does not depend on it. Recovering them is
COLMAP's structure from motion (pycolmap, CPU): SIFT features, every pair matched, then
incremental mapping. The largest model is written as nerfstudio's
transforms.json (with its sparse points as sparse_pc.ply, which splatfacto starts from), so
scripts/train_splat.ps1 trains on it like on a robot export.

Then it measures what the app judges the capture by - which pictures were placed, how well, and for
each one its distance along its centre ray (as a share of the first picture's), its tilt from the
fabric's normal and the side it looks from - into pose_report.json. The fabric's plane is fitted
to the sparse points. Needs pycolmap in .venv-splat (scripts/setup_splatting.ps1 installs it).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402
from persistence import write_json

# Sparse points with a larger reprojection error than this are left out of the plane fit.
PLANE_FIT_MAX_ERROR_PX = 2.0

# A knit lying flat is the case COLMAP's defaults handle worst, and with them a capture whose pictures
# all matched still came out as 15 of 52 placed, in 4 pieces. Two things make a plane hard:
#   - the lens cannot be calibrated from it, so the camera starts from the field of view the app
#     holds (a prior focal length) and keeps it, refining only the lens distortion;
#   - a hand moving over it gives views a few degrees apart, and the defaults want 16 degrees between
#     the first two views and 1.5 degrees for any point - so these angles are lowered to what a
#     hand-held walk over a cloth gives, and a picture is placed from fewer agreeing points.
INIT_MIN_TRI_ANGLE_DEG = 4.0
MIN_TRI_ANGLE_DEG = 0.5
MIN_POSE_INLIERS = 15


def equalise_exposure(images: Path, percentile: float = 95.0, limits=(0.5, 2.0)) -> None:
    """Undo the camera's automatic exposure and white balance across the pictures, in place: a
    per-channel gain brings each picture's bright tones (the white yarn, in nearly every frame) to
    the set's median. Left as shot, the same stitch had a different brightness and colour in every
    picture - 2x and 60 % across one sweep - and the splat could only blur it."""
    from PIL import Image

    paths = sorted(images.glob("*.png")) + sorted(images.glob("*.jpg"))
    bright = {}
    for path in paths:
        with Image.open(path) as image:
            small = np.asarray(image.convert("RGB").resize((max(8, image.width // 4), max(6, image.height // 4))), dtype=float)
        bright[path] = np.percentile(small.reshape(-1, 3), percentile, axis=0)
    if not bright:
        return
    reference = np.median(np.array(list(bright.values())), axis=0)
    for path, value in bright.items():
        gain = np.clip(reference / np.maximum(value, 1.0), *limits)
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=float) * gain[None, None, :]
        Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)).save(path)
    print(f"exposure equalised across {len(bright)} pictures", flush=True)


# A camera COLMAP placed this many times farther from the others than the typical one is a stray:
# a picture pinned by a handful of wrong matches. One such camera, 17,000 units out when the rest
# sat within 3, made nerfstudio scale the whole scene by its distance - the cloth shrank to a dot
# and every training on it came out as grey mush.
STRAY_CAMERA_FACTOR = 10.0


def mask_board(images: Path, output: Path) -> dict[str, float]:
    """A mask per picture with the board's paper black (charuco.board_paper_mask), written twice: as
    COLMAP reads masks (colmap_masks/<picture>.png) and as nerfstudio does (masks/<picture>).
    Returns each masked picture's masked share."""
    import cv2

    import charuco

    spec = charuco.load_board_spec()
    if spec is None:
        raise SystemExit("--mask-board: no board spec (config/charuco_board.json)")
    for folder in ("masks", "colmap_masks"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    shares = {}
    for picture in sorted(images.iterdir()):
        board, _ = charuco.board_paper_mask(cv2.imread(str(picture)), spec)
        keep = np.where(board, 0, 255).astype(np.uint8)
        cv2.imwrite(str(output / "masks" / picture.name), keep)
        cv2.imwrite(str(output / "colmap_masks" / (picture.name + ".png")), keep)
        if board.any():
            shares[picture.name] = round(float(board.mean()), 3)
    print(f"0/4 Board masked in {len(shares)} of {sum(1 for _ in images.iterdir())} pictures", flush=True)
    return shares


def drop_stray_cameras(transforms_path: Path) -> list[str]:
    """Removes cameras far from the rest from transforms.json, in place. Returns their picture names."""
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = transforms.get("frames", [])
    if len(frames) < 4:
        return []
    positions = np.array([np.array(f["transform_matrix"], dtype=float)[:3, 3] for f in frames])
    middle = np.median(positions, axis=0)
    distance = np.linalg.norm(positions - middle, axis=1)
    limit = STRAY_CAMERA_FACTOR * max(float(np.median(distance)), 1e-9)
    strays = [Path(f["file_path"]).name for f, d in zip(frames, distance) if d > limit]
    if strays:
        transforms["frames"] = [f for f, d in zip(frames, distance) if d <= limit]
        transforms["stray_cameras_dropped"] = strays
        write_json(transforms_path, transforms)
        print(f"dropped {len(strays)} stray camera(s) placed far from the rest: {', '.join(strays)}", flush=True)
    return strays


def measure(recon_dir: Path, names: list[str], models: int) -> dict:
    from nerfstudio.process_data.colmap_utils import qvec2rotmat, read_images_binary, read_points3D_binary

    images = read_images_binary(recon_dir / "images.bin")
    points = read_points3D_binary(recon_dir / "points3D.bin")
    xyz = np.array([p.xyz for p in points.values()], dtype=float).reshape(-1, 3)
    errors = np.array([p.error for p in points.values()], dtype=float)
    trusted = xyz[errors <= PLANE_FIT_MAX_ERROR_PX] if (errors <= PLANE_FIT_MAX_ERROR_PX).sum() >= 50 else xyz

    cameras = {}
    for image in images.values():
        rotation = qvec2rotmat(image.qvec)
        cameras[image.name] = (-rotation.T @ image.tvec, rotation.T @ np.array([0.0, 0.0, 1.0]))

    centre = np.median(trusted, axis=0)
    _, _, axes = np.linalg.svd(trusted - centre, full_matrices=False)
    across, normal = axes[0], axes[2]
    if np.mean([normal @ (position - centre) for position, _ in cameras.values()]) < 0.0:
        normal = -normal
    along = np.cross(normal, across)

    rows = []
    for name in names:
        if name not in cameras:
            continue
        position, look = cameras[name]
        facing = float(normal @ look)
        depth = float(normal @ (centre - position)) / facing if facing < -1e-6 else None
        tilt = math.degrees(math.acos(min(1.0, max(-1.0, -facing))))
        seen = position + look * depth if depth is not None else centre
        towards = position - seen
        rows.append({"name": name, "depth": depth, "tilt_from_vertical_deg": tilt,
                     "azimuth_deg": math.degrees(math.atan2(float(towards @ along), float(towards @ across)))})

    depths = [row["depth"] for row in rows if row["depth"] is not None]
    first = next((row["depth"] for row in rows if row["name"] == names[0] and row["depth"] is not None), max(depths, default=None))
    for row in rows:
        row["relative_distance"] = None if row["depth"] is None or not first else row["depth"] / first
    return {
        "total": len(names),
        "registered": len(rows),
        "unregistered": [name for name in names if name not in cameras],
        "models": models,
        "points": int(len(xyz)),
        "mean_reprojection_error_px": float(errors.mean()) if len(errors) else None,
        "first_picture_placed": rows[0]["name"] == names[0] if rows else False,
        "images": rows,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fov", type=float, default=55.0, help="the lens's horizontal field of view, degrees (the app passes its own)")
    parser.add_argument("--refine-focal", action="store_true",
                        help="solve the focal length too - for a dome of views at several distances (a flat sweep cannot)")
    parser.add_argument("--mask-board", action="store_true",
                        help="leave the ChArUco board's paper out of the pose solve and the training - for a scan whose "
                             "board slid under the cloth while it was taken")
    args = parser.parse_args()
    session, output = Path(args.session), Path(args.output)

    # A session's pictures are what its sidecars name - a hand-held one's and a robot run's alike -
    # which leaves out the masks and annotated copies kept beside them. A folder holding combined.json
    # stands for the sessions it lists, whose pictures are solved together under their session's name.
    combined = session / "combined.json"
    settings = json.loads(combined.read_text()) if combined.is_file() else {}
    folders = [paths.scan_session(folder) for folder in settings["sessions"]] if settings else [session]
    # A combined solve may say which pictures to leave out: video frames (no pose, often blurred)
    # and anything below a sharpness, so that several sessions' worth stays matchable in an hour.
    skip_video = bool(settings.get("skip_video", False))
    min_sharpness = float(settings.get("min_sharpness", 0.0))
    sources = {}
    for folder in folders:
        for sidecar in folder.glob("*.json"):
            record = json.loads(sidecar.read_text())
            picture = Path(record.get("path") or "").name if isinstance(record, dict) else ""
            if not picture or not (folder / picture).is_file():
                continue
            if skip_video and record.get("camera_position") is None:
                continue
            if min_sharpness and record.get("sharpness") is not None and float(record["sharpness"]) < min_sharpness:
                continue
            sources[f"{folder.name}__{picture}" if len(folders) > 1 else picture] = folder / picture
    names = sorted(sources)
    if len(names) < 3:
        raise SystemExit(f"only {len(names)} pictures in {session}; the pose solve needs many more")
    # Only what an earlier solve wrote is cleared: the app's log of this very run (train.log) is open
    # in the same folder.
    for stale in ("images", "sparse", "colmap", "database.db", "transforms.json", "sparse_pc.ply", "pose_report.json",
                  "masks", "colmap_masks"):
        path = output / stale
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    images = output / "images"
    images.mkdir(parents=True)
    for name in names:
        shutil.copy2(sources[name], images / name)
    if settings.get("equalise_exposure", True):
        equalise_exposure(images)
    masked = mask_board(images, output) if args.mask_board else {}

    import pycolmap
    from PIL import Image

    with Image.open(images / names[0]) as probe:
        width, height = probe.size
    focal = width / (2.0 * math.tan(math.radians(args.fov) / 2.0))
    database = output / "database.db"
    print(f"1/4 Finding features in {len(names)} pictures (lens {args.fov:.0f} deg, focal {focal:.0f} px)", flush=True)
    reader = pycolmap.ImageReaderOptions(camera_model="SIMPLE_RADIAL", camera_params=f"{focal},{width / 2.0},{height / 2.0},0")
    if masked:
        reader.mask_path = str(output / "colmap_masks")
    pycolmap.extract_features(str(database), str(images), camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader)
    print(f"2/4 Matching every pair ({len(names) * (len(names) - 1) // 2} pairs)", flush=True)
    pycolmap.match_exhaustive(str(database))
    print("3/4 Placing the cameras (incremental mapping)", flush=True)
    options = pycolmap.IncrementalPipelineOptions()
    # Pictures from several heights let the lens be solved too; a single height does not.
    options.ba_refine_focal_length = bool(settings.get("refine_focal", False) or args.refine_focal)
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = True
    options.mapper.init_min_tri_angle = INIT_MIN_TRI_ANGLE_DEG
    options.mapper.filter_min_tri_angle = MIN_TRI_ANGLE_DEG
    options.mapper.abs_pose_min_num_inliers = MIN_POSE_INLIERS
    options.triangulation.min_angle = MIN_TRI_ANGLE_DEG
    models = pycolmap.incremental_mapping(str(database), str(images), str(output / "sparse"), options)
    if not models:
        raise SystemExit("no pictures could be placed: they share too few matching features")
    largest = max(models.values(), key=lambda model: sum(bool(image.has_pose) for image in model.images.values()))
    recon_dir = output / "colmap"
    recon_dir.mkdir()
    largest.write(str(recon_dir))

    print("4/4 Writing transforms.json and measuring the capture", flush=True)
    from nerfstudio.process_data.colmap_utils import colmap_to_json

    colmap_to_json(recon_dir=recon_dir, output_dir=output)
    strays = drop_stray_cameras(output / "transforms.json")
    if masked:
        # nerfstudio leaves a frame's masked pixels out of the training loss.
        transforms = json.loads((output / "transforms.json").read_text())
        for frame in transforms["frames"]:
            frame["mask_path"] = f"masks/{Path(frame['file_path']).name}"
        write_json(output / "transforms.json", transforms)
    report = measure(recon_dir, names, len(models))
    report["stray_cameras"] = strays
    if masked:
        report["board_masked_share"] = masked
    report["registered"] -= len(strays)
    report["unregistered"] += strays
    write_json(output / "pose_report.json", report)
    print(f"Placed {report['registered']} of {report['total']} pictures in {len(models)} model(s), "
          f"mean reprojection error {report['mean_reprojection_error_px']:.2f} px", flush=True)


if __name__ == "__main__":
    main()
