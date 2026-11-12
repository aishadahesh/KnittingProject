"""Replay or inspect a recorded scan on the UR5e URDF kinematics: the arm at each picture's pose, the picture beside it.

    uv run python scripts\\replay_session.py ur5_20261101_155954     (a scan folder under robot_scans)
    uv run python scripts\\replay_session.py --list                  (the scans there, with their picture counts)
    uv run python scripts\\replay_session.py <scan> --snapshot out.png  (one rendered frame, no window)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402
import scan_geometry  # noqa: E402
import session_replay  # noqa: E402

PHOTO_WIDTH = 360


def fabric_patch(views):
    heights = [v.fabric_z for v in views if v.fabric_z is not None]
    fabric_z = float(np.median(heights)) if heights else float(min(v.camera[2] for v in views)) - 0.05
    corners = [v.footprint for v in views if v.footprint is not None] or [np.array([v.subject[:2]]) for v in views if v.subject is not None]
    points = np.vstack(corners) if corners else np.array([v.camera[:2] for v in views])
    low, high = points.min(axis=0) - 0.01, points.max(axis=0) + 0.01
    return fabric_z, low, high


def photo(view, width=PHOTO_WIDTH):
    try:
        with Image.open(view.image) as image:
            image = image.convert("RGB")
            height = max(1, round(image.height * width / image.width))
            return np.asarray(image.resize((width, height)), dtype=np.uint8)
    except OSError:
        return np.full((width * 3 // 4, width, 3), 40, dtype=np.uint8)


def caption(views, index, solved):
    view, (_, why) = views[index], solved[index]
    record = view.record
    numbers = [f"{index + 1} / {len(views)}   {view.session}", view.label, view.image.name]
    if record.get("camera_height_above_fabric_m") is not None:
        numbers.append(f"{float(record['camera_height_above_fabric_m']) * 1000:.0f} mm above the fabric, "
                       f"tilted {float(record.get('tilt_from_vertical_deg') or 0):.0f} deg")
    if record.get("sharpness") is not None:
        numbers.append(f"sharpness {float(record['sharpness']):.0f}")
    if why:
        numbers.append("NOT REACHED by the IK - the arm stays at the pose before")
    return "\n".join(numbers)


def snapshot(views, solved, unreached, patch, index, out: Path):
    width, height = 1080, 720
    canvas = Image.new("RGB", (width, height), (20, 23, 28))
    draw = ImageDraw.Draw(canvas)

    joints = solved[index][0]
    transforms = scan_geometry.link_world_transforms(joints)
    link_names = ["base", "shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"]
    pts_3d = [transforms[name][:3, 3] for name in link_names if name in transforms]

    fabric_z, low, high = patch
    centre = np.array([(low[0] + high[0]) * 0.5, (low[1] + high[1]) * 0.5, fabric_z + 0.25], dtype=float)

    def project(p3):
        rel = np.asarray(p3, dtype=float) - centre
        sx = int(width * 0.36 + rel[0] * 520.0 - rel[1] * 260.0)
        sy = int(height * 0.56 - rel[2] * 520.0 - rel[1] * 140.0)
        return sx, sy

    quad = [
        project([low[0], low[1], fabric_z]),
        project([high[0], low[1], fabric_z]),
        project([high[0], high[1], fabric_z]),
        project([low[0], high[1], fabric_z]),
    ]
    draw.polygon(quad, fill=(85, 78, 62), outline=(140, 130, 105))

    for i, view in enumerate(views):
        px, py = project(view.camera)
        r = 5 if i == index else 3
        color = (240, 55, 55) if i in unreached else ((55, 220, 90) if i == index else (150, 155, 165))
        draw.ellipse((px - r, py - r, px + r, py + r), fill=color)

    if len(pts_3d) >= 2:
        proj_links = [project(p) for p in pts_3d]
        draw.line(proj_links, fill=(195, 205, 220), width=5)
        for px, py in proj_links:
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(90, 150, 230))

    draw.text((20, 20), caption(views, index, solved), fill=(235, 240, 248))

    image = np.asarray(canvas, dtype=np.uint8).copy()
    picture = photo(views[index])
    image[:picture.shape[0], -picture.shape[1]:] = picture
    Image.fromarray(image).save(out)
    print(f"Saved {out}")


def replay(views, solved, unreached, patch, start=0):
    for idx in range(start, len(views)):
        print(caption(views, idx, solved))
        print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("session", nargs="?", help="a scan folder's name under robot_scans")
    parser.add_argument("--list", action="store_true", help="list the scans under robot_scans")
    parser.add_argument("--tool-offset", type=float, default=None,
                        help=f"the camera point's distance along the flange (default: config/params.json, else {session_replay.scan_geometry.ENDOSCOPE_TIP_OFFSET})")
    parser.add_argument("--snapshot", type=Path, help="render one frame to this PNG instead of opening the viewer")
    parser.add_argument("--picture", type=int, default=1, help="the picture to start on (1 is the first)")
    args = parser.parse_args()

    scans = paths.ROBOT_SCANS_DIR
    if args.list or not args.session:
        for folder in sorted(p for p in scans.iterdir() if p.is_dir()) if scans.is_dir() else []:
            print(f"  {folder.name:<28} {len(session_replay.load_views(folder)):>4} posed pictures")
        if not args.session:
            return
    session = paths.scan_session(args.session)
    if not session.is_dir():
        raise SystemExit(f"no scan folder {session}")
    views = session_replay.load_views(session)
    if not views:
        raise SystemExit(f"{session.name} has no pictures with an arm pose")

    offset = session_replay.tool_offset() if args.tool_offset is None else args.tool_offset
    print(f"Solving the arm's joints for {len(views)} pictures (tool offset {offset * 1000:.1f} mm)...", flush=True)
    solved = session_replay.solve_joints(views, offset)
    unreached = {i for i, (_, why) in enumerate(solved) if why}
    if unreached:
        print(f"  {len(unreached)} pose(s) not reached: pictures {', '.join(str(i + 1) for i in sorted(unreached))}")
    patch = fabric_patch(views)

    start = min(len(views), max(1, args.picture)) - 1
    if args.snapshot:
        snapshot(views, solved, unreached, patch, start, args.snapshot)
    else:
        replay(views, solved, unreached, patch, start)


if __name__ == "__main__":
    main()
