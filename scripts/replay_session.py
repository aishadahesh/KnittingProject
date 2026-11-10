"""Replay a recorded scan on the MuJoCo UR5e: the arm at each picture's pose, the picture beside it.

    uv run python scripts\\replay_session.py ur5_20261101_155954     (a scan folder under robot_scans)
    uv run python scripts\\replay_session.py --list                  (the scans there, with their picture counts)
    uv run python scripts\\replay_session.py <scan> --snapshot out.png  (one rendered frame, no window)

Keys: Page Down / Page Up the next / previous picture, Home / End the first / last, Enter play or pause.
Mouse (MuJoCo's viewer): drag to orbit, right-drag to pan, scroll to zoom.

Every camera point of the scan is a grey dot, the current one green with its picture's footprint on
the fabric and the line to the point it was aimed at; a red dot is a pose the IK could not reach
(the arm stays where it was). The camera point is the planned one, not a calibrated lens pose: see
session_replay.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402
import session_replay  # noqa: E402

# GLFW key codes. Arrows, brackets and most letters are taken by MuJoCo's own viewer.
KEY_ENTER, KEY_PAGE_UP, KEY_PAGE_DOWN, KEY_HOME, KEY_END = 257, 266, 267, 268, 269
PHOTO_WIDTH = 360            # the least; on a large window the picture takes a quarter of its width
MOVE_SECONDS = 0.6           # the arm's glide from one picture's pose to the next
PLAY_SECONDS = 1.2           # time on each picture while playing

GREY, GREEN, RED, YELLOW = (0.6, 0.6, 0.65, 0.9), (0.2, 0.85, 0.35, 1.0), (0.95, 0.2, 0.2, 1.0), (0.95, 0.75, 0.2, 1.0)
FABRIC = (0.85, 0.78, 0.62, 0.55)


def _geom(mj, scn, kind, size, pos, mat, rgba):
    if scn.ngeom >= scn.maxgeom:
        return None
    geom = scn.geoms[scn.ngeom]
    mj.mjv_initGeom(geom, kind, np.asarray(size, dtype=float), np.asarray(pos, dtype=float),
                    np.asarray(mat, dtype=float).flatten(), np.asarray(rgba, dtype=np.float32))
    scn.ngeom += 1
    return geom


def _dot(mj, scn, at, radius, rgba):
    _geom(mj, scn, mj.mjtGeom.mjGEOM_SPHERE, (radius, 0, 0), at, np.eye(3), rgba)


def _line(mj, scn, a, b, rgba, width=0.0015):
    geom = _geom(mj, scn, mj.mjtGeom.mjGEOM_CAPSULE, (0, 0, 0), (0, 0, 0), np.eye(3), rgba)
    if geom is not None:
        mj.mjv_connector(geom, mj.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, dtype=float), np.asarray(b, dtype=float))


def fabric_patch(views):
    """The fabric's height and the box round every picture's footprint (or aim point) on it."""
    heights = [v.fabric_z for v in views if v.fabric_z is not None]
    fabric_z = float(np.median(heights)) if heights else float(min(v.camera[2] for v in views)) - 0.05
    corners = [v.footprint for v in views if v.footprint is not None] or [np.array([v.subject[:2]]) for v in views if v.subject is not None]
    points = np.vstack(corners) if corners else np.array([v.camera[:2] for v in views])
    low, high = points.min(axis=0) - 0.01, points.max(axis=0) + 0.01
    return fabric_z, low, high


def draw_scene(mj, scn, views, index, unreached, patch, start=0):
    """Draw from geom `start` on: 0 for the viewer's own scene, the model's count for a renderer's."""
    scn.ngeom = start
    fabric_z, low, high = patch
    centre, half = (low + high) / 2, (high - low) / 2
    _geom(mj, scn, mj.mjtGeom.mjGEOM_BOX, (half[0], half[1], 0.0005), (centre[0], centre[1], fabric_z - 0.0005), np.eye(3), FABRIC)
    for i, view in enumerate(views):
        if i != index:
            _dot(mj, scn, view.camera, 0.0025, RED if i in unreached else GREY)
    view = views[index]
    _dot(mj, scn, view.camera, 0.005, RED if index in unreached else GREEN)
    if view.footprint is not None:
        quad = [np.r_[corner, fabric_z + 0.0005] for corner in view.footprint]
        for a, b in zip(quad, quad[1:] + quad[:1]):
            _line(mj, scn, a, b, GREEN, 0.001)
        for corner in quad:
            _line(mj, scn, view.camera, corner, (0.2, 0.85, 0.35, 0.45), 0.0007)
    if view.subject is not None:
        _line(mj, scn, view.camera, view.subject, YELLOW, 0.001)
        _dot(mj, scn, view.subject, 0.002, YELLOW)


def frame_camera(cam, patch):
    """Look at the fabric from the side, far enough back to see the whole arm."""
    fabric_z, low, high = patch
    centre = (low + high) / 2
    cam.lookat[:] = (centre[0] * 0.6, centre[1] * 0.6, fabric_z + 0.25)
    cam.distance, cam.azimuth, cam.elevation = 1.4, 210.0, -25.0


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
    """One frame rendered off screen: the check that the scene is where it should be, without a window."""
    import mujoco as mj
    import scan_geometry

    _, model, data, _ = scan_geometry.load_model()
    data.qpos[:6] = solved[index][0]
    mj.mj_forward(model, data)
    camera = mj.MjvCamera()
    frame_camera(camera, patch)
    model.vis.global_.offwidth, model.vis.global_.offheight = max(model.vis.global_.offwidth, 1080), max(model.vis.global_.offheight, 720)
    with mj.Renderer(model, height=720, width=1080) as renderer:
        renderer.update_scene(data, camera)
        draw_scene(mj, renderer.scene, views, index, unreached, patch, start=renderer.scene.ngeom)
        image = renderer.render()
    picture = photo(views[index])
    image[:picture.shape[0], -picture.shape[1]:] = picture
    Image.fromarray(image).save(out)
    print(f"Saved {out}")


def replay(views, solved, unreached, patch, start=0):
    import mujoco as mj
    import mujoco.viewer
    import scan_geometry

    _, model, data, _ = scan_geometry.load_model()
    state = {"index": start, "go": None, "playing": False, "since": time.monotonic()}

    def on_key(key):
        last = len(views) - 1
        moves = {KEY_PAGE_DOWN: state["index"] + 1, KEY_PAGE_UP: state["index"] - 1, KEY_HOME: 0, KEY_END: last}
        if key in moves:
            state["go"], state["playing"] = min(last, max(0, moves[key])), False
        elif key == KEY_ENTER:
            state["playing"] = not state["playing"]
            if state["playing"] and state["index"] == last:
                state["go"] = 0
            state["since"] = time.monotonic()

    start_q = solved[start][0]
    move = {"from": start_q, "to": start_q, "at": 0.0}
    shown = None
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key, show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            frame_camera(viewer.cam, patch)
        while viewer.is_running():
            now = time.monotonic()
            if state["playing"] and now - state["since"] >= PLAY_SECONDS:
                state["since"] = now
                if state["index"] < len(views) - 1:
                    state["go"] = state["index"] + 1
                else:
                    state["playing"] = False
            if state["go"] is not None and state["go"] != state["index"]:
                t = min(1.0, (now - move["at"]) / MOVE_SECONDS)
                current = move["from"] + (move["to"] - move["from"]) * (3 * t * t - 2 * t ** 3)
                move = {"from": current, "to": solved[state["go"]][0], "at": now}
                state["index"] = state["go"]
            state["go"] = None

            t = min(1.0, (now - move["at"]) / MOVE_SECONDS)
            with viewer.lock():
                data.qpos[:6] = move["from"] + (move["to"] - move["from"]) * (3 * t * t - 2 * t ** 3)
                mj.mj_forward(model, data)
                draw_scene(mj, viewer.user_scn, views, state["index"], unreached, patch)

            viewport = viewer.viewport
            key = (state["index"], state["playing"], None if viewport is None else (viewport.width, viewport.height))
            if key != shown and viewport is not None and viewport.width > PHOTO_WIDTH + 20:
                shown = key
                picture = photo(views[state["index"]], max(PHOTO_WIDTH, viewport.width // 4))
                h, w = picture.shape[:2]
                rect = mj.MjrRect(viewport.width - w - 10, viewport.height - h - 10, w, h)
                viewer.set_images((rect, picture))
                help_text = "PgDn / PgUp  next / previous     Home / End  first / last     Enter  " + ("pause" if state["playing"] else "play")
                viewer.set_texts([(None, mj.mjtGridPos.mjGRID_TOPLEFT, caption(views, state["index"], solved), None),
                                  (mj.mjtFontScale.mjFONTSCALE_100, mj.mjtGridPos.mjGRID_BOTTOMLEFT, help_text, None)])
            viewer.sync()
            time.sleep(1 / 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("session", nargs="?", help="a scan folder's name under robot_scans")
    parser.add_argument("--list", action="store_true", help="list the scans under robot_scans")
    parser.add_argument("--tool-offset", type=float, default=None,
                        help=f"the camera point's distance along the flange (default: config/params.json, else {session_replay.DEFAULT_TOOL_OFFSET})")
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
