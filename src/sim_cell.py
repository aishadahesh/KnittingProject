"""TeleopApp's simulated cell: a table, and the knitted fabric on it, set into TeleopApp's 3D scene.

With TeleopApp on its emulator, the whole capture pipeline - Teach & calibrate, Check, Rehearse,
Run, the frame, sharpness and fabric checks, saving - runs against its virtual UR5e, and the
pictures come from its gripper camera (see teleop_transport.TeleopCameraBackend). What that
camera needs to see is built here, as TeleopApp props:

- the fabric: a flat mesh whose texture is the knitted layout Scan Mode renders, written as an
  OBJ + MTL + PNG, because a mesh prop with no colour is the one TeleopApp draws textured;
- a table under it, reaching to the floor, so the scene reads as the real cell does.

Placement is in this app's plan frame (the MuJoCo UR5e's, which is the controller's base frame),
under the camera at Home, and converted into TeleopApp's Y-up world. That conversion is derived
from the two model loaders, not assumed: `frame_map` measures TeleopApp's reported tool point
against this app's forward kinematics and picks the mapping that agrees, refusing when none
does - a table placed in the wrong frame would put the fabric where no plan expects it.

The fabric's corners are handed back in the plan frame, so the app can adopt them as the taught
fabric outline: every automatic frame is then kept on the simulated cloth, exactly as it would be
on a real one whose corners were marked by hand.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

import paths
import scan_geometry as geo
import ur5_robot

# Candidate maps from the plan frame into TeleopApp's world. TeleopApp loads the URDF with its base
# turned -90 degrees about X to make Y up; whether that URDF's base link is the controller's base
# or the ROS base_link (turned 180 degrees about Z) decides which of these it is.
FRAME_CANDIDATES = {
    "base_link": np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
    "base":      np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
}
FRAME_TOLERANCE = 0.01

# The cell, in metres. The fabric lies FABRIC_DROP below the camera at Home, which is where Home
# is taught to look down from; the table reaches past it by TABLE_MARGIN on every side.
FABRIC_WIDTH = 0.40
FABRIC_DROP = 0.25
TABLE_MARGIN = 0.15
TABLE_THICKNESS = 0.03
LEG_SIDE = 0.04
# The fabric mesh floats this far above the table top, so the two never z-fight.
FABRIC_LIFT = 0.001
TEXTURE_MAX_SIDE = 2048

TABLE_COLOR = [0.45, 0.33, 0.22, 1.0]
LEG_COLOR = [0.30, 0.30, 0.32, 1.0]
PROP_PREFIX = "simcell "


def frame_map(transport) -> np.ndarray:
    """The plan-to-TeleopApp rotation, chosen by where TeleopApp says the tool point is right now."""
    ours = geo.camera_position(transport.measured_joints(), transport.tool_offset)
    theirs = np.asarray(transport.app_tool_point(), dtype=float)
    misses = {name: float(np.linalg.norm(rotation @ ours - theirs)) for name, rotation in FRAME_CANDIDATES.items()}
    name = min(misses, key=misses.get)
    if misses[name] > FRAME_TOLERANCE:
        raise RuntimeError("TeleopApp's world does not match this app's frame under any known mapping - its tool point is "
                           + ", ".join(f"{miss * 1000:.0f} mm off as `{candidate}`" for candidate, miss in misses.items())
                           + ". Check that its ur5 basePose is zero and its tool is the camera, then try again")
    return FRAME_CANDIDATES[name]


def fabric_layout(tool_offset: float, texture_size) -> dict:
    """Where the fabric and table go in the plan frame: under Home's camera, sized to the texture."""
    home = geo.camera_position(ur5_robot.DEFAULT_HOME_JOINTS, tool_offset)
    width = FABRIC_WIDTH
    depth = FABRIC_WIDTH * float(texture_size[1]) / float(texture_size[0])
    centre = np.array([home[0], home[1], home[2] - FABRIC_DROP])
    half = np.array([width / 2.0, depth / 2.0])
    corners = [np.r_[centre[:2] + sign * half, centre[2]] for sign in (np.array([-1, -1]), np.array([1, -1]), np.array([1, 1]), np.array([-1, 1]))]
    return {"centre": centre, "width": width, "depth": depth, "corners": corners}


def write_fabric_mesh(texture: Image.Image, width: float, depth: float, folder: Path) -> Path:
    """A flat, textured quad in TeleopApp's frame: Y up, the cloth spanning X and Z.

    A fresh folder per build: TeleopApp caches a model by its path, so rewriting one in place
    would keep showing the old knit.
    """
    folder.mkdir(parents=True, exist_ok=True)
    image = texture.convert("RGB")
    if max(image.size) > TEXTURE_MAX_SIDE:
        scale = TEXTURE_MAX_SIDE / float(max(image.size))
        image = image.resize((max(1, int(image.size[0] * scale)), max(1, int(image.size[1] * scale))), Image.LANCZOS)
    image.save(folder / "fabric.png")

    (folder / "fabric.mtl").write_text("newmtl knit\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nmap_Kd fabric.png\n")
    x, z = width / 2.0, depth / 2.0
    # Counter-clockwise seen from above (+Y), so the face points up at the camera.
    (folder / "fabric.obj").write_text(
        "mtllib fabric.mtl\n"
        f"v {-x} 0 {-z}\nv {-x} 0 {z}\nv {x} 0 {z}\nv {x} 0 {-z}\n"
        "vt 0 1\nvt 0 0\nvt 1 0\nvt 1 1\n"
        "vn 0 1 0\n"
        "usemtl knit\n"
        "f 1/1/1 2/2/1 3/3/1 4/4/1\n")
    return folder / "fabric.obj"


def props_for(layout: dict, rotation: np.ndarray, mesh: Path) -> dict:
    """The props that make the cell, by name, as setProp specs in TeleopApp's world."""
    def at(point):
        return [float(value) for value in rotation @ np.asarray(point, dtype=float)]

    centre = layout["centre"]
    table_w, table_d = layout["width"] + 2.0 * TABLE_MARGIN, layout["depth"] + 2.0 * TABLE_MARGIN
    top = centre[2] - FABRIC_LIFT
    leg_height = max(top - TABLE_THICKNESS, 0.01)

    props = {
        f"{PROP_PREFIX}table": {"shape": "box", "position": at(centre - [0.0, 0.0, FABRIC_LIFT + TABLE_THICKNESS / 2.0]),
                                "size": [table_w, TABLE_THICKNESS, table_d], "color": TABLE_COLOR},
        f"{PROP_PREFIX}fabric": {"shape": "mesh", "file": str(mesh), "position": at(centre), "size": [1.0, 1.0, 1.0]},
    }
    for index, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
        foot = centre + [sx * (table_w / 2.0 - LEG_SIDE), sy * (table_d / 2.0 - LEG_SIDE), 0.0]
        foot[2] = leg_height / 2.0
        props[f"{PROP_PREFIX}leg {index + 1}"] = {"shape": "box", "position": at(foot),
                                                   "size": [LEG_SIDE, leg_height, LEG_SIDE], "color": LEG_COLOR}
    return props


def build(transport, texture: Image.Image) -> list[list[float]]:
    """Set the table and the knitted fabric into TeleopApp's scene; return the fabric's corners (plan frame).

    Replaces a cell built before, and leaves every other prop alone.
    """
    rotation = frame_map(transport)
    layout = fabric_layout(transport.tool_offset, texture.size)
    mesh = write_fabric_mesh(texture, layout["width"], layout["depth"], paths.SIM_CELL_DIR / time.strftime("cell_%Y%m%d_%H%M%S"))

    for name in [name for name in (transport.client.call("getProps") or {}) if name.startswith(PROP_PREFIX)]:
        transport.client.call("removeProp", name)
    for name, spec in props_for(layout, rotation, mesh).items():
        transport.client.call("setProp", name, spec)
    return [[float(value) for value in corner] for corner in layout["corners"]]


def remove(transport) -> int:
    names = [name for name in (transport.client.call("getProps") or {}) if name.startswith(PROP_PREFIX)]
    for name in names:
        transport.client.call("removeProp", name)
    return len(names)
