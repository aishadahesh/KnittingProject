"""The measured work cell (config/workcell.json): table, carton and fabric in the UR5e controller base frame, metres.

Every surface shares one slope - the table is tilted against the robot (~2 deg along y, ~0.6 along x) -
so a surface is its height at the base origin plus that slope: z = sx * x + sy * y + top_z_at_origin.

The carton, the fabric and its parts are stored where they lay when the carton was photographed. The
carton pose (`carton_pose`: a turn about the base's z axis, then a shift) says where they lie now, so a
nudged carton is re-located from a few pictures instead of being mapped again.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from paths import FABRIC_PARTS_JSON, WORKCELL_CARTON_JPG, WORKCELL_JSON
from persistence import write_json


@lru_cache(maxsize=1)
def cell() -> dict:
    return json.loads(WORKCELL_JSON.read_text(encoding="utf-8"))


def carton_pose() -> tuple[np.ndarray, np.ndarray]:
    """(2x2 turn, 2-vector shift): a photographed point (x, y) lies now at turn @ (x, y) + shift."""
    pose = cell().get("carton_pose", {"turn_deg": 0.0, "shift": [0.0, 0.0]})
    angle = np.radians(float(pose["turn_deg"]))
    turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return turn, np.asarray(pose["shift"], dtype=float)


def place(points_xy) -> np.ndarray:
    """Photographed (x, y) points (Nx2) where they lie now."""
    turn, shift = carton_pose()
    return np.asarray(points_xy, dtype=float).reshape(-1, 2) @ turn.T + shift


def save_carton_pose(turn_deg: float, shift_xy) -> None:
    """Record where the carton lies now; everything placed on it follows."""
    data = dict(cell(), carton_pose={"turn_deg": float(turn_deg), "shift": [float(v) for v in shift_xy]})
    write_json(WORKCELL_JSON, data)
    cell.cache_clear()
    _photographed_parts.cache_clear()


def reference_photo(res_m: float):
    """The carton's photo (BGR) at `res_m` per pixel, where it was photographed: (image, x0, y_top)."""
    import cv2

    carton = cell()["carton"]
    photo = cv2.imread(str(WORKCELL_CARTON_JPG))
    scale = (carton["x"][1] - carton["x"][0]) / photo.shape[1] / res_m
    return cv2.resize(photo, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), carton["x"][0], carton["y"][1]


def surface_z(top_z_at_origin: float, x: float, y: float) -> float:
    sx, sy = cell()["surface_slope"]
    return float(sx * x + sy * y + top_z_at_origin)


@lru_cache(maxsize=1)
def _photographed_parts() -> tuple[dict, ...]:
    if not FABRIC_PARTS_JSON.exists():
        return ()
    return tuple(json.loads(FABRIC_PARTS_JSON.read_text(encoding="utf-8"))["parts"])


def parts() -> list[dict]:
    """The fabric's numbered parts (scripts/map_fabric_parts.py) where they lie now: number, row, the
    rectangle's 4 corners (x, y) and its centre (x, y, z on the carton's top). Empty until mapped."""
    top = cell()["carton"]["top_z_at_origin"]
    placed = []
    for part in _photographed_parts():
        (x0, x1), (y0, y1) = part["x"], part["y"]
        corners = place([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        x, y = corners.mean(axis=0)
        placed.append({"number": part["number"], "row": part["row"], "corners": corners.tolist(),
                       "centre": [float(x), float(y), surface_z(top, x, y)]})
    return placed


def fabric_center() -> tuple[float, float, float]:
    """The fabric's middle, on the carton's top."""
    fabric = cell()["fabric"]
    x, y = place([(sum(fabric["x"]) / 2.0, sum(fabric["y"]) / 2.0)])[0]
    return (float(x), float(y), surface_z(cell()["carton"]["top_z_at_origin"], x, y))


def fabric_corners() -> np.ndarray:
    """The fabric's 4 corners (x, y) where it lies now."""
    (x0, x1), (y0, y1) = cell()["fabric"]["x"], cell()["fabric"]["y"]
    return place([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def part_centre(number) -> list[float] | None:
    """Where fabric part `number` lies now (x, y, z), or None when there is no such part."""
    return next((part["centre"] for part in parts() if part["number"] == number), None)


def fabric_size() -> tuple[float, float]:
    fabric = cell()["fabric"]
    return (fabric["x"][1] - fabric["x"][0], fabric["y"][1] - fabric["y"][0])
