"""The measured work cell (config/workcell.json): table, carton and fabric in the UR5e controller base frame, metres.

Every surface shares one slope - the table is tilted against the robot (~2 deg along y, ~0.6 along x) -
so a surface is its height at the base origin plus that slope: z = sx * x + sy * y + top_z_at_origin.
"""

from __future__ import annotations

import json
from functools import lru_cache

from paths import FABRIC_PARTS_JSON, WORKCELL_JSON


@lru_cache(maxsize=1)
def cell() -> dict:
    return json.loads(WORKCELL_JSON.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def parts() -> list[dict]:
    """The fabric's numbered parts (scripts/map_fabric_parts.py); empty until they have been mapped."""
    if not FABRIC_PARTS_JSON.exists():
        return []
    return json.loads(FABRIC_PARTS_JSON.read_text(encoding="utf-8"))["parts"]


def surface_z(top_z_at_origin: float, x: float, y: float) -> float:
    sx, sy = cell()["surface_slope"]
    return float(sx * x + sy * y + top_z_at_origin)


def fabric_center() -> tuple[float, float, float]:
    """The fabric's middle, on the carton's top."""
    fabric = cell()["fabric"]
    x, y = sum(fabric["x"]) / 2.0, sum(fabric["y"]) / 2.0
    return (x, y, surface_z(cell()["carton"]["top_z_at_origin"], x, y))


def fabric_size() -> tuple[float, float]:
    fabric = cell()["fabric"]
    return (fabric["x"][1] - fabric["x"][0], fabric["y"][1] - fabric["y"][0])
