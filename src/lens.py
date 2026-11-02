"""The measured lens (scripts/calibrate_lens.py writes it), for whoever places pixels in space."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from paths import CONFIG_DIR

INTRINSICS_JSON = CONFIG_DIR / "camera_intrinsics.json"


def load_intrinsics(path: Path = INTRINSICS_JSON) -> dict | None:
    """The measured lens, or None when it has not been measured."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and "fx" in data else None


def intrinsics_for(width: int, height: int, measured: dict) -> dict:
    """The measured lens as nerfstudio's transforms.json keys, scaled to this picture size."""
    sx, sy = width / float(measured["width"]), height / float(measured["height"])
    fx = float(measured["fx"]) * sx
    return {"fl_x": fx, "fl_y": float(measured["fy"]) * sy,
            "cx": float(measured["cx"]) * sx, "cy": float(measured["cy"]) * sy, "w": int(width), "h": int(height),
            "k1": float(measured.get("k1", 0.0)), "k2": float(measured.get("k2", 0.0)), "k3": float(measured.get("k3", 0.0)),
            "p1": float(measured.get("p1", 0.0)), "p2": float(measured.get("p2", 0.0)), "camera_model": "OPENCV",
            "camera_angle_x": 2.0 * float(np.arctan(width / (2.0 * fx)))}


def project(points_cam: np.ndarray, intrinsics: dict) -> np.ndarray:
    """Camera-frame points (x right, y down, z forward) to pixels, through the lens's distortion.
    `intrinsics` holds fl_x, fl_y, cx, cy and optionally k1, k2, k3, p1, p2 (OpenCV's model)."""
    z = np.maximum(points_cam[:, 2], 1e-6)
    x, y = points_cam[:, 0] / z, points_cam[:, 1] / z
    k1, k2, k3 = (float(intrinsics.get(k, 0.0)) for k in ("k1", "k2", "k3"))
    p1, p2 = (float(intrinsics.get(k, 0.0)) for k in ("p1", "p2"))
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.stack([float(intrinsics["fl_x"]) * xd + float(intrinsics["cx"]), float(intrinsics["fl_y"]) * yd + float(intrinsics["cy"])], axis=1)
