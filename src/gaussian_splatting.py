"""Optional Gaussian Splatting reconstruction from a real UR5 scan.

A scan already produces the two things a splatting trainer needs: images of the
fabric from many angles, and -- because a robot took them -- a known camera pose
for every one. That second part is the useful bit. Normal photogrammetry has to
recover poses with structure-from-motion, which on a flat, repetitively textured
knit is exactly the case SfM handles worst. Robot forward kinematics sidesteps
it entirely.

So this module does not reconstruct anything itself. It converts a scan into the
dataset formats trainers expect (``transforms.json`` and a COLMAP text model),
and can then hand that dataset to whichever trainer the user has installed. The
heavy CUDA training stays external and optional; preparing the dataset does not
require it.

Two things must be right for the exported poses to be worth anything, and both
are surfaced rather than hidden:

  * The hand-eye transform -- where the camera sits relative to the tool flange.
    The default is a plain forward-looking offset, which is a guess. An
    uncalibrated guess yields a misaligned reconstruction, so it is a settable
    parameter and the export records what was used.
  * The intrinsics. Without a calibration file these are derived from an assumed
    field of view, which is likewise approximate.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


# Camera sits ahead of the tool flange looking along tool +Z. A placeholder for
# a real hand-eye calibration, not a substitute for one.
DEFAULT_CAMERA_OFFSET_M = 0.12
DEFAULT_FOV_DEGREES = 55.0


@dataclass
class SplatDataset:
    """Where an exported dataset landed and what went into it."""

    root: Path
    image_count: int
    transforms_path: Path
    colmap_dir: Path
    session_id: str = ""
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "image_count": int(self.image_count),
            "transforms_path": str(self.transforms_path),
            "colmap_dir": str(self.colmap_dir),
            "session_id": str(self.session_id),
            "warnings": list(self.warnings),
        }


def hand_eye_matrix(offset_m: float = DEFAULT_CAMERA_OFFSET_M, rotation_rotvec=None) -> np.ndarray:
    """Tool-flange to camera transform.

    Defaults to a pure translation along the tool's +Z, which is where a
    gripper-mounted camera looking at the work usually sits.
    """
    matrix = np.eye(4, dtype=float)
    if rotation_rotvec is not None:
        matrix[:3, :3] = Rotation.from_rotvec(np.asarray(rotation_rotvec, dtype=float)).as_matrix()
    matrix[:3, 3] = [0.0, 0.0, float(offset_m)]
    return matrix


def pose_to_matrix(pose) -> np.ndarray:
    """UR pose ``[x, y, z, rx, ry, rz]`` to a 4x4 base-to-tool transform.

    The rotation part is an axis-angle (rotation vector), which is how UR
    controllers report and accept orientation.
    """
    pose = np.asarray(pose, dtype=float).reshape(-1)[:6]
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = Rotation.from_rotvec(pose[3:6]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def camera_to_world(pose, hand_eye: np.ndarray | None = None) -> np.ndarray:
    """Camera pose in the robot base frame, in OpenCV axes (x right, y down, z forward)."""
    hand_eye = hand_eye_matrix() if hand_eye is None else np.asarray(hand_eye, dtype=float)
    return pose_to_matrix(pose) @ hand_eye


# OpenCV camera axes to OpenGL/NeRF axes: flip Y and Z. transforms.json is a
# NeRF-convention format, so skipping this flips the reconstruction upside down
# and back to front -- a failure that looks like bad calibration rather than a
# convention mismatch, which is why it is spelled out here.
_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def intrinsics_from_fov(width: int, height: int, fov_degrees: float = DEFAULT_FOV_DEGREES) -> dict[str, float]:
    """Pinhole intrinsics assuming a horizontal field of view."""
    width = max(1, int(width))
    height = max(1, int(height))
    fov = math.radians(float(np.clip(float(fov_degrees), 5.0, 170.0)))
    fx = width / (2.0 * math.tan(fov / 2.0))
    return {
        "fl_x": float(fx),
        "fl_y": float(fx),
        "cx": float(width) / 2.0,
        "cy": float(height) / 2.0,
        "w": int(width),
        "h": int(height),
        "camera_angle_x": float(fov),
    }


def export_dataset(
    captures,
    output_dir: str | Path,
    *,
    session_id: str = "",
    hand_eye: np.ndarray | None = None,
    fov_degrees: float = DEFAULT_FOV_DEGREES,
    copy_images: bool = True,
) -> SplatDataset:
    """Write a splatting-ready dataset from capture records.

    Each capture must carry an image path and the pose it was taken from --
    ``actual_tcp_pose`` where the robot reported one, otherwise the commanded
    ``target_position``. Captures missing either are skipped and reported in
    ``warnings`` rather than silently dropped, since a quietly short dataset
    reconstructs badly for no visible reason.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    colmap_dir = output_dir / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    colmap_dir.mkdir(parents=True, exist_ok=True)

    hand_eye = hand_eye_matrix() if hand_eye is None else np.asarray(hand_eye, dtype=float)
    warnings: list[str] = []
    frames: list[dict[str, Any]] = []
    colmap_images: list[str] = []
    width = height = 0

    for index, capture in enumerate(captures):
        source = Path(str(capture.get("path") or capture.get("image_path") or ""))
        if not source.exists():
            warnings.append(f"missing image: {source.name or '(no path)'}")
            continue
        pose = capture.get("actual_tcp_pose") or capture.get("target_position")
        if pose is None or len(np.asarray(pose, dtype=float).reshape(-1)) < 6:
            warnings.append(f"no camera pose for {source.name}")
            continue

        if width == 0:
            width = int(capture.get("camera_image_width", 0))
            height = int(capture.get("camera_image_height", 0))
            if width <= 0 or height <= 0:
                from PIL import Image

                with Image.open(source) as probe:
                    width, height = probe.size

        name = source.name
        target = images_dir / name
        if copy_images:
            if not target.exists():
                shutil.copy2(source, target)
            relative = f"images/{name}"
        else:
            relative = str(source)

        c2w = camera_to_world(pose, hand_eye)
        frames.append({
            "file_path": relative,
            "transform_matrix": (c2w @ _CV_TO_GL).tolist(),
            "colmap_im_id": index + 1,
            "robot_pose": [float(v) for v in np.asarray(pose, dtype=float).reshape(-1)[:6]],
            "angle": str(capture.get("angle", "")),
            "row": int(capture.get("row", 0)),
            "col": int(capture.get("col", 0)),
        })

        # COLMAP stores world-to-camera, the inverse of what transforms.json
        # holds, with the rotation as a wxyz quaternion.
        w2c = np.linalg.inv(c2w)
        quat = Rotation.from_matrix(w2c[:3, :3]).as_quat()  # xyzw
        qw, qx, qy, qz = float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])
        tx, ty, tz = (float(v) for v in w2c[:3, 3])
        colmap_images.append(
            f"{index + 1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {tx:.9f} {ty:.9f} {tz:.9f} 1 {name}"
        )

    if not frames:
        warnings.append("no usable captures; nothing was exported")
        width, height = max(width, 1), max(height, 1)

    intrinsics = intrinsics_from_fov(width or 1, height or 1, fov_degrees)
    transforms = {
        **intrinsics,
        "camera_model": "PINHOLE",
        "session_id": str(session_id),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pose_source": "robot forward kinematics (no structure-from-motion)",
        "hand_eye_transform": np.asarray(hand_eye, dtype=float).tolist(),
        "intrinsics_source": f"assumed {float(fov_degrees):.1f} degree horizontal field of view",
        "frames": frames,
    }
    transforms_path = output_dir / "transforms.json"
    with transforms_path.open("w", encoding="utf-8") as handle:
        json.dump(transforms, handle, indent=2)

    _write_colmap_model(colmap_dir, colmap_images, intrinsics)

    return SplatDataset(
        root=output_dir,
        image_count=len(frames),
        transforms_path=transforms_path,
        colmap_dir=colmap_dir,
        session_id=str(session_id),
        warnings=warnings,
    )


def _write_colmap_model(colmap_dir: Path, image_lines: list[str], intrinsics: dict[str, float]) -> None:
    """COLMAP text model, for trainers that read that instead of transforms.json."""
    with (colmap_dir / "cameras.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Camera list with one line of data per camera:\n")
        handle.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        handle.write(
            f"1 PINHOLE {int(intrinsics['w'])} {int(intrinsics['h'])} "
            f"{intrinsics['fl_x']:.6f} {intrinsics['fl_y']:.6f} "
            f"{intrinsics['cx']:.6f} {intrinsics['cy']:.6f}\n"
        )
    with (colmap_dir / "images.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Image list with two lines of data per image:\n")
        handle.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        handle.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for line in image_lines:
            handle.write(line + "\n")
            # Second line per image is the 2D feature list. Empty: the poses
            # come from the robot, so there are no matched features to record.
            handle.write("\n")
    with (colmap_dir / "points3D.txt").open("w", encoding="utf-8") as handle:
        handle.write("# 3D point list with one line of data per point:\n")
        handle.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")


class SplatTrainer:
    """Runs an external Gaussian Splatting trainer over an exported dataset.

    The command is whatever the user has installed -- the Inria reference
    implementation, nerfstudio's ``ns-train``, gsplat -- given as a template
    with ``{dataset}`` and ``{output}`` placeholders. Nothing about a specific
    trainer is assumed, and it runs in a thread with its log captured so a long
    training does not freeze the UI.
    """

    def __init__(self, command_template: str = "", dataset: SplatDataset | None = None, output_dir: str | Path | None = None):
        self.command_template = str(command_template)
        self.dataset = dataset
        self.output_dir = Path(output_dir) if output_dir else None
        self.status = "Not started"
        self.returncode: int | None = None
        self.log: list[str] = []
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def resolved_command(self) -> str:
        dataset_root = str(self.dataset.root) if self.dataset else ""
        output = str(self.output_dir) if self.output_dir else ""
        return self.command_template.format(dataset=dataset_root, output=output)

    def start(self) -> bool:
        if self.running:
            self.status = "Training already running"
            return False
        if not self.command_template.strip():
            self.status = "No trainer command configured"
            return False
        if self.dataset is None or self.dataset.image_count <= 0:
            self.status = "Export a dataset with images before training"
            return False
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        command = self.resolved_command()
        with self._lock:
            self.log = [f"$ {command}"]
        self.returncode = None
        self.status = "Training started"
        self._thread = threading.Thread(target=self._run, args=(command,), daemon=True, name="gsplat-train")
        self._thread.start()
        return True

    def _run(self, command: str) -> None:
        try:
            self._process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self.status = f"Could not start trainer: {exc}"
            self._append(str(exc))
            return
        assert self._process.stdout is not None
        for line in self._process.stdout:
            self._append(line.rstrip())
        self._process.wait()
        self.returncode = int(self._process.returncode)
        self.status = (
            "Training finished" if self.returncode == 0 else f"Trainer exited with code {self.returncode}"
        )

    def _append(self, line: str) -> None:
        with self._lock:
            self.log.append(line)
            # Only the tail is ever displayed, and a long training prints tens
            # of thousands of lines.
            if len(self.log) > 400:
                self.log = self.log[-400:]

    def tail(self, count: int = 12) -> list[str]:
        with self._lock:
            return list(self.log[-int(count):])

    def stop(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass
        self.status = "Training stopped"

    def output_artifacts(self) -> list[Path]:
        """Reconstruction files the trainer produced, newest first."""
        if self.output_dir is None or not self.output_dir.exists():
            return []
        artifacts = [
            path
            for pattern in ("*.ply", "*.splat", "*.ckpt")
            for path in self.output_dir.rglob(pattern)
        ]
        artifacts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return artifacts
