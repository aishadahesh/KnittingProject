"""Save a scanned model to Google Drive, through rclone (https://rclone.org), from the app.

A model is saved as one zip of what cannot be made again: its scan folder - and, for a model built from
several captures (combined.json), each capture's folder - with the raw photos and the arm's poses, the
cameras' poses, the masks, the scale, every splat (the Gaussians, .ply) and the colours, yarns and
stitch fits read from it. What can be made again is left out: training checkpoints, COLMAP databases,
the photo copies inside splat_dataset*, renders and a fit's region.ply. Paths in the zip start at the
scan folder's name, so it unzips straight back into robot_scans.

  rclone_path    the rclone program (on PATH, or where winget installed it), or None
  connected      whether the Drive remote is set up
  connect        set it up: a browser opens for the person to sign in to their Google account (rclone
                 is given access only to the files it makes)
  scan_folders   the folders a model's data lives in
  pack           zip them (returns the zip and what went in)
  upload         start copying a zip to Drive (a process; its stderr carries rclone's progress)
  saved          what is on Drive already
  progress_of    the share done, read from one of rclone's progress lines

No imgui here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

REMOTE = "gdrive"
FOLDER = "KnittingProject_data"
STORED = {".png", ".jpg", ".ply", ".mp4", ".npy"}          # already compressed: stored as they are
REBUILDABLE = {".ckpt", ".db"}
HIDE_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def rclone_path() -> str | None:
    found = shutil.which("rclone")
    if found:
        return found
    packages = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    candidates = sorted(packages.glob("Rclone.Rclone_*/rclone-*/rclone.exe")) if packages.is_dir() else []
    return str(candidates[-1]) if candidates else None


def _run(*args, timeout=60) -> subprocess.CompletedProcess:
    return subprocess.run([rclone_path(), *args], capture_output=True, text=True, timeout=timeout, creationflags=HIDE_WINDOW)


def connected() -> bool:
    if rclone_path() is None:
        return False
    try:
        return f"{REMOTE}:" in _run("listremotes").stdout.split()
    except (OSError, subprocess.SubprocessError):
        return False


def connect() -> subprocess.Popen:
    """Set the Drive remote up; rclone opens the browser for the Google sign-in and waits for it."""
    return subprocess.Popen([rclone_path(), "config", "create", REMOTE, "drive", "scope=drive.file"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, creationflags=HIDE_WINDOW)


def scan_folders(output: Path) -> list[Path]:
    """The scan folder a model (its splat_output folder) belongs to, after the captures it was built from."""
    scan = Path(output).parent
    folders = []
    combined = scan / "combined.json"
    if combined.is_file():
        for session in json.loads(combined.read_text(encoding="utf-8")).get("sessions", []):
            path = Path(session)
            path = path if path.is_dir() else scan.parent / path.name
            if path.is_dir() and path != scan:
                folders.append(path)
    return folders + [scan]


def keep(relative: Path) -> bool:
    """Whether a file (its path from robot_scans) is one that cannot be made again."""
    parts = [p.lower() for p in relative.parts]
    suffix = relative.suffix.lower()
    if suffix in REBUILDABLE or relative.name == "region.ply" or "renders" in parts:
        return False
    if any(p.startswith("splat_dataset") for p in parts) and suffix in (".png", ".jpg"):
        return any("mask" in p for p in parts)
    return True


def pack(output: Path, out_dir: Path, report=lambda share: None) -> tuple[Path, dict]:
    """Zip a model's scan folders into out_dir/<scan folder>.zip: (the zip, what went in)."""
    folders = scan_folders(output)
    root = folders[-1].parent
    files = [f for folder in folders for f in sorted(folder.rglob("*")) if f.is_file() and keep(f.relative_to(root))]
    total = max(sum(f.stat().st_size for f in files), 1)
    target = Path(out_dir) / f"{folders[-1].name}.zip"
    done = 0
    with zipfile.ZipFile(target, "w") as archive:
        for f in files:
            method = zipfile.ZIP_STORED if f.suffix.lower() in STORED else zipfile.ZIP_DEFLATED
            archive.write(f, f.relative_to(root).as_posix(), compress_type=method)
            done += f.stat().st_size
            report(done / total)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return target, {"folders": [f.name for f in folders], "files": len(files), "bytes": target.stat().st_size,
                    "sha256": digest.hexdigest(), "unzip_into": "robot_scans"}


def upload(path: Path) -> subprocess.Popen:
    """Copy a file into the Drive folder; rclone's progress comes on stderr, a line a second."""
    return subprocess.Popen([rclone_path(), "copyto", str(path), f"{REMOTE}:{FOLDER}/{Path(path).name}",
                             "--stats", "1s", "--stats-one-line", "-v", "--retries", "10", "--low-level-retries", "20",
                             "--drive-chunk-size", "64M"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, creationflags=HIDE_WINDOW)


def saved() -> list[dict]:
    """What is in the Drive folder: [{"name", "bytes", "modified"}], newest first."""
    result = _run("lsjson", f"{REMOTE}:{FOLDER}", "--files-only")
    if result.returncode != 0:
        return []
    entries = [{"name": e["Name"], "bytes": e["Size"], "modified": e["ModTime"][:16].replace("T", " ")} for e in json.loads(result.stdout or "[]")]
    return sorted(entries, key=lambda e: e["modified"], reverse=True)


def progress_of(line: str) -> float | None:
    """The share done in one of rclone's --stats-one-line lines ("... 12 MiB / 150 MiB, 8%, ..."), or None."""
    match = re.search(r",\s*(\d+)%", line)
    return int(match.group(1)) / 100.0 if match else None
