"""Save a scanned model to Google Drive, through rclone (https://rclone.org), from the app.

A model is saved as folders, as they are in robot_scans (so the Drive folder can be opened and shared
as it is): its scan folder - and, for a model built from several captures (combined.json), each
capture's folder - with what cannot be made again: the raw photos and the arm's poses, the cameras'
poses, the masks, the scale, every splat (the Gaussians, .ply) and the colours, yarns and stitch fits
read from it. What can be made again is left out: training checkpoints, COLMAP databases, the photo
copies inside splat_dataset*, renders and a fit's region.ply. Saving again copies only what changed.

  rclone_path    the rclone program (on PATH, or where winget installed it), or None
  connected      whether the Drive remote is set up
  connect        set it up: a browser opens for the person to sign in to their Google account (rclone
                 is given access only to the files it makes)
  scan_folders   the folders a model's data lives in
  files_to_save  the files of those folders worth saving
  upload         start copying them to Drive (a process; its stderr carries rclone's progress)
  share_link     a link to a folder on Drive, for anyone who has it to view
  saved          the scan folders on Drive already
  progress_of    the share done, read from one of rclone's progress lines

No imgui here.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import paths

REMOTE = "gdrive"
FOLDER = "KnittingProject_data"
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
    scan = Path(output).resolve().parent
    folders = []
    combined = scan / "combined.json"
    if combined.is_file():
        for session in json.loads(combined.read_text(encoding="utf-8")).get("sessions", []):
            path = paths.scan_session(session, scan.parent)
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


def files_to_save(output: Path) -> tuple[Path, list[Path]]:
    """robot_scans, and every file of the model's scan folders worth saving, as paths from it."""
    folders = scan_folders(output)
    root = folders[-1].parent
    return root, [f.relative_to(root) for folder in folders for f in sorted(folder.rglob("*"))
                  if f.is_file() and keep(f.relative_to(root))]


def upload(output: Path, list_file: Path) -> tuple[subprocess.Popen, int, int]:
    """Copy a model's files into the Drive folder as they are, folder by folder (files already there
    and unchanged are skipped). Returns the process - rclone's progress comes on its stderr, a line a
    second - and how many files and bytes go."""
    root, files = files_to_save(output)
    Path(list_file).write_text("\n".join(f.as_posix() for f in files), encoding="utf-8")
    process = subprocess.Popen([rclone_path(), "copy", str(root), f"{REMOTE}:{FOLDER}", "--files-from", str(list_file),
                                "--stats", "1s", "--stats-one-line", "-v", "--transfers", "8", "--checkers", "8",
                                "--retries", "10", "--low-level-retries", "20"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, creationflags=HIDE_WINDOW)
    return process, len(files), sum((root / f).stat().st_size for f in files)


def share_link(name: str = "") -> str:
    """A link to a folder on Drive (the whole data folder by default). Anyone with the link can view it."""
    result = _run("link", f"{REMOTE}:{FOLDER}/{name}".rstrip("/"))
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "rclone link failed")
    return result.stdout.strip()


def saved() -> list[dict]:
    """The scan folders in the Drive folder: [{"name", "modified"}], by name."""
    result = _run("lsjson", f"{REMOTE}:{FOLDER}", "--dirs-only")
    if result.returncode != 0:
        return []
    return sorted(({"name": e["Name"], "modified": e["ModTime"][:16].replace("T", " ")} for e in json.loads(result.stdout or "[]")),
                  key=lambda e: e["name"])


def progress_of(line: str) -> float | None:
    """The share done in one of rclone's --stats-one-line lines ("... 12 MiB / 150 MiB, 8%, ..."), or None."""
    match = re.search(r",\s*(\d+)%", line)
    return int(match.group(1)) / 100.0 if match else None
