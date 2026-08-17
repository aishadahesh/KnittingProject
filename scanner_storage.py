"""Persistent local storage for the fabric scanner workflow.

The scanner keeps image files on disk and stores searchable metadata/state in a
small SQLite database. This avoids putting large PNG blobs into the database
while still letting the app restore previous patterns, settings, captures, and
analysis results when it is reopened.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np


DB_DIR_NAME = "scanner_data"
DB_FILE_NAME = "scanner.db"
JSON_INDEX_FILE_NAME = "captures_index.json"


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _state_get(state, key: str, default=None):
    if hasattr(state, "get"):
        return state.get(key, default)
    return getattr(state, key, default)


def database_path(project_root: str | Path) -> Path:
    return Path(project_root) / DB_DIR_NAME / DB_FILE_NAME


class ScannerStorage:
    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root)
        self.db_path = database_path(self.project_root)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        # captures_index.json is a derived export of the entire database, so its
        # cost scales with all history rather than with the one row just added
        # (13 MB / 0.6 s at the time of writing, against a 0.2 ms insert).
        # Writing it per capture made every scan slower than the one before.
        # Writes are deferred to flush_json_index(); SQLite stays the record of
        # truth in the meantime, so nothing is at risk.
        self._index_dirty = False
        # Bumped by every write, so derived views can be cached against it.
        self.revision = 0
        self._summary_cache = None

    @property
    def json_index_path(self) -> Path:
        return self.db_path.parent / JSON_INDEX_FILE_NAME

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS scanner_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS pattern_sets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signature TEXT NOT NULL UNIQUE,
                    settings_json TEXT NOT NULL,
                    patterns_json TEXT NOT NULL,
                    estimated_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS captures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_signature TEXT,
                    image_path TEXT NOT NULL,
                    row_index INTEGER NOT NULL,
                    col_index INTEGER NOT NULL,
                    station_index INTEGER NOT NULL,
                    target_index INTEGER NOT NULL,
                    angle TEXT NOT NULL,
                    capture_mode TEXT NOT NULL,
                    capture_settings_json TEXT NOT NULL,
                    rgb_json TEXT,
                    record_json TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_signature TEXT,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            capture_columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(captures)").fetchall()
            }
            if "record_json" not in capture_columns:
                conn.execute("ALTER TABLE captures ADD COLUMN record_json TEXT")

    def scanner_state_snapshot(self, state) -> dict[str, Any]:
        data = getattr(state, "_data", {})
        snapshot = {}
        for key, value in data.items():
            if key in {"scanner_storage", "scanner_process", "embedded_scanner"}:
                continue
            if str(key).startswith("scanner_") or key in {"app_mode", "ui_theme"}:
                snapshot[str(key)] = _json_ready(value)
        return snapshot

    def save_scanner_state(self, state) -> None:
        snapshot = self.scanner_state_snapshot(state)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO scanner_state (id, state_json, updated_at)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (json.dumps(snapshot, sort_keys=True), now),
            )

    def restore_scanner_state(self, state) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT state_json FROM scanner_state WHERE id = 1").fetchone()
        if row is None:
            return False
        try:
            snapshot = json.loads(row[0])
        except json.JSONDecodeError:
            return False
        data = getattr(state, "_data", {})
        for key, value in snapshot.items():
            if key in data:
                current = data[key]
                if isinstance(current, np.ndarray):
                    try:
                        data[key] = np.asarray(value, dtype=current.dtype)
                    except Exception:
                        data[key] = np.asarray(value)
                else:
                    data[key] = value
            elif str(key).startswith("scanner_"):
                data[key] = value
        return True

    def pattern_signature(self, state) -> str:
        keys = [
            "scanner_rows",
            "scanner_cols",
            "scanner_pattern_rows",
            "scanner_pattern_cols",
            "scanner_pattern_density",
            "scanner_random_seed",
            "scanner_pattern_repeat_rows",
            "scanner_pattern_repeat_cols",
            "scanner_repeat_spacing_x",
            "scanner_repeat_spacing_y",
            "scanner_color_variants",
        ]
        payload = {key: _json_ready(_state_get(state, key)) for key in keys}
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def save_pattern_set(self, state, patterns, estimates) -> str:
        signature = self.pattern_signature(state)
        settings = self.scanner_state_snapshot(state)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO pattern_sets
                    (signature, settings_json, patterns_json, estimated_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(signature) DO UPDATE SET
                    settings_json = excluded.settings_json,
                    patterns_json = excluded.patterns_json,
                    estimated_json = excluded.estimated_json,
                    updated_at = excluded.updated_at
                """,
                (
                    signature,
                    json.dumps(settings, sort_keys=True),
                    json.dumps(_json_ready(patterns), sort_keys=True),
                    json.dumps(_json_ready(estimates), sort_keys=True),
                    now,
                    now,
                ),
            )
        self.save_scanner_state(state)
        self.revision += 1
        # Deferred like record_capture: this runs while the scanner is being
        # constructed, i.e. inside the "Start scanning" click, where a 13 MB
        # export costs more than everything else in the sequence except the
        # tile renders. flush_json_index() picks it up at the end of the scan
        # and whenever the Database panel is drawn.
        self._index_dirty = True
        return signature

    def record_capture(self, state, record: dict[str, Any], capture_settings: dict[str, Any] | None = None) -> None:
        signature = self.pattern_signature(state)
        settings = capture_settings or self.scanner_state_snapshot(state)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO captures
                    (pattern_signature, image_path, row_index, col_index, station_index, target_index,
                     angle, capture_mode, capture_settings_json, rgb_json, record_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signature,
                    str(record.get("path", "")),
                    int(record.get("row", 0)),
                    int(record.get("col", 0)),
                    int(record.get("station", 0)),
                    int(record.get("target_index", 0)),
                    str(record.get("angle", "")),
                    str(_state_get(state, "scanner_capture_mode", "natural")),
                    json.dumps(_json_ready(settings), sort_keys=True),
                    json.dumps(_json_ready(record.get("rgb", [])), sort_keys=True),
                    json.dumps(_json_ready(record), sort_keys=True),
                    now,
                ),
            )
        self.revision += 1
        self._index_dirty = True

    def save_analysis(self, state, result: dict[str, Any]) -> None:
        signature = self.pattern_signature(state)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO analyses (pattern_signature, result_json, created_at)
                VALUES (?, ?, ?)
                """,
                (signature, json.dumps(_json_ready(result), sort_keys=True), now),
            )
        self.save_scanner_state(state)
        self.revision += 1
        self.write_json_index()

    def captures_for_signature(self, signature: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT record_json, image_path, row_index, col_index, station_index,
                       target_index, angle, rgb_json
                FROM captures
                WHERE pattern_signature = ?
                ORDER BY id
                """,
                (signature,),
            ).fetchall()
        captures = []
        for row in rows:
            if row[0]:
                try:
                    captures.append(json.loads(row[0]))
                    continue
                except json.JSONDecodeError:
                    pass
            try:
                rgb = json.loads(row[7]) if row[7] else []
            except json.JSONDecodeError:
                rgb = []
            captures.append({
                "path": row[1],
                "row": int(row[2]),
                "col": int(row[3]),
                "station": int(row[4]),
                "target_index": int(row[5]),
                "angle": str(row[6]),
                "rgb": rgb,
            })
        return captures

    def latest_analysis(self, signature: str | None = None) -> dict[str, Any] | None:
        with self._connect() as conn:
            if signature is None:
                row = conn.execute(
                    "SELECT result_json FROM analyses ORDER BY id DESC LIMIT 1"
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT result_json FROM analyses WHERE pattern_signature = ? ORDER BY id DESC LIMIT 1",
                    (signature,),
                ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return None

    def _load_json(self, text: str | None, fallback):
        if not text:
            return fallback
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return fallback

    @staticmethod
    def _lighting_label(settings: dict[str, Any]) -> str:
        if float(settings.get("scanner_lighting_enabled", 1.0) or 0.0) < 0.5:
            return "No Lighting Effect"
        az = float(settings.get("scanner_light_azimuth", -35.0) or -35.0)
        el = float(settings.get("scanner_light_elevation", 48.0) or 48.0)
        return f"Sunlight az {az:.0f} el {el:.0f}"

    @staticmethod
    def _colors_label(colors) -> str:
        values = []
        for color in colors or []:
            if isinstance(color, (list, tuple)) and len(color) >= 3:
                rgb = [int(round(float(v) * 255.0)) if float(v) <= 1.0 else int(round(float(v))) for v in color[:3]]
                values.append(f"rgb({rgb[0]},{rgb[1]},{rgb[2]})")
        return ", ".join(values) if values else "no colors"

    def database_summary(self) -> dict[str, Any]:
        with self._connect() as conn:
            pattern_rows = conn.execute(
                """
                SELECT id, signature, settings_json, patterns_json, estimated_json, created_at, updated_at
                FROM pattern_sets
                ORDER BY id
                """
            ).fetchall()
            capture_rows = conn.execute(
                """
                SELECT id, pattern_signature, image_path, row_index, col_index, station_index,
                       target_index, angle, capture_mode, capture_settings_json, rgb_json,
                       record_json, created_at
                FROM captures
                ORDER BY id
                """
            ).fetchall()
            analysis_rows = conn.execute(
                """
                SELECT id, pattern_signature, result_json, created_at
                FROM analyses
                ORDER BY id
                """
            ).fetchall()

        patterns_by_signature = {}
        patterns = []
        for row in pattern_rows:
            settings = self._load_json(row[2], {})
            pattern_items = self._load_json(row[3], [])
            estimates = self._load_json(row[4], [])
            shared_colors = settings.get("scanner_color_variants", [])
            pattern = {
                "id": int(row[0]),
                "name": f"Pattern {int(row[0])}",
                "signature": str(row[1]),
                "created_at": str(row[5]),
                "updated_at": str(row[6]),
                "settings": settings,
                "shared_colors": shared_colors,
                "shared_colors_label": self._colors_label(shared_colors),
                "lighting_mode": self._lighting_label(settings),
                "patterns": pattern_items,
                "estimated_colors": estimates,
            }
            patterns_by_signature[pattern["signature"]] = pattern
            patterns.append(pattern)

        analyses = []
        latest_analysis_by_signature = {}
        for row in analysis_rows:
            result = self._load_json(row[2], {})
            item = {
                "id": int(row[0]),
                "pattern_signature": str(row[1]),
                "created_at": str(row[3]),
                "result": result,
            }
            analyses.append(item)
            latest_analysis_by_signature[item["pattern_signature"]] = item

        captures = []
        for row in capture_rows:
            signature = str(row[1] or "")
            pattern = patterns_by_signature.get(signature, {})
            settings = self._load_json(row[9], {})
            record = self._load_json(row[11], {})
            row_index = int(row[3])
            col_index = int(row[4])
            pattern_cell = {}
            for cell in pattern.get("patterns", []):
                if int(cell.get("row", -1)) == row_index and int(cell.get("col", -1)) == col_index:
                    pattern_cell = cell
                    break
            estimate_cell = {}
            for cell in pattern.get("estimated_colors", []):
                if int(cell.get("row", -1)) == row_index and int(cell.get("col", -1)) == col_index:
                    estimate_cell = cell
                    break
            image_path = str(row[2])
            path = Path(image_path)
            capture = {
                "id": int(row[0]),
                "pattern_id": pattern.get("id"),
                "pattern_name": pattern.get("name", "Unknown pattern"),
                "pattern_signature": signature,
                "bitmap": pattern_cell.get("bitmap"),
                "selected_colors": pattern.get("shared_colors", settings.get("scanner_color_variants", [])),
                "selected_colors_label": pattern.get("shared_colors_label", self._colors_label(settings.get("scanner_color_variants", []))),
                "lighting_mode": pattern.get("lighting_mode", self._lighting_label(settings)),
                "camera_angle": str(row[7]),
                "scan_station": int(row[5]) + 1,
                "target_index": int(row[6]),
                "batch_id": f"row{row_index + 1:02d}_col{col_index + 1:02d}",
                "batch_row": row_index + 1,
                "batch_col": col_index + 1,
                "capture_mode": str(row[8]),
                "scan_run": path.parent.name if path.parent.name else "",
                "image_path": image_path,
                "average_rgb": record.get("rgb", self._load_json(row[10], None)),
                "estimated_color": estimate_cell.get("rgb"),
                "capture_settings": settings,
                "created_at": str(row[12]),
                # Dynamic per-capture patch detection metadata (may be absent
                # on captures made before this was tracked).
                "patch_bbox": record.get("patch_bbox"),
                "patch_confidence": record.get("patch_confidence"),
                "camera_zoom_level": record.get("camera_zoom_level"),
                "camera_angle_deg": record.get("camera_angle_deg"),
                "camera_pose": record.get("camera_pose"),
                "patch_image_path": record.get("patch_image_path"),
                "debug_image_path": record.get("debug_image_path"),
            }
            captures.append(capture)

        summary = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "database_path": str(self.db_path),
            "json_path": str(self.json_index_path),
            "pattern_count": len(patterns),
            "capture_count": len(captures),
            "analysis_count": len(analyses),
            "patterns": patterns,
            "captures": captures,
            "analyses": analyses,
            "categories": {
                "patterns": sorted({str(item.get("pattern_name", "Unknown pattern")) for item in captures}),
                "selected_colors": sorted({str(item.get("selected_colors_label", "")) for item in captures}),
                "lighting_modes": sorted({str(item.get("lighting_mode", "")) for item in captures}),
                "camera_angles": sorted({str(item.get("camera_angle", "")) for item in captures}),
                "batches": sorted({str(item.get("batch_id", "")) for item in captures}),
                "scan_runs": sorted({str(item.get("scan_run", "")) for item in captures}),
            },
            "latest_analysis_by_pattern": latest_analysis_by_signature,
        }
        return _json_ready(summary)

    def cached_database_summary(self) -> dict[str, Any]:
        """database_summary() memoised against the write counter.

        The Database panel asks for this every frame it is drawn, and building it
        re-decodes every historical row's JSON blobs. Recomputing only after an
        actual write is what keeps that panel interactive.
        """
        cached = self._summary_cache
        if cached is not None and cached[0] == self.revision:
            return cached[1]
        summary = self.database_summary()
        self._summary_cache = (self.revision, summary)
        return summary

    def write_json_index(self) -> Path:
        """Exports the whole database to captures_index.json, unconditionally."""
        summary = self.cached_database_summary()
        self.json_index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.json_index_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        self._index_dirty = False
        return self.json_index_path

    def flush_json_index(self, force: bool = False) -> Path | None:
        """Exports the index only if a capture has been recorded since the last one.

        Free to call repeatedly, so callers can simply invoke it whenever the
        index might be wanted rather than tracking who owes a write.
        """
        if not force and not self._index_dirty:
            return None
        return self.write_json_index()
