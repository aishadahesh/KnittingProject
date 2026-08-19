"""Persistent local storage for the fabric scanner workflow.

The scanner keeps image files on disk and stores searchable metadata/state in a
small SQLite database. This avoids putting large PNG blobs into the database
while still letting the app restore previous patterns, settings, captures, and
analysis results when it is reopened.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np


DB_DIR_NAME = "scanner_data"
DB_FILE_NAME = "scanner.db"
JSON_INDEX_FILE_NAME = "captures_index.json"

# Everything that changes the images a scan produces, or the RGB measured from
# them. Two runs agreeing on all of this would photograph the same scene the
# same way, which is what makes the second one a repeat rather than new data.
#
# Kept separate from pattern_signature deliberately: that one is the key rows
# are stored under, so widening it would orphan every capture already in the
# database from its pattern set. This is only ever compared, never stored as a
# foreign key.
SCAN_CONFIG_KEYS = (
    # Fabric grid and the patterns generated onto it
    "scanner_rows", "scanner_cols", "scanner_layout_pattern",
    "scanner_pattern_rows", "scanner_pattern_cols",
    "scanner_pattern_density", "scanner_random_seed",
    "scanner_fabric_width", "scanner_fabric_length",
    # How each pattern repeats across its cell
    "scanner_pattern_repeat_rows", "scanner_pattern_repeat_cols",
    "scanner_repeat_spacing_x", "scanner_repeat_spacing_y",
    # Yarn colours
    "scanner_color_variants", "scanner_cell_color_sets",
    # Lighting
    "scanner_lighting_enabled", "scanner_light_azimuth", "scanner_light_elevation",
    "scanner_light_sun_intensity", "scanner_light_shadow", "scanner_light_sheen",
    # Camera and capture
    "scanner_angles", "scanner_capture_mode", "scanner_camera_zoom",
    "scanner_camera_workflow", "scanner_image_every", "scanner_save_images",
    "scanner_capture_width",
    # Which robot took them
    "scanner_execution_mode",
)

# Deliberately excluded, because they do not change a single pixel:
# scanner_speed and scanner_dwell (timing), scanner_add_camera (a viewer
# toggle), scanner_selected_cell and scanner_single_* (UI selection for a
# different workflow), scanner_robot_ip/port, app_mode, ui_theme.


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


def scan_config_signature(settings: dict[str, Any]) -> str:
    """Identity of a scan configuration, for spotting a repeat of one.

    A pure function of a settings snapshot rather than of live state, so the
    same digest can be computed for a run recorded months ago out of the
    settings stored beside it.

    Floats are rounded before hashing: a slider that lands on 0.6200000001
    describes the same scan as one that lands on 0.62, and an exact-equality
    digest would call them different every time.
    """
    def canonical(value):
        value = _json_ready(value)
        if isinstance(value, float):
            return round(value, 6)
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, list):
            return [canonical(item) for item in value]
        if isinstance(value, dict):
            return {str(k): canonical(v) for k, v in sorted(value.items())}
        return value

    payload = {key: canonical(settings.get(key)) for key in SCAN_CONFIG_KEYS}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


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

                CREATE TABLE IF NOT EXISTS robot_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL UNIQUE,
                    robot_mode TEXT NOT NULL,
                    robot_ip TEXT,
                    transport TEXT,
                    is_real INTEGER NOT NULL DEFAULT 0,
                    camera_backend TEXT,
                    camera_is_real INTEGER NOT NULL DEFAULT 0,
                    pattern_signature TEXT,
                    settings_json TEXT,
                    status TEXT,
                    capture_count INTEGER NOT NULL DEFAULT 0,
                    splat_output_path TEXT,
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            capture_columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(captures)").fetchall()
            }
            if "record_json" not in capture_columns:
                conn.execute("ALTER TABLE captures ADD COLUMN record_json TEXT")
            # Real-UR5 columns, added the same way record_json was: existing
            # databases full of simulation captures keep working and simply read
            # back the simulation defaults for these.
            for column, ddl in (
                ("robot_mode", "ALTER TABLE captures ADD COLUMN robot_mode TEXT DEFAULT 'simulation'"),
                ("session_id", "ALTER TABLE captures ADD COLUMN session_id TEXT"),
                ("robot_ip", "ALTER TABLE captures ADD COLUMN robot_ip TEXT"),
                ("target_position_json", "ALTER TABLE captures ADD COLUMN target_position_json TEXT"),
                ("lighting_condition", "ALTER TABLE captures ADD COLUMN lighting_condition TEXT"),
                ("splat_output_path", "ALTER TABLE captures ADD COLUMN splat_output_path TEXT"),
            ):
                if column not in capture_columns:
                    conn.execute(ddl)
            if "config_signature" not in capture_columns:
                conn.execute("ALTER TABLE captures ADD COLUMN config_signature TEXT")
                self._backfill_config_signatures(conn)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_captures_config ON captures(config_signature)"
            )

    @staticmethod
    def _backfill_config_signatures(conn) -> None:
        """Derive the config digest for rows written before it was stored.

        Runs once, in the same transaction that adds the column, so scans
        already in the database can be recognised as duplicates rather than the
        check only working for runs made from here on. Every row already carries
        the settings it was captured under, so nothing has to be guessed.
        """
        rows = conn.execute(
            "SELECT id, capture_settings_json FROM captures WHERE config_signature IS NULL"
        ).fetchall()
        updates = []
        for row_id, settings_json in rows:
            try:
                settings = json.loads(settings_json) if settings_json else {}
            except json.JSONDecodeError:
                continue
            if isinstance(settings, dict):
                updates.append((scan_config_signature(settings), int(row_id)))
        if updates:
            conn.executemany(
                "UPDATE captures SET config_signature = ? WHERE id = ?", updates
            )

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

    def record_capture(
        self,
        state,
        record: dict[str, Any],
        capture_settings: dict[str, Any] | None = None,
        signature: str | None = None,
    ) -> None:
        """Stores one capture.

        The robot-specific columns are read off the record itself and default to
        the simulation values, so the existing Scan Mode call site needs no
        changes and its rows keep reading back exactly as before.

        ``signature`` and ``capture_settings`` may be supplied to avoid reading
        the live AppState here. The real UR5 scan records from its own thread,
        where walking that state while the UI thread mutates it would be a race;
        it snapshots both once when the run starts and passes them in.
        """
        signature = self.pattern_signature(state) if signature is None else str(signature)
        settings = capture_settings or self.scanner_state_snapshot(state)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        target_position = record.get("target_position")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO captures
                    (pattern_signature, image_path, row_index, col_index, station_index, target_index,
                     angle, capture_mode, capture_settings_json, rgb_json, record_json, created_at,
                     robot_mode, session_id, robot_ip, target_position_json, lighting_condition,
                     splat_output_path, config_signature)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signature,
                    str(record.get("path", "")),
                    int(record.get("row", 0)),
                    int(record.get("col", 0)),
                    int(record.get("station", 0)),
                    int(record.get("target_index", 0)),
                    str(record.get("angle", "")),
                    str(record.get("capture_mode", _state_get(state, "scanner_capture_mode", "natural"))),
                    json.dumps(_json_ready(settings), sort_keys=True),
                    json.dumps(_json_ready(record.get("rgb", [])), sort_keys=True),
                    json.dumps(_json_ready(record), sort_keys=True),
                    now,
                    str(record.get("robot_mode", "simulation")),
                    str(record.get("session_id", "")) or None,
                    str(record.get("robot_ip", "")) or None,
                    json.dumps(_json_ready(target_position)) if target_position is not None else None,
                    str(record.get("lighting_condition", "")) or None,
                    str(record.get("splat_output_path", "")) or None,
                    scan_config_signature(settings),
                ),
            )
        self.revision += 1
        self._index_dirty = True

    # -- Duplicate scan detection --------------------------------------------

    def scan_config_signature_for_state(self, state) -> str:
        """The config digest for the settings currently on screen."""
        return scan_config_signature(self.scanner_state_snapshot(state))

    def find_scan_runs_for_config(self, signature: str) -> list[dict[str, Any]]:
        """Previous scan runs that used this exact configuration.

        Grouped by run rather than returned per capture, because what the user
        needs to decide is "have I already scanned this scene", and a run is the
        unit they would be repeating. The run name is the folder the images were
        written to, which is what the Database panel shows them under.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT image_path, created_at
                FROM captures
                WHERE config_signature = ?
                ORDER BY id
                """,
                (str(signature),),
            ).fetchall()

        runs: dict[str, dict[str, Any]] = {}
        for image_path, created_at in rows:
            run = Path(str(image_path)).parent.name or "(unknown run)"
            entry = runs.setdefault(run, {
                "scan_run": run,
                "capture_count": 0,
                "first_capture_at": str(created_at),
                "last_capture_at": str(created_at),
            })
            entry["capture_count"] += 1
            entry["last_capture_at"] = str(created_at)
        return sorted(runs.values(), key=lambda item: item["last_capture_at"], reverse=True)

    # -- Real UR5 sessions ---------------------------------------------------

    def start_robot_session(self, session: dict[str, Any]) -> str:
        """Opens a row for one real-robot scan run and returns its session id.

        A session ties a run's captures together with the hardware that made
        them -- which arm, at which address, through which camera -- so a scan
        stays attributable long after the run.
        """
        session_id = str(session.get("session_id", ""))
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO robot_sessions
                    (session_id, robot_mode, robot_ip, transport, is_real, camera_backend,
                     camera_is_real, pattern_signature, settings_json, status, capture_count,
                     splat_output_path, started_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    str(session.get("robot_mode", "real_ur5")),
                    str(session.get("robot_ip", "")),
                    str(session.get("transport", "")),
                    1 if session.get("is_real") else 0,
                    str(session.get("camera_backend", "")),
                    1 if session.get("camera_is_real") else 0,
                    str(session.get("pattern_signature", "")),
                    json.dumps(_json_ready(session.get("settings", {})), sort_keys=True),
                    str(session.get("status", "running")),
                    int(session.get("capture_count", 0)),
                    str(session.get("splat_output_path", "")) or None,
                    now,
                    now,
                ),
            )
        self.revision += 1
        self._index_dirty = True
        return session_id

    def update_robot_session(self, session_id: str, **fields: Any) -> None:
        allowed = {
            "status", "capture_count", "splat_output_path", "robot_ip",
            "camera_backend", "pattern_signature",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            return
        assignments = ", ".join(f"{key} = ?" for key in updates)
        values = [
            int(value) if key == "capture_count" else (None if value is None else str(value))
            for key, value in updates.items()
        ]
        with self._connect() as conn:
            conn.execute(
                f"UPDATE robot_sessions SET {assignments}, updated_at = ? WHERE session_id = ?",
                (*values, time.strftime("%Y-%m-%d %H:%M:%S"), str(session_id)),
            )
        self.revision += 1
        self._index_dirty = True

    def set_session_splat_output(self, session_id: str, output_path: str) -> None:
        """Records a Gaussian Splatting result against a session and its captures."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE robot_sessions SET splat_output_path = ?, updated_at = ? WHERE session_id = ?",
                (str(output_path), time.strftime("%Y-%m-%d %H:%M:%S"), str(session_id)),
            )
            conn.execute(
                "UPDATE captures SET splat_output_path = ? WHERE session_id = ?",
                (str(output_path), str(session_id)),
            )
        self.revision += 1
        self._index_dirty = True

    def robot_sessions(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, robot_mode, robot_ip, transport, is_real, camera_backend,
                       camera_is_real, pattern_signature, settings_json, status, capture_count,
                       splat_output_path, started_at, updated_at
                FROM robot_sessions
                ORDER BY id DESC
                """
            ).fetchall()
        sessions = []
        for row in rows:
            sessions.append({
                "session_id": str(row[0]),
                "robot_mode": str(row[1]),
                "robot_ip": str(row[2] or ""),
                "transport": str(row[3] or ""),
                "is_real": bool(row[4]),
                "camera_backend": str(row[5] or ""),
                "camera_is_real": bool(row[6]),
                "pattern_signature": str(row[7] or ""),
                "settings": self._load_json(row[8], {}),
                "status": str(row[9] or ""),
                "capture_count": int(row[10] or 0),
                "splat_output_path": str(row[11] or ""),
                "started_at": str(row[12]),
                "updated_at": str(row[13]),
            })
        return sessions

    def save_analysis(self, state, result: dict[str, Any], signature: str | None = None) -> None:
        signature = self.pattern_signature(state) if signature is None else str(signature)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO analyses (pattern_signature, result_json, created_at)
                VALUES (?, ?, ?)
                """,
                (signature, json.dumps(_json_ready(result), sort_keys=True), now),
            )
        # Skipped when state is absent: the real UR5 analysis runs off-thread
        # and walking the live state there would race the UI.
        if state is not None:
            self.save_scanner_state(state)
        self.revision += 1
        self.write_json_index()

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
                       record_json, created_at, robot_mode, session_id, robot_ip,
                       target_position_json, lighting_condition, splat_output_path
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
            # Real captures report the lighting they were actually taken under;
            # simulated ones derive it from the scanner's lighting settings.
            lighting_condition = str(row[17] or "")
            capture = {
                "id": int(row[0]),
                "pattern_id": pattern.get("id"),
                "pattern_name": pattern.get("name", "Unknown pattern"),
                "pattern_signature": signature,
                "bitmap": pattern_cell.get("bitmap"),
                "selected_colors": pattern.get("shared_colors", settings.get("scanner_color_variants", [])),
                "selected_colors_label": pattern.get("shared_colors_label", self._colors_label(settings.get("scanner_color_variants", []))),
                "lighting_mode": lighting_condition or pattern.get("lighting_mode", self._lighting_label(settings)),
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
                # Which robot produced this capture, and everything specific to
                # a real run. Simulation rows report "simulation" and leave the
                # rest empty.
                "robot_mode": str(row[13] or "simulation"),
                "session_id": str(row[14] or ""),
                "robot_ip": str(row[15] or ""),
                "target_position": self._load_json(row[16], None),
                "lighting_condition": lighting_condition,
                "splat_output_path": str(row[18] or ""),
                "camera_backend": record.get("camera_backend", ""),
                "camera_is_real": bool(record.get("camera_is_real", False)),
            }
            captures.append(capture)

        sessions = self.robot_sessions()

        summary = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "database_path": str(self.db_path),
            "json_path": str(self.json_index_path),
            "pattern_count": len(patterns),
            "capture_count": len(captures),
            "analysis_count": len(analyses),
            "robot_session_count": len(sessions),
            "patterns": patterns,
            "captures": captures,
            "analyses": analyses,
            "robot_sessions": sessions,
            "categories": {
                "patterns": sorted({str(item.get("pattern_name", "Unknown pattern")) for item in captures}),
                "selected_colors": sorted({str(item.get("selected_colors_label", "")) for item in captures}),
                "lighting_modes": sorted({str(item.get("lighting_mode", "")) for item in captures}),
                "camera_angles": sorted({str(item.get("camera_angle", "")) for item in captures}),
                "batches": sorted({str(item.get("batch_id", "")) for item in captures}),
                "scan_runs": sorted({str(item.get("scan_run", "")) for item in captures}),
                "robot_modes": sorted({str(item.get("robot_mode", "simulation")) for item in captures}),
                "sessions": sorted({str(item.get("session_id", "")) for item in captures if item.get("session_id")}),
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
