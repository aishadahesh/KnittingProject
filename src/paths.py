"""Where everything lives on disk.

Before this, five modules each derived the project root from their own
``__file__`` and joined filenames onto it. That works right up until a file
moves, at which point every one of those sites has to be found and corrected
independently -- and the ones that are missed fail quietly, because a missing
config file usually surfaces as a default value rather than an error.

Layout is declared once here instead. ``app.py`` stays in the project root;
modules live in ``src/``, configuration and saved parameters in ``config/``,
and images in ``assets/``.
"""

from __future__ import annotations

from pathlib import Path


# This file is src/paths.py, so the project root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "config"
ASSETS_DIR = PROJECT_ROOT / "assets"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Static configuration: the knit parameter definitions and the state schema.
CONFIG_JSON = CONFIG_DIR / "config.json"
STATE_SCHEMA_JSON = CONFIG_DIR / "state_schema.json"

# initial_params.json is the frozen reset baseline and Scan Mode's fabric
# template; params.json is the working autosave the app writes as you edit.
# Both are model parameters, so they sit with the configuration.
INITIAL_PARAMS_JSON = CONFIG_DIR / "initial_params.json"
PARAMS_JSON = CONFIG_DIR / "params.json"

# Dear ImGui's window layout. Written by the UI at runtime rather than being
# configuration anyone edits, so it stays out of config/.
LAYOUT_INI = PROJECT_ROOT / "imgui_layout.ini"

# Runtime output directories.
SCANNER_IMAGES_DIR = PROJECT_ROOT / "scanner_images"
SCANNER_DATA_DIR = PROJECT_ROOT / "scanner_data"
ROBOT_SCANS_DIR = PROJECT_ROOT / "robot_scans"


def resolve(path) -> Path:
    """Resolve a possibly-relative path against the project root.

    Paths stored in config.json (the reference image, for one) are written
    relative to the project root so the file stays portable between machines.
    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
