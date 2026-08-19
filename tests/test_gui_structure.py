"""Structural invariants for the GUI modules.

These are the rules the module layout has to keep obeying, written as tests so
they survive the refactor rather than living in a comment that goes stale. They
parse source with `ast` instead of importing, so they cost nothing and work
without a display.

The one that matters most is `test_no_module_imports_a_foreign_mode`. The whole
point of splitting `gui.py` per mode is that scan-mode code stops living
alongside edit-mode code; without a check, the two drift back together one
convenience import at a time.
"""

import ast
import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parent.parent / "src"
APP = Path(__file__).resolve().parent.parent / "app.py"

# One module per mode. A module here may not import another module here.
MODE_MODULES = ("gui_edit", "gui_scan", "gui_puzzle", "gui_database", "gui_ur5")

# Modules that must stay usable without a UI: they are imported by headless
# callers (tests, scripts, the scan runner) where imgui does not exist.
IMGUI_FREE_MODULES = ("fabric_scanner", "scanner_core", "embedded_scanner",
                      "rgb_analysis", "robot_camera", "ur5_robot", "ur5_scan",
                      "gaussian_splatting", "paths")


def _existing(names):
    """Only the modules that exist yet -- the split lands in stages."""
    return [name for name in names if (SRC / f"{name}.py").exists()]


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_names(tree: ast.Module) -> set[str]:
    """Every module name imported, including inside functions.

    Function-level imports are the interesting case: they are how a cycle gets
    reintroduced after being removed, precisely because they do not fail at
    import time.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


def _top_level_names(tree: ast.Module) -> set[str]:
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def test_app_imports_only_the_documented_entry_points():
    """app.py's contract with gui.py: four drawing functions, nothing else."""
    tree = _parse(APP)
    imported_from_gui = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "gui"
        for alias in node.names
    }
    assert imported_from_gui, "app.py should import its drawing entry points from gui"
    # Whatever app.py asks for, gui must actually provide.
    sys.path.insert(0, str(SRC))
    import gui

    for name in imported_from_gui:
        assert hasattr(gui, name), f"app.py imports gui.{name}, which does not exist"


def test_no_top_level_name_is_defined_twice():
    """A moved function must not be left behind in its old home as well."""
    modules = _existing(("gui", "gui_common", "scanner_core", "embedded_scanner", *MODE_MODULES))
    owner: dict[str, str] = {}
    clashes = []
    for name in modules:
        tree = _parse(SRC / f"{name}.py")
        for symbol in _top_level_names(tree):
            if symbol.startswith("__"):
                continue
            if symbol in owner:
                clashes.append(f"{symbol}: defined in both {owner[symbol]}.py and {name}.py")
            else:
                owner[symbol] = name
    assert not clashes, "duplicate definitions:\n  " + "\n  ".join(clashes)


def test_no_module_imports_a_foreign_mode():
    """Each mode's code stays in its own module.

    This is the rule the split exists to enforce: Scan Mode must not reach into
    Edit Mode's module, and vice versa. `gui.py` is exempt -- dispatching to
    every mode is its whole job.
    """
    violations = []
    for name in _existing(MODE_MODULES):
        imported = _imported_names(_parse(SRC / f"{name}.py"))
        for other in MODE_MODULES:
            if other != name and other in imported:
                violations.append(f"{name}.py imports {other}")
    assert not violations, "modes must not import each other:\n  " + "\n  ".join(violations)


def test_no_mode_module_imports_gui():
    """Dependencies point one way: gui -> modes, never back.

    `gui_ur5.py` used to import `gui` from inside three functions to dodge the
    cycle. Walking function-level imports is what caught that, and what keeps it
    from quietly returning.
    """
    offenders = [
        name for name in _existing(MODE_MODULES)
        if "gui" in _imported_names(_parse(SRC / f"{name}.py"))
    ]
    assert not offenders, (
        "these modules import gui, creating a cycle: " + ", ".join(offenders)
        + " -- the shared helpers belong in scanner_core or gui_common"
    )


@pytest.mark.parametrize("module", IMGUI_FREE_MODULES)
def test_core_modules_are_imgui_free(module):
    """Logic modules stay importable without a UI toolkit."""
    path = SRC / f"{module}.py"
    if not path.exists():
        pytest.skip(f"{module}.py does not exist yet")
    imported = _imported_names(_parse(path))
    assert "imgui" not in imported and "imgui_bundle" not in imported, (
        f"{module}.py imports imgui; it is meant to be usable headlessly"
    )


def test_import_graph_is_acyclic():
    """No cycles among the project's own modules, function-level imports included."""
    own = {path.stem for path in SRC.glob("*.py")}
    graph = {
        name: _imported_names(_parse(SRC / f"{name}.py")) & own
        for name in own
    }

    visiting: set[str] = set()
    done: set[str] = set()
    cycles: list[str] = []

    def visit(node, trail):
        if node in visiting:
            start = trail.index(node)
            cycles.append(" -> ".join(trail[start:] + [node]))
            return
        if node in done:
            return
        visiting.add(node)
        for neighbour in sorted(graph.get(node, ())):
            visit(neighbour, trail + [neighbour])
        visiting.discard(node)
        done.add(node)

    for node in sorted(graph):
        visit(node, [node])

    assert not cycles, "import cycles:\n  " + "\n  ".join(sorted(set(cycles)))
