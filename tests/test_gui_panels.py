"""Render every mode's panels against a hidden window, with no imgui backend.

`gui.py` had no test coverage at all, which is an awkward position from which to
move 2400 lines of it between modules. This harness draws each mode for a frame
and asserts nothing raises.

That is a weaker claim than "the UI is correct", but it catches the failure this
refactor actually risks: an unbalanced `imgui.begin`/`end`. imgui asserts on that
internally, so simply completing a frame is the balance check. It also records
`total_vtx_count` per mode, which is the oracle for "this was pure code
movement" -- a move that changes the drawn geometry is a move that changed
behaviour.

Skips cleanly where no GL context can be created, so a headless CI still gets
the static checks in test_gui_structure.py.
"""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

moderngl = pytest.importorskip("moderngl")
imgui_bundle = pytest.importorskip("imgui_bundle")
glfw = pytest.importorskip("glfw")
from imgui_bundle import imgui  # noqa: E402

MODES = ("edit", "scan", "puzzle", "database", "ur5")


def _make_hidden_window():
    """A real but invisible GL context, matching how app.py makes its own.

    A standalone moderngl context has no default framebuffer, and the renderer
    finishes every draw with `ctx.screen.use()` -- so the tests need a context
    that actually has a screen. A hidden window gives one, and exercises the
    same code path production does.
    """
    if not glfw.init():
        return None
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    glfw.window_hint(glfw.VISIBLE, False)
    window = glfw.create_window(640, 480, "test", None, None)
    if window:
        glfw.make_context_current(window)
    return window


def _can_create_context():
    window = _make_hidden_window()
    if window is None:
        return False
    glfw.destroy_window(window)
    return True


pytestmark = pytest.mark.skipif(
    not _can_create_context(), reason="no GL context available for headless rendering"
)


@pytest.fixture(scope="module")
def gl_context():
    """The GL context and the window that owns it.

    The window handle is not optional: draw_viewport polls mouse and keyboard
    state through glfw, which faults on a null window rather than returning a
    default.
    """
    from types import SimpleNamespace

    window = _make_hidden_window()
    if window is None:
        pytest.skip("could not create a hidden GL window")
    ctx = moderngl.create_context()
    yield SimpleNamespace(ctx=ctx, window=window)
    try:
        ctx.release()
    except Exception:
        pass
    glfw.destroy_window(window)
    glfw.terminate()


@pytest.fixture(scope="module")
def imgui_context():
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1600, 900)
    io.delta_time = 1.0 / 60.0
    # imgui 1.92 builds its font atlas lazily and expects the backend to own
    # textures. Claiming that here is what lets new_frame run with no backend
    # attached (io.fonts.get_tex_data_as_rgba32() no longer exists).
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    yield ctx
    imgui.destroy_context(ctx)


@pytest.fixture(scope="module")
def ref_tex(gl_context):
    """The reference-image texture draw_viewport blends as a backdrop."""
    from PIL import Image

    from rendering import pil_to_texture

    return pil_to_texture(gl_context.ctx, Image.new("RGB", (8, 8), (40, 40, 40)))


@pytest.fixture
def state(gl_context, tmp_path):
    from app_state import AppState
    from rendering import Camera, MeshRenderer

    renderer = MeshRenderer(gl_context.ctx, 320, 240)
    app_state = AppState(Camera(), renderer)
    # Redirected before anything draws: draw_sidebar calls state.maybe_autosave(),
    # which would otherwise overwrite the real config/params.json every run.
    app_state.save_path = str(tmp_path / "params.json")
    app_state.load_path = str(tmp_path / "params.json")
    app_state.autosave_enabled = False
    # Build the mesh, as app.py does at startup. Without it the renderer has no
    # pick data, and draw_viewport's geometry closures all return early -- which
    # let a NameError inside one of them pass the suite and still crash the app.
    app_state.rebuild_spline_from_params()
    assert renderer.mesh_pick_data, "fixture must produce pick data for the viewport paths"
    return app_state, renderer


def _draw_one_frame(gui, app_state, renderer, mode, window=None, ref_tex=None):
    """Draw a mode and return the vertex count it produced.

    Renders twice and reports the second. imgui lays a window out on the frame
    it first appears and emits nothing for it until the frame after, so a
    single pass over a freshly-switched mode measures zero. Drawing twice also
    means the count reflects a settled layout, which is what makes it usable as
    a before/after oracle for code movement.
    """
    gui._set_app_mode(app_state, mode)
    for _ in range(2):
        imgui.new_frame()
        gui.draw_menu_bar(app_state)
        gui.draw_sidebar(app_state, renderer, window)
        # Arrives with the sidebar split; called when present so the test keeps
        # covering the same surface before and after.
        draw_mode_windows = getattr(gui, "draw_mode_windows", None)
        if draw_mode_windows is not None:
            draw_mode_windows(app_state, renderer, window)
        if mode != "database":
            # app.py skips the viewport in Database Mode, which takes the window over.
            gui.draw_viewport(app_state, renderer, ref_tex, window)
        imgui.render()
    return int(imgui.get_draw_data().total_vtx_count)


@pytest.mark.parametrize("mode", MODES)
def test_mode_draws_a_complete_frame(imgui_context, state, gl_context, ref_tex, mode):
    """Each mode draws without raising and without unbalancing begin/end."""
    import gui

    app_state, renderer = state
    gl = gl_context
    vertices = _draw_one_frame(gui, app_state, renderer, mode, gl.window, ref_tex)
    assert vertices > 0, f"{mode} mode drew nothing"


def test_every_mode_is_reachable_from_every_other(imgui_context, state, gl_context, ref_tex):
    """Mode switching is exercised in both directions.

    _set_app_mode tears down the previous mode -- it closes the embedded
    scanner, the UR5 controller and the solver. Cycling every ordered pair is
    what covers those teardown paths, which is where a split most easily drops
    a cleanup.
    """
    import gui

    app_state, renderer = state
    gl = gl_context
    for first in MODES:
        for second in MODES:
            if first == second:
                continue
            _draw_one_frame(gui, app_state, renderer, first, gl.window, ref_tex)
            _draw_one_frame(gui, app_state, renderer, second, gl.window, ref_tex)


def test_scan_mode_draws_without_the_mujoco_runtime(imgui_context, state, gl_context, ref_tex, monkeypatch):
    """Scan Mode's UI must not depend on the scan runtime starting.

    Keeps MuJoCo out of the test run, and asserts the separation the split
    relies on: the panel is drawable even when the scanner cannot be built.
    """
    import gui

    def _refuse(*args, **kwargs):
        raise RuntimeError("EmbeddedMujocoScanner is stubbed out in tests")

    monkeypatch.setattr(gui, "EmbeddedMujocoScanner", _refuse)
    app_state, renderer = state
    gl = gl_context
    assert _draw_one_frame(gui, app_state, renderer, "scan", gl.window, ref_tex) > 0
    assert app_state.get("embedded_scanner") is None


def test_ur5_mode_does_not_touch_hardware_when_drawn(imgui_context, state, gl_context, ref_tex):
    """Drawing the UR5 panel must not connect to a robot or open a camera."""
    import gui

    app_state, renderer = state
    gl = gl_context
    _draw_one_frame(gui, app_state, renderer, "ur5", gl.window, ref_tex)
    controller = app_state.get("ur5_controller")
    assert controller is not None
    assert controller.robot.connected is False
    assert controller.camera is None


def test_vertex_counts_are_recorded_per_mode(imgui_context, state, gl_context, ref_tex):
    """Prints the per-mode vertex counts used as the code-movement oracle.

    Run with -s before and after a stage: a pure move leaves every number
    unchanged.
    """
    import gui

    app_state, renderer = state
    gl = gl_context
    counts = {mode: _draw_one_frame(gui, app_state, renderer, mode, gl.window, ref_tex)
              for mode in MODES}
    print("\nvertex counts per mode:")
    for mode, count in counts.items():
        print(f"  {mode:9} {count}")
    assert all(count > 0 for count in counts.values())
