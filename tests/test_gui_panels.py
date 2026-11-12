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

from pathlib import Path

import numpy as np
import pytest


moderngl = pytest.importorskip("moderngl")
imgui_bundle = pytest.importorskip("imgui_bundle")
glfw = pytest.importorskip("glfw")
from imgui_bundle import imgui  # noqa: E402

MODES = ("scan", "capture", "model", "edit")


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
    io.set_ini_filename(None)
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
    app_state.project_root = str(tmp_path)
    app_state.autosave_enabled = False
    app_state.scanner_rows = 1
    app_state.scanner_cols = 1
    app_state.scanner_pattern_repeat_rows = 1
    app_state.scanner_pattern_repeat_cols = 1
    # Build the mesh, as app.py does at startup. Without it the renderer has no
    # pick data, and draw_viewport's geometry closures all return early -- which
    # let a NameError inside one of them pass the suite and still crash the app.
    app_state.rebuild_spline_from_params()
    assert renderer.mesh_pick_data, "fixture must produce pick data for the viewport paths"
    yield app_state, renderer
    controller = app_state.get('ur5_controller')
    if controller is not None:
        controller.close()
    glfw.make_context_current(gl_context.window)
    renderer.close()


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
        scene_renderer = app_state.active_scene_renderer()
        imgui.new_frame()
        gui.draw_menu_bar(app_state)
        gui.draw_sidebar(app_state, scene_renderer, window)
        # Arrives with the sidebar split; called when present so the test keeps
        # covering the same surface before and after.
        draw_mode_windows = getattr(gui, "draw_mode_windows", None)
        if draw_mode_windows is not None:
            draw_mode_windows(app_state, scene_renderer, window)
        gui.draw_viewport(app_state, scene_renderer, ref_tex, window)
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


def test_scan_mode_draws_without_the_mujoco_runtime(imgui_context, state, gl_context, ref_tex):
    import gui

    app_state, renderer = state
    gl = gl_context
    assert _draw_one_frame(gui, app_state, renderer, "scan", gl.window, ref_tex) > 0
    controller = app_state.get("ur5_controller")
    assert controller is None or controller.robot.connected is False


def test_ur5_mode_does_not_touch_hardware_when_drawn(imgui_context, state, gl_context, ref_tex, monkeypatch):
    """Drawing the Capture panel must not connect to a robot or open a camera."""
    import gui
    import robot_camera

    def refuse_probe(*args, **kwargs):
        pytest.fail("Drawing opened camera discovery")

    monkeypatch.setattr(robot_camera, "list_devices", refuse_probe)

    app_state, renderer = state
    gl = gl_context
    _draw_one_frame(gui, app_state, renderer, "capture", gl.window, ref_tex)
    controller = app_state.get("ur5_controller")
    assert controller is not None
    assert controller.robot.connected is False
    assert controller.camera is None


def test_a_running_scan_reports_its_status_in_the_sidebar(imgui_context, state, gl_context, ref_tex):
    from types import SimpleNamespace

    import gui
    import ur5_scan

    app_state, renderer = state
    _draw_one_frame(gui, app_state, renderer, "capture", gl_context.window, ref_tex)
    controller = app_state.get("ur5_controller")
    progress = ur5_scan.ScanProgress(running=True, status="Moving to view 3/12")
    controller.runner = SimpleNamespace(running=True, paused=False, progress=progress, capture_records=[], session_id="", video_feed=None)
    try:
        _draw_one_frame(gui, app_state, renderer, "scan", gl_context.window, ref_tex)
    finally:
        controller.runner = None
    assert app_state.scanner_status == "Moving to view 3/12"


def test_replacing_geometry_and_closing_renderer_releases_gpu_resources(state):
    app_state, renderer = state
    buffers = list(renderer.mesh_buffers) + [renderer.pt_buffer]
    assert all(buffer is not None for buffer in buffers)
    app_state.rebuild_spline_mesh()
    assert all(isinstance(buffer.mglo, moderngl.InvalidObject) for buffer in buffers)

    resources = list(renderer.mesh_buffers) + [renderer.pt_buffer, renderer.fbo, renderer.bg_vbo]
    renderer.close()
    renderer.close()
    assert all(isinstance(resource.mglo, moderngl.InvalidObject) for resource in resources)


def test_capture_path_draws_a_checked_plan(imgui_context, state, gl_context, ref_tex):
    """The viewpoint preview is drawn from a real solved plan, not just an empty panel.

    Everything the preview reads -- positions, the subject, per-viewpoint verdicts -- comes
    from the solve, so drawing it with no plan present covers none of it.
    """
    import gui
    import scan_geometry
    import ur5_robot

    app_state, renderer = state
    gl = gl_context
    _draw_one_frame(gui, app_state, renderer, "capture", gl.window, ref_tex)
    controller = app_state.get("ur5_controller")
    controller.plan = scan_geometry.build_capture_plan(ur5_robot.DEFAULT_HOME_JOINTS, pattern="spin", count=3, fabric_z=0.20)

    assert _draw_one_frame(gui, app_state, renderer, "capture", gl.window, ref_tex) > 0
    assert controller.robot.connected is False


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


def test_reset_layout_brings_the_controls_back_in_front(imgui_context, state, gl_context, ref_tex, tmp_path, monkeypatch):
    """Knitting Control and 3D View keep fixed positions and sizes across all modes and after Reset Layout."""
    import gui
    import paths

    monkeypatch.setattr(paths, "LAYOUT_INI", tmp_path / "imgui_layout.ini")
    app_state, renderer = state
    gui._reset_layout(app_state)
    for mode in MODES:
        _draw_one_frame(gui, app_state, renderer, mode, gl_context.window, ref_tex)
        ctrl_win = imgui.internal.find_window_by_name("Knitting Control")
        vp_win = imgui.internal.find_window_by_name("3D View")
        assert (ctrl_win.pos.x, ctrl_win.pos.y) == (20, 20)
        assert (ctrl_win.size.x, ctrl_win.size.y) == (360, 820)
        assert (vp_win.pos.x, vp_win.pos.y) == (390, 20)
        assert (vp_win.size.x, vp_win.size.y) == (1190, 820)
    assert not app_state.get("_layout_reset_pending", False)


def test_model_mode_hands_its_repeating_loops_to_edit_mode(imgui_context, state, gl_context, ref_tex):
    """'Open in Edit Mode' makes the fitted repeat Edit Mode's pattern: its tile of stitches and
    openings, and its loop numbers, in Edit Mode, with one undo step to go back."""
    import gui

    app_state, renderer = state
    bitmap = np.ones((7, 3), dtype=np.float32)
    bitmap[2:4, 1] = 0.0
    _draw_one_frame(gui, app_state, renderer, "model", gl_context.window, ref_tex)
    undo_before = len(app_state.undo_stack)
    app_state.model_to_edit = {"parameters": {"stitch_width": 3.03, "dy": 2.23, "stitch_bulge": 0.65, "stitch_z": 0.1,
                                              "radius": 0.3, **{f"loop_height_{r + 1}": 3.9 for r in range(7)}},
                               "bitmap": bitmap}
    imgui.new_frame()
    gui.draw_sidebar(app_state, app_state.active_scene_renderer(), gl_context.window)
    imgui.render()
    assert app_state.get("app_mode") == "edit"
    assert app_state.bitmap.shape == (7, 3) and app_state.bitmap[2, 1] == 0.0
    pidx = app_state._pidx
    assert abs(app_state.params[pidx["stitch_width"]] - 3.03) < 1e-6 and abs(app_state.params[pidx["dy"]] - 2.23) < 1e-6
    assert len(app_state.ctrl_rows) == 7
    assert len(app_state.undo_stack) == undo_before + 1
    assert app_state.get("model_to_edit") is None


def test_edit_mode_builds_a_crochet_fabric_and_takes_one_from_model_mode(imgui_context, state, gl_context, ref_tex):
    """Edit Mode's crochet fabric is drawn from the stitch equations; Model Mode's crochet model opens
    in it with its numbers and grid; Undo goes back to the knit fabric."""
    import gui

    app_state, renderer = state
    _draw_one_frame(gui, app_state, renderer, "edit", gl_context.window, ref_tex)
    app_state.fabric_type = "crochet"
    vertices = _draw_one_frame(gui, app_state, renderer, "edit", gl_context.window, ref_tex)
    assert vertices > 0 and app_state.crochet_active()
    assert len(app_state.renderer.meshes) > 0

    app_state.fabric_type = "knit"
    _draw_one_frame(gui, app_state, renderer, "model", gl_context.window, ref_tex)
    stitches = np.array([[1, 0, 0], [1, 1, 1], [1, 1, 1]], dtype=np.float32)
    app_state.model_to_edit = {"fabric": "crochet", "stitches": stitches,
                               "parameters": {"stitch_width": 3.03, "row_height": 5.2, "wraps": 1, "yarn_radius": 0.4, "twist": 0.75, "lean": 0.0}}
    imgui.new_frame()
    gui.draw_sidebar(app_state, app_state.active_scene_renderer(), gl_context.window)
    imgui.render()
    assert app_state.get("app_mode") == "edit" and app_state.fabric_type == "crochet"
    assert np.array_equal(app_state.crochet_stitches, stitches)
    assert abs(app_state.crochet_parameters().row_height - 5.2) < 1e-9
    app_state.undo_last()
    assert app_state.fabric_type == "knit"


def test_edit_mode_fits_its_crochet_equations_to_a_scan(imgui_context, state, gl_context, ref_tex, monkeypatch):
    """'Fit these equations to the scan' in Edit Mode: the crochet fabric takes the numbers fitted to a
    scan, keeps its repeat's size, and one Undo puts the old numbers back."""
    import time

    import gui
    import scan_fit
    import strand_equations
    from test_lace_lattice import _lace_splat

    splat = _lace_splat(0.0, points=120_000)
    prepared = {"model": None, "splat": splat, "extraction": strand_equations.extract(splat, 1.0, (0.0, 0.0, 1.0)), "mm_per_unit": 1.0}
    fake_model = {"name": "synthetic lace", "ply": Path("synthetic.ply"), "mm_per_unit": 1.0, "psnr": 30.0, "run": None, "dataset": None}
    prepared["model"] = fake_model
    monkeypatch.setattr(scan_fit, "find_models", lambda: [fake_model])
    monkeypatch.setattr(scan_fit, "prepare", lambda model, progress=None: prepared)
    gui._SCAN_FIT.models = None
    gui._SCAN_FIT.prepared = None

    app_state, renderer = state
    app_state.fabric_type = "crochet"
    app_state.crochet_stitches = np.ones((6, 3), dtype=np.float32)
    before = dict(app_state.crochet_params)
    _draw_one_frame(gui, app_state, renderer, "edit", gl_context.window, ref_tex)
    gui._start_scan_fit(app_state, "crochet")
    deadline = time.time() + 120
    while gui._SCAN_FIT.job is not None and time.time() < deadline:
        gui._SCAN_FIT.job[0].join(timeout=1.0)
        _draw_one_frame(gui, app_state, renderer, "edit", gl_context.window, ref_tex)
    assert gui._SCAN_FIT.job is None and gui._SCAN_FIT.result is not None, gui._SCAN_FIT.message
    fitted = app_state.crochet_parameters()
    assert abs(fitted.stitch_width - 3.0) < 0.2 and abs(fitted.row_height - 2.0) < 0.2     # the test lace's 3 x 2 mm stitch
    assert np.asarray(app_state.crochet_stitches).shape == (6, 3)
    assert gui._SCAN_FIT.texture is not None
    app_state.undo_last()
    assert dict(app_state.crochet_params) == before


def test_a_finished_photo_fit_gives_edit_mode_the_learned_stitch(imgui_context, state, gl_context, ref_tex, tmp_path):
    """fit_stitch_model.py's stitch_model.json, read back by scan_fit, becomes Edit Mode's crochet fabric:
    its numbers, its stitch grid and its learned templates; Undo takes it off again."""
    import json
    from types import SimpleNamespace

    import crochet_core
    import gui
    import scan_fit

    p = crochet_core.CrochetParams(stitch_width=1.5, row_height=5.0, yarn_radius=0.5)
    stitches = [[1, 1, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
    path = tmp_path / "stitch_model.json"
    path.write_text(json.dumps({"stitch_width": 1.5, "row_height": 5.0, "yarn_radius": 0.5, "wraps": 1, "twist": 0.75, "lean": 0.0,
                                "stitches": stitches, "templates": crochet_core.hand_templates(p),
                                "scores": {"model_psnr": 13.1, "splat_psnr": 13.9, "before_model_psnr": 7.7}}), encoding="utf-8")
    finished = SimpleNamespace(running=False, lines=["done"], result=lambda: (*scan_fit.load_stitch_model(path), "held-out pictures: 13.10 dB"))
    app_state, renderer = state
    _draw_one_frame(gui, app_state, renderer, "edit", gl_context.window, ref_tex)
    gui._SCAN_FIT.photo_fit = finished
    gui._finish_photo_fit(app_state)
    assert app_state.fabric_type == "crochet" and gui._SCAN_FIT.photo_fit is None
    learned = app_state.crochet_parameters()
    assert learned.templates and abs(learned.row_height - 5.0) < 1e-9
    assert np.array_equal(np.asarray(app_state.crochet_stitches), np.asarray(stitches, dtype=np.float32))
    vertices = _draw_one_frame(gui, app_state, renderer, "edit", gl_context.window, ref_tex)
    assert vertices > 0 and len(app_state.renderer.meshes) > 0
    app_state.undo_last()
    assert not app_state.crochet_parameters().templates
