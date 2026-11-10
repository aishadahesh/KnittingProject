"""3D OpenGL wrist camera rendering, lens intrinsics/distortion, and lazy tiled textures."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

import robot_camera
import scan_geometry
import ur5_robot
from scanner_core import TiledFabricTexture

REPEATS = 64


def _knit_tile(width=480, height=105):
    tile = Image.new("RGB", (width, height), (26, 18, 12))
    draw = ImageDraw.Draw(tile)
    for i in range(6):
        x = (i + 0.5) * width / 6
        draw.arc([x - width / 14, 4, x + width / 14, height - 4], 200, 340, fill=(196, 128, 72), width=7)
        draw.arc([x - width / 14, -height // 2, x + width / 14, height // 2], 20, 160, fill=(150, 96, 54), width=7)
    return tile


def test_tiled_fabric_texture_caps_rasterize_size():
    tex = TiledFabricTexture(_knit_tile(), REPEATS, REPEATS)
    assert tex.size == (480 * REPEATS, 105 * REPEATS)
    raster = tex.rasterize(max_dim=1024)
    assert max(raster.size) <= 1024
    arr = np.asarray(raster.convert("L"), dtype=np.float32)
    assert float(np.abs(np.diff(arr, axis=0)).mean() + np.abs(np.diff(arr, axis=1)).mean()) > 4.0


def test_sim_camera_backend_prefers_live_gl_frame_and_falls_back_to_pinhole():
    plan = scan_geometry.build_capture_plan(
        ur5_robot.DEFAULT_HOME_JOINTS,
        pattern="grid",
        grid_rows=2,
        grid_cols=2,
        grid_views_per_station=2,
        fabric_z=0.0,
    )
    pose = list(plan.poses[0])
    backend = robot_camera.SimCameraBackend(
        size=(320, 240),
        pose_provider=lambda: (pose, 42.0, (-0.45, -0.08, 0.0)),
    )
    pinhole = backend.read()
    assert pinhole.shape == (240, 320, 3)

    gl_img = np.full((240, 320, 3), 187, dtype=np.uint8)
    backend.update_gl_frame(gl_img, pose)
    from_gl = backend.read()
    assert np.array_equal(from_gl, gl_img)


def test_render_wrist_camera_respects_calibrated_intrinsics_and_distortion():
    glfw = pytest.importorskip("glfw")
    moderngl = pytest.importorskip("moderngl")
    if not glfw.init():
        pytest.skip("GLFW unavailable")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    glfw.window_hint(glfw.VISIBLE, False)
    window = glfw.create_window(320, 240, "test_wrist", None, None)
    if window is None:
        glfw.terminate()
        pytest.skip("No GL context available")
    glfw.make_context_current(window)
    try:
        from app_state import AppState
        from rendering import Camera, MeshRenderer

        ctx = moderngl.create_context()
        renderer = MeshRenderer(ctx, 320, 240)
        state = AppState(Camera(), renderer)
        state.rebuild_spline_from_params()

        pose = [-0.45, -0.08, 0.22, float(np.pi), 0.0, 0.0]
        base = renderer.render_wrist_camera(
            pose,
            40.0,
            (-0.45, -0.08, 0.0),
            0.18,
            320,
            240,
            state.get_material_uniforms(),
            visible_rows=state.row_visible,
            model_mat=state.current_model_matrix(),
            intrinsics={"fl_x": 380.0, "fl_y": 380.0, "cx": 160.0, "cy": 120.0},
        )
        distorted = renderer.render_wrist_camera(
            pose,
            40.0,
            (-0.45, -0.08, 0.0),
            0.18,
            320,
            240,
            state.get_material_uniforms(),
            visible_rows=state.row_visible,
            model_mat=state.current_model_matrix(),
            intrinsics={"fl_x": 380.0, "fl_y": 380.0, "cx": 160.0, "cy": 120.0, "k1": -0.22, "k2": 0.05},
        )
        assert base.shape == (240, 320, 3)
        assert distorted.shape == (240, 320, 3)
        assert not np.array_equal(base, distorted)
        renderer.close()
        ctx.release()
    finally:
        glfw.destroy_window(window)
        glfw.terminate()
