"""3D OpenGL wrist camera rendering, lens intrinsics/distortion, and standalone GL SimCameraBackend."""

import numpy as np
import pytest

import robot_camera
import ur5_robot


def test_distortion_map_produces_valid_remap():
    k = np.array([[380.0, 0.0, 160.0], [0.0, 380.0, 120.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    d = np.array([-0.22, 0.05, 0.0, 0.0, 0.0], dtype=np.float64)
    k_render, (xs, ys) = robot_camera.distortion_map(k, d, (320, 240))
    assert k_render.shape == (3, 3)
    assert xs.shape == (240, 320) and xs.dtype == np.float32
    assert ys.shape == (240, 320) and ys.dtype == np.float32
    assert np.isfinite(k_render).all()
    assert np.isfinite(xs).all() and np.isfinite(ys).all()


def test_sim_camera_backend_renders_3d_gl_scene_and_syncs_meshes():
    moderngl = pytest.importorskip("moderngl")
    from app_state import AppState
    from rendering import Camera, MeshRenderer

    try:
        ctx = moderngl.create_standalone_context()
    except Exception:
        pytest.skip("No standalone GL context available")
    renderer = MeshRenderer(ctx, 320, 240)
    try:
        state = AppState(Camera(), renderer)
        state.rebuild_spline_from_params()
        prepared, colors = state.last_prepared_meshes
        grid_span = state.canonical_grid_span()
        mat_uniforms = state.get_material_uniforms()
    finally:
        renderer.close()
        ctx.release()

    pose = [-0.45, -0.08, 0.22, float(np.pi), 0.0, 0.0]

    scene_box = {
        "joints": list(ur5_robot.DEFAULT_HOME_JOINTS),
        "grid_span": grid_span,
        "fabric_center": (-0.45, -0.08, 0.0),
        "fabric_size": 0.18,
        "mesh_epoch": 0,
        "prepared_meshes": [],
        "colors": [],
        "material_uniforms": mat_uniforms,
    }

    backend = robot_camera.SimCameraBackend(
        320,
        240,
        pose_provider=lambda: (pose, 55.0, (-0.45, -0.08, 0.0)),
        scene_provider=lambda: scene_box,
        intrinsics={"fl_x": 380.0, "fl_y": 380.0, "cx": 160.0, "cy": 120.0},
    )
    try:
        board_only = backend.read()
        assert board_only.shape == (240, 320, 3)
        assert float(board_only.std()) > 5.0

        scene_box["prepared_meshes"] = prepared
        scene_box["colors"] = colors
        scene_box["mesh_epoch"] = 1
        with_meshes = backend.read()
        assert with_meshes.shape == (240, 320, 3)
        assert not np.array_equal(board_only, with_meshes)
    finally:
        backend.close()


def test_render_wrist_camera_respects_calibrated_intrinsics_and_invariant_world_mapping():
    moderngl = pytest.importorskip("moderngl")
    from app_state import AppState
    from rendering import Camera, MeshRenderer

    try:
        ctx = moderngl.create_standalone_context()
    except Exception:
        pytest.skip("No standalone GL context available")
    renderer = MeshRenderer(ctx, 320, 240)
    try:
        state = AppState(Camera(), renderer)
        state.rebuild_spline_from_params()

        center_single, _, upm_single = renderer._scene_world_mapping(
            np.eye(4, dtype=np.float32), (-0.45, -0.08, 0.0), 0.18, grid_span=80.0
        )
        renderer.mesh_pick_data = []
        center_empty, _, upm_empty = renderer._scene_world_mapping(
            np.eye(4, dtype=np.float32), (-0.45, -0.08, 0.0), 0.18, grid_span=80.0
        )
        np.testing.assert_allclose(center_single, np.zeros(3, dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(center_empty, center_single, atol=1e-6)
        assert np.isclose(upm_single, upm_empty)

        state.rebuild_spline_from_params()
        grid_span = state.canonical_grid_span()
        pose = [-0.45, -0.08, 0.22, float(np.pi), 0.0, 0.0]
        base = renderer.render_wrist_camera(
            pose,
            ur5_robot.DEFAULT_HOME_JOINTS,
            width=320,
            height=240,
            fabric_center=(-0.45, -0.08, 0.0),
            fabric_size=0.18,
            material_uniforms=state.get_material_uniforms(),
            intrinsics={"fl_x": 380.0, "fl_y": 380.0, "cx": 160.0, "cy": 120.0},
            grid_span=grid_span,
        )
        distorted = renderer.render_wrist_camera(
            pose,
            ur5_robot.DEFAULT_HOME_JOINTS,
            width=320,
            height=240,
            fabric_center=(-0.45, -0.08, 0.0),
            fabric_size=0.18,
            material_uniforms=state.get_material_uniforms(),
            intrinsics={"fl_x": 380.0, "fl_y": 380.0, "cx": 160.0, "cy": 120.0},
            dist_coeffs=(-0.22, 0.05, 0.0, 0.0, 0.0),
            grid_span=grid_span,
        )
        assert base.shape == (240, 320, 3)
        assert distorted.shape == (240, 320, 3)
        assert not np.array_equal(base, distorted)
    finally:
        renderer.close()
        ctx.release()


def test_render_unit_cell_tile_and_macro_fabric_sheet():
    moderngl = pytest.importorskip("moderngl")
    from app_state import AppState
    from gui import frame_camera_for_mode
    from rendering import Camera, MeshRenderer

    try:
        ctx = moderngl.create_standalone_context()
    except Exception:
        pytest.skip("No standalone GL context available")
    renderer = MeshRenderer(ctx, 320, 240)
    try:
        state = AppState(Camera(), renderer)
        state.rebuild_spline_from_params()
        prepared, colors = state.last_prepared_meshes
        orig_mesh_count = len(renderer.meshes)

        tile = renderer.render_unit_cell_tile(
            prepared,
            colors,
            center_xy=(0.0, 0.0),
            x_period=10.0,
            y_period=10.0,
            material_uniforms=state.get_material_uniforms(),
            tile_h=96,
        )
        assert tile.ndim == 3 and tile.shape[0] == 96 and tile.shape[2] == 3
        assert tile.dtype == np.uint8
        assert len(renderer.meshes) == orig_mesh_count

        grid_span = float(state.canonical_grid_span())
        slots = (
            state.all_swatch_slot_rects()
            if hasattr(state, "all_swatch_slot_rects")
            else [(-0.4 * grid_span, -0.4 * grid_span, 0.4 * grid_span, 0.4 * grid_span)]
        )
        pose = [-0.45, -0.08, 0.22, float(np.pi), 0.0, 0.0]
        board_only = renderer.render_wrist_camera(
            pose,
            ur5_robot.DEFAULT_HOME_JOINTS,
            width=240,
            height=180,
            fabric_center=(-0.45, -0.08, 0.0),
            fabric_size=0.18,
            material_uniforms={**state.get_material_uniforms(), "hide_unit_cell_mesh": True},
            intrinsics={"fl_x": 280.0, "fl_y": 280.0, "cx": 120.0, "cy": 90.0},
            grid_span=grid_span,
        )
        red_tile = np.full((64, 64, 3), (220, 40, 30), dtype=np.uint8)
        red_tile[::8, :, :] = (40, 200, 60)
        with_macro_sheet = renderer.render_wrist_camera(
            pose,
            ur5_robot.DEFAULT_HOME_JOINTS,
            width=240,
            height=180,
            fabric_center=(-0.45, -0.08, 0.0),
            fabric_size=0.18,
            material_uniforms={
                **state.get_material_uniforms(),
                "swatch_tile_images": [red_tile],
                "swatch_tiles_epoch": 1,
                "swatch_slots": slots,
                "swatch_repeats": (24, 24),
                "hide_unit_cell_mesh": True,
            },
            intrinsics={"fl_x": 280.0, "fl_y": 280.0, "cx": 120.0, "cy": 90.0},
            grid_span=grid_span,
        )
        assert with_macro_sheet.shape == (180, 240, 3)
        assert not np.array_equal(board_only, with_macro_sheet)

        frame_camera_for_mode(state, "edit")
        edit_dist = float(state.camera.dist)
        frame_camera_for_mode(state, "scan")
        scan_dist = float(state.camera.dist)
        assert scan_dist > edit_dist
    finally:
        renderer.close()
        ctx.release()
