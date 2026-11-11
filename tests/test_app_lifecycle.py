import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize('fail_frame', [False, True], ids=['saved_model_frames', 'frame_failure'])
def test_startup_and_shutdown_with_a_saved_model(tmp_path, fail_frame):
    code = textwrap.dedent('''
        import sys
        import json
        import threading
        from pathlib import Path
        import app

        root = Path(sys.argv[1])
        app.paths.PARAMS_JSON = root / 'params.json'
        app.paths.LAYOUT_INI = root / 'layout.ini'
        saved = json.loads(app.paths.INITIAL_PARAMS_JSON.read_text())
        saved['gui_state'].pop('bitmap_size', None)
        saved['gui_state']['row_visible'] = [True, False, True]
        saved['gui_state']['app_mode'] = 'edit'
        app.paths.PARAMS_JSON.write_text(json.dumps(saved))
        fail_frame = sys.argv[2] == 'True'
        storage = app.ScannerStorage
        app.ScannerStorage = lambda _: storage(root)
        create_window = app.glfw.create_window

        def hidden_window(*args, **kwargs):
            app.glfw.window_hint(app.glfw.VISIBLE, False)
            return create_window(*args, **kwargs)

        app.glfw.create_window = hidden_window
        renderers = []
        draw_sidebar = app.draw_sidebar

        def draw_frame(state, renderer, window):
            if not renderers:
                renderers.append(state.renderer)
                assert state.ur5_controller is not None
                assert state.ur5_controller.robot.connected
                assert state.ur5_controller.camera is not None and state.ur5_controller.camera.running
            if fail_frame:
                raise RuntimeError('injected frame failure')
            draw_sidebar(state, renderer, window)

        app.draw_sidebar = draw_frame
        swap_buffers = app.glfw.swap_buffers
        frames = []

        def finish_frame(window):
            swap_buffers(window)
            frames.append(True)
            if len(frames) == 2:
                app.glfw.set_window_should_close(window, True)

        app.glfw.swap_buffers = finish_frame
        try:
            app.main()
        except RuntimeError as exc:
            if str(exc) in ('GLFW init failed', 'GLFW window creation failed'):
                sys.exit(77)
            assert fail_frame and str(exc) == 'injected frame failure', str(exc)
        else:
            assert not fail_frame, 'Frame failure was suppressed'
            assert len(frames) == 2
        assert renderers and all(renderer.fbo is None for renderer in renderers)
        assert app.imgui.get_current_context() is None
        assert not any(thread.name == 'yarn-sim' for thread in threading.enumerate())
    ''')
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path), str(fail_frame)], capture_output=True, text=True, timeout=90)
    if result.returncode == 77:
        pytest.skip('No GL context available')
    assert result.returncode == 0, result.stdout + result.stderr
