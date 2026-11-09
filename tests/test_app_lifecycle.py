import subprocess
import sys
import textwrap


def test_frame_failure_closes_window_renderers_and_simulation(tmp_path):
    code = textwrap.dedent('''
        import sys
        import threading
        from pathlib import Path
        import app

        root = Path(sys.argv[1])
        app.paths.PARAMS_JSON = root / 'params.json'
        app.paths.LAYOUT_INI = root / 'layout.ini'
        storage = app.ScannerStorage
        app.ScannerStorage = lambda _: storage(root)
        create_window = app.glfw.create_window

        def hidden_window(*args):
            app.glfw.window_hint(app.glfw.VISIBLE, False)
            return create_window(*args)

        app.glfw.create_window = hidden_window
        renderers = []

        def fail_frame(state, renderer, window):
            renderers.extend([state.renderer, state.orbit_renderer, state.puzzle_renderer])
            raise RuntimeError('injected frame failure')

        app.draw_sidebar = fail_frame
        try:
            app.main()
        except RuntimeError as exc:
            if str(exc) in ('GLFW init failed', 'GLFW window creation failed'):
                sys.exit(77)
            assert str(exc) == 'injected frame failure', str(exc)
        else:
            raise AssertionError('Frame failure was suppressed')
        assert renderers and all(renderer.fbo is None for renderer in renderers)
        assert app.imgui.get_current_context() is None
        assert not any(thread.name == 'yarn-sim' for thread in threading.enumerate())
    ''')
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], capture_output=True, text=True, timeout=90)
    if result.returncode == 77:
        import pytest
        pytest.skip('No GL context available')
    assert result.returncode == 0, result.stdout + result.stderr
