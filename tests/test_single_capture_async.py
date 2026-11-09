"""Regression tests for the non-blocking single-image capture pipeline."""

import queue
import threading
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw


from embedded_scanner import EmbeddedMujocoScanner  # noqa: E402
from fabric_scanner import _detect_knitting_patch  # noqa: E402


class _State(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


def test_start_single_capture_only_schedules_work():
    scanner = EmbeddedMujocoScanner.__new__(EmbeddedMujocoScanner)
    scanner._capture_job = None
    scanner.app_state = _State(single_capture_job_seq=0)
    scanner.args = SimpleNamespace(camera_zoom=1.0)
    scanner.plan = SimpleNamespace(station_cells=[(0, 0)])
    scanner.running = True
    scanner.paused = False
    scanner.single_capture_mode = False
    scanner.single_target_active = False
    scanner._camera_preview_dirty = False
    scanner._single_target_index = lambda *_args: (0, 0)
    scanner._single_capture_image_size = lambda: (4096, 3072)

    def should_not_render(*_args, **_kwargs):
        raise AssertionError("the button callback must not render")

    scanner._render_robot_camera_image = should_not_render

    assert scanner.start_single_capture(0, 0, 0, 1.0)
    assert scanner._capture_job["stage"] == "positioning"
    assert scanner._capture_job["image_size"] == (4096, 3072)
    assert scanner.app_state.single_capture_job["status"] == "positioning"


def test_worker_compose_uses_full_requested_size_but_bounds_preview_and_analysis():
    scanner = EmbeddedMujocoScanner.__new__(EmbeddedMujocoScanner)
    scanner.plan = SimpleNamespace(poses=[[0, 0, 0, 0, 0, 0]])
    seen = {}

    def render(*_args, **kwargs):
        seen["image_size"] = kwargs["image_size"]
        return {"full_image": Image.new("RGB", (1800, 1350), (80, 40, 20))}

    scanner._render_robot_camera_image = render
    scanner._fabric_rgb_stats = lambda image: {
        "rgb": [80, 40, 20],
        "pixel_count": image.width * image.height,
        "total_pixels": image.width * image.height,
        "method": "test",
    }
    job = {
        "kind": "single",
        "tcp": [0, 0, 0],
        "target_index": 0,
        "station_id": 0,
        "image_size": (4096, 3072),
        "cancelled": False,
    }

    scanner._capture_compose(job)

    assert seen["image_size"] == (4096, 3072)
    assert job["preview_image"].width == scanner.CAMERA_PREVIEW_MAX_WIDTH
    assert job["analysis_stats"]["total_pixels"] <= scanner.ANALYSIS_MAX_DIMENSION ** 2


def test_high_resolution_patch_detection_maps_proxy_box_to_source_pixels():
    image = Image.new("RGB", (2400, 1800), (18, 20, 24))
    ImageDraw.Draw(image).rectangle((480, 360, 1919, 1439), fill=(190, 70, 30))

    (x0, y0, x1, y1), confidence = _detect_knitting_patch(image)

    assert abs(x0 - 480) < 8
    assert abs(y0 - 360) < 8
    assert abs(x1 - 1920) < 8
    assert abs(y1 - 1440) < 8
    assert confidence > 0.5


def test_background_save_is_atomic_and_signals_completion(tmp_path):
    scanner = EmbeddedMujocoScanner.__new__(EmbeddedMujocoScanner)
    scanner._save_queue = queue.Queue()
    scanner._save_stop = threading.Event()
    scanner.status = ""
    scanner._save_thread = threading.Thread(target=scanner._save_worker, daemon=True)
    scanner._save_thread.start()
    group = {"remaining": 1, "errors": [], "done": threading.Event()}
    output = tmp_path / "capture.png"

    scanner._queue_image_save(Image.new("RGB", (64, 48), (12, 34, 56)), output, save_group=group)
    assert group["done"].wait(3.0)
    scanner._save_stop.set()
    scanner._save_thread.join(timeout=2.0)

    assert group["errors"] == []
    assert output.exists()
    assert Image.open(output).getpixel((0, 0)) == (12, 34, 56)
    assert not list(tmp_path.glob("*.tmp"))


def test_batch_capture_finishes_and_queues_images(tmp_path):
    scanner = EmbeddedMujocoScanner.__new__(EmbeddedMujocoScanner)
    scanner.args = SimpleNamespace(image_dir=str(tmp_path), capture_mode="natural")
    scanner.plan = SimpleNamespace(station_cells=[(0, 0)], view_names=["top"])
    scanner.scanner = SimpleNamespace(CAMERA_CAPTURE_FOCUSED="focused")
    preview = []
    saved = []
    records = []
    scanner._show_robot_camera_image = lambda image, **kwargs: preview.append(image)
    scanner._queue_image_save = lambda image, path: saved.append((image, path))
    scanner._record_capture_analysis = lambda *args, **kwargs: records.append(args)
    image = Image.new("RGB", (64, 48))

    scanner._capture_stage_finish({"result": {"full_image": image}, "target_index": 0, "station_id": 0})

    assert preview == [image]
    assert len(records) == len(saved) == 1
    assert saved[0][0] is image
    assert saved[0][1].parent == tmp_path


def test_image_write_failure_is_preserved_when_cleanup_also_fails(tmp_path, monkeypatch, caplog):
    scanner = EmbeddedMujocoScanner.__new__(EmbeddedMujocoScanner)
    scanner._save_queue = queue.Queue()
    scanner._save_stop = threading.Event()
    scanner._save_stop.set()
    group = {"remaining": 1, "errors": [], "done": threading.Event()}

    def fail_save(*args, **kwargs):
        raise OSError("image write failed")

    def fail_cleanup(*args, **kwargs):
        raise OSError("cleanup failed")

    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    scanner._queue_image_save(SimpleNamespace(save=fail_save), tmp_path / "capture.png", save_group=group)
    scanner._save_worker()

    assert group["errors"] == ["image write failed"]
    assert group["done"].is_set()
    assert scanner._save_queue.unfinished_tasks == 0
    assert "cleanup failed" in caplog.text
