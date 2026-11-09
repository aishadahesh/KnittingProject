"""Pictures picked from the camera's stream: moved-on views are kept, the same view is not, a standing
frame is kept whatever its sharpness, and the recorded video gives the same choice again."""

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageFilter


import robot_camera  # noqa: E402
import ur5_scan  # noqa: E402

CLOTH = Image.fromarray(np.random.default_rng(3).integers(0, 255, (900, 2400, 3), dtype=np.uint8)).filter(ImageFilter.GaussianBlur(1.5))
FPS = 30


def frames(offset_of, count):
    for index in range(1, count + 1):
        left = offset_of(index)
        yield robot_camera.CameraFrame(CLOTH.crop((left, 100, left + 640, 460)), index / FPS, index, "test", False)


def fed(tmp_path, offset_of, count=60, standing_at=()):
    sampler = ur5_scan.VideoSampler(tmp_path)
    for frame in frames(offset_of, count):
        sampler.feed(frame, frame.index in standing_at)
    return sampler


def test_a_camera_that_moves_gives_several_frames_each_with_a_sidecar(tmp_path):
    sampler = fed(tmp_path, lambda index: min(1700, index * 12))

    assert sampler.kept >= 3
    sidecars = sorted(tmp_path.glob("video_*.json"))
    assert len(sidecars) == sampler.kept
    record = json.loads(sidecars[0].read_text())
    assert Path(record["path"]).is_file() and record["video_frame"]


def test_a_camera_that_stays_still_gives_one_frame(tmp_path):
    assert fed(tmp_path, lambda index: 200).kept == 1


def test_a_standing_frame_is_kept_however_pale_and_a_moving_one_must_match_it(tmp_path):
    pale = CLOTH.filter(ImageFilter.GaussianBlur(6))
    sampler = ur5_scan.VideoSampler(tmp_path)
    sampler._standing_sharpness = 1e9
    sampler.feed(robot_camera.CameraFrame(pale.crop((0, 100, 640, 460)), 0.0, 1, "test", False), standing=False)
    sampler.feed(robot_camera.CameraFrame(pale.crop((0, 100, 640, 460)), ur5_scan.VIDEO_WINDOW, 2, "test", False), standing=False)
    assert sampler.kept == 0 and sampler.rejected["blurred"] == 1
    sampler.feed(robot_camera.CameraFrame(pale.crop((900, 100, 1540, 460)), 1.0, 3, "test", False), standing=True)
    assert sampler.kept == 1 and json.loads(next(tmp_path.glob("video_*.json")).read_text())["standing"]


def test_the_recorded_video_gives_the_same_pictures_again(tmp_path):
    pytest.importorskip("cv2")

    sequence = iter(frames(lambda index: min(1700, index * 12), 60))
    finished = threading.Event()

    def latest_frame():
        frame = next(sequence, None)
        if frame is None:
            finished.set()
        elif frame.index == 20:
            live.still()
        return frame

    live = ur5_scan.LiveVideo(SimpleNamespace(latest_frame=latest_frame), ur5_scan.VideoSampler(tmp_path))
    live.start()
    try:
        assert finished.wait(30), live.sampler.error
    finally:
        live.stop()
    chosen_live = live.sampler.kept
    assert (tmp_path / ur5_scan.VIDEO_FILE).is_file() and live.sampler.stills

    again = ur5_scan.extract_from_video(tmp_path)
    assert again.kept == pytest.approx(chosen_live, abs=2)
    assert len(list(tmp_path.glob("video_*.png"))) == again.kept
