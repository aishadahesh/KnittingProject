"""The hand-held capture's live estimates and its "enough" verdict.

A textured plane is photographed through a pinhole from known distances, tilts and sides; the
estimates read back from the homographies must be those, and the targets must count them.
"""

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import free_capture
import robot_camera

WIDTH, HEIGHT, FOV = 1280, 720, 60.0


def _texture(side=2000):
    """Detail at every scale, as a cloth with its edges and surroundings has - so it matches near and far."""
    import cv2

    rng = np.random.default_rng(1)
    total = sum(cv2.GaussianBlur(rng.random((side, side)), (0, 0), sigma) * sigma for sigma in (2, 4, 8, 16, 32))
    grey = ((total - total.min()) / (total.max() - total.min()) * 255).astype(np.uint8)
    return cv2.merge([grey, np.roll(grey, 7, 0), np.roll(grey, 13, 1)])


def _photograph(texture, distance, tilt_deg, side_deg, metres_per_px=0.0002):
    """The plane z=0 seen from `distance` along a ray tilted `tilt_deg` from straight down, from side `side_deg`."""
    import cv2

    tilt, side = math.radians(tilt_deg), math.radians(side_deg)
    centre = distance * np.array([math.sin(tilt) * math.cos(side), math.sin(tilt) * math.sin(side), -math.cos(tilt)])
    forward = -centre / np.linalg.norm(centre)
    up = np.array([0.0, 1.0, 0.0]) if abs(forward[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    rotation = np.stack([right, np.cross(forward, right), forward])
    plane_to_image = free_capture.working_intrinsics(WIDTH, HEIGHT, FOV) @ np.column_stack([rotation[:, 0], rotation[:, 1], -rotation @ centre])
    half = texture.shape[0] / 2.0
    pixels_to_plane = np.array([[metres_per_px, 0, -half * metres_per_px], [0, metres_per_px, -half * metres_per_px], [0, 0, 1]])
    return Image.fromarray(cv2.warpPerspective(texture, plane_to_image @ pixels_to_plane, (WIDTH, HEIGHT)))


class _Frame:
    def __init__(self, image, index):
        self.image, self.index = image, index

    def metadata(self):
        return {}


VIEWS = ((0.30, 0, 0), (0.25, 10, 0), (0.20, 25, 0), (0.18, 30, 90), (0.15, 30, 180), (0.12, 25, 270), (0.10, 20, 45), (0.08, 10, 0))


def test_each_view_is_placed_at_its_distance_tilt_and_side(tmp_path):
    pytest.importorskip("cv2")
    texture = _texture()
    session = free_capture.FreeCaptureSession(tmp_path, FOV)
    # The close views are the texture enlarged, so softer than any real close-up: no floor here.
    floor, free_capture.MIN_SHARPNESS = free_capture.MIN_SHARPNESS, 0.0
    try:
        for index, view in enumerate(VIEWS):
            session._judge(_Frame(_photograph(texture, *view), index))
    finally:
        free_capture.MIN_SHARPNESS = floor
    assert len(session.records) == len(VIEWS), [event.text for event in session.events]
    for record, (distance, tilt, side) in zip(session.records, VIEWS):
        assert record["relative_distance"] == pytest.approx(distance / VIEWS[0][0], abs=0.02)
        assert record["tilt_from_vertical_deg"] == pytest.approx(tilt, abs=1.0)
        if tilt >= free_capture.MIN_SIDE_TILT_DEG:
            assert (record["azimuth_deg"] - side + 180.0) % 360.0 - 180.0 == pytest.approx(0.0, abs=2.0)
        assert Path(record["path"]).exists() and Path(record["path"]).with_suffix(".json").exists()
    assert all(record["linked_to"] is not None for record in session.records[1:])
    assert all(event.kind == "kept" for event in session.events)
    session.close()


def test_a_repeat_a_blurred_frame_and_an_undo(tmp_path):
    pytest.importorskip("cv2")
    from PIL import ImageFilter

    texture = _texture()
    session = free_capture.FreeCaptureSession(tmp_path, FOV)
    for index, view in enumerate(VIEWS[:3]):
        session._judge(_Frame(_photograph(texture, *view), index))
    session._judge(_Frame(_photograph(texture, *VIEWS[0]), 3))
    assert session.last_event.kind == "skipped" and "same view" in session.last_event.text
    steady = _photograph(texture, *VIEWS[3])
    moving = steady.filter(ImageFilter.GaussianBlur(3))
    width = free_capture.BURST_SHARPNESS_WIDTH
    session._judge(_Frame(moving, 4), robot_camera.sharpness(moving, width), robot_camera.sharpness(steady, width))
    assert session.last_event.kind == "skipped" and "blurred" in session.last_event.text
    assert len(session.records) == 3 and session.skipped == 2

    last = Path(session.records[-1]["path"])
    session.discard_last()
    assert len(session.records) == 2 and not last.exists()
    session._judge(_Frame(_photograph(texture, *VIEWS[2]), 5))
    assert len(session.records) == 3 and Path(session.records[-1]["path"]).name != last.name
    session.close()


def test_auto_waits_for_the_first_picture_and_for_the_view_to_move_on(tmp_path):
    session = free_capture.FreeCaptureSession(tmp_path, FOV)
    assert not session.auto_ready(0.0)[0]
    session.records.append({"path": "x"})
    session.overlap = (200, 0.05)
    assert "Move on" in session.auto_ready(0.0)[1]
    session.overlap = (0, 0.0)
    assert "Lost" in session.auto_ready(0.0)[1]
    session.overlap = (200, 0.3)
    assert session.auto_ready(0.0)[0]
    session.close()


def test_enough_needs_every_band_every_side_and_overlap():
    def row(distance, tilt=0.0, side=None):
        return {"relative_distance": distance, "tilt_from_vertical_deg": tilt, "azimuth_deg": side}

    rows = [row(1.0)]
    assert not all(check.done for check in free_capture.readiness(rows))
    for label, _, low, high, need in free_capture.DISTANCE_BANDS:
        middle = low + 0.01 if math.isinf(high) else (low + high) / 2.0
        rows += [row(middle, 30.0, (index * 360.0 / free_capture.SIDE_SECTORS) + 5.0) for index in range(need)]
    assert all(check.done for check in free_capture.readiness(rows))

    straight_down = [dict(r, tilt_from_vertical_deg=5.0) for r in rows]
    assert not next(c for c in free_capture.readiness(straight_down) if c.label.startswith("sides")).done
    unmatched = rows + [row(None)] * len(rows)
    assert not free_capture.readiness(unmatched)[-1].done


def test_the_pose_solve_must_place_most_pictures_precisely():
    assert all(c.done for c in free_capture.solve_checks({"total": 40, "registered": 38, "mean_reprojection_error_px": 0.8}))
    assert not free_capture.solve_checks({"total": 40, "registered": 30, "mean_reprojection_error_px": 0.8})[0].done
    assert not free_capture.solve_checks({"total": 40, "registered": 40, "mean_reprojection_error_px": 2.5})[1].done
