"""Re-locating a nudged carton from partial pictures aligned with its photo."""

import cv2
import numpy as np
import pytest

import fabric_relocate
import workcell

RES = fabric_relocate.RES_M
X0, Y_TOP = -0.30, 0.70


def _photo():
    """A carton-like photo: panels of distinct colours with fine random texture, 0.50 x 0.30 m."""
    rng = np.random.default_rng(3)
    h, w = int(0.30 / RES), int(0.50 / RES)
    image = np.zeros((h, w, 3), np.uint8)
    for k, x in enumerate(range(0, w, 28)):
        image[:, x:x + 28] = rng.integers(60, 200, 3)
        image[h // 2:, x:x + 28] = rng.integers(60, 200, 3)
    noise = cv2.GaussianBlur(rng.normal(0, 25, (h, w)).astype(np.float32), (0, 0), 1.5)
    return np.clip(image + noise[..., None], 0, 255).astype(np.uint8)


def _views(photo, turn_deg, shift, spots, size=0.13):
    """Top-down pictures of the carton moved by (turn_deg, shift), in the photographed frame (prior: unmoved)."""
    turn = fabric_relocate.turn_matrix(turn_deg)
    views = []
    for cx, cy in spots:
        n = int(size / RES)
        gx, gy = np.meshgrid(cx - size / 2 + np.arange(n) * RES, cy + size / 2 - np.arange(n) * RES)
        then = (np.stack([gx.ravel(), gy.ravel()], 1) - shift) @ turn
        u = ((then[:, 0] - X0) / RES).reshape(gx.shape).astype(np.float32)
        v = ((Y_TOP - then[:, 1]) / RES).reshape(gx.shape).astype(np.float32)
        views.append((cv2.remap(photo, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 90, 200)),
                      float(cx - size / 2), float(cy + size / 2)))
    return views


SPOTS = [(x, y) for y in (0.48, 0.62) for x in (-0.22, -0.08, 0.06, 0.18)]


def test_an_unmoved_carton_is_found_where_it_was():
    photo = _photo()
    turn_deg, shift, correlation, _ = fabric_relocate.locate(_views(photo, 0.0, np.zeros(2), SPOTS), (photo, X0, Y_TOP), (0.0, [0, 0]))
    assert turn_deg == pytest.approx(0.0, abs=0.1)
    assert shift == pytest.approx([0.0, 0.0], abs=0.001)
    assert correlation > fabric_relocate.MIN_CORRELATION


def test_a_nudged_and_turned_carton_is_re_located():
    photo = _photo()
    true_turn, true_shift = 2.0, np.array([-0.045, 0.030])
    turn_deg, shift, _, _ = fabric_relocate.locate(_views(photo, true_turn, true_shift, SPOTS), (photo, X0, Y_TOP), (0.0, [0, 0]))
    assert turn_deg == pytest.approx(true_turn, abs=0.2)
    middle = np.array([X0 + 0.25, Y_TOP - 0.15])
    moved = middle @ fabric_relocate.turn_matrix(turn_deg).T + shift
    assert moved == pytest.approx(middle @ fabric_relocate.turn_matrix(true_turn).T + true_shift, abs=0.002)


def test_too_little_of_the_carton_seen_is_refused():
    photo = _photo()
    with pytest.raises(ValueError, match="cover"):
        fabric_relocate.locate(_views(photo, 0.0, np.zeros(2), SPOTS[:1], size=0.05), (photo, X0, Y_TOP), (0.0, [0, 0]))


def test_the_parts_follow_the_carton_pose(monkeypatch):
    before = workcell.parts()
    if not before:
        pytest.skip("no fabric parts mapped")
    turn = fabric_relocate.turn_matrix(3.0)
    monkeypatch.setattr(workcell, "carton_pose", lambda: (turn, np.array([0.01, -0.02])))
    after = workcell.parts()
    expected = np.asarray(before[0]["corners"]) @ turn.T + [0.01, -0.02]
    assert np.asarray(after[0]["corners"]) == pytest.approx(expected)
