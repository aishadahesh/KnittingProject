"""Camera poses from the ChArUco board: pictures that see it, and through the arm those that do not."""

import json
import math

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import charuco
import hand_eye

SPEC = dict(charuco.DEFAULT_BOARD)
PX_PER_MM = 12.0
WIDTH, HEIGHT = 640, 480
FX = 1000.0
LENS = {"fx": FX, "fy": FX, "cx": 320.0, "cy": 240.0, "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0, "width": WIDTH, "height": HEIGHT}
INTRINSICS = {"fl_x": FX, "fl_y": FX, "cx": 320.0, "cy": 240.0, "w": WIDTH, "h": HEIGHT}
# The board's middle, where the cloth lies: 70-210 mm across, 50-140 mm down.
CLOTH_MM = (70, 50, 210, 140)
# Where the camera sits on the flange, and where the arm's base is in the board's frame - unknown to
# the export, which fits them per session.
TRUE_CAMERA_FROM_FLANGE = hand_eye.hand_eye_from_params([0.05, -0.03, 0.8], [0.02, -0.01, -0.06])
TRUE_WORLD_FROM_BASE = hand_eye.hand_eye_from_params([math.pi, 0.0, 0.3], [0.30, -0.45, 0.02])
# The pendant's tool-centre setting on the day: the reported pose sits this far along the tool axis.
PENDANT_TCP_M = 0.1085


@pytest.fixture(scope="module")
def page():
    """The board as printed, with a knit-like cloth over its middle."""
    image = cv2.cvtColor(charuco.board_image(SPEC, PX_PER_MM, (SPEC["squares_x"] * SPEC["square_mm"], SPEC["squares_y"] * SPEC["square_mm"])),
                         cv2.COLOR_GRAY2BGR)
    x0, y0, x1, y1 = (int(v * PX_PER_MM) for v in CLOTH_MM)
    rng = np.random.default_rng(0)
    cloth = cv2.GaussianBlur(rng.integers(120, 255, (y1 - y0, x1 - x0, 3)).astype(np.uint8), (9, 9), 0)
    image[y0:y1, x0:x1] = cloth
    return image


def look_at(centre_mm, target_mm, yaw_deg):
    """board_to_camera for a camera at `centre_mm` (board frame: z into the paper, so above it is
    negative z) looking at `target_mm`, rolled by `yaw_deg`."""
    centre, target = np.array(centre_mm, float) / 1000.0, np.array(target_mm, float) / 1000.0
    forward = (target - centre) / np.linalg.norm(target - centre)
    right = Rotation.from_euler("z", yaw_deg, degrees=True).apply([1.0, 0.0, 0.0])
    right = right - (right @ forward) * forward
    right /= np.linalg.norm(right)
    camera_to_board = np.eye(4)
    camera_to_board[:3, :3], camera_to_board[:3, 3] = np.stack([right, np.cross(forward, right), forward], axis=1), centre
    return np.linalg.inv(camera_to_board)


def render(page, board_to_camera, mirrored=False):
    s = 1.0 / (PX_PER_MM * 1000.0)
    from_page = np.array([[s, 0.0, 0.5 * s], [0.0, s, 0.5 * s], [0.0, 0.0, 1.0]])
    matrix = np.array([[FX, 0.0, 320.0], [0.0, FX, 240.0], [0.0, 0.0, 1.0]])
    plane = np.column_stack([board_to_camera[:3, 0], board_to_camera[:3, 1], board_to_camera[:3, 3]])
    picture = cv2.warpPerspective(page, matrix @ plane @ from_page, (WIDTH, HEIGHT), flags=cv2.INTER_AREA, borderValue=(90, 90, 90))
    return picture[:, ::-1].copy() if mirrored else picture


def reported_pose(board_to_camera):
    """What the controller reports for the arm pose that put the camera there."""
    flange_from_base = np.linalg.inv(TRUE_CAMERA_FROM_FLANGE) @ board_to_camera @ TRUE_WORLD_FROM_BASE
    base_to_flange = np.linalg.inv(flange_from_base)
    tool = base_to_flange @ hand_eye.hand_eye_from_params([0.0, 0.0, 0.0], [0.0, 0.0, PENDANT_TCP_M])
    return [*tool[:3, 3], *Rotation.from_matrix(tool[:3, :3]).as_rotvec()]


def scan(tmp_path, page, mirrored=False, session="ur5_test"):
    """A scan as the panel plans one: views round the cloth that see the board, and close-ups of
    the cloth's middle that do not."""
    views = []
    for i, (x, y) in enumerate([(40, 30), (140, 25), (245, 30), (255, 95), (245, 165), (140, 168), (40, 160), (30, 95)]):
        for tilt, yaw in ((0, 20 * i), (18, 45 + 20 * i)):
            offset = 130 * math.tan(math.radians(tilt))
            views.append(look_at((x + offset, y, -130), (x, y, 0), yaw))
    for i, (x, y) in enumerate([(120, 80), (140, 95), (160, 110), (130, 110), (150, 80)]):
        views.append(look_at((x + 6 * (i % 2), y, -50), (x, y, 0), 15 * i))
    captures = []
    for index, board_to_camera in enumerate(views):
        path = tmp_path / f"ur5_{index:04d}.png"
        cv2.imwrite(str(path), render(page, board_to_camera, mirrored))
        captures.append({"path": str(path), "session_id": session, "row": index, "col": 0,
                         "actual_tcp_pose": reported_pose(board_to_camera), "camera_image_width": WIDTH, "camera_image_height": HEIGHT})
    return captures, views


def centres(transforms):
    """Camera centres in the board frame (z into the paper), from transforms.json's world (z up)."""
    return [np.diag([1.0, -1.0, -1.0]) @ np.array(frame["transform_matrix"])[:3, 3] for frame in transforms["frames"]]


def test_a_picture_of_the_board_gives_the_camera_pose(page):
    truth = look_at((120, 40, -140), (100, 30, 0), 30)
    detection = charuco.Detector(SPEC).detect(render(page, truth))
    assert detection is not None and not detection["mirrored"]
    pose = charuco.solve_pose(detection, INTRINSICS)
    assert pose["rms_px"] < 0.5
    assert np.linalg.norm(pose["board_to_camera"][:3, 3] - truth[:3, 3]) < 0.0005
    turned = Rotation.from_matrix(pose["board_to_camera"][:3, :3] @ truth[:3, :3].T).magnitude()
    assert math.degrees(turned) < 0.2


def test_a_mirrored_picture_is_recognised_as_mirrored(page):
    detection = charuco.Detector(SPEC).detect(render(page, look_at((120, 40, -140), (100, 30, 0), 0), mirrored=True))
    assert detection is not None and detection["mirrored"]


def test_close_ups_of_the_cloth_see_no_board(page):
    assert charuco.Detector(SPEC).detect(render(page, look_at((140, 95, -50), (140, 95, 0), 0))) is None


@pytest.mark.parametrize("mirrored", [False, True])
def test_export_places_every_picture_in_the_board_frame(tmp_path, page, mirrored):
    captures, truth = scan(tmp_path, page, mirrored)
    dataset = charuco.export_dataset(captures, tmp_path / "ds", session_id="ur5_test", spec=SPEC, measured_lens=LENS)
    transforms = json.loads(dataset.transforms_path.read_text())

    assert dataset.image_count == len(captures)
    sources = [frame["pose_from"] for frame in transforms["frames"]]
    assert sources.count("arm") == 5 and sources.count("board") == len(captures) - 5
    # Placed through the arm, despite the pendant's tool-centre setting and the unknown mount.
    assert transforms["mount_fit"]["rms_px"] < 1.0
    for frame, centre, board_to_camera in zip(transforms["frames"], centres(transforms), truth):
        true_centre = np.linalg.inv(board_to_camera)[:3, 3]
        if mirrored:
            # A mirrored picture is exported flipped: the same camera, seen through a mirrored scene.
            continue
        assert np.linalg.norm(centre - true_centre) < 0.001, frame["file_path"]
    assert transforms["camera_model"] == "OPENCV"
    assert bool(transforms.get("mirrored")) == mirrored
    assert transforms["ply_file_path"] == "fabric_seed.ply" and (tmp_path / "ds" / "fabric_seed.ply").stat().st_size > 1000


def test_the_lens_is_calibrated_from_the_scan_itself(tmp_path, page):
    captures, _ = scan(tmp_path, page)
    dataset = charuco.export_dataset(captures, tmp_path / "ds", spec=SPEC, measured_lens=None)
    transforms = json.loads(dataset.transforms_path.read_text())
    assert "this scan's own" in transforms["intrinsics_source"]
    assert transforms["fl_x"] == pytest.approx(FX, rel=0.02)


def test_scans_of_several_sessions_are_refused(tmp_path, page):
    captures, _ = scan(tmp_path, page)
    captures[0]["session_id"] = "ur5_other"
    with pytest.raises(ValueError, match="2 scans"):
        charuco.export_dataset(captures, tmp_path / "ds", spec=SPEC, measured_lens=LENS)


def test_a_scan_without_the_board_is_refused(tmp_path):
    blank = np.full((HEIGHT, WIDTH, 3), 200, np.uint8)
    path = tmp_path / "plain.png"
    cv2.imwrite(str(path), blank)
    with pytest.raises(ValueError, match="board is in none"):
        charuco.export_dataset([{"path": str(path), "session_id": "s"}], tmp_path / "ds", spec=SPEC, measured_lens=LENS)


def test_the_board_paper_is_masked_but_cloth_lying_on_it_is_kept():
    # A close picture of part of the board (6 mm squares, about 16 px each), with light-blue cloth
    # lying over its left third and dark blue table beyond the sheet's right edge.
    spec = {"squares_x": 47, "squares_y": 33, "square_mm": 6.0, "marker_mm": 4.5, "dictionary": "DICT_5X5_1000"}
    page = charuco.board_image(spec, px_per_mm=2.7, page_mm=(47 * 6 + 20, 33 * 6 + 20))
    picture = cv2.cvtColor(page[100:580, 330:970], cv2.COLOR_GRAY2BGR).copy()
    sheet_right = (page.shape[1] - int(round(47 * 6 * 2.7))) // 2 + int(round(47 * 6 * 2.7)) - 330    # in the picture
    sheet_bottom = (page.shape[0] - int(round(33 * 6 * 2.7))) // 2 + int(round(33 * 6 * 2.7)) - 100
    picture[:, sheet_right + 12:] = (90, 20, 10)                       # the table (BGR)
    picture[:, :200] = (235, 200, 160)                                  # the cloth over the sheet
    board, markers = charuco.board_paper_mask(picture, spec)
    assert markers >= charuco.MASK_MIN_MARKERS
    assert board[:sheet_bottom - 10, 260:sheet_right - 10].mean() > 0.95  # the open paper: masked
    assert board[:, :190].mean() < 0.02                                 # the cloth on it: kept
    assert board[:, sheet_right + 20:].mean() < 0.02                    # the table beyond the sheet: kept


def test_a_picture_without_the_board_masks_nothing():
    picture = np.full((480, 640, 3), (235, 200, 160), dtype=np.uint8)
    board, markers = charuco.board_paper_mask(picture, {"squares_x": 47, "squares_y": 33, "square_mm": 6.0,
                                                        "marker_mm": 4.5, "dictionary": "DICT_5X5_1000"})
    assert markers == 0 and not board.any()
