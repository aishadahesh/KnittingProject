"""Camera poses from a printed ChArUco board under the cloth, instead of from the arm.

Every earlier way of placing the cameras failed on the knit for its own reason. The arm's poses
with the mount calibration were ~11 px off (40 px at worst) - a knit stitch is 20-30 px, so the
splat averaged neighbouring stitches into grey - and they silently broke on a day the pendant
carried another tool-centre setting (the reported TCP sat 108 mm from the plan). COLMAP, matching
the pictures, cannot tell one stitch from the next and placed whole pictures a stitch over.

A ChArUco board is a checkerboard with an ArUco marker in every white square. Each marker has an
ID, so even a corner of the board peeking out from under the cloth says exactly which corners are
seen, and the board is flat with known geometry - so a picture that sees a dozen corners gets its
camera pose to a fraction of a pixel (solvePnP). The board is the world frame: its origin is a
corner of the print, x along the long side, z up out of the paper, metres.

A close-up of the middle of the cloth does not see the board. Those pictures are placed through the
arm, but with a mount calibration fitted per session from that session's own board views
(fit_robot_world): the robot base to board and flange to camera transforms are solved together by
minimising the board corners' reprojection error. Fitted from the same reported poses the session
recorded, whatever tool-centre setting the pendant had that day cancels out.

The lens is calibrated from the session's own board views too, when there are enough of them -
it is then measured at exactly the focus the scan was taken at - otherwise config/camera_intrinsics.json
(scripts/calibrate_lens.py) is used, and with neither the export refuses: a guessed lens is half of
what went wrong before.

scripts/make_charuco_board.py prints the board and writes config/charuco_board.json, which says
which board this is; a session whose pictures see it exports through export_dataset here.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import gaussian_splatting
import hand_eye
from paths import CONFIG_DIR
from persistence import write_json

BOARD_JSON = CONFIG_DIR / "charuco_board.json"
# A4 landscape with 10 mm squares: 28 x 19 squares is 280 x 190 mm, inside a printer's margins. At
# 12-15 cm the camera (30 deg across) sees 6-8 cm of it - a few dozen corners round the cloth.
DEFAULT_BOARD = {"squares_x": 28, "squares_y": 19, "square_mm": 10.0, "marker_mm": 7.5, "dictionary": "DICT_5X5_1000"}

# A picture's pose is trusted from the board only with this many corners seen, fitting this well.
MIN_CORNERS = 8
MAX_POSE_RMS_PX = 1.5
HOMOGRAPHY_TOLERANCE_PX = 4.0
# The lens is calibrated from a session's own board views only when it has this many, each seeing
# at least this many corners; otherwise the measured lens file is used.
MIN_LENS_VIEWS = 12
MIN_LENS_CORNERS = 15
# The mount needs board views at several arm poses to be fitted; with fewer, only the pictures that
# see the board themselves are exported.
MIN_MOUNT_VIEWS = 4
# The fitted mount is good when the arm-placed board corners land this close to where they were seen.
GOOD_MOUNT_RMS_PX = 2.0
# Board frame (OpenCV: y down the print, z into the paper) to world (z up out of the paper).
_BOARD_TO_WORLD = np.diag([1.0, -1.0, -1.0, 1.0])


def load_board_spec(path: Path = BOARD_JSON) -> dict | None:
    """The printed board, or None when none has been printed (no config/charuco_board.json)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return {**DEFAULT_BOARD, **data} if isinstance(data, dict) else None


def save_board_spec(spec: dict, path: Path = BOARD_JSON) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_json(Path(path), spec, indent=2)


def make_board(spec: dict):
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, str(spec["dictionary"])))
    return cv2.aruco.CharucoBoard((int(spec["squares_x"]), int(spec["squares_y"])),
                                  float(spec["square_mm"]) / 1000.0, float(spec["marker_mm"]) / 1000.0, dictionary)


def board_image(spec: dict, px_per_mm: float = 12.0, page_mm=(297.0, 210.0)) -> np.ndarray:
    """The board drawn on a page, at `px_per_mm`: square sizes come out a whole number of pixels so
    the print is exactly to scale at that resolution."""
    board = make_board(spec)
    width_px = int(round(float(spec["squares_x"]) * float(spec["square_mm"]) * px_per_mm))
    height_px = int(round(float(spec["squares_y"]) * float(spec["square_mm"]) * px_per_mm))
    drawn = board.generateImage((width_px, height_px), marginSize=0, borderBits=1)
    page = np.full((int(round(page_mm[1] * px_per_mm)), int(round(page_mm[0] * px_per_mm))), 255, dtype=np.uint8)
    top, left = (page.shape[0] - height_px) // 2, (page.shape[1] - width_px) // 2
    page[top:top + height_px, left:left + width_px] = drawn
    return page


class Detector:
    """Finds the board's corners in a picture - in the picture as it is, or mirrored: a camera
    driver that flips its image left to right (one did, on some days) shows the markers mirrored,
    and a mirrored marker reads as no marker at all."""

    def __init__(self, spec: dict):
        import cv2

        self.spec = spec
        self.board = make_board(spec)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        # A corner is taken with one decoded marker beside it, not OpenCV's two: at the edge of a close
        # picture - where a patch scan sees the board - most corners have one (5 corners became 10).
        # The homography check in detect() still throws out any that land wrong.
        charuco_parameters = cv2.aruco.CharucoParameters()
        charuco_parameters.minMarkers = 1
        self._detector = cv2.aruco.CharucoDetector(self.board, charuco_parameters, parameters)
        # Small, soft markers - 4.5 mm seen from 15 cm by a 640 px fixed-focus camera is ~25 px across,
        # 3-4 px a bit - decoded 3 of ~40 as they were, 14 after this: twice the size, sharpened, and a
        # more forgiving decoder.
        loose = cv2.aruco.DetectorParameters()
        loose.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        # The error correction stays OpenCV's own: raised (0.9) it read 14 markers where 3 were read -
        # but some as the wrong marker, and a wrong marker is a corner put elsewhere on the board.
        loose.adaptiveThreshWinSizeMax = 63
        loose.adaptiveThreshWinSizeStep = 6
        loose.perspectiveRemovePixelPerCell = 8
        loose.perspectiveRemoveIgnoredMarginPerCell = 0.2
        self._small = cv2.aruco.CharucoDetector(self.board, charuco_parameters, loose)

    def _find(self, grey):
        """The board's corners as found plainly, else in the enlarged, sharpened picture (scaled back)."""
        import cv2

        corners, ids, _, _ = self._detector.detectBoard(grey)
        if ids is not None and len(ids) >= MIN_CORNERS:
            return corners, ids
        up = cv2.resize(grey, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        up = cv2.addWeighted(up, 1.8, cv2.GaussianBlur(up, (0, 0), 3), -0.8, 0)
        big_corners, big_ids, _, _ = self._small.detectBoard(up)
        if big_ids is not None and (ids is None or len(big_ids) > len(ids)):
            # Pixel centres: x in the doubled picture is 2x + 0.5.
            return (np.asarray(big_corners, dtype=np.float32) - 0.5) / 2.0, big_ids
        return corners, ids

    def detect(self, image: np.ndarray, min_corners: int = MIN_CORNERS) -> dict | None:
        """{'object': Nx3 board points (m), 'image': Nx2 pixels, 'mirrored': bool} or None when
        fewer than `min_corners` corners are seen."""
        import cv2

        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        for mirrored in (False, True):
            view = grey[:, ::-1].copy() if mirrored else grey
            corners, ids = self._find(view)
            if ids is None or len(ids) < min_corners:
                continue
            object_points, image_points = self.board.matchImagePoints(corners, ids)
            object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
            image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
            ids = np.asarray(ids).reshape(-1)
            # The board is flat, so its corners and their pixels are one homography apart (to within
            # the lens's distortion over a part of the frame). A corner that breaks it was misread.
            if len(ids) >= 6:
                _, inliers = cv2.findHomography(object_points[:, :2], image_points, cv2.RANSAC, HOMOGRAPHY_TOLERANCE_PX)
                if inliers is None:
                    continue
                keep = inliers.ravel().astype(bool)
                object_points, image_points, ids = object_points[keep], image_points[keep], ids[keep]
                if len(ids) < min_corners:
                    continue
            return {"object": object_points, "image": image_points, "ids": ids, "mirrored": mirrored,
                    "size": (view.shape[1], view.shape[0])}
        return None


# The board's printed sheet, for masking it out of a scan: its white border round the squares (m),
# the most a pixel may be coloured and still be the paper (HSV saturation, 0-255; the yarn is blue),
# and the fewest markers that pin the sheet down (one alone extrapolates badly across the frame).
SHEET_BORDER_M = 0.004
PAPER_MAX_SATURATION = 25
DARK_PAPER_VALUE, DARK_PAPER_MAX_SATURATION = 110, 80
MASK_MIN_MARKERS = 2


def board_paper_mask(image: np.ndarray, spec: dict) -> tuple[np.ndarray, int]:
    """Where a picture shows the board's paper: (bool mask, markers read). Empty when too few markers.

    The sheet is found from its markers (a homography from the board's plane), and inside it only the
    neutral paper counts - cloth lying over the sheet's edge stays in. For a scan whose board moved
    under the cloth while the pictures were taken: masked, the board neither places the cameras nor
    is trained into the splat."""
    import cv2

    board = make_board(spec)
    corners_on_board = {int(i): np.asarray(p, dtype=np.float64)[:, :2] for i, p in zip(np.asarray(board.getIds()).ravel(), board.getObjPoints())}
    parameters = cv2.aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMax = 63
    parameters.adaptiveThreshWinSizeStep = 6
    parameters.perspectiveRemovePixelPerCell = 8
    parameters.perspectiveRemoveIgnoredMarginPerCell = 0.2
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Enlarged and sharpened, as Detector reads small markers.
    up = cv2.resize(grey, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    up = cv2.addWeighted(up, 1.8, cv2.GaussianBlur(up, (0, 0), 3), -0.8, 0)
    found, ids, _ = cv2.aruco.ArucoDetector(board.getDictionary(), parameters).detectMarkers(up)
    empty = np.zeros(grey.shape, dtype=bool)
    if ids is None:
        return empty, 0
    on_board, in_picture = [], []
    for marker, marker_id in zip(found, np.asarray(ids).ravel()):
        if int(marker_id) in corners_on_board:
            on_board.append(corners_on_board[int(marker_id)])
            in_picture.append((np.asarray(marker, dtype=np.float64).reshape(4, 2) - 0.5) / 2.0)
    if len(on_board) < MASK_MIN_MARKERS:
        return empty, len(on_board)
    homography, _ = cv2.findHomography(np.concatenate(on_board), np.concatenate(in_picture), cv2.RANSAC, HOMOGRAPHY_TOLERANCE_PX)
    if homography is None:
        return empty, len(on_board)
    # Every pixel carried back onto the board's plane: inside the sheet or not.
    ys, xs = np.mgrid[0:grey.shape[0], 0:grey.shape[1]]
    pixels = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64).reshape(-1, 1, 2)
    plane = cv2.perspectiveTransform(pixels, np.linalg.inv(homography)).reshape(-1, 2)
    width = float(spec["squares_x"]) * float(spec["square_mm"]) / 1000.0
    height = float(spec["squares_y"]) * float(spec["square_mm"]) / 1000.0
    sheet = ((plane[:, 0] > -SHEET_BORDER_M) & (plane[:, 0] < width + SHEET_BORDER_M)
             & (plane[:, 1] > -SHEET_BORDER_M) & (plane[:, 1] < height + SHEET_BORDER_M)).reshape(grey.shape)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # In the black squares the camera's noise reads as colour: dark paper may be more saturated.
    paper = (hsv[..., 1] < PAPER_MAX_SATURATION) | ((hsv[..., 2] < DARK_PAPER_VALUE) & (hsv[..., 1] < DARK_PAPER_MAX_SATURATION))
    mask = cv2.morphologyEx((sheet & paper).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return cv2.dilate(mask, np.ones((5, 5), np.uint8)).astype(bool) & sheet, len(on_board)


def camera_matrix(intrinsics: dict) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV's K and distortion (k1, k2, p1, p2, k3) from transforms.json-style intrinsics."""
    matrix = np.array([[float(intrinsics["fl_x"]), 0.0, float(intrinsics["cx"])],
                       [0.0, float(intrinsics["fl_y"]), float(intrinsics["cy"])], [0.0, 0.0, 1.0]])
    distortion = np.array([float(intrinsics.get(key, 0.0)) for key in ("k1", "k2", "p1", "p2", "k3")])
    return matrix, distortion


def solve_pose(detection: dict, intrinsics: dict) -> dict | None:
    """The board-to-camera transform (OpenCV camera axes) for one detection, and how well it fits."""
    import cv2

    matrix, distortion = camera_matrix(intrinsics)
    ok, rvec, tvec = cv2.solvePnP(detection["object"], detection["image"], matrix, distortion, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(detection["object"], detection["image"], matrix, distortion, rvec, tvec)
    projected, _ = cv2.projectPoints(detection["object"], rvec, tvec, matrix, distortion)
    rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - detection["image"]) ** 2, axis=1))))
    board_to_camera = np.eye(4)
    board_to_camera[:3, :3], board_to_camera[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
    if board_to_camera[2, 3] <= 0.0:
        return None
    return {"board_to_camera": board_to_camera, "rms_px": rms, "corners": int(len(detection["image"]))}


def calibrate_lens(detections: list[dict], size) -> dict:
    """The lens from board views, in the same form scripts/calibrate_lens.py writes."""
    import cv2

    object_points = [d["object"].astype(np.float32) for d in detections]
    image_points = [d["image"].astype(np.float32).reshape(-1, 1, 2) for d in detections]
    error, matrix, distortion, _, _ = cv2.calibrateCamera(object_points, image_points, tuple(int(v) for v in size), None, None)
    k1, k2, p1, p2, k3 = (float(v) for v in distortion.ravel()[:5])
    fx = float(matrix[0, 0])
    return {"fx": fx, "fy": float(matrix[1, 1]), "cx": float(matrix[0, 2]), "cy": float(matrix[1, 2]),
            "k1": k1, "k2": k2, "p1": p1, "p2": p2, "k3": k3, "width": int(size[0]), "height": int(size[1]),
            "fov_degrees": float(np.degrees(2.0 * np.arctan(size[0] / (2.0 * fx)))),
            "reprojection_error_px": float(error), "views": len(detections), "board": "charuco",
            "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def _nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    return u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt


def fit_robot_world(views: list[dict], intrinsics: dict) -> dict:
    """The robot base in the board's frame and the camera on the flange, from board views taken at
    known arm poses. Each view: {'tcp': 4x4 base-to-flange, 'detection': ..., 'board_to_camera': 4x4}."""
    import cv2

    base_to_flange = [np.linalg.inv(v["tcp"]) for v in views]
    board_to_camera = [v["board_to_camera"] for v in views]
    rows = [np.hstack([np.kron(np.eye(3), a[:3, :3]), -np.kron(b[:3, :3].T, np.eye(3))])
            for a, b in zip(board_to_camera, base_to_flange)]
    _, _, vt = np.linalg.svd(np.vstack(rows))
    solution = vt[-1]
    rotation_y = solution[:9].reshape(3, 3, order="F")
    rotation_x = solution[9:].reshape(3, 3, order="F")
    scale = np.cbrt(np.linalg.det(rotation_x)) or 1.0
    rotation_x, rotation_y = _nearest_rotation(rotation_x / scale), _nearest_rotation(rotation_y / scale)
    lhs = np.vstack([np.hstack([a[:3, :3], -np.eye(3)]) for a in board_to_camera])
    rhs = np.concatenate([rotation_x @ b[:3, 3] - a[:3, 3] for a, b in zip(board_to_camera, base_to_flange)])
    translations = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    world_from_base, camera_from_flange = np.eye(4), np.eye(4)
    world_from_base[:3, :3], world_from_base[:3, 3] = rotation_y, translations[:3]
    camera_from_flange[:3, :3], camera_from_flange[:3, 3] = rotation_x, translations[3:]
    matrix, distortion = camera_matrix(intrinsics)

    def residuals(params):
        world_base = hand_eye.hand_eye_from_params(params[:3], params[3:6])
        camera_flange = hand_eye.hand_eye_from_params(params[6:9], params[9:12])
        base_world = np.linalg.inv(world_base)
        out = []
        for view, flange_base in zip(views, base_to_flange):
            b2c = camera_flange @ flange_base @ base_world
            rvec = cv2.Rodrigues(b2c[:3, :3])[0]
            projected, _ = cv2.projectPoints(view["detection"]["object"], rvec, b2c[:3, 3], matrix, distortion)
            out.append((projected.reshape(-1, 2) - view["detection"]["image"]).ravel())
        return np.concatenate(out)

    start = np.concatenate([hand_eye.params_from_matrix(world_from_base), hand_eye.params_from_matrix(camera_from_flange)])
    fitted = least_squares(residuals, start, method="lm", x_scale="jac")
    error = residuals(fitted.x).reshape(-1, 2)
    per_view, offset = [], 0
    for view in views:
        count = len(view["detection"]["image"])
        per_view.append(float(np.sqrt(np.mean(np.sum(error[offset:offset + count] ** 2, axis=1)))))
        offset += count
    return {"world_from_base": hand_eye.hand_eye_from_params(fitted.x[:3], fitted.x[3:6]),
            "camera_from_flange": hand_eye.hand_eye_from_params(fitted.x[6:9], fitted.x[9:12]),
            "rms_px": float(np.sqrt(np.mean(np.sum(error ** 2, axis=1)))), "max_view_px": float(max(per_view)),
            "views": len(views)}


def arm_tcp(capture: dict) -> np.ndarray | None:
    """The flange pose the session recorded for a picture - the reported one where there is one."""
    pose = capture.get("actual_tcp_pose") or capture.get("target_position")
    if pose is None or len(pose) < 6:
        return None
    return hand_eye.tcp_matrix(pose)


def _read(path: Path):
    import cv2

    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def write_coloured_seed(path: Path, frames: list[dict], images: dict, intrinsics: dict, spacing_m: float = 0.0004,
                        heights_m=(0.001, 0.003, 0.005)) -> int:
    """Seed points on thin layers over the board where the cameras looked, each coloured with the
    median of what the pictures show there - the cloth lies on the board, a few millimetres up."""
    import cv2

    matrix, distortion = camera_matrix(intrinsics)
    footprints = []
    for frame in frames:
        board_to_camera = frame["board_to_camera"]
        corners_px = np.array([[0, 0], [intrinsics["w"], 0], [intrinsics["w"], intrinsics["h"]], [0, intrinsics["h"]]], dtype=float)
        rays = np.c_[(corners_px - [intrinsics["cx"], intrinsics["cy"]]) / [intrinsics["fl_x"], intrinsics["fl_y"]], np.ones(4)]
        camera_to_board = np.linalg.inv(board_to_camera)
        centre, directions = camera_to_board[:3, 3], rays @ camera_to_board[:3, :3].T
        hits = [centre - centre[2] / d[2] * d for d in directions if abs(d[2]) > 1e-6 and -centre[2] / d[2] > 0]
        footprints.extend(hits)
    if not footprints:
        return 0
    footprints = np.array(footprints)
    low, high = footprints[:, :2].min(axis=0), footprints[:, :2].max(axis=0)
    area_points = (high - low) / spacing_m
    spacing = spacing_m * max(1.0, math.sqrt(area_points[0] * area_points[1] * len(heights_m) / gaussian_splatting.SEED_MAX_POINTS))
    xs, ys = np.arange(low[0], high[0], spacing), np.arange(low[1], high[1], spacing)
    grid = np.array([[x, y, -h] for h in heights_m for y in ys for x in xs])     # board z points into the paper
    counts = np.zeros(len(grid))
    sums = np.zeros((len(grid), 3))
    for frame in frames:
        image = images.get(frame["file_path"])
        if image is None:
            continue
        board_to_camera = frame["board_to_camera"]
        in_camera = grid @ board_to_camera[:3, :3].T + board_to_camera[:3, 3]
        front = in_camera[:, 2] > 1e-3
        rvec = cv2.Rodrigues(board_to_camera[:3, :3])[0]
        projected, _ = cv2.projectPoints(grid, rvec, board_to_camera[:3, 3], matrix, distortion)
        projected = projected.reshape(-1, 2)
        inside = front & (projected[:, 0] >= 0) & (projected[:, 1] >= 0) & (projected[:, 0] < image.shape[1] - 1) & (projected[:, 1] < image.shape[0] - 1)
        pixels = projected[inside].astype(int)
        sums[inside] += image[pixels[:, 1], pixels[:, 0], ::-1]
        counts[inside] += 1
    seen = counts > 0
    if not seen.any():
        return 0
    points = (_BOARD_TO_WORLD[:3, :3] @ grid[seen].T).T
    rgb = (sums[seen] / counts[seen, None]).clip(0, 255).astype(np.uint8)
    table = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    table["x"], table["y"], table["z"] = points[:, 0], points[:, 1], points[:, 2]
    table["red"], table["green"], table["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(table)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with Path(path).open("wb") as handle:
        handle.write(header.encode("ascii"))
        table.tofile(handle)
    return len(table)


def survey(captures, spec: dict | None = None, sample: int = 6) -> dict:
    """A quick look at a few of a session's pictures: how many see the board. For the panel, which
    offers the board's poses only for a session that was scanned over it."""
    spec = spec or load_board_spec()
    if spec is None:
        return {"board": False, "seen": 0, "looked": 0}
    detector = Detector(spec)
    picked = list(captures)[:: max(1, len(captures) // sample)][:sample]
    seen = looked = 0
    for capture in picked:
        image = _read(Path(str(capture.get("path") or capture.get("image_path") or "")))
        if image is None:
            continue
        looked += 1
        seen += detector.detect(image) is not None
    return {"board": True, "seen": seen, "looked": looked}


def export_dataset(captures, output_dir: str | Path, *, session_id: str = "", spec: dict | None = None,
                   measured_lens: dict | None = None, equalise_exposure: bool = True,
                   allow_several_sessions: bool = False) -> gaussian_splatting.SplatDataset:
    """A splatting dataset whose cameras are placed by the ChArUco board.

    Pictures that see the board are placed by it; the rest of a session through the arm, with the
    mount fitted from that session's board views. Several sessions are refused unless allowed: the
    board makes their frames agree, but only if the cloth was not touched in between."""
    spec = spec or load_board_spec()
    if spec is None:
        raise ValueError("no ChArUco board is set up: print one with scripts/make_charuco_board.py first")
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    colmap_dir = output_dir / "sparse" / "0"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    colmap_dir.mkdir(parents=True, exist_ok=True)
    for stale in (gaussian_splatting.SEED_PLY_NAME, "sparse_pc.ply"):
        if (output_dir / stale).exists():
            (output_dir / stale).unlink()

    sessions = sorted({str(c.get("session_id") or c.get("combined_from") or "") for c in captures})
    if len(sessions) > 1 and not allow_several_sessions:
        raise ValueError(f"these pictures come from {len(sessions)} scans ({', '.join(s or '?' for s in sessions)}); a splat needs "
                         "one scan of an untouched cloth - export one session")
    warnings: list[str] = []
    detector = Detector(spec)
    looked = []
    for capture in captures:
        source = Path(str(capture.get("path") or capture.get("image_path") or ""))
        image = _read(source)
        if image is None:
            warnings.append(f"missing image: {source.name or '(no path)'}")
            continue
        looked.append({"capture": capture, "source": source, "image": image, "detection": detector.detect(image)})
    seen = [item for item in looked if item["detection"] is not None]
    if not seen:
        raise ValueError(f"the board is in none of the {len(looked)} pictures: scan with the cloth on the printed board, "
                         "high enough (12-15 cm) that the board shows round it")
    mirrored_votes = [item["detection"]["mirrored"] for item in seen]
    mirrored = sum(mirrored_votes) * 2 > len(mirrored_votes)
    if any(v != mirrored for v in mirrored_votes):
        warnings.append("some pictures are mirrored and some are not; the odd ones are left out")
        seen = [item for item in seen if item["detection"]["mirrored"] == mirrored]
    if mirrored:
        warnings.append("the camera's pictures are mirrored left to right; they are exported flipped")
        for item in looked:
            item["image"] = item["image"][:, ::-1].copy()
    height, width = looked[0]["image"].shape[:2]

    lens_views = [item["detection"] for item in seen if len(item["detection"]["image"]) >= MIN_LENS_CORNERS]
    if len(lens_views) >= MIN_LENS_VIEWS:
        measured = calibrate_lens(lens_views, (width, height))
        lens_source = (f"calibrated from this scan's own {len(lens_views)} board views "
                       f"({measured['reprojection_error_px']:.2f} px), at the focus it was taken at")
        if measured["reprojection_error_px"] > 1.0:
            warnings.append(f"the lens calibrated from the board views fits to {measured['reprojection_error_px']:.2f} px (want under 1): "
                            "was the board flat, and focus fixed?")
    else:
        measured = measured_lens if measured_lens is not None else hand_eye.load_intrinsics()
        if measured is None:
            raise ValueError(f"the lens is not measured: only {len(lens_views)} pictures see {MIN_LENS_CORNERS}+ board corners "
                             f"(the scan can calibrate it itself from {MIN_LENS_VIEWS}), and there is no config/camera_intrinsics.json - "
                             "run scripts/calibrate_lens.py, or scan with more of the board in view")
        lens_source = "measured lens (config/camera_intrinsics.json)"
        if mirrored:
            measured = {**measured, "cx": float(measured["width"]) - float(measured["cx"]), "p2": -float(measured.get("p2", 0.0))}
    intrinsics = hand_eye.intrinsics_for(width, height, measured)

    placed: dict[int, dict] = {}
    rejected = 0
    for index, item in enumerate(looked):
        if item["detection"] is None or item["detection"]["mirrored"] != mirrored:
            continue
        pose = solve_pose(item["detection"], intrinsics)
        if pose is None or pose["rms_px"] > MAX_POSE_RMS_PX:
            rejected += 1
            continue
        placed[index] = {"board_to_camera": pose["board_to_camera"], "from": "board", "rms_px": pose["rms_px"], "corners": pose["corners"]}
    if rejected:
        warnings.append(f"{rejected} pictures saw the board but fitted it worse than {MAX_POSE_RMS_PX} px and were placed by the arm instead")

    mount = None
    arm_views = [{"tcp": arm_tcp(looked[i]["capture"]), "detection": looked[i]["detection"], "board_to_camera": p["board_to_camera"]}
                 for i, p in placed.items() if arm_tcp(looked[i]["capture"]) is not None]
    unplaced = [i for i in range(len(looked)) if i not in placed]
    if unplaced:
        if len(arm_views) >= MIN_MOUNT_VIEWS:
            mount = fit_robot_world(arm_views, intrinsics)
            if mount["rms_px"] > GOOD_MOUNT_RMS_PX:
                warnings.append(f"the arm places the board views to {mount['rms_px']:.1f} px (worst {mount['max_view_px']:.1f}); pictures "
                                "without the board are placed that well - over 2 px blurs fine stitches")
            for index in unplaced:
                tcp = arm_tcp(looked[index]["capture"])
                if tcp is None:
                    warnings.append(f"no arm pose for {looked[index]['source'].name}; left out")
                    continue
                placed[index] = {
                    "board_to_camera": mount["camera_from_flange"] @ np.linalg.inv(tcp) @ np.linalg.inv(mount["world_from_base"]),
                    "from": "arm",
                }
        else:
            warnings.append(f"{len(unplaced)} pictures without the board are left out: placing them through the arm needs the board "
                            f"in at least {MIN_MOUNT_VIEWS} pictures with an arm pose ({len(arm_views)} here)")

    gains = gaussian_splatting.exposure_gains([item["source"] for item in looked]) if equalise_exposure else {}
    frames, colmap_images, seed_frames, seed_images = [], [], [], {}
    from PIL import Image

    for index in sorted(placed):
        item, place = looked[index], placed[index]
        name = item["source"].name
        relative = f"images/{name}"
        rgb = item["image"][:, :, ::-1].astype(float) * np.asarray(gains.get(str(item["source"]), np.ones(3)))[None, None, :]
        Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(images_dir / name)
        camera_to_world = _BOARD_TO_WORLD @ np.linalg.inv(place["board_to_camera"])
        frame = {"file_path": relative, "transform_matrix": (camera_to_world @ gaussian_splatting._CV_TO_GL).tolist(),
                 "colmap_im_id": index + 1, "pose_from": place["from"],
                 "row": int(item["capture"].get("row", 0)), "col": int(item["capture"].get("col", 0)),
                 "angle": str(item["capture"].get("angle", ""))}
        if place["from"] == "board":
            frame.update({"board_corners": place["corners"], "board_rms_px": round(place["rms_px"], 3)})
        frames.append(frame)
        seed_frames.append({"file_path": relative, "board_to_camera": place["board_to_camera"]})
        seed_images[relative] = item["image"]
        w2c = np.linalg.inv(camera_to_world)
        quat = Rotation.from_matrix(w2c[:3, :3]).as_quat()
        colmap_images.append(f"{index + 1} {quat[3]:.9f} {quat[0]:.9f} {quat[1]:.9f} {quat[2]:.9f} "
                             f"{w2c[0, 3]:.9f} {w2c[1, 3]:.9f} {w2c[2, 3]:.9f} 1 {name}")

    seed = write_coloured_seed(output_dir / gaussian_splatting.SEED_PLY_NAME, seed_frames, seed_images, intrinsics) if frames else 0
    from_board = sum(1 for f in frames if f["pose_from"] == "board")
    board_rms = [f["board_rms_px"] for f in frames if f["pose_from"] == "board"]
    transforms = {
        **intrinsics,
        "camera_model": "OPENCV",
        "session_id": str(session_id),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pose_source": f"ChArUco board ({from_board} pictures placed by the board, {len(frames) - from_board} through the arm "
                       "with the mount fitted to this scan's board views)",
        "charuco_board": spec,
        "board_pose_rms_px": float(np.median(board_rms)) if board_rms else None,
        **({"mount_fit": {"rms_px": mount["rms_px"], "max_view_px": mount["max_view_px"], "views": mount["views"],
                          "camera_from_flange": mount["camera_from_flange"].tolist(),
                          "world_from_base": (_BOARD_TO_WORLD @ mount["world_from_base"]).tolist()}} if mount else {}),
        "intrinsics_source": lens_source,
        "lens": {k: v for k, v in measured.items() if k != "pictures_used"},
        **({"mirrored": True} if mirrored else {}),
        **({"ply_file_path": gaussian_splatting.SEED_PLY_NAME} if seed else {}),
        **({"exposure_equalised": True} if gains else {}),
        "frames": frames,
    }
    transforms_path = output_dir / "transforms.json"
    write_json(transforms_path, transforms, indent=2)
    gaussian_splatting._write_colmap_model(colmap_dir, colmap_images, intrinsics)
    return gaussian_splatting.SplatDataset(root=output_dir, image_count=len(frames), transforms_path=transforms_path,
                                           colmap_dir=colmap_dir, session_id=str(session_id), warnings=warnings, offered=len(captures))
