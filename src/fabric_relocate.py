"""Where a nudged carton lies now, from a few pictures aligned with its reference photo.

The reference photo shows the carton where it was photographed. Each picture, taken at a known arm pose,
is projected onto the carton's top and resampled in the photo's frame as the last known carton pose
(`prior`: now = turn @ photographed + shift) predicts it, and the pictures are pasted into one image.
Whatever the carton moved since is a shift and turn between that image and the photo, found coarsely by
template matching, then exactly by ECC alignment over the area the pictures cover. The whole carton is
matched at once: a small patch of this fabric repeats every panel and every few knit rows, the carton's
outline and its run of panel colours do not.

Images are BGR uint8. Top-down images are in the photo's frame: column = (x - x0) / res, row = (y_top - y) / res.
"""

from __future__ import annotations

import cv2
import numpy as np

RES_M = 0.001
COARSE_M = 0.004
SEARCH_M = 0.12
MIN_CORRELATION = 0.6
MIN_COVERED = 0.15
EDGE_INSET_M = 0.01
COARSE_TURN_DEG = 6.0
COARSE_TURN_STEP_DEG = 1.0
ECC_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-7)
ECC_SCALES = (0.5, 1.0)
CONTRAST = cv2.createCLAHE(2.0, (8, 8))


def turn_matrix(turn_deg: float) -> np.ndarray:
    angle = np.radians(turn_deg)
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def project_view(image, camera_base, camera_matrix, distortion, slope, top_z, prior, res_m=RES_M):
    """The picture resampled as a top-down image in the photo's frame, where `prior` (turn_deg, shift)
    says the carton lies: (image, x0, y_top), black where the camera did not see."""
    h, w = image.shape[:2]
    turn, shift = turn_matrix(prior[0]), np.asarray(prior[1], dtype=float)
    base_camera = np.linalg.inv(camera_base)
    rays = cv2.undistortPoints(np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], np.float32), camera_matrix, distortion).reshape(-1, 2)
    directions = np.c_[rays, np.ones(4)] @ base_camera[:3, :3].T
    origin = base_camera[:3, 3]
    depth = origin[2] - (slope[0] * origin[0] + slope[1] * origin[1] + top_z)
    footprint = (origin[:2] + directions[:, :2] * (-depth / directions[:, 2])[:, None] - shift) @ turn
    x0, y_top = footprint[:, 0].min(), footprint[:, 1].max()
    gx, gy = np.meshgrid(x0 + np.arange(int((footprint[:, 0].max() - x0) / res_m)) * res_m,
                         y_top - np.arange(int((y_top - footprint[:, 1].min()) / res_m)) * res_m)
    now = np.stack([gx.ravel(), gy.ravel()], 1) @ turn.T + shift
    world = np.c_[now, slope[0] * now[:, 0] + slope[1] * now[:, 1] + top_z]
    uv, _ = cv2.projectPoints(world, cv2.Rodrigues(camera_base[:3, :3])[0], camera_base[:3, 3], camera_matrix, distortion)
    u, v = (uv[:, 0, i].reshape(gx.shape).astype(np.float32) for i in (0, 1))
    return cv2.remap(image, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), float(x0), float(y_top)


def _grey(image):
    return cv2.GaussianBlur(CONTRAST.apply(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)), (0, 0), 1.0).astype(np.float32)


def paste(views, reference, res_m=RES_M, search_m=SEARCH_M):
    """The projected views in one image covering the reference photo plus `search_m` all round:
    (image, covered mask, x0, y_top)."""
    ref_image, rx0, ry_top = reference
    margin = int(search_m / res_m)
    canvas = np.zeros((ref_image.shape[0] + 2 * margin, ref_image.shape[1] + 2 * margin, 3), np.uint8)
    x0, y_top = rx0 - margin * res_m, ry_top + margin * res_m
    for image, vx0, vy_top in views:
        r, c = int(round((y_top - vy_top) / res_m)), int(round((vx0 - x0) / res_m))
        r0, c0 = max(r, 0), max(c, 0)
        r1, c1 = min(r + image.shape[0], canvas.shape[0]), min(c + image.shape[1], canvas.shape[1])
        if r1 <= r0 or c1 <= c0:
            continue
        piece = image[r0 - r:r1 - r, c0 - c:c1 - c]
        seen = piece.max(axis=2) > 0
        canvas[r0:r1, c0:c1][seen] = piece[seen]
    return canvas, (canvas.max(axis=2) > 0).astype(np.uint8), x0, y_top


def rigid_fit(then, now):
    """(turn_deg, shift): the turn and shift carrying `then` onto `now` in the least-squares sense."""
    a, b = np.asarray(then, float), np.asarray(now, float)
    u, _, vt = np.linalg.svd((a - a.mean(0)).T @ (b - b.mean(0)))
    turn = (u @ np.diag([1.0, np.linalg.det(u @ vt)]) @ vt).T
    return float(np.degrees(np.arctan2(turn[1, 0], turn[0, 0]))), b.mean(0) - a.mean(0) @ turn.T


def locate(views, reference, prior, res_m=RES_M, search_m=SEARCH_M):
    """The carton's pose now (turn_deg, shift, correlation, covered share of the photo) from views
    projected with `prior` (see project_view) and the reference photo (image, x0, y_top) at `res_m`."""
    ref_image, rx0, ry_top = reference
    canvas, covered, cx0, cy_top = paste(views, reference, res_m, search_m)
    margin = int(search_m / res_m)
    share = float(covered[margin:-margin, margin:-margin].mean())
    if share < MIN_COVERED:
        raise ValueError(f"the pictures cover {share:.0%} of the carton, need {MIN_COVERED:.0%}")
    padded = np.zeros_like(canvas)
    padded[margin:margin + ref_image.shape[0], margin:margin + ref_image.shape[1]] = ref_image
    k = res_m / COARSE_M
    rows, cols = np.nonzero(covered)
    top, left = rows.min(), cols.min()
    seen = cv2.resize(_grey(canvas[top:rows.max() + 1, left:cols.max() + 1]), None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    seen_mask = cv2.resize(covered[top:rows.max() + 1, left:cols.max() + 1], seen.shape[::-1], interpolation=cv2.INTER_NEAREST)
    small_ref = cv2.resize(_grey(padded), None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    centre = (seen.shape[1] / 2.0, seen.shape[0] / 2.0)
    best = (-np.inf, 0.0, (0, 0))
    for angle in np.arange(-COARSE_TURN_DEG, COARSE_TURN_DEG + 1e-9, COARSE_TURN_STEP_DEG):
        turned = cv2.getRotationMatrix2D(centre, angle, 1.0)
        mask = cv2.warpAffine(seen_mask, turned, seen.shape[::-1], flags=cv2.INTER_NEAREST)
        if seen.shape[0] >= small_ref.shape[0] or seen.shape[1] >= small_ref.shape[1]:
            raise ValueError("the pictures reach further than the search window round the photo")
        scores = cv2.matchTemplate(small_ref, cv2.warpAffine(seen, turned, seen.shape[::-1]), cv2.TM_CCOEFF_NORMED, mask=mask)
        _, score, _, location = cv2.minMaxLoc(np.nan_to_num(scores, nan=-1.0))
        if score > best[0]:
            best = (score, angle, location)
    _, angle, (bx, by) = best
    turned = cv2.getRotationMatrix2D(centre, angle, 1.0)
    to_small = np.array([[k, 0, -left * k], [0, k, -top * k], [0, 0, 1]])
    to_padded = np.vstack([turned + np.array([[0, 0, bx], [0, 0, by]]), [0, 0, 1]])
    warp = (np.diag([1 / k, 1 / k, 1]) @ to_padded @ to_small)[:2].astype(np.float32)
    inverse = cv2.invertAffineTransform(warp)
    inset = int(EDGE_INSET_M / res_m)
    on_padded = np.zeros(covered.shape, np.uint8)
    on_padded[margin + inset:margin + ref_image.shape[0] - inset, margin + inset:margin + ref_image.shape[1] - inset] = 1
    mask = covered * cv2.warpAffine(on_padded, inverse, covered.shape[::-1], flags=cv2.INTER_NEAREST)
    template, image = _grey(canvas), _grey(padded)
    for scale in ECC_SCALES:
        start = np.c_[warp[:, :2], warp[:, 2] * scale].astype(np.float32)
        try:
            _, found = cv2.findTransformECC(*(cv2.resize(i, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for i in (template, image)),
                                            start, cv2.MOTION_EUCLIDEAN, ECC_CRITERIA,
                                            cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST), 3)
        except cv2.error as exc:
            raise ValueError(f"the pictures did not line up with the reference photo ({exc.err})") from exc
        warp = np.c_[found[:, :2], found[:, 2] / scale].astype(np.float32)
    aligned = cv2.warpAffine(image, warp, template.shape[::-1], flags=cv2.WARP_INVERSE_MAP)
    used = mask > 0
    correlation = float(np.corrcoef(template[used], aligned[used])[0, 1])
    if correlation < MIN_CORRELATION:
        raise ValueError(f"the pictures line up with the reference photo only weakly (correlation {correlation:.2f})")
    rows, cols = np.nonzero(mask[::20, ::20])
    pixels = np.c_[cols * 20, rows * 20].astype(float)
    in_ref = pixels @ warp[:, :2].T + warp[:, 2] - margin
    then = np.c_[rx0 + in_ref[:, 0] * res_m, ry_top - in_ref[:, 1] * res_m]
    prior_xy = np.c_[cx0 + pixels[:, 0] * res_m, cy_top - pixels[:, 1] * res_m]
    now = prior_xy @ turn_matrix(prior[0]).T + np.asarray(prior[1], float)
    turn_deg, shift = rigid_fit(then, now)
    return turn_deg, shift, float(correlation), share
