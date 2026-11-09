"""The camera mount and lens fitted to one scan's own pictures: a splat dataset without a board.

The arm's poses with the saved mount calibration are ~11 px off - the mount was fitted from one dark
nut's centre, with a guessed lens - and that blurs a knit, whose stitches are 20-30 px. COLMAP, which
solves every picture's pose from the pictures alone, cannot tell one stitch from the next. This sits
between the two. Within one scan the arm's relative motion is exact to a fraction of a millimetre and
the tool-centre setting is constant, so what is wrong is shared by every picture: the mount (6 numbers)
and the lens (focal length, centre, two distortion terms). Those 11 numbers are fitted to everything
the pictures agree on:

  1. SIFT features in every picture.
  2. Pairs of pictures the arm says overlap, matched, and kept only where the match agrees with the
     geometry the arm predicts - which is what stops a stitch matching its neighbour - then checked
     against an epipolar RANSAC.
  3. Matches chained into tracks, each track triangulated.
  4. Bundle adjustment of the mount, the lens and the points together, robust to the odd bad match;
     then the gate is tightened to the fitted model and it is run again.

A scan taken under another tool-centre setting (Oct 18: the reported pose sat 108.5 mm along the tool
from the plan) starts the mount from the saved calibration moved by that offset; a camera driver that
mirrors its image is tried both ways. Whichever start explains the most matches wins.

export_dataset writes the result as a splatting dataset: every camera the arm's pose with the fitted
mount, the fitted lens with its distortion, and the triangulated points, coloured, as the seed.
"""

from __future__ import annotations

import math
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import gaussian_splatting
import hand_eye
from persistence import write_json

FEATURES_PER_PICTURE = 2500
RATIO = 0.8
NEIGHBOURS = 12                 # pictures each one is matched against, the most overlapping first
MIN_OVERLAP = 0.25              # share of a picture's footprint the other must cover to be matched
GATES_PX = (150.0,)             # how far a match may first sit from where the arm puts it; each later
ROUNDS = 5                      # round's gate is 4x the last round's error, down to MIN_GATE_PX
MIN_GATE_PX = 3.0
MIN_TRACK_ANGLE_DEG = 2.0
MIN_TRACKS = 50
MIN_PAIR_MATCHES = 12
MAX_TRACKS = 30000
HUBER_PX = 2.0
MAX_EVALUATIONS = 400


def _matrix(rotvec, translation) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = Rotation.from_rotvec(np.asarray(rotvec, dtype=float)).as_matrix(), np.asarray(translation, dtype=float)
    return out


def arm_matrices(captures) -> list[np.ndarray]:
    """Each picture's arm pose, in the one frame a session's mount is fitted in (hand_eye.arm_pose)."""
    return [hand_eye.tcp_matrix(hand_eye.arm_pose(c)) for c in captures]


def project(points_world: np.ndarray, world_to_camera: np.ndarray, lens: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pixels (through the distortion) and depths of world points in one camera."""
    cam = points_world @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    z = cam[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    x, y = cam[:, 0] / safe, cam[:, 1] / safe
    r2 = x * x + y * y
    radial = 1.0 + lens["k1"] * r2 + lens["k2"] * r2 * r2
    return np.stack([lens["f"] * x * radial + lens["cx"], lens["f"] * y * radial + lens["cy"]], axis=1), z


def _undistort(pixels: np.ndarray, lens: dict) -> np.ndarray:
    """Normalised image coordinates of distorted pixels (a few fixed-point steps)."""
    xd = (pixels[:, 0] - lens["cx"]) / lens["f"]
    yd = (pixels[:, 1] - lens["cy"]) / lens["f"]
    x, y = xd.copy(), yd.copy()
    for _ in range(8):
        r2 = x * x + y * y
        radial = 1.0 + lens["k1"] * r2 + lens["k2"] * r2 * r2
        x, y = xd / radial, yd / radial
    return np.stack([x, y], axis=1)


def _footprint(camera_to_world: np.ndarray, lens: dict, size, plane_z: float) -> np.ndarray | None:
    w, h = size
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
    rays = np.c_[_undistort(corners, lens), np.ones(4)] @ camera_to_world[:3, :3].T
    centre = camera_to_world[:3, 3]
    if np.any(rays[:, 2] >= -1e-6):
        return None
    return centre[:2] + ((plane_z - centre[2]) / rays[:, 2])[:, None] * rays[:, :2]


def _overlap(a: np.ndarray, b: np.ndarray, samples: int = 12) -> float:
    """Share of footprint a's area covered by footprint b (sampled)."""
    import matplotlib.path as mpath

    lo, hi = a.min(axis=0), a.max(axis=0)
    grid = np.stack(np.meshgrid(np.linspace(lo[0], hi[0], samples), np.linspace(lo[1], hi[1], samples)), -1).reshape(-1, 2)
    inside_a = mpath.Path(a).contains_points(grid)
    if not inside_a.any():
        return 0.0
    return float(mpath.Path(b).contains_points(grid[inside_a]).mean())


class _Pictures:
    def __init__(self, paths, mirrored: bool):
        import cv2

        self.grey, self.colour, self.keypoints, self.descriptors = [], [], [], []
        sift = cv2.SIFT_create(FEATURES_PER_PICTURE)
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if mirrored:
                image = image[:, ::-1].copy()
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            points, descriptors = sift.detectAndCompute(grey, None)
            if descriptors is None:
                points, descriptors = [], np.zeros((0, 128), np.float32)
            # RootSIFT: matches better than plain SIFT for the same cost.
            descriptors = np.sqrt(descriptors / (np.abs(descriptors).sum(axis=1, keepdims=True) + 1e-9)).astype(np.float32)
            self.colour.append(image)
            self.keypoints.append(np.array([p.pt for p in points], dtype=float).reshape(-1, 2))
            self.descriptors.append(descriptors)
        self.size = (self.colour[0].shape[1], self.colour[0].shape[0])


def _candidate_matches(pictures: _Pictures, pairs) -> dict:
    import cv2

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    found = {}
    for i, j in pairs:
        if len(pictures.descriptors[i]) < 2 or len(pictures.descriptors[j]) < 2:
            continue
        knn = matcher.knnMatch(pictures.descriptors[i], pictures.descriptors[j], k=2)
        good = [(m.queryIdx, m.trainIdx) for m, n in (pair for pair in knn if len(pair) == 2) if m.distance < RATIO * n.distance]
        if good:
            found[(i, j)] = np.array(good, dtype=int)
    return found


def _gate(pictures, candidates, cameras, lens, plane_z, gate_px):
    """Matches that land within `gate_px` of where the model puts them: each keypoint of picture i is
    carried to j through the point it would be on the fabric plane (or, later, near it), and the
    match is kept if j's keypoint is near - then the pair is checked with an epipolar RANSAC."""
    import cv2

    kept = {}
    for (i, j), matches in candidates.items():
        a = pictures.keypoints[i][matches[:, 0]]
        b = pictures.keypoints[j][matches[:, 1]]
        rays = np.c_[_undistort(a, lens), np.ones(len(a))] @ cameras[i][:3, :3].T
        centre = cameras[i][:3, 3]
        depth = (plane_z - centre[2]) / np.where(np.abs(rays[:, 2]) < 1e-9, -1e-9, rays[:, 2])
        on_plane = centre + depth[:, None] * rays
        predicted, z = project(on_plane, np.linalg.inv(cameras[j]), lens)
        near = (depth > 0) & (z > 0) & (np.linalg.norm(predicted - b, axis=1) < gate_px)
        if near.sum() < MIN_PAIR_MATCHES:
            continue
        a_n, b_n = _undistort(a[near], lens), _undistort(b[near], lens)
        _, inliers = cv2.findFundamentalMat(a_n, b_n, cv2.FM_RANSAC, 1.5 / lens["f"], 0.999)
        if inliers is None or inliers.sum() < MIN_PAIR_MATCHES:
            continue
        kept[(i, j)] = matches[near][inliers.ravel().astype(bool)]
    return kept


def _tracks(matches: dict, counts: list[int]) -> list[list[tuple[int, int]]]:
    offsets = np.concatenate([[0], np.cumsum(counts)])
    parent = np.arange(int(offsets[-1]))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j), pairs in matches.items():
        for a, b in pairs:
            ra, rb = find(offsets[i] + a), find(offsets[j] + b)
            if ra != rb:
                parent[ra] = rb
    groups: dict[int, list[tuple[int, int]]] = {}
    image_of = np.searchsorted(offsets, np.arange(int(offsets[-1])), side="right") - 1
    for (i, j), pairs in matches.items():
        for a, b in pairs:
            for node in (offsets[i] + a, offsets[j] + b):
                groups.setdefault(find(node), set()).add(int(node))
    tracks = []
    for nodes in groups.values():
        observed = [(int(image_of[n]), int(n - offsets[image_of[n]])) for n in nodes]
        if len({image for image, _ in observed}) == len(observed) and len(observed) >= 2:
            tracks.append(sorted(observed))
    return tracks


def _points_for(track_of, image_of, seen, cameras, lens, count):
    """Every track's point for this mount and lens: the point nearest all its rays (vectorised)."""
    rays = np.c_[_undistort(seen, lens), np.ones(len(seen))]
    rays = np.einsum("nij,nj->ni", cameras[image_of, :3, :3], rays)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    centres = cameras[image_of, :3, 3]
    across = np.eye(3)[None] - rays[:, :, None] * rays[:, None, :]      # projects onto the plane across each ray
    a = np.zeros((count, 3, 3))
    b = np.zeros((count, 3))
    np.add.at(a, track_of, across)
    np.add.at(b, track_of, np.einsum("nij,nj->ni", across, centres))
    a += np.eye(3)[None] * 1e-12
    return np.linalg.solve(a, b[:, :, None])[:, :, 0]


def _adjust(tracks, points, pictures, arms, mount, lens, refine_lens=True):
    """The mount (6) and lens (f, cx, cy, k1, k2) fitted to every match, Huber-robust.

    Only those 11 numbers are searched: for any of them each track's point is the point nearest its
    rays, and the residual is how far that point reprojects from where it was seen. Searching the
    points too (a full bundle adjustment) left the focal length crawling along the valley where a
    longer lens and deeper points look alike - 400 evaluations moved it a fifth of the way."""
    del points
    observations = [(t, image, pictures.keypoints[image][key]) for t, track in enumerate(tracks) for image, key in track]
    track_of = np.array([o[0] for o in observations])
    image_of = np.array([o[1] for o in observations])
    seen = np.array([o[2] for o in observations], dtype=float)
    lens_keys = ("f", "cx", "cy", "k1", "k2")
    start = np.concatenate([Rotation.from_matrix(mount[:3, :3]).as_rotvec(), mount[:3, 3],
                            [lens[k] for k in lens_keys] if refine_lens else []])
    arms = np.asarray(arms)

    def unpack(x):
        return _matrix(x[:3], x[3:6]), ({**lens, **dict(zip(lens_keys, x[6:11]))} if refine_lens else dict(lens))

    def evaluate(x):
        fitted_mount, fitted_lens = unpack(x)
        cameras = arms @ fitted_mount
        pts = _points_for(track_of, image_of, seen, cameras, fitted_lens, len(tracks))
        world_to_camera = np.linalg.inv(cameras)
        cam = np.einsum("nij,nj->ni", world_to_camera[image_of, :3, :3], pts[track_of]) + world_to_camera[image_of, :3, 3]
        z = np.where(cam[:, 2] < 1e-6, 1e-6, cam[:, 2])
        u, v = cam[:, 0] / z, cam[:, 1] / z
        r2 = u * u + v * v
        radial = 1.0 + fitted_lens["k1"] * r2 + fitted_lens["k2"] * r2 * r2
        predicted = np.stack([fitted_lens["f"] * u * radial + fitted_lens["cx"], fitted_lens["f"] * v * radial + fitted_lens["cy"]], axis=1)
        return (predicted - seen).ravel(), pts

    scale = np.r_[np.full(3, 0.01), np.full(3, 0.001), [10.0, 5.0, 5.0, 0.01, 0.01] if refine_lens else []]
    # Cauchy, not Huber: a stitch matched to its neighbour is tens of pixels off, and under Huber a
    # few thousand of those still outweighed every good match.
    fitted = least_squares(lambda x: evaluate(x)[0], start, loss="cauchy", f_scale=HUBER_PX, x_scale=scale,
                           diff_step=1e-6, method="trf", max_nfev=MAX_EVALUATIONS)
    residual, pts = evaluate(fitted.x)
    fitted_mount, fitted_lens = unpack(fitted.x)
    return fitted_mount, fitted_lens, pts, np.linalg.norm(residual.reshape(-1, 2), axis=1), track_of


def _track_medians(error, track_of, count) -> np.ndarray:
    """Each track's median reprojection error."""
    order = np.argsort(track_of, kind="stable")
    starts_at = np.searchsorted(track_of[order], np.arange(count))
    ends_at = np.r_[starts_at[1:], len(order)]
    return np.array([np.median(error[order[s:e]]) if e > s else np.inf for s, e in zip(starts_at, ends_at)])


def _wide_enough(tracks, pictures, cameras, lens, min_deg: float = MIN_TRACK_ANGLE_DEG):
    """Tracks whose rays meet at a usable angle. Rays nearly parallel put the point anywhere along
    them, and a point a metre off reprojects hundreds of pixels off - it outweighed the rest."""
    kept = []
    for track in tracks:
        rays = np.array([cameras[image][:3, :3] @ np.r_[_undistort(pictures.keypoints[image][key][None], lens)[0], 1.0] for image, key in track])
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        widest = math.degrees(math.acos(float(np.clip((rays @ rays.T).min(), -1.0, 1.0))))
        if widest >= min_deg:
            kept.append(track)
    return kept


def _start_mounts(captures, calibration) -> list[tuple[str, np.ndarray]]:
    """The mounts to start from: the saved calibration, and - when this session's controller reported
    its tool point a tool length from the plan - the calibration moved by that length."""
    saved = hand_eye.calibration_matrix(calibration)
    starts = [("saved mount", saved)]
    offsets = []
    for capture in captures:
        planned, reported = capture.get("target_position"), capture.get("actual_tcp_pose")
        if planned and reported and len(planned) >= 6 and len(reported) >= 6:
            axis = Rotation.from_rotvec(planned[3:6]).as_matrix()[:, 2]
            offsets.append(float(np.dot(np.subtract(reported[:3], planned[:3]), axis)))
    if offsets and abs(np.median(offsets)) > gaussian_splatting.TOOL_CENTRE_MISMATCH_M:
        along = float(np.median(offsets))
        for sign in (1.0, -1.0):
            # Along the arm pose's own tool axis: the mount's translation is in that frame.
            shifted = saved.copy()
            shifted[:3, 3] = saved[:3, 3] + np.array([0.0, 0.0, sign * along])
            starts.append((f"saved mount moved {sign * along * 1000:+.0f} mm along the tool", shifted))
    return starts


def fit(captures, calibration: dict, plane_z: float, *, mirrored=None, measured_lens: dict | None = None, log=print) -> dict:
    """Fit the mount and lens of one session to its pictures. `plane_z` is the fabric's height in the
    arm's frame; `mirrored` None tries the pictures both ways."""
    paths = [Path(str(c.get("path") or c.get("image_path") or "")) for c in captures]
    arms = arm_matrices(captures)
    best = None
    for flip in ((False, True) if mirrored is None else (bool(mirrored),)):
        pictures = _Pictures(paths, flip)
        width, height = pictures.size
        if measured_lens:
            scale = width / float(measured_lens["width"])
            lens = {"f": float(measured_lens["fx"]) * scale, "cx": float(measured_lens["cx"]) * scale,
                    "cy": float(measured_lens["cy"]) * scale, "k1": float(measured_lens.get("k1", 0.0)), "k2": float(measured_lens.get("k2", 0.0))}
        else:
            f = width / (2.0 * math.tan(math.radians(float(calibration.get("fov_degrees", 30.0))) / 2.0))
            lens = {"f": f, "cx": width / 2.0, "cy": height / 2.0, "k1": 0.0, "k2": 0.0}
        for label, mount in _start_mounts(captures, calibration):
            cameras = [arm @ mount for arm in arms]
            footprints = [_footprint(c, lens, pictures.size, plane_z) for c in cameras]
            pairs = []
            for i, fi in enumerate(footprints):
                if fi is None:
                    continue
                scored = sorted(((_overlap(fi, fj), j) for j, fj in enumerate(footprints) if j > i and fj is not None), reverse=True)
                pairs += [(i, j) for share, j in scored[:NEIGHBOURS] if share >= MIN_OVERLAP]
            candidates = _candidate_matches(pictures, pairs)
            kept = _gate(pictures, candidates, cameras, lens, plane_z, GATES_PX[0])
            total = sum(len(m) for m in kept.values())
            log(f"  {'mirrored, ' if flip else ''}{label}: {len(pairs)} overlapping pairs, {len(kept)} agree with the arm "
                f"({total} matches)")
            if best is None or total > best["matches"]:
                best = {"mirrored": flip, "start": label, "mount": mount, "lens": lens, "pictures": pictures,
                        "candidates": candidates, "matches": total}
    if best is None or best["matches"] == 0:
        raise RuntimeError("no pair of pictures agrees with the arm's poses: is the fabric height right, and is this one scan?")

    pictures, mount, lens = best["pictures"], best["mount"], best["lens"]
    log(f"starting from: {'mirrored pictures, ' if best['mirrored'] else ''}{best['start']}")
    rounds = []
    gate = GATES_PX[0]
    for _ in range(ROUNDS):
        cameras = [arm @ mount for arm in arms]
        kept = _gate(pictures, best["candidates"], cameras, lens, plane_z, gate)
        tracks = _tracks(kept, [len(k) for k in pictures.keypoints])
        tracks.sort(key=len, reverse=True)
        tracks = _wide_enough(tracks[:MAX_TRACKS], pictures, np.asarray(cameras), lens)
        if len(tracks) < MIN_TRACKS:
            raise RuntimeError(f"only {len(tracks)} usable tracks at a {gate:.0f} px gate: the arm's poses and the pictures do not agree")
        for _ in range(2):
            mount_next, lens_next, points, error, track_of = _adjust(tracks, None, pictures, arms, mount, lens, refine_lens=True)
            per_track = _track_medians(error, track_of, len(tracks))
            limit = max(2.0, 3.0 * float(np.median(error)))
            if (per_track <= limit).all():
                break
            tracks = [t for t, e in zip(tracks, per_track) if e <= limit]
        else:
            # The last pass dropped tracks: fit once more so the points match the tracks kept.
            mount_next, lens_next, points, error, track_of = _adjust(tracks, None, pictures, arms, mount, lens, refine_lens=True)
        if not 0.6 * lens["f"] < lens_next["f"] < 1.6 * lens["f"]:
            raise RuntimeError(f"the lens came out implausible (focal length {lens_next['f']:.0f} px, from {lens['f']:.0f}): "
                               "the matches do not pin it down - this scan cannot be fitted")
        mount, lens = mount_next, lens_next
        median = float(np.median(error))
        rms = float(np.sqrt(np.mean(np.minimum(error, 10.0) ** 2)))
        rounds.append({"gate_px": gate, "pairs": len(kept), "tracks": len(tracks), "median_px": median, "rms_px": rms})
        log(f"  gate {gate:.0f} px: {len(kept)} pairs, {len(tracks)} tracks, reprojection median {median:.2f} px, f {lens['f']:.0f}")
        next_gate = float(np.clip(4.0 * median, MIN_GATE_PX, gate))
        if next_gate >= gate * 0.9 and len(rounds) > 1:
            break
        gate = next_gate

    # Each picture's own residual, and the coloured points for the seed.
    per_image = {}
    observations = [(t, image, key) for t, track in enumerate(tracks) for image, key in track]
    for (t, image, key), e in zip(observations, error):
        per_image.setdefault(image, []).append(float(e))
    colours = []
    for track in tracks:
        image, key = track[0]
        x, y = np.clip(pictures.keypoints[image][key].astype(int), 0, [pictures.size[0] - 1, pictures.size[1] - 1])
        colours.append(pictures.colour[image][y, x, ::-1])
    # A track's points are kept when its observations fit to under 2 px on average (its median).
    good = _track_medians(error, track_of, len(tracks)) < 2.0
    return {"mount": mount, "lens": lens, "mirrored": best["mirrored"], "start": best["start"], "size": pictures.size,
            "points": points[good], "colours": np.array(colours, dtype=np.uint8)[good] if len(colours) else np.zeros((0, 3), np.uint8),
            "rounds": rounds, "per_image_median_px": {i: float(np.median(v)) for i, v in per_image.items()},
            "pictures_with_points": len(per_image), "pictures": len(paths)}


def _write_ply(path: Path, points: np.ndarray, colours: np.ndarray) -> int:
    table = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    table["x"], table["y"], table["z"] = points[:, 0], points[:, 1], points[:, 2]
    table["red"], table["green"], table["blue"] = colours[:, 0], colours[:, 1], colours[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(table)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        table.tofile(handle)
    return len(table)


def export_dataset(captures, fitted: dict, output_dir: str | Path, *, session_id: str = "",
                   max_picture_px: float = 3.0, equalise_exposure: bool = True) -> gaussian_splatting.SplatDataset:
    """The session as a splatting dataset placed by the fitted mount and lens. A picture whose own
    points fit worse than `max_picture_px` (it moved, or the arm missed) is left out."""

    output_dir = Path(output_dir)
    images_dir, colmap_dir = output_dir / "images", output_dir / "sparse" / "0"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    colmap_dir.mkdir(parents=True, exist_ok=True)
    width, height = fitted["size"]
    lens = fitted["lens"]
    intrinsics = {"fl_x": float(lens["f"]), "fl_y": float(lens["f"]), "cx": float(lens["cx"]), "cy": float(lens["cy"]),
                  "w": int(width), "h": int(height), "k1": float(lens["k1"]), "k2": float(lens["k2"]), "k3": 0.0, "p1": 0.0, "p2": 0.0,
                  "camera_angle_x": 2.0 * math.atan(width / (2.0 * float(lens["f"])))}
    arms = arm_matrices(captures)
    paths = [Path(str(c.get("path") or c.get("image_path") or "")) for c in captures]
    gains = gaussian_splatting.exposure_gains(paths) if equalise_exposure else {}
    warnings, frames, colmap_images = [], [], []
    per_image = fitted["per_image_median_px"]
    for index, (capture, arm, source) in enumerate(zip(captures, arms, paths)):
        residual = per_image.get(index)
        if residual is None:
            warnings.append(f"{source.name}: no point matched to another picture; kept, placed by the arm")
        elif residual > max_picture_px:
            warnings.append(f"{source.name}: its points fit to {residual:.1f} px; left out")
            continue
        gaussian_splatting.equalised_image(source, gains.get(str(source), np.ones(3)), fitted["mirrored"]).save(images_dir / source.name)
        camera_to_world = arm @ fitted["mount"]
        frames.append({"file_path": f"images/{source.name}", "transform_matrix": (camera_to_world @ gaussian_splatting._CV_TO_GL).tolist(),
                       "colmap_im_id": index + 1, "fit_median_px": residual, "view": str(capture.get("capture_view", capture.get("angle", "")))})
        w2c = np.linalg.inv(camera_to_world)
        quat = Rotation.from_matrix(w2c[:3, :3]).as_quat()
        colmap_images.append(f"{index + 1} {quat[3]:.9f} {quat[0]:.9f} {quat[1]:.9f} {quat[2]:.9f} "
                             f"{w2c[0, 3]:.9f} {w2c[1, 3]:.9f} {w2c[2, 3]:.9f} 1 {source.name}")
    seed = _write_ply(output_dir / gaussian_splatting.SEED_PLY_NAME, fitted["points"], fitted["colours"]) if len(fitted["points"]) else 0
    last = fitted["rounds"][-1]
    transforms = {
        **intrinsics, "camera_model": "OPENCV", "session_id": str(session_id), "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pose_source": (f"arm poses with the camera mount fitted to this scan's own pictures (session fit: {last['tracks']} tracks, "
                        f"reprojection median {last['median_px']:.2f} px)"),
        "intrinsics_source": "fitted to this scan's own pictures, with the mount",
        "session_fit": {"mount": fitted["mount"].tolist(), "lens": {k: float(v) for k, v in lens.items()}, "mirrored": fitted["mirrored"],
                        "start": fitted["start"], "rounds": fitted["rounds"], "pictures_with_points": fitted["pictures_with_points"]},
        **({"mirrored": True} if fitted["mirrored"] else {}),
        **({"ply_file_path": gaussian_splatting.SEED_PLY_NAME} if seed else {}),
        **({"exposure_equalised": True} if gains else {}),
        "frames": frames,
    }
    transforms_path = output_dir / "transforms.json"
    write_json(transforms_path, transforms, indent=2)
    gaussian_splatting._write_colmap_model(colmap_dir, colmap_images, intrinsics)
    return gaussian_splatting.SplatDataset(root=output_dir, image_count=len(frames), transforms_path=transforms_path,
                                           colmap_dir=colmap_dir, session_id=str(session_id), warnings=warnings, offered=len(captures))
