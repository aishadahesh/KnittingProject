"""Finding each yarn's own Gaussians: yarns made of small flat Gaussians on twisted plies, crossing,
touching, and passing out of sight."""


import json

import numpy as np
from scipy.spatial.transform import Rotation

import scan_fit  # noqa: E402
import yarn_tracks  # noqa: E402

RADIUS = 0.5            # mm; 1 unit = 1 mm here
TWIST_DEG = 30.0


def _yarn(start, end, offset=(0.0, 0.0, 0.0), gap=None, rng=None):
    """A straight yarn from `start` to `end` (mm) as Gaussians on its surface: rings every 0.1 mm of
    14 Gaussians, each lying along a ply (its long axis turned TWIST_DEG round the tube from the
    yarn's direction). `gap` (from, to) along the yarn leaves a stretch out of sight."""
    rng = rng or np.random.default_rng(0)
    start, end = np.asarray(start, float) + offset, np.asarray(end, float) + offset
    tangent = (end - start) / np.linalg.norm(end - start)
    side = np.cross(tangent, [0, 0, 1.0])
    side = side / np.linalg.norm(side) if np.linalg.norm(side) > 1e-6 else np.array([1.0, 0, 0])
    up = np.cross(side, tangent)
    length = np.linalg.norm(end - start)
    positions, rotations = [], []
    for s in np.arange(0.0, length, 0.1):
        if gap is not None and gap[0] <= s <= gap[1]:
            continue
        for k in range(14):
            angle = 2 * np.pi * k / 14 + rng.uniform(-0.1, 0.1)
            normal = np.cos(angle) * side + np.sin(angle) * up
            around = np.cross(tangent, normal)
            twist = np.radians(TWIST_DEG)
            axis = np.cos(twist) * tangent + np.sin(twist) * around
            third = np.cross(normal, axis)                 # right-handed: long x middle = short
            positions.append(start + s * tangent + RADIUS * normal)
            rotations.append(np.stack([axis, third, normal], axis=1))   # columns: long, middle, short axes
    q = Rotation.from_matrix(np.array(rotations)).as_quat()            # x, y, z, w
    return np.array(positions), np.c_[q[:, 3], q[:, :3]]


def _fabric():
    """Yarn A along x; yarn B over it at 70 deg, touching; yarn C beside A, out of sight for 1.5 mm."""
    parts = [
        _yarn((-6, 0, 0), (6, 0, 0)),
        _yarn((-6 * np.cos(np.radians(70)), -6 * np.sin(np.radians(70)), 0), (6 * np.cos(np.radians(70)), 6 * np.sin(np.radians(70)), 0),
              offset=(0, 0, 2 * RADIUS)),
        _yarn((-6, 3, 0), (6, 3, 0), gap=(5.3, 6.8)),
    ]
    positions = np.concatenate([p for p, _ in parts])
    quaternion = np.concatenate([q for _, q in parts])
    truth = np.concatenate([np.full(len(p), k) for k, (p, _) in enumerate(parts)])
    count = len(positions)
    scales = np.tile([0.35, 0.17, 0.10], (count, 1))
    return positions, scales, quaternion, np.ones(count), truth


def test_long_axes_are_read_from_the_rotation():
    positions, quaternion = _yarn((0, 0, 0), (5, 0, 0))
    axes = yarn_tracks.long_axes(np.tile([0.35, 0.17, 0.1], (len(positions), 1)), quaternion)
    # A twisted ply: 30 deg off the yarn's direction, all round the tube.
    assert np.allclose(np.degrees(np.arccos(np.abs(axes[:, 0]))), TWIST_DEG, atol=0.5)


def test_crossing_touching_and_hidden_yarns_are_told_apart():
    positions, scales, quaternion, opacity, truth = _fabric()
    found = yarn_tracks.find_yarns(positions, scales, quaternion, opacity)
    yarn = found["yarn"]
    assert abs(found["radius"] - RADIUS) < 0.15                          # the radius, measured
    assert len(np.unique(yarn)) == 3                                      # three yarns, C joined across its gap
    for k in range(3):
        mine = yarn[truth == k]
        main = np.bincount(mine).argmax()
        assert (mine == main).mean() > 0.95                               # each yarn's Gaussians in one yarn
        assert (yarn[truth != k] == main).mean() < 0.03                   # and no other yarn's


def test_region_takes_its_yarn_and_the_rims_nearest_it():
    rng = np.random.default_rng(0)
    lace = np.c_[rng.uniform(-2, -0.5, 50), rng.uniform(0, 1, 50), np.zeros(50)]
    band = np.c_[rng.uniform(0.5, 2, 50), rng.uniform(0, 1, 50), np.zeros(50)]
    rims = np.array([[-1.0, 0.5, 0.05], [1.0, 0.5, 0.05]])
    table = np.array([[0.0, 0.5, -1.0]])
    positions = np.vstack([lace, band, rims, table])
    labels = np.r_[np.zeros(50, int), np.ones(50, int), [2, 2], [3]]
    roles = ["yarn", "yarn", "edge", "table"]
    lace_part = yarn_tracks.region_members(positions, labels, roles, 0)
    band_part = yarn_tracks.region_members(positions, labels, roles, 1)
    assert lace_part[:50].all() and not lace_part[50:100].any()
    assert lace_part[100] and not lace_part[101] and band_part[101]
    assert not lace_part[102] and not band_part[102]


def test_saved_frame_matches_scan_fits_carrying(tmp_path):
    rng = np.random.default_rng(1)
    q1, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q2, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    frame = {"mm_per_unit": 24.6, "extraction_centre": [1.0, 2.0, 0.5], "extraction_axes": q1.tolist(),
             "flat_centre": [0.1, -0.2, 0.3], "flat_axes": q2.tolist()}
    path = tmp_path / "stitch_model.json"
    path.write_text(json.dumps({"frame": frame}), encoding="utf-8")
    prepared = yarn_tracks.fit_frame(path)
    points = rng.normal(size=(5, 3))
    expected = ((points @ q1 + frame["extraction_centre"]) / 24.6) @ q2 + frame["flat_centre"]
    assert np.allclose(scan_fit.to_splat_frame(prepared, points), expected)
    path.write_text("{}", encoding="utf-8")
    assert yarn_tracks.fit_frame(path) is None
