"""Finding each yarn's own Gaussians: yarns made of small flat Gaussians on twisted plies, crossing,
touching, and passing out of sight."""


import numpy as np
from scipy.spatial.transform import Rotation

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


def test_yarns_are_ordered_across_their_direction():
    positions, scales, quaternion, opacity, truth = _fabric()
    keep = truth != 1                                                     # the two parallel yarns, A at y=0, C at y=3
    found = yarn_tracks.find_yarns(positions[keep], scales[keep], quaternion[keep], opacity[keep])
    order = yarn_tracks.order_yarns(positions[keep], found["yarn"], found["direction"])
    first = [np.bincount(found["yarn"][truth[keep] == k]).argmax() for k in (0, 2)]
    assert order in (first, first[::-1])
    assert yarn_tracks.order_yarns(positions[keep], found["yarn"], found["direction"], flip=True) == order[::-1]
