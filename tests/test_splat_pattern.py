"""The splat dome: every view aimed at the subject, at each distance and tilt asked for, in short steps."""

import math

import numpy as np
import pytest


def test_the_splat_dome_aims_every_view_at_the_subject_from_each_distance_and_tilt():
    pytest.importorskip("scipy")
    from scipy.spatial.transform import Rotation

    import scan_geometry

    subject = np.array([-0.45, -0.08, 0.0])
    tool = np.r_[-0.45, -0.25, 0.20, Rotation.from_matrix(np.diag([1.0, -1.0, -1.0])).as_rotvec()]
    views, _ = scan_geometry.splat_views(tool, subject, 12, 3, 2, 0.10, 36.0)

    assert len(views) == 2 * (1 + 3 * 12)
    distances, tilts = set(), set()
    for view in views:
        look = Rotation.from_rotvec(view.rotvec).as_matrix()[:, 2]
        toward = (subject - view.position) / np.linalg.norm(subject - view.position)
        assert float(np.dot(look, toward)) == pytest.approx(1.0, abs=1e-6)
        distances.add(round(float(np.linalg.norm(view.position - subject)), 4))
        tilts.add(round(math.degrees(math.acos(-look[2]))))
    assert distances == {0.10, 0.15}
    assert tilts == {0, 12, 24, 36}
    steps = [np.linalg.norm(b.position - a.position) for a, b in zip(views, views[1:])]
    assert max(steps) < 0.07


def test_the_dome_can_be_given_its_distances_outright():
    """Three close layers in the camera's sharp band and one far enough to see the board."""
    pytest.importorskip("scipy")
    from scipy.spatial.transform import Rotation

    import scan_geometry

    subject = np.array([-0.45, -0.08, 0.0])
    tool = np.r_[-0.45, -0.25, 0.20, Rotation.from_matrix(np.diag([1.0, -1.0, -1.0])).as_rotvec()]
    views, sphere = scan_geometry.splat_views(tool, subject, 8, 2, 99, 0.5, 30.0, layer_radii=[0.085, 0.09, 0.095, 0.15])

    assert len(views) == 4 * (1 + 2 * 8)
    assert {round(float(np.linalg.norm(v.position - subject)), 4) for v in views} == {0.085, 0.09, 0.095, 0.15}
    assert sphere.standoff == pytest.approx(0.085)
