"""The GPU splat renderer: the same picture as the CPU one, and the height colours that show relief."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import gaussian_splatting  # noqa: E402
import splat_gl  # noqa: E402


def _context():
    moderngl = pytest.importorskip("moderngl")
    try:
        return moderngl.create_standalone_context()
    except Exception as exc:              # no GPU / GL driver on this machine
        pytest.skip(f"no OpenGL context: {exc}")


def _ball():
    rng = np.random.default_rng(0)
    count = 4000
    return {"positions": (rng.normal(size=(count, 3)) * 0.05).astype(np.float32),
            "colours": np.tile(np.array([0.9, 0.2, 0.2], dtype=np.float32), (count, 1)),
            "opacity": np.full(count, 0.95, dtype=np.float32), "radius": np.full(count, 0.01, dtype=np.float32),
            "sigma": np.full(count, 0.01, dtype=np.float32), "count": count}


def test_the_gpu_draws_the_picture_the_cpu_draws():
    ctx = _context()
    splat, camera = _ball(), (20.0, 30.0, 1.0, (0.0, 0.0))
    frame = gaussian_splatting.orbit_frame(splat["positions"])
    renderer = splat_gl.SplatRenderer(ctx)
    renderer.render([("ball", splat, None, 1.0)], camera, frame, (240, 180), background=(0, 0, 0))
    gpu = renderer.read().astype(float)
    cpu = np.asarray(gaussian_splatting.render_splat(splat, 20.0, 30.0, 1.0, (240, 180), background=(0, 0, 0), frame=frame)).astype(float)
    assert gpu[80:100, 110:130, 0].mean() > 150                                     # red ball in the middle
    assert np.abs(gpu - cpu).mean() < 6                                             # and the same picture round it
    covered_gpu, covered_cpu = gpu.max(axis=2) > 40, cpu.max(axis=2) > 40
    assert (covered_gpu & covered_cpu).sum() / (covered_gpu | covered_cpu).sum() > 0.9   # the same outline
    # Seen from the other side and from above, the ball stays a ball: the camera really turns round it.
    renderer.render([("ball", splat, None, 1.0)], (200.0, 80.0, 1.0, (0.0, 0.0)), frame, (240, 180), background=(0, 0, 0))
    assert renderer.read()[80:100, 110:130, 0].mean() > 150


def test_a_see_through_layer_is_laid_over_at_its_strength():
    ctx = _context()
    ball = _ball()
    blue = dict(ball, colours=np.tile(np.array([0.1, 0.2, 0.9], dtype=np.float32), (ball["count"], 1)))
    frame = gaussian_splatting.orbit_frame(ball["positions"])
    renderer = splat_gl.SplatRenderer(ctx)
    renderer.render([("red", ball, None, 1.0), ("blue", blue, None, 0.5)], (0.0, 30.0, 1.0, (0.0, 0.0)), frame, (240, 180))
    middle = renderer.read()[85:95, 115:125].astype(float).mean(axis=(0, 1))
    assert 90 < middle[0] < 170 and 90 < middle[2] < 170       # half red, half blue


def test_height_colours_show_the_relief_not_the_bow():
    # A cloth sagging in a bowl, with small bumps on it: the bumps come out red wherever they sit.
    u, v = np.meshgrid(np.linspace(-1, 1, 120), np.linspace(-1, 1, 120))
    bump = np.exp(-((np.mod(u, 0.4) - 0.2) ** 2 + (np.mod(v, 0.4) - 0.2) ** 2) / 0.002) * 0.05
    points = np.stack([u.ravel(), v.ravel(), (0.4 * (u ** 2 + v ** 2) + bump).ravel()], axis=1)
    colours = splat_gl.height_colours(points, splat_gl.cloth_plane(points))
    on_bump, off_bump = bump.ravel() > 0.04, bump.ravel() < 1e-4
    red = colours[:, 0] - colours[:, 2]
    assert red[on_bump].mean() > 0.5 and red[off_bump].mean() < red[on_bump].mean() - 0.5
    # Both the bowl's middle and its rim, off the bumps, are coloured alike.
    middle, rim = off_bump & (np.hypot(u, v).ravel() < 0.3), off_bump & (np.hypot(u, v).ravel() > 0.9)
    assert np.abs(colours[middle].mean(axis=0) - colours[rim].mean(axis=0)).max() < 0.25


def test_the_single_gaussians_can_be_shown_as_ellipses_or_centres():
    ctx = _context()
    ball = _ball()
    frame = gaussian_splatting.orbit_frame(ball["positions"])
    renderer = splat_gl.SplatRenderer(ctx)
    pictures = {}
    for style in range(len(splat_gl.STYLES)):
        renderer.render([("ball", ball, None, 1.0)], (20.0, 30.0, 1.0, (0.0, 0.0)), frame, (240, 180), background=(0, 0, 0), style=style)
        pictures[style] = renderer.read().astype(float)
    covered = {style: (picture.max(axis=2) > 40).mean() for style, picture in pictures.items()}
    # Ellipses stop at one sigma and centres are dots: each covers less of the picture than the soft splat.
    assert covered[splat_gl.SOFT] > covered[splat_gl.ELLIPSES] > covered[splat_gl.CENTRES] > 0.005
    # The ellipses' dark rims split the ball into many separate Gaussians: far more edges than the soft splat.
    edges = {style: (np.abs(np.diff(p[..., 0], axis=1)) > 40).sum() for style, p in pictures.items()}
    assert edges[splat_gl.ELLIPSES] > 3 * edges[splat_gl.SOFT]


def test_the_soft_splat_can_show_its_gaussians_outlines_in_it():
    ctx = _context()
    ball = _ball()
    ball.update(sigma=np.full(ball["count"], 0.03, dtype=np.float32), radius=np.full(ball["count"], 0.03, dtype=np.float32))
    frame = gaussian_splatting.orbit_frame(ball["positions"])
    renderer = splat_gl.SplatRenderer(ctx)
    pictures = {}
    for style in (splat_gl.SOFT, splat_gl.OUTLINED):
        renderer.render([("ball", ball, None, 1.0)], (20.0, 30.0, 1.0, (0.0, 0.0)), frame, (240, 180), background=(0, 0, 0), style=style)
        pictures[style] = renderer.read().astype(float)
    soft, outlined = pictures[splat_gl.SOFT], pictures[splat_gl.OUTLINED]
    # Same picture underneath (the ball still red where no outline runs), with orange lines through it.
    orange = (outlined[..., 1] > soft[..., 1] + 40) & (outlined[..., 0] > 150)
    assert 0.005 < orange.mean() < 0.6                    # thin lines, a pixel or so wide
    assert np.median(np.abs(outlined - soft)[~orange]) < 10


def test_axes_are_drawn_as_3d_lines_coloured_by_the_way_they_point():
    ctx = _context()
    rng = np.random.default_rng(1)
    count = 300
    # Needles along x on a sheet at z = 0, and needles standing up (along z) on a sheet at z = 0.3.
    positions = np.concatenate([rng.uniform(-1, 1, (count, 3)) * (1, 1, 0), rng.uniform(-1, 1, (count, 3)) * (1, 1, 0) + (0, 0, 0.3)])
    scales = np.concatenate([np.tile((0.05, 0.005, 0.005), (count, 1)), np.tile((0.005, 0.005, 0.05), (count, 1))]).astype(np.float32)
    splat = {"positions": positions.astype(np.float32), "colours": np.full((2 * count, 3), 0.8, dtype=np.float32),
             "opacity": np.ones(2 * count, dtype=np.float32), "scales": scales,
             "quaternion": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (2 * count, 1)),
             "radius": scales.max(axis=1), "sigma": np.full(2 * count, 0.01, dtype=np.float32), "count": 2 * count}
    segments = splat_gl.long_axes(splat)
    assert segments.shape == (2 * count, 2, 6)
    assert np.allclose(segments[:count, 0, 3:], (1, 0, 0)) and np.allclose(segments[count:, 0, 3:], (0, 0, 1))
    assert np.allclose(np.linalg.norm(segments[:, 1, :3] - segments[:, 0, :3], axis=1), 2 * splat_gl.AXIS_SIGMAS * 0.05)
    # Seen from the side, the two sheets are two separate bands: red below, blue above.
    frame = (np.zeros(3), 1.0)
    renderer = splat_gl.SplatRenderer(ctx)
    renderer.render([("needles", splat, None, 1.0)], (90.0, 0.0, 1.0, (0.0, 0.0)), frame, (240, 180), background=(0, 0, 0),
                    style=splat_gl.AXES)
    image = renderer.read().astype(int)
    red_rows = np.flatnonzero(((image[..., 0] > 100) & (image[..., 2] < 60)).any(axis=1))
    blue_rows = np.flatnonzero(((image[..., 2] > 100) & (image[..., 0] < 60)).any(axis=1))
    assert len(red_rows) and len(blue_rows) and blue_rows.mean() < red_rows.mean() - 10


def test_axes_on_the_soft_splat_hide_behind_the_cloth():
    ctx = _context()
    # A solid sheet at z = 0 (flat Gaussians, longest axis along x: red) over needles standing up
    # under it (along z: blue). Seen from above, only the sheet's own axes may show.
    u, v = np.meshgrid(np.linspace(-1, 1, 60), np.linspace(-1, 1, 60))
    sheet = np.stack([u.ravel(), v.ravel(), np.zeros(u.size)], axis=1)
    needles = np.random.default_rng(2).uniform(-0.7, 0.7, (200, 3)) * (1, 1, 0) + (0, 0, -0.3)
    positions = np.concatenate([sheet, needles]).astype(np.float32)
    scales = np.concatenate([np.tile((0.04, 0.035, 0.005), (len(sheet), 1)), np.tile((0.005, 0.005, 0.06), (len(needles), 1))])
    count = len(positions)
    splat = {"positions": positions, "colours": np.full((count, 3), 0.8, dtype=np.float32), "opacity": np.ones(count, dtype=np.float32),
             "scales": scales.astype(np.float32), "quaternion": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (count, 1)),
             "radius": scales.max(axis=1).astype(np.float32), "sigma": np.full(count, 0.03, dtype=np.float32), "count": count}
    frame = (np.zeros(3), 1.0)
    renderer = splat_gl.SplatRenderer(ctx)

    def blue(style):
        renderer.render([("sheet", splat, None, 1.0)], (30.0, 60.0, 1.5, (0.0, 0.0)), frame, (240, 180), background=(0, 0, 0), style=style)
        image = renderer.read().astype(int)
        return int(((image[..., 2] > 150) & (image[..., 0] < 80) & (image[..., 1] < 80)).sum())

    assert blue(splat_gl.AXES) > 200                  # the axes alone: the needles show through
    assert blue(splat_gl.SPLAT_AXES) < 10             # on the soft splat: hidden by the sheet
