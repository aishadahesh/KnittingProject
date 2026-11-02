"""The stitch grid on a photo: cloth millimetres carried into a camera's picture, the grid drawn there,
and each stitch's place read from the picture as a post or an opening."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_fit  # noqa: E402
import strand_equations  # noqa: E402
from test_lace_lattice import _lace_splat  # noqa: E402

LENS = {"fl_x": 600.0, "fl_y": 600.0, "cx": 320.0, "cy": 240.0, "w": 640.0, "h": 480.0}


def _scene():
    """The synthetic lace (3 mm stitches, 2 mm rows, an opening in every 3 x 6 repeat), mapped as a
    scan would be, with the splat's frame the cloth's own (mm), and a camera 60 mm straight above."""
    splat = _lace_splat(0.0, points=150_000)
    flat, centre, axes = strand_equations.lay_flat(splat, (0.0, 0.0, 1.0), return_frame=True)
    extraction = strand_equations.extract(flat, 1.0, (0.0, 0.0, 1.0))
    maps = extraction.maps
    import lace_lattice
    prepared = {"extraction": extraction, "mm_per_unit": 1.0, "flat_centre": centre, "flat_axes": axes, "low": extraction.map_origin,
                "cloth": lace_lattice.cloth_region(maps["support"]), "bow": lace_lattice.bow_map(maps["support"], maps["height"])}
    middle = scan_fit.to_splat_frame(prepared, np.array([[0.0, 0.0, 0.0]]))[0]
    up = scan_fit.to_splat_frame(prepared, np.array([[0.0, 0.0, 1.0]]))[0] - middle
    eye = middle + 60.0 * up / np.linalg.norm(up)
    forward = -up / np.linalg.norm(up)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])                      # world -> camera rows (OpenCV axes)
    viewmat = np.eye(4)
    viewmat[:3, :3], viewmat[:3, 3] = rotation, -rotation @ eye
    camera = {"viewmat": viewmat, "path": None, "name": "above"}
    # The picture: dark blue table, white where yarn is.
    photo = np.zeros((480, 640, 3), dtype=np.uint8)
    photo[:] = (0, 0, 54)
    xy, ahead = scan_fit.project(splat["positions"].astype(float), camera, LENS)
    inside = ahead & (xy[:, 0] >= 0) & (xy[:, 0] < 640) & (xy[:, 1] >= 0) & (xy[:, 1] < 480)
    photo[xy[inside, 1].astype(int), xy[inside, 0].astype(int)] = (230, 230, 220)
    from scipy import ndimage
    white = ndimage.binary_dilation(photo[..., 0] > 100, iterations=2)
    photo[white] = (230, 230, 220)
    return splat, prepared, camera, photo, extraction


def test_cloth_points_land_where_the_camera_sees_them():
    splat, prepared, camera, photo, extraction = _scene()
    xy, ahead = scan_fit.project(scan_fit.to_splat_frame(prepared, np.array([[0.0, 0.0, 0.0]])), camera, LENS)
    assert ahead.all() and np.allclose(xy[0], (320.0, 240.0), atol=1.0)        # straight below the camera: the middle


def test_the_photo_vote_finds_the_openings():
    splat, prepared, camera, photo, extraction = _scene()
    # The made lattice, in the cloth frame the map uses: made -> laid flat -> the map's own axes.
    def to_cloth(points):
        flat = (np.asarray(points, dtype=float) - prepared["flat_centre"]) @ prepared["flat_axes"].T
        return (flat - extraction.centre) @ extraction.axes.T

    made = to_cloth([[1.0, 0.0, 0.0]])[0] - to_cloth([[0.0, 0.0, 0.0]])[0]
    angle = math.degrees(math.atan2(made[1], made[0])) % 180.0
    origin = to_cloth([[0.0, 0.0, 0.0]])[0]                         # the made lattice's (0, 0) in the cloth frame
    a = math.radians(angle)
    u0, v0 = origin[0] * math.cos(a) + origin[1] * math.sin(a), -origin[0] * math.sin(a) + origin[1] * math.cos(a)
    lattice = {"angle_deg": angle, "shift": [u0, v0], "stitch_width": 3.0, "row_height": 2.0}
    places = scan_fit.stitch_places(prepared, lattice)
    assert len(places) > 20
    share = scan_fit.photo_openings(prepared, lattice, places, [camera], LENS, photo[None], [0], (0.0, 0.0, 0.21))
    # Each 9 x 12 mm repeat has a round opening (1.6 mm radius) at (4.5, 6) mm: the stitch places
    # nearest it see the table far more often than the rest.
    centre = np.array([[column * 3.0, (row + 0.5) * 2.0] for column, row in places])
    offset = np.mod(centre - np.array([4.5, 6.0]) + np.array([4.5, 6.0]), np.array([9.0, 12.0])) - np.array([4.5, 6.0])
    near = np.linalg.norm(offset, axis=1) < 2.0          # the nearest posts stand 1.8 mm from its middle
    assert near.any() and (~near).any()
    assert share[near].mean() > share[~near].mean() + 0.2
    picture = scan_fit.grid_picture(prepared, lattice, camera, LENS, photo,
                                    [p for p, s in zip(places, share) if s > 0.35])
    assert picture.size == (640, 480) and np.asarray(picture).std() > 0
