"""Scan Mode's camera captures, at the resolution the camera can actually show.

Two defects lived here, both invisible at the default capture size and both
obvious at the 2000 px one the resolution slider now allows:

* The whole-grid ("natural") capture textured itself from the grid preview
  composite, which caps the *entire* layout at 1400 px. Across a five-cell row
  at 64 repeats that is ~4 px per repeat, so the stitches were gone from the
  source before the camera looked at it and the capture came out a flat smear.
* The lens vignette's margins and blur were pixel constants tuned against
  CAMERA_IMAGE_SIZE, so a capture at twice that size got half the relative
  feathering -- a hard black oval instead of a lens.

No GL and no MuJoCo: render_camera_image is pure PIL/numpy over a plan.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import fabric_scanner as fs  # noqa: E402
from scanner_core import TiledFabricTexture  # noqa: E402

REPEATS = 64
PREVIEW_MAX_DIM = 1400  # the cap _scanner_generate_tiled_layout builds the composite at


def _knit_tile(width=480, height=105):
    """Stands in for one exact-period fabric crop: strong, countable structure."""
    tile = Image.new("RGB", (width, height), (26, 18, 12))
    draw = ImageDraw.Draw(tile)
    for i in range(6):
        x = (i + 0.5) * width / 6
        draw.arc([x - width / 14, 4, x + width / 14, height - 4], 200, 340, fill=(196, 128, 72), width=7)
        draw.arc([x - width / 14, -height // 2, x + width / 14, height // 2], 20, 160, fill=(150, 96, 54), width=7)
    return tile


def _plan(rows=4, cols=5):
    args = SimpleNamespace(
        rows=rows, cols=cols, number_of_angles=6,
        width=max(0.06, cols * 2 * 0.045), length=max(0.06, rows * 4 * 0.040),
        edge_margin=0.004, square_margin=0.006, surface_wave=0.003,
        view_radius=0.018, angle_lift=0.014, approach_lift=0.040,
        center=[-0.45, -0.08, 0.30], max_span=fs.DEFAULT_MAX_SPAN.tolist(),
        speed=1.5, dwell=0.0, add_camera=True, save_images=False,
        image_dir=".", image_every="view", capture_mode="natural", camera_zoom=1.0,
        palette=None, cell_color_sets=None, model_json="", model_curves=None,
        cell_model_curves=None,
        pattern_repeat_rows=REPEATS, pattern_repeat_cols=REPEATS,
        pattern_repeat_spacing_x=1.0, pattern_repeat_spacing_y=1.0,
        batch_texture_width=420, batch_texture_height=340,
        scanner_lighting=None, display_batch_colors=None,
    )
    return fs.build_plan(args)


def _attach_textures(plan, per_cell=True):
    """The imagery Scan Mode hands the scanner: lazy per-cell tiles, plus the
    capped preview composite built from them."""
    tiles = [
        TiledFabricTexture(_knit_tile(), REPEATS, REPEATS)
        for _ in range(plan.grid_rows * plan.grid_cols)
    ]
    cell_px = tiles[0].size
    scale = min(
        1.0,
        PREVIEW_MAX_DIM / (plan.grid_cols * cell_px[0]),
        PREVIEW_MAX_DIM / (plan.grid_rows * cell_px[1]),
    )
    slot = (max(1, int(cell_px[0] * scale)), max(1, int(cell_px[1] * scale)))
    composite = Image.new("RGB", (plan.grid_cols * slot[0], plan.grid_rows * slot[1]))
    cell_preview = tiles[0].rasterize(int(np.clip(max(slot) * 8, 256, 3072))).resize(slot, Image.LANCZOS)
    for row in range(plan.grid_rows):
        for col in range(plan.grid_cols):
            composite.paste(cell_preview, (col * slot[0], row * slot[1]))
    plan.rendered_fabric_image = composite
    plan.rendered_fabric_image_is_full_layout = True
    plan.rendered_fabric_image_lit = True
    if per_cell:
        plan.scan_tiled_pattern_images = tiles
    return plan



def _flat_plan(level=(150, 150, 150)):
    """A plan textured in one uniform colour: any variation is the vignette."""
    plan = _plan()
    plan.rendered_fabric_image = Image.new("RGB", (64, 64), level)
    plan.rendered_fabric_image_is_full_layout = True
    plan.rendered_fabric_image_lit = True
    plan.scan_tiled_pattern_images = [
        TiledFabricTexture(Image.new("RGB", (64, 64), level), REPEATS, REPEATS)
        for _ in range(plan.grid_rows * plan.grid_cols)
    ]
    return plan

def _capture(plan, size, mode=None, station_cell=(0, 0)):
    station_id = next(
        i for i, cell in enumerate(plan.station_cells)
        if tuple(int(v) for v in cell) == tuple(station_cell)
    )
    idx = next(
        i for i, sid in enumerate(plan.station_ids)
        if int(sid) == station_id and plan.view_names[i] not in ("approach", "retreat")
    )
    return fs.render_camera_image(
        plan, plan.poses[idx][:3], idx, station_id, plan.view_names[idx],
        target_pose=plan.poses[idx],
        capture_mode=mode or fs.CAMERA_CAPTURE_NATURAL,
        camera_zoom=1.0, image_size=size,
    )


def _detail(image):
    """High-frequency energy: how much stitch structure survived into the image."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    fabric = gray[gray > 12.0]
    if fabric.size < 1000:
        return 0.0
    return float(np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean())


def test_natural_capture_beats_the_preview_composite_resolution():
    """Per-cell tiles carry stitch detail the 1400 px composite cannot hold."""
    from_preview = _capture(_attach_textures(_plan(), per_cell=False), (2000, 1500))
    from_cells = _capture(_attach_textures(_plan(), per_cell=True), (2000, 1500))

    assert _detail(from_cells) > _detail(from_preview) * 1.5


def test_natural_capture_does_not_glue_whole_cells():
    """The cap keeps a lazy texture from being stamped out at full size.

    One cell at 64x64 repeats glues to 30720x6720 -- 600 MB, twenty times over
    per frame. rasterize() is asked for the quad's size instead.
    """
    requested = []

    class RecordingTexture(TiledFabricTexture):
        def rasterize(self, max_dim=None):
            requested.append(max_dim)
            return super().rasterize(max_dim)

    plan = _attach_textures(_plan(), per_cell=True)
    plan.scan_tiled_pattern_images = [
        RecordingTexture(_knit_tile(), REPEATS, REPEATS)
        for _ in range(plan.grid_rows * plan.grid_cols)
    ]
    _capture(plan, (2000, 1500))

    assert requested, "the per-cell textures were never used"
    assert all(cap is not None for cap in requested)
    assert max(requested) <= 2 * 2000


def _vignette_edge_fraction(gray, width, height):
    """Where the vignette gives way to open frame, as a fraction of the image.

    Walked along the bottom-left diagonal, which is empty background in this
    scene, and expressed relatively so the answer is comparable across sizes.
    """
    # Sampled between the vignette's edge and the fabric's: further out along
    # this diagonal is the neighbouring cell, not open background.
    plateau = float(np.median([
        gray[int(height * (1 - f)), int(width * f)] for f in (0.16, 0.20, 0.24)
    ]))
    for frac in np.linspace(0.004, 0.15, 400):
        if gray[int(height * (1 - frac)), int(width * frac)] >= 0.5 * plateau:
            return float(frac)
    raise AssertionError("the vignette never opens up along the diagonal")


def test_vignette_is_the_same_lens_at_any_capture_size():
    """A bigger capture gets a bigger lens, not a harder-edged one.

    With the margins and blur left as pixel constants, a 2048 px capture's dark
    oval reached 23% further into the frame than the 1024 px one it was tuned
    for, and with proportionally less feathering -- a stamped-on oval rather
    than a lens.
    """
    plan = _flat_plan()
    fractions = []
    for size in ((1024, 768), (2048, 1536)):
        gray = np.asarray(_capture(plan, size).convert("L"), dtype=np.float32)
        fractions.append(_vignette_edge_fraction(gray, *size))

    assert fractions[0] == pytest.approx(fractions[1], abs=0.005)

def _resolved_row_pitch(image):
    """Pixels between consecutive stitch rows in the middle of the capture.

    Autocorrelation of the row-brightness profile: the first strong peak is the
    fabric's vertical repeat as the camera actually resolved it. Returns None
    when the capture holds no periodic structure at all.
    """
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    height, width = gray.shape
    band = gray[int(height * 0.35):int(height * 0.65), int(width * 0.40):int(width * 0.60)]
    profile = band.mean(axis=1)
    profile = profile - profile.mean()
    correlation = np.correlate(profile, profile, "full")[len(profile) - 1:]
    correlation /= max(float(correlation[0]), 1e-9)
    for lag in range(3, len(correlation) - 1):
        if correlation[lag] > correlation[lag - 1] and correlation[lag] >= correlation[lag + 1]:
            if correlation[lag] > 0.35:
                return lag
    return None


def test_natural_capture_resolves_individual_stitch_rows():
    """End to end: a single capture shows stitches, not an average colour.

    Two things had to be true at once. The camera has to frame the cell it is
    at -- a fixed 74 degree field of view put 64 repeats across ~450 px, or 7 px
    per row -- and the texture behind it has to carry that detail, which the
    1400 px grid preview composite could not.
    """
    plan = _attach_textures(_plan(), per_cell=True)
    pitch = _resolved_row_pitch(_capture(plan, (2000, 1500)))

    assert pitch is not None, "the capture has no resolvable fabric structure"
    assert pitch >= 14


def test_natural_capture_fills_the_frame_with_the_cell_it_is_at():
    """The selected cell covers the frame, including at the edge of the grid.

    A corner cell is the case that exposed this: framed to contain its long
    axis, the short one left a third of the capture as empty background beside
    the fabric, so a capture that was centred on the cell did not look it.
    """
    target = (0, 0)  # the corner cell -- fabric on two sides only
    plan = _plan()
    plan.rendered_fabric_image = Image.new("RGB", (64, 64), (90, 90, 90))
    plan.rendered_fabric_image_is_full_layout = True
    plan.rendered_fabric_image_lit = True
    plan.scan_tiled_pattern_images = [
        TiledFabricTexture(
            Image.new("RGB", (64, 64), (220, 40, 40) if (r, c) == target else (90, 90, 90)),
            REPEATS, REPEATS,
        )
        for r in range(plan.grid_rows) for c in range(plan.grid_cols)
    ]

    width, height = 1024, 768
    image = _capture(plan, (width, height), station_cell=target)
    pixels = np.asarray(image.convert("RGB")).astype(int)
    is_target = (pixels[:, :, 0] > 90) & (pixels[:, :, 0] > pixels[:, :, 1] * 2)

    # The middle of the frame is the target cell, all of it.
    middle = is_target[int(height * 0.25):int(height * 0.75), int(width * 0.25):int(width * 0.75)]
    assert middle.mean() > 0.99

    # And it is centred: the cell's centroid sits on the frame's centre.
    ys, xs = np.where(is_target)
    assert abs(float(xs.mean()) - width / 2) < width * 0.05
    assert abs(float(ys.mean()) - height / 2) < height * 0.05


def test_single_capture_targets_the_cell_the_user_picked():
    """Row/col must survive the robot's serpentine station order.

    Stations run along the scan path, so odd rows are laid out right to left.
    Indexing them row-major sent a request for row 2, col 1 of a five-wide grid
    to row 2, col 5 -- and the capture labelled itself with the cell it went
    to, so nothing looked wrong about the wrong square.
    """
    from embedded_scanner import EmbeddedMujocoScanner

    plan = _plan()
    # Serpentine really is the layout, otherwise this test proves nothing.
    assert tuple(plan.station_cells[plan.grid_cols]) != (1, 0)

    scanner = EmbeddedMujocoScanner.__new__(EmbeddedMujocoScanner)
    scanner.plan = plan
    scanner.scanner = fs
    scanner.target_index = 0

    for row in range(plan.grid_rows):
        for col in range(plan.grid_cols):
            target_index, station_id = scanner._single_target_index(row, col, 0)
            assert tuple(int(v) for v in plan.station_cells[station_id]) == (row, col)
            assert int(plan.station_ids[target_index]) == station_id
