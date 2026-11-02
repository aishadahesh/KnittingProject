"""The splat viewer's renderer and the floater clean-up."""

import numpy as np

import gaussian_splatting


def write_full_splat(path, positions, colours, opacity_logit, log_scale):
    """A 3DGS-layout PLY, as nerfstudio exports: positions, SH colour, opacity and log scales."""
    properties = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = ("ply\nformat binary_little_endian 1.0\n" + f"element vertex {len(positions)}\n"
              + "".join(f"property float {name}\n" for name in properties) + "end_header\n")
    table = np.zeros(len(positions), dtype=[(name, "<f4") for name in properties])
    table["x"], table["y"], table["z"] = positions.T
    dc = (np.asarray(colours, dtype=float) - 0.5) / gaussian_splatting.SH_C0
    table["f_dc_0"], table["f_dc_1"], table["f_dc_2"] = dc.T
    table["opacity"] = opacity_logit
    table["scale_0"] = table["scale_1"] = table["scale_2"] = log_scale
    table["rot_0"] = 1.0
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        table.tofile(handle)


def test_blobs_blend_into_a_smooth_solid_rather_than_squares(tmp_path):
    rng = np.random.default_rng(0)
    positions = rng.normal(size=(4000, 3)).astype(np.float32) * 0.05
    write_full_splat(tmp_path / "ball.ply", positions, np.tile([0.9, 0.2, 0.2], (4000, 1)), np.full(4000, 3.0), np.full(4000, np.log(0.01)))
    splat = gaussian_splatting.load_splat_ply(tmp_path / "ball.ply")
    assert splat["sigma"].shape == (4000,) and np.allclose(splat["sigma"], 0.01, atol=1e-4)
    image = np.asarray(gaussian_splatting.render_splat(splat, 20.0, 30.0, 1.0, (240, 180), background=(0, 0, 0))).astype(float)
    centre = image[80:100, 110:130]
    assert centre[..., 0].mean() > 150 and centre[..., 1].mean() < 80      # solid red in the middle
    assert image[:5, :5].mean() < 5                                        # background untouched
    # Smooth: neighbouring pixels differ gently, not in hard square steps.
    steps = np.abs(np.diff(image[90, 60:180, 0]))
    assert np.percentile(steps, 90) < 60


def test_floaters_and_invisible_blobs_are_cleaned_away(tmp_path):
    rng = np.random.default_rng(1)
    body = rng.normal(size=(3000, 3)) * 0.02
    far = rng.normal(size=(200, 3)) * 0.01 + [0.5, 0.0, 0.0]            # detached floaters
    positions = np.concatenate([body, far]).astype(np.float32)
    colours = np.tile([0.8, 0.8, 0.8], (3200, 1))
    colours[:100] = 0.0                                                   # black: invisible on black
    opacity = np.full(3200, 3.0)
    opacity[100:150] = -6.0                                               # transparent
    write_full_splat(tmp_path / "noisy.ply", positions, colours, opacity, np.full(3200, np.log(0.005)))
    dropped = gaussian_splatting.clean_splat(tmp_path / "noisy.ply", tmp_path / "clean.ply", object_only=True)
    assert dropped["total"] == 3200
    assert dropped["dark"] == 100 and dropped["invisible"] == 50
    # All 200 far blobs go; a few stragglers at the body's thin edge may go with them.
    assert 200 <= dropped["detached"] <= 260
    assert dropped["kept"] == 3200 - 150 - dropped["detached"]
    cleaned = gaussian_splatting.load_splat_ply(tmp_path / "clean.ply")
    assert cleaned["count"] == dropped["kept"]
    assert cleaned["positions"][:, 0].max() < 0.2
