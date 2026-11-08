"""Reading a fabric's yarns from Gaussians coloured by the photos: grouping, the table, the openings' rims."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import fabric_colours as F  # noqa: E402


def _cloth():
    """A strip of light-blue lace beside a charcoal band, each in light and shade, with openings in the
    lace showing the blue table and a rim round each where yarn and table blend."""
    rng = np.random.default_rng(0)
    lace_xy = rng.uniform([0, 0], [10, 20], (3000, 2))
    band_xy = rng.uniform([10.5, 0], [16, 20], (1500, 2))
    shade = lambda n: rng.uniform(0.35, 1.0, (n, 1))
    lace = F.linear_to_srgb(F.srgb_to_linear(np.array([0.72, 0.77, 0.82])) * shade(3000))
    band = F.linear_to_srgb(F.srgb_to_linear(np.array([0.2, 0.21, 0.22])) * shade(1500))
    centres = np.array([[5.0, y] for y in (3.0, 9.0, 15.0)])
    hole = np.min(np.linalg.norm(lace_xy[:, None] - centres[None], axis=2), axis=1)
    table = np.array([0.08, 0.1, 0.45])
    rim_mix = 0.4
    lace[hole < 1.2] = table
    lace[(hole >= 1.2) & (hole < 1.6)] = F.linear_to_srgb(rim_mix * F.srgb_to_linear(np.array([0.72, 0.77, 0.82]))
                                                         + (1 - rim_mix) * F.srgb_to_linear(table))
    positions = np.concatenate([np.c_[lace_xy, np.zeros(3000)], np.c_[band_xy, np.zeros(1500)]])
    photo = np.concatenate([lace, band])
    truth = np.r_[np.where(hole < 1.2, 2, np.where(hole < 1.6, 3, 0)), np.ones(1500, int)]      # lace, band, table, rim
    return positions, photo, truth


def test_the_yarns_the_table_and_the_rims_are_told_apart():
    positions, photo, truth = _cloth()
    views, weight = np.full(len(photo), 8), np.ones(len(photo))
    labels = F.group_yarns(positions, photo, views, weight)
    groups, labels = F.describe_groups(labels, photo, views, weight)
    roles = [g["role"] for g in groups]
    assert roles.count("yarn") == 2 and "table" in roles
    lace, band = groups[0], groups[1]
    # Each yarn's colour: its hue exactly, its lightness that of its lit side (a little under fully lit).
    for found, made in ((lace, (0.72, 0.77, 0.82)), (band, (0.2, 0.21, 0.22))):
        made = F.srgb_to_lab(np.array(made))
        assert np.linalg.norm(found["lab"][1:] - made[1:]) < 3 and -12 < found["lab"][0] - made[0] <= 1
    # Nearly every lace and band Gaussian lands in its own yarn.
    assert (labels[truth == 0] == 0).mean() > 0.95 and (labels[truth == 1] == 1).mean() > 0.95
    # The table is a table, and the rims are no yarn of their own.
    table_groups = [k for k, g in enumerate(groups) if g["role"] == "table"]
    not_yarn = [k for k, g in enumerate(groups) if g["role"] != "yarn"]
    assert np.isin(labels[truth == 2], table_groups).mean() > 0.8          # where the table meets a rim it may be rim
    assert np.isin(labels[truth == 2], not_yarn).mean() > 0.97
    assert not np.isin(labels[truth == 3], [k for k, g in enumerate(groups) if g["role"] == "yarn" and k > 1]).any()


def test_asking_for_one_colour_merges_the_yarns():
    positions, photo, truth = _cloth()
    views, weight = np.full(len(photo), 8), np.ones(len(photo))
    groups, labels = F.describe_groups(F.group_yarns(positions, photo, views, weight, wanted=1), photo, views, weight)
    assert len(groups) == 1 and len(np.unique(labels)) == 1


def test_a_blend_of_a_yarn_and_the_table_is_an_edge():
    yarn = {"role": "yarn", "share": 0.8, "rgb": np.array([0.85, 0.85, 0.83])}
    table = {"role": "table", "share": 0.05, "rgb": np.array([0.05, 0.06, 0.3])}
    mix = F.linear_to_srgb(0.3 * F.srgb_to_linear(yarn["rgb"]) + 0.7 * F.srgb_to_linear(table["rgb"]))
    rim = {"role": "yarn", "share": 0.05, "rgb": mix}
    other = {"role": "yarn", "share": 0.1, "rgb": np.array([0.8, 0.2, 0.2])}          # a real red yarn: not a blend
    groups = [yarn, table, rim, other]
    for g in groups:
        g["lab"] = F.srgb_to_lab(g["rgb"])
    F.mark_edges(groups)
    assert rim["role"] == "edge" and abs(rim["blend_of_yarn"] - 0.3) < 0.05
    assert other["role"] == "yarn" and yarn["role"] == "yarn"
