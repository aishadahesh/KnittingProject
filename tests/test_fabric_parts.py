"""The fabric's numbered parts: found from colour edges, numbered from the +x end, evens far and odds near."""

import numpy as np
import pytest

import fabric_parts

RES = 0.001
X0, Y_TOP = -0.40, 0.80
FABRIC = {"x": [-0.38, 0.36], "y": [0.38, 0.78]}
SEAM = 0.56
WIDTH = 0.028
FIRST = -0.30


def _lab_image(columns=20, label_band=True):
    """Two rows of 2.8 cm panels in alternating colours, a label band at -x and a plain hem at +x."""
    xs = X0 + np.arange(int(0.80 / RES)) * RES
    ys = Y_TOP - np.arange(int(0.45 / RES)) * RES
    image = np.full((len(ys), len(xs), 3), (60.0, 128.0, 128.0))
    rng = np.random.default_rng(1)
    for k in range(columns):
        inside = (xs >= FIRST + k * WIDTH) & (xs < FIRST + (k + 1) * WIDTH)
        for far, rows in ((True, ys >= SEAM), (False, ys < SEAM)):
            image[np.ix_(rows, inside)] = (rng.uniform(80, 200), rng.uniform(100, 160), rng.uniform(100, 160))
    if label_band:
        image[:, (xs >= FIRST - 0.017) & (xs < FIRST)] = (30.0, 128.0, 128.0)
    return image


def test_the_parts_are_found_and_numbered_from_the_plus_x_end():
    parts = fabric_parts.find_parts(_lab_image(), RES, X0, Y_TOP, FABRIC)
    assert [part["number"] for part in parts] == list(range(40))
    last = FIRST + 20 * WIDTH
    assert parts[0]["row"] == "far" and parts[1]["row"] == "near"
    assert parts[0]["x"] == pytest.approx([last - WIDTH, last], abs=0.003)
    assert parts[1]["x"] == pytest.approx(parts[0]["x"])
    assert parts[39]["x"] == pytest.approx([FIRST, FIRST + WIDTH], abs=0.003)
    assert parts[0]["y"][0] == pytest.approx(SEAM, abs=0.003)


def test_a_row_with_the_wrong_number_of_columns_is_refused():
    with pytest.raises(ValueError, match="regular columns"):
        fabric_parts.find_parts(_lab_image(columns=19), RES, X0, Y_TOP, FABRIC)
