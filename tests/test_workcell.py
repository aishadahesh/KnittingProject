"""The measured work cell: the fabric lies on the carton, the carton on the table, at the measured sizes."""

import pytest

import workcell


def _inside(inner, outer):
    return all(outer[axis][0] <= inner[axis][0] < inner[axis][1] <= outer[axis][1] for axis in ("x", "y"))


def test_the_fabric_lies_on_the_carton_and_the_carton_on_the_table():
    cell = workcell.cell()
    assert _inside(cell["fabric"], cell["carton"])
    assert _inside(cell["carton"], cell["table"])
    assert cell["carton"]["top_z_at_origin"] > cell["table"]["top_z_at_origin"]


def test_the_sizes_are_the_measured_ones():
    carton = workcell.cell()["carton"]
    assert workcell.fabric_size() == pytest.approx((0.68, 0.38), abs=0.03)
    assert (carton["x"][1] - carton["x"][0], carton["y"][1] - carton["y"][0]) == pytest.approx((0.86, 0.38), abs=0.01)


def test_the_fabric_centre_is_on_the_tilted_carton_top():
    x, y, z = workcell.fabric_center()
    sx, sy = workcell.cell()["surface_slope"]
    assert z == pytest.approx(sx * x + sy * y + workcell.cell()["carton"]["top_z_at_origin"])
    assert workcell.surface_z(0.0, 0.0, 1.0) == pytest.approx(sy)
