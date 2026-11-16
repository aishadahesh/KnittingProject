"""Targeting a fabric part: every capture pattern centres on its centre, inside the whole fabric's outline."""

import numpy as np
import pytest

import gui_ur5
import scan_geometry
import ur5_robot
import workcell


class _State(dict):
    def get(self, key, default=None):
        return super().get(key, default)


def _settings(number, pattern):
    centre = workcell.part_centre(number)
    state = _State(teleop_pattern=pattern, teleop_fabric_part=number, teleop_fabric_z=centre[2],
                   teleop_fabric_corners=[[x, y, centre[2]] for x, y in workcell.fabric_corners()])
    return centre, gui_ur5._capture_settings(state)


@pytest.mark.parametrize("pattern", ["centred", "splat", "centre", "orbit"])
def test_every_pattern_centres_on_the_targeted_part(pattern):
    if not workcell.parts():
        pytest.skip("no fabric parts mapped")
    centre, settings = _settings(0, pattern)
    assert settings["subject_xy"] == pytest.approx(centre[:2])
    plan = scan_geometry.build_capture_plan(list(ur5_robot.DEFAULT_HOME_JOINTS), **settings)
    assert np.asarray(plan.subject)[:2] == pytest.approx(centre[:2], abs=1e-6)
    assert plan.viewpoints


def test_no_target_leaves_the_subject_to_the_outline():
    assert gui_ur5._part_subject(_State()) is None
