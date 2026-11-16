"""Simulation-time filtering and shared v4 configuration without ROS or Isaac."""
import copy

import pytest

from src.simulation_config.joint_filter import JointTargetFilter
from src.simulation_config.robot_profile import build_simulation_config, default_robot_profile


def test_lowpass_is_invariant_to_subdividing_simulation_time_and_reset():
    filters = [JointTargetFilter({'j': {'time_constant_s': .04}}, {'j'}) for _ in range(2)]
    for target_filter in filters:
        target_filter.apply('j', 0, 0, 99, .01)
    expected = filters[0].apply('j', 1, 0, 99, .02)
    filters[1].apply('j', 1, 0, 99, .01)
    assert filters[1].apply('j', 1, 0, 99, .01) == pytest.approx(expected)
    filters[0].reset()
    assert filters[0].apply('j', 7, 0, 99, .01) == 7


def test_explicit_zero_lookahead_ignores_reconstructed_sdk_velocity():
    target_filter = JointTargetFilter({'j': {}}, {'j', 'legacy'})
    assert target_filter.apply('j', 2, 1000, 102, .01) == 2
    assert target_filter.apply('legacy', 2, 1000, 102, .01) == 102


@pytest.mark.parametrize('settings', [
    {'bad': {}}, {'j': {'time_constant_s': -1}},
    {'j': {'velocity_lookahead_s': float('nan')}}, {'j': {'unknown': 1}},
])
def test_invalid_filter_config(settings):
    with pytest.raises(ValueError):
        JointTargetFilter(settings, {'j'})


@pytest.mark.parametrize('dt', [0, -1, float('nan'), float('inf')])
def test_invalid_simulation_interval_rejected(dt):
    with pytest.raises(ValueError, match='positive finite simulation dt'):
        JointTargetFilter({'j': {}}, {'j'}).apply('j', 0, 0, 0, dt)


def test_robot_profile_merges_gains_and_replaces_filter_selection_without_mutation():
    data = default_robot_profile().as_dict()
    baseline = build_simulation_config(data)
    data['simulation_overrides'] = {
        'arm_right_joint_7_control': {'stiffness': 1200.},
        'head_joint_1_control': {'damping': 80.},
        'torso_joint_4_control': {'stiffness': 1600.},
        'joint_command_filters': {'astribot_arm_right_joint_7': {'time_constant_s': .02}},
    }
    original = copy.deepcopy(data)
    config = build_simulation_config(data)
    for name, updates in data['simulation_overrides'].items():
        if name != 'joint_command_filters':
            assert config[name] == {**baseline[name], **updates}
    assert config['joint_command_filters'] == data['simulation_overrides']['joint_command_filters']
    assert data == original
    config['joint_command_filters']['astribot_arm_right_joint_7']['time_constant_s'] = 1.
    assert data == original
    data['simulation_overrides']['joint_command_filters'] = {}
    assert build_simulation_config(data)['joint_command_filters'] == {}
    data['simulation_overrides']['joint_command_filters'] = {'unknown_joint': {}}
    with pytest.raises(ValueError, match='unknown joint'):
        build_simulation_config(data)
