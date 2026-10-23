"""Check isolated feedforward and validation without starting ROS or Isaac."""
from types import SimpleNamespace
import re

import pytest
import torch

from tests.unit.test_isaaclab_gripper import load_env_class


@pytest.mark.parametrize('fixed_base,expected_index', [(True, 1), (False, 7)])
def test_gravity_force_uses_dof_index_and_floating_base_offset(monkeypatch, fixed_base, expected_index):
    cls = load_env_class(monkeypatch)
    env = object.__new__(cls)
    env.joint_name_to_index = {'astribot_torso_joint_1': 1}
    forces = torch.arange(10, dtype=torch.float32).reshape(1, -1)
    env.robot = SimpleNamespace(
        is_fixed_base=fixed_base,
        root_physx_view=SimpleNamespace(get_gravity_compensation_forces=lambda: forces),
    )
    assert env._torso_joint_1_gravity_effort().item() == expected_index


@pytest.mark.parametrize('enabled,scale,expected', [(True, 0.5, 15.0), (True, 0, 0), (False, 1, 0)])
def test_feedforward_is_scaled_and_written_only_to_torso(monkeypatch, enabled, scale, expected):
    cls = load_env_class(monkeypatch)
    env = object.__new__(cls)
    env.device = 'cpu'
    env.gravity_compensation = enabled
    env.torso_joint_1_control = {'gravity_compensation_scale': scale}
    env.joint_name_to_index = {'astribot_torso_joint_1': 1}
    env.controller_mode = []
    env.robot = SimpleNamespace(num_joints=3, is_fixed_base=True,
                               root_physx_view=SimpleNamespace(
                                   get_gravity_compensation_forces=lambda: torch.tensor([[0., 30., 0.]])))
    env.robot.set_joint_position_target = lambda target: None
    writes = []
    env.robot.set_joint_effort_target = lambda target, joint_ids: writes.append((target.clone(), joint_ids))
    env._apply_control()
    assert len(writes) == 1
    assert writes[0][1] == [1]
    assert writes[0][0].shape == (1, 1)
    assert writes[0][0].item() == expected


@pytest.mark.parametrize('settings', [
    {'stiffness': -1}, {'damping': float('nan')},
    {'gravity_compensation_scale': float('inf')}, {'effort_limit': 0}, {'unknown': 1},
])
def test_invalid_control_settings_fail(monkeypatch, settings):
    cls = load_env_class(monkeypatch)
    with pytest.raises(ValueError):
        cls._validate_torso_joint_1_control(settings)


def test_switching_trials_updates_physics_and_effort_diagnostics_only_for_torso(monkeypatch):
    cls = load_env_class(monkeypatch)
    env = object.__new__(cls)
    env.device = 'cpu'
    env.joint_name_to_index = {'astribot_torso_joint_1': 2}
    actuator = SimpleNamespace(**{key: torch.zeros((1, 1)) for key in
                                 ('stiffness', 'damping', 'effort_limit', 'effort_limit_sim')})
    writes = []
    env.robot = SimpleNamespace(actuators={'torso_joint_1_override': actuator})
    for name in ('write_joint_stiffness_to_sim', 'write_joint_damping_to_sim',
                 'write_joint_effort_limit_to_sim', 'set_joint_effort_target'):
        setattr(env.robot, name, lambda value, joint_ids, name=name: writes.append((name, value.clone(), joint_ids)))
    env.configure_torso_joint_1({'stiffness': 3200, 'damping': 160, 'gravity_compensation_scale': 0.5})
    assert actuator.stiffness.item() == 3200
    assert actuator.damping.item() == 160
    assert actuator.effort_limit.item() == actuator.effort_limit_sim.item() == 400
    assert all(ids == [2] for _, _, ids in writes)
    assert writes[-1][1].item() == 0  # Clear the previous trial's feedforward.


@pytest.mark.parametrize('fixed_base', [True, False])
def test_three_torso_scales_use_one_force_snapshot_and_clear_disabled_joint(monkeypatch, fixed_base):
    cls = load_env_class(monkeypatch)
    env = object.__new__(cls)
    env.device = 'cpu'
    env.gravity_compensation = True
    env.controller_mode = []
    env.joint_name_to_index = {f'astribot_torso_joint_{i}': i for i in (1, 2, 3)}
    for i, scale in enumerate((1.1, 0.9, 0), start=1):
        setattr(env, f'torso_joint_{i}_control', {'gravity_compensation_scale': scale})
    forces = torch.zeros((1, 10))
    offset = 0 if fixed_base else 6
    forces[0, 1 + offset] = -50
    forces[0, 2 + offset] = 80
    snapshots, writes = [], []
    def gravity():
        snapshots.append(True)
        return forces
    env.robot = SimpleNamespace(
        num_joints=4, is_fixed_base=fixed_base,
        root_physx_view=SimpleNamespace(get_gravity_compensation_forces=gravity),
        set_joint_position_target=lambda value: None,
        set_joint_effort_target=lambda value, joint_ids: writes.append((value.item(), joint_ids)),
    )
    env._apply_control()
    assert len(snapshots) == 1
    assert [ids for _, ids in writes] == [[1], [2], [3]]
    assert [value for value, _ in writes] == pytest.approx([-55, 72, 0])


@pytest.mark.parametrize('name,settings', [
    ('torso_joint_2_control', {'stiffness': -1}),
    ('torso_joint_3_control', {'damping': float('nan')}),
    ('gripper_left_control', {'velocity_limit_sim': 0}),
    ('gripper_right_control', {'velocity_limit_sim': float('inf')}),
    ('gripper_left_control', {'velocity_limit': 16}),
    ('head_control', {}),
])
def test_invalid_torso_and_gripper_overrides_fail(monkeypatch, name, settings):
    cls = load_env_class(monkeypatch)
    with pytest.raises(ValueError):
        cls._validate_joint_control(name, settings)


def test_gripper_trial_updates_all_six_signed_dofs_and_preserves_other_side(monkeypatch):
    cls = load_env_class(monkeypatch)
    env = object.__new__(cls)
    env.device = 'cpu'
    names = [f'astribot_gripper_{side}_joint_{suffix}'
             for side in ('left', 'right') for suffix in cls._GRIPPER_JOINT_DIRECTIONS]
    env.joint_name_to_index = {name: i for i, name in enumerate(names)}
    actuator = SimpleNamespace(**{key: torch.zeros((1, 6)) for key in (
        'stiffness', 'damping', 'effort_limit', 'effort_limit_sim', 'velocity_limit', 'velocity_limit_sim')})
    writes = []
    env.robot = SimpleNamespace(actuators={'gripper_left_override': actuator})
    for method in ('write_joint_stiffness_to_sim', 'write_joint_damping_to_sim',
                   'write_joint_effort_limit_to_sim', 'write_joint_velocity_limit_to_sim',
                   'set_joint_effort_target'):
        setattr(env.robot, method, lambda value, joint_ids, method=method:
                writes.append((method, value.clone(), joint_ids)))
    env.configure_joint_control('gripper_left_control', {'velocity_limit_sim': 16})
    assert all(ids == list(range(6)) and value.shape == (1, 6) for _, value, ids in writes)
    assert actuator.velocity_limit_sim.tolist() == [[16] * 6]
    assert actuator.velocity_limit.tolist() == [[16] * 6]
    assert writes[-1][1].count_nonzero() == 0
    env.configure_joint_control('gripper_left_control', {})
    assert actuator.velocity_limit_sim.tolist() == [[1] * 6]  # No trial leakage.


def test_actuator_patterns_partition_torso_and_both_grippers(monkeypatch):
    cls = load_env_class(monkeypatch)
    excluded = '|'.join(cls._CONTROL_PATTERNS.values())
    all_other = f'^(?!(?:{excluded})$).*$'
    names = [f'astribot_torso_joint_{i}' for i in (1, 2, 3, 4)]
    names += [f'astribot_gripper_{side}_joint_{suffix}'
              for side in ('left', 'right') for suffix in cls._GRIPPER_JOINT_DIRECTIONS]
    names += ['astribot_head_joint_1', 'astribot_arm_left_joint_1']
    for joint in names:
        groups = [name for name, pattern in cls._CONTROL_PATTERNS.items()
                  if re.fullmatch(pattern, joint)]
        assert len(groups) + bool(re.fullmatch(all_other, joint)) == 1
