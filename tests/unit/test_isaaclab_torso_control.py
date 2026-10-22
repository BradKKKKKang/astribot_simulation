"""Check isolated feedforward and validation without starting ROS or Isaac."""
from types import SimpleNamespace

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
    env.robot = SimpleNamespace(num_joints=3)
    env.robot.set_joint_position_target = lambda target: None
    writes = []
    env.robot.set_joint_effort_target = lambda target, joint_ids: writes.append((target.clone(), joint_ids))
    env._torso_joint_1_gravity_effort = lambda: 30.0
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
