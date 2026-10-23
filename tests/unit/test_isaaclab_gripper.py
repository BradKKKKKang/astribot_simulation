"""Exercise real control methods with CPU tensors and no ROS or Isaac app."""

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SIGNS = {'L1': -1, 'L2': -1, 'L11': 1, 'R1': 1, 'R2': 1, 'R11': -1}


def load_env_class(monkeypatch):
    """Stub only runtime dependencies; retain the real base compensation method."""
    import sys

    def stub(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    def load(name, relative_path):
        spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    monkeypatch.delenv('ROS_VERSION', raising=False)
    stub('cv_bridge', CvBridge=object)
    stub('geometry_msgs.msg', WrenchStamped=object, Quaternion=object)
    stub('sensor_msgs.msg', Imu=object, PointCloud2=object, PointField=object)
    stub('std_msgs.msg', Header=object)
    stub('src.simu_utils.simu_common_tools', SimuCommonTools=object,
         astribot_simu_log=lambda *args: None)
    stub('src.simu_utils.robot_ros_interface', MultiRobotRosInterface=object)
    base = load('_gripper_test_base', 'src/astribot_envs/astribot_base_env.py')
    monkeypatch.setitem(sys.modules, 'src.astribot_envs.astribot_base_env', base)
    stub('isaaclab.app', AppLauncher=object)
    module = load('_gripper_test_isaac', 'src/astribot_envs/astribot_isaaclab_env.py')
    return module.AstribotIsaacLabEnv


@pytest.fixture
def env(monkeypatch):
    cls = load_env_class(monkeypatch)
    instance = object.__new__(cls)
    names = ['astribot_arm_left_joint_1'] + [
        f'astribot_gripper_{side}_joint_{suffix}'
        for side in ('left', 'right') for suffix in SIGNS
    ]
    instance.joint_name_to_index = {name: i for i, name in enumerate(names)}
    instance.joint_names_all = [names[0]] + [
        f'astribot_gripper_{side}_joint_L1' for side in ('left', 'right')
    ]
    instance.device = 'cpu'
    instance.real_time_fps = 100
    instance.controller_mode = [1, 1, 1]
    instance.joint_velocity_command_all = [0.0, 0.0, 0.0]
    instance.robot = SimpleNamespace(
        num_joints=len(names),
        data=SimpleNamespace(**{
            key: torch.zeros(1, len(names))
            for key in ('joint_pos', 'joint_vel', 'joint_acc', 'applied_torque')
        }),
    )
    instance.robot.set_joint_position_target = lambda value: setattr(
        instance.robot, 'targets', value.clone()
    )
    return instance


@pytest.mark.parametrize('left,right', [(0, 100), (25, 75), (50, 50), (100, 0), (-5, 120)])
def test_position_commands_and_feedback(env, left, right):
    env.joint_position_command_all = [0.37, left, right]
    env._apply_control()
    assert env.robot.targets[0, 0].item() == pytest.approx(0.37)
    env.robot.data.joint_pos = env.robot.targets.clone()
    for side, command in [('left', left), ('right', right)]:
        expected = max(0, min(100, command))
        for suffix, sign in SIGNS.items():
            name = f'astribot_gripper_{side}_joint_{suffix}'
            idx = env.joint_name_to_index[name]
            assert env.robot.targets[0, idx].item() == pytest.approx(sign * expected * 0.0093)
            assert env.get_joint_position(name) == pytest.approx(expected)


def test_compensation_uses_dataset_units_before_clamping(env):
    env.joint_position_command_all = [0.37, 50, 99]
    env.joint_velocity_command_all = [0, 20, 20]
    env._apply_control()
    env.robot.data.joint_pos = env.robot.targets.clone()
    assert env.get_joint_position(env.joint_names_all[1]) == pytest.approx(52)
    assert env.get_joint_position(env.joint_names_all[2]) == pytest.approx(100)


def test_velocity_mode_uses_each_grippers_feedback_in_dataset_units(env):
    env.joint_position_command_all = [0.37, 50, 75]
    env._apply_control()
    env.robot.data.joint_pos = env.robot.targets.clone()
    env.controller_mode = [1, 2, 2]
    env.joint_velocity_command_all = [0, 20, -30]
    env._apply_control()
    env.robot.data.joint_pos = env.robot.targets.clone()
    assert env.get_joint_position(env.joint_names_all[1]) == pytest.approx(52)
    assert env.get_joint_position(env.joint_names_all[2]) == pytest.approx(72)


def test_sdk_derived_velocity_does_not_shift_gripper_targets_when_compensation_disabled(env):
    env.robot_list = ['astribot_arm_left', 'astribot_gripper_left', 'astribot_gripper_right']
    env.robot_joint_map = {part: [joint] for part, joint in zip(env.robot_list, env.joint_names_all)}
    env.vel_compensation_map = dict(zip(env.robot_list, (True, False, False)))
    env.get_ft_sensor_data = lambda part: None
    env.robot_dict = {}
    for part, position, velocity in zip(env.robot_list, (0.37, 50, 75), (0.2, 750, -320)):
        env.robot_dict[part] = SimpleNamespace(
            publish_joint_states=lambda *args: None,
            get_joint_position_command=lambda position=position: [position],
            get_joint_velocity_command=lambda velocity=velocity: [velocity],
            get_joint_torque_command=lambda: [0.0],
            get_controller_mode=lambda: 1,
        )
    (env.joint_names_all, env.controller_mode, env.joint_position_command_all,
     env.joint_velocity_command_all, env.joint_torque_command_all) = env.update_joint_states()
    assert env.joint_velocity_command_all == [0.2, 0, 0]
    env._apply_control()
    env.robot.data.joint_pos = env.robot.targets.clone()
    assert env.get_joint_position(env.joint_names_all[0]) == pytest.approx(0.39)
    assert env.get_joint_position(env.joint_names_all[1]) == pytest.approx(50)
    assert env.get_joint_position(env.joint_names_all[2]) == pytest.approx(75)


@pytest.mark.parametrize('value', [-0.2, 0, 30, 100.2])
def test_feedback_preserves_direction_and_excursions(env, value):
    for name, idx in env.joint_name_to_index.items():
        expected = value
        if 'gripper' in name:
            expected *= SIGNS[name.rsplit('_', 1)[-1]] * 0.0093
        for field in ('joint_pos', 'joint_vel', 'joint_acc'):
            getattr(env.robot.data, field)[0, idx] = expected
        assert env.get_joint_position(name) == pytest.approx(value)
        assert env.get_joint_velocity(name) == pytest.approx(value)
        assert env.get_joint_acceleration(name) == pytest.approx(value)
