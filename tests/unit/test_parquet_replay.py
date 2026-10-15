"""Offline data and SDK-dispatch checks; no ROS, renderer or hardware."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

spec = importlib.util.spec_from_file_location(
    "replay_parquet", Path(__file__).resolve().parents[2] / "scripts/replay_parquet.py"
)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def dataset(tmp_path, *, commands=None, times=None, episodes=None, names=None):
    commands = np.tile(np.arange(25, dtype=float), (4, 1)) if commands is None else commands
    data = {replay.COMMAND_COLUMN: commands.tolist()}
    if times is not None:
        data["timestamp"] = times
    if episodes is not None:
        data["episode_index"] = episodes
    path = tmp_path / "episode.parquet"
    pq.write_table(pa.table(data), path)
    if names is not None:
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta/info.json").write_text(json.dumps({"features": {
            replay.COMMAND_COLUMN: {"shape": [25], "names": names}
        }}))
    return path


def test_metadata_reorder_segment_and_timing(tmp_path):
    source = np.tile(np.arange(25, dtype=float), (4, 1))
    path = dataset(tmp_path, commands=source[:, ::-1], times=[10, 10.1, 10.4, 11],
                   names=replay.JOINT_NAMES[::-1])
    args = replay.parser().parse_args([str(path), "--start-frame", "1", "--end-frame", "3",
                                      "--speed", "2"])
    commands, times = replay.load_segment(args)
    np.testing.assert_array_equal(commands, source[1:3])
    np.testing.assert_allclose(times, [0, 0.15])
    assert [len(row) for row in replay.split_command(commands[0])] == [4, 7, 1, 7, 1, 2]
    assert replay.split_command(commands[0])[0] == [3, 4, 5, 6]


@pytest.mark.parametrize("extra", [["--start-frame", "-1"], ["--end-frame", "5"],
                                  ["--start-frame", "2", "--end-frame", "2"]])
def test_invalid_segment(tmp_path, extra):
    path = dataset(tmp_path, times=[0, 1, 2, 3])
    with pytest.raises(ValueError, match="segment"):
        replay.load_segment(replay.parser().parse_args([str(path), *extra]))


@pytest.mark.parametrize("case", ["nan", "gripper", "duplicate_time", "episodes", "names"])
def test_invalid_input(tmp_path, case):
    commands = np.zeros((4, 25))
    if case == "nan":
        commands[1, 7] = np.nan
    if case == "gripper":
        commands[1, 22] = 101
    path = dataset(tmp_path, commands=commands,
                   times=[0, 1, 1, 3] if case == "duplicate_time" else [0, 1, 2, 3],
                   episodes=[0, 0, 1, 1] if case == "episodes" else None,
                   names=["unknown"] * 25 if case == "names" else None)
    with pytest.raises(ValueError):
        replay.load_segment(replay.parser().parse_args([str(path)]))


def test_missing_time_requires_explicit_fps(tmp_path):
    path = dataset(tmp_path)
    with pytest.raises(ValueError, match="missing columns"):
        replay.load_segment(replay.parser().parse_args([str(path)]))
    _, times = replay.load_segment(replay.parser().parse_args([str(path), "--fps", "20"]))
    np.testing.assert_allclose(times, [0, 0.05, 0.1, 0.15])


def test_dispatch_and_viewer_exit(monkeypatch):
    robot = Mock()
    robot.get_current_joints_position.return_value = [[0] * (b - a) for _, a, b in replay.PARTS]
    robot.get_control_rights_status.return_value = True
    viewer = Mock()
    viewer.poll.return_value = None
    middleware = Mock()
    middleware.ok.return_value = True
    commands = np.tile(np.arange(25, dtype=float), (2, 1))
    clock = [0.0]
    monkeypatch.setattr(replay.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(replay, "wait_until", lambda deadline, *_: clock.__setitem__(0, deadline))
    replay.replay(robot, commands, [0, 0.1], 0.01, viewer, middleware)
    call = robot.set_joints_position.call_args
    assert call.args == ([name for name, _, _ in replay.PARTS], replay.split_command(commands[-1]))
    assert call.kwargs == {"control_way": "direct", "use_wbc": False}
    assert robot.set_joints_position.call_args_list[0].args[1] == robot.get_current_joints_position.return_value
    robot.get_control_rights_status.return_value = False
    with pytest.raises(RuntimeError, match="control rights"):
        replay.replay(robot, commands, [0, 0.1], 0.01, viewer, middleware)


def test_control_clock_holds_frames_and_skips_missed_ticks(monkeypatch):
    clock = [0.0]
    sent = []
    robot = Mock()
    robot.get_current_joints_position.return_value = [[0] * (b - a) for _, a, b in replay.PARTS]
    robot.get_control_rights_status.return_value = True

    def send(names, positions, **kwargs):
        sent.append((clock[0], positions))
        # Simulate a delayed SDK call in the dataset phase.
        if len(sent) == 4:
            clock[0] += 0.025

    robot.set_joints_position.side_effect = send
    monkeypatch.setattr(replay.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(replay, "wait_until", lambda deadline, *_: clock.__setitem__(
        0, max(clock[0], deadline)))
    commands = np.tile(np.arange(3, dtype=float)[:, None], (1, 25))
    replay.replay(robot, commands, [0, 0.03, 0.06], 0.01, Mock(), Mock())
    dataset_calls = sent[2:]
    assert [round(t - 0.01, 3) for t, _ in dataset_calls] == [0, 0.01, 0.035, 0.04, 0.05, 0.06]
    assert [pose[0][0] for _, pose in dataset_calls] == [0, 0, 1, 1, 1, 2]


def test_missing_sdk_dependency_reports_interpreter(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError("No module named 'cryptography'")
    monkeypatch.setattr(replay.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="SDK dependency cryptography") as error:
        replay.load_sdk()
    assert replay.sys.executable in str(error.value)
    assert "requirements-replay.txt" in str(error.value)


def test_sdk_dependency_failure_precedes_ros_probe(monkeypatch):
    monkeypatch.setenv("ROS_VERSION", "2")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    monkeypatch.setenv("DISPLAY", ":1")
    sdk = Mock(side_effect=RuntimeError("SDK dependency cryptography"))
    probe = Mock()
    monkeypatch.setattr(replay, "load_sdk", sdk)
    monkeypatch.setattr(replay, "wait_for_simulation", probe)
    with pytest.raises(RuntimeError, match="SDK dependency cryptography"):
        replay.run(replay.parser().parse_args(["unused.parquet"]), None, None)
    probe.assert_not_called()


def test_viewer_quit_stops_replay():
    viewer = Mock(returncode=0)
    viewer.poll.return_value = 0
    with pytest.raises(KeyboardInterrupt):
        replay.wait_until(0, viewer, Mock())


@pytest.mark.parametrize('outcome', ['ready', 'timeout', 'not_simulation'])
def test_probe_context_and_cleanup(monkeypatch, outcome):
    import sys
    from types import ModuleType, SimpleNamespace

    def module(name, **attributes):
        value = ModuleType(name)
        value.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    context = Mock()
    node = Mock()
    subscriptions = {}
    node.create_subscription.side_effect = (
        lambda msg_type, topic, callback, qos: subscriptions.setdefault(topic, callback)
    )
    executor = Mock()
    executor_type = Mock(return_value=executor)
    context_type = Mock(return_value=context)
    ros = module('rclpy', init=Mock(), create_node=Mock(return_value=node), spin_once=Mock())
    module('rclpy.context', Context=context_type)
    module('rclpy.executors', SingleThreadedExecutor=executor_type)
    module('rclpy.qos', qos_profile_sensor_data=object())
    module('astribot_msgs')
    module('astribot_msgs.msg', RobotJointState=object())
    module('sensor_msgs')
    module('sensor_msgs.msg', Image=object())

    def deliver(**kwargs):
        for topic, callback in subscriptions.items():
            if topic.endswith('/image_raw'):
                callback(SimpleNamespace(width=640, height=480))
            else:
                frame_id = 'simulation' if outcome == 'ready' else 'hardware'
                if outcome == 'ready' and topic == '/astribot_chassis/joint_space_states':
                    frame_id = ''
                callback(SimpleNamespace(header=SimpleNamespace(
                    frame_id=frame_id)))

    executor.spin_once.side_effect = deliver
    if outcome == 'timeout':
        ticks = iter([0, 2])
        monkeypatch.setattr(replay.time, 'monotonic', lambda: next(ticks))
        with pytest.raises(RuntimeError, match='timeout'):
            replay.wait_for_simulation(1)
    elif outcome == 'not_simulation':
        with pytest.raises(RuntimeError, match='not from simulation'):
            replay.wait_for_simulation(1)
    else:
        replay.wait_for_simulation(1)
        assert len(subscriptions) == 10
    ros.init.assert_called_once_with(context=context)
    ros.create_node.assert_called_once_with(
        f'parquet_replay_probe_{replay.os.getpid()}', context=context)
    executor_type.assert_called_once_with(context=context)
    executor.add_node.assert_called_once_with(node)
    ros.spin_once.assert_not_called()
    executor.shutdown.assert_called_once_with(timeout_sec=1.0)
    node.destroy_node.assert_called_once_with()
    context.try_shutdown.assert_called_once_with()
