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
    "replay_extract_event", Path(__file__).resolve().parents[2] / "scripts/replay_extract_event.py"
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


def test_replay_defaults_to_direct_control_mode(tmp_path):
    path = dataset(tmp_path, times=[0, 1 / 30, 2 / 30, 3 / 30])
    args = replay.parser().parse_args([str(path)])
    assert args.control_mode == "direct"


@pytest.mark.parametrize('extra,published', [([], True), (['--dry-run'], False),
                                          (['--no-web-publish'], False)])
def test_extract_only_publishes_only_after_success(tmp_path, monkeypatch, extra, published):
    from types import ModuleType
    from src.placement_analysis import web_publish
    extractor = ModuleType('src.placement_analysis.extraction')
    extractor.extract_events = Mock()
    monkeypatch.setitem(replay.sys.modules, extractor.__name__, extractor)
    publish = Mock()
    monkeypatch.setattr(web_publish, 'publish_bundle', publish)
    output = tmp_path / 'bundle'
    argv = [str(tmp_path / 'capture'), '--extract-only', '--output', str(output),
            '--task-name', '按压按钮', *extra]
    assert replay.main(argv) == 0
    if published:
        publish.assert_called_once_with(output, task_name='按压按钮')
    else:
        publish.assert_not_called()
    publish.reset_mock()
    extractor.extract_events.side_effect = ValueError('incomplete capture')
    assert replay.main(argv) == 1
    publish.assert_not_called()


def test_cpu_phase_forwards_publication_options(monkeypatch):
    args = replay.parser().parse_args(['capture', '--extract-only', '--task-name', '按钮', '--no-web-publish'])
    run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(replay.subprocess, 'run', run)
    replay.run_cpu_phase(args)
    command = run.call_args.args[0]
    assert command[command.index('--task-name') + 1] == '按钮'
    assert '--no-web-publish' in command


def test_tqdm_progress_updates_and_closes(monkeypatch):
    import io
    import types

    records = []

    class FakeTqdm:
        def __init__(self, total, desc, unit, dynamic_ncols, file, **kwargs):
            self.total, self.n = total, 0
            records.append((total, desc, unit, dynamic_ncols, file))
            assert kwargs == {'position': 0, 'leave': False, 'mininterval': 0.5}

        def update(self, amount):
            self.n += amount

        def close(self):
            records.append("closed")

    tqdm_package = types.ModuleType("tqdm")
    tqdm_auto = types.ModuleType("tqdm.auto")
    tqdm_auto.tqdm = FakeTqdm
    monkeypatch.setitem(__import__("sys").modules, "tqdm", tqdm_package)
    monkeypatch.setitem(__import__("sys").modules, "tqdm.auto", tqdm_auto)
    progress = replay.TqdmProgress(stream=io.StringIO(), enabled=True)
    progress.start_episode(1, 3, "episode_000000", 4)
    progress.update_frame(2)
    progress.finish_episode()
    assert records[0][0:3] == (4, "Replay 1/3 episode_000000", "frame")
    assert records[0][3] is True
    assert records[1] == "closed"
    progress.start_episode(2, 3, "episode_000001", 4)
    progress.update_frame(1)
    bar = progress.bar
    progress.close()
    assert bar.n == 1  # failure must not be displayed as 100% complete


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
    replay.replay(robot, commands, [0, 0.1], 0.01, viewer, middleware, control_mode="direct")
    call = robot.set_joints_position.call_args
    assert call.args == ([name for name, _, _ in replay.PARTS], replay.split_command(commands[-1]))
    assert call.kwargs == {"control_way": "direct", "use_wbc": False}
    assert robot.set_joints_position.call_args_list[0].args[1] == robot.get_current_joints_position.return_value
    robot.get_control_rights_status.return_value = False
    with pytest.raises(RuntimeError, match="control rights"):
        replay.replay(robot, commands, [0, 0.1], 0.01, viewer, middleware, control_mode="direct")


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
    replay.replay(robot, commands, [0, 0.03, 0.06], 0.01, Mock(), Mock(), control_mode="direct")
    dataset_calls = sent[2:]
    assert np.min(np.diff([t for t, _ in sent])) >= .01 - 1e-12
    assert [pose[0][0] for _, pose in dataset_calls] == [0, 0, 1, 1, 2]


def test_move_preserves_frames_and_intervals_despite_sdk_overhead(monkeypatch):
    clock = [0.0]
    robot = Mock()
    robot.get_control_rights_status.return_value = True

    def move(*args, duration, **kwargs):
        clock[0] += duration + 0.02
        return "move success"

    robot.move_joints_position.side_effect = move
    viewer, middleware = Mock(), Mock()
    viewer.poll.return_value = None
    middleware.ok.return_value = True
    monkeypatch.setattr(replay.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(replay.time, "sleep", lambda dt: clock.__setitem__(0, clock[0] + dt))
    commands = np.tile(np.arange(3, dtype=float)[:, None], (1, 25))
    replay.replay(robot, commands, [0, 0.03, 0.08], 3.0, viewer, middleware)
    calls = robot.move_joints_position.call_args_list
    assert [call.args[1] for call in calls] == [replay.split_command(q) for q in commands]
    assert [call.kwargs["duration"] for call in calls] == pytest.approx([3, 0.03, 0.05])
    assert all(call.kwargs["use_wbc"] is False for call in calls)
    robot.set_joints_position.assert_not_called()


def test_move_failure_stops_before_next_frame():
    robot = Mock()
    robot.get_control_rights_status.return_value = True
    robot.move_joints_position.return_value = "move failed"
    viewer = Mock()
    viewer.poll.return_value = None
    with pytest.raises(RuntimeError, match="SDK move failed"):
        replay.replay(robot, np.zeros((2, 25)), [0, 0.03], 3, viewer, Mock())
    assert robot.move_joints_position.call_count == 1


def test_move_capture_window_closes_after_feedback_drain(monkeypatch):
    robot = Mock()
    robot.get_control_rights_status.return_value = True
    robot.move_joints_position.return_value = "move success"
    viewer = Mock()
    viewer.poll.return_value = None
    middleware = Mock()
    middleware.ok.return_value = True
    sequence = []
    capture = Mock()
    capture.end_window.side_effect = lambda: sequence.append("end_window")
    clock = [0.0]
    monkeypatch.setattr(replay.time, "monotonic", lambda: clock[0])

    def wait(deadline, *_):
        clock[0] = max(clock[0], deadline)
        sequence.append("drain")

    monkeypatch.setattr(replay, "wait_until", wait)
    replay.replay(robot, np.zeros((2, 25)), [0, 1 / 30], 0.01,
                  viewer, middleware, control_mode="move", capture=capture)
    assert capture.end_window.call_count == 1
    assert sequence[-2:] == ["drain", "end_window"]


@pytest.mark.parametrize('ready_after', [.25, 10.0])
def test_feedback_drain_waits_for_all_parts_with_a_timeout(monkeypatch, ready_after):
    clock = [0.0]
    capture = Mock()
    capture.feedback_covers_window.side_effect = lambda: clock[0] >= ready_after
    monkeypatch.setattr(replay.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(replay, 'wait_until', lambda deadline, *_: clock.__setitem__(0, deadline))
    replay.drain_feedback(capture, None, Mock())
    assert clock[0] == pytest.approx(min(ready_after, replay.FEEDBACK_DRAIN_S + replay.FEEDBACK_TIMEOUT_S))
    capture.end_window.assert_called_once()


def test_30fps_direct_replay_uses_one_100hz_clock(monkeypatch):
    clock, sent, frames = [0.0], [], []
    robot = Mock()
    robot.get_control_rights_status.return_value = True
    robot.get_current_joints_position.return_value = [[0] * (b-a) for _, a, b in replay.PARTS]
    robot.set_joints_position.side_effect = lambda *a, **kw: sent.append(clock[0])
    monkeypatch.setattr(replay.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(replay, 'wait_until', lambda deadline, *_: clock.__setitem__(0, max(clock[0], deadline)))
    commands = np.zeros((435, 25))
    replay.replay(robot, commands, np.arange(435) / 30, .01, None, Mock(),
                  control_mode='direct', progress=frames.append)
    np.testing.assert_allclose(np.diff(sent), .01, atol=1e-10)
    assert frames == list(range(1, 436))


def test_missing_sdk_dependency_reports_interpreter(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError("No module named 'cryptography'")
    monkeypatch.setattr(replay.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="SDK dependency cryptography") as error:
        replay.load_sdk()
    assert replay.sys.executable in str(error.value)
    assert "cryptography>=3.4" in str(error.value)


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


@pytest.mark.parametrize('outcome', ['ready', 'timeout', 'not_simulation', 'no_cameras'])
@pytest.mark.parametrize('cameras', [True, False])
def test_probe_context_and_cleanup(monkeypatch, outcome, cameras):
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
                if outcome != 'no_cameras':
                    callback(SimpleNamespace(width=640, height=480))
            else:
                frame_id = 'hardware' if outcome == 'not_simulation' else 'simulation'
                if frame_id == 'simulation' and topic == '/astribot_chassis/joint_space_states':
                    frame_id = ''
                callback(SimpleNamespace(header=SimpleNamespace(
                    frame_id=frame_id)))

    executor.spin_once.side_effect = deliver
    if outcome == 'timeout':
        ticks = iter([0, 2])
        monkeypatch.setattr(replay.time, 'monotonic', lambda: next(ticks))
        with pytest.raises(RuntimeError, match='timeout'):
            replay.wait_for_simulation(1, cameras=cameras)
    elif outcome == 'not_simulation':
        with pytest.raises(RuntimeError, match='not from simulation'):
            replay.wait_for_simulation(1, cameras=cameras)
    elif outcome == 'no_cameras' and cameras:
        ticks = iter([0, 0, 2])
        monkeypatch.setattr(replay.time, 'monotonic', lambda: next(ticks))
        with pytest.raises(RuntimeError, match='--no-viewer') as error:
            replay.wait_for_simulation(1, cameras=cameras)
        assert f'missing: {sorted(replay.CAMERAS)}' in str(error.value)
    else:
        replay.wait_for_simulation(1, cameras=cameras)
        assert len(subscriptions) == (10 if cameras else 7)
    ros.init.assert_called_once_with(context=context)
    ros.create_node.assert_called_once_with(
        f'parquet_replay_probe_{replay.os.getpid()}', context=context)
    executor_type.assert_called_once_with(context=context)
    executor.add_node.assert_called_once_with(node)
    ros.spin_once.assert_not_called()
    executor.shutdown.assert_called_once_with(timeout_sec=1.0)
    node.destroy_node.assert_called_once_with()
    context.try_shutdown.assert_called_once_with()


def test_directory_dry_run_validates_every_episode_without_runtime(tmp_path, monkeypatch):
    import builtins
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        assert not name.startswith(('rclpy', 'astribot_sdk', 'astribot_ros_middleware', 'pxr', 'isaac'))
        return original(name, *args, **kwargs)

    root = tmp_path / 'data'
    chunk = root / 'chunk-000'
    chunk.mkdir(parents=True)
    path = dataset(chunk, times=[0, 1/30, 2/30, 3/30])
    path.rename(chunk / 'episode_000001.parquet')
    path = dataset(root, times=[0, 1/30, 2/30, 3/30])
    path.rename(root / 'episode_000000.parquet')
    report = tmp_path / 'report.json'
    monkeypatch.setattr(builtins, '__import__', guard)
    monkeypatch.setattr(replay, 'run', Mock(side_effect=AssertionError('must not start replay')))
    assert replay.main([str(root), '--dry-run', '--report', str(report)]) == 0
    data = json.loads(report.read_text())
    assert data['episode_count'] == 2
    assert data['frames'] == 8
    assert data['duration_s'] == pytest.approx(.2)
    assert len({e['episode_id'] for e in data['episodes']}) == 2


def test_bad_episode_prevents_entire_batch_from_starting(tmp_path, monkeypatch):
    good = dataset(tmp_path, times=[0, .1, .2, .3])
    good.rename(tmp_path / 'episode_0.parquet')
    dataset(tmp_path, times=[0, .1, .1, .3])
    run = Mock()
    monkeypatch.setattr(replay, 'run', run)
    assert replay.main([str(tmp_path), '--no-viewer']) == 1
    run.assert_not_called()


@pytest.mark.parametrize('case', ['empty', 'duplicate', 'segment'])
def test_directory_input_rejects_ambiguous_or_partial_scope(tmp_path, case):
    if case != 'empty':
        dataset(tmp_path, times=[0, .1, .2, .3])
    if case == 'duplicate':
        child = tmp_path / 'chunk'
        child.mkdir()
        dataset(child, times=[0, .1, .2, .3])
    argv = [str(tmp_path), '--dry-run']
    if case == 'segment':
        argv += ['--start-frame', '1']
    assert replay.main(argv) == 1


def test_moving_chassis_is_rejected_before_fixed_base_replay(tmp_path):
    commands = np.zeros((4, 25))
    commands[1, 0] = .01
    path = dataset(tmp_path, commands=commands, times=[0, .1, .2, .3])
    with pytest.raises(ValueError, match='moving chassis'):
        replay.load_segment(replay.parser().parse_args([str(path)]))


def test_direct_replay_sends_all_source_frames_despite_long_delay(monkeypatch):
    clock = [0.0]
    robot = Mock()
    robot.get_control_rights_status.return_value = True
    robot.get_current_joints_position.return_value = [[0] * (b-a) for _, a, b in replay.PARTS]
    sent = []
    def send(names, positions, **kwargs):
        sent.append(positions[0][0])
        if len(sent) == 3:  # first dataset call stalls across several source frames
            clock[0] += .2
    robot.set_joints_position.side_effect = send
    monkeypatch.setattr(replay.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(replay, 'wait_until', lambda deadline, *_: clock.__setitem__(0, max(clock[0], deadline)))
    commands = np.tile(np.arange(4)[:, None], (1, 25)).astype(float)
    replay.replay(robot, commands, [0, .03, .06, .09], .01, None, Mock(), control_mode='direct')
    assert sent[2:] == [0, 1, 2, 3]


def test_move_capture_window_excludes_initial_move(monkeypatch):
    clock = [0.0]
    robot = Mock()
    robot.get_control_rights_status.return_value = True
    def move(*args, duration, **kwargs):
        clock[0] += duration
        return 'success'
    robot.move_joints_position.side_effect = move
    capture = Mock()
    windows = []
    capture.begin.side_effect = lambda: windows.append(clock[0])
    capture.end_window.side_effect = lambda: windows.append(clock[0])
    monkeypatch.setattr(replay.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(replay, 'wait_until', lambda deadline, *_: clock.__setitem__(0, max(clock[0], deadline)))
    replay.replay(robot, np.zeros((3, 25)), [0, .03, .06], 3, None, Mock(), capture=capture)
    assert windows == pytest.approx([3, 3.06 + replay.FEEDBACK_DRAIN_S])
    assert capture.record_call.call_count == 3
    assert capture.record_call.call_args_list[0].kwargs['phase'] == 'initial'
    assert [call.args[0] for call in capture.record_call.call_args_list] == [0, 1, 2]


def test_headless_batch_reuses_sdk_and_saves_each_episode(tmp_path, monkeypatch):
    from src.placement_analysis import replay_capture
    root = tmp_path / 'data'
    root.mkdir()
    first = dataset(root, times=[0, .03, .06, .09])
    first.rename(root / 'episode_0.parquet')
    dataset(root, times=[0, .03, .06, .09]).rename(root / 'episode_1.parquet')
    args = replay.parser().parse_args([str(root), '--no-viewer', '--output-root', str(tmp_path / 'capture')])
    report = replay.validate_dataset(args, replay.discover_episodes(args))
    robot, middleware = Mock(), Mock()
    sdk = Mock(return_value=robot)
    monkeypatch.setenv('ROS_VERSION', '2')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('ROS_DOMAIN_ID', '197')
    monkeypatch.delenv('DISPLAY', raising=False)
    monkeypatch.delenv('WAYLAND_DISPLAY', raising=False)
    monkeypatch.setattr(replay, 'load_sdk', lambda: (sdk, middleware))
    probe = Mock()
    monkeypatch.setattr(replay, 'wait_for_simulation', probe)
    dispatch = Mock()
    monkeypatch.setattr(replay, 'replay', dispatch)
    captures = [Mock(), Mock()]
    factory = Mock(side_effect=captures)
    monkeypatch.setattr(replay_capture, 'ReplayCapture', factory)
    replay.run(args, report=report)
    sdk.assert_called_once_with(freq=replay.CONTROL_HZ)
    probe.assert_called_once_with(args.timeout, cameras=False, profile=replay.robot_profile(args))
    assert dispatch.call_count == 2
    for index, capture in enumerate(captures):
        capture.save.assert_called_once()
        assert capture.save.call_args.args[0].name == f'episode_{index}'
        capture.close.assert_called_once()
    batch = json.loads((args.output_root / 'batch_manifest.json').read_text())
    assert batch['status'] == 'complete'
    assert batch['completed'] == ['episode_0', 'episode_1']
    robot.stop_robot.assert_called_once()
    middleware.shutdown.assert_called_once()


def test_resume_requires_matching_source_settings_and_intact_capture(tmp_path):
    import hashlib
    path = dataset(tmp_path, times=[0, .1, .2, .3])
    args = replay.parser().parse_args([str(path)])
    source = replay.validate_dataset(args, [path])['episodes'][0]
    settings = replay.replay_settings(args)
    output = tmp_path / 'capture'
    output.mkdir()
    for name in ('api_commands.parquet', 'sim_states.parquet'):
        (output / name).write_bytes(b'capture fixture')
    metadata = {'status': 'complete', 'capture_schema': 'replay_parquet_v1', 'source': source,
                'input_sha256': source['input_sha256'], 'replay_settings': settings,
                'capture_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}}
    (output / 'run_metadata.json').write_text(json.dumps(metadata))
    assert replay.completed_capture(output, source, settings, True)
    with pytest.raises(FileExistsError):
        replay.completed_capture(output, source, settings, False)
    with pytest.raises(FileExistsError):
        replay.completed_capture(output, source, {**settings, 'speed': 2}, True)
    (output / 'sim_states.parquet').write_bytes(b'truncated')
    with pytest.raises(FileExistsError):
        replay.completed_capture(output, source, settings, True)


def test_resume_retries_matching_failed_capture_in_place(tmp_path):
    path = dataset(tmp_path, times=[0, .1, .2, .3])
    args = replay.parser().parse_args([str(path)])
    source = replay.validate_dataset(args, [path])['episodes'][0]
    settings = replay.replay_settings(args)
    output = tmp_path / 'capture'
    output.mkdir()
    (output / 'run_metadata.json').write_text(json.dumps({
        'status': 'failed', 'input_sha256': source['input_sha256'],
        'source': {'metadata_sha256': source['metadata_sha256']},
        'replay_settings': settings,
    }))
    assert replay.completed_capture(output, source, settings, True) is False


def test_completed_resume_does_not_connect_sdk(tmp_path, monkeypatch):
    path = dataset(tmp_path, times=[0, .1, .2, .3])
    monkeypatch.setattr(replay, 'completed_capture', lambda *args: True)
    run = Mock()
    monkeypatch.setattr(replay, 'run', run)
    output = tmp_path / 'capture'
    cpu = Mock()
    monkeypatch.setattr(replay, 'run_cpu_phase', cpu)
    assert replay.main([str(path), '--resume', '--output-root', str(output),
                        '--output', str(tmp_path / 'bundle')]) == 0
    assert [call.kwargs for call in cpu.call_args_list] == [{'check': True}, {}]
    run.assert_not_called()
    assert json.loads((output / 'batch_manifest.json').read_text())['status'] == 'complete'


def test_pipeline_preflights_then_replays_then_extracts_after_cleanup(tmp_path, monkeypatch):
    from types import SimpleNamespace
    path = dataset(tmp_path, times=[0, .1, .2, .3])
    capture_root, output = tmp_path / 'capture', tmp_path / 'Web bundle'
    calls = []

    def cpu(command, *, timeout):
        assert command[0] == '/custom CPU/python'
        if '--check-extraction' in command:
            assert timeout == 60
            calls.append('preflight')
        else:
            assert calls == ['preflight', 'SDK cleanup']
            assert command[command.index('--extract-only') + 1] == str(capture_root)
            assert command[command.index('--output') + 1] == str(output)
            assert '--extract-python' not in command  # worker must not recursively launch itself
            assert json.loads((capture_root / 'batch_manifest.json').read_text())['status'] == 'complete'
            calls.append('extract')
        return SimpleNamespace(returncode=0)

    def sdk(args, *, report):
        replay.write_json(args.output_root / 'batch_manifest.json', {'status': 'complete'})
        calls.append('SDK cleanup')

    monkeypatch.setattr(replay.subprocess, 'run', cpu)
    monkeypatch.setattr(replay, 'run', sdk)
    assert replay.main([str(path), '--no-viewer', '--output-root', str(capture_root),
                        '--output', str(output), '--extract-python', '/custom CPU/python']) == 0
    assert calls == ['preflight', 'SDK cleanup', 'extract']


def test_cpu_preflight_failure_prevents_motion(tmp_path, monkeypatch):
    from types import SimpleNamespace
    path = dataset(tmp_path, times=[0, .1, .2, .3])
    sdk = Mock()
    monkeypatch.setattr(replay, 'run', sdk)
    monkeypatch.setattr(replay.subprocess, 'run', Mock(return_value=SimpleNamespace(returncode=1)))
    assert replay.main([str(path), '--output-root', str(tmp_path / 'capture'),
                        '--output', str(tmp_path / 'bundle')]) == 1
    sdk.assert_not_called()
    assert not (tmp_path / 'capture').exists()


@pytest.mark.parametrize('interrupted', [False, True])
def test_failed_replay_preserves_captures_without_extracting(tmp_path, monkeypatch, interrupted):
    path = dataset(tmp_path, times=[0, .1, .2, .3])
    capture_root = tmp_path / 'capture'
    cpu = Mock()
    monkeypatch.setattr(replay, 'run_cpu_phase', cpu)

    def fail(args, *, report):
        replay.write_json(args.output_root / 'batch_manifest.json', {'status': 'partial'})
        raise KeyboardInterrupt() if interrupted else RuntimeError('replay failed')

    monkeypatch.setattr(replay, 'run', fail)
    assert replay.main([str(path), '--output-root', str(capture_root),
                        '--output', str(tmp_path / 'bundle')]) == (130 if interrupted else 1)
    assert json.loads((capture_root / 'batch_manifest.json').read_text())['status'] == 'partial'
    assert cpu.call_count == 1
    assert cpu.call_args.kwargs == {'check': True}


def test_extraction_failure_keeps_completed_capture_for_retry(tmp_path, monkeypatch):
    path = dataset(tmp_path, times=[0, .1, .2, .3])
    capture_root = tmp_path / 'capture'
    cpu = Mock(side_effect=[None, RuntimeError('CPU extraction failed')])
    monkeypatch.setattr(replay, 'run_cpu_phase', cpu)
    monkeypatch.setattr(replay, 'run', lambda args, **kwargs: replay.write_json(
        args.output_root / 'batch_manifest.json', {'status': 'complete'}))
    assert replay.main([str(path), '--output-root', str(capture_root),
                        '--output', str(tmp_path / 'bundle')]) == 1
    assert json.loads((capture_root / 'batch_manifest.json').read_text())['status'] == 'complete'


@pytest.mark.parametrize('conflict', ['existing', 'capture', 'source', 'report'])
def test_output_conflicts_fail_before_motion(tmp_path, monkeypatch, conflict):
    root = tmp_path / 'data'
    root.mkdir()
    dataset(root, times=[0, .1, .2, .3])
    capture_root, output = tmp_path / 'capture', tmp_path / 'bundle'
    if conflict == 'existing':
        output.mkdir()
        (output / 'placement.json').write_text('preserve existing dataset')
    elif conflict == 'capture':
        output = capture_root / 'bundle'
    elif conflict == 'source':
        output = root / 'bundle'
    extra = ['--report', str(output / 'report.json')] if conflict == 'report' else []
    sdk, cpu = Mock(), Mock()
    monkeypatch.setattr(replay, 'run', sdk)
    monkeypatch.setattr(replay, 'run_cpu_phase', cpu)
    assert replay.main([str(root), '--output-root', str(capture_root), '--output', str(output), *extra]) == 1
    sdk.assert_not_called()
    cpu.assert_not_called()
    if conflict == 'existing':
        assert (output / 'placement.json').read_text() == 'preserve existing dataset'


def test_task_and_identification_options_forward_to_cpu_without_output(monkeypatch):
    args = replay.parser().parse_args(['bundle', '--identify-only', '--task-config', 'task.json',
                                      '--data-root', '/new-root', '--dry-run'])
    args.output = Path('/unused/default')
    run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(replay.subprocess, 'run', run)
    replay.run_cpu_phase(args)
    command = run.call_args.args[0]
    assert '--identify-only' in command and '--dry-run' in command
    assert command[command.index('--task-config') + 1] == 'task.json'
    assert command[command.index('--data-root') + 1] == '/new-root'
    assert '--detector-config' not in command and '--output' not in command


@pytest.mark.parametrize('no_identify', [False, True])
def test_task_extract_automatically_passes_identification_settings(tmp_path, monkeypatch, no_identify):
    from types import ModuleType
    from src.placement_analysis import identification, pipeline_stages
    extractor = ModuleType('src.placement_analysis.extraction')
    output = tmp_path / 'bundle'
    output.mkdir()
    (output / 'placement.json').write_text('{}')
    extractor.extract_events = Mock()
    identify, evaluate = Mock(), Mock()
    monkeypatch.setattr(identification, 'identify_bundle', identify)
    monkeypatch.setattr(pipeline_stages, 'evaluate_bundle', evaluate)
    monkeypatch.setitem(replay.sys.modules, extractor.__name__, extractor)
    options = ['--no-identify'] if no_identify else []
    assert replay.main(['capture', '--extract-only', '--task-config', str(replay.DEFAULT_DETECTOR_CONFIG),
                        '--data-root', str(tmp_path), '--no-web-publish', '--output', str(output), *options]) == 0
    kwargs = extractor.extract_events.call_args.kwargs
    assert kwargs['task_config'] == replay.DEFAULT_DETECTOR_CONFIG
    assert kwargs['data_root'] == tmp_path and kwargs['detector_config'] is None
    assert identify.call_count == (0 if no_identify else 1)
    evaluate.assert_called_once_with(output, task_config=replay.DEFAULT_DETECTOR_CONFIG, observations=None)


def test_no_progress_reaches_identify_and_cpu_child(tmp_path, monkeypatch):
    from src.placement_analysis import identification, pipeline_stages
    identify, evaluate = Mock(), Mock()
    monkeypatch.setattr(identification, 'identify_bundle', identify)
    monkeypatch.setattr(pipeline_stages, 'evaluate_bundle', evaluate)
    assert replay.main([str(tmp_path), '--identify-only', '--no-progress', '--no-web-publish']) == 0
    assert identify.call_args.kwargs['no_progress'] is True
    evaluate.assert_called_once_with(tmp_path, task_config=None, dry_run=False, observations=None)
    args = replay.parser().parse_args([str(tmp_path), '--identify-only', '--no-progress'])
    run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(replay.subprocess, 'run', run)
    replay.run_cpu_phase(args)
    assert '--no-progress' in run.call_args.args[0]
