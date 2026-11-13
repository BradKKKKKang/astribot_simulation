"""Causal HDF5 replay and simulation-time conditioning contracts."""
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.simulation_config.joint_filter import JointTargetFilter
from src.simu_utils.joint_alignment_stepper import JointAlignmentStepper

SPEC = importlib.util.spec_from_file_location(
    'alignment_replay', Path(__file__).parents[2] / 'scripts/joint_align/replay_artifacts_sdk_100hz.py')
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


def test_lowpass_is_invariant_to_subdividing_simulation_time_and_reset():
    filters = [JointTargetFilter({'j': {'time_constant_s': .04}}, {'j'}) for _ in range(2)]
    for f in filters:
        f.apply('j', 0, 0, 99, .004)
    expected = filters[0].apply('j', 1, 0, 99, .008)
    filters[1].apply('j', 1, 0, 99, .004)
    assert filters[1].apply('j', 1, 0, 99, .004) == pytest.approx(expected)
    filters[0].reset()
    assert filters[0].apply('j', 7, 0, 99, .004) == 7


def test_explicit_zero_lookahead_ignores_reconstructed_sdk_velocity():
    f = JointTargetFilter({'j': {}}, {'j', 'legacy'})
    assert f.apply('j', 2, 1000, 102, .004) == 2
    assert f.apply('legacy', 2, 1000, 102, .004) == 102


@pytest.mark.parametrize('settings', [{'bad': {}}, {'j': {'time_constant_s': -1}},
                                      {'j': {'velocity_lookahead_s': float('nan')}},
                                      {'j': {'unknown': 1}}])
def test_invalid_filter_config(settings):
    with pytest.raises(ValueError):
        JointTargetFilter(settings, {'j'})


def write_hdf5(path):
    with h5py.File(path, 'w') as f:
        for part, count in REPLAY.PART_CONFIGS:
            g = f.create_group('joints_dict/' + part)
            c, s = g.create_group('command'), g.create_group('states')
            # Deliberately create out of order and use suffix 10 to exercise numeric sorting.
            for suffix, times, values in [(10, [.020, .024], [8, 9]), (2, [0., .004], [1, 2])]:
                c[f'timestamp_{suffix}'] = times
                c[f'command_{suffix}'] = np.repeat(np.asarray(values)[:, None], count, axis=1)
            s['timestamp_0'] = np.arange(7) * .004
            s['position_0'] = np.zeros((7, count))
            s['velocity_0'] = np.ones((7, count))


def test_hdf5_250hz_preserves_gap_and_missing_torque(tmp_path):
    p = tmp_path / 'episode.hdf5'
    write_hdf5(p)
    _, grid, command, states = REPLAY.load_hdf5_episode(p)
    np.testing.assert_allclose(np.diff(grid), .004)
    np.testing.assert_equal(command[:, 0], [1, 2, 2, 2, 2, 8, 9])
    assert command.shape == (7, 22)
    assert all(np.isnan(v[:, 3]).all() for v in states.values())


def test_hdf5_rejects_nonmonotonic_chunks(tmp_path):
    p = tmp_path / 'episode.hdf5'
    write_hdf5(p)
    with h5py.File(p, 'a') as f:
        f['joints_dict/astribot_torso/command/timestamp_10'][0] = .003
    with pytest.raises(ValueError, match='invalid timestamps'):
        REPLAY.load_hdf5_episode(p)


def test_25dim_sdk_mapping_and_fixed_chassis():
    frame = np.arange(22, dtype=float)
    full = REPLAY._sdk_frame(frame)
    assert full.shape == (25,)
    assert full[14] == frame[11] and full[22] == frame[19]
    assert REPLAY._part_command(full) == REPLAY._part_command(frame)
    full[0] = 1
    with pytest.raises(ValueError, match='zero chassis'):
        REPLAY._part_command(full)


def test_preposition_clips_encoder_overshoot_without_modifying_reference():
    measured = np.zeros(22)
    measured[11], measured[19] = -.03, 100.02
    command = REPLAY._preposition_target(measured)
    assert command[11] == 0 and command[19] == 100
    assert measured[11] == -.03 and measured[19] == 100.02


def test_stepper_rejects_missing_parts_and_never_assigns_commands():
    from types import SimpleNamespace
    interface = SimpleNamespace(dof=2, controller_mode=1, joint_position_command=[1., 2.])
    env = SimpleNamespace(robot_list=['arm'], robot_dict={'arm': interface})
    assert JointAlignmentStepper.targets_match(env, {'arm': [1., 2.]})
    assert not JointAlignmentStepper.targets_match(env, {'arm': [3., 4.]})
    assert interface.joint_position_command == [1., 2.]
    with pytest.raises(ValueError, match='every configured'):
        JointAlignmentStepper.targets_match(env, {})


def test_stepper_advances_once_and_rejects_repeated_sequence():
    import json
    import queue
    import threading
    from types import SimpleNamespace
    stepper = object.__new__(JointAlignmentStepper)
    stepper.queue = queue.Queue()
    stepper.dt, stepper.stepped, stepper.steps = .004, False, 0
    interface = SimpleNamespace(dof=1, controller_mode=1, joint_position_command=[1.])
    executed = []
    env = SimpleNamespace(robot_list=['arm'], robot_dict={'arm': interface},
                          robot_joint_map={'arm': ['joint']},
                          get_joint_positions=lambda _: [len(executed)],
                          get_joint_velocities=lambda _: [0.], step=lambda _: executed.append(1))
    def submit(request):
        item = {'request': json.dumps(request), 'done': threading.Event(), 'cancelled': False}
        stepper.queue.put(item)
        stepper.process_one(env)
        assert item['done'].is_set()
        return item['response']
    assert submit({'op': 'begin'})['physics_time_s'] == 0
    request = {'op': 'step', 'step': 0, 'expected': {'arm': [1.]}}
    assert submit(request)['physics_time_s'] == .004
    assert 'error' in submit(request)
    assert len(executed) == 1
    submit({'op': 'end'})
    assert not stepper.stepped


def test_physics_metrics_ignore_slow_wall_clock():
    states = {j: np.array([[10., 0., 1., np.nan], [10.008, .008, 1., np.nan]])
              for j in REPLAY.JOINTS}
    rows = [{'joint': j, 'physics_time_s': .004, 'position': .004, 'velocity': 1.,
             'header_stamp_ns': 99_000_000_000, 'received_monotonic_ns': 999_000_000_000}
            for j in REPLAY.JOINTS]
    metrics = REPLAY._metric_for_rows(states, rows, 10., 10.008, 0, 0, 'physics_time')
    assert all(v['position_rmse'] < 1e-12 and v['velocity_rmse'] == 0
               for v in metrics['joints'].values())


def test_fixed_settling_uses_sdk_and_verified_steps_without_state_assignment():
    from types import SimpleNamespace
    calls, requests = [], []
    target = np.arange(len(REPLAY.JOINTS), dtype=float)
    def request(op, **fields):
        requests.append((op, fields))
        return {'steps': len(requests) - 1, 'physics_time_s': (len(requests) - 1) * .004}
    step = SimpleNamespace(call=request, rows=lambda response: [dict(response)])
    robot = SimpleNamespace(set_joints_position=lambda *a, **kw: calls.append((a, kw)))
    rows = REPLAY.settle_physics(robot, step, target, 3)
    assert len(calls) == 3
    assert [fields['step'] for op, fields in requests if op == 'step'] == [0, 1, 2]
    assert all(fields['expected'] == dict(zip(REPLAY.PART_NAMES, REPLAY._part_command(target)))
               for op, fields in requests if op == 'step')
    assert rows[0]['physics_time_s'] == 0
    assert rows[0]['steps'] == 3


def test_robot_profile_accepts_arm_gain_and_validates_named_filter_override():
    from src.simulation_config.robot_profile import RobotProfile, default_robot_profile, build_simulation_config
    data = default_robot_profile().as_dict()
    data['simulation_overrides'] = {
        'arm_right_joint_7_control': {'stiffness': 1200.},
        'head_joint_1_control': {'damping': 80.},
        'torso_joint_4_control': {'stiffness': 1600.},
        'joint_command_filters': {'astribot_arm_right_joint_7': {'time_constant_s': .02}}}
    config = build_simulation_config(RobotProfile(data))
    assert config['arm_right_joint_7_control']['stiffness'] == 1200.
    assert config['arm_right_joint_7_control']['damping'] == 40.
    assert config['head_joint_1_control']['damping'] == 80.
    assert config['head_joint_1_control']['stiffness'] == 800.
    assert config['torso_joint_4_control']['stiffness'] == 1600.
    assert config['torso_joint_4_control']['damping'] == 40.
    assert config['joint_command_filters']['astribot_arm_right_joint_7']['time_constant_s'] == .02
    data['simulation_overrides']['joint_command_filters'] = {'unknown_joint': {}}
    with pytest.raises(ValueError, match='unknown joint'):
        build_simulation_config(RobotProfile(data))


def test_selection_rejects_velocity_regression_and_different_training_data(tmp_path, monkeypatch):
    import json
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / 'scripts/joint_align'))
    from fit_response import JOINTS, control_key, select_completed_trials
    trials, configs = {}, {}
    for label, q, v, gain in [('baseline', 1., 1., 800), ('worse_velocity', .1, 2., 1600),
                              ('both_better', .7, .8, 1200)]:
        root = tmp_path / label
        folder = root / 'replay/episode_0'
        folder.mkdir(parents=True)
        (folder / 'run_metadata.json').write_text(json.dumps(
            {'time_mode': 'physics_step', 'status': 'complete', 'input_sha256': 'same-data'}))
        (folder / 'tracking_error.json').write_text(json.dumps(
            {'physics_time': {'joints': {j: {'position_rmse': q, 'velocity_rmse': v} for j in JOINTS}}}))
        trials[label] = root
        configs[label] = {control_key(j): {'stiffness': gain} for j in JOINTS}
        configs[label]['joint_command_filters'] = {j: {'time_constant_s': 0.} for j in JOINTS}
    selected, report = select_completed_trials(trials, configs)
    assert all(row['selected'] == 'both_better' for row in report['joints'].values())
    assert selected[control_key(JOINTS[0])]['stiffness'] == 1200
    assert configs['baseline'][control_key(JOINTS[0])]['stiffness'] == 800
    (trials['both_better'] / 'replay/episode_0/run_metadata.json').write_text(json.dumps(
        {'time_mode': 'physics_step', 'status': 'complete', 'input_sha256': 'different-data'}))
    with pytest.raises(ValueError, match='same complete training'):
        select_completed_trials(trials, configs)
