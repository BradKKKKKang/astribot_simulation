"""CPU-only behavior checks; no ROS, Isaac Sim, SDK, or GPU imports."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from src.placement_analysis.events import (
    DetectorConfig, compose_pose, detect_presses, detect_stops, detect_transitions, initial_phase_mask,
    motion_metrics, relative_pose,
)
from src.placement_analysis.recorder import PlacementRecorder, save_bundle


def test_independent_hands_and_no_initial_event():
    times = np.arange(40) * .01
    config = DetectorConfig(confirm_duration_s=.03, larger_is_open=True)
    left = np.r_[np.full(10, 100), np.zeros(10), np.full(20, 100)]
    right = np.r_[np.zeros(20), np.full(20, 100)]
    events = detect_transitions(times, left, config)
    assert [(e['event_type'], e['frame_index']) for e in events] == [('grasp', 10), ('release', 20)]
    assert [(e['event_type'], e['frame_index']) for e in detect_transitions(times, right, config)] == [('release', 20)]
    assert detect_transitions(times, np.zeros(40), config) == []


def test_debounce_deadband_partial_close_and_gap():
    config = DetectorConfig(confirm_duration_s=.03, larger_is_open=True)
    values = [100]*8 + [0, 100]*3 + [60, 40, 20] + [20]*8 + [30, 50, 90] + [90]*8
    events = detect_transitions(np.arange(len(values))*.01, values, config)
    assert [e['event_type'] for e in events] == ['grasp', 'release']
    assert events[0]['frame_index'] == 16
    assert events[1]['frame_index'] == 25  # departure from closed, not fully open
    gap_times = np.r_[np.arange(10)*.01, 1+np.arange(10)*.01]
    assert detect_transitions(gap_times, [100]*10 + [0]*10, config) == []
    # Invert dataset polarity explicitly rather than guessing from a trajectory.
    inverted = DetectorConfig(confirm_duration_s=.03, larger_is_open=False)
    assert detect_transitions(np.arange(20)*.01, [0]*10 + [100]*10, inverted)[0]['event_type'] == 'grasp'


@pytest.mark.parametrize('times,values', [([0, 0], [0, 1]), ([0, .01], [0, float('nan')]),
                                        ([0, .01], [0, 200])])
def test_reject_invalid_input(times, values):
    with pytest.raises(ValueError):
        detect_transitions(times, values)


def test_motion_uses_sample_times_and_low_acceleration_is_not_low_speed():
    times = np.arange(20)*.01
    positions = np.c_[2*times, np.zeros((20, 2))]
    speed, acceleration = motion_metrics(times, positions)
    np.testing.assert_allclose(speed, 2, atol=1e-10)
    np.testing.assert_allclose(acceleration, 0, atol=1e-9)
    curved = np.c_[times**2, np.zeros((20, 2))]
    speed, acceleration = motion_metrics(times, curved)
    np.testing.assert_allclose(speed, 2*times, atol=1e-10)
    np.testing.assert_allclose(acceleration, 2, atol=1e-9)


def test_pose_conversion_includes_base_rotation_and_tcp_offset():
    q = np.sqrt(.5)
    base = [1, 2, 0, q, 0, 0, q]  # base +X points along world +Y
    local = [1, 0, .5, 1, 0, 0, 0]
    world = compose_pose(base, local)
    np.testing.assert_allclose(world[:3], [1, 3, .5], atol=1e-12)
    np.testing.assert_allclose(relative_pose(base, world), local, atol=1e-12)


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __getitem__(self, item):
        return Tensor(self.value[item])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def test_recorder_measured_tcp_frames_event_snapshot_and_typed_empty_output(tmp_path):
    settings = json.loads((Path(__file__).parents[2] / 'config/placement_analysis.json').read_text())
    settings['confirm_duration_s'] = .02
    names = [settings['base_body'], settings['tcp']['left']['body'], settings['tcp']['right']['body']]
    joint_names = [f'astribot_gripper_{s}_joint_L1' for s in ('left', 'right')]
    data = SimpleNamespace(joint_names=joint_names)
    robot = SimpleNamespace(body_names=names, data=data)
    env = SimpleNamespace(robot=robot, joint_name_to_index=dict(zip(joint_names, range(2))),
                          _gripper_joint_to_percent=lambda name, value: value)
    recorder = PlacementRecorder(env, settings, 'single-arm')
    poses = [[1, 2, 0, 1, 0, 0, 0], [1.5, 2.2, 1, 1, 0, 0, 0], [1.5, 1.8, 1, 1, 0, 0, 0]]
    for index in range(30):
        data.body_link_pose_w = Tensor([poses])
        data.joint_pos = Tensor([[0 if index < 10 or index >= 20 else 100, 0]])
        recorder.sample(index*.01)
    events = recorder.finish()
    transitions = [e for e in events if e['event_type'] != 'dwell']
    assert [(e['side'], e['event_type']) for e in transitions] == [('left', 'grasp'), ('left', 'release')]
    np.testing.assert_allclose(transitions[0]['tcp_pose_world'][:3], [1.5, 2.005, 1])
    np.testing.assert_allclose(transitions[0]['tcp_pose_base'][:3], [.5, .005, 1])
    assert transitions[0]['low_motion']
    assert 'motion' in transitions[0]['detection_sources']
    assert all(e['initial_phase'] and e['exclusion_reasons'] for e in events)
    assert len(events[0]['snapshot']['body_poses_world']) == 3
    metadata = {'body_names': names}
    save_bundle(tmp_path / 'full', events, metadata, recorder.snapshot(0))
    table = pq.read_table(tmp_path / 'full/events.parquet')
    assert table.num_rows == len(events) == 4
    assert table['body_poses_world'][0].as_py() == events[0]['snapshot']['body_poses_world']
    save_bundle(tmp_path / 'empty', [], metadata, recorder.snapshot(0))
    empty = pq.read_table(tmp_path / 'empty/events.parquet')
    assert empty.num_rows == 0 and empty.schema == table.schema
    assert not (tmp_path / 'full/trajectory.parquet').exists()
    assert json.loads((tmp_path / 'full/placement.json').read_text())['schema_version'] == 2
    # An instant low-motion frame is insufficient evidence for the combined
    # path when no sustained stop meets the configured minimum duration.
    recorder.config = replace(recorder.config, dwell_duration_s=1)
    instant = recorder.finish()
    assert len(instant) == 2
    assert all(e['low_motion'] and e['detection_sources'] == ['gripper'] for e in instant)


def test_stop_windows_require_both_metrics_duration_and_motion_rearm():
    times = np.arange(150) * .01
    positions = np.zeros((150, 3))
    speed, acceleration = np.zeros(150), np.zeros(150)
    # Initial idle, one real motion, a stop, threshold chatter, then a second
    # proper movement and stop. Chatter at the same point must not duplicate it.
    speed[30:50] = .2
    positions[30:50, 0] = np.linspace(0, .1, 20)
    positions[50:100, 0] = .1
    speed[75:77] = .03
    speed[100:120] = .2
    positions[100:120, 0] = np.linspace(.1, .2, 20)
    positions[120:, 0] = .2
    config = DetectorConfig(dwell_duration_s=.15)
    stops = detect_stops(times, positions, speed, acceleration, config)
    assert [(e['transition_start_index'], e['transition_end_index']) for e in stops] == [(0, 29), (50, 74), (120, 149)]
    assert detect_stops(times, positions, np.full(150, .1), acceleration, config) == []
    assert detect_stops(times, positions, np.zeros(150), np.ones(150), config) == []
    assert detect_stops(times[:10], positions[:10], speed[:10], acceleration[:10], config) == []
    gap = np.r_[np.arange(10)*.01, 1+np.arange(10)*.01]
    assert detect_stops(gap, positions[:20], np.zeros(20), np.zeros(20), config) == []


def test_initial_departure_is_per_hand_sustained_and_return_does_not_exclude():
    times = 20 + np.arange(100) * .01
    positions = np.zeros((100, 3))
    positions[10:12, 0] = .1  # transient spike must not arm
    positions[40:70, 0] = .1
    config = DetectorConfig(initial_departure_duration_s=.1)
    mask, departure, distance = initial_phase_mask(times, positions, config)
    assert departure == 40
    assert mask[:40].all() and not mask[40:].any()
    assert distance[80] == 0 and not mask[80]  # return is legitimate task motion
    stationary, no_departure, _ = initial_phase_mask(times, np.zeros_like(positions), config)
    assert stationary.all() and no_departure is None


def test_recorder_motion_only_press_candidates_and_same_hand_association():
    settings = json.loads((Path(__file__).parents[2] / 'config/placement_analysis.json').read_text())
    settings.update(dwell_duration_s=.1, confirm_duration_s=.02, initial_departure_duration_s=.05)
    names = [settings['base_body'], settings['tcp']['left']['body'], settings['tcp']['right']['body']]
    joint_names = [f'astribot_gripper_{s}_joint_L1' for s in ('left', 'right')]
    data = SimpleNamespace(joint_names=joint_names)
    env = SimpleNamespace(robot=SimpleNamespace(body_names=names, data=data),
                          joint_name_to_index=dict(zip(joint_names, range(2))),
                          _gripper_joint_to_percent=lambda name, value: value)
    recorder = PlacementRecorder(env, settings, 'press')
    for index in range(200):
        x = 0 if index < 30 else min((index-30)*.005, .2)
        poses = [[0, 0, 0, 1, 0, 0, 0], [x, .2, 1, 1, 0, 0, 0], [0, -.2, 1, 1, 0, 0, 0]]
        data.body_link_pose_w = Tensor([poses])
        # Only the right gripper closes while the left arm makes a press stop.
        data.joint_pos = Tensor([[0, 100 if index >= 100 else 0]])
        recorder.sample(5 + index * .01)
    events = recorder.finish()
    press = next(e for e in events if e['side'] == 'left' and e['event_type'] == 'dwell' and not e['initial_phase'])
    assert press['detection_sources'] == ['motion'] and not press['related_event_ids']
    assert press['departure_time_s'] is not None
    assert press['episode_elapsed_s'] > 1
    right = next(e for e in events if e['event_type'] == 'grasp')
    assert right['initial_phase'] and 'initial_pose' in right['exclusion_reasons']
    assert right['event_id'] not in press['related_event_ids']


def press_trajectory(rate=80):
    times = np.arange(0, 7, 1 / rate)
    z = np.interp(times, [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7],
                  [.8, .8, .74, .8, .8, .74, .74, .8, .8, .8, .74, .8, .8])
    return times, np.c_[np.zeros((len(times), 2)), z]


@pytest.mark.parametrize('rate', [30, 80, 120])
def test_press_cycles_require_no_low_acceleration_or_fixed_count(rate):
    times, positions = press_trajectory(rate)
    positions[:, 2] += .0005 * np.sin(times * 53)  # feedback jitter
    config = DetectorConfig(motion_detector='press')
    events = detect_presses(times, positions, np.full(len(times), 100), config)
    assert len(events) == 3
    stamps = [times[event['frame_index']] for event in events]
    assert 1.4 < stamps[0] < 1.6 and 3 <= stamps[1] <= 3.5 and 5.4 < stamps[2] < 5.6
    for event in events:
        assert event['transition_start_index'] < event['frame_index'] < event['confirmed_index']
        assert event['press_depth_m'] >= config.press_min_depth_m
        assert event['press_retract_m'] >= config.press_min_retract_m
    # A subset with one completed stroke yields one candidate, never a forced 3.
    keep = times < 2.4
    assert len(detect_presses(times[keep], positions[keep], np.full(keep.sum(), 100), config)) == 1


def test_press_rejects_idle_horizontal_transit_open_gripper_and_missing_lift():
    times, positions = press_trajectory()
    closed = np.full(len(times), 100)
    config = DetectorConfig(motion_detector='press')
    assert detect_presses(times, np.zeros_like(positions), closed, config) == []
    transit = positions.copy()
    transit[:, 0] = times * .3
    assert detect_presses(times, transit, closed, config) == []
    assert detect_presses(times, positions, np.zeros(len(times)), config) == []
    # Polarity is explicit, and tasks with an open pressing tool may disable gating.
    assert len(detect_presses(times, positions, np.zeros(len(times)), replace(config, larger_is_open=True))) == 3
    assert len(detect_presses(times, positions, np.zeros(len(times)), replace(config, press_require_closed=False))) == 3
    keep = times < 1.51
    assert detect_presses(times[keep], positions[keep], closed[keep], config) == []
    # A missing interval covers the only valley. Neither smoothing nor detection
    # may bridge it to claim a completed press.
    keep = (times < 1.4) | ((times > 1.6) & (times < 2.4))
    assert detect_presses(times[keep], positions[keep], closed[keep], config) == []


def test_press_recorder_and_parquet_preserve_measured_pose_and_evidence(tmp_path):
    settings = json.loads((Path(__file__).parents[2] / 'config/placement_press_buttons.json').read_text())
    names = [settings['base_body'], settings['tcp']['left']['body'], settings['tcp']['right']['body']]
    joint_names = [f'astribot_gripper_{s}_joint_L1' for s in ('left', 'right')]
    recorder = PlacementRecorder(None, settings, 'press', body_names=names, joint_names=joint_names)
    times, positions = press_trajectory()
    for timestamp, point in zip(times, positions):
        recorder.append_sample(timestamp,
                               [[0, 0, 0, 1, 0, 0, 0], [*point, 1, 0, 0, 0], [0, -.2, .8, 1, 0, 0, 0]],
                               [-.93, 0], [100, 0])
    events = recorder.finish()
    assert len(events) == 3 and all(e['side'] == 'left' and e['event_type'] == 'press' for e in events)
    for event in events:
        assert event['detection_sources'] == ['motion']
        assert event['snapshot'] == recorder.snapshot(event['frame_index'])
        assert event['tcp_pose_world'][2] == positions[event['frame_index'], 2]
    save_bundle(tmp_path, events, {'detector': {'motion_detector': 'press'}}, recorder.snapshot(0))
    bundle = json.loads((tmp_path / 'placement.json').read_text())
    assert bundle['schema_version'] == 3
    rows = pq.read_table(tmp_path / 'events.parquet').to_pylist()
    assert rows[0]['press_depth_m'] == events[0]['press_depth_m']
    assert rows[0]['press_lateral_m'] == events[0]['press_lateral_m']


@pytest.mark.parametrize('settings', [dict(motion_detector='unknown'), dict(press_smoothing_s=0),
                                    dict(press_min_depth_m=float('nan')), dict(press_require_closed='yes'),
                                    dict(press_min_phase_s=2)])
def test_press_settings_reject_invalid_values(settings):
    with pytest.raises(ValueError):
        DetectorConfig(**settings)
