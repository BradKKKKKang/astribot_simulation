"""Offline SDK-capture contract tests; no ROS, Isaac runtime, SDK or GPU."""
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from src.placement_analysis.ros_feedback import JOINTS, feedback_joint_positions, synchronize_feedback


def capture_fixture(count=80):
    epoch, wall = 10_000_000_000, 1_000_000_000_000
    meta = {'status': 'complete', 'command_frames': count,
            'epoch_monotonic_ns': epoch, 'epoch_system_ns': wall,
            'control_path': 'Astribot SDK -> ROS2 -> Isaac Lab -> ROS2',
            'input_parquet': 'fixture/samples.parquet', 'input_sha256': 'fixture'}
    api = [{'frame_index': i, 'call_start_monotonic_ns': epoch + i * 10_000_000,
            'call_end_monotonic_ns': epoch + i * 10_000_000 + 1000} for i in range(count)]
    rows = []
    # Include preposition and tail samples; part headers are slightly staggered.
    for i in range(-2, count + 3):
        for j, joint in enumerate(JOINTS):
            t = i * .01 + j * .00001
            gripper = 'gripper' in joint
            value = (100 if .2 <= t < .5 and 'left' in joint else 0) if gripper else t * (j + 1)
            rows.append({'part': joint.rsplit('_joint_', 1)[0], 'joint': joint,
                         'position': value, 'header_stamp_ns': wall + round(t * 1e9),
                         'frame_id': 'simulation'})
    return rows, api, meta


def test_synchronization_uses_header_time_measured_values_and_command_window():
    rows, api, meta = capture_fixture()
    t, q, quality = synchronize_feedback(rows, api, meta, JOINTS, .1)
    # The final command's 10 ms hold interval is included; the 0.2 s
    # recorder drain tail and all preposition samples are excluded.
    np.testing.assert_allclose(t, np.arange(81)*.01)
    np.testing.assert_allclose(q[:, 5], t * 6)
    assert quality['discarded_anchor_samples'] == quality['gap_count'] == 0
    assert quality['command_window_s'][0] == 0
    # SDK payload is not used to reconstruct positions.
    assert q[30, JOINTS.index('astribot_gripper_left_joint_1')] == 100


def test_synchronization_crops_async_feedback_boundary_and_reports_trim():
    rows, api, meta = capture_fixture()
    rows = [r for r in rows if not (
        r['header_stamp_ns'] < meta['epoch_system_ns'] + .16e9
        or r['header_stamp_ns'] > meta['epoch_system_ns'] + .74e9)]
    t, _, quality = synchronize_feedback(rows, api, meta, JOINTS, .1)
    assert t[0] == pytest.approx(.17)
    assert t[-1] == pytest.approx(.73)
    assert quality['command_window_s'] == [0, pytest.approx(.800001)]
    assert quality['synchronized_window_s'] == [pytest.approx(.17), pytest.approx(.73)]
    assert quality['boundary_trim_s'] == [pytest.approx(.17), pytest.approx(.070001)]


@pytest.mark.parametrize('failure', ['missing', 'hardware', 'clock', 'nan', 'stale', 'api'])
def test_reject_incomplete_non_simulation_or_invalid_captures(failure):
    rows, api, meta = capture_fixture()
    if failure == 'missing':
        rows = [r for r in rows if r['joint'] != JOINTS[-1]]
    elif failure == 'hardware':
        rows[0]['frame_id'] = 'hardware'
    elif failure == 'clock':
        rows[22]['header_stamp_ns'] = rows[0]['header_stamp_ns']
    elif failure == 'nan':
        rows[0]['position'] = float('nan')
    elif failure == 'stale':
        rows = [r for r in rows if r['joint'] != JOINTS[-1] or r['header_stamp_ns'] < meta['epoch_system_ns'] - 200_000_000]
    else:
        api.pop()
    with pytest.raises(ValueError):
        synchronize_feedback(rows, api, meta, JOINTS, .1)


def test_interior_gap_is_not_filled_with_stationary_samples():
    rows, api, meta = capture_fixture()
    rows = [r for r in rows if r['joint'] != JOINTS[-1] or not .25 < (r['header_stamp_ns']-meta['epoch_system_ns'])/1e9 < .48]
    t, _, quality = synchronize_feedback(rows, api, meta, JOINTS, .1)
    assert not np.any((t > .26) & (t < .48))
    assert quality['gap_count'] == 1
    assert quality['discarded_anchor_samples'] > 0


def test_gripper_percent_expands_signed_fingers_without_changing_arm_feedback():
    values = np.arange(len(JOINTS), dtype=float) / 100
    values[JOINTS.index('astribot_gripper_left_joint_1')] = 50
    values[JOINTS.index('astribot_gripper_right_joint_1')] = 120  # physical overshoot
    joints, grippers = feedback_joint_positions(JOINTS, values)
    assert grippers == [50, 100]
    assert joints['astribot_gripper_left_joint_L1'] == -.465
    assert joints['astribot_gripper_left_joint_R1'] == .465
    assert joints['astribot_arm_left_joint_2'] == values[JOINTS.index('astribot_arm_left_joint_2')]
    assert len(joints) == 32


def load_extractor():
    return importlib.import_module('src.placement_analysis.extraction')


def write_fixture(path):
    rows, api, meta = capture_fixture()
    path.mkdir()
    pq.write_table(pa.Table.from_pylist(rows), path / 'sim_states.parquet')
    pq.write_table(pa.Table.from_pylist(api), path / 'api_commands.parquet')
    (path / 'run_metadata.json').write_text(json.dumps(meta))


def test_dry_run_capture_validation_has_no_runtime_imports_or_output(tmp_path, monkeypatch):
    import builtins
    original = builtins.__import__

    def guard(name, *args, **kwargs):
        assert not name.startswith(('pxr', 'rclpy', 'isaac', 'omni', 'astribot_sdk', 'astribot_ros_middleware'))
        return original(name, *args, **kwargs)

    capture = tmp_path / 'episode'
    write_fixture(capture)
    monkeypatch.setattr(builtins, '__import__', guard)
    output = tmp_path / 'output'
    assert load_extractor().extract_events([capture], output, dry_run=True) is None
    assert not output.exists()


def test_raw_replay_window_excludes_blocking_sdk_preposition():
    rows, api, meta = capture_fixture()
    epoch = meta['epoch_monotonic_ns']
    api[0]['call_start_monotonic_ns'] = epoch - 3_000_000_000
    api[0]['call_end_monotonic_ns'] = epoch - 1000
    api[0]['phase'] = 'initial'
    meta['dataset_window_monotonic_ns'] = [epoch, epoch + 800_000_000]
    t, _, quality = synchronize_feedback(rows, api, meta, JOINTS, .1)
    assert t[0] == 0
    assert t[-1] == pytest.approx(.8)
    assert quality['command_window_s'] == [0, .8]
    meta['dataset_window_monotonic_ns'][1] = epoch + 500_000_000
    with pytest.raises(ValueError, match='capture window'):
        synchronize_feedback(rows, api, meta, JOINTS, .1)


def test_raw_direct_capture_window_can_start_before_first_sdk_call():
    rows, api, meta = capture_fixture()
    epoch = meta['epoch_monotonic_ns']
    for row in api:
        row['call_start_monotonic_ns'] += 10_000
        row['call_end_monotonic_ns'] += 10_000
    meta['dataset_window_monotonic_ns'] = [epoch, epoch + 800_000_000]
    t, _, _ = synchronize_feedback(rows, api, meta, JOINTS, .1)
    assert t[0] == 0


def test_completed_batch_expands_all_episodes_and_rejects_partial(tmp_path):
    module = load_extractor()
    first, second = tmp_path / 'episode_0', tmp_path / 'episode_1'
    write_fixture(first)
    write_fixture(second)
    manifest = {'status': 'complete', 'episodes': [{'episode_id': p.name} for p in (first, second)],
                'completed': [first.name, second.name]}
    (tmp_path / 'batch_manifest.json').write_text(json.dumps(manifest))
    assert module.discover_captures([tmp_path]) == [first, second]
    assert module.extract_events([tmp_path], tmp_path / 'output', dry_run=True) is None
    manifest['completed'].pop()
    manifest['status'] = 'partial'
    (tmp_path / 'batch_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='incomplete'):
        module.extract_events([tmp_path], tmp_path / 'output', dry_run=True)


def test_raw_replay_capture_saves_extractable_feedback_and_hashes(tmp_path):
    import hashlib
    import threading
    from src.placement_analysis.replay_capture import ReplayCapture
    rows, api, meta = capture_fixture()
    capture = ReplayCapture.__new__(ReplayCapture)
    capture.lock = threading.Lock()
    capture.active = True
    capture.errors = []
    capture.rows, capture.calls = rows, api
    capture.epoch_mono, capture.epoch_wall = meta['epoch_monotonic_ns'], meta['epoch_system_ns']
    capture.window = [capture.epoch_mono, capture.epoch_mono + 800_000_000]
    capture.dataset_end_monotonic_ns = capture.epoch_mono + 800_000_000
    source = {'input_parquet': 'data/episode_0.parquet', 'input_sha256': 'source fixture', 'frames': 25}
    output = tmp_path / 'episode_0'
    saved = capture.save(output, source, {'control_mode': 'move'})
    assert saved['capture_schema'] == 'replay_parquet_v1'
    assert saved['source_frames'] == 25
    for name, digest in saved['capture_sha256'].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert capture.rows == []  # release the previous episode's feedback
    assert load_extractor().extract_events([output], tmp_path / 'bundle', dry_run=True) is None


def test_capture_window_uses_last_dataset_command_endpoint(monkeypatch):
    from src.placement_analysis.replay_capture import ReplayCapture
    capture = ReplayCapture.__new__(ReplayCapture)
    capture.window = [100, None]
    capture.dataset_end_monotonic_ns = 250
    monkeypatch.setattr('src.placement_analysis.replay_capture.time.monotonic_ns', lambda: 999)
    capture.end_window()
    assert capture.window == [100, 250]


def test_terminal_feedback_requires_every_part_to_pass_command_endpoint():
    import threading
    from src.placement_analysis.replay_capture import ReplayCapture
    from src.placement_analysis.ros_feedback import PART_CONFIGS
    capture = ReplayCapture.__new__(ReplayCapture)
    capture.lock, capture.errors = threading.Lock(), []
    capture.epoch_mono, capture.epoch_wall = 100, 1000
    capture.window = [100, 200]
    capture.latest = {part: [{'header_stamp_ns': 1100}] for part, _ in PART_CONFIGS}
    capture.latest['astribot_head'][0]['header_stamp_ns'] = 1099
    assert not capture.feedback_covers_window()
    capture.latest['astribot_head'][0]['header_stamp_ns'] = 1100
    assert capture.feedback_covers_window()


def test_failed_synchronization_preserves_raw_capture_and_boundary_diagnostics(tmp_path):
    import threading
    from src.placement_analysis.replay_capture import ReplayCapture
    rows, api, meta = capture_fixture()
    # Missing a joint remains an error under the boundary cropping policy.
    rows = [r for r in rows if r['joint'] != JOINTS[-1]
            or (r['header_stamp_ns'] - meta['epoch_system_ns']) / 1e9 < -.2]
    capture = ReplayCapture.__new__(ReplayCapture)
    capture.lock, capture.errors = threading.Lock(), []
    capture.active, capture.rows, capture.calls = True, rows, api
    capture.epoch_mono, capture.epoch_wall = meta['epoch_monotonic_ns'], meta['epoch_system_ns']
    capture.window = [capture.epoch_mono, capture.epoch_mono + 800_000_000]
    source = {'input_parquet': 'fixture.parquet', 'input_sha256': 'fixture', 'frames': 25}
    output = tmp_path / 'failed'
    with pytest.raises(RuntimeError, match='(feedback does not cover|insufficient feedback).*raw capture saved'):
        capture.save(output, source, {})
    saved = json.loads((output / 'run_metadata.json').read_text())
    assert saved['status'] == 'failed'
    assert pq.read_table(output / 'sim_states.parquet').to_pylist() == rows
    assert pq.read_table(output / 'api_commands.parquet').to_pylist() == api
    with pytest.raises(ValueError, match='completed SDK replay capture'):
        load_extractor().extract_events([output], tmp_path / 'bundle', dry_run=True)


@pytest.mark.parametrize('failure', ['hardware', 'wrong_part', 'duplicate', 'nan', 'length', 'stamp'])
def test_raw_capture_rejects_invalid_state_messages(failure):
    from types import SimpleNamespace
    from src.placement_analysis.replay_capture import state_rows
    msg = SimpleNamespace(name=['astribot_head_joint_1', 'astribot_head_joint_2'], position=[.1, .2],
                          header=SimpleNamespace(frame_id='simulation',
                                                 stamp=SimpleNamespace(sec=1, nanosec=0)))
    if failure == 'hardware':
        msg.header.frame_id = 'hardware'
    elif failure == 'wrong_part':
        msg.name[0] = 'astribot_torso_joint_1'
    elif failure == 'duplicate':
        msg.name[0] = msg.name[1]
    elif failure == 'nan':
        msg.position[0] = float('nan')
    elif failure == 'length':
        msg.position.pop()
    else:
        msg.header.stamp.sec = 0
    with pytest.raises(ValueError):
        state_rows(msg, 'astribot_head')


@pytest.mark.parametrize('raw_capture', [False, True])
def test_full_capture_to_viewer_bundle_on_cpu(tmp_path, monkeypatch, raw_capture):
    pytest.importorskip('pxr.Usd')
    usd = ROOT / 'astribot_descriptions/usd/astribot_s1_usd/astribot_s1_textured.usda'
    if not usd.exists():
        pytest.skip('USD submodule asset unavailable')
    capture = tmp_path / 'episode'
    write_fixture(capture)
    if raw_capture:
        path = capture / 'run_metadata.json'
        meta = json.loads(path.read_text())
        epoch = meta['epoch_monotonic_ns']
        meta['capture_schema'] = 'replay_parquet_v1'
        meta['dataset_window_monotonic_ns'] = [epoch, epoch + 800_000_000]
        path.write_text(json.dumps(meta))
    module = load_extractor()
    # Mesh exporter has its own USD tests; avoid duplicating a 42 MB export.
    def geometry(stage, root, names, output):
        output.write_bytes(b'geometry fixture')
        return {'body_count': len(names)}
    monkeypatch.setattr(module, 'export_robot_glb', geometry)
    output = tmp_path / 'output'
    assert module.extract_events([capture], output, usd=usd) == output / 'placement.json'
    bundle = json.loads((output / 'placement.json').read_text())
    assert bundle['schema_version'] == 2
    events = [e for e in bundle['events'] if e['event_type'] != 'dwell']
    assert [e['event_type'] for e in events] == ['grasp', 'release']
    assert all(e['side'] == 'left' for e in events)
    assert len(events[0]['snapshot']['body_poses_world']) == 33
    assert bundle['metadata']['time_basis'].startswith('ROS header wall time')
    assert bundle['metadata']['status'] == 'complete'
    assert pq.read_table(output / 'events.parquet').num_rows == len(bundle['events'])
    with pytest.raises(FileExistsError):
        module.extract_events([capture], output)


def test_unified_extract_only_dry_run_never_connects_sdk_or_loads_usd(tmp_path, monkeypatch):
    import builtins
    from replay_extract_event import main
    original = builtins.__import__

    def guard(name, *args, **kwargs):
        assert not name.startswith(('pxr', 'rclpy', 'isaac', 'omni', 'astribot_sdk', 'astribot_ros_middleware'))
        return original(name, *args, **kwargs)

    capture, output = tmp_path / 'episode', tmp_path / 'bundle'
    write_fixture(capture)
    monkeypatch.setattr(builtins, '__import__', guard)
    assert main([str(capture), '--extract-only', '--dry-run', '--output', str(output)]) == 0
    assert not output.exists()


def test_unified_model_export_calls_internal_exporter_without_replay(tmp_path, monkeypatch):
    from unittest.mock import Mock
    import replay_extract_event as entry
    from src.placement_analysis import robot_export
    export, sdk = Mock(), Mock()
    monkeypatch.setattr(robot_export, 'export_reference_robot', export)
    monkeypatch.setattr(entry, 'run', sdk)
    output = tmp_path / 'S1 model'
    assert entry.main(['--export-robot-only', '--output', str(output)]) == 0
    export.assert_called_once_with(output, usd=robot_export.DEFAULT_USD,
                                   pose_config=robot_export.DEFAULT_POSE_CONFIG)
    sdk.assert_not_called()
