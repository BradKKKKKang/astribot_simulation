"""Configured replay contracts using pure CPU input and message mocks."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import replay_extract_event as replay
from src.simulation_config.robot_profile import default_robot_profile, RobotProfile
from src.placement_analysis.replay_capture import ReplayCapture, state_rows
from src.placement_analysis.task_config import load_task_config, DEFAULT_DETECTOR_CONFIG


def custom_profile():
    data = default_robot_profile().as_dict()
    data['dataset_joint_names'] = ['q' + str(i) for i in range(25)]
    data['parts'][0]['name'] = 'custom_torso'
    data['parts'][0]['feedback_joints'] = ['torso' + str(i) for i in range(4)]
    data['fixed_chassis']['part'] = 'fixed_base'
    return RobotProfile(data)


def test_profile_dataset_order_feedback_names_and_command_layout(tmp_path):
    profile = custom_profile()
    commands = np.zeros((3,25)); commands[:,3] = [0,.1,.2]
    path = tmp_path/'episode.parquet'
    names = list(reversed(profile.dataset_joint_names))
    pq.write_table(pa.table({replay.COMMAND_COLUMN: commands[:,::-1].tolist(), 'timestamp':[0,.01,.02]}),path)
    metadata = tmp_path/'info.json'
    metadata.write_text(json.dumps({'features':{replay.COMMAND_COLUMN:{'shape':[25],'names':names}}}))
    args = replay.parser().parse_args([str(path),'--metadata',str(metadata)])
    args.robot_profile = profile
    loaded, _ = replay.load_segment(args,quiet=True)
    np.testing.assert_allclose(loaded,commands)
    assert replay.split_command(loaded[-1],profile)[0][0] == .2
    message = SimpleNamespace(name=['torso3','torso2','torso1','torso0'],position=[.3,.2,.1,0],
                              header=SimpleNamespace(frame_id='simulation',stamp=SimpleNamespace(sec=1,nanosec=2)))
    assert [row['joint'] for row in state_rows(message,'custom_torso',profile)] == message.name
    capture = ReplayCapture.__new__(ReplayCapture)
    capture.profile, capture.calls = profile, []
    capture.record_call(0,0,loaded[0],1,2)
    assert len(capture.calls[0]['position']) == len(profile.joint_names)


def test_profile_precedence_and_capture_provenance_independent_of_detection():
    task = load_task_config(DEFAULT_DETECTOR_CONFIG)
    task['_resolved_sections']['runtime']['replay'] = {'speed_scale':.5,'control_hz':80}
    args = replay.parser().parse_args(['episode.parquet','--speed','2'])
    replay.apply_experiment_settings(args,task,{'--speed'})
    assert args.speed == 2 and replay.runtime_settings(args)['control_hz'] == 80
    before = replay.replay_settings(args)
    task['press_min_depth_m'] = .09
    task['_resolved_sections']['candidates']['press_min_depth_m'] = .09
    task['_resolved_sections']['jev']['transport']['timeout_s'] = 5
    replay.apply_experiment_settings(args,task,{'--speed'})
    assert replay.replay_settings(args) == before
    task['_resolved_sections']['robot']['simulation_overrides'] = {'gripper_left_control':{'stiffness':900}}
    replay.apply_experiment_settings(args,task,{'--speed'})
    assert replay.replay_settings(args)['robot_control_sha256'] != before['robot_control_sha256']


def test_capture_only_dry_run_uses_config_without_loading_sdk_or_usd(tmp_path,monkeypatch):
    path=tmp_path/'episode.parquet'
    pq.write_table(pa.table({replay.COMMAND_COLUMN:[[0.]*25]*3,'timestamp':[0.,.01,.02]}),path)
    sdk=Mock(side_effect=AssertionError('no SDK during validation'))
    monkeypatch.setattr(replay,'load_sdk',sdk)
    assert replay.main([str(path),'--capture-only','--dry-run','--usd',str(tmp_path/'missing.usda'),
                        '--output-root',str(tmp_path/'capture')]) == 0
    sdk.assert_not_called()
    assert not (tmp_path/'capture').exists()


def test_default_detector_replay_and_extraction_share_placement():
    from scripts import replay_extract_event as replay
    args = replay.parser().parse_args(['fixture.parquet'])
    settings = replay.pipeline_robot_settings(args)
    replay.apply_experiment_settings(args, settings)
    assert replay.replay_settings(args)['robot_placement'] == settings['_resolved_sections']['robot_placement']


def test_v3_resume_and_dryrun_require_actual_placement_evidence(tmp_path):
    import hashlib
    from src.placement_analysis.extraction import extract_events
    from tests.unit.test_placement_extraction_stages import write_capture
    settings = load_task_config(DEFAULT_DETECTOR_CONFIG)
    args = replay.parser().parse_args(['fixture.parquet'])
    replay.apply_experiment_settings(args, settings)
    capture = write_capture(tmp_path / 'episode')
    # Dry-run checks capture evidence even though it does not load USD.
    with pytest.raises(ValueError, match='legacy coordinates'):
        extract_events([capture], tmp_path / 'out', detector_config=DEFAULT_DETECTOR_CONFIG, dry_run=True)
    metadata_path = capture / 'run_metadata.json'
    metadata = json.loads(metadata_path.read_text())
    source = {'input_sha256': metadata['input_sha256'], 'metadata_sha256': None}
    metadata.update(capture_schema='replay_parquet_v1', source=source,
                    replay_settings=replay.replay_settings(args),
                    capture_sha256={name: hashlib.sha256((capture/name).read_bytes()).hexdigest()
                                    for name in ('sim_states.parquet', 'api_commands.parquet')})
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='legacy coordinates'):
        replay.completed_capture(capture, source, replay.replay_settings(args), True, profile=args.robot_profile)


def test_legacy_capture_rejects_grounded_feedback_received_after_probe():
    from threading import Lock
    capture = ReplayCapture.__new__(ReplayCapture)
    capture.robot_placement = None
    capture.lock, capture.errors = Lock(), []
    capture.on_placement(SimpleNamespace(data='{}'))
    assert capture.errors == ['legacy capture cannot use explicit grounded simulator coordinates']
