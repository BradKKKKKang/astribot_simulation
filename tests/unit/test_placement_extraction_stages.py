"""CPU differentials, cache dependencies and extraction checkpoint contracts."""
import copy
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.placement_analysis.events import DetectorConfig, motion_metrics
from src.placement_analysis.extraction import discover_captures, extract_events, prepare_extraction, detector_settings
from src.placement_analysis.extraction_worker import StageCache, extract_episode
from src.placement_analysis.recorder import PlacementRecorder
from src.placement_analysis.robot_export import DEFAULT_USD
from src.simulation_config.robot_profile import default_robot_profile, profile_for_settings

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'config/placement_distribution/penholder.json'


@pytest.fixture(autouse=True)
def legacy_capture_configuration(tmp_path, monkeypatch):
    """These recorded fixtures predate grounded capture evidence: retain v2 coordinates."""
    import sys
    from src.simulation_config.experiment_config import resolve_experiment
    sections = resolve_experiment(ROOT / 'config/placement_distribution/penholder.json')['sections']
    sections.pop('robot_placement', None)
    path = tmp_path / 'legacy-experiment.json'
    path.write_text(json.dumps({'schema_version': 2, 'kind': 'placement_test', **sections}))
    monkeypatch.setattr(sys.modules[__name__], 'CONFIG', path)


def write_capture(path):
    path.mkdir()
    profile = default_robot_profile()
    epoch, wall = 10_000_000_000, 1_000_000_000_000
    metadata = {'status': 'complete', 'command_frames': 80, 'epoch_monotonic_ns': epoch,
                'epoch_system_ns': wall, 'control_path': 'Astribot SDK -> ROS2 -> Isaac Lab -> ROS2',
                'input_parquet': 'fixture/samples.parquet', 'input_sha256': 'fixture'}
    api = [{'frame_index': i, 'call_start_monotonic_ns': epoch + i * 10_000_000,
            'call_end_monotonic_ns': epoch + i * 10_000_000 + 1000} for i in range(80)]
    rows = []
    for i in range(-2, 83):
        for j, joint in enumerate(profile.joint_names):
            t = i * .01 + j * .00001
            value = ((100 if .2 <= t < .5 and 'left' in joint else 0)
                     if 'gripper' in joint else t * (j + 1) / 10)
            rows.append({'part': joint.rsplit('_joint_', 1)[0], 'joint': joint,
                         'position': value, 'header_stamp_ns': wall + round(t * 1e9), 'frame_id': 'simulation'})
    pq.write_table(pa.Table.from_pylist(rows), path / 'sim_states.parquet')
    pq.write_table(pa.Table.from_pylist(api), path / 'api_commands.parquet')
    (path / 'run_metadata.json').write_text(json.dumps(metadata))
    return path


@pytest.mark.parametrize('smoothing', [1, 2, 5, 12])
def test_candidate_metrics_exact_windows_boundaries_and_gaps(smoothing):
    times = np.r_[0., .01, .02, .03, .3, .31, .6, .61, .62, .63, .64]
    positions = np.c_[np.sin(times), times**2, np.cos(times)]
    cfg = DetectorConfig(smoothing_samples=smoothing)
    full = motion_metrics(times, positions, cfg)
    indices = [0, 3, 4, 5, 6, 10]
    selected = motion_metrics(times, positions, cfg, indices=indices)
    for original, sparse in zip(full, selected):
        np.testing.assert_equal(original[indices], sparse[indices])
        assert np.isnan(sparse[[1, 2, 7, 8, 9]]).all()


@pytest.fixture
def cpu_model():
    pytest.importorskip('pxr')
    if not DEFAULT_USD.is_file():
        pytest.skip('USD asset unavailable')
    settings, detector = detector_settings(CONFIG)
    stage, root, names, model = prepare_extraction(DEFAULT_USD, settings=settings)
    return settings, detector, names, model


def test_minimal_fk_events_match_full_trajectory_and_stage_dependencies(tmp_path, cpu_model, monkeypatch):
    from src.placement_analysis.extraction_worker import normalized_capture
    import src.placement_analysis.recorder as recorder_module
    settings, detector, names, model = cpu_model
    path = write_capture(tmp_path / 'episode')
    profile = profile_for_settings(settings)
    cache = StageCache(tmp_path / 'store')
    optimized = extract_episode(path, settings, model, profile, cache)
    assert optimized['cache_hits'] == {'synchronized': False, 'endpoint_fk': False}
    _, times, positions, _, _, _ = normalized_capture(path, profile, detector.max_gap_s)
    baseline = PlacementRecorder(None, settings, path.name, body_names=names, joint_names=model.joint_names)
    for t, values in zip(times, positions):
        joints, openings = profile.feedback_joint_positions(profile.joint_names, values)
        full = model.snapshot(joints)
        baseline.append_sample(t, full['body_poses_world'], full['joint_positions_rad'], openings)
    # Baseline computes every local fit, as the original implementation did.
    original_metrics = motion_metrics
    monkeypatch.setattr(recorder_module, 'motion_metrics',
                        lambda t, p, c, **kw: original_metrics(t, p, c))
    assert baseline.finish() == optimized['events']
    assert len(optimized['events']) == 2
    assert optimized['reference'] == baseline.snapshot(0)
    selected_settings = copy.deepcopy(settings)
    selected_settings['event_rules'] = {'exclude_initial': False}
    selected = extract_episode(path, selected_settings, model, profile, cache)
    assert all(event['analysis_selected'] for event in selected['events'])
    assert all(event['analysis_exclusion_reasons'] == [] for event in selected['events'])
    assert selected['events'][0]['exclusion_reasons']  # detection annotations remain
    changed = copy.deepcopy(settings)
    changed['tcp']['left']['position'][0] += .03
    changed['confirm_duration_s'] = .09
    again = extract_episode(path, changed, model, profile, cache)
    assert again['cache_hits'] == {'synchronized': True, 'endpoint_fk': True}
    changed['tcp']['left']['body'] = names[-1]
    changed_endpoint = extract_episode(path, changed, model, profile, cache)
    assert changed_endpoint['cache_hits'] == {'synchronized': True, 'endpoint_fk': False}
    gap_changed = extract_episode(path, settings, model, profile, cache, alignment_max_gap_s=.2)
    assert gap_changed['cache_hits'] == {'synchronized': False, 'endpoint_fk': False}
    meta_path = path / 'run_metadata.json'
    metadata = json.loads(meta_path.read_text()); metadata['revision'] = 2
    meta_path.write_text(json.dumps(metadata))
    source_changed = extract_episode(path, settings, model, profile, cache)
    assert source_changed['cache_hits'] == {'synchronized': False, 'endpoint_fk': False}
    old_fingerprint = model.fingerprint()
    base_path = model.bodies[settings['base_body']]
    # S1's base has an incoming world-fixed joint: its parent frame is the
    # kinematic dependency; authored base geometry transforms are not.
    model.incoming[base_path][2].SetTranslateOnly(model.Gf.Vec3d(1, 0, 0))
    assert model.fingerprint() != old_fingerprint
    model_changed = extract_episode(path, settings, model, profile, cache)
    assert model_changed['cache_hits'] == {'synchronized': True, 'endpoint_fk': False}


def test_partial_batch_is_explicit(tmp_path):
    capture = write_capture(tmp_path / 'episode0')
    (tmp_path / 'batch_manifest.json').write_text(json.dumps({
        'episodes': [{'episode_id': 'episode0'}, {'episode_id': 'episode1'}],
        'completed': ['episode0'], 'status': 'running'}))
    with pytest.raises(ValueError, match='incomplete'):
        discover_captures([tmp_path])
    assert discover_captures([tmp_path], incremental=True) == [capture]


def test_interrupted_resume_incremental_and_workers_match(tmp_path, cpu_model, monkeypatch):
    import src.placement_analysis.extraction as extraction
    import src.placement_analysis.robot_export as robot_export
    from src.placement_analysis.bundle import load_bundle
    captures = [write_capture(tmp_path / f'episode{i}') for i in range(2)]
    # Geometry itself has independent USD tests. This test isolates extraction.
    monkeypatch.setattr(robot_export, 'cached_robot_geometry',
                        lambda stage, root, names, store, **kw: store.put_bytes(b'fixture', 'glb'))
    output, store = tmp_path / 'serial', tmp_path / 'artifacts'
    original = extraction.extract_episode
    def interrupted(path, *args, **kwargs):
        if Path(path).name == 'episode1':
            raise RuntimeError('interrupted fixture')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(extraction, 'extract_episode', interrupted)
    with pytest.raises(RuntimeError, match='interrupted fixture'):
        extract_events(captures, output, detector_config=CONFIG, store_root=store)
    partial = load_bundle(output)
    assert partial['metadata']['status'] == 'partial'
    assert len(partial['episodes']) == 1
    # Completed episode must never execute again after interruption.
    def resumed(path, *args, **kwargs):
        assert Path(path).name == 'episode1'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(extraction, 'extract_episode', resumed)
    extract_events(captures, output, detector_config=CONFIG, store_root=store, resume=True)
    serial = load_bundle(output)
    assert serial['metadata']['status'] == 'complete'
    assert not (output / 'events.parquet').exists()
    assert not (output / 'robot.glb').exists()
    monkeypatch.setattr(extraction, 'extract_episode', original)
    parallel = tmp_path / 'parallel'
    extract_events(captures, parallel, detector_config=CONFIG, store_root=store, workers=2)
    assert load_bundle(parallel)['events'] == serial['events']
    from src.placement_analysis.bundle import save_bundle_revision
    def mark_downstream():
        expanded = load_bundle(output)
        expanded['events'][0]['object'] = 'recognized_fixture'
        expanded['metadata']['stages'].update(identification='complete', evaluation='complete')
        save_bundle_revision(output, expanded['events'], expanded['metadata'], expanded['reference_snapshot'])
    mark_downstream()
    previous_analysis_id = serial['metadata']['analysis_id']
    third = write_capture(tmp_path / 'episode2')
    extract_events([*captures, third], output, detector_config=CONFIG, store_root=store,
                   resume=True, incremental=True)
    appended = load_bundle(output)
    assert len(appended['episodes']) == 3
    assert appended['metadata']['analysis_id'] != previous_analysis_id
    assert appended['metadata']['stages']['identification'] == 'not_requested'
    assert appended['metadata']['stages']['evaluation'] == 'not_requested'
    mark_downstream()
    # Identification configuration is not an extraction dependency.
    changed_settings = copy.deepcopy(cpu_model[0])
    changed_settings.setdefault('identification', {})['model'] = 'unrelated-model'
    def no_episode_work(*args, **kwargs):
        pytest.fail('completed extraction reran for identification-only change')
    monkeypatch.setattr(extraction, 'extract_episode', no_episode_work)
    extract_events([*captures, third], output, resolved={'settings': changed_settings},
                   store_root=store, resume=True, incremental=True)
    preserved = load_bundle(output)
    assert preserved['events'][0]['object'] == 'recognized_fixture'
    assert preserved['metadata']['stages']['identification'] == 'complete'
    assert preserved['metadata']['stages']['evaluation'] == 'complete'
    changed_settings['event_rules']['exclude_initial'] = not changed_settings['event_rules']['exclude_initial']
    with pytest.raises(ValueError, match='configuration/store differs'):
        extract_events([*captures, third], output, resolved={'settings': changed_settings},
                       store_root=store, resume=True, incremental=True)
    with pytest.raises(ValueError, match='include all'):
        extract_events([third], output, detector_config=CONFIG, store_root=store, resume=True)


def test_feedback_alignment_accepts_explicit_profile_names():
    from src.placement_analysis.ros_feedback import synchronize_feedback
    metadata = {'status': 'complete', 'command_frames': 2,
                'epoch_monotonic_ns': 100, 'epoch_system_ns': 1000}
    api = [{'frame_index': i, 'call_start_monotonic_ns': 100 + i * 10_000_000,
            'call_end_monotonic_ns': 101 + i * 10_000_000} for i in range(2)]
    rows = [{'joint': joint, 'part': 'arm', 'frame_id': 'simulation',
             'header_stamp_ns': 1000 + i * 10_000_000, 'position': i}
            for i in range(3) for joint in ('shoulder', 'wrist')]
    times, positions, _ = synchronize_feedback(
        rows, api, metadata, ('shoulder', 'wrist'), .1,
        anchor_joint='shoulder', part_by_joint={'shoulder': 'arm', 'wrist': 'arm'})
    np.testing.assert_equal(times, [0, .01, .02])
    np.testing.assert_equal(positions, [[0, 0], [1, 1], [2, 2]])


def test_competing_pipeline_stage_cannot_write_extraction(tmp_path, cpu_model, monkeypatch):
    import fcntl
    import src.placement_analysis.robot_export as robot_export
    capture = write_capture(tmp_path / 'episode')
    output = tmp_path / 'bundle'
    output.mkdir()
    # A resumable empty checkpoint permits reaching the shared writer gate.
    (output / '.extraction-checkpoint.json').write_text('{}')
    monkeypatch.setattr(robot_export, 'cached_robot_geometry',
                        lambda *a, **k: pytest.fail('competing stage reached geometry/write work'))
    with (output / '.pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='another pipeline coordinator'):
            extract_events([capture], output, detector_config=CONFIG, resume=True)
    assert (output / '.extraction-checkpoint.json').read_text() == '{}'
    assert not (output / 'placement.json').exists()


@pytest.mark.parametrize('changed', ['source', 'model'])
def test_coordinator_rejects_worker_provenance_drift(tmp_path, cpu_model, monkeypatch, changed):
    import src.placement_analysis.extraction as extraction
    import src.placement_analysis.robot_export as robot_export
    capture = write_capture(tmp_path / 'episode')
    output = tmp_path / 'bundle'
    monkeypatch.setattr(robot_export, 'cached_robot_geometry',
                        lambda stage, root, names, store, **kw: store.put_bytes(b'fixture', 'glb'))
    original = extraction.extract_episode
    def drift(*args, **kwargs):
        result = original(*args, **kwargs)
        if changed == 'source':
            result['episode']['capture_sha256']['sim_states.parquet'] = '0' * 64
        else:
            result['kinematics_sha256'] = '0' * 64
        return result
    monkeypatch.setattr(extraction, 'extract_episode', drift)
    with pytest.raises(ValueError, match='worker source/model changed'):
        extract_events([capture], output, detector_config=CONFIG)
    assert not (output / 'placement.json').exists()
    assert json.loads((output / '.extraction-checkpoint.json').read_text())['episodes'] == {}


def test_v2_replay_to_real_cpu_extraction_with_recorded_feedback(tmp_path, cpu_model, monkeypatch):
    """Mock only SDK capture; run real CLI children, USD FK and artifact export."""
    from scripts import replay_extract_event as replay
    from src.placement_analysis.bundle import load_bundle
    source = tmp_path / 'source.parquet'
    pq.write_table(pa.table({replay.COMMAND_COLUMN: [[0.] * 25] * 80,
                             'timestamp': np.arange(80) * .01}), source)
    capture_root, output = tmp_path / 'captures', tmp_path / 'bundle'
    def capture(args, *, report):
        args.output_root.mkdir()
        path = write_capture(args.output_root / report['episodes'][0]['episode_id'])
        replay.write_json(args.output_root / 'batch_manifest.json', {
            'status': 'complete', 'episodes': [{'episode_id': path.name}], 'completed': [path.name]})
    monkeypatch.setattr(replay, 'run', capture)
    assert replay.main([str(source), '--experiment', str(CONFIG), '--no-viewer',
                        '--no-identify', '--no-web-publish', '--no-progress',
                        '--output-root', str(capture_root), '--output', str(output)]) == 0
    bundle = load_bundle(output)
    assert len(bundle['events']) == 2
    assert bundle['metadata']['status'] == 'complete'
    assert not {'scene', 'task', 'success'} & bundle['metadata']['task_config']['_resolved_sections'].keys()
    assert 'evaluation' not in bundle['metadata']
    assert bundle['robot_model']


def test_v3_grounded_capture_uses_measured_world_pose_and_rejects_legacy(tmp_path):
    pytest.importorskip('pxr')
    from src.simulation_config.robot_placement import identity
    settings, _ = detector_settings(ROOT / 'config/placement_distribution/penholder.json')
    settings['_resolved_sections']['robot_placement'].update(xy=[1., 2.], yaw=.5)
    _, _, names, model = prepare_extraction(DEFAULT_USD, settings=settings)
    profile = profile_for_settings(settings)
    path = write_capture(tmp_path / 'episode')
    with pytest.raises(ValueError, match='legacy coordinates'):
        extract_episode(path, settings, model, profile, StageCache(tmp_path / 'cache'))
    metadata_path = path / 'run_metadata.json'
    metadata = json.loads(metadata_path.read_text())
    epoch, wall = metadata['epoch_monotonic_ns'], metadata['epoch_system_ns']
    metadata['dataset_window_monotonic_ns'] = [epoch, epoch + 790_001_000]
    actual = list(model.placement['base_pose_world'])
    actual[0] += 2e-6
    sample = {'source': 'isaaclab.body_pose_w', 'observed_system_ns': wall,
              'base_body': profile.base_body, 'actual_base_pose_world': actual,
              'declaration': model.placement['declaration'], 'resolved': model.placement,
              'robot_control_sha256': identity(profile.control_provenance())}
    metadata['robot_placement_attestations'] = [sample, {**sample, 'observed_system_ns': wall + 1_000_000_000}]
    metadata_path.write_text(json.dumps(metadata))
    result = extract_episode(path, settings, model, profile, StageCache(tmp_path / 'cache'))
    np.testing.assert_allclose(result['reference']['body_poses_world'][names.index(profile.base_body)], actual)
    assert result['episode']['robot_placement_actual']['actual_base_pose_world'] == actual
    assert result['episode']['robot_kinematics_sha256'] != model.fingerprint()
    for event in result['events']:
        assert event['base_pose_world'] == pytest.approx(actual)
