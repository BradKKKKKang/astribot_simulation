"""Offline source mapping and resumable JEV tests with real, tiny video files."""
import copy
import json
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.placement_analysis import identification as ident
from src.placement_analysis import jev
from src.placement_analysis.recorder import save_bundle
from src.placement_analysis.task_config import DEFAULT_DETECTOR_CONFIG, load_task_config, validate_task_config


@pytest.fixture
def sample(tmp_path):
    settings = load_task_config(DEFAULT_DETECTOR_CONFIG)
    root = tmp_path / 'data-root'
    dataset = root / settings['task']['dataset']
    (dataset / 'meta').mkdir(parents=True)
    info = {'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
            'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.avi',
            'chunks_size': 1000, 'features': {'images_dict.head.rgb': {'dtype': 'video'}}}
    ident.write_json(dataset / 'meta/info.json', info)
    source = dataset / 'data/chunk-000/episode_000000.parquet'
    source.parent.mkdir(parents=True)
    pq.write_table(pa.table({'timestamp': np.arange(12) / 10, 'frame_index': range(12),
                             'episode_index': [0] * 12, 'index': range(12)}), source)
    video = dataset / 'videos/chunk-000/images_dict.head.rgb/episode_000000.avi'
    video.parent.mkdir(parents=True)
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
    assert writer.isOpened()
    for i in range(12):
        writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    writer.release()
    capture = tmp_path / 'capture'
    capture.mkdir()
    epoch = 10_000_000_000
    commands = [dict(frame_index=i, source_frame_index=row, phase='dataset',
                     call_start_monotonic_ns=epoch+int(t*1e9))
                for i, (row, t) in enumerate([(0, 0), (0, .025), (1, .05), (2, .1), (3, .15)])]
    pq.write_table(pa.Table.from_pylist(commands), capture / 'api_commands.parquet')
    metadata = dict(capture_schema='replay_parquet_v1', status='complete',
                    input_parquet='/old/press_buttons/data/chunk-000/episode_000000.parquet',
                    input_sha256=ident.sha256(source), command_frames=len(commands),
                    epoch_monotonic_ns=epoch, dataset_window_monotonic_ns=[epoch, epoch+160_000_000],
                    replay_settings=dict(start_frame=2, end_frame=6, speed=2))
    ident.write_json(capture / 'run_metadata.json', metadata)
    episode = dict(episode_id='episode_000000', capture_directory=str(capture), sha256=ident.sha256(source))
    events = [dict(event_id=f'episode_000000:{i}', episode_id=episode['episode_id'], side='left',
                   event_type='press', timestamp_s=t, initial_phase=False, exclusion_reasons=[],
                   tcp_pose_world=[0, 0, 0, 1, 0, 0, 0], tcp_pose_base=[0, 0, 0, 1, 0, 0, 0],
                   base_pose_world=[0, 0, 0, 1, 0, 0, 0], snapshot={})
              for i, t in enumerate([.01, .04, .06, .11])]
    events.append({**events[0], 'event_id': 'excluded', 'exclusion_reasons': ['initial_time']})
    output = tmp_path / 'bundle'
    save_bundle(output, events, {'episodes': [episode], 'task_config': settings, 'status': 'complete'}, {})
    (output / 'robot.glb').write_bytes(b'test model')
    return dict(evaluation_config=Path(__file__).parents[2] / 'config/experiments/evaluation/press_cube.json', output=output, root=root, settings=settings, video=video, source=source,
                commands=commands, metadata=metadata, episode=episode, events=events)


def successful(payload, key, timeout):
    return {'answers': {'interaction_object': {'choice': 'red', 'probabilities': {'red': .8, 'blue': .2, 'yellow': 0.0}}}}


def read_bundle(sample):
    return json.loads((sample['output'] / 'placement.json').read_text())


def test_configs_pending_guard_and_mutual_exclusion():
    from scripts import replay_extract_event as replay
    assert load_task_config(DEFAULT_DETECTOR_CONFIG)['task']['dataset'] == 'press_cube_merged'
    for name, dataset in [('cap_pen', 'cap_pen_merged')]:
        path = DEFAULT_DETECTOR_CONFIG.with_name(name + '.json')
        pending = load_task_config(path, require_ready=False)
        assert pending['task']['dataset'] == dataset
        assert pending['event_rules'] is None and 'motion_detector' not in pending
        assert pending['type'] is None
        with pytest.raises(ValueError, match='待标注'):
            load_task_config(path)
        assert replay.main(['--task-config', str(path), '--dry-run']) == 0
        assert replay.main(['--task-config', str(path)]) == 1
    with pytest.raises(SystemExit):
        replay.parser().parse_args(['--task-config', 'a', '--detector-config', 'b'])


@pytest.mark.parametrize('change', [dict(type=None), dict(type='union'), dict(motion_detector='press'),
                                    dict(identification={'enabled': False}),
                                    dict(identification={'event_types': ['grasp']})])
def test_task_type_and_identification_reject_conflicting_or_incomplete_rules(change):
    settings = load_task_config(DEFAULT_DETECTOR_CONFIG)
    settings.update(change)
    with pytest.raises(ValueError):
        validate_task_config(settings)


@pytest.mark.parametrize('dry_run', [False, True])
def test_penholder_assigns_fixed_object_without_source_video_or_model(sample, monkeypatch, dry_run):
    path = DEFAULT_DETECTOR_CONFIG.with_name('penholder.json')
    settings = load_task_config(path)
    assert settings['type'] == 'gripper' and settings['identification']['enabled'] is False
    bundle = read_bundle(sample)
    for event in bundle['events']:
        event['event_type'] = 'grasp'
    # No usable source metadata: a disabled model must not need it.
    bundle['metadata']['episodes'] = []
    ident.write_json(sample['output'] / 'placement.json', bundle)
    before = (sample['output'] / 'placement.json').read_bytes()
    source = Mock(side_effect=AssertionError('must not read source frames'))
    client = Mock(side_effect=AssertionError('must not call JEV'))
    monkeypatch.setattr(ident, 'EpisodeSource', source)
    monkeypatch.delenv('DECISION_API_KEY', raising=False)
    counts = ident.identify_bundle(sample['output'], task_config=path, client=client, dry_run=dry_run)
    source.assert_not_called()
    client.assert_not_called()
    assert counts['assigned'] == 4 and counts['failed'] == 0
    if dry_run:
        assert (sample['output'] / 'placement.json').read_bytes() == before
    else:
        assert [e['object'] for e in read_bundle(sample)['events']] == ['笔'] * 4 + [None]
        assert pq.read_table(sample['output'] / 'events.parquet')['object'].to_pylist() == ['笔'] * 4 + [None]


def test_mapping_speed_start_offset_repeated_commands_and_bounds(sample, tmp_path):
    source = ident.EpisodeSource(sample['episode'], sample['settings'], sample['root'])
    frames = []
    for event in sample['events'][:4]:
        mapped = source.locate(event, sample['settings']['identification'])
        path = tmp_path / f"{event['timestamp_s']}.png"
        actual = ident.video_frame(mapped, path)
        frames.append(actual['video_frame_index'])
        assert abs(cv2.imread(str(path)).mean() - actual['video_frame_index'] * 20) < 3
        assert cv2.imread(str(path)).shape == (48, 64, 3)  # full frame
    assert frames == [2, 2, 3, 4]  # two repeated commands, nonzero start, 2x playback
    for time in [-.001, .161]:
        with pytest.raises(ValueError, match='outside mapped'):
            ident.source_row(time, sample['commands'], sample['metadata'], 12)
    bad = copy.deepcopy(sample['metadata'])
    bad['replay_settings']['start_frame'] = 12
    with pytest.raises(ValueError, match='out of bounds'):
        ident.source_row(.01, sample['commands'], bad, 12)
    for offset in [-3, 12]:
        mapped = source.locate(sample['events'][0], {**sample['settings']['identification'], 'frame_offset': offset})
        with pytest.raises(ValueError, match='out of bounds'):
            ident.video_frame(mapped)


def test_failure_continuation_parquet_consistency_reuse_and_changed_inputs(sample, monkeypatch):
    monkeypatch.setenv('DECISION_API_KEY', 'unit-test-secret')
    client = Mock(side_effect=[successful(None, None, None), {'answers': {'interaction_object': {'choice': 'orange'}}},
                               TimeoutError('timeout unit-test-secret'), successful(None, None, None)])
    counts = ident.identify_bundle(sample['output'], data_root=sample['root'], client=client)
    assert counts == dict(success=2, failed=2, skipped=1, reused=0)
    bundle = read_bundle(sample)
    rows = pq.read_table(sample['output'] / 'events.parquet').to_pylist()
    assert bundle['schema_version'] == 4
    assert [e['object'] for e in bundle['events']] == [r['object'] for r in rows] == ['红色方块', None, None, '红色方块', None]
    for event, row in zip(bundle['events'], rows):
        assert row['identification']['status'] == event['identification']['status']
        if event['identification']['status'] != 'skipped':
            assert dict(row['identification']['probabilities']) == event['identification']['probabilities']
            assert row['identification']['source']['parquet_row'] == event['identification']['source']['parquet_row']
    assert 'unit-test-secret' not in (sample['output'] / 'placement.json').read_text()
    assert bundle['events'][1]['identification']['raw_response']
    assert len(list((sample['output'] / 'identification/events').glob('*.json'))) == 4
    client = Mock(side_effect=successful)
    counts = ident.identify_bundle(sample['output'], data_root=sample['root'], client=client)
    assert counts == dict(success=2, failed=0, skipped=1, reused=2)
    assert client.call_count == 2
    ident.identify_bundle(sample['output'], data_root=sample['root'], client=Mock(side_effect=AssertionError('must reuse')))
    settings = sample['settings']
    settings['identification']['question'] += '请结合接触位置判断。'
    config = sample['output'].parent / 'changed.json'
    ident.write_json(config, settings)
    client.reset_mock()
    counts = ident.identify_bundle(sample['output'], task_config=config, data_root=sample['root'], client=client)
    assert counts['success'] == 4 and client.call_count == 4
    assert all('左手' in call.args[0]['questions']['interaction_object']['instructions'] for call in client.call_args_list)


def test_checkpoint_recovers_after_interruption(sample, monkeypatch):
    monkeypatch.setenv('DECISION_API_KEY', 'test-key')
    client = Mock(side_effect=[successful(None, None, None), KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        ident.identify_bundle(sample['output'], data_root=sample['root'], client=client)
    client = Mock(side_effect=successful)
    result = ident.identify_bundle(sample['output'], data_root=sample['root'], client=client)
    assert result['reused'] == 1 and result['success'] == 3


def test_dry_run_no_decode_write_or_api(sample, monkeypatch):
    before = {p: ident.sha256(p) for p in sample['output'].rglob('*') if p.is_file()}
    client = Mock(side_effect=AssertionError('no API calls'))
    monkeypatch.setattr(cv2, 'imwrite', Mock(side_effect=AssertionError('no screenshots')))
    counts = ident.identify_bundle(sample['output'], data_root=sample['root'], dry_run=True, client=client)
    assert counts['success'] == 4
    client.assert_not_called()
    assert before == {p: ident.sha256(p) for p in sample['output'].rglob('*') if p.is_file()}


@pytest.mark.parametrize('failure', ['missing_video', 'sha_mismatch', 'missing_mapping', 'no_key'])
def test_missing_inputs_fail_explicitly_preserve_locations(sample, monkeypatch, failure):
    monkeypatch.delenv('DECISION_API_KEY', raising=False)
    if failure == 'missing_video':
        sample['video'].unlink()
    elif failure == 'sha_mismatch':
        pq.write_table(pa.table({'timestamp': [0.], 'episode_index': [0]}), sample['source'])
    elif failure == 'missing_mapping':
        sample['metadata']['capture_schema'] = 'old'
        ident.write_json(Path(sample['episode']['capture_directory']) / 'run_metadata.json', sample['metadata'])
    client = Mock()
    counts = ident.identify_bundle(sample['output'], data_root=sample['root'], client=client)
    assert counts['failed'] == 4 and counts['skipped'] == 1
    client.assert_not_called()
    bundle = read_bundle(sample)
    expected = {'missing_video': 'missing source video', 'sha_mismatch': 'SHA256 mismatch',
                'missing_mapping': 'source-frame mapping', 'no_key': 'DECISION_API_KEY'}[failure]
    assert expected in bundle['events'][0]['identification']['error']
    assert [e['tcp_pose_world'] for e in bundle['events']] == [e['tcp_pose_world'] for e in sample['events']]


def test_each_episode_uses_own_times_and_arbitrary_event_count(sample, monkeypatch):
    monkeypatch.setenv('DECISION_API_KEY', 'test-key')
    bundle = read_bundle(sample)
    bundle['metadata']['episodes'].append({**sample['episode'], 'episode_id': 'second'})
    bundle['events'].append({**sample['events'][0], 'event_id': 'second:0', 'episode_id': 'second', 'timestamp_s': .155})
    ident.write_json(sample['output'] / 'placement.json', bundle)
    result = ident.identify_bundle(sample['output'], data_root=sample['root'], client=successful)
    assert result['success'] == 5
    assert read_bundle(sample)['events'][-1]['identification']['source']['parquet_row'] == 5


def test_shared_client_custom_choices_and_payload(sample):
    source = ident.EpisodeSource(sample['episode'], sample['settings'], sample['root'])
    mapped = source.locate(sample['events'][0], sample['settings']['identification'])
    image = sample['output'] / 'image.png'
    ident.video_frame(mapped, image)
    payload = jev.build_payload(image, question_id='custom', question='左手在操作什么？', choices={'a': '笔', 'b': '笔盖'})
    assert payload['model'] == jev.DEFAULT_MODEL
    assert payload['questions']['custom']['criteria'] == {'a': '笔', 'b': '笔盖'}


def test_changed_source_frame_reidentifies_and_legacy_schema_preserves_annotations(sample, monkeypatch):
    monkeypatch.setenv('DECISION_API_KEY', 'test-key')
    ident.identify_bundle(sample['output'], data_root=sample['root'], client=successful)
    bundle = read_bundle(sample)
    bundle['events'][0]['timestamp_s'] = .155
    bundle['schema_version'] = 1
    for event in bundle['events']:
        event.pop('initial_phase', None)
    ident.write_json(sample['output'] / 'placement.json', bundle)
    client = Mock(side_effect=successful)
    counts = ident.identify_bundle(sample['output'], data_root=sample['root'], client=client)
    assert counts['success'] == 1 and counts['reused'] == 3
    assert client.call_count == 1
    assert read_bundle(sample)['metadata']['source_schema_version'] == 1
    assert read_bundle(sample)['events'][0]['identification']['source']['video_frame_index'] == 5


def test_explicit_new_root_is_verified_even_if_old_source_exists(sample):
    metadata = sample['metadata']
    metadata['input_parquet'] = str(sample['source'])
    capture = Path(sample['episode']['capture_directory'])
    ident.write_json(capture / 'run_metadata.json', metadata)
    import shutil
    new_root = sample['root'].with_name('renumbered')
    shutil.copytree(sample['root'], new_root)
    replacement = new_root / sample['settings']['task']['dataset'] / 'data/chunk-000/episode_000000.parquet'
    pq.write_table(pa.table({'timestamp': [0.], 'episode_index': [0]}), replacement)
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        ident.EpisodeSource(sample['episode'], sample['settings'], new_root)


def renumbered_source(sample, tmp_path):
    import shutil
    import pyarrow.compute as pc
    original = tmp_path / 'original.parquet'
    shutil.copyfile(sample['source'], original)
    current = sample['source'].with_name('episode_000001.parquet')
    table = pq.read_table(original)
    for name, offset in [('episode_index', 1), ('index', 12)]:
        table = table.set_column(table.schema.get_field_index(name), name, pc.add(table[name], offset))
    pq.write_table(table, current)
    shutil.copyfile(sample['video'], sample['video'].with_name('episode_000001.avi'))
    # The old same-number path now refers to another episode, as in the real merge.
    pq.write_table(pa.table({'timestamp': [0.], 'episode_index': [0]}), sample['source'])
    return original, current


def test_verified_renumbering_keeps_original_capture_and_uses_new_video(sample, tmp_path, monkeypatch):
    from src.placement_analysis.source_relocation import register_source_relocation
    original, current = renumbered_source(sample, tmp_path)
    capture = Path(sample['episode']['capture_directory'])
    before = (capture / 'run_metadata.json').read_bytes()
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        ident.EpisodeSource(sample['episode'], sample['settings'], sample['root'])
    entry = register_source_relocation(capture, original, current, sample['source'].parents[2],
                                       sample['settings']['task']['dataset'])
    assert entry['renumbering']['offsets'] == {'episode_index': 1, 'index': 12}
    assert (capture / 'run_metadata.json').read_bytes() == before
    source = ident.EpisodeSource(sample['episode'], sample['settings'], sample['root'])
    mapped = source.locate(sample['events'][0], sample['settings']['identification'])
    assert mapped['video'].endswith('episode_000001.avi')
    assert mapped['dataset_episode_index'] == 1 and mapped['parquet_row'] == 2
    assert mapped['original_parquet_sha256'] == sample['metadata']['input_sha256']
    assert mapped['parquet_sha256'] == ident.sha256(current)
    assert mapped['source_relocation_sha256'] == ident.sha256(capture / 'source_relocation.json')
    monkeypatch.setenv('DECISION_API_KEY', 'test-key')
    assert ident.identify_bundle(sample['output'], data_root=sample['root'], client=successful)['success'] == 4
    row = pq.read_table(sample['output'] / 'events.parquet').to_pylist()[0]
    assert row['identification']['source']['original_parquet_sha256'] == sample['metadata']['input_sha256']
    assert row['identification']['source']['source_relocation_sha256'] == mapped['source_relocation_sha256']


@pytest.mark.parametrize('tamper', ['original', 'current', 'trajectory_with_updated_hash', 'offsets', 'path', 'capture'])
def test_relocation_proof_cannot_bypass_source_integrity(sample, tmp_path, tamper):
    from src.placement_analysis.source_relocation import register_source_relocation
    import pyarrow.compute as pc
    original, current = renumbered_source(sample, tmp_path)
    capture = Path(sample['episode']['capture_directory'])
    register_source_relocation(capture, original, current, sample['source'].parents[2],
                               sample['settings']['task']['dataset'])
    manifest = capture / 'source_relocation.json'
    entry = json.loads(manifest.read_text())
    if tamper == 'original':
        (capture / 'source_original.parquet').write_bytes(b'invalid original proof')
    elif tamper in ('current', 'trajectory_with_updated_hash'):
        table = pq.read_table(current)
        table = table.set_column(table.schema.get_field_index('timestamp'), 'timestamp', pc.add(table['timestamp'], .1))
        pq.write_table(table, current)
        if tamper == 'trajectory_with_updated_hash':
            entry['current_sha256'] = ident.sha256(current)
    elif tamper == 'offsets':
        entry['renumbering']['offsets']['index'] = 13
    elif tamper == 'path':
        entry['parquet'] = '../outside.parquet'
    elif tamper == 'capture':
        sample['metadata']['source_frames'] = 100
        ident.write_json(capture / 'run_metadata.json', sample['metadata'])
    ident.write_json(manifest, entry)
    with pytest.raises(ValueError, match='relocation'):
        ident.EpisodeSource(sample['episode'], sample['settings'], sample['root'])


def test_register_requires_exact_old_hash_and_only_renumbering(sample, tmp_path):
    from src.placement_analysis.source_relocation import register_source_relocation
    original, current = renumbered_source(sample, tmp_path)
    capture = Path(sample['episode']['capture_directory'])
    with pytest.raises(ValueError, match='exact original'):
        register_source_relocation(capture, current, current, sample['source'].parents[2],
                                   sample['settings']['task']['dataset'])
    table = pq.read_table(current).slice(1)
    pq.write_table(table, current)
    with pytest.raises(ValueError, match='row counts'):
        register_source_relocation(capture, original, current, sample['source'].parents[2],
                                   sample['settings']['task']['dataset'])
    assert not (capture / 'source_relocation.json').exists()


def test_identification_logs_stages_and_reuse_without_requesting_again(sample, monkeypatch, capsys):
    monkeypatch.setenv('DECISION_API_KEY', 'test-secret')
    client = Mock(side_effect=successful)
    ident.identify_bundle(sample['output'], data_root=sample['root'], client=client, no_progress=True)
    log = capsys.readouterr().err
    assert '4 个有效事件，跳过 1 个候选' in log
    assert '[1/4] episode_000000:0' in log and '[4/4] episode_000000:3' in log
    assert '源数据已核验' in log and '截取完整视频帧' in log
    assert log.count('JEV 请求：') == 4
    assert 'timeout=60s' in log and log.count('检查点已保存') == 4
    assert '保存 placement.json、events.parquet' in log
    assert 'test-secret' not in log
    client.reset_mock()
    ident.identify_bundle(sample['output'], data_root=sample['root'], client=client, no_progress=True)
    log = capsys.readouterr().err
    assert log.count('复用成功') == 4 and 'JEV 请求：' not in log
    client.assert_not_called()


@pytest.mark.parametrize('model_choice', ['red', 'unknown', None, ['invalid']])
def test_highest_probability_ignores_model_choice(model_choice):
    config = load_task_config(DEFAULT_DETECTOR_CONFIG)['identification']
    raw = {'answers': {config['question_id']: {'choice': model_choice, 'probabilities': {
        'red': .0028, 'blue': .0012, 'yellow': .9960, 'not_a_candidate': 1.0,
    }}}}
    choice, probabilities = ident.parse_answer(raw, config)
    assert choice == 'yellow'
    assert probabilities == {'red': .0028, 'yellow': .9960, 'blue': .0012}


def test_highest_probability_ties_follow_configuration_order():
    config = load_task_config(DEFAULT_DETECTOR_CONFIG)['identification']
    raw = {'answers': {config['question_id']: {'choice': 'yellow', 'probabilities': {
        'yellow': .5, 'blue': 0, 'red': .5,
    }}}}
    assert ident.parse_answer(raw, config)[0] == 'red'


@pytest.mark.parametrize('value', [None, float('nan'), float('inf'), -.1, 1.1, True, '0.9'])
def test_incomplete_or_invalid_probability_is_not_used_to_guess(value):
    config = load_task_config(DEFAULT_DETECTOR_CONFIG)['identification']
    raw = {'answers': {config['question_id']: {'choice': 'red', 'probabilities': {
        'red': .1, 'blue': .1, 'yellow': value,
    }}}}
    with pytest.raises(ValueError, match='probability'):
        ident.parse_answer(raw, config)
    del raw['answers'][config['question_id']]['probabilities']['yellow']
    with pytest.raises(ValueError, match='probability'):
        ident.parse_answer(raw, config)


@pytest.mark.parametrize('old_status', ['success', 'failed'])
def test_cached_responses_are_reselected_by_probability_without_api(sample, monkeypatch, old_status):
    monkeypatch.setenv('DECISION_API_KEY', 'test-key')
    raw = {'answers': {'interaction_object': {'choice': 'red', 'probabilities': {
        'red': .0028, 'blue': .0012, 'yellow': .9960,
    }}}}
    ident.identify_bundle(sample['output'], data_root=sample['root'], client=Mock(return_value=raw))
    # Represent an older run that trusted choice, including previously invalid choice failures.
    bundle = read_bundle(sample)
    for event in bundle['events'][:4]:
        event['object'] = '红色方块' if old_status == 'success' else None
        result = event['identification']
        result.pop('selection_policy')
        result.update(status=old_status, candidate_key='red', error=None if old_status == 'success' else 'invalid choice')
        ident.write_json(sample['output'] / 'identification/events' / f"{ident.fingerprint(event['event_id'])}.json", event)
    save_bundle(sample['output'], bundle['events'], bundle['metadata'], bundle['reference_snapshot'])
    monkeypatch.delenv('DECISION_API_KEY')
    client = Mock(side_effect=AssertionError('cached probabilities need no new request'))
    counts = ident.identify_bundle(sample['output'], data_root=sample['root'], client=client)
    assert counts == dict(success=0, failed=0, skipped=1, reused=4)
    client.assert_not_called()
    bundle = read_bundle(sample)
    rows = pq.read_table(sample['output'] / 'events.parquet').to_pylist()
    for event, row in zip(bundle['events'][:4], rows[:4]):
        assert event['object'] == row['object'] == '黄色方块'
        assert event['identification']['candidate_key'] == row['identification']['candidate_key'] == 'yellow'
        assert event['identification']['selection_policy'] == row['identification']['selection_policy'] == jev.SELECTION_POLICY
        saved = json.loads((sample['output'] / 'identification/events' / f"{ident.fingerprint(event['event_id'])}.json").read_text())
        assert saved['object'] == '黄色方块'


def test_jev_cli_uses_same_probability_selection(monkeypatch, capsys):
    from scripts import jev_test
    monkeypatch.setenv('DECISION_API_KEY', 'test-key')
    monkeypatch.setattr(jev_test.sys, 'argv', ['jev_test.py', '--image', 'unused.png'])
    monkeypatch.setattr(jev_test, 'build_payload', Mock(return_value={}))
    monkeypatch.setattr(jev_test, 'ask_model', Mock(return_value={'answers': {jev_test.QUESTION_ID: {
        'choice': 'red', 'probabilities': {'red': .0028, 'blue': .0012, 'yellow': .9960},
    }}}))
    assert jev_test.main() == 0
    output = capsys.readouterr().out
    assert '模型选择（最高概率）: 黄色方块' in output
    assert '红色方块: 0.28%' in output and '黄色方块: 99.60%' in output
