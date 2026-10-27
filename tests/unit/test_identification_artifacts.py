"""Shared-frame/request cache and independent evaluation stage integration."""
import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from tests.unit.test_placement_identification import sample, successful
from src.placement_analysis.artifact_store import ArtifactStore
from src.placement_analysis.bundle import load_bundle, save_bundle_revision
from src.placement_analysis.identification import identify_bundle
from src.placement_analysis.pipeline_stages import evaluate_bundle


@pytest.fixture
def artifact_sample(sample):
    old = json.loads((sample['output'] / 'placement.json').read_text())
    output = sample['output'].parent / 'modern'
    old['metadata']['type'] = 'press'
    save_bundle_revision(output, old['events'], old['metadata'], old['reference_snapshot'],
                         robot_model=sample['output'] / 'robot.glb')
    return {**sample, 'output': output}


def configuration(sample, **changes):
    sections = copy.deepcopy(sample['settings']['_resolved_sections'])
    for key, value in changes.items():
        sections[key] = value
    path = sample['output'].parent / 'experiment.json'
    path.write_text(json.dumps({'schema_version': 1, **sections}))
    return path


def test_frames_and_requests_shared_timeout_decision_independent(artifact_sample, monkeypatch):
    sample = artifact_sample
    monkeypatch.setenv('DECISION_API_KEY', 'unit-test-secret')
    client = Mock(side_effect=successful)
    before = copy.deepcopy(load_bundle(sample['output'])['metadata']['task_config'])
    first = identify_bundle(sample['output'], data_root=sample['root'], client=client, workers=2)
    assert first['failed'] == 0 and first['success'] == 3 and first['reused'] == 1
    assert client.call_count == 3  # two events identify the same source image/question
    section = copy.deepcopy(sample['settings']['_resolved_sections']['jev'])
    section['transport']['timeout_s'] = 2
    section['decision'] = {'policy': 'max_probability_v1', 'min_probability': .95}
    config = configuration(sample, jev=section)
    client.reset_mock()
    second = identify_bundle(sample['output'], task_config=config, data_root=sample['root'], client=client)
    assert second['unknown'] == 4 and second['failed'] == 0
    client.assert_not_called()
    updated = load_bundle(sample['output'])
    assert updated['metadata']['task_config'] == before
    assert updated['metadata']['identification_config']['transport']['timeout_s'] == 2
    assert all(event['object'] is None for event in updated['events'])
    assert updated['metadata']['stages']['identification'] == 'complete'


def test_question_change_reuses_frame_and_retries_only_requests(artifact_sample, monkeypatch):
    sample = artifact_sample
    monkeypatch.setenv('DECISION_API_KEY', 'unit-test-secret')
    identify_bundle(sample['output'], data_root=sample['root'], client=successful)
    before = load_bundle(sample['output'])
    refs = [e['identification'].get('screenshot_ref') for e in before['events']]
    section = copy.deepcopy(sample['settings']['_resolved_sections']['jev'])
    section['request']['question'] += '现在再判断。'
    client = Mock(side_effect=successful)
    identify_bundle(sample['output'], task_config=configuration(sample, jev=section),
                    data_root=sample['root'], client=client)
    assert client.call_count == 3
    assert [e['identification'].get('screenshot_ref') for e in load_bundle(sample['output'])['events']] == refs


def test_dry_run_writes_nothing_and_candidate_type_cannot_change(artifact_sample):
    sample = artifact_sample
    base = sample['output'].parent
    before = {str(p): p.read_bytes() for p in base.rglob('*') if p.is_file()}
    identify_bundle(sample['output'], data_root=sample['root'], dry_run=True,
                    client=Mock(side_effect=AssertionError('must not request')))
    assert before == {str(p): p.read_bytes() for p in base.rglob('*') if p.is_file()}
    section = copy.deepcopy(sample['settings']['_resolved_sections']['candidates'])
    section['type'] = 'gripper'
    with pytest.raises(ValueError, match='candidate type'):
        identify_bundle(sample['output'], task_config=configuration(sample, candidates=section))


def test_evaluation_runs_without_model_and_preserves_candidate_provenance(artifact_sample):
    sample = artifact_sample
    sections = copy.deepcopy(sample['settings']['_resolved_sections'])
    sections['task']['events'] = {'press': {'event_type': 'press'}}
    sections['task']['stages'] = {}
    sections['success'] = {'mode': 'all', 'predicates': [
        {'id': 'enough_candidates', 'kind': 'event_count', 'event': 'press', 'min_count': 1}]}
    path = sample['output'].parent / 'evaluation-config.json'
    path.write_text(json.dumps({'schema_version': 1, **sections}))
    before = load_bundle(sample['output'])
    result = evaluate_bundle(sample['output'], task_config=path)
    assert result['summary'] == {'passed': 1, 'failed': 0, 'unknown': 0}
    after = load_bundle(sample['output'])
    assert after['events'] == before['events']
    assert after['metadata']['task_config'] == before['metadata']['task_config']
    assert after['metadata']['evaluation_summary'] == result['summary']


def test_failure_is_visible_and_requests_can_resume(artifact_sample, monkeypatch):
    sample = artifact_sample
    monkeypatch.setenv('DECISION_API_KEY', 'unit-test-secret')
    client = Mock(side_effect=RuntimeError('unit-test-secret remote failed'))
    result = identify_bundle(sample['output'], data_root=sample['root'], client=client)
    assert result['failed'] == 4
    bundle = load_bundle(sample['output'])
    assert bundle['metadata']['stages']['identification'] == 'partial'
    assert 'unit-test-secret' not in json.dumps(bundle)
    result = identify_bundle(sample['output'], data_root=sample['root'], client=successful)
    assert result['failed'] == 0


def test_invalid_response_retains_audit_context_and_retries(artifact_sample, monkeypatch):
    from src.placement_analysis.bundle import store_for_manifest
    sample = artifact_sample
    monkeypatch.setenv('DECISION_API_KEY', 'unit-test-secret')
    invalid = {'answers': {'interaction_object': {'probabilities': {'red': -1}}}}
    assert identify_bundle(sample['output'], data_root=sample['root'],
                           client=lambda *args: invalid)['failed'] == 4
    bundle = load_bundle(sample['output'])
    store = store_for_manifest(sample['output'] / 'placement.json', bundle)
    for event in bundle['events'][:4]:
        result = event['identification']
        assert result['source']['video_frame_index'] >= 0
        assert store.read_json(result['raw_response_ref']) == invalid
        assert store.resolve(result['screenshot_ref']).is_file()
    client = Mock(side_effect=successful)
    assert identify_bundle(sample['output'], data_root=sample['root'], client=client)['failed'] == 0
    assert client.call_count == 3


def test_decision_only_uses_archived_evidence_and_invalidates_evaluation(artifact_sample, monkeypatch):
    sample = artifact_sample
    monkeypatch.setenv('DECISION_API_KEY', 'unit-test-secret')
    identify_bundle(sample['output'], data_root=sample['root'], client=successful)
    evaluate_bundle(sample['output'])
    assert load_bundle(sample['output'])['metadata']['stages']['evaluation'] == 'complete'
    sample['video'].unlink()
    sample['source'].unlink()
    section = copy.deepcopy(sample['settings']['_resolved_sections']['jev'])
    section['decision'] = {'policy': 'max_probability_v1', 'min_probability': .95}
    client = Mock(side_effect=AssertionError('decision-only must not call model'))
    counts = identify_bundle(sample['output'], task_config=configuration(sample, jev=section), client=client)
    assert counts['unknown'] == 4 and counts['failed'] == 0
    client.assert_not_called()
    metadata = load_bundle(sample['output'])['metadata']
    assert metadata['stages']['evaluation'] == 'not_requested'
    assert 'evaluation' not in metadata and 'evaluation_summary' not in metadata


def test_explicit_contact_evidence_is_consumed_and_audited(artifact_sample):
    sample = artifact_sample
    config = configuration(sample, success={'predicates': [
        {'id': 'button_contact', 'kind': 'contact', 'evidence': 'button_contact'}]})
    assert evaluate_bundle(sample['output'], task_config=config)['summary']['unknown'] == 1
    observations = {'episodes': {'episode_000000': {'button_contact': {
        'value': True, 'kind': 'contact', 'source': 'contact-sensor-recording:frame-12'}}}}
    path = sample['output'].parent / 'observations.json'
    path.write_text(json.dumps(observations))
    result = evaluate_bundle(sample['output'], task_config=config, observations=path)
    assert result['summary']['passed'] == 1 and result['observations'] == observations
    assert evaluate_bundle(sample['output'], task_config=config)['summary']['passed'] == 1
    with pytest.raises(ValueError, match='unknown episode'):
        evaluate_bundle(sample['output'], task_config=config, observations={'episodes': {'other': {}}})


def test_pipeline_writers_share_exclusive_lock(artifact_sample):
    import fcntl
    sample = artifact_sample
    with (sample['output'] / '.pipeline.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='pipeline stage'):
            identify_bundle(sample['output'])
        with pytest.raises(RuntimeError, match='pipeline stage'):
            evaluate_bundle(sample['output'])


def test_explicit_source_override_revalidates_instead_of_using_archived_frame(artifact_sample, monkeypatch):
    sample = artifact_sample
    monkeypatch.setenv('DECISION_API_KEY', 'unit-test-secret')
    identify_bundle(sample['output'], data_root=sample['root'], client=successful)
    sample['video'].unlink()
    client = Mock(side_effect=AssertionError('missing source must not request'))
    counts = identify_bundle(sample['output'], data_root=sample['root'], client=client)
    assert counts['failed'] == 4
    client.assert_not_called()


def test_evidence_duplicate_keys_rejected(artifact_sample):
    sample = artifact_sample
    path = sample['output'].parent / 'ambiguous-evidence.json'
    path.write_text('{"episodes":{},"episodes":{}}')
    with pytest.raises(ValueError, match='duplicate'):
        evaluate_bundle(sample['output'], observations=path)


def test_manifest_only_empty_episode_is_evaluated(artifact_sample):
    from src.placement_analysis.bundle import store_for_manifest, write_episode_shard, save_manifest_revision
    sample = artifact_sample
    path = sample['output'] / 'placement.json'
    manifest = json.loads(path.read_text())
    store = store_for_manifest(path, manifest)
    empty = write_episode_shard(store, 'empty', [])
    save_manifest_revision(sample['output'], manifest['metadata'], manifest['reference_snapshot'],
                           [*manifest['episodes'], empty], manifest['robot_model'], store=store)
    report = evaluate_bundle(sample['output'])
    assert [row['episode_id'] for row in report['episodes']] == ['episode_000000', 'empty']
