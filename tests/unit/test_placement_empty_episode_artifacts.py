"""Completed zero-candidate episodes survive independent downstream stages."""
from src.placement_analysis.bundle import load_bundle, save_manifest_revision, store_for_manifest, write_episode_shard
from src.placement_analysis.identification import identify_bundle
from src.evaluation.placement import evaluate_bundle
from tests.unit.test_identification_artifacts import artifact_sample, sample, successful  # noqa: F401


def test_zero_event_episode_survives_identification_and_evaluation(artifact_sample, monkeypatch):
    fixture = artifact_sample
    output = fixture['output']
    bundle = load_bundle(output)
    store = store_for_manifest(output / 'placement.json', bundle)
    empty = write_episode_shard(store, 'completed-empty', [])
    original_shards = bundle['episodes']
    metadata = bundle['metadata']
    metadata['episodes'].append({'episode_id': 'completed-empty'})
    save_manifest_revision(output, metadata, store.put_json(bundle['reference_snapshot']),
                           [empty, *original_shards], bundle['robot_model'],
                           display=bundle['display'], store=store)
    expected_order = ['completed-empty', *[item['episode_id'] for item in original_shards]]
    monkeypatch.setenv('DECISION_API_KEY', 'unit-test-secret')
    result = identify_bundle(output, data_root=fixture['root'], client=successful)
    assert result['failed'] == 0
    identified = load_bundle(output)
    assert [item['episode_id'] for item in identified['episodes']] == expected_order
    assert identified['episodes'][0] == empty
    report = evaluate_bundle(output, task_config=fixture['evaluation_config'])
    assert 'completed-empty' in [item['episode_id'] for item in report['episodes']]
    evaluated = load_bundle(output)
    assert [item['episode_id'] for item in evaluated['episodes']] == expected_order
    assert evaluated['episodes'][0] == empty
    assert evaluated['events'] == identified['events']


def test_offline_v3_independent_success_consumes_explicit_observation(artifact_sample, tmp_path):
    import json
    from src.simulation_config.experiment_config import resolve_experiment
    fixture = artifact_sample
    resolved = resolve_experiment(fixture['evaluation_config'])
    value = {'schema_version': 3, 'kind': 'evaluation', **resolved['sections']}
    value['success'] = {'predicates': [{'id': 'contact', 'kind': 'contact', 'evidence': 'contact'}]}
    path = tmp_path / 'evaluation.json'; path.write_text(json.dumps(value))
    missing = evaluate_bundle(fixture['output'], task_config=path, dry_run=True)
    assert missing['summary']['unknown'] == len(missing['episodes'])
    evidence = {'episodes': {episode['episode_id']: {
        'contact': {'kind': 'contact', 'source': 'recorded_sensor', 'value': True}}
        for episode in missing['episodes']}}
    observed = evaluate_bundle(fixture['output'], task_config=path, dry_run=True, observations=evidence)
    assert observed['summary']['passed'] == len(missing['episodes'])
    assert observed['config']['success'] == value['success']
    assert 'success' not in observed['config']['task']
