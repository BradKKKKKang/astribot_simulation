"""Completed zero-candidate episodes survive independent downstream stages."""
from src.placement_analysis.bundle import load_bundle, save_manifest_revision, store_for_manifest, write_episode_shard
from src.placement_analysis.identification import identify_bundle
from src.placement_analysis.pipeline_stages import evaluate_bundle
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
    report = evaluate_bundle(output)
    assert 'completed-empty' in [item['episode_id'] for item in report['episodes']]
    evaluated = load_bundle(output)
    assert [item['episode_id'] for item in evaluated['episodes']] == expected_order
    assert evaluated['episodes'][0] == empty
    assert evaluated['events'] == identified['events']
