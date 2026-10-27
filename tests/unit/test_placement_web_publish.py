"""Atomic Web registration checks; no ROS, USD or simulation."""
import json

import pytest

from src.placement_analysis.artifact_store import ArtifactStore
from src.placement_analysis.bundle import save_bundle_revision
from src.placement_analysis.web_publish import publish_bundle


def make_bundle(path, *, status='complete', events=None):
    path.mkdir(parents=True)
    (path / 'placement.json').write_text(json.dumps({
        'schema_version': 4, 'metadata': {'status': status}, 'robot_model': 'robot.glb', 'events': events or [],
    }))
    (path / 'robot.glb').write_bytes(b'companion robot')
    return path


def catalog(viewer):
    return json.loads((viewer / '.placement-catalog.json').read_text())


def test_publish_registers_sources_without_copies_and_preserves_archives(tmp_path):
    viewer = tmp_path / 'viewer'
    archive = viewer / 'public/data/old'
    archive.mkdir(parents=True)
    (archive / 'notes.txt').write_text('user archive')
    source = make_bundle(tmp_path / 'first')
    assert publish_bundle(source, task_name='按压按钮', viewer=viewer) == source / 'placement.json'
    first = catalog(viewer)
    old_revision = first['tasks']['按压按钮']['revision']
    new_source = make_bundle(tmp_path / 'second', events=[{'id': 1}])
    publish_bundle(new_source, task_name='按压按钮', viewer=viewer)
    latest = catalog(viewer)
    assert latest['tasks']['按压按钮']['revision'] != old_revision
    assert old_revision in latest['revisions']
    assert (archive / 'notes.txt').read_text() == 'user archive'
    assert not (viewer / 'public/data/按压按钮').exists()
    assert len(next(iter(latest['stores'].values()))['artifacts']) == 1  # shared GLB; private manifests use the transformed bundle route


def test_bundle_already_in_public_data_remains_untouched(tmp_path):
    viewer = tmp_path / 'viewer'
    source = make_bundle(viewer / 'public/data/demo')
    original = (source / 'placement.json').stat().st_ino
    assert publish_bundle(source, viewer=viewer) == source / 'placement.json'
    assert (source / 'placement.json').stat().st_ino == original


@pytest.mark.parametrize('failure', ['partial', 'missing_model', 'wrong_model'])
def test_failed_bundle_keeps_previous_catalog(tmp_path, failure):
    viewer = tmp_path / 'viewer'
    source = make_bundle(tmp_path / 'task')
    publish_bundle(source, viewer=viewer)
    before = (viewer / '.placement-catalog.json').read_bytes()
    bundle = json.loads((source / 'placement.json').read_text())
    if failure == 'partial':
        bundle['metadata']['status'] = 'partial'
    elif failure == 'missing_model':
        (source / 'robot.glb').unlink()
    else:
        bundle['robot_model'] = '../robot.glb'
    (source / 'placement.json').write_text(json.dumps(bundle))
    with pytest.raises((ValueError, FileNotFoundError)):
        publish_bundle(source, viewer=viewer)
    assert (viewer / '.placement-catalog.json').read_bytes() == before


@pytest.mark.parametrize('name', ['', '.', '..', '../escape', '/absolute', 'a/b', '.hidden', 'a\\b'])
def test_task_name_is_one_visible_directory(tmp_path, name):
    with pytest.raises(ValueError, match='directory name'):
        publish_bundle(make_bundle(tmp_path / 'task'), task_name=name, viewer=tmp_path / 'viewer')


def test_schema5_catalog_allows_only_referenced_viewer_artifacts(tmp_path):
    store = ArtifactStore(tmp_path / 'store')
    model = store.put_bytes(b'model', 'glb')
    private = store.put_bytes(b'private screenshot', 'png')
    snapshot = {'body_poses_world': [[0, 0, 0, 1, 0, 0, 0]]}
    output = tmp_path / 'out'
    save_bundle_revision(output, [{'event_id': 'a:0', 'episode_id': 'a', 'snapshot': snapshot,
        'identification': {'status': 'success', 'screenshot_ref': private}}],
        {'status': 'complete'}, snapshot, store=store, robot_model=model)
    viewer = tmp_path / 'viewer'
    publish_bundle(output, viewer=viewer)
    registered = next(iter(catalog(viewer)['stores'].values()))['artifacts']
    assert model['path'].split('/')[-1] in registered
    assert private['path'].split('/')[-1] not in registered
    assert len(registered) == 3  # model, deduplicated reference/event snapshot, shard
    original = (viewer / '.placement-catalog.json').read_bytes()
    store.resolve(model).write_bytes(b'bad')
    with pytest.raises(ValueError, match='digest'):
        publish_bundle(output, viewer=viewer)
    assert (viewer / '.placement-catalog.json').read_bytes() == original


def test_atomic_registry_replace_failure_preserves_prior_publication(tmp_path, monkeypatch):
    from src.placement_analysis import artifact_store
    viewer = tmp_path / 'viewer'
    source = make_bundle(tmp_path / 'source')
    publish_bundle(source, viewer=viewer)
    before = (viewer / '.placement-catalog.json').read_bytes()
    changed = json.loads((source / 'placement.json').read_text())
    changed['events'] = [{'id': 2}]
    (source / 'placement.json').write_text(json.dumps(changed))
    def fail(*args):
        raise OSError('interrupted registry update')
    monkeypatch.setattr(artifact_store.os, 'replace', fail)
    with pytest.raises(OSError):
        publish_bundle(source, viewer=viewer)
    assert (viewer / '.placement-catalog.json').read_bytes() == before


def test_evaluation_reference_is_validated_before_publication(tmp_path):
    store = ArtifactStore(tmp_path / 'store')
    model = store.put_bytes(b'model', 'glb')
    evaluation = store.put_json({'status': 'unknown', 'evidence': []})
    output, viewer = tmp_path / 'out', tmp_path / 'viewer'
    save_bundle_revision(output, [], {'status': 'complete', 'evaluation': evaluation},
                         {}, store=store, robot_model=model)
    publish_bundle(output, viewer=viewer)
    assert evaluation['path'].split('/')[-1] in next(iter(catalog(viewer)['stores'].values()))['artifacts']
    store.resolve(evaluation).write_text('{}')
    with pytest.raises(ValueError, match='digest'):
        publish_bundle(output, viewer=viewer)
