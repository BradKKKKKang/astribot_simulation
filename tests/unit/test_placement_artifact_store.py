"""Immutable placement revisions, without ROS, USD or GPU dependencies."""
import json
from pathlib import Path

import pytest

from src.placement_analysis.artifact_store import ArtifactStore
from src.placement_analysis.bundle import load_bundle, save_bundle_revision, write_episode_shard, save_manifest_revision


def event(episode='one'):
    return {'episode_id': episode, 'event_id': f'{episode}:0', 'snapshot': {'body_poses_world': [[0, 0, 0, 1, 0, 0, 0]]}}


def test_content_dedup_and_immutable_revision(tmp_path):
    store = ArtifactStore(tmp_path / 'store')
    ref = store.put_json({'b': 2, 'a': 1})
    inode = store.resolve(ref).stat().st_ino
    assert store.put_json({'a': 1, 'b': 2}) == ref
    assert store.resolve(ref).stat().st_ino == inode
    output = tmp_path / 'analysis'
    item = event()
    save_bundle_revision(output, [item], {'status': 'complete'}, item['snapshot'], store=store)
    first = json.loads((output / 'placement.json').read_text())
    first_bytes = store.read_bytes(first['episodes'][0]['events'])
    item['object'] = 'red'
    save_bundle_revision(output, [item], {'status': 'complete'}, item['snapshot'], store=store)
    second = json.loads((output / 'placement.json').read_text())
    assert first['episodes'] != second['episodes']
    assert store.read_bytes(first['episodes'][0]['events']) == first_bytes
    assert load_bundle(output)['events'][0]['snapshot'] == item['snapshot']
    shard = store.read_json(second['episodes'][0]['events'])
    assert 'snapshot' not in shard['events'][0]
    assert shard['events'][0]['snapshot_ref'] == second['reference_snapshot']


def test_artifact_rejects_bad_paths_digest_and_symlinks(tmp_path):
    store = ArtifactStore(tmp_path / 'store')
    ref = store.put_bytes(b'one')
    for invalid in ({**ref, 'path': '../outside'}, {**ref, 'sha256': '0' * 64},
                    {**ref, 'path': '/' + ref['path']}, {**ref, 'size_bytes': 99}):
        with pytest.raises((ValueError, FileNotFoundError)):
            store.resolve(invalid)
    path = store.resolve(ref)
    path.write_bytes(b'bad')
    with pytest.raises(ValueError, match='digest'):
        store.read_bytes(ref)
    with pytest.raises(ValueError, match='digest'):
        store.put_bytes(b'one')
    path.unlink()
    target = tmp_path / 'outside'
    target.write_bytes(b'one')
    path.symlink_to(target)
    with pytest.raises(ValueError, match='symbolic'):
        store.resolve(ref)


def test_incremental_manifest_reuses_episode_refs_without_reading_old_shards(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / 'store')
    one, two = event('one'), event('two')
    shards = [write_episode_shard(store, 'one', [one])]
    reference = store.put_json(one['snapshot'])
    save_manifest_revision(tmp_path / 'out', {}, reference, shards, store=store)
    shards.append(write_episode_shard(store, 'two', [two]))
    monkeypatch.setattr(store, 'read_json', lambda *_: pytest.fail('old shards must not be reloaded'))
    save_manifest_revision(tmp_path / 'out', {}, reference, shards, store=store)
    assert len(json.loads((tmp_path / 'out/placement.json').read_text())['episodes']) == 2


@pytest.mark.parametrize('version', [1, 2, 3, 4])
def test_legacy_read_preserves_original(tmp_path, version):
    path = tmp_path / 'placement.json'
    value = {'schema_version': version, 'metadata': {}, 'events': [event()]}
    path.write_text(json.dumps(value))
    assert load_bundle(path) == value


def test_atomic_manifest_failure_preserves_previous(tmp_path, monkeypatch):
    from src.placement_analysis import artifact_store
    output = tmp_path / 'out'
    one = event()
    save_bundle_revision(output, [one], {}, one['snapshot'])
    before = (output / 'placement.json').read_bytes()
    def fail(*args):
        raise OSError('simulated replace failure')
    monkeypatch.setattr(artifact_store.os, 'replace', fail)
    with pytest.raises(OSError):
        save_bundle_revision(output, [], {}, one['snapshot'])
    assert (output / 'placement.json').read_bytes() == before


def test_geometry_cache_tracks_dependency_bytes_mapping_and_reuses_export(tmp_path, monkeypatch):
    pytest.importorskip('pxr')
    from pxr import Sdf, Usd, UsdGeom
    from src.placement_analysis import robot_export
    dependency = tmp_path / 'dependency.usda'
    layer = Sdf.Layer.CreateNew(str(dependency))
    layer.Save()
    usd = tmp_path / 'model.usda'
    stage = Usd.Stage.CreateNew(str(usd))
    stage.GetRootLayer().subLayerPaths = ['dependency.usda']
    UsdGeom.Xform.Define(stage, '/robot')
    stage.GetRootLayer().Save()
    first = robot_export.geometry_cache_key(usd, ['base'], root_path='/robot')
    assert first != robot_export.geometry_cache_key(usd, ['other'], root_path='/robot')
    assert first != robot_export.geometry_cache_key(usd, ['base'], root_path='/another')
    layer.comment = 'dependency change'
    layer.Save()
    assert first != robot_export.geometry_cache_key(usd, ['base'], root_path='/robot')
    calls = []
    def export(stage, root, names, destination):
        calls.append(names)
        destination.write_bytes(b'geometry')
    monkeypatch.setattr(robot_export, 'export_robot_glb', export)
    store = ArtifactStore(tmp_path / 'store')
    one = robot_export.cached_robot_geometry(stage, '/robot', ['base'], store, usd=usd)
    two = robot_export.cached_robot_geometry(stage, '/robot', ['base'], store, usd=usd)
    assert one == two
    assert len(calls) == 1


def test_convenience_writer_preserves_empty_shards_and_prior_episode_order(tmp_path):
    store = ArtifactStore(tmp_path / 'store')
    output = tmp_path / 'out'
    one = event('one')
    empty = write_episode_shard(store, 'empty', [])
    populated = write_episode_shard(store, 'one', [one])
    reference = store.put_json(one['snapshot'])
    save_manifest_revision(output, {}, reference, [empty, populated], store=store)
    # A downstream stage receives only event rows, while metadata may use another order.
    metadata = {'episodes': [{'episode_id': 'one'}, {'episode_id': 'empty'}, {'episode_id': 'new-empty'}]}
    save_bundle_revision(output, [one], metadata, one['snapshot'], store=store)
    bundle = load_bundle(output)
    assert [row['episode_id'] for row in bundle['episodes']] == ['empty', 'one', 'new-empty']
    assert bundle['episodes'][0] == empty
    assert bundle['episodes'][2]['event_count'] == 0
    assert len(bundle['events']) == 1


def test_convenience_writer_creates_explicit_zero_event_episodes_on_first_revision(tmp_path):
    save_bundle_revision(tmp_path / 'out', [], {'episodes': [{'episode_id': 'zero'}]}, {})
    bundle = load_bundle(tmp_path / 'out')
    assert [row['episode_id'] for row in bundle['episodes']] == ['zero']
    assert bundle['episodes'][0]['event_count'] == 0
    assert bundle['events'] == []


def test_geometry_cache_ignores_unused_shader_and_texture_assets(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Sdf, Usd, UsdGeom, UsdShade
    from src.placement_analysis.robot_export import cached_robot_geometry, geometry_cache_key

    usd = tmp_path / 'robot.usda'
    stage = Usd.Stage.CreateNew(str(usd))
    UsdGeom.SetStageUpAxis(stage, 'Z')
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.Xform.Define(stage, '/robot/base')
    mesh = UsdGeom.Mesh.Define(stage, '/robot/base/visuals/mesh')
    mesh.GetPointsAttr().Set([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.GetFaceVertexCountsAttr().Set([3])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    shader = UsdShade.Shader.Define(stage, '/robot/material/shader')
    shader.GetPrim().CreateAttribute('info:mdl:sourceAsset', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('OmniPBR.mdl'))
    shader.CreateInput('texture', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('unused.png'))
    stage.GetRootLayer().Save()
    first_key = geometry_cache_key(usd, ['base'], root_path='/robot')
    store = ArtifactStore(tmp_path / 'store')
    ref = cached_robot_geometry(stage, '/robot', ['base'], store, usd=usd)
    assert store.read_bytes(ref)[:4] == b'glTF'
    (tmp_path / 'unused.png').write_bytes(b'unused texture bytes')
    (tmp_path / 'OmniPBR.mdl').write_text('unused shader source')
    assert geometry_cache_key(usd, ['base'], root_path='/robot') == first_key
    assert cached_robot_geometry(stage, '/robot', ['base'], store, usd=usd) == ref


@pytest.mark.parametrize('dependency_kind', ['sublayer', 'reference', 'payload'])
def test_geometry_cache_rejects_missing_composition_dependencies(tmp_path, dependency_kind):
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom
    from src.placement_analysis.robot_export import geometry_cache_key

    usd = tmp_path / 'robot.usda'
    stage = Usd.Stage.CreateNew(str(usd))
    prim = UsdGeom.Xform.Define(stage, '/robot/base').GetPrim()
    if dependency_kind == 'sublayer':
        stage.GetRootLayer().subLayerPaths = ['missing-geometry.usda']
    elif dependency_kind == 'reference':
        prim.GetReferences().AddReference('missing-geometry.usda')
    else:
        prim.GetPayloads().AddPayload('missing-geometry.usda')
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match='composition dependencies'):
        geometry_cache_key(usd, ['base'], root_path='/robot')
