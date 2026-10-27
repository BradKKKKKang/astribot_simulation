"""Synthetic capture -> real CPU USD FK -> real GLB/schema-5 bundle acceptance.

Attestation envelopes deliberately carry ``mock=True`` and an evidence-origin
label. Their poses come from real complete-geometry grounding, but no ROS/SDK or
physics feedback is produced by these tests.
"""
import builtins
import copy
import json
from pathlib import Path
import struct

import numpy as np
import pytest

pytest.importorskip('pxr', reason='CPU USD bindings required for grounding and export')

from src.placement_analysis.bundle import load_bundle, store_for_manifest
from src.placement_analysis.extraction import extract_events, prepare_extraction
from src.placement_analysis.extraction_worker import StageCache, capture_digests, extract_episode
from src.placement_analysis.robot_export import DEFAULT_USD, geometry_cache_key
from src.simulation_config.experiment_config import resolve_experiment
from src.simulation_config.poses import compose_pose, relative_pose
from src.simulation_config.robot_placement import identity, pose_matrix, resolve_robot_placement
from src.simulation_config.robot_profile import profile_for_settings
from tests.unit.test_placement_extraction_stages import write_capture

ROOT = Path(__file__).resolve().parents[2]
DECLARATION = {'frame':'world','xy':[1.25,-.75],'yaw':.65,'grounding':'geometry_min'}
MOCK_ORIGIN = 'synthetic CPU fixture; not ROS or runtime pose measurement'


def configuration(tmp_path, *, declaration=None):
    sections = resolve_experiment(ROOT/'config/placement_distribution/penholder.json')['sections']
    sections.pop('robot_placement',None)
    if declaration is not None:
        sections['robot_placement'] = copy.deepcopy(declaration)
    path = tmp_path / ('grounded.json' if declaration is not None else 'legacy.json')
    path.write_text(json.dumps({'schema_version':3 if declaration is not None else 2,
                               'kind':'placement_test',**sections}))
    resolved = resolve_experiment(path)
    profile = profile_for_settings(resolved['settings'])
    stage, root, names, model = prepare_extraction(profile.model_path,settings=resolved['settings'])
    return resolved, profile, stage, root, names, model


def synthetic_capture(path, profile, placement=None, *, measured_offset=2e-6):
    path.parent.mkdir(parents=True,exist_ok=True)
    write_capture(path)
    metadata_path = path/'run_metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['mock'] = True
    metadata['evidence_origin'] = MOCK_ORIGIN
    if placement is not None:
        epoch, wall = metadata['epoch_monotonic_ns'],metadata['epoch_system_ns']
        metadata['dataset_window_monotonic_ns'] = [epoch,epoch+790_001_000]
        actual = list(placement['base_pose_world'])
        actual[0] += measured_offset
        actual[1] -= measured_offset
        sample = {'source':'isaaclab.body_pose_w','mock':True,'evidence_origin':MOCK_ORIGIN,
                  'declaration':placement['declaration'],'resolved':placement,
                  'base_body':profile.base_body,'actual_base_pose_world':actual,
                  'robot_control_sha256':identity(profile.control_provenance())}
        metadata['robot_placement_attestations'] = [
            {**sample,'observed_system_ns':wall+offset}
            for offset in (-10_000_000,400_000_000,820_000_000)]
    metadata_path.write_text(json.dumps(metadata))
    return path,metadata


def test_v3_mock_attestation_real_fk_export_and_bundle_preserve_coordinate_provenance(tmp_path,monkeypatch):
    original_import = builtins.__import__
    def no_runtime(name,*args,**kwargs):
        if name.split('.')[0] in {'isaaclab','isaacsim','omni','rclpy','rospy','astribot_sdk'}:
            raise AssertionError('CPU extraction imported runtime: '+name)
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',no_runtime)
    grounded,profile,_,_,names,model = configuration(tmp_path,declaration=DECLARATION)
    legacy,_,_,_,legacy_names,legacy_model = configuration(tmp_path)
    # Independent complete-geometry resolution supplies the synthetic envelope.
    placement = resolve_robot_placement(profile,DECLARATION)
    assert placement == model.placement
    assert placement['geometry_min_z'] == 0 and placement['geometry_count'] > 0
    assert names == legacy_names
    capture,capture_metadata = synthetic_capture(tmp_path/'grounded-capture/episode0',profile,placement)
    legacy_capture,_ = synthetic_capture(tmp_path/'legacy-capture/episode0',profile)
    source_digests = capture_digests(capture)
    store = tmp_path/'artifacts'
    # Real geometry export, immutable shards and manifest publication; no exporter mocks.
    for source,config,name in ((legacy_capture,legacy,'legacy-bundle'),(capture,grounded,'grounded-bundle')):
        extract_events([source],tmp_path/name,resolved=config,store_root=store,no_progress=True)
    before,after = (load_bundle(tmp_path/name) for name in ('legacy-bundle','grounded-bundle'))
    assert len(before['events']) == len(after['events']) == 2
    actual = capture_metadata['robot_placement_attestations'][0]['actual_base_pose_world']
    for old,new in zip(before['events'],after['events']):
        for field in ('event_id','event_type','side','frame_index','analysis_selected'):
            assert old[field] == new[field]
        np.testing.assert_allclose(new['tcp_pose_base'],old['tcp_pose_base'],atol=1e-7)
        np.testing.assert_allclose(new['base_pose_world'],actual,atol=1e-10)
        expected_world = compose_pose(actual,old['tcp_pose_base'])
        np.testing.assert_allclose(pose_matrix(new['tcp_pose_world']),pose_matrix(expected_world),atol=1e-7)
        np.testing.assert_allclose(new['snapshot']['joint_positions_rad'],old['snapshot']['joint_positions_rad'])
        for old_body,new_body in zip(old['snapshot']['body_poses_world'],new['snapshot']['body_poses_world']):
            expected_body = compose_pose(actual,relative_pose(old['base_pose_world'],old_body))
            np.testing.assert_allclose(pose_matrix(new_body),pose_matrix(expected_body),atol=1e-7)
    assert after['metadata']['tcp'] == before['metadata']['tcp']
    assert after['metadata']['status'] == 'complete'
    assert after['metadata']['robot_placement'] == placement
    assert after['metadata']['robot_kinematics_sha256'] != before['metadata']['robot_kinematics_sha256']
    episode = after['metadata']['episodes'][0]
    assert episode['capture_sha256'] == source_digests == capture_digests(capture)
    assert episode['source'] == capture_metadata['input_parquet']
    assert episode['sha256'] == capture_metadata['input_sha256']
    assert episode['robot_placement_actual'] == capture_metadata['robot_placement_attestations'][0]
    assert episode['robot_placement_actual']['mock'] is True
    assert episode['robot_placement_actual']['evidence_origin'] == MOCK_ORIGIN
    # Measured numerical offset is preserved rather than replaced by the plan.
    assert episode['robot_kinematics_sha256'] != after['metadata']['robot_kinematics_sha256']
    np.testing.assert_allclose(after['reference_snapshot']['body_poses_world'][names.index(profile.base_body)],actual)
    assert 'robot_placement' not in before['metadata']
    assert 'robot_placement_actual' not in before['metadata']['episodes'][0]
    # Export stays body-local; world placement lives in the referenced snapshots.
    assert before['robot_model'] == after['robot_model']
    manifest_path = tmp_path/'grounded-bundle/placement.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['schema_version'] == 5
    artifact_store = store_for_manifest(manifest_path,manifest)
    glb = artifact_store.read_bytes(manifest['robot_model'])
    magic,version,length = struct.unpack_from('<4sII',glb)
    assert (magic,version,length) == (b'glTF',2,len(glb))
    json_size = struct.unpack_from('<I',glb,12)[0]
    document = json.loads(glb[20:20+json_size])
    assert [node['name'] for node in document['nodes']] == names
    assert document['meshes'] and document['accessors']
    shard = artifact_store.read_json(manifest['episodes'][0]['events'])
    assert shard['events'][0]['tcp_pose_world'] == after['events'][0]['tcp_pose_world']
    assert artifact_store.read_json(shard['events'][0]['snapshot_ref']) == after['events'][0]['snapshot']


def test_legacy_capture_cannot_acquire_grounded_coordinates_or_lose_evidence(tmp_path):
    grounded,profile,_,_,_,model = configuration(tmp_path,declaration=DECLARATION)
    legacy,_,_,_,_,legacy_model = configuration(tmp_path)
    capture,_ = synthetic_capture(tmp_path/'legacy-capture/episode0',profile)
    before = capture_digests(capture)
    with pytest.raises(ValueError,match='legacy coordinates cannot be reinterpreted'):
        extract_events([capture],tmp_path/'forbidden',resolved=grounded,dry_run=True)
    assert capture_digests(capture) == before
    assert not (tmp_path/'forbidden').exists()
    placed,metadata = synthetic_capture(tmp_path/'grounded-capture/episode0',profile,model.placement)
    with pytest.raises(ValueError,match='original explicit robot_placement'):
        extract_episode(placed,legacy['settings'],legacy_model,profile,StageCache(tmp_path/'cache'))
    metadata['robot_placement_attestations'].pop()
    (placed/'run_metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='continuously cover the capture window'):
        extract_episode(placed,grounded['settings'],model,profile,StageCache(tmp_path/'cache'))


def test_actual_placement_changes_fk_cache_and_preserves_body_local_geometry_cache(tmp_path):
    grounded,profile,stage,root,names,model = configuration(tmp_path,declaration=DECLARATION)
    cache = StageCache(tmp_path/'cache')
    first,_ = synthetic_capture(tmp_path/'first/episode0',profile,model.placement,measured_offset=2e-6)
    first_result = extract_episode(first,grounded['settings'],model,profile,cache)
    repeated = extract_episode(first,grounded['settings'],model,profile,cache)
    assert repeated['cache_hits'] == {'synchronized':True,'endpoint_fk':True}
    second,_ = synthetic_capture(tmp_path/'second/episode0',profile,model.placement,measured_offset=5e-6)
    second_result = extract_episode(second,grounded['settings'],model,profile,cache)
    assert second_result['cache_hits'] == {'synchronized':False,'endpoint_fk':False}
    assert first_result['episode']['stage_keys'] != second_result['episode']['stage_keys']
    assert first_result['episode']['robot_kinematics_sha256'] != second_result['episode']['robot_kinematics_sha256']
    for a,b in zip(first_result['events'],second_result['events']):
        np.testing.assert_allclose(a['tcp_pose_base'],b['tcp_pose_base'],atol=1e-10)
        np.testing.assert_allclose(np.array(b['tcp_pose_world'][:3])-a['tcp_pose_world'][:3],[3e-6,-3e-6,0],atol=1e-10)
    declaration = {**DECLARATION,'xy':[-.4,.8],'yaw':-.35}
    other,_,other_stage,other_root,other_names,other_model = configuration(tmp_path,declaration=declaration)
    third,_ = synthetic_capture(tmp_path/'third/episode0',profile,other_model.placement)
    third_result = extract_episode(third,other['settings'],other_model,profile,cache)
    assert third_result['episode']['stage_keys'] != first_result['episode']['stage_keys']
    assert grounded['fingerprints']['extraction'] != other['fingerprints']['extraction']
    assert model.fingerprint() != other_model.fingerprint()
    assert geometry_cache_key(DEFAULT_USD,names,root_path=root.GetPath(),stage=stage) == geometry_cache_key(
        DEFAULT_USD,other_names,root_path=other_root.GetPath(),stage=other_stage)
