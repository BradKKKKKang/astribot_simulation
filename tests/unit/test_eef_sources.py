"""Sample-source routing and episode placement without ROS or simulator startup."""
import copy
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.spatial.transform import Rotation

from src.eef_review.alignment import place, pose_matrix
from src.eef_review.catalog import admission
from src.eef_review.common import atomic_json, digest
from src.eef_review.formal_index import build
from src.eef_review.mapping import describe
from src.eef_review.sources import CombinedReader, SampleReader
from src.eef_review.trajectory import adapt
from tests.unit.test_eef_formal import formal_root


def sample_package(tmp_path, dataset='realomin_10kh', no_pose=False):
    root = tmp_path / dataset
    group_id = dataset + '/configuration'
    if dataset == 'HABIT':
        contract = {'published_configuration': {'coordinates': {'common_base_known': False, 'position_unit': 'UNKNOWN'},
                    'pose_encoding': 'xyz_rotation_first_two_rows'}, 'source': 'dataset', 'robot': 'human'}
    else:
        contract = {'schema': 'umi_aligned_sample_mapping_v1', 'eef_sides': ['left', 'right'], 'source_policies': {}}
    group = {'dataset': dataset, 'directory': 'configuration', 'details': {}, 'contract': contract, 'episodes': 1}
    policy = {'mode': 'chassis' if dataset == 'HABIT' else 'episode_first_state',
              'mapping_sha256': digest(contract), 'position_unit_assumption': 'meter', 'evidence': 'explicit user choice'}
    job = {'dataset': dataset, 'root': 'task', 'group_directory': 'configuration',
           'output_root': 'configuration/task', 'selected_episodes': [7]}
    atomic_json(root / 'SAMPLE_MANIFEST.json', {'id': 'sample-v1', 'groups': {group_id: group}, 'jobs': [job]})
    atomic_json(root / 'MAPPING_GROUPS.json', {group_id: group})
    directory = root / job['output_root']
    atomic_json(directory / 'MAPPING.json', {'contract': contract})
    atomic_json(directory / 'SOURCE_INFO.json', {'robot_type': 'human',
                'features': {k: {'shape': [34]} for k in ('observation.state', 'action', 'eef_mask')}})
    valid = {'left': 0 if no_pose else 2, 'right': 0 if no_pose else 2}
    atomic_json(directory / 'ROOT_COMPLETE.json', {'status': 'passed', 'manifest_id': 'sample-v1',
                'selected_episodes': [7], 'dataset': dataset, 'root': 'task',
                'episodes': [{'episode': 7, 'frames': 2, 'valid_pose_frames': valid}]})
    atomic_json(directory / 'episode_000007/source.json', {'source_dataset': dataset, 'source_root': 'task',
                'source_episode_index': 7, 'frames': 2, 'fps': 10, 'eef_valid_pose_frames': valid,
                'sample_time_column': 'sample_timestamp', 'videos': []})
    pq.write_table(pa.table({'timestamp': [0., .1]}), directory / 'episode_000007/eef.parquet')
    # A retained historical package must not enter the current directory.
    atomic_json(directory / 'episode_000999/source.json', {'historical': True})
    return {'id': dataset, 'label': dataset, 'root': str(root)}, {group_id: policy}, directory, group


def trajectory():
    state = np.zeros((3, 34)); mask = np.zeros((3, 34), dtype=bool)
    for start, angle in ((9, .9), (19, -1.2)):
        for i in range(3):
            state[i, start:start + 3] = [.1 * i, -.02 * i, .03 * i]
            state[i, start + 3:start + 9] = Rotation.from_euler('xyz', [.1 * i, .05 * i, angle]).as_matrix()[:2].ravel()
        mask[:, start:start + 10] = True
    action = state.copy(); action[:, 9] += .04; action[:, 19] -= .02
    return adapt({'timestamp': [2., 2.1, 2.2], 'observation.state': state, 'action': action, 'eef_mask': mask})


def test_independent_arm_placement_preserves_state_action_offsets_and_source_values():
    source = trajectory(); original = copy.deepcopy(source)
    anchors = {'astribot_arm_left': [.4, .25, .9, *Rotation.from_euler('xyz', [.2, -.4, .5]).as_quat()],
               'astribot_arm_right': [.4, -.25, .9, *Rotation.from_euler('xyz', [-.1, .3, -.6]).as_quat()]}
    result = place(source, {'mode': 'episode_first_state'}, anchors)
    assert source == original
    for side in ('left', 'right'):
        A0 = pose_matrix(anchors['astribot_arm_' + side]); S0 = pose_matrix(source['frames'][0]['state']['targets'][side])
        np.testing.assert_allclose(pose_matrix(result['frames'][0]['state']['targets'][side]), A0, atol=1e-12)
        for i in range(3):
            for channel in ('state', 'action'):
                actual = result['frames'][i][channel]
                expected = source['frames'][i][channel]
                np.testing.assert_allclose(np.linalg.inv(A0) @ pose_matrix(actual['targets'][side]),
                                           np.linalg.inv(S0) @ pose_matrix(expected['targets'][side]), atol=1e-12)
                assert actual['raw'] == expected['raw']
                assert actual['source_targets'] == expected['targets']
        assert result['frames'][0]['action']['targets'][side][:3] != anchors['astribot_arm_' + side][:3]
    assert result['alignment']['per_arm']['left']['data_to_chassis'] != result['alignment']['per_arm']['right']['data_to_chassis']
    assert place(source)['frames'] == original['frames']


def test_episode_placement_never_substitutes_a_later_frame_for_missing_first_state():
    source = trajectory(); source['frames'][0]['state']['targets']['left'] = None
    with pytest.raises(ValueError, match='missing first State EEF pose: left'):
        place(source, {'mode': 'episode_first_state'}, {'astribot_arm_left': [0, 0, 0, 0, 0, 0, 1]})


@pytest.mark.parametrize('dataset', ['HABIT', 'realomin_10kh'])
def test_sample_contract_policy_and_selected_episode_integrity(tmp_path, dataset):
    source, policies, directory, group = sample_package(tmp_path, dataset)
    reader = SampleReader(source, {'alignment_policies': policies}, tmp_path / 'cache')
    try:
        page = reader.catalog.query({})
        assert page['total'] == 1 and page['episodes'][0]['episode'] == 7
        row = page['episodes'][0]
        assert row['review']['status'] == 'admitted'
        assert row['review']['unit_evidence']['assumed']
        assert row['mapping_version'] == digest(group['contract'])
        assert describe(group)['profile']
        assert reader.metadata(row['id'])['source']['frames'] == 2
        changed = json.loads((directory / 'ROOT_COMPLETE.json').read_text()); changed['status'] = 'changed'
        atomic_json(directory / 'ROOT_COMPLETE.json', changed)
        with pytest.raises(ValueError, match='completion changed'):
            reader.metadata(row['id'])
    finally:
        reader.preview.executor.shutdown()
    policy = next(iter(policies.values())); policy['mapping_sha256'] = 'stale'
    with pytest.raises(ValueError, match='another mapping version'):
        admission(group, policy=policy)


def test_no_pose_episode_remains_visible_but_not_computable(tmp_path):
    source, policies, _, _ = sample_package(tmp_path, no_pose=True)
    reader = SampleReader(source, {'alignment_policies': policies}, tmp_path / 'cache')
    try:
        assert reader.catalog.query({'review': 'no_eef'})['total'] == 1
        assert reader.catalog.query({'review': 'admitted'})['total'] == 0
        assert reader.catalog.sample() == []
    finally:
        reader.preview.executor.shutdown()


def test_unknown_robot_type_is_nullable_and_does_not_confirm_units():
    group = {'dataset': 'FastUMI_aligned', 'details': {}, 'contract': {
        'schema': 'umi_aligned_sample_mapping_v1', 'eef_sides': ['left', 'right'], 'source_policies': {}}}
    assert admission(group, robot_type=None)['status'] == 'pending_units'


def test_mixed_pagination_routing_filters_and_formal_identity(formal_root, tmp_path):
    base, sample, _, index, _ = formal_root
    build(base, sample, index, {}, workers=1)
    source, policies, _, _ = sample_package(tmp_path)
    reader = CombinedReader({'sample_sources': [source], 'alignment_policies': policies,
                             'remote_root': str(base), 'remote_index': str(index)}, tmp_path / 'media')
    try:
        page = reader.query({'limit': 2})
        assert page['total'] == 3 and len(page['episodes']) == 2
        assert [e['source_kind'] for e in page['episodes']] == ['sample', 'formal']
        assert len(reader.query({'offset': 1, 'limit': 2})['episodes']) == 2
        assert reader.query({'offset': 99})['episodes'] == []
        assert reader.query({'source': source['id']})['total'] == 1
        assert reader.query({'source': 'formal'})['total'] == 2
        assert reader.query({'source': 'missing'})['total'] == 0
        assert reader.query({'q': 'realomin'})['total'] == 1
        assert reader.query({'ids': []})['total'] == 0
        key = page['episodes'][0]['id']
        assert reader.query({'exclude_ids': [key]})['total'] == 2
        assert reader.query({'ids': [key]})['episodes'][0]['id'] == key
        assert reader.metadata(key)['catalog_entry']['source_version'] == reader.version
        formal_key = page['episodes'][1]['id']
        assert formal_key == digest(['InternData-A1', 'task', 0])[:24]
        assert reader.metadata(formal_key)['catalog_entry']['source_id'] == 'formal'
        with reader.data_file(key) as path:
            assert pq.read_table(path)['timestamp'].to_pylist() == [0., .1]
    finally:
        reader.preview.executor.shutdown()
