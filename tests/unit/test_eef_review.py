"""Contract, transport and publication tests without ROS or Isaac Sim."""
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.eef_review.catalog import Catalog, admission
from src.eef_review.common import atomic_json, confined, digest
from src.eef_review.media import byte_range, InvalidRange, send_file
from src.eef_review.trajectory import adapt, decode_pose
from src.eef_review.preview import PreviewCache


def group(profile='sim_lift2', dataset='InternData-A1', unit='meter'):
    return {'dataset': dataset, 'details': {'profile': profile},
            'contract': {'action_kind': 'absolute_pose', 'coordinates': {'position_unit': unit, 'reference_frame': 'robot'}}}


@pytest.mark.parametrize('profile,dataset', [('franka', 'RoboMIND'), ('sim_franka', 'InternData-A1'),
    ('fr3', 'RoboMIND'), ('droid_root', 'MolmoAct2-DROID-Dataset'), ('simulated_franka', 'InternData-M1')])
def test_reviewed_franka_profiles_require_metric_evidence(profile, dataset):
    item = group(profile, dataset)
    assert admission(item, 'panda')['status'] == 'admitted'
    item['contract']['coordinates']['position_unit'] = 'source'
    assert admission(item, 'panda')['status'] == 'pending_units'
    evidence = {'unit': 'meter', 'evidence': 'dataset owner confirmation', 'mapping_sha256': digest(item['contract'])}
    assert admission(item, 'panda', evidence)['status'] == 'admitted'
    item['contract']['coordinates']['target_frame'] = 'changed'
    assert admission(item, 'panda', evidence)['status'] == 'pending_units'


def test_franka_does_not_bypass_unknown_bases_or_absolute_pose_requirement():
    item = group('franka', 'RoboMINDv2')
    item['contract']['coordinates'].update(reference_frame='arm_base', common_base_known=False)
    assert admission(item)['reason'] == 'common arm base unknown'
    item = group('franka', 'RoboMIND')
    item['contract']['action_kind'] = 'delta_pose'
    assert admission(item)['status'] == 'deferred'


def test_metadata_conflict_and_unverified_units():
    assert admission(group(), 'panda')['status'] == 'deferred'
    assert 'conflicts' in admission(group(), 'panda')['reason']
    assert admission(group('simulated_agilex', 'InternData-M1'), 'panda')['status'] == 'deferred'
    item = group(unit='source')
    assert admission(item)['status'] == 'pending_units'
    evidence = {'unit': 'meter', 'evidence': 'dataset owner confirmation', 'mapping_sha256': digest(item['contract'])}
    assert admission(item, unit_evidence=evidence)['status'] == 'admitted'
    item['contract']['coordinates']['reference_frame'] = 'new_frame'
    assert admission(item, unit_evidence=evidence)['status'] == 'pending_units'


def test_episode_identity_and_diverse_sampling():
    m = {'groups': {'InternData-A1/a': group()}, 'jobs': [
        {'dataset': 'InternData-A1', 'root': root, 'group_directory': 'a', 'output_root': 'd/' + root,
         'selected_episodes': [0, 1, 2]} for root in ('root1', 'root2')]}
    catalog = Catalog(m)
    assert len(catalog.entries) == 6
    selected = catalog.sample(2)
    assert len({catalog.entries[key]['root'] for key in selected}) == 2
    assert all('_directory' not in entry for entry in catalog.public())


def test_rotation_rows_not_columns_and_invalid_masks():
    rotation = Rotation.from_euler('xyz', [.3, -.5, 1.2])
    values = [1, 2, 3, *rotation.as_matrix()[:2].ravel()]
    pose, issue = decode_pose(values, [True] * 9)
    assert issue is None
    np.testing.assert_allclose(Rotation.from_quat(pose[3:]).as_matrix(), rotation.as_matrix(), atol=1e-10)
    assert decode_pose(values, [False] + [True] * 8)[0] is None
    assert decode_pose([0] * 9, [True] * 9)[1] == 'degenerate_rotation'
    assert decode_pose([np.nan] + values[1:], [True] * 9)[1] == 'nonfinite_pose'


def test_time_and_gripper_values_are_not_normalized_twice():
    values = np.zeros((3, 34)); masks = np.zeros((3, 34), bool)
    values[:, 9:18] = [0.4, 0.2, 0.8, 1, 0, 0, 0, 1, 0]; masks[:, 9:19] = True
    values[:, 18] = [0, 50, 100]
    data = {'timestamp': [4, 4.1, 4.3], 'frame_index': [40, 41, 43], 'observation.state': values,
            'action': values, 'eef_mask': masks}
    result = adapt(data)
    np.testing.assert_allclose([f['sample_timestamp'] for f in result['frames']], [0, .1, .3])
    assert [f['state']['grippers']['left'] for f in result['frames']] == [0, 50, 100]
    assert result['frames'][0]['state']['targets']['torso'] is None
    assert result['frames'][2]['frame_index'] == 43
    data['sample_timestamp'] = [0, .1, .2]
    with pytest.raises(ValueError, match='sample_timestamp'):
        adapt(data)
    del data['sample_timestamp']; data['timestamp'] = [4, 4, 5]
    with pytest.raises(ValueError, match='strictly increasing'):
        adapt(data)


def test_franka_single_arm_keeps_right_slot_and_missing_left_pose():
    values = np.zeros((2, 34)); masks = np.zeros((2, 34), bool)
    values[:, 19:28] = [.4, -.2, .6, 1, 0, 0, 0, 1, 0]; masks[:, 19:29] = True
    values[:, 28] = [0, 100]
    result = adapt({'timestamp': [0, .1], 'observation.state': values, 'action': values, 'eef_mask': masks})
    for mode in ('state', 'action'):
        assert [f[mode]['grippers']['right'] for f in result['frames']] == [0, 100]
        for frame in result['frames']:
            assert frame[mode]['targets']['left'] is None
            assert frame[mode]['grippers']['left'] is None
            assert frame[mode]['targets']['right'][:3] == [.4, -.2, .6]


@pytest.mark.parametrize('header,expected', [(None, (0, 99, 200)), ('bytes=0-9', (0, 9, 206)),
    ('bytes=50-', (50, 99, 206)), ('bytes=-10', (90, 99, 206)), ('bytes=-500', (0, 99, 206)),
    ('bytes=90-150', (90, 99, 206))])
def test_ranges(header, expected):
    assert byte_range(header, 100) == expected


@pytest.mark.parametrize('header', ['bytes=100-', 'bytes=4-3', 'bytes=-0', 'bytes=-', 'bytes=0-1,4-5', 'items=1-2'])
def test_invalid_ranges(header):
    with pytest.raises(InvalidRange):
        byte_range(header, 100)


def test_http_range_head_cache_and_bounded_reads(tmp_path):
    path = tmp_path / 'sample.mp4'; path.write_bytes(bytes(range(256)) * 4096)
    stats = {'read_bytes': 0}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            send_file(self, path, stats=stats)
        do_HEAD = do_GET
        def log_message(self, *_):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    url = f'http://127.0.0.1:{server.server_port}/video'
    try:
        with urlopen(Request(url, headers={'Range': 'bytes=256-511'})) as response:
            assert response.status == 206 and response.read() == bytes(range(256))
            assert response.headers['Content-Range'] == f'bytes 256-511/{path.stat().st_size}'
            etag = response.headers['ETag']
        with urlopen(Request(url, method='HEAD', headers={'Range': 'bytes=-10'})) as response:
            assert response.status == 206 and response.read() == b''
            assert response.headers['Content-Length'] == '10'
        with pytest.raises(HTTPError) as caught:
            urlopen(Request(url, headers={'If-None-Match': etag}))
        assert caught.value.code == 304
        with pytest.raises(HTTPError) as caught:
            urlopen(Request(url, headers={'Range': 'bytes=999999999-'}))
        assert caught.value.code == 416
        assert caught.value.headers['Content-Range'] == f'bytes */{path.stat().st_size}'
        assert stats['read_bytes'] == 256
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_source_path_traversal_and_symlinks(tmp_path):
    root = tmp_path / 'source'; root.mkdir()
    (root / 'external').symlink_to(tmp_path)
    for name in ('../secret', '/etc/passwd', 'external/secret'):
        with pytest.raises(ValueError):
            confined(root, name)


def test_atomic_publication_rejects_nonfinite_without_changing_old_result(tmp_path):
    path = tmp_path / 'result.json'; atomic_json(path, {'status': 'complete'})
    with pytest.raises(ValueError):
        atomic_json(path, {'invalid': float('nan')})
    assert json.loads(path.read_text()) == {'status': 'complete'}
    assert list(tmp_path.iterdir()) == [path]


def test_active_preview_is_not_evicted(tmp_path):
    cache = PreviewCache(tmp_path, limit=20)
    (tmp_path / 'a.mp4').write_bytes(b'a' * 15)
    with cache.acquire('a'):
        with pytest.raises(OSError, match='protected'):
            cache.evict(10)
        assert (tmp_path / 'a.mp4').exists()
    assert cache.renew('a')
    with pytest.raises(OSError, match='protected'):
        cache.evict(10)
    cache.leases['a'] = 0
    cache.evict(10)
    assert not (tmp_path / 'a.mp4').exists()
    cache.executor.shutdown()


def test_queue_deduplicates_and_configuration_changes_key(tmp_path):
    import asyncio
    from src.eef_review.gateway import ReviewService
    config = {'cache': str(tmp_path), 'tool_offsets_m': {'left': [0, -.15, 0]},
              'source_base_to_chassis': np.eye(4).tolist(), 'sdk_to_usd_translation_m': [0, 0, -.097],
              'thresholds': {'position_mm': 20}}
    service = ReviewService(config)
    service.sdk_version = 'sdk-version'; service.implementation = 'code-version'
    service.initial_pose = {'joints': [0, 1]}; service.model = {'kinematics_fingerprint': 'usd-version'}
    async def episode(_):
        return {'source_version': 'source-version'}
    async def trajectory(_):
        return {'frames': [], 'parquet_sha256': 'parquet-version'}
    service.episode = episode; service.trajectory = trajectory
    service.public_episode = lambda *_: {'review': {'status': 'admitted'}, 'mapping_version': 'mapping-version'}
    async def check():
        first, second = await asyncio.gather(service.enqueue('episode', 'state'), service.enqueue('episode', 'state'))
        assert first['key'] == second['key'] and second['deduplicated']
        assert service.pending.qsize() == 1
        action = await service.enqueue('episode', 'action')
        assert action['key'] != first['key']
        service.initial_pose['joints'][0] = 1
        changed = await service.enqueue('episode', 'state')
        assert changed['key'] != first['key']
        service.config['tool_offsets_m']['left'][1] = -.16
        changed_tool = await service.enqueue('episode', 'state')
        assert changed_tool['key'] != changed['key']
        service.sdk_version = 'new-sdk'
        assert (await service.enqueue('episode', 'state'))['key'] != changed_tool['key']
        assert not (tmp_path / 'results').exists()
    asyncio.run(check())


def test_failed_frame_has_no_success_residual(tmp_path, monkeypatch):
    from src.eef_review import fk_worker
    from types import SimpleNamespace
    names = ['astribot_torso_link_4', 'astribot_arm_left_link_7', 'astribot_arm_right_link_7']
    monkeypatch.setattr(fk_worker, 'model', lambda _: (None, None, None, names, SimpleNamespace(fingerprint=lambda: 'model'), 'geometry'))
    monkeypatch.setattr(fk_worker, 'snapshot', lambda *_: [[0, 0, 0, 1, 0, 0, 0]] * 3)
    sdk_row = {'ik_success': False, 'status': 'ik_failed', 'joints': [0] * 25,
               'sdk_fk': {side: [0, 0, 0, 0, 0, 0, 1] for side in ('torso', 'left', 'right')},
               'joint_jump_deg': 0, 'limit_violations': [], 'torso_drift_deg': 0, 'solve_s': .01, 'seed_wait_s': .001}
    atomic_json(tmp_path / 'sdk.json', {'frames': [sdk_row], 'command_messages_observed': 0})
    job = {'key': 'k', 'episode_id': 'e', 'mode': 'state', 'provenance': {}, 'model': {'kinematics_fingerprint': 'model'},
           'config': {'tool_offsets_m': {side: [0, 0, 0] for side in ('torso', 'left', 'right')},
                      'sdk_to_usd_translation_m': [0, 0, 0], 'chunk_size': 128,
                      'thresholds': {'joint_jump_deg': 15, 'position_mm': 20, 'angle_deg': 5}},
           'trajectory': {'frames': [{'state': {'targets': {side: [1, 0, 0, 0, 0, 0, 1] for side in ('torso', 'left', 'right')}}}]}}
    fk_worker.run(job, tmp_path)
    row = json.loads((tmp_path / 'chunk-00000.json').read_text())['frames'][0]
    summary = json.loads((tmp_path / 'result.json').read_text())['summary']
    assert row['anomaly'] and all(e is None for e in row['errors'].values())
    assert summary['success_frames'] == 0 and summary['per_arm']['left']['position_mm'] is None
