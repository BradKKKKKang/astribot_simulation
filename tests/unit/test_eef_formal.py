"""Formal root indexing, complete episode extraction and bounded video clipping."""
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from src.eef_review.common import atomic_json, digest
from src.eef_review.formal_index import build, FormalCatalog
from src.eef_review.formal_reader import FormalReader
from src.eef_review.preview import PreviewCache, probe


@pytest.fixture
def formal_root(tmp_path):
    base, sample = tmp_path / 'formal', tmp_path / 'sample'
    root = base / 'InternData-A1/task'; root.mkdir(parents=True)
    contract = {'coordinates': {'position_unit': 'meter', 'reference_frame': 'robot', 'target_frame': 'robot_base'}, 'action_kind': 'absolute_pose'}
    group = {'dataset': 'InternData-A1', 'processing': 'test', 'details': {'profile': 'sim_lift2'}, 'contract': contract}
    group_id = 'InternData-A1/test'
    atomic_json(sample / 'SAMPLE_MANIFEST.json', {'id': 'snapshot', 'groups': {group_id: group}, 'jobs': [
        {'dataset': 'InternData-A1', 'root': 'task', 'group_directory': 'test', 'output_root': 'selected/task', 'selected_episodes': [1]}]})
    atomic_json(sample / 'selected/task/MAPPING.json', {'contract': contract, 'source_plan_hash': 'same-plan'})
    atomic_json(root / 'COMPLETE.json', {'plan_hash': 'same-plan'})
    features = {name: {'shape': [34]} for name in ['observation.state', 'action', 'eef_mask']}
    for camera in ('head', 'left', 'right'):
        features['observation.images.' + camera] = {'dtype': 'video', 'shape': [48, 64, 3],
            'info': {'video.codec': 'h264', 'video.fps': 10, 'video.width': 64, 'video.height': 48}}
        video = root / f'videos/observation.images.{camera}/chunk-000/file-000.mp4'
        video.parent.mkdir(parents=True, exist_ok=True); video.write_bytes(b'fixture-only')
    info = {'codebase_version': 'v3.0', 'fps': 10, 'robot_type': 'ARX Lift-2', 'chunks_size': 1000,
            'total_episodes': 2, 'total_frames': 6, 'features': features,
            'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
            'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'}
    atomic_json(root / 'meta/info.json', info)
    episode_rows = []
    for episode in range(2):
        row = {'episode_index': episode, 'length': 3, 'tasks': ['task ' + str(episode)],
               'data/chunk_index': 0, 'data/file_index': episode,
               'dataset_from_index': episode * 3, 'dataset_to_index': (episode + 1) * 3}
        for camera in ('head', 'left', 'right'):
            prefix = 'videos/observation.images.' + camera + '/'
            row.update({prefix + 'chunk_index': 0, prefix + 'file_index': 0,
                        prefix + 'from_timestamp': 1 + .3 * episode, prefix + 'to_timestamp': 1.3 + .3 * episode})
        episode_rows.append(row)
    episode_path = root / 'meta/episodes/chunk-000/file-000.parquet'; episode_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(episode_rows), episode_path)
    values = [0.] * 34; values[9:18] = [.4, .2, .8, 1, 0, 0, 0, 1, 0]; values[18] = 50
    mask = [False] * 34; mask[9:19] = [True] * 10
    rows = [{'index': i, 'episode_index': i // 3, 'frame_index': i % 3, 'timestamp': 5 + (i % 3) / 10,
             'task_index': i // 3, 'observation.state': values, 'action': values, 'eef_mask': mask} for i in range(6)]
    data = root / 'data/chunk-000'; data.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows[:2]), data / 'file-000.parquet')
    pq.write_table(pa.Table.from_pylist(rows[2:]), data / 'file-001.parquet')
    index = tmp_path / 'cache/index.sqlite'
    return base, sample, root, index, rows


def test_index_admits_every_episode_and_supports_sql_pagination(formal_root):
    base, sample, root, index, rows = formal_root
    manifest = build(base, sample, index, {}, workers=1)
    assert manifest['episodes'] == 2 and manifest['groups']['InternData-A1/test']['episodes'] == 2
    catalog = FormalCatalog(index)
    page = catalog.query({'limit': 1, 'offset': 0})
    assert page['total'] == 2 and page['episodes'][0]['episode'] == 0  # absent from selected_episodes
    first = page['episodes'][0]['id']
    assert catalog.query({'exclude_ids': [first]})['episodes'][0]['episode'] == 1
    assert catalog.query({'ids': [first]})['total'] == 1
    assert catalog.query({'ids': []})['total'] == 0
    assert catalog.query({'q': "' OR 1=1 --"})['total'] == 0
    assert catalog.query({'q': 'Franka'})['total'] == 0
    assert catalog.query({'review': 'excluded'})['total'] == 0


@pytest.mark.parametrize('dataset,profile,robot_type,root_name', [
    ('InternData-A1', 'sim_franka', 'franka', 'task'),
    ('InternData-M1', 'simulated_franka', 'panda', 'simulated/franka'),
    ('MolmoAct2-DROID-Dataset', 'droid_root', 'panda', ''),
    ('RoboMIND', 'franka', 'franka', 'Franka/task'),
    ('RoboMIND', 'fr3', 'fr3', 'Franka-FR3-Dual/task'),
])
def test_formal_franka_roots_include_unsampled_episodes(formal_root, dataset, profile, robot_type, root_name):
    base, sample, root, index, rows = formal_root
    destination = base / dataset / root_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination != root:
        root.rename(destination)
    info = json.loads((destination / 'meta/info.json').read_text())
    info['robot_type'] = robot_type
    atomic_json(destination / 'meta/info.json', info)
    manifest = json.loads((sample / 'SAMPLE_MANIFEST.json').read_text())
    group = manifest['groups'].pop('InternData-A1/test')
    group['dataset'] = dataset; group['details']['profile'] = profile
    manifest['groups'][dataset + '/test'] = group
    manifest['jobs'][0].update(dataset=dataset, root=root_name)
    atomic_json(sample / 'SAMPLE_MANIFEST.json', manifest)
    built = build(base, sample, index, {}, workers=1)
    assert built['episodes'] == 2
    reader = FormalReader(base, index.parent / 'media', index)
    try:
        page = reader.catalog.query({'dataset': dataset})
        assert page['total'] == 2
        assert all(e['review']['status'] == 'admitted' for e in page['episodes'])
        key = page['episodes'][0]['id']  # episode 0 was not sampled
        assert reader.metadata(key)['robot_type'] == robot_type
        assert key in reader.catalog.sample(1)
        with reader.data_file(key) as path:
            assert pq.read_table(path)['observation.state'].to_pylist() == [r['observation.state'] for r in rows[:3]]
    finally:
        reader.preview.executor.shutdown()


def test_extracts_episode_across_shards_without_reapplying_eef_transforms(formal_root):
    base, sample, root, index, rows = formal_root
    build(base, sample, index, {}, workers=1)
    reader = FormalReader(base, index.parent / 'media', index)
    key = reader.catalog.query({'limit': 1})['episodes'][0]['id']
    metadata = reader.metadata(key)
    assert metadata['source']['source_kind'] == 'formal'
    assert metadata['source']['videos'][0]['clip']['start'] == 1
    assert metadata['source']['videos'][0]['requires_preview']
    with reader.data_file(key) as path:
        data = pq.read_table(path).to_pydict()
        assert data['frame_index'] == [0, 1, 2]
        assert data['episode_index'] == [0, 0, 0]
        np.testing.assert_allclose(data['sample_timestamp'], [0, .1, .2], atol=1e-12)
        assert data['observation.state'] == [row['observation.state'] for row in rows[:3]]
    assert not path.exists()
    atomic_json(root / 'COMPLETE.json', {'plan_hash': 'new-plan'})
    with pytest.raises(ValueError, match='source version changed'):
        reader.camera(key, 'observation.images.head')
    reader.preview.executor.shutdown()


def test_incomplete_episode_metadata_is_not_published(formal_root):
    base, sample, root, index, _ = formal_root
    info = json.loads((root / 'meta/info.json').read_text()); info['total_frames'] += 1
    atomic_json(root / 'meta/info.json', info)
    with pytest.raises(RuntimeError, match='failed audit'):
        build(base, sample, index, {}, workers=1)
    assert not index.exists()


def test_formal_startup_never_falls_back_to_a_cached_sample_catalog(tmp_path):
    import asyncio
    from src.eef_review.gateway import ReviewService
    service = ReviewService({'cache': str(tmp_path), 'source_mode': 'formal', 'unit_evidence': {}})
    atomic_json(tmp_path / 'manifest.json', {'groups': {}, 'jobs': []})
    async def offline(_):
        raise ConnectionError('reader restarting')
    service.upstream_json = offline
    with pytest.raises(ValueError, match='source mode differs'):
        asyncio.run(service.refresh_catalog())
    async def formal(_):
        return {'source_kind': 'formal', 'id': 'formal-version', 'groups': {}}
    service.upstream_json = formal
    asyncio.run(service.refresh_catalog())
    assert service.catalog.version == 'formal-version'


def test_formal_catalog_recovers_from_old_startup_cache_before_opening_franka(tmp_path):
    import asyncio
    from src.eef_review.catalog import RemoteCatalog
    from src.eef_review.gateway import ReviewService
    service = ReviewService({'cache': str(tmp_path), 'source_mode': 'formal', 'unit_evidence': {}})
    service.catalog = RemoteCatalog({'id': 'old', 'groups': {}})
    group = {'dataset': 'InternData-M1', 'details': {'profile': 'simulated_franka'},
             'contract': {'coordinates': {'position_unit': 'meter'}}}
    row = {'id': 'new-episode', 'group': 'InternData-M1/franka', 'source_version': 'new'}
    async def upstream(path):
        if path == '/manifest':
            return {'source_kind': 'formal', 'id': 'new', 'groups': {'InternData-M1/franka': group}}
        return {'catalog_entry': row}
    service.upstream_json = upstream
    metadata = asyncio.run(service.episode('new-episode'))
    assert service.catalog.version == 'new'
    assert service.catalog.groups[metadata['catalog_entry']['group']] == group
    cached = json.loads((tmp_path / 'metadata/new-episode.json').read_text())
    assert cached['manifest'] == 'new'


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='FFmpeg required for real clip validation')
def test_formal_video_clip_starts_at_the_requested_source_frame(tmp_path):
    source = tmp_path / 'source.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i',
                    'color=black:size=64x48:rate=10', '-vf', 'geq=lum=N*6+16:cb=128:cr=128',
                    '-t', '3', '-c:v', 'libx264', '-crf', '0', '-pix_fmt', 'yuv420p', str(source)], check=True)
    cache = PreviewCache(tmp_path / 'clips')
    selection = {'start': 1., 'end': 1.3, 'frames': 3, 'fps': 10., 'width': 64, 'height': 48}
    job = cache.request(source, {'version': 'source'}, 10, selection)
    cache.executor.submit(lambda: None).result(timeout=20)
    result = cache.jobs[job['key']]
    assert result['status'] == 'complete', result
    assert result['first_source_pts'] == 1. and result['last_source_pts'] == 1.2
    output = cache.root / (job['key'] + '.mp4')
    assert probe(output)[1] == [0., .1, .2]
    def pixels(path, start):
        return np.frombuffer(subprocess.check_output(['ffmpeg', '-v', 'error', '-ss', str(start), '-i', str(path),
                  '-frames:v', '3', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-']), dtype=np.uint8).astype(float)
    assert np.mean(np.abs(pixels(source, 1) - pixels(output, 0))) < 2.5
    cache.evict(cache.limit)
    assert not output.exists()
    again = cache.request(source, {'version': 'source'}, 10, selection)
    cache.executor.submit(lambda: None).result(timeout=20)
    assert again['key'] == job['key'] and output.exists()
    assert cache.jobs[job['key']]['status'] == 'complete'
    cache.executor.shutdown(wait=True)
