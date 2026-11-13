"""Read an indexed formal episode without copying or modifying the source root."""
from contextlib import contextmanager
from functools import lru_cache
import math
from pathlib import Path
import tempfile
import threading

from .common import confined, digest
from .formal_index import FormalCatalog
from .preview import PreviewCache


def file_version(path):
    stat = path.stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


class FormalReader:
    def __init__(self, root, cache, index, *, preview=None):
        self.root = Path(root).resolve()
        self.catalog = FormalCatalog(index)
        self.reader_version = digest([Path(__file__).read_text(), Path(__file__).with_name('formal_index.py').read_text()])
        self.catalog.manifest = {**self.catalog.manifest, 'reader_version': self.reader_version,
                                 'id': digest([self.catalog.version, self.reader_version])}
        self.catalog.version = self.catalog.manifest['id']
        self.preview = preview if preview is not None else PreviewCache(cache)
        self.scratch = Path(cache).parent / 'formal' / 'episode-reads'
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.read_slots = threading.BoundedSemaphore(2)
        self.stats = {'read_bytes': 0, 'requests': 0, 'disconnects': 0}

    def checked_root(self, record):
        root = confined(self.root, record['dataset'] + '/' + record['root'])
        if digest((root / 'COMPLETE.json').read_bytes()) != record['complete_sha256']:
            raise ValueError('formal source version changed; rebuild the index before reading this root')
        return root

    def episode(self, key):
        row, root, _ = self.catalog.get(key)
        return row, self.checked_root(root)

    @lru_cache(maxsize=256)
    def metadata(self, key):
        row, record, episode = self.catalog.get(key)
        root = self.checked_root(record)
        info = record['info']
        group = self.catalog.groups[row['group']]
        videos = []
        for camera, feature in info['cameras'].items():
            prefix = 'videos/' + camera + '/'
            if prefix + 'file_index' not in episode:
                continue
            relative = info['video_path'].format(video_key=camera, chunk_index=episode[prefix + 'chunk_index'],
                                                  file_index=episode[prefix + 'file_index'])
            path = confined(root, relative)
            video_info = feature.get('info', {})
            fps = float(video_info.get('video.fps', info['fps']))
            start, end = float(episode[prefix + 'from_timestamp']), float(episode[prefix + 'to_timestamp'])
            if not all(math.isfinite(v) for v in (fps, start, end)) or fps <= 0 or start < 0 or end <= start:
                raise ValueError('invalid formal video interval')
            if not math.isclose(fps, float(info['fps']), rel_tol=1e-4):
                raise ValueError('formal source video/data FPS differ; explicit frame selection is required')
            if abs((end - start) * fps - episode['length']) > 1.01:
                raise ValueError('formal video interval differs from episode frame count')
            shape = feature['shape']
            width, height = int(video_info.get('video.width', shape[1])), int(video_info.get('video.height', shape[0]))
            source_version = file_version(path)
            clip = {'start': start, 'end': end, 'frames': episode['length'], 'fps': fps, 'width': width, 'height': height}
            videos.append({'camera': camera, 'source_name': relative, 'source_version': source_version,
                           'requires_preview': True, 'kind': 'formal_episode_preview', 'clip': clip,
                           'selection': {'mode': 'formal_episode_clip', 'first_source_time': start,
                                         'last_source_time': start + (episode['length'] - 1) / fps,
                                         'frames': episode['length'], 'output_fps': fps},
                           'output': {'codec': 'h264', 'width': width, 'height': height, 'frames': episode['length'],
                                      'fps': fps, 'source_version': source_version, 'selection': clip,
                                      'root_version': record['source_version']}})
        source = {'schema': 'eef_review_formal_episode_v1', 'source_kind': 'formal',
                  'source_dataset': row['dataset'], 'source_root': row['root'], 'source_episode_index': row['episode'],
                  'frames': episode['length'], 'fps': info['fps'], 'sample_time_column': 'sample_timestamp',
                  'original_ids_and_eef_values_preserved': True, 'source_episode_metadata': episode,
                  'root_version': record['source_version'], 'formal_plan_hash': record['formal_plan_hash'], 'videos': videos}
        mapping = {'contract': group['contract'], 'processing': group['processing'],
                   'formal_plan_hash': record['formal_plan_hash'], 'stored_values_copied_without_new_coordinate_or_gripper_transform': True}
        return {'source': source, 'mapping': mapping, 'robot_type': info.get('robot_type', ''),
                'source_version': digest([self.reader_version, record['source_version'], source]),
                'catalog_entry': row}

    @contextmanager
    def data_file(self, key):
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        row, record, episode = self.catalog.get(key)
        root = self.checked_root(record)
        info = record['info']
        relative = info['data_path'].format(chunk_index=episode['data/chunk_index'], file_index=episode['data/file_index'])
        first = confined(root, relative)
        columns = ['observation.state', 'action', 'eef_mask', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index']
        with self.read_slots, tempfile.TemporaryDirectory(prefix=key + '-', dir=self.scratch) as temporary:
            schema = pq.read_schema(first)
            selected = [name for name in columns if name in schema.names]
            tables = [pq.read_table(first, columns=selected, filters=[('episode_index', '=', row['episode'])])]
            count = len(tables[0])
            if count < episode['length']:
                # An episode may cross a shard boundary. Discover only this root's
                # data shards, and read successors until its declared length ends.
                paths = sorted((root / 'data').glob('chunk-*/*.parquet'))
                offset = paths.index(first)
                for path in paths[offset + 1:]:
                    table = pq.read_table(path, columns=selected, filters=[('episode_index', '=', row['episode'])])
                    if not len(table):
                        break
                    tables.append(table); count += len(table)
                    if count >= episode['length']:
                        break
            table = pa.concat_tables(tables)
            if len(table) != episode['length']:
                raise ValueError('formal extraction did not recover the complete episode')
            frames = np.asarray(table['frame_index'].to_pylist())
            times = np.asarray(table['timestamp'].to_pylist(), dtype=float)
            if not np.array_equal(frames, np.arange(len(table))) or not np.isfinite(times).all():
                raise ValueError('formal episode has invalid frame IDs/timestamps')
            if 'index' in selected and not np.array_equal(table['index'].to_pylist(), np.arange(episode['dataset_from_index'], episode['dataset_to_index'])):
                raise ValueError('formal source indices disagree with episode metadata')
            sample = times - times[0]
            if not np.allclose(sample, np.arange(len(table)) / info['fps'], atol=max(1e-4, .02 / info['fps']), rtol=0):
                raise ValueError('formal timestamps require a nonuniform video frame selection')
            table = table.append_column('sample_timestamp', pa.array(sample))
            path = Path(temporary) / 'eef.parquet'
            pq.write_table(table, path, compression='zstd')
            self.checked_root(record)
            yield path

    def camera(self, key, camera):
        _, record, _ = self.catalog.get(key)
        root = self.checked_root(record)
        video = next(v for v in self.metadata(key)['source']['videos'] if v['camera'] == camera)
        path = confined(root, video['source_name'])
        if file_version(path) != video['source_version']:
            raise ValueError('formal video changed since this episode was indexed/read')
        return path, video
