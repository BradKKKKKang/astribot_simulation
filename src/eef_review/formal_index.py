"""Build and query a disk-backed index of complete formal LeRobot episodes.

Only metadata is indexed. Source data/video files remain on the cloud mount.
The reviewed sample root registry supplies the admitted robot/mapping scope,
but every episode in those formal roots is indexed, without a five-item cap.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time
import zlib

from .catalog import Catalog, admission
from .common import atomic_json, confined, digest, read_json


def pack(value):
    return zlib.compress(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode())


def unpack(value):
    return json.loads(zlib.decompress(value))


def inspect_root(base, sample_root, job, group, units):
    import pyarrow.parquet as pq
    directory = confined(base, job['dataset'] + '/' + job['root'])
    info_raw = (directory / 'meta/info.json').read_bytes()
    info = json.loads(info_raw)
    complete_raw = (directory / 'COMPLETE.json').read_bytes()
    complete = json.loads(complete_raw)
    mapping = read_json(confined(sample_root, job['output_root']) / 'MAPPING.json')
    if digest(mapping['contract']) != digest(group['contract']):
        raise ValueError('sample root contract differs from admitted group')
    if complete['plan_hash'] != mapping['source_plan_hash']:
        plan = read_json(directory / 'meta/conversion.json')['plan']
        if plan['profile'] != group['details']['profile'] or plan['hash'] != complete['plan_hash']:
            raise ValueError('formal profile/completion mismatch')
        # A new source version may change video encoding or receipts. Its EEF
        # meaning must still match the reviewed coordinate/unit contract.
        for field in ('coordinates', 'streams', 'gripper_normalization', 'action_kind', 'pose_encoding'):
            if plan.get(field) != mapping['contract'].get(field):
                raise ValueError('formal EEF contract changed: ' + field)
    review = admission(group, info.get('robot_type', ''), units)
    if review['status'] != 'admitted':
        raise ValueError(review['reason'])
    if info.get('codebase_version') != 'v3.0':
        raise ValueError('formal reader currently requires LeRobot v3 metadata')
    for field in ('observation.state', 'action', 'eef_mask'):
        if info['features'][field]['shape'] != [34]:
            raise ValueError('formal EEF shape differs from the reviewed 34D contract')
    selected_info = {k: info[k] for k in ('fps', 'data_path', 'video_path', 'total_episodes', 'total_frames', 'chunks_size', 'robot_type') if k in info}
    selected_info['cameras'] = {k: v for k, v in info['features'].items() if v.get('dtype') == 'video'}
    paths = sorted((directory / 'meta/episodes').glob('chunk-*/*.parquet'))
    if not paths:
        raise ValueError('no episode metadata')
    rows, versions = [], []
    for path in paths:
        file = pq.ParquetFile(path)
        columns = [name for name in file.schema_arrow.names if name in
                   ('episode_index', 'length', 'tasks', 'dataset_from_index', 'dataset_to_index')
                   or name.startswith('data/') or name.startswith('videos/')]
        versions.append([str(path.relative_to(directory)), digest(path.read_bytes())])
        for batch in file.iter_batches(batch_size=2048, columns=columns):
            for row in batch.to_pylist():
                if row['length'] <= 0 or row['dataset_to_index'] - row['dataset_from_index'] != row['length']:
                    raise ValueError('episode frame interval differs from length')
                rows.append(row)
    if len(rows) != info['total_episodes'] or sum(row['length'] for row in rows) != info['total_frames']:
        raise ValueError('episode metadata totals differ from info.json')
    if len({row['episode_index'] for row in rows}) != len(rows):
        raise ValueError('duplicate episode indices')
    if (directory / 'COMPLETE.json').read_bytes() != complete_raw or (directory / 'meta/info.json').read_bytes() != info_raw:
        raise ValueError('formal root changed during indexing')
    record = {'dataset': job['dataset'], 'root': job['root'], 'group': job['dataset'] + '/' + job['group_directory'],
              'profile': group['details']['profile'], 'info': selected_info, 'review': review,
              'complete_sha256': digest(complete_raw), 'info_sha256': digest(info_raw),
              'formal_plan_hash': complete['plan_hash'], 'mapping_version': digest(group['contract']),
              'episode_metadata_versions': versions}
    record['source_version'] = digest(record)
    return record, rows


def build(base, sample_root, output, units, workers=8):
    base, sample_root, output = Path(base), Path(sample_root), Path(output)
    manifest = read_json(sample_root / 'SAMPLE_MANIFEST.json')
    catalog = Catalog(manifest, units)
    groups = {key: group for key, group in catalog.groups.items()
              if admission(group, unit_evidence=units.get(key))['status'] == 'admitted'}
    jobs = [job for job in manifest['jobs'] if job['dataset'] + '/' + job['group_directory'] in groups]
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_suffix('.building.sqlite')
    connection = sqlite3.connect(staging)
    connection.executescript('''
        PRAGMA journal_mode=DELETE;
        CREATE TABLE IF NOT EXISTS roots (id INTEGER PRIMARY KEY, dataset TEXT, root TEXT, group_id TEXT,
            profile TEXT, record BLOB, UNIQUE(dataset,root));
        CREATE TABLE IF NOT EXISTS episodes (id TEXT PRIMARY KEY, root_id INTEGER, episode INTEGER,
            frames INTEGER, task TEXT, metadata BLOB, UNIQUE(root_id,episode));
        CREATE INDEX IF NOT EXISTS episode_root ON episodes(root_id,episode);
        CREATE INDEX IF NOT EXISTS root_group ON roots(group_id);
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT);
    ''')
    build_scope = digest([manifest['id'], units, sorted(groups), 'formal-index-v2'])
    prior = connection.execute("SELECT value FROM metadata WHERE key='build_scope'").fetchone()
    if prior and prior[0] != build_scope:
        connection.close()
        raise ValueError('staging index belongs to another source/units/scope version; choose another output path')
    connection.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', ('build_scope', build_scope))
    connection.commit()
    done = {(d, r) for d, r in connection.execute('SELECT dataset,root FROM roots')}
    remaining = [job for job in jobs if (job['dataset'], job['root']) not in done]
    started = time.monotonic()
    errors = []

    def read(job):
        try:
            group_id = job['dataset'] + '/' + job['group_directory']
            return job, inspect_root(base, sample_root, job, groups[group_id], units.get(group_id)), None
        except Exception as error:
            return job, None, str(error)

    # Bounded batches avoid retaining metadata for half a million episodes in RAM.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(remaining), workers * 2):
            for job, value, error in pool.map(read, remaining[start:start + workers * 2]):
                if error:
                    errors.append({'dataset': job['dataset'], 'root': job['root'], 'error': error})
                    continue
                root, episodes = value
                with connection:
                    cursor = connection.execute('INSERT INTO roots(dataset,root,group_id,profile,record) VALUES (?,?,?,?,?)',
                                                (root['dataset'], root['root'], root['group'], root['profile'], pack(root)))
                    root_id = cursor.lastrowid
                    connection.executemany('INSERT INTO episodes VALUES (?,?,?,?,?,?)', [
                        (digest([root['dataset'], root['root'], row['episode_index']])[:24], root_id,
                         row['episode_index'], row['length'], ' / '.join(row.get('tasks') or []), pack(row)) for row in episodes])
                done.add((root['dataset'], root['root']))
            atomic_json(output.with_suffix('.progress.json'), {'roots': len(done), 'total_roots': len(jobs),
                        'errors': errors, 'elapsed_s': time.monotonic() - started})
            print(f'INDEX {len(done)}/{len(jobs)} roots; errors={len(errors)}', flush=True)
    if errors:
        connection.close()
        raise RuntimeError(f'{len(errors)} roots failed audit; see {output.with_suffix(".progress.json")}')
    roots = [unpack(row[0]) for row in connection.execute('SELECT record FROM roots ORDER BY dataset,root')]
    count, frames = connection.execute('SELECT count(*),sum(frames) FROM episodes').fetchone()
    for group_id, group in groups.items():
        members = [root for root in roots if root['group'] == group_id]
        group['roots'] = len(members)
        group['episodes'] = sum(root['info']['total_episodes'] for root in members)
        group['frames'] = sum(root['info']['total_frames'] for root in members)
    result = {'schema': 'eef_review_formal_index_v1', 'source_kind': 'formal', 'roots': len(roots),
              'episodes': count, 'frames': frames, 'groups': groups,
              'scope': 'All formal episodes in reviewed robot roots, including Franka; unresolved units/bases deferred',
              'root_registry_sample_id': manifest['id'], 'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'root_versions': [[root['dataset'], root['root'], root['source_version']] for root in roots]}
    result['id'] = digest(result)
    with connection:
        connection.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', ('manifest', json.dumps(result)))
    assert connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    connection.close()
    staging.replace(output)
    atomic_json(output.with_suffix('.json'), result)
    return result


class IndexedEntries:
    def __init__(self, catalog):
        self.catalog = catalog
    def __len__(self):
        return self.catalog.manifest['episodes']
    def __getitem__(self, key):
        return self.catalog.get(key)[0]
    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False


class FormalCatalog:
    def __init__(self, path):
        self.path = Path(path).resolve()
        with self.connect() as connection:
            self.manifest = json.loads(connection.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
        self.version = self.manifest['id']
        self.groups = self.manifest['groups']
        self.entries = IndexedEntries(self)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(f'{self.path.as_uri()}?mode=ro', uri=True)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def public_row(self, row):
        group = self.groups[row['group_id']]
        return {'id': row['id'], 'dataset': row['dataset'], 'root': row['root'], 'episode': row['episode'],
                'group': row['group_id'], 'profile': row['profile'], 'task': row['task'], 'frames': row['frames'],
                'coordinates': group['contract']['coordinates'],
                'review': {'status': 'admitted', 'reason': 'audited formal EEF root; source base assumed identical to SDK chassis'},
                'mapping_version': digest(group['contract']), 'source_version': self.version, 'source_kind': 'formal'}

    def get(self, key):
        with self.connect() as connection:
            row = connection.execute('SELECT e.*,r.dataset,r.root,r.group_id,r.profile,r.record FROM episodes e JOIN roots r ON r.id=e.root_id WHERE e.id=?', (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return self.public_row(row), unpack(row['record']), unpack(row['metadata'])

    def query(self, query):
        where, values = [], []
        for key, column in [('group', 'r.group_id'), ('dataset', 'r.dataset')]:
            if query.get(key):
                where.append(column + '=?'); values.append(query[key])
        if query.get('review') not in (None, '', 'admitted'):
            where.append('0')
        if query.get('q'):
            where.append("instr(lower(r.dataset||' '||r.root||' '||r.profile||' '||e.task||' '||e.episode),lower(?))>0")
            values.append(query['q'])
        for key, operator in [('ids', 'IN'), ('exclude_ids', 'NOT IN')]:
            if key in query:
                where.append(f'e.id {operator} (SELECT value FROM json_each(?))')
                values.append(json.dumps(query[key]))
        clause = ' WHERE ' + ' AND '.join(where) if where else ''
        joined = ' FROM episodes e JOIN roots r ON r.id=e.root_id'
        limit, offset = min(200, max(1, int(query.get('limit', 80)))), max(0, int(query.get('offset', 0)))
        with self.connect() as connection:
            total = connection.execute('SELECT count(*)' + joined + clause, values).fetchone()[0]
            rows = connection.execute('SELECT e.id,e.episode,e.frames,e.task,r.dataset,r.root,r.group_id,r.profile' + joined + clause
                                      + ' ORDER BY e.root_id,e.episode LIMIT ? OFFSET ?', [*values, limit, offset]).fetchall()
        return {'total': total, 'episodes': [self.public_row(row) for row in rows], 'groups': sorted(self.groups),
                'source_version': self.version, 'source_kind': 'formal'}

    def sample(self, per_group=1):
        result = []
        with self.connect() as connection:
            for group in sorted(self.groups):
                roots = connection.execute('SELECT id FROM roots WHERE group_id=? ORDER BY id', (group,)).fetchall()
                selected = []
                for offset in range(per_group):
                    for root in roots:
                        row = connection.execute('SELECT id FROM episodes WHERE root_id=? ORDER BY episode LIMIT 1 OFFSET ?', (root['id'], offset)).fetchone()
                        if row:
                            selected.append(row['id'])
                        if len(selected) >= per_group:
                            break
                    if len(selected) >= per_group:
                        break
                result.extend(selected)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--sample-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    result = build(args.root, args.sample_root, args.output, read_json(args.config)['unit_evidence'], args.workers)
    print(json.dumps({k: v for k, v in result.items() if k not in ('groups', 'root_versions')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
