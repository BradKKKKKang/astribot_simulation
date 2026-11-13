"""Explicit sample sources alongside the existing paginated formal index."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path

from .catalog import admission
from .cloud import Reader
from .common import confined, digest, read_json
from .formal_reader import FormalReader
from .mapping import alignment_policy, episode_review, group_label
from .preview import PreviewCache


class SampleReader(Reader):
    def __init__(self, source, config, cache, preview=None):
        super().__init__(source['root'], cache, units=config.get('unit_evidence'),
                         policies=config.get('alignment_policies'), source_id=source['id'], preview=preview)
        manifest = self.catalog.manifest
        self.roots = {}

        def inspect(job):
            root = confined(self.root, job['output_root'])
            raw = (root / 'ROOT_COMPLETE.json').read_bytes()
            receipt = json.loads(raw)
            if (receipt.get('status') != 'passed' or receipt.get('manifest_id') != manifest['id']
                    or receipt.get('selected_episodes') != job['selected_episodes']
                    or (receipt.get('dataset'), receipt.get('root')) != (job['dataset'], job['root'])):
                raise ValueError('sample completion differs from selected manifest: ' + job['output_root'])
            mapping, info = read_json(root / 'MAPPING.json'), read_json(root / 'SOURCE_INFO.json')
            group_id = job['dataset'] + '/' + job['group_directory']
            group = self.catalog.groups[group_id]
            if digest(mapping['contract']) != digest(group['contract']):
                raise ValueError('sample mapping differs from group: ' + group_id)
            if any(info['features'][name]['shape'] != [34] for name in ('observation.state', 'action', 'eef_mask')):
                raise ValueError('sample requires 34D state/action/mask')
            policy = alignment_policy(group, config.get('alignment_policies', {}))
            review = admission(group, info.get('robot_type', ''), config.get('unit_evidence', {}).get(group_id), policy)
            return job['output_root'], {'receipt': receipt, 'receipt_sha256': digest(raw),
                                        'mapping': mapping, 'info': info, 'review': review}

        with ThreadPoolExecutor(max_workers=6) as pool:
            self.roots.update(pool.map(inspect, manifest['jobs']))
        for row in self.catalog.entries.values():
            record = self.roots[row['_root_directory']]
            episode = next(e for e in record['receipt']['episodes'] if e['episode'] == row['episode'])
            row.update(frames=episode['frames'], source_label=source['label'],
                       valid_pose_frames=episode['valid_pose_frames'],
                       review=episode_review(record['review'], episode['valid_pose_frames']))
        self.catalog.version = digest([manifest['id'], source, config.get('unit_evidence', {}),
                                       config.get('alignment_policies', {}),
                                       sorted((key, value['receipt_sha256']) for key, value in self.roots.items())])

    def episode(self, key):
        row, directory = super().episode(key)
        record = self.roots[row['_root_directory']]
        if digest((directory.parent / 'ROOT_COMPLETE.json').read_bytes()) != record['receipt_sha256']:
            raise ValueError('sample completion changed; reload the registered source')
        return row, directory

    def metadata(self, key):
        value = super().metadata(key)
        row = self.catalog.entries[key]
        if value['source']['frames'] != row['frames']:
            raise ValueError('sample frame count differs from completion')
        if value['source'].get('eef_valid_pose_frames') != row['valid_pose_frames']:
            raise ValueError('sample EEF availability differs from completion')
        return value


class CombinedEntries:
    def __init__(self, owner):
        self.owner = owner

    def __len__(self):
        return sum(len(reader.catalog.entries) for reader in self.owner.readers.values())


class CombinedReader:
    def __init__(self, config, cache):
        self.preview = PreviewCache(cache)
        self.stats = {'read_bytes': 0, 'requests': 0, 'disconnects': 0}
        self.readers, self.sources, self.sample_routes = {}, [], {}
        for source in config['sample_sources']:
            if source['id'] == 'formal' or source['id'] in self.readers:
                raise ValueError('duplicate/reserved sample source ID')
            reader = SampleReader(source, config, cache, self.preview)
            self.readers[source['id']] = reader
            self.sources.append({'id': source['id'], 'label': source['label'], 'kind': 'sample'})
            for key in reader.catalog.entries:
                if key in self.sample_routes:
                    raise ValueError('duplicate sample identity')
                self.sample_routes[key] = source['id']
        self.readers['formal'] = FormalReader(config['remote_root'], cache, config['remote_index'], preview=self.preview)
        self.sources.append({'id': 'formal', 'label': Path(config['remote_root']).name, 'kind': 'formal'})
        self.groups, self.group_options = {}, []
        for source in self.sources:
            reader = self.readers[source['id']]
            for key, group in reader.catalog.groups.items():
                if key in self.groups:
                    raise ValueError('mapping group occurs in multiple registered sources: ' + key)
                self.groups[key] = group
                self.group_options.append({'id': key, 'label': group_label(key, group),
                                           'source_id': source['id'], 'episodes': group.get('episodes', 0)})
        implementation = digest([(name, digest(Path(__file__).with_name(name).read_bytes())) for name in
                                 ('sources.py', 'cloud.py', 'catalog.py', 'mapping.py')])
        self.version = digest([[(key, reader.catalog.version) for key, reader in self.readers.items()], implementation])
        self.entries = CombinedEntries(self)
        self.manifest = {'id': self.version, 'source_kind': 'mixed', 'groups': self.groups,
                         'sources': self.sources, 'group_options': self.group_options, 'episodes': len(self.entries)}
        self.catalog = self

    def route(self, key):
        return self.sample_routes.get(key, 'formal')

    def query(self, query):
        offset, limit = max(0, int(query.get('offset', 0))), min(200, max(1, int(query.get('limit', 80))))
        total, rows = 0, []
        for source in self.sources:
            if query.get('source') and query['source'] != source['id']:
                continue
            reader = self.readers[source['id']]
            page = reader.catalog.query({**query, 'offset': offset, 'limit': max(1, limit - len(rows))})
            total += page['total']
            if len(rows) < limit:
                rows.extend({**row, 'source_id': source['id'], 'source_label': source['label'],
                             'source_kind': source['kind'], 'source_version': self.version} for row in page['episodes'])
            offset = max(0, offset - page['total'])
        return {'total': total, 'episodes': rows, 'groups': sorted(self.groups), 'group_options': self.group_options,
                'sources': self.sources, 'source_version': self.version, 'source_kind': 'mixed'}

    def sample(self, per_group=1):
        return [key for reader in self.readers.values() for key in reader.catalog.sample(per_group)]

    def metadata(self, key):
        source_id = self.route(key)
        reader = self.readers[source_id]
        value = reader.metadata(key)
        row = value.get('catalog_entry')
        if row is None:
            row = {k: v for k, v in reader.catalog.entries[key].items() if not k.startswith('_')}
        source = next(s for s in self.sources if s['id'] == source_id)
        return {**value, 'catalog_entry': {**row, 'source_id': source_id, 'source_label': source['label'],
                                          'source_kind': source['kind'], 'source_version': self.version}}

    @contextmanager
    def data_file(self, key):
        reader = self.readers[self.route(key)]
        if isinstance(reader, SampleReader):
            reader.metadata(key)
        with reader.data_file(key) as path:
            yield path

    def camera(self, key, camera):
        return self.readers[self.route(key)].camera(key, camera)
