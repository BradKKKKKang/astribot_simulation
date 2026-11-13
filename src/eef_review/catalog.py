"""Manifest-only directory index and conservative robot/unit admission."""
from collections import defaultdict
import re

from .common import digest
from .mapping import alignment_policy, describe

FRANKA_CANDIDATES = {
    ('InternData-A1', 'sim_franka'), ('InternData-M1', 'simulated_franka'),
    ('MolmoAct2-DROID-Dataset', 'droid_root'), ('RoboMIND', 'franka'),
    ('RoboMIND', 'fr3'), ('RoboMINDv2', 'franka'),
}

CANDIDATES = FRANKA_CANDIDATES | {
    ('InternData-A1', 'sim_lift2'), ('InternData-A1', 'sim_split_aloha'),
    ('InternData-A1', 'sim_genie1'), ('PrimeBot-Challenge', 'primebot_x2w_fold_cloth'),
    ('RW-RL-Dataset', 'g2'), ('RW-RL-Dataset', 'r1lite'),
}


def admission(group, robot_type='', unit_evidence=None, policy=None):
    description = describe(group)
    profile, coord = description['profile'], description['coordinates']
    if description['has_pose'] is False:
        return {'status': 'no_eef', 'reason': 'mapping declares no EEF pose'}
    if policy is not None:
        alignment_policy(group, {group['dataset'] + '/' + group['directory']: policy})
        return {'status': 'admitted', 'reason': 'user-selected ' + policy['mode'] + '; saved positions interpreted as meters',
                'unit_evidence': {'unit': 'meter', 'assumed': True, 'evidence': policy['evidence'],
                                  'mapping_sha256': policy['mapping_sha256']}}
    identity = ' '.join([profile, robot_type or '', group['dataset'], coord.get('target_frame') or '']).lower()
    if 'agilex' in profile and group['dataset'] == 'InternData-M1':
        return {'status': 'deferred', 'reason': 'independent_arm_bases; robot_type=panda conflicts with Agilex mapping'}
    if ((group['dataset'], profile) not in FRANKA_CANDIDATES
            and any(re.search(r'(?<![a-z])' + term + r'(?![a-z])', identity)
                    for term in ('franka', 'panda', 'fr3', 'droid'))):
        return {'status': 'deferred', 'reason': 'Franka identity conflicts with the registered mapping profile'}
    if description['kind'] == 'legacy' and (group['dataset'], profile) not in CANDIDATES:
        return {'status': 'deferred', 'reason': 'source origin unconfirmed or outside reviewed scope'}
    if coord.get('common_base_known') is False or coord.get('reference_frame') == 'arm_base':
        return {'status': 'deferred', 'reason': 'common arm base unknown'}
    if coord.get('position_unit') not in ('m', 'meter', 'meters'):
        if not unit_evidence or unit_evidence.get('unit') != 'meter' or not unit_evidence.get('evidence'):
            return {'status': 'pending_units', 'reason': 'position_unit=source; metric evidence required'}
        if unit_evidence.get('mapping_sha256') != digest(group['contract']):
            return {'status': 'pending_units', 'reason': 'unit evidence belongs to another mapping version'}
    if description['action_kind'] != 'absolute_pose':
        return {'status': 'deferred', 'reason': 'absolute EEF action semantics require confirmation'}
    return {'status': 'admitted', 'reason': 'metric pose; source base assumed identical to SDK chassis',
            'unit_evidence': unit_evidence or {'unit': 'meter', 'evidence': 'mapping contract'}}


class Catalog:
    def __init__(self, manifest, units=None, policies=None, source_id=None):
        self.manifest = manifest
        self.version = digest(manifest)
        self.entries = {}
        self.groups = manifest['groups']
        units = units or {}
        policies = policies or {}
        for job in manifest['jobs']:
            group_id = job['dataset'] + '/' + job['group_directory']
            group = self.groups[group_id]
            policy = alignment_policy(group, policies) if policies else None
            description = describe(group)
            review = admission(group, unit_evidence=units.get(group_id), policy=policy)
            for episode in job['selected_episodes']:
                identity = [job['dataset'], job['root'], int(episode)]
                if source_id:
                    identity.insert(0, source_id)
                key = digest(identity)[:24]
                if key in self.entries:
                    raise ValueError('duplicate episode identity')
                self.entries[key] = {
                    'id': key, 'dataset': job['dataset'], 'root': job['root'],
                    'episode': int(episode), 'group': group_id,
                    'profile': description['profile'],
                    'coordinates': description['coordinates'], 'review': review,
                    'mapping_version': digest(group['contract']), 'source_version': self.version,
                    '_directory': job['output_root'] + f'/episode_{episode:06d}',
                    '_root_directory': job['output_root'],
                    **({'source_id': source_id, 'source_kind': 'sample'} if source_id else {}),
                    **({'alignment_policy': policy} if policy else {}),
                }

    def public(self, include_excluded=False):
        return [{k: v for k, v in row.items() if not k.startswith('_')}
                for row in self.entries.values()
                if include_excluded or row['review']['status'] in ('admitted', 'pending_units')]

    def sample(self, per_group=1):
        """Round robin roots, then episode number, without selecting excluded robots."""
        grouped = defaultdict(lambda: defaultdict(list))
        for row in self.entries.values():
            if row['review']['status'] in ('admitted', 'pending_units'):
                grouped[row['group']][row['root']].append(row['id'])
        result = []
        for roots in grouped.values():
            selected = []
            for index in range(max(map(len, roots.values()))):
                for rows in roots.values():
                    if index < len(rows):
                        selected.append(rows[index])
                if len(selected) >= per_group:
                    break
            result.extend(selected[:per_group])
        return result

    def query(self, query):
        rows = self.public(include_excluded=True)
        for key in ('group', 'dataset'):
            if query.get(key):
                rows = [row for row in rows if row[key] == query[key]]
        if query.get('review'):
            rows = [row for row in rows if row['review']['status'] == query['review']]
        if query.get('q'):
            term = query['q'].lower()
            rows = [row for row in rows if term in ' '.join(str(row.get(key, '')) for key in
                    ('dataset', 'root', 'profile', 'task', 'episode')).lower()]
        for key in ('ids', 'exclude_ids'):
            if key in query:
                ids = set(query[key])
                rows = [row for row in rows if (row['id'] in ids) == (key == 'ids')]
        offset, limit = max(0, int(query.get('offset', 0))), min(200, max(1, int(query.get('limit', 80))))
        return {'total': len(rows), 'episodes': rows[offset:offset + limit], 'groups': sorted(self.groups),
                'source_version': self.version, 'source_kind': 'sample'}


class RemoteCatalog:
    """Small manifest plus on-demand entry metadata; full pagination stays in SQL."""
    def __init__(self, manifest):
        self.manifest = manifest
        self.version = manifest['id']
        self.groups = manifest['groups']
        self.entries = {}
