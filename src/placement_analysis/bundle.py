"""Schema 5 manifests and compatibility expansion for CPU consumers."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import json

from .artifact_store import ArtifactStore, atomic_write, json_bytes


def bundle_path(path):
    path = Path(path).expanduser().resolve()
    return path / 'placement.json' if path.is_dir() else path


def store_for_manifest(path, manifest):
    location = manifest.get('artifact_store', {}).get('path')
    if not isinstance(location, str) or not location:
        raise ValueError('schema 5 manifest requires artifact_store.path')
    return ArtifactStore(bundle_path(path).parent / location)


def write_episode_shard(store, episode_id, events):
    thin = []
    seen = set()
    for original in events:
        event = copy.deepcopy(original)
        event_id = event.get('event_id')
        if not isinstance(event_id, str) or event_id in seen:
            raise ValueError('invalid or duplicate event identity')
        seen.add(event_id)
        if str(event.get('episode_id')) != str(episode_id):
            raise ValueError('event episode does not match shard')
        event.setdefault('object', None)
        event.setdefault('identification', {'status': 'not_requested'})
        snapshot = event.pop('snapshot', None)
        if snapshot is not None:
            event['snapshot_ref'] = store.put_json(snapshot)
        elif event.get('snapshot_ref'):
            store.resolve(event['snapshot_ref'])
        thin.append(event)
    return {'episode_id': str(episode_id), 'event_count': len(thin),
            'event_types': sorted({event['event_type'] for event in thin if 'event_type' in event}),
            'events': store.put_json({'episode_id': str(episode_id), 'events': thin})}


def default_display(metadata, episode_shards=None):
    detector = metadata.get('detector', {})
    kind = metadata.get('type', metadata.get('task_config', {}).get('type',
        detector.get('type', 'press' if detector.get('motion_detector') == 'press' else 'gripper')))
    kinds = ['press'] if kind == 'press' else ['grasp', 'release']
    available = sorted(set(kinds).union(*(item.get('event_types', []) for item in (episode_shards or []))))
    return {'event_types': available, 'default_event_types': kinds,
            'event_labels': {'grasp': '抓取', 'release': '释放', 'dwell': '停留', 'press': '按压候选'},
            'operation_label': '按压' if kind == 'press' else '夹爪开合',
            'candidate_help': '候选点由离线分析生成；视图筛选不改变分析结果。'}


def save_manifest_revision(output, metadata, reference_snapshot_ref, episode_shards,
                           robot_model_ref=None, *, display=None, store=None):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    store = store or ArtifactStore(output.parent / '.placement-artifacts')
    if len({item['episode_id'] for item in episode_shards}) != len(episode_shards):
        raise ValueError('duplicate episode shards')
    # References produced by the store are immutable; publication validates the full closure.
    manifest = {'schema_version': 5, 'metadata': copy.deepcopy(metadata),
                'artifact_store': {'path': os.path.relpath(store.root, output)},
                'robot_model': robot_model_ref, 'reference_snapshot': reference_snapshot_ref,
                'episodes': copy.deepcopy(episode_shards),
                'display': {**default_display(metadata, episode_shards), **copy.deepcopy(display or {})}}
    store.put_json(manifest)  # revision remains available even after placement.json advances
    atomic_write(output / 'placement.json', json_bytes(manifest))
    return output / 'placement.json'


def save_bundle_revision(output, events, metadata, reference_snapshot, *, store=None,
                         robot_model=None, display=None, episode_shards=None):
    output = Path(output).expanduser().resolve()
    previous = json.loads((output / 'placement.json').read_text()) if (output / 'placement.json').is_file() else None
    store = store or (store_for_manifest(output / 'placement.json', previous)
                      if previous and previous.get('schema_version') == 5
                      else ArtifactStore(output.parent / '.placement-artifacts'))
    if robot_model is None:
        if previous and previous.get('schema_version') == 5:
            old_store = store_for_manifest(output / 'placement.json', previous)
            old_ref = previous.get('robot_model')
            robot_model = (old_ref if old_store.root == store.root else store.put_file(old_store.resolve(old_ref))) if old_ref else None
        elif (output / 'robot.glb').is_file():
            robot_model = store.put_file(output / 'robot.glb')
    elif not isinstance(robot_model, dict):
        robot_model = store.put_file(robot_model)
    if episode_shards is None:
        # Event rows cannot represent completed episodes with zero candidates.
        # Keep the manifest's order, then add explicitly declared/new episodes.
        prior_episodes = previous.get('episodes', []) if previous and previous.get('schema_version') == 5 else []
        episode_ids = dict.fromkeys(str(row['episode_id']) for row in prior_episodes)
        episode_ids.update(dict.fromkeys(str(row['episode_id']) for row in metadata.get('episodes', [])))
        grouped = {episode: [] for episode in episode_ids}
        for event in events:
            grouped.setdefault(str(event['episode_id']), []).append(event)
        episode_shards = [write_episode_shard(store, episode, items) for episode, items in grouped.items()]
    reference_ref = (reference_snapshot if isinstance(reference_snapshot, dict) and 'sha256' in reference_snapshot
                     else store.put_json(reference_snapshot))
    return save_manifest_revision(output, metadata, reference_ref, episode_shards, robot_model,
                                  display=display, store=store)


def load_bundle(path):
    path = bundle_path(path)
    manifest = json.loads(path.read_text())
    version = manifest.get('schema_version')
    if version in (1, 2, 3, 4):
        return manifest
    if version != 5:
        raise ValueError('unsupported placement bundle schema')
    store = store_for_manifest(path, manifest)
    bundle = copy.deepcopy(manifest)
    bundle['reference_snapshot'] = store.read_json(manifest['reference_snapshot'])
    bundle['events'] = []
    seen = set()
    for episode in manifest['episodes']:
        shard = store.read_json(episode['events'])
        if shard['episode_id'] != episode['episode_id']:
            raise ValueError('episode shard identity mismatch')
        for event in shard['events']:
            if event['episode_id'] != episode['episode_id'] or event['event_id'] in seen:
                raise ValueError('invalid or duplicate event identity')
            seen.add(event['event_id'])
            if event.get('snapshot_ref'):
                event['snapshot'] = store.read_json(event['snapshot_ref'])
            bundle['events'].append(event)
    return bundle
