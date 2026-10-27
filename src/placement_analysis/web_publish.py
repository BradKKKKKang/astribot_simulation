"""Atomically register immutable revisions; never copy bundles into public/data."""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
import time

from .artifact_store import ArtifactStore, atomic_write, json_bytes
from .bundle import store_for_manifest

DEFAULT_VIEWER = Path(__file__).resolve().parents[2] / 'web/placement-viewer'


def task_directory_name(value):
    name = value.strip()
    if not name or name.startswith('.') or Path(name).name != name or '\\' in name:
        raise ValueError('task name must be a single, non-hidden directory name')
    return name


def publish_bundle(output, *, task_name=None, viewer=DEFAULT_VIEWER):
    output, viewer = Path(output).resolve(), Path(viewer).resolve()
    name = task_directory_name(task_name if task_name is not None else output.name)
    source = output / 'placement.json'
    bundle = json.loads(source.read_bytes())
    if bundle.get('metadata', {}).get('status') != 'complete':
        raise ValueError('only complete placement bundles can be published')
    refs = []
    if bundle.get('schema_version') == 5:
        store = store_for_manifest(source, bundle)
        def include(ref):
            store.resolve(ref)
            refs.append(ref)
        include(bundle['robot_model'])
        include(bundle['reference_snapshot'])
        if bundle['metadata'].get('evaluation'):
            include(bundle['metadata']['evaluation'])
        seen_events, seen_episodes = set(), set()
        for episode in bundle['episodes']:
            if episode['episode_id'] in seen_episodes:
                raise ValueError('duplicate episode shard')
            seen_episodes.add(episode['episode_id'])
            include(episode['events'])
            shard = store.read_json(episode['events'])
            if shard['episode_id'] != episode['episode_id']:
                raise ValueError('episode shard identity mismatch')
            for event in shard['events']:
                if event.get('episode_id') != episode['episode_id'] or event.get('event_id') in seen_events:
                    raise ValueError('invalid or duplicate event identity')
                seen_events.add(event.get('event_id'))
                if event.get('snapshot_ref'):
                    include(event['snapshot_ref'])
    else:
        if bundle.get('robot_model', 'robot.glb') != 'robot.glb':
            raise ValueError('legacy bundle must use its companion robot.glb')
        store = ArtifactStore(output.parent / '.placement-artifacts')
        robot = store.put_file(output / 'robot.glb')
        refs.append(robot)
        bundle['robot_model'] = robot
    manifest_ref = store.put_json(bundle)
    revision = manifest_ref['sha256']
    store_id = hashlib.sha256(str(store.root).encode()).hexdigest()[:24]
    registry = viewer / '.placement-catalog.json'
    viewer.mkdir(parents=True, exist_ok=True)
    with (viewer / '.placement-catalog.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        catalog = json.loads(registry.read_text()) if registry.is_file() else {
            'version': 1, 'tasks': {}, 'revisions': {}, 'stores': {}}
        catalog['stores'].setdefault(store_id, {'root': str(store.root), 'artifacts': {}})
        registered = catalog['stores'][store_id]
        if registered['root'] != str(store.root):
            raise ValueError('artifact store identifier collision')
        for ref in refs:
            registered['artifacts'][Path(ref['path']).name] = ref
        catalog['revisions'][revision] = {'store_id': store_id, 'manifest': manifest_ref,
                                          'source_manifest': str(source)}
        catalog['tasks'][name] = {'name': name, 'revision': revision,
                                  'updated_at': time.time() * 1000}
        atomic_write(registry, json_bytes(catalog))
    print(f'WEB_PUBLISHED: {name} ({source}, revision {revision[:12]})', flush=True)
    return source
