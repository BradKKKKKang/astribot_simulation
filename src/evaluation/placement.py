"""Offline orchestration boundaries; no SDK, ROS or simulation imports."""
import copy
import fcntl
from pathlib import Path

from src.placement_analysis.artifact_store import atomic_write, json_bytes
from src.placement_analysis.bundle import load_bundle, save_bundle_revision, store_for_manifest
from .task_evaluation import evaluate_success


def _evaluate_bundle(path, *, task_config=None, dry_run=False, observations=None):
    path = Path(path).expanduser().resolve()
    output = path if path.is_dir() else path.parent
    bundle = load_bundle(path)
    metadata = bundle['metadata']
    if task_config:
        from src.simulation_config.experiment_config import resolve_experiment
        resolved = resolve_experiment(task_config, require_ready=not dry_run)
        if resolved['kind'] != 'evaluation':
            raise ValueError('offline evaluation requires an evaluation experiment')
        sections = resolved['sections']
        sections = {**sections, 'success': sections['task']['success']}
    else:
        sections = metadata.get('task_config', {}).get('_resolved_sections', {})
        if not {'scene', 'task', 'success'} <= sections.keys():
            raise ValueError('an explicit evaluation experiment is required for this placement bundle')
    if isinstance(observations, (str, Path)):
        from src.simulation_config.experiment_config import read_json
        observations = read_json(Path(observations).expanduser())
    if observations is None:
        observations = metadata.get('observations', {'episodes': {}})
    if (not isinstance(observations, dict) or set(observations) != {'episodes'}
            or not isinstance(observations['episodes'], dict)):
        raise ValueError('observations must be an object containing an episodes mapping')
    results = []
    complete_shards = {row['episode_id'] for row in bundle.get('episodes', [])} if bundle['schema_version'] == 5 else set()
    episode_ids = list(dict.fromkeys([e['episode_id'] for e in metadata.get('episodes', [])]
                                    + [e['episode_id'] for e in bundle.get('episodes', [])]
                                    + [e['episode_id'] for e in bundle['events']]))
    if set(observations['episodes']) - set(episode_ids):
        raise ValueError('observations reference an unknown episode')
    for episode in episode_ids:
        events = [event for event in bundle['events'] if event['episode_id'] == episode]
        evidence = observations['episodes'].get(episode, {})
        if not isinstance(evidence, dict) or 'events_complete' in evidence:
            raise ValueError('episode observations must be an object; events_complete is pipeline-owned')
        result = evaluate_success(events, sections['scene'], sections['task'], sections['success'],
                                  observations={**evidence, 'events_complete': metadata.get('status') == 'complete'
                                                or episode in complete_shards})
        results.append({'episode_id': episode, **result})
    report = {'schema_version': 1, 'episodes': results,
              'summary': {status: sum(row['status'] == status for row in results)
                          for status in ('passed', 'failed', 'unknown')},
              'config': {name: copy.deepcopy(sections[name]) for name in ('scene', 'task', 'success')},
              'observations': copy.deepcopy(observations)}
    if not dry_run:
        if bundle['schema_version'] == 5:
            store = store_for_manifest(output / 'placement.json', bundle)
            metadata['evaluation'] = store.put_json(report)
            metadata['evaluation_summary'] = report['summary']
            metadata['observations'] = copy.deepcopy(observations)
            metadata.setdefault('stages', {})['evaluation'] = 'complete'
            save_bundle_revision(output, bundle['events'], metadata, bundle['reference_snapshot'],
                                 store=store, display=bundle.get('display'))
        else:
            atomic_write(output / 'evaluation.json', json_bytes(report))
    return report


def evaluate_bundle(path, *, task_config=None, dry_run=False, observations=None):
    if dry_run:
        return _evaluate_bundle(path, task_config=task_config, dry_run=True, observations=observations)
    path = Path(path).expanduser().resolve()
    output = path if path.is_dir() else path.parent
    with (output / '.pipeline.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another pipeline stage owns this bundle') from error
        return _evaluate_bundle(path, task_config=task_config, observations=observations)
