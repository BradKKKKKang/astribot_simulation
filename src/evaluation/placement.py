"""Offline orchestration boundaries; no SDK, ROS or simulation imports."""
import copy
import fcntl
from pathlib import Path

from src.placement_analysis.artifact_store import atomic_write, json_bytes
from src.placement_analysis.bundle import load_bundle, save_bundle_revision, store_for_manifest
from .task_evaluation import evaluate_task


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
    else:
        sections = metadata.get('task_config', {}).get('_resolved_sections', {})
        if not {'scene', 'task'} <= sections.keys():
            raise ValueError('an explicit evaluation experiment is required for this placement bundle')
        if 'success' not in sections['task']:
            sections = copy.deepcopy(sections)
            sections['task']['success'] = sections['success']
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
        result = evaluate_task(events, sections['scene'], sections['task'],
                                  observations={**evidence, 'events_complete': metadata.get('status') == 'complete'
                                                or episode in complete_shards})
        results.append({'episode_id': episode, **result})
    report = {'schema_version': 1, 'episodes': results,
              'summary': {status: sum(row['status'] == status for row in results)
                          for status in ('passed', 'failed', 'unknown')},
              'config': {name: copy.deepcopy(sections[name]) for name in ('scene', 'task')},
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


def main(argv=None):
    """Evaluate existing placement evidence with an explicit evaluation experiment."""
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--observations', type=Path, help='explicit per-episode contact and object state evidence')
    parser.add_argument('--dry-run', action='store_true', help='evaluate without writing bundle revisions')
    args = parser.parse_args(argv)
    report = evaluate_bundle(args.bundle, task_config=args.experiment,
                             observations=args.observations, dry_run=args.dry_run)
    print(json.dumps(report['summary'], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
