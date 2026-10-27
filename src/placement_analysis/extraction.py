"""CPU-only extraction from SDK -> ROS -> Isaac captures, with resumable shards."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import multiprocessing
import os
from pathlib import Path

from .events import DetectorConfig
from .recorder import PlacementRecorder
from .reference_pose import UsdKinematics
from .robot_export import DEFAULT_USD, open_robot_stage
from src.simulation_config.robot_profile import profile_for_settings
from .task_config import DEFAULT_DETECTOR_CONFIG, EVENT_TYPES, load_task_config
from .extraction_worker import (StageCache, capture_digests, digest, extract_episode,
                                file_digest, initialize_worker, normalized_capture,
                                run_episode_worker)


def load_capture(path, detector):
    """Compatibility reader; the extraction pipeline reads each source only once."""
    path = Path(path).expanduser().resolve(strict=True)
    metadata, times, positions, quality, _, _ = normalized_capture(
        path, profile_for_settings({}), detector.max_gap_s)
    return path, metadata, times, positions, quality


def discover_captures(inputs, *, incremental=False):
    """Strict complete batches by default; explicit incremental accepts completed shards."""
    paths = []
    for value in inputs:
        path = Path(value).expanduser().resolve(strict=True)
        manifest = path / 'batch_manifest.json'
        if (path / 'run_metadata.json').is_file():
            paths.append(path)
        elif manifest.is_file():
            batch = json.loads(manifest.read_text())
            ids = [episode['episode_id'] for episode in batch['episodes']]
            completed = batch.get('completed', [])
            if (not ids or len(set(ids)) != len(ids)
                    or any(not isinstance(name, str) or Path(name).name != name or name in ('.', '..') for name in ids)
                    or not isinstance(completed, list) or len(set(completed)) != len(completed)
                    or any(name not in ids for name in completed)):
                raise ValueError(f'{path}: batch has invalid episode IDs or completion records')
            if not incremental and (batch.get('status') != 'complete' or completed != ids):
                raise ValueError(f'{path}: batch is incomplete; finish replay first or select incremental extraction')
            paths.extend(path / name for name in ids if not incremental or name in completed)
        else:
            raise ValueError(f'{path}: expected an episode capture or a replay batch')
    if len({path.name for path in paths}) != len(paths):
        raise ValueError('episode directory names must be unique in one bundle')
    if not paths:
        raise ValueError('no completed episode captures available')
    return paths


def detector_settings(detector_config):
    settings = json.loads(Path(detector_config).expanduser().read_text())
    if 'task' in settings or 'schema_version' in settings:
        settings = load_task_config(detector_config, validate_identification=False)
    if settings.get('type') not in EVENT_TYPES or 'motion_detector' in settings:
        raise ValueError('detector configuration requires type: gripper or press')
    detector = DetectorConfig(**{k: v for k, v in settings.items()
                                 if k in DetectorConfig.__dataclass_fields__})
    return settings, detector


def prepare_extraction(usd=DEFAULT_USD, detector_config=DEFAULT_DETECTOR_CONFIG, *, settings=None):
    """Check model/TCP contracts on the CPU before replay acquires control."""
    if settings is None:
        settings, _ = detector_settings(detector_config)
    profile = profile_for_settings(settings)
    stage, root, names = open_robot_stage(usd, root_path=profile.root_path)
    model = UsdKinematics(stage, root.GetPath(), names)
    declaration = settings.get('_resolved_sections', {}).get('robot_placement')
    if declaration is not None:
        from src.simulation_config.robot_placement import resolve_robot_placement
        placement = resolve_robot_placement(profile, declaration, stage=stage)
        model.world_transform = model.Gf.Matrix4d(placement['world_transform'])
        model.placement = placement
    if set(profile.feedback_joint_positions(profile.joint_names, [0.] * len(profile.joint_names))[0]) != set(model.joint_names):
        raise ValueError('USD joints do not match the SDK feedback mapping')
    PlacementRecorder(None, settings, 'preflight', body_names=names, joint_names=model.joint_names)
    return stage, root, names, model


def _metadata(settings, detector, model_path, model, names, geometry, task):
    result = {
        'units': 'meters', 'up_axis': 'Z', 'quaternion_order': 'wxyz',
        'position_source': 'USD link-frame FK from synchronized Isaac ROS joint feedback',
        'gripper_source': 'Isaac ROS measured SDK percentage, clipped to [0,100]',
        'replay_mode': 'Astribot SDK -> ROS2 command -> Isaac Lab -> ROS2 state capture; CPU extraction',
        'time_basis': 'ROS header wall time; not PhysX simulation time',
        'base_assumption': 'fixed base at authored USD world pose; no mobile base reconstruction',
        'finger_pose_source': 'other fingers inferred from measured L1 percentage and signed linkage',
        'placement_meaning': 'TCP transition and motion candidates; not object centers or confirmed contact',
        'type': detector.type, 'candidate_policy': 'one detector selected by type; preserve startup candidates for audit',
        'candidate_basis': ('world Z down/bottom/up with time smoothing, bounded XY excursion and gripper gating'
                            if detector.type == 'press' else 'confirmed gripper open/closed transitions'),
        'initial_phase_basis': 'per-hand sustained departure from initial TCP in base coordinates',
        'detector': asdict(detector), 'tcp': settings['tcp'], 'event_rules': settings.get('event_rules'),
        'tcp_calibration': settings.get('tcp_calibration', ''),
        'base_body': settings['base_body'], 'body_names': names, 'joint_names': model.joint_names,
        'robot_usd': str(model_path), 'robot_usd_sha256': file_digest(model_path),
        'robot_kinematics_sha256': model.fingerprint(), 'robot_geometry': geometry,
    }
    if task:
        result['task_config'] = task
    if hasattr(model, 'placement'):
        result['robot_placement'] = model.placement
        result['base_assumption'] = 'fixed base verified from actual Isaac body pose feedback; per-episode provenance retained'
    return result


def extract_events(inputs, output, *, usd=DEFAULT_USD, detector_config=None, task_config=None,
                   data_root=None, dry_run=False, no_progress=False, store_root=None,
                   workers=1, resume=False, incremental=False, resolved=None):
    """Publish schema 5 revisions; identification is an explicit separate phase.

    A single coordinator owns checkpoints and manifests. At most ``workers``
    CPU episodes are in flight, and only immutable intermediate stages are shared.
    Resume requires unchanged candidate/kinematic dependencies; append completed
    captures using incremental mode. Existing legacy bundles are never overwritten.
    """
    if type(workers) is not int or not 1 <= workers <= max(1, os.cpu_count() or 1):
        raise ValueError('workers must be between 1 and the available CPU count')
    if task_config is not None and detector_config is not None:
        raise ValueError('--task-config and --detector-config are mutually exclusive')
    task = load_task_config(task_config, require_ready=not dry_run,
                            validate_identification=False) if task_config else None
    if task and task['task']['status'] != 'ready':
        print(f"{task['task']['name']}: 待标注; configuration valid, extraction disabled")
        return None
    settings, detector = (detector_settings(task_config or detector_config or DEFAULT_DETECTOR_CONFIG)
                          if resolved is None else (resolved['settings'], DetectorConfig(**{
                              k: v for k, v in resolved['settings'].items() if k in DetectorConfig.__dataclass_fields__})))
    profile = profile_for_settings(settings)
    sections = settings.get('_resolved_sections', {})
    alignment = sections.get('runtime', {}).get('alignment', {}).get('max_gap_s', detector.max_gap_s)
    inputs = list(inputs)
    captures = discover_captures(inputs, incremental=incremental)
    batches_complete = all(json.loads((Path(value).expanduser() / 'batch_manifest.json').read_text()).get('status') == 'complete'
                           for value in inputs if (Path(value).expanduser() / 'batch_manifest.json').is_file())
    if dry_run:
        for path in captures:
            capture_metadata, times, _, quality, _, _ = normalized_capture(path, profile, alignment)
            if sections.get('robot_placement') is not None:
                from src.simulation_config.robot_placement import validate_capture_placement
                validate_capture_placement(capture_metadata, sections['robot_placement'], profile)
            elif capture_metadata.get('robot_placement_attestations'):
                raise ValueError('grounded capture requires its original explicit robot_placement configuration')
            print(f'Validated {path.name}: {len(times)} feedback samples; {quality["gap_count"]} gaps', flush=True)
        print('Captures valid. No simulation, ROS, SDK, USD loading or output writes.')
        return None
    output = Path(output).expanduser().resolve()
    checkpoint_path = output / '.extraction-checkpoint.json'
    if output.exists() and (not output.is_dir() or any(output.iterdir())) and not resume:
        raise FileExistsError(f'refusing to overwrite existing bundle: {output}; use resume for extraction checkpoints')
    if resume and output.exists() and any(output.iterdir()) and not checkpoint_path.is_file():
        raise ValueError('resume requires an extraction checkpoint; existing bundles are not overwritten')
    model_path = (profile.model_path if usd == DEFAULT_USD and sections else Path(usd)).expanduser().resolve(strict=True)
    stage, root, names, model = prepare_extraction(model_path, settings=settings)
    from .artifact_store import ArtifactStore, atomic_write, json_bytes
    from .bundle import save_manifest_revision, write_episode_shard
    from .robot_export import cached_robot_geometry
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.pipeline.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another pipeline coordinator owns this output') from exc
        prior = json.loads(checkpoint_path.read_text()) if resume and checkpoint_path.exists() else None
        root_path = Path(store_root).expanduser().resolve() if store_root else (
            Path(prior['store_root']) if prior else output.parent / '.placement-artifacts')
        store = ArtifactStore(root_path)
        mapping = profile.as_dict()
        mapping = {'parts': [{key: part[key] for key in ('name', 'feedback_joints', 'model_joints')}
                             for part in mapping['parts']], 'grippers': mapping['grippers']}
        signature = digest({'version': 1, 'detector': asdict(detector), 'tcp': settings['tcp'], 'event_rules': settings.get('event_rules'),
                            'base_body': settings['base_body'], 'event_rules': settings.get('event_rules'),
                            'robot_feedback_mapping': mapping,
                            'kinematics': model.fingerprint(), 'alignment_max_gap_s': alignment})
        if prior and (prior['signature'] != signature or Path(prior['store_root']) != store.root):
            raise ValueError('resume configuration/store differs; extract into a new output to reuse stage caches')
        current_ids = [path.name for path in captures]
        input_digests = {path.name: capture_digests(path) for path in captures}
        analysis_id = digest({'extraction': signature, 'inputs': [
            {'episode_id': name, 'capture_sha256': input_digests[name]} for name in current_ids]})
        manifest_path = output / 'placement.json'
        previous_manifest = json.loads(manifest_path.read_text()) if resume and manifest_path.is_file() else None
        if prior and any(name not in current_ids for name in prior['episodes']):
            raise ValueError('resume inputs must include all previously extracted episodes')
        checkpoint = prior or {'version': 1, 'signature': signature, 'store_root': str(store.root), 'episodes': {}}
        atomic_write(checkpoint_path, json_bytes(checkpoint))
        geometry = cached_robot_geometry(stage, root.GetPath(), names, store, usd=model_path)
        metadata = _metadata(settings, detector, model_path, model, names, geometry,
                             task or (settings if 'task' in settings else None))
        metadata['analysis_id'] = analysis_id
        metadata['analysis_version'] = 1
        if previous_manifest and 'task_config' in previous_manifest['metadata']:
            metadata['task_config'] = previous_manifest['metadata']['task_config']
        metadata['alignment_max_gap_s'] = alignment
        metadata['incremental'] = incremental
        metadata['capture_batches_complete'] = batches_complete
        cache = StageCache(store.root)
        pending = []
        for path in captures:
            old = checkpoint['episodes'].get(path.name)
            if old and old['episode']['capture_sha256'] != input_digests[path.name]:
                raise ValueError(f'{path.name}: completed capture changed; extract into a new output')
            if old:
                shard = store.read_json(old['shard']['events'])
                for event in shard['events']:
                    if event.get('snapshot_ref'):
                        store.resolve(event['snapshot_ref'])
                store.resolve(old['reference'])
            else:
                pending.append(path)
        # A no-op resume must not replace recognized/evaluated shards with the
        # original extraction checkpoint. Preserve downstream stages and refs.
        if (not pending and previous_manifest
                and previous_manifest['metadata'].get('analysis_id') == analysis_id):
            previous_metadata = previous_manifest['metadata']
            previous_metadata['status'] = 'complete' if batches_complete else 'partial'
            previous_metadata['capture_batches_complete'] = batches_complete
            previous_metadata.setdefault('stages', {}).update(
                capture='complete' if batches_complete else 'partial', extraction='complete')
            save_manifest_revision(output, previous_metadata, previous_manifest['reference_snapshot'],
                                   previous_manifest['episodes'], previous_manifest.get('robot_model'),
                                   display=previous_manifest.get('display'), store=store)
            print(f'PLACEMENT_SAVED: {manifest_path}', flush=True)
            return manifest_path
        def publish():
            completed = [checkpoint['episodes'][name] for name in current_ids if name in checkpoint['episodes']]
            atomic_write(checkpoint_path, json_bytes(checkpoint))
            if completed:
                metadata['episodes'] = [item['episode'] for item in completed]
                extraction_complete = len(completed) == len(captures)
                metadata['status'] = 'complete' if extraction_complete and batches_complete else 'partial'
                metadata['stages'] = {'capture': 'complete' if batches_complete else 'partial',
                                      'extraction': 'complete' if extraction_complete else 'partial',
                                      'identification': 'not_requested', 'evaluation': 'not_requested'}
                save_manifest_revision(output, metadata, completed[0]['reference'],
                                       [item['shard'] for item in completed], geometry,
                                       display=sections.get('display'), store=store)
        publish()
        def accept(result):
            episode = result['episode']
            if (episode['capture_sha256'] != input_digests[episode['episode_id']]
                    or result['kinematics_sha256'] != metadata['robot_kinematics_sha256']):
                raise ValueError(f"{episode['episode_id']}: worker source/model changed during extraction")
            checkpoint['episodes'][episode['episode_id']] = {
                'episode': episode, 'shard': write_episode_shard(store, episode['episode_id'], result['events']),
                'reference': store.put_json(result['reference'])}
            publish()
            if not no_progress:
                print(f"{episode['episode_id']}: saved {episode['event_count']} candidates; cache {result['cache_hits']}", flush=True)
        if workers == 1:
            for path in pending:
                accept(extract_episode(path, settings, model, profile, cache, alignment_max_gap_s=alignment))
        elif pending:
            # Bounded submission avoids ProcessPoolExecutor.map retaining one
            # serialized episode result for every capture behind a slow episode.
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                     initializer=initialize_worker,
                                     initargs=(str(model_path), settings, str(store.root))) as executor:
                iterator = iter(pending)
                futures = [executor.submit(run_episode_worker, (str(path), alignment))
                           for path in [next(iterator, None) for _ in range(min(workers, len(pending)))]
                           if path is not None]
                while futures:
                    future = futures.pop(0)
                    accept(future.result())
                    path = next(iterator, None)
                    if path is not None:
                        futures.append(executor.submit(run_episode_worker, (str(path), alignment)))
        print(f'PLACEMENT_SAVED: {output / "placement.json"}', flush=True)
        return output / 'placement.json'
