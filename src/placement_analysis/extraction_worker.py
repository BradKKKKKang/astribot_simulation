"""CPU episode extraction with immutable, independently keyed stage caches.

Workers own no manifests. Each returns one episode; the coordinator alone
publishes coherent bundle revisions. Atomic unique temporary files allow two
runs to share the same cache safely.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pyarrow.parquet as pq

from .events import DetectorConfig
from .recorder import PlacementRecorder
from .ros_feedback import synchronize_feedback

CAPTURE_FILES = ('sim_states.parquet', 'api_commands.parquet', 'run_metadata.json')
STAGE_VERSION = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def capture_digests(path):
    return {name: file_digest(Path(path) / name) for name in CAPTURE_FILES}


class StageCache:
    """Private intermediate arrays; not browser artifacts or mutable manifests."""
    def __init__(self, root):
        self.root = Path(root) / 'stages'

    def _path(self, stage, key):
        return self.root / stage / key[:2] / (key + '.npz')

    def get(self, stage, key):
        path = self._path(stage, key)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as archive:
            result = {name: archive[name] for name in archive.files}
        metadata = json.loads(str(result.pop('_metadata')))
        return metadata, result

    def put(self, stage, key, metadata, **arrays):
        path = self._path(stage, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix='.', suffix='.tmp', dir=path.parent)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                np.savez(stream, _metadata=json.dumps(metadata, sort_keys=True), **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def normalized_capture(path, profile, max_gap_s, cache=None, source_digests=None):
    path = Path(path).expanduser().resolve(strict=True)
    source_digests = source_digests or capture_digests(path)
    key = digest({'version': STAGE_VERSION, 'capture': source_digests,
                  'feedback_joint_names': profile.joint_names, 'anchor_joint': profile.anchor_joint,
                  'part_by_joint': profile.feedback_parts, 'max_gap_s': max_gap_s})
    cached = cache.get('synchronized', key) if cache else None
    if cached is not None:
        details, arrays = cached
        return details['capture_metadata'], arrays['times'], arrays['positions'], details['quality'], key, True
    metadata = json.loads((path / 'run_metadata.json').read_text())
    if 'Astribot SDK' not in metadata.get('control_path', '') or 'Isaac Lab' not in metadata['control_path']:
        raise ValueError(f'{path}: SDK -> ROS -> Isaac capture required')
    rows = pq.read_table(path / 'sim_states.parquet').to_pylist()
    api = pq.read_table(path / 'api_commands.parquet').to_pylist()
    if capture_digests(path) != source_digests:
        raise ValueError(f'{path.name}: capture changed while reading synchronized input')
    times, positions, quality = synchronize_feedback(rows, api, metadata, profile.joint_names, max_gap_s,
                                                       anchor_joint=profile.anchor_joint,
                                                       part_by_joint=profile.feedback_parts)
    if cache:
        cache.put('synchronized', key, {'capture_metadata': metadata, 'quality': quality},
                  times=times, positions=positions)
    return metadata, times, positions, quality, key, False


def extract_episode(path, settings, model, profile, cache, *, alignment_max_gap_s=None):
    """Minimal endpoint FK for the trajectory; full poses only for saved frames."""
    path = Path(path)
    detector = DetectorConfig(**{key: value for key, value in settings.items()
                                 if key in DetectorConfig.__dataclass_fields__})
    source_digests = capture_digests(path)
    metadata, times, positions, quality, synchronized_key, sync_hit = normalized_capture(
        path, profile, detector.max_gap_s if alignment_max_gap_s is None else alignment_max_gap_s,
        cache, source_digests)
    declared_kinematics_sha256 = model.fingerprint()
    declaration = settings.get('_resolved_sections', {}).get('robot_placement')
    actual_placement = None
    if declaration is not None:
        import copy
        from src.simulation_config.robot_placement import validate_capture_placement, pose_matrix
        actual_placement = validate_capture_placement(metadata, declaration, profile)
        resolved = actual_placement['resolved']
        if resolved != model.placement:
            raise ValueError('capture grounded geometry/profile differs from extraction model')
        # Preserve measured world coordinates, including tiny numerical offsets.
        model = copy.copy(model)
        model.world_transform = (model.Gf.Matrix4d(resolved['world_transform'])
                                 * pose_matrix(resolved['base_pose_world']).GetInverse()
                                 * pose_matrix(actual_placement['actual_base_pose_world']))
    elif metadata.get('robot_placement_attestations'):
        raise ValueError('grounded capture requires its original explicit robot_placement configuration')
    endpoints = list(dict.fromkeys([settings['base_body'],
                                   *[settings['tcp'][side]['body'] for side in ('left', 'right')]]))
    # The conversion result is included so linkage/profile mapping changes cannot
    # reuse FK. TCP offsets, candidate thresholds and identification are absent.
    converted = [profile.feedback_joint_positions(profile.joint_names, values) for values in positions]
    joints = np.asarray([[values.get(name, 0) for name in model.joint_names] for values, _ in converted])
    openings = np.asarray([opening for _, opening in converted])
    kinematics_sha256 = model.fingerprint()
    fk_key = digest({'version': STAGE_VERSION, 'synchronized': synchronized_key,
                     'kinematics': kinematics_sha256, 'endpoints': endpoints,
                     'joint_values_sha256': hashlib.sha256(joints.tobytes()).hexdigest()})
    cached = cache.get('endpoint_fk', fk_key) if cache else None
    if cached is None:
        poses = np.asarray([model.snapshot(values, body_names=endpoints)['body_poses_world']
                            for values, _ in converted])
        if cache:
            cache.put('endpoint_fk', fk_key, {'body_names': endpoints}, poses=poses)
    else:
        poses = cached[1]['poses']
    snapshots = {}
    def snapshot(index):
        if index not in snapshots:
            snapshots[index] = model.snapshot(converted[index][0])
        return snapshots[index]
    recorder = PlacementRecorder(None, settings, path.name, body_names=endpoints,
                                 joint_names=model.joint_names, snapshot_provider=snapshot)
    for timestamp, pose, joint, opening in zip(times, poses, joints, openings):
        recorder.append_sample(timestamp, pose, joint, opening)
    events = recorder.finish()
    from .task_config import EVENT_TYPES
    episode = {'episode_id': path.name, 'source': metadata['input_parquet'],
               'sha256': metadata['input_sha256'], 'capture_directory': str(path),
               'capture_sha256': source_digests, 'frames': len(times), 'event_count': len(events),
               'feedback_quality': quality,
               'default_excluded_count': sum(bool(event['exclusion_reasons']) for event in events),
               'candidate_counts': {kind: sum(event['event_type'] == kind for event in events)
                                    for kind in EVENT_TYPES[detector.type]},
               'sample_start_s': float(times[0]), 'sample_end_s': float(times[-1]),
               'stage_keys': {'synchronized': synchronized_key, 'endpoint_fk': fk_key}}
    if actual_placement is not None:
        episode['robot_placement_actual'] = actual_placement
        episode['robot_kinematics_sha256'] = kinematics_sha256
    return {'events': events, 'episode': episode, 'reference': snapshot(0),
            'kinematics_sha256': declared_kinematics_sha256,
            'cache_hits': {'synchronized': sync_hit, 'endpoint_fk': cached is not None}}


_WORKER = None


def initialize_worker(usd, settings, store_root):
    global _WORKER
    from .extraction import prepare_extraction
    from src.simulation_config.robot_profile import profile_for_settings
    _, _, _, model = prepare_extraction(usd, settings=settings)
    _WORKER = (settings, model, profile_for_settings(settings), StageCache(store_root))


def run_episode_worker(job):
    path, alignment_max_gap_s = job
    settings, model, profile, cache = _WORKER
    return extract_episode(path, settings, model, profile, cache,
                           alignment_max_gap_s=alignment_max_gap_s)
