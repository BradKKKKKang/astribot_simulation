"""Independent recognition stage over immutable placement artifacts.

Frames, model requests and decisions have separate identities. The source
candidate configuration is never replaced by a later recognition configuration.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path

from . import jev
from .artifact_store import atomic_write, json_bytes
from .bundle import load_bundle, save_bundle_revision, store_for_manifest
from .task_config import EVENT_TYPES, load_task_config, validate_task_config


def identity(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


class RecognitionFailure(RuntimeError):
    def __init__(self, error, context):
        super().__init__(str(error))
        self.context = context


class RecognitionCache:
    def __init__(self, store):
        self.store = store

    def _path(self, kind, key):
        if kind not in ('frames', 'requests') or len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('invalid recognition cache identity')
        return self.store.root / 'recognition' / kind / f'{key}.json'

    def get(self, kind, key):
        path = self._path(kind, key)
        if not path.exists():
            return None
        ref = json.loads(path.read_text())
        self.store.resolve(ref)
        return ref

    @contextmanager
    def lock(self, kind, key):
        path = self._path(kind, key).with_suffix('.lock')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def put(self, kind, key, ref):
        self.store.resolve(ref)
        atomic_write(self._path(kind, key), json_bytes(ref))


class VideoFrames:
    """One decoder per source video, used on the coordinator thread only."""
    def __init__(self):
        self.handles = {}

    def describe(self, source):
        import cv2
        path = source['video']
        if path not in self.handles:
            handle = cv2.VideoCapture(path)
            if not handle.isOpened():
                handle.release()
                raise ValueError(f'cannot open video: {path}')
            self.handles[path] = handle
        handle = self.handles[path]
        fps = handle.get(cv2.CAP_PROP_FPS)
        count = int(handle.get(cv2.CAP_PROP_FRAME_COUNT))
        timestamp = source['source_video_time_s']
        if not math.isfinite(fps) or fps <= 0 or count <= 0 or not 0 <= timestamp < count / fps:
            raise ValueError('invalid video timing or out-of-bounds source timestamp')
        index = math.floor(timestamp * fps + .5) + source['frame_offset']
        if not 0 <= index < count:
            raise ValueError('source video frame out of bounds')
        source.update(video_fps=fps, video_frame_count=count, video_frame_index=index,
                      video_time_s=index / fps)

    def png(self, source):
        import cv2
        handle = self.handles[source['video']]
        index = source['video_frame_index']
        if not handle.set(cv2.CAP_PROP_POS_FRAMES, index):
            raise ValueError(f'cannot seek video frame {index}')
        ok, frame = handle.read()
        actual = int(round(handle.get(cv2.CAP_PROP_POS_FRAMES))) - 1
        if not ok or frame is None or actual != index:
            raise ValueError(f'cannot decode exact video frame {index}; actual={actual}')
        ok, encoded = cv2.imencode('.png', frame)
        if not ok:
            raise ValueError('cannot encode evidence frame')
        return encoded.tobytes()

    def close(self):
        for handle in self.handles.values():
            handle.release()
        self.handles.clear()


def decide(raw, config, decision):
    choice, probabilities = jev.select_highest_probability(raw, config['question_id'], jev.ordered_choices(config))
    if decision.get('policy', jev.SELECTION_POLICY) != jev.SELECTION_POLICY:
        raise ValueError('unsupported identification decision policy')
    probability = decision.get('min_probability', 0.0)
    margin = decision.get('min_margin', 0.0)
    for value in (probability, margin):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('identification decision thresholds must be in [0, 1]')
    ordered = sorted(probabilities.values(), reverse=True)
    accepted = probabilities[choice] >= probability and ordered[0] - ordered[1] >= margin
    return choice if accepted else None, probabilities


def _identify_artifacts(output, *, task_config=None, data_root=None, dry_run=False,
                       client=None, no_progress=False, workers=1, store_root=None):
    from .identification import EpisodeSource, MAPPING_VERSION
    output = Path(output).expanduser().resolve()
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError('identification workers must be between 1 and 32')
    manifest = json.loads((output / 'placement.json').read_text())
    store = store_for_manifest(output / 'placement.json', manifest)
    if store_root is not None and Path(store_root).expanduser().resolve() != store.root:
        raise ValueError('identification must use the bundle artifact store')
    bundle = load_bundle(output)
    metadata = bundle['metadata']
    settings = (load_task_config(task_config, require_ready=not dry_run, validate_identification=True) if task_config else
                validate_task_config(metadata.get('task_config', {}), require_ready=not dry_run))
    if settings['task']['status'] != 'ready':
        return {'pending_annotation': True}
    config = settings['identification']
    sections = settings.get('_resolved_sections', {})
    jev_section = sections.get('jev', {})
    decision = jev_section.get('decision', config.get('decision', {}))
    endpoint = jev_section.get('transport', {}).get('endpoint', config.get('endpoint', jev.ENDPOINT))
    source_type = metadata.get('type', metadata.get('detector', {}).get('type'))
    if source_type and source_type != settings['type']:
        raise ValueError('identification cannot change the extracted candidate type; re-extract first')
    event_types = EVENT_TYPES[settings['type']]
    episodes = {row['episode_id']: row for row in metadata.get('episodes', [])}
    cache = RecognitionCache(store)
    api_key = os.environ.get('DECISION_API_KEY', '').strip()
    counts = dict(success=0, failed=0, skipped=0, reused=0, assigned=0, unknown=0)
    previous_decisions = identity([{'event_id': e['event_id'], 'object': e.get('object'),
                                    'identification': e.get('identification')} for e in bundle['events']])

    def safe(value):
        if isinstance(value, str):
            return value.replace(api_key, '[REDACTED]') if api_key else value
        if isinstance(value, dict):
            return {safe(key): safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [safe(item) for item in value]
        return value

    def infer(job):
        event, source, frame_ref = job
        prompt = config['question'].format(side={'left': '左手', 'right': '右手'}[event['side']])
        request_key = identity({'version': 1, 'image': frame_ref['sha256'], 'endpoint': endpoint,
                                'model': config['model'], 'question_id': config['question_id'],
                                'question': prompt, 'choices': list(jev.ordered_choices(config).items())})
        context = {'source': source, 'screenshot_ref': frame_ref, 'fingerprint': request_key,
                   'decision_fingerprint': identity(decision),
                   'selection_policy': decision.get('policy', jev.SELECTION_POLICY)}
        try:
            with cache.lock('requests', request_key):
                previous = event.get('identification', {})
                stored_response = (previous.get('raw_response_ref') if
                                   previous.get('fingerprint') == request_key and
                                   previous.get('status') in ('success', 'unknown') else None)
                response_ref = cache.get('requests', request_key) or stored_response
                reused = response_ref is not None
                if response_ref is None:
                    if not api_key and client is None:
                        raise ValueError('DECISION_API_KEY is not set')
                    payload = jev.build_payload(store.resolve(frame_ref), config['model'],
                                               question_id=config['question_id'], question=prompt,
                                               choices=jev.ordered_choices(config))
                    raw = (client(payload, api_key, config['timeout_s']) if client else
                           jev.ask_model(payload, api_key, config['timeout_s'], endpoint=endpoint))
                    response_ref = store.put_json(safe(raw))
                    context['raw_response_ref'] = response_ref
                    jev.select_highest_probability(raw, config['question_id'], jev.ordered_choices(config))
                    cache.put('requests', request_key, response_ref)
                context['raw_response_ref'] = response_ref
                raw = store.read_json(response_ref)
                choice, probabilities = decide(raw, config, decision)
        except Exception as error:
            raise RecognitionFailure(error, context) from error
        result = {**context, 'status': 'success' if choice is not None else 'unknown',
                  'candidate_key': choice, 'probabilities': probabilities}
        return config['choices'][choice] if choice is not None else None, result, reused

    def checkpoint(event):
        key = identity({'event_id': event['event_id'], 'request': event['identification'].get('fingerprint'),
                        'decision': event['identification'].get('decision_fingerprint')})
        ref = store.put_json({'event_id': event['event_id'], 'object': event.get('object'),
                              'identification': event['identification']})
        atomic_write(output / 'recognition' / f'{key}.json', json_bytes(ref))

    def record_failure(event, error):
        counts['failed'] += 1
        if not dry_run:
            event.update(object=None, identification={**safe(getattr(error, 'context', {})),
                         'status': 'failed', 'error': safe(f'{type(error).__name__}: {error}')})
            checkpoint(event)

    def finish_one(event, future):
        try:
            name, result, reused = future.result()
            event.update(object=name, identification=result)
            if result['status'] == 'unknown':
                counts['unknown'] += 1
            elif reused:
                counts['reused'] += 1
            else:
                counts['success'] += 1
            checkpoint(event)
        except Exception as error:
            record_failure(event, error)

    grouped = {}
    for event in bundle['events']:
        grouped.setdefault(event['episode_id'], []).append(event)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for episode_id, events in grouped.items():
            frames = VideoFrames()
            source = None
            pending = []
            try:
                for event in events:
                    selected = event['event_type'] in event_types and event.get('analysis_selected', not (
                        settings['event_rules']['exclude_initial'] and
                        (event.get('exclusion_reasons') or event.get('initial_phase'))))
                    if not selected:
                        counts['skipped'] += 1
                        if not dry_run:
                            event.update(object=None, identification={'status': 'skipped'})
                        continue
                    if config.get('enabled') is False:
                        counts['assigned'] += 1
                        if not dry_run:
                            event.update(object=config['object'], identification={
                                'status': 'skipped', 'selection_policy': 'configured_object'})
                        continue
                    try:
                        # An immutable verified frame is sufficient to change the
                        # question/model/decision without reopening source videos.
                        prior = event.get('identification', {})
                        saved_source, saved_frame = prior.get('source'), prior.get('screenshot_ref')
                        original_settings = metadata.get('task_config', {})
                        if (data_root is None and isinstance(saved_source, dict) and saved_frame
                                and saved_source.get('mapping_version') == MAPPING_VERSION
                                and settings['task']['dataset'] == original_settings.get('task', {}).get('dataset')
                                and settings['dataset_root'] == original_settings.get('dataset_root')
                                and saved_source.get('original_parquet_sha256') == episodes.get(episode_id, {}).get('sha256')
                                and saved_source.get('event_time_s') == event['timestamp_s']
                                and saved_source.get('camera') == config['camera']
                                and saved_source.get('frame_offset') == config['frame_offset']):
                            store.resolve(saved_frame)
                            if dry_run:
                                counts['success'] += 1
                                continue
                            pending.append((event, pool.submit(infer, (event, saved_source, saved_frame))))
                            if len(pending) >= workers:
                                finish_one(*pending.pop(0))
                            continue
                        if source is None:
                            if episode_id not in episodes:
                                raise ValueError(f'no capture source for {episode_id}')
                            source = EpisodeSource(episodes[episode_id], settings, data_root)
                        mapped = source.locate(event, config)
                        frames.describe(mapped)
                        if dry_run:
                            counts['success'] += 1
                            continue
                        frame_key = identity({'version': 1, 'video': mapped['video_sha256'],
                                              'frame': mapped['video_frame_index'], 'encoding': 'opencv-png'})
                        with cache.lock('frames', frame_key):
                            frame_ref = cache.get('frames', frame_key)
                            if frame_ref is None:
                                frame_ref = store.put_bytes(frames.png(mapped), 'png')
                                cache.put('frames', frame_key, frame_ref)
                        pending.append((event, pool.submit(infer, (event, mapped, frame_ref))))
                        # Bound both queued image requests and pending result memory.
                        if len(pending) >= workers:
                            finish_one(*pending.pop(0))
                    except Exception as error:
                        record_failure(event, error)
                for event, future in pending:
                    finish_one(event, future)
            finally:
                frames.close()
    if not dry_run:
        current_decisions = identity([{'event_id': e['event_id'], 'object': e.get('object'),
                                      'identification': e.get('identification')} for e in bundle['events']])
        if current_decisions != previous_decisions:
            metadata.pop('evaluation', None)
            metadata.pop('evaluation_summary', None)
            metadata.setdefault('stages', {})['evaluation'] = 'not_requested'
        metadata['identification_config'] = copy.deepcopy(jev_section or config)
        metadata['identification_summary'] = counts
        metadata.setdefault('stages', {})['identification'] = 'partial' if counts['failed'] else 'complete'
        save_bundle_revision(output, bundle['events'], metadata, bundle['reference_snapshot'],
                             store=store, display=manifest.get('display'))
    if not no_progress:
        print(f'Identification {"validation" if dry_run else "complete"}: {counts}', flush=True)
    if dry_run and counts['failed']:
        raise ValueError(f'identification validation failed for {counts["failed"]} events')
    return counts


def identify_artifacts(output, **kwargs):
    """A single coordinator owns each mutable manifest; caches remain shared."""
    if kwargs.get('dry_run', False):
        return _identify_artifacts(output, **kwargs)
    output = Path(output).expanduser().resolve()
    with (output / '.pipeline.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another pipeline stage owns this bundle') from error
        return _identify_artifacts(output, **kwargs)
