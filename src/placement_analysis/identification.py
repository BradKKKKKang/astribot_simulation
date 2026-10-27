"""Map feedback events to verified source video frames and identify their objects."""
from bisect import bisect_right
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import pyarrow.parquet as pq

from . import jev
from .identification_progress import IdentificationProgress
from .recorder import save_bundle
from .task_config import EVENT_TYPES, load_task_config, validate_task_config
from .source_relocation import MANIFEST as RELOCATION_MANIFEST, resolve_source_relocation, sha256

MAPPING_VERSION = 1


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_row(event_time, commands, metadata, row_count):
    """Use the last issued dataset command (zero-order hold), never replay time as video time.

    source_frame_index is relative to the selected segment in replay_parquet_v1.
    Repeated 100 Hz commands intentionally map to the same original source row.
    """
    if metadata.get('capture_schema') != 'replay_parquet_v1':
        raise ValueError('capture has no supported source-frame mapping (replay_parquet_v1 required)')
    rows = [r for r in commands if r.get('phase', 'dataset') == 'dataset']
    if not rows or not math.isfinite(event_time):
        raise ValueError('missing dataset commands or invalid event time')
    epoch = metadata['epoch_monotonic_ns']
    starts = [(r['call_start_monotonic_ns'] - epoch) / 1e9 for r in rows]
    indices = [r['source_frame_index'] for r in rows]
    if (any(b <= a for a, b in zip(starts, starts[1:]))
            or any(type(i) is not int or i < 0 for i in indices)
            or any(b < a for a, b in zip(indices, indices[1:]))):
        raise ValueError('invalid command timing or source frame ordering')
    end = (metadata['dataset_window_monotonic_ns'][1] - epoch) / 1e9
    if event_time < starts[0] or event_time > end:
        raise ValueError(f'event time {event_time} outside mapped command window [{starts[0]}, {end}]')
    index = bisect_right(starts, event_time) - 1
    command = rows[index]
    offset = metadata['replay_settings']['start_frame']
    absolute = offset + command['source_frame_index']
    limit = metadata['replay_settings'].get('end_frame')
    if not 0 <= absolute < min(row_count, limit if limit is not None else row_count):
        raise ValueError(f'original Parquet row {absolute} out of bounds')
    return absolute, command


def dataset_file(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('dataset metadata path escapes dataset root')
    return path


class EpisodeSource:
    def __init__(self, episode, settings, data_root=None):
        self.episode = episode
        if not episode.get('capture_directory'):
            raise ValueError('source-frame mapping unavailable: episode has no capture_directory')
        capture = Path(episode['capture_directory']).expanduser()
        for name, expected in episode.get('capture_sha256', {}).items():
            if name in ('run_metadata.json', 'api_commands.parquet') and sha256(capture / name) != expected:
                raise ValueError(f'capture SHA256 mismatch: {capture / name}')
        self.metadata = json.loads((capture / 'run_metadata.json').read_text())
        if self.metadata.get('status') != 'complete':
            raise ValueError('completed capture required')
        self.commands = pq.read_table(capture / 'api_commands.parquet', columns=[
            'frame_index', 'source_frame_index', 'phase', 'call_start_monotonic_ns']).to_pylist()
        if len(self.commands) != self.metadata.get('command_frames'):
            raise ValueError('incomplete API command mapping')
        expected = self.metadata.get('capture_sha256', {}).get('api_commands.parquet')
        if expected and sha256(capture / 'api_commands.parquet') != expected:
            raise ValueError('API commands SHA256 mismatch')
        original = Path(self.metadata['input_parquet']).expanduser()
        # An explicit root always wins, including when the stale original still exists.
        if data_root is None and original.is_file():
            info_path = next((p / 'meta/info.json' for p in original.parents
                              if (p / 'meta/info.json').is_file()), None)
            if info_path is None:
                raise ValueError(f'no dataset meta/info.json for {original}')
            self.root = info_path.parent.parent
            source = original
            self.info = json.loads(info_path.read_text())
        else:
            self.root = Path(data_root or settings['dataset_root']).expanduser() / settings['task']['dataset']
            info_path = self.root / 'meta/info.json'
            self.info = json.loads(info_path.read_text())
            match = re.fullmatch(r'episode_(\d+)', original.stem)
            if match is None:
                raise ValueError(f'cannot relocate source episode: {original.name}')
            number = int(match[1])
            source = dataset_file(self.root, self.info['data_path'].format(
                episode_index=number, episode_chunk=number // self.info['chunks_size']))
        expected = self.metadata.get('input_sha256')
        if not expected or (episode.get('sha256') and episode['sha256'] != expected):
            raise ValueError('bundle and capture original Parquet SHA256 mismatch')
        self.relocation = {}
        if (not source.is_file() or sha256(source) != expected) and (capture / RELOCATION_MANIFEST).is_file():
            source, self.relocation = resolve_source_relocation(
                capture, self.root, settings['task']['dataset'], expected)
        elif not source.is_file() or sha256(source) != expected:
            raise ValueError(f'original Parquet SHA256 mismatch: {source}; dataset may have been renumbered')
        self.source, self.source_sha, self.original_sha = source, sha256(source), expected
        self.info_path = info_path
        self.info_sha = sha256(info_path)
        parquet = pq.ParquetFile(source)
        self.table = parquet.read(columns=[name for name in
            ('timestamp', 'frame_index', 'episode_index', settings['identification']['camera'])
            if name in parquet.schema_arrow.names])
        self.videos = {}

    def locate(self, event, config):
        row_index, command = source_row(event['timestamp_s'], self.commands, self.metadata, len(self.table))
        row = self.table.slice(row_index, 1).to_pylist()[0]
        camera = config['camera']
        feature = self.info.get('features', {}).get(camera, {})
        if feature.get('dtype') != 'video':
            raise ValueError(f'{camera}: video feature missing in dataset metadata')
        number = row['episode_index']
        video = dataset_file(self.root, self.info['video_path'].format(
            episode_index=number, episode_chunk=number // self.info['chunks_size'], video_key=camera))
        timestamp = float(row['timestamp'])
        # Some datasets carry an explicit per-row video reference and timestamp.
        reference = row.get(camera)
        if isinstance(reference, dict):
            if 'path' in reference:
                video = dataset_file(self.root, reference['path'])
            if 'timestamp' in reference:
                timestamp = float(reference['timestamp'])
        if not video.is_file():
            raise FileNotFoundError(f'missing source video: {video}')
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError('invalid original video timestamp')
        if video not in self.videos:
            self.videos[video] = sha256(video)
        return {
            'mapping_version': MAPPING_VERSION, 'mapping_policy': 'last issued dataset command',
            'parquet': str(self.source), 'parquet_sha256': self.source_sha,
            'original_parquet_sha256': self.original_sha, **self.relocation,
            'parquet_row': row_index, 'dataset_frame_index': row.get('frame_index'),
            'dataset_episode_index': number, 'event_time_s': event['timestamp_s'],
            'command_index': command['frame_index'], 'source_frame_index': command['source_frame_index'],
            'command_time_s': (command['call_start_monotonic_ns'] - self.metadata['epoch_monotonic_ns']) / 1e9,
            'metadata_file': str(self.info_path), 'metadata_sha256': self.info_sha,
            'camera': camera, 'video': str(video), 'video_sha256': self.videos[video],
            'source_video_time_s': timestamp, 'frame_offset': config['frame_offset'],
        }


def video_frame(source, output=None):
    """Validate exact frame bounds; dry-run opens headers but never decodes/writes a frame."""
    import cv2
    capture = cv2.VideoCapture(source['video'])
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {source['video']}")
        fps, count = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError('video has invalid FPS or frame count')
        timestamp = source['source_video_time_s']
        if timestamp >= count / fps:
            raise ValueError(f'source video timestamp {timestamp} out of bounds')
        index = math.floor(timestamp * fps + .5) + source['frame_offset']
        if not 0 <= index < count:
            raise ValueError(f'video frame {index} out of bounds [0, {count})')
        details = dict(video_fps=fps, video_frame_count=count, video_frame_index=index,
                       video_time_s=index / fps)
        if output is not None:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise ValueError(f'cannot seek video frame {index}')
            ok, frame = capture.read()
            actual = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
            if not ok or frame is None or actual != index:
                raise ValueError(f'cannot decode exact video frame {index}; actual={actual}')
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), frame):
                raise OSError(f'cannot save screenshot: {output}')
        return details
    finally:
        capture.release()


def parse_answer(raw, config):
    return jev.select_highest_probability(raw, config['question_id'], jev.ordered_choices(config))


def identify_bundle(output, *, task_config=None, data_root=None, dry_run=False, client=None,
                    no_progress=False, workers=1, store_root=None):
    """Persist each event checkpoint; reuse only verified identical inputs and successes."""
    output = Path(output).expanduser().resolve()
    if output.is_file():
        output = output.parent
    bundle = json.loads((output / 'placement.json').read_text())
    if bundle.get('schema_version') == 5:
        from .identification_artifacts import identify_artifacts
        return identify_artifacts(output, task_config=task_config, data_root=data_root,
                                  dry_run=dry_run, client=client, no_progress=no_progress,
                                  workers=workers, store_root=store_root)
    if (bundle.get('schema_version') not in (1, 2, 3, 4) or not isinstance(bundle.get('metadata'), dict)
            or not isinstance(bundle.get('events'), list)):
        raise ValueError('expected a schema 1–4 placement bundle')
    if bundle.get('schema_version') == 1:
        # Preserve schema-1 filtering semantics without inventing startup annotations.
        bundle['metadata']['source_schema_version'] = 1
    settings = (load_task_config(task_config, require_ready=not dry_run, validate_identification=True) if task_config else
                validate_task_config(bundle['metadata'].get('task_config', {}), require_ready=not dry_run))
    if settings['task']['status'] != 'ready':
        print(f"{settings['task']['name']}: 待标注; configuration valid, no identification performed")
        return {'pending_annotation': True}
    config = settings['identification']
    event_types = EVENT_TYPES[settings['type']]
    if config.get('enabled') is False:
        counts = dict(success=0, failed=0, skipped=len(bundle['events']), reused=0, assigned=0)
        for event in bundle['events']:
            selected = event['event_type'] in event_types and not (
                settings['event_rules']['exclude_initial']
                and (event.get('exclusion_reasons') or event.get('initial_phase')))
            counts['assigned'] += int(selected)
            if not dry_run:
                event.update(object=config['object'] if selected else None,
                             identification={'status': 'skipped', 'selection_policy': 'configured_object'})
        if not dry_run:
            bundle['metadata'].update(identification_config=config, identification_summary=counts)
            write_json(output / 'identification_config.json', config)
            save_bundle(output, bundle['events'], bundle['metadata'], bundle['reference_snapshot'])
        print(f"{settings['task']['name']}: JEV disabled; {counts['assigned']} events use configured object {config['object']}")
        return counts
    episodes = {e['episode_id']: e for e in bundle['metadata'].get('episodes', [])}
    sources, counts = {}, dict(success=0, failed=0, skipped=0, reused=0)
    api_key = os.environ.get('DECISION_API_KEY', '').strip()
    endpoint = config.get('endpoint', jev.ENDPOINT)
    client = client or (lambda payload, key, timeout: jev.ask_model(payload, key, timeout, endpoint=endpoint))

    def safe(value):
        if isinstance(value, str):
            return value.replace(api_key, '[REDACTED]') if api_key else value
        if isinstance(value, dict):
            return {safe(k): safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [safe(v) for v in value]
        return value

    selected_total = sum(event['event_type'] in event_types and not (
        settings['event_rules']['exclude_initial'] and (event.get('exclusion_reasons') or event.get('initial_phase')))
        for event in bundle['events'])
    with IdentificationProgress(selected_total, enabled=not no_progress, dry_run=dry_run,
                                redact=safe) as progress:
        progress.log(f"{'识别检查' if dry_run else '开始识别'}：{output} | {len(episodes)} episodes | "
                     f"{selected_total} 个有效事件，跳过 {len(bundle['events']) - selected_total} 个候选 | "
                     f"model={config['model']}，timeout={config['timeout_s']:g}s")
        for event in bundle['events']:
            selected = (event['event_type'] in event_types and not (
                settings['event_rules']['exclude_initial'] and (event.get('exclusion_reasons') or event.get('initial_phase'))))
            if not selected:
                counts['skipped'] += 1
                if not dry_run:
                    event.update(object=None, identification={'status': 'skipped', 'error': 'event not selected by task rules'})
                continue
            with progress.event(event['event_id'], counts):
                result = {'status': 'failed', 'candidate_key': None, 'probabilities': {}, 'screenshot': None,
                          'source': None, 'error': None, 'fingerprint': None, 'raw_response': None,
                          'selection_policy': jev.SELECTION_POLICY}
                object_name = None
                event_key = fingerprint(event['event_id'])
                checkpoint = output / 'identification/events' / f'{event_key}.json'
                try:
                    episode_id = event['episode_id']
                    if episode_id not in sources:
                        progress.stage('校验源 Parquet、capture 与命令 SHA256')
                        # Extracted bundles are grouped by episode; do not retain a whole
                        # dataset's command tables while identifying a long batch.
                        sources.clear()
                        try:
                            if episode_id not in episodes:
                                raise ValueError(f'source-frame mapping unavailable: no metadata for {episode_id}')
                            sources[episode_id] = EpisodeSource(episodes[episode_id], settings, data_root)
                            progress.log(f"{progress.current} | 源数据已核验：{sources[episode_id].source.name}")
                        except Exception as error:
                            sources[episode_id] = error
                    source = sources[episode_id]
                    if isinstance(source, Exception):
                        raise source
                    progress.stage('映射事件到原视频帧并校验视频')
                    result['source'] = source.locate(event, config)
                    result['source'].update(video_frame(result['source']))
                    frame_description = (f"frame={result['source']['video_frame_index']}，"
                                         f"video_time={result['source']['video_time_s']:.3f}s")
                    prompt = config['question'].format(side={'left': '左手', 'right': '右手'}[event['side']])
                    identity = {key: result['source'][key] for key in (
                        'mapping_version', 'parquet_sha256', 'parquet_row', 'metadata_sha256',
                        'video_sha256', 'camera', 'video_frame_index', 'frame_offset')}
                    identity.update(identification={key: config[key] for key in (
                        'camera', 'frame_offset', 'question_id', 'model')}, question=prompt,
                        choices=list(jev.ordered_choices(config).items()), endpoint=endpoint)
                    result['fingerprint'] = fingerprint(identity)
                    if dry_run:
                        counts['success'] += 1
                        progress.log(f'{progress.current} | 检查通过：{frame_description}')
                        continue
                    try:
                        previous = json.loads(checkpoint.read_text()) if checkpoint.is_file() else event
                    except (ValueError, OSError):
                        previous = {}
                    cached = previous.get('identification', {})
                    screenshot = output / 'screenshots' / f"{result['fingerprint']}.png"
                    raw_path = output / 'identification/responses' / f"{event_key}-{result['fingerprint']}.json"
                    if (cached.get('fingerprint') == result['fingerprint'] and screenshot.is_file()
                            and cached.get('screenshot_sha256') == sha256(screenshot) and raw_path.is_file()
                            and cached.get('screenshot') == str(screenshot.relative_to(output))
                            and cached.get('raw_response') == str(raw_path.relative_to(output))):
                        # Re-evaluate the raw probabilities even for old successful
                        # checkpoints. The input fingerprint describes the request,
                        # so changing selection policy needs no new model request.
                        try:
                            choice, probabilities = parse_answer(json.loads(raw_path.read_text()), config)
                        except (ValueError, OSError) as error:
                            progress.log(f'{progress.current} | 缓存概率不可用，将重新请求：{error}')
                        else:
                            result.update(status='success', candidate_key=choice, probabilities=probabilities,
                                          screenshot=str(screenshot.relative_to(output)),
                                          screenshot_sha256=cached['screenshot_sha256'],
                                          raw_response=str(raw_path.relative_to(output)))
                            event.update(object=config['choices'][choice], identification=result)
                            write_json(checkpoint, {'event_id': event['event_id'], 'object': event['object'],
                                                    'identification': result})
                            counts['reused'] += 1
                            progress.log(f"{progress.current} | 复用成功（最高概率）：{event['object']} "
                                         f"{probabilities[choice]:.2%}，{frame_description}")
                            continue
                    progress.stage(f'截取完整视频帧：{frame_description}')
                    video_frame(result['source'], screenshot)
                    result.update(screenshot=str(screenshot.relative_to(output)), screenshot_sha256=sha256(screenshot))
                    if not api_key:
                        raise ValueError('DECISION_API_KEY is not set')
                    payload = jev.build_payload(screenshot, config['model'], question_id=config['question_id'],
                                               question=prompt, choices=jev.ordered_choices(config))
                    progress.stage(f"JEV 请求：model={config['model']}，timeout={config['timeout_s']:g}s，等待响应")
                    request_started = time.monotonic()
                    raw = client(payload, api_key, config['timeout_s'])
                    progress.log(f'{progress.current} | JEV 响应已返回，用时 {time.monotonic() - request_started:.1f}s')
                    write_json(raw_path, safe(raw))
                    result['raw_response'] = str(raw_path.relative_to(output))
                    choice, probabilities = parse_answer(raw, config)
                    result.update(status='success', candidate_key=choice, probabilities=probabilities)
                    object_name = config['choices'][choice]
                    counts['success'] += 1
                except Exception as error:
                    result['error'] = safe(f'{type(error).__name__}: {error}')
                    counts['failed'] += 1
                    progress.log(f"{progress.current} | 识别失败：{result['error']}")
                if not dry_run:
                    event.update(object=object_name, identification=result)
                    write_json(checkpoint, {'event_id': event['event_id'], 'object': object_name, 'identification': result})
                    if result['status'] == 'success':
                        progress.log(f"{progress.current} | 识别成功（最高概率）：{object_name} "
                                     f"{result['probabilities'][result['candidate_key']]:.2%}；检查点已保存")
        if not dry_run:
            progress.stage('保存 placement.json、events.parquet 和配置快照')
            bundle['metadata'].update(identification_config=config, identification_summary=counts)
            write_json(output / 'identification_config.json', config)
            save_bundle(output, bundle['events'], bundle['metadata'], bundle['reference_snapshot'])
        progress.log(f'Identification {"validation" if dry_run else "complete"}: {counts} | '
                     f'elapsed={time.monotonic() - progress.started:.1f}s')
    if dry_run and counts['failed']:
        raise ValueError(f"identification validation failed for {counts['failed']} events")
    return counts
