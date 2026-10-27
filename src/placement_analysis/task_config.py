"""Task configuration layered over the existing flat detector/TCP contract."""
import copy
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DETECTOR_CONFIG = ROOT / 'config/placement/press_cube.json'
EVENT_TYPES = {'gripper': ('grasp', 'release'), 'press': ('press',)}


def validate_task_config(settings, *, require_ready=True, validate_identification=True):
    if not isinstance(settings, dict):
        raise ValueError('task configuration must be a JSON object')
    settings = copy.deepcopy(settings)
    from .experiment_config import check_fields
    from .events import DetectorConfig
    check_fields(settings, set(DetectorConfig.__dataclass_fields__) | {
        'task', 'dataset_root', 'base_body', 'tcp', 'tcp_calibration',
        'event_rules', 'identification', '_resolved_sections', '_fingerprints'}, 'task configuration')
    task = settings.get('task', {})
    if not isinstance(task, dict):
        raise ValueError('task must be an object')
    check_fields(task, {'name', 'dataset', 'status'}, 'task')
    for field in ('name', 'dataset'):
        value = task.get(field)
        if not isinstance(value, str) or not value.strip() or Path(value).name != value or value in ('.', '..'):
            raise ValueError(f'task.{field} must be a nonempty directory name')
    if task.get('status') not in ('ready', 'needs_annotation'):
        raise ValueError('task.status must be ready or needs_annotation')
    operation = settings.get('type')
    if not (isinstance(operation, str) and operation in EVENT_TYPES) and not (
            operation is None and task['status'] == 'needs_annotation'):
        raise ValueError('type must be gripper or press (null only while awaiting annotation)')
    if 'motion_detector' in settings or (isinstance(settings.get('event_rules'), dict)
                                         and 'motion_detector' in settings['event_rules']):
        raise ValueError('use type as the single candidate selector; remove motion_detector')
    identification = settings.get('identification', {})
    if not isinstance(identification, dict):
        raise ValueError('identification must be an object')
    check_fields(identification, {'enabled', 'object', 'camera', 'frame_offset', 'question_id',
        'question', 'choices', 'choice_order', 'model', 'timeout_s', 'endpoint', 'decision'}, 'identification')
    if type(identification.get('enabled', True)) is not bool:
        raise ValueError('identification.enabled must be boolean')
    if not isinstance(settings.get('dataset_root'), str) or not settings['dataset_root']:
        raise ValueError('dataset_root is required')
    if not settings.get('base_body') or set(settings.get('tcp', {})) != {'left', 'right'}:
        raise ValueError('shared robot base_body and both TCPs are required')
    # Only a credential environment variable is supported; never snapshot secrets.
    def check_secrets(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if any(token in key.lower() for token in ('api_key', 'authorization', 'password', 'secret', 'token')):
                    raise ValueError('credentials must be supplied through DECISION_API_KEY, not task configuration')
                check_secrets(child)
        elif isinstance(value, list):
            for child in value:
                check_secrets(child)
    check_secrets(settings)
    if task['status'] == 'needs_annotation':
        if require_ready:
            raise ValueError(f"{task['name']}: 待标注 (needs_annotation); complete type and task rules first")
        return settings
    from .events import DetectorConfig
    DetectorConfig(**{k: v for k, v in settings.items() if k in DetectorConfig.__dataclass_fields__})
    rules = settings.get('event_rules')
    if not isinstance(rules, dict) or type(rules.get('exclude_initial')) is not bool:
        raise ValueError('event_rules.exclude_initial must be boolean')
    check_fields(rules, {'exclude_initial'}, 'event_rules')
    if not validate_identification:
        return settings
    if 'event_types' in identification:
        raise ValueError('identification event types are defined by type; remove identification.event_types')
    if identification.get('enabled') is False:
        if not isinstance(identification.get('object'), str) or not identification['object'].strip():
            raise ValueError('identification.object is required when model identification is disabled')
        return settings
    if not isinstance(identification.get('camera'), str) or not identification['camera']:
        raise ValueError('identification.camera is required')
    if type(identification.get('frame_offset')) is not int:
        raise ValueError('identification.frame_offset must be an integer number of video frames')
    for field in ('question_id', 'question', 'model'):
        if not isinstance(identification.get(field), str) or not identification[field].strip():
            raise ValueError(f'identification.{field} is required')
    # Substitute the event hand only; no ordering or expected-answer context is injected.
    try:
        identification['question'].format(side='左手')
    except (KeyError, ValueError, IndexError) as error:
        raise ValueError('question supports only the {side} placeholder') from error
    choices = identification.get('choices')
    if not isinstance(choices, dict) or len(choices) < 2 or not all(
            isinstance(k, str) and k and isinstance(v, str) and v.strip() for k, v in choices.items()):
        raise ValueError('identification.choices needs at least two nonempty keys and readable names')
    order = identification.get('choice_order', list(choices))
    if (not isinstance(order, list) or not all(isinstance(key, str) for key in order)
            or len(order) != len(set(order)) or set(order) != set(choices)):
        raise ValueError('identification.choice_order must list every choice exactly once')
    identification['choices'] = {key: choices[key] for key in order}
    timeout = identification.get('timeout_s')
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('identification.timeout_s must be finite and positive')
    return settings


def load_task_config(path, *, require_ready=True, validate_identification=False):
    from .experiment_config import resolve_experiment
    settings = resolve_experiment(path, require_ready=require_ready)['settings']
    return validate_task_config(settings, require_ready=require_ready,
                                validate_identification=validate_identification)
