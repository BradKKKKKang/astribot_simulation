"""Resolve explicit modular JSON experiments without runtime or network effects."""
import copy
import hashlib
import json
import math
from pathlib import Path

from .robot_profile import RobotProfile, default_robot_profile

SECTIONS = ('robot', 'scene', 'task', 'candidates', 'success', 'jev', 'source', 'runtime', 'output')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate configuration field: {key}')
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).expanduser().read_text(), object_pairs_hook=_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'nonfinite JSON value: {value}')))


def check_fields(value, allowed, label):
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be an object')
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f'{label}: unknown fields: {sorted(unknown)}')


def check_credentials(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if any(word in key.lower() for word in ('api_key', 'authorization', 'password', 'secret', 'token')):
                raise ValueError('credentials must be supplied through DECISION_API_KEY, not configuration')
            check_credentials(child)
    elif isinstance(value, list):
        for item in value:
            check_credentials(item)


def _merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        # A new reference replaces the prior resource; never merge a descriptor
        # into an already expanded resource and retain its stale fields.
        result[key] = (_merge(result[key], value)
                       if isinstance(result.get(key), dict) and isinstance(value, dict) and 'ref' not in value
                       else copy.deepcopy(value))
    return result


def _section(spec, directory, seen=(), *, scene_assets=False):
    if not isinstance(spec, dict):
        raise ValueError('experiment sections must be objects or explicit ref/overrides objects')
    if 'ref' not in spec:
        result = copy.deepcopy(spec)
        if scene_assets:
            from .scene_assets import expand_scene_assets
            result = expand_scene_assets(result, directory)
        return result
    check_fields(spec, {'ref', 'overrides'}, 'profile reference')
    if not isinstance(spec['ref'], str) or not spec['ref'] or '$' in spec['ref']:
        raise ValueError('profile ref must be a literal path (environment expansion is unsupported)')
    path = (directory / spec['ref']).expanduser().resolve(strict=True)
    if path in seen:
        raise ValueError('cyclic profile reference')
    override = spec.get('overrides', {})
    if not isinstance(override, dict):
        raise ValueError('profile overrides must be an object')
    result = _merge(_section(read_json(path), path.parent, (*seen, path), scene_assets=scene_assets), override)
    if scene_assets:
        from .scene_assets import expand_scene_assets
        result = expand_scene_assets(result, directory)
    return result


def _positive(value, label, zero=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f'{label} must be finite and {"nonnegative" if zero else "positive"}')


def validate_scene(scene):
    check_fields(scene, {'name', 'objects', 'regions', 'frames', 'simulation'}, 'scene')
    from .scene_profile import validate_scene_frames
    validate_scene_frames(scene)
    objects, regions = scene.get('objects', {}), scene.get('regions', {})
    if not isinstance(objects, dict) or not isinstance(regions, dict):
        raise ValueError('scene objects and regions must be named objects')
    from .scene_assets import validate_scene_objects
    validate_scene_objects(scene)
    for name, region in regions.items():
        check_fields(region, {'frame', 'min', 'max'}, f'scene.regions.{name}')
        if region.get('frame') not in {'world', 'base', *scene.get('frames', {})}:
            raise ValueError('region frame must be world, base or a declared calibrated scene frame')
        frame = region['frame']
        while frame not in ('world', 'base'):
            owner = frame.partition('.')[0]
            if objects.get(owner, {}).get('asset', {}).get('physics', {}).get('body_type') == 'dynamic':
                raise ValueError('region frame cannot depend on a dynamic object initial frame')
            frame = scene['frames'][frame]['parent']
        for field in ('min', 'max'):
            vector = region.get(field)
            if not isinstance(vector, list) or len(vector) != 3 or not all(type(v) in (int, float) and math.isfinite(v) for v in vector):
                raise ValueError('region bounds must contain three finite coordinates')
        if any(a > b for a, b in zip(region['min'], region['max'])):
            raise ValueError('region min must not exceed max')


def validate_sections(sections, require_ready, *, legacy=False):
    from .candidates import DetectorConfig
    RobotProfile(sections['robot'])
    if legacy:
        validate_scene(sections['scene'])
        check_fields(sections['task'], {'name', 'dataset', 'status', 'event_rules', 'events', 'stages'}, 'task')
        if {'name', 'dataset', 'status'} - set(sections['task']):
            raise ValueError('task requires name, dataset and status')
    check_fields(sections['candidates'], set(DetectorConfig.__dataclass_fields__), 'candidates')
    for key, value in sections['candidates'].items():
        if key == 'type' or key == 'grasp_onset_threshold' and value is None:
            continue
        if key in ('larger_is_open', 'press_require_closed'):
            if type(value) is not bool:
                raise ValueError(f'candidates.{key} must be boolean')
        elif key == 'smoothing_samples':
            if type(value) is not int:
                raise ValueError('candidates.smoothing_samples must be an integer')
        elif type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'candidates.{key} must be a finite number')
    check_fields(sections['source'], {'dataset_root', 'dataset', 'column', 'metadata_policy'}, 'source')
    if 'column' in sections['source'] and (not isinstance(sections['source']['column'], str) or not sections['source']['column']):
        raise ValueError('source.column must be a nonempty string')
    if sections['source'].get('metadata_policy', 'assume_canonical') not in ('assume_canonical', 'require'):
        raise ValueError('source.metadata_policy must be assume_canonical or require')
    check_fields(sections['runtime'], {'alignment', 'replay'}, 'runtime')
    alignment = sections['runtime'].get('alignment', {})
    check_fields(alignment, {'max_gap_s'}, 'runtime.alignment')
    if 'max_gap_s' in alignment:
        _positive(alignment['max_gap_s'], 'alignment.max_gap_s')
    replay = sections['runtime'].get('replay', {})
    check_fields(replay, {'control_hz', 'move_hz', 'speed_scale', 'feedback_drain_s', 'feedback_timeout_s'}, 'runtime.replay')
    for field, value in replay.items():
        _positive(value, f'runtime.replay.{field}', zero=field == 'feedback_drain_s')
    check_fields(sections['output'], {'root', 'bundle', 'publish'}, 'output')
    for field in ('root', 'bundle'):
        if field in sections['output'] and (not isinstance(sections['output'][field], str) or not sections['output'][field]):
            raise ValueError(f'output.{field} must be a nonempty path')
    if 'publish' in sections['output'] and type(sections['output']['publish']) is not bool:
        raise ValueError('output.publish must be boolean')
    jev = sections['jev']
    check_fields(jev, {'enabled', 'object', 'request', 'decision', 'transport'}, 'jev')
    check_fields(jev.get('request', {}), {'camera', 'frame_offset', 'question_id', 'question', 'choices', 'choice_order', 'model'}, 'jev.request')
    check_fields(jev.get('decision', {}), {'policy', 'min_probability', 'min_margin'}, 'jev.decision')
    check_fields(jev.get('transport', {}), {'endpoint', 'timeout_s'}, 'jev.transport')
    request = jev.get('request', {})
    if isinstance(request.get('choices'), dict):
        order = request.setdefault('choice_order', list(request['choices']))
        if (not isinstance(order, list) or not all(isinstance(key, str) for key in order)
                or len(order) != len(set(order)) or set(order) != set(request['choices'])):
            raise ValueError('jev.request.choice_order must list every choice exactly once')
    elif 'choice_order' in request:
        raise ValueError('jev.request.choice_order requires choices')
    decision = jev.get('decision', {})
    if decision.get('policy', 'max_probability_v1') != 'max_probability_v1':
        raise ValueError('unsupported JEV decision policy')
    for field in ('min_probability', 'min_margin'):
        if field in decision:
            threshold = decision[field]
            if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError(f'JEV decision {field} must be in [0, 1]')
    if legacy:
        from src.evaluation.task_evaluation import validate_evaluation_config
        validate_evaluation_config(sections['scene'], sections['task'], sections['success'])
    # Request completeness and transport validity are checked only at identification.
    from .placement import validate_task_config
    validate_task_config(normalized_settings(sections), require_ready=require_ready, validate_identification=False)


def normalized_settings(sections):
    robot, task, jev = sections['robot'], sections['task'], sections['jev']
    identification = copy.deepcopy(jev.get('request', {}))
    identification.update(jev.get('transport', {}))
    identification['enabled'] = jev.get('enabled', True)
    if 'object' in jev:
        identification['object'] = jev['object']
    identification['decision'] = copy.deepcopy(jev.get('decision', {'policy': 'max_probability_v1'}))
    return {**copy.deepcopy(sections['candidates']),
            'base_body': robot['base_body'], 'tcp': copy.deepcopy(robot['tcp']),
            'tcp_calibration': robot.get('tcp_calibration', ''),
            'task': {key: task[key] for key in ('name', 'dataset', 'status')},
            'dataset_root': sections['source'].get('dataset_root'),
            'event_rules': copy.deepcopy(task.get('event_rules')),
            'identification': identification}


def _legacy_sections(value):
    from .candidates import DetectorConfig
    robot = default_robot_profile().as_dict()
    for field in ('base_body', 'tcp', 'tcp_calibration'):
        if field in value:
            robot[field] = copy.deepcopy(value[field])
    identification = value.get('identification', {})
    return {'robot': robot, 'scene': {'objects': {}, 'regions': {}},
            'task': {**copy.deepcopy(value['task']), 'event_rules': copy.deepcopy(value.get('event_rules'))},
            'candidates': {k: copy.deepcopy(v) for k, v in value.items() if k in DetectorConfig.__dataclass_fields__},
            'success': {'predicates': []},
            'jev': {'enabled': identification.get('enabled', True),
                    **({'object': identification['object']} if 'object' in identification else {}),
                    'request': {k: copy.deepcopy(v) for k, v in identification.items() if k in ('camera', 'frame_offset', 'question_id', 'question', 'choices', 'choice_order', 'model')},
                    'decision': copy.deepcopy(identification.get('decision', {'policy': 'max_probability_v1'})),
                    'transport': {k: identification[k] for k in ('timeout_s', 'endpoint') if k in identification}},
            'source': {'dataset_root': value['dataset_root']}, 'runtime': {}, 'output': {}}


def _placement_sections(sections):
    """Adapt modular authoring to the existing immutable capture/settings contract."""
    candidates = copy.deepcopy(sections['candidates'])
    selection = candidates.pop('selection', {'exclude_initial': True})
    return {**sections, 'candidates': candidates,
            'task': {'name': sections['name'], 'status': sections['status'],
                     'dataset': sections['source'].get('dataset'), 'event_rules': selection}}


def _validate_evaluation_runtime(runtime):
    check_fields(runtime, {'simulation', 'evaluation'}, 'evaluation.runtime')
    simulation = runtime.get('simulation', {})
    check_fields(simulation, {'dt', 'decimation'}, 'runtime.simulation')
    if 'dt' in simulation:
        _positive(simulation['dt'], 'runtime.simulation.dt')
    if 'decimation' in simulation and (type(simulation['decimation']) is not int or simulation['decimation'] < 1):
        raise ValueError('runtime.simulation.decimation must be a positive integer')
    evaluation = runtime.get('evaluation', {})
    check_fields(evaluation, {'max_steps'}, 'runtime.evaluation')
    if 'max_steps' in evaluation and (type(evaluation['max_steps']) is not int or evaluation['max_steps'] < 1):
        raise ValueError('runtime.evaluation.max_steps must be a positive integer')


def resolve_experiment(path, *, require_ready=True):
    path = Path(path).expanduser().resolve(strict=True)
    value = read_json(path)
    check_credentials(value)
    if not isinstance(value, dict):
        raise ValueError('experiment must be a JSON object')
    version = value.get('schema_version')
    if version is not None and (type(version) is not int or version not in (1, 2, 3)):
        raise ValueError('unsupported experiment schema_version')
    if version in (2, 3):
        kind = value.get('kind')
        if kind not in ('placement_test', 'evaluation'):
            raise ValueError('experiment.kind must be placement_test or evaluation')
        names = (('robot', 'candidates', 'jev', 'source', 'runtime', 'output')
                 if kind == 'placement_test' else ('robot', 'scene', 'task', 'output'))
        if version == 3:
            names += ('robot_placement',)
            if kind == 'evaluation':
                names += ('success', 'runtime')
        allowed = {'schema_version', 'kind', 'name', *names}
        if kind == 'placement_test':
            allowed.add('status')
        check_fields(value, allowed, kind)
        required = {'name', *names} - {'runtime', 'output', 'robot_placement'}
        if required - set(value):
            raise ValueError(f'missing experiment sections: {sorted(required - set(value))}')
        name = value['name']
        if not isinstance(name, str) or not name.strip() or Path(name).name != name or name in ('.', '..'):
            raise ValueError('experiment.name must be a nonempty directory name')
        sections = {key: _section(value.get(key, {}), path.parent, scene_assets=key == 'scene')
                    for key in names if key != 'robot_placement' or key in value}
        sections['name'] = name
        check_credentials(sections)
        if 'robot_placement' in sections:
            from .robot_placement import validate_robot_placement
            sections['robot_placement'] = validate_robot_placement(sections['robot_placement'])
        if kind == 'evaluation':
            RobotProfile(sections['robot'])
            validate_scene(sections['scene'])
            task = sections['task']
            check_fields(task, {'name', 'events', 'stages'} | ({'success'} if version == 2 else set()), 'evaluation.task')
            if not isinstance(task.get('name'), str) or not task['name']:
                raise ValueError('evaluation.task requires name')
            if version == 2:
                if 'success' not in task:
                    raise ValueError('v2 evaluation.task requires success')
                sections['success'] = task.pop('success')
            from src.evaluation.task_evaluation import validate_evaluation_config
            validate_evaluation_config(sections['scene'], task, sections['success'])
            _validate_evaluation_runtime(sections.get('runtime', {}))
            check_fields(sections['output'], {'root'}, 'evaluation.output')
            if 'root' in sections['output'] and (not isinstance(sections['output']['root'], str)
                                                or not sections['output']['root']):
                raise ValueError('evaluation.output.root must be a nonempty path')
            from .scene_profile import scene_simulation_provenance
            fingerprints = {key: fingerprint(section) for key, section in sections.items()}
            fingerprints['scene_simulation'] = fingerprint(scene_simulation_provenance(sections['scene']))
            return {'schema_version': version, 'kind': kind, 'sections': sections, 'fingerprints': fingerprints}
        sections['status'] = value.get('status', 'ready')
        detector_sections = _placement_sections(sections)
        validate_sections(detector_sections, require_ready)
    else:
        # Explicit adapters for previously supported modular v1 and flat inputs.
        if version == 1:
            check_fields(value, {'schema_version', *SECTIONS}, 'experiment')
            required = set(SECTIONS) - {'runtime', 'output'}
            if required - set(value):
                raise ValueError(f'missing experiment sections: {sorted(required - set(value))}')
            sections = {key: _section(value.get(key, {}), path.parent, scene_assets=key == 'scene') for key in SECTIONS}
        else:
            from .placement import validate_task_config
            validate_task_config(value, require_ready=require_ready, validate_identification=False)
            sections = _legacy_sections(value)
        version = 1
        check_credentials(sections)
        validate_sections(sections, require_ready, legacy=True)
        detector_sections = sections
    from .candidates import DetectorConfig
    from dataclasses import asdict
    if detector_sections['candidates'].get('type') is not None:
        detector_sections['candidates'] = asdict(DetectorConfig(**detector_sections['candidates']))
        sections['candidates'] = copy.deepcopy(detector_sections['candidates'])
        if version in (2, 3):
            sections['candidates']['selection'] = copy.deepcopy(detector_sections['task']['event_rules'])
    fingerprints = {key: fingerprint(section) for key, section in sections.items()}
    robot = sections['robot']
    fingerprints['robot_control'] = fingerprint(RobotProfile(robot).control_provenance())
    from .scene_profile import scene_simulation_provenance
    fingerprints['scene_simulation'] = fingerprint(scene_simulation_provenance(sections.get('scene', {})))
    fingerprints['robot_kinematics'] = fingerprint({key: robot[key] for key in (
        'model_path', 'root_path', 'base_body', 'tcp', 'parts', 'grippers', 'initial_joint_positions') if key in robot})
    fingerprints['robot_tcp'] = fingerprint({key: robot[key] for key in ('base_body', 'tcp')})
    jev = sections['jev']
    from .defaults import JEV_ENDPOINT as ENDPOINT
    request = copy.deepcopy(jev.get('request', {}))
    # Candidate order breaks probability ties and is semantic, unlike object key order elsewhere.
    if isinstance(request.get('choices'), dict):
        request['choices'] = [(key, request['choices'][key]) for key in request.get('choice_order', list(request['choices']))]
    fingerprints['jev_request'] = fingerprint({'enabled': jev.get('enabled', True), 'object': jev.get('object'),
                                               'request': request, 'endpoint': jev.get('transport', {}).get('endpoint', ENDPOINT)})
    fingerprints['jev_decision'] = fingerprint(jev.get('decision', {'policy': 'max_probability_v1'}))
    fingerprints['jev_transport'] = fingerprint(jev.get('transport', {}))
    fingerprints['extraction'] = fingerprint({'robot': fingerprints['robot_kinematics'],
        'candidates': fingerprint(detector_sections['candidates']), 'alignment': sections['runtime'].get('alignment', {}),
        'event_rules': detector_sections['task'].get('event_rules'),
        **({'robot_placement': fingerprints['robot_placement']} if 'robot_placement' in sections else {})})
    settings = normalized_settings(detector_sections)
    settings['_resolved_sections'], settings['_fingerprints'] = copy.deepcopy(sections), fingerprints.copy()
    return {'schema_version': version, 'kind': 'placement_test', 'sections': sections,
            'settings': settings, 'fingerprints': fingerprints}
