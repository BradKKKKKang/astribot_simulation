"""Three-valued task predicates over explicit offline or runtime observations.

Detected motion candidates and visual labels are not contact observations.
Missing evidence yields unknown; only an explicit contact observation can
satisfy a contact predicate. This module never calls JEV or a simulator.
"""
import math


def _fields(value, allowed, label):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f'{label}: unknown fields or non-object')


def validate_evaluation_config(scene, task, success):
    from src.simulation_config.scene_profile import validate_scene_frames
    validate_scene_frames(scene)
    events = task.get('events', {})
    stages = task.get('stages', {})
    if not isinstance(events, dict) or not isinstance(stages, dict):
        raise ValueError('task.events and task.stages must be named objects')
    if 'instruction' in task and (not isinstance(task['instruction'], str) or not task['instruction'].strip()):
        raise ValueError('task.instruction must be a nonempty string')
    order = task.get('stage_order')
    if order is not None and (not isinstance(order, list) or not all(isinstance(s, str) for s in order)
                              or len(order) != len(set(order)) or set(order) != set(stages)):
        raise ValueError('task.stage_order must list every stage exactly once')
    roles = {role for obj in scene.get('objects', {}).values() for role in obj.get('roles', [])}
    for name, selector in events.items():
        _fields(selector, {'event_type', 'side', 'object', 'role', 'exclude_initial'}, f'task event {name}')
        if selector.get('event_type') not in ('grasp', 'release', 'press', 'dwell'):
            raise ValueError('task event requires a supported event_type')
        if 'side' in selector and selector['side'] not in ('left', 'right'):
            raise ValueError('task event side must be left or right')
        if 'object' in selector and selector['object'] not in scene.get('objects', {}):
            raise ValueError('task event references an unknown scene object')
        if 'role' in selector and selector['role'] not in roles:
            raise ValueError('task event references an unknown scene role')
        if 'exclude_initial' in selector and type(selector['exclude_initial']) is not bool:
            raise ValueError('task event exclude_initial must be boolean')
    for name, stage in stages.items():
        _fields(stage, {'events', 'ordered', 'success'}, f'task stage {name}')
        if 'success' in stage:
            if set(stage) != {'success'} or order is None:
                raise ValueError('predicate stages require stage_order and only success')
            # Stage criteria cannot recursively reference other stages.
            validate_evaluation_config(scene, {'events': events}, stage['success'])
            if not stage['success'].get('predicates'):
                raise ValueError('stage.success requires predicates')
            continue
        if (not isinstance(stage.get('events'), list) or not stage['events']
                or any(event not in events for event in stage['events'])
                or type(stage.get('ordered', True)) is not bool):
            raise ValueError('task stage requires declared events and boolean ordered')
    _fields(success, {'name', 'mode', 'predicates'}, 'success')
    if success.get('mode', 'all') not in ('all', 'any'):
        raise ValueError('success.mode must be all or any')
    predicates = success.get('predicates', [])
    if not isinstance(predicates, list):
        raise ValueError('success.predicates must be a list')
    ids = []
    for predicate in predicates:
        _fields(predicate, {'id', 'kind', 'event', 'stage', 'min_count', 'max_count', 'region', 'object', 'evidence', 'expected',
                            'relative_to', 'point', 'min', 'max', 'min_axis_dot'}, 'success predicate')
        if not isinstance(predicate.get('id'), str) or not predicate['id']:
            raise ValueError('success predicate id is required')
        ids.append(predicate['id'])
        kind = predicate.get('kind')
        if kind not in ('event_count', 'stage', 'in_region', 'object_in_region', 'object_relative_region', 'evidence', 'contact'):
            raise ValueError('unknown success predicate kind')
        if kind in ('event_count', 'in_region') and predicate.get('event') not in events:
            raise ValueError('success predicate requires a declared task event')
        if kind == 'stage' and predicate.get('stage') not in stages:
            raise ValueError('success predicate requires a declared task stage')
        if kind in ('in_region', 'object_in_region') and predicate.get('region') not in scene.get('regions', {}):
            raise ValueError('success predicate references unknown region')
        if kind == 'object_in_region' and predicate.get('object') not in scene.get('objects', {}):
            raise ValueError('success predicate references unknown scene object')
        if kind == 'object_relative_region':
            if any(predicate.get(key) not in scene.get('objects', {}) for key in ('object', 'relative_to')):
                raise ValueError('relative predicate requires two declared objects')
            if predicate['object'] == predicate['relative_to']:
                raise ValueError('relative predicate requires distinct objects')
            from src.simulation_config.isaac_scene_config import vector
            for key in ('point', 'min', 'max'):
                vector(predicate.get(key), 3, 'relative predicate.' + key)
            if any(a > b for a, b in zip(predicate['min'], predicate['max'])):
                raise ValueError('relative predicate min must not exceed max')
            dot = predicate.get('min_axis_dot', -1.)
            if type(dot) not in (int, float) or not math.isfinite(dot) or not -1 <= dot <= 1:
                raise ValueError('min_axis_dot must be in [-1, 1]')
        if kind in ('contact', 'evidence') and (not isinstance(predicate.get('evidence'), str) or not predicate['evidence']):
            raise ValueError('contact/evidence predicates require an explicit observation key')
        for bound in ('min_count', 'max_count'):
            if bound in predicate and (type(predicate[bound]) is not int or predicate[bound] < 0):
                raise ValueError('event count bounds must be nonnegative integers')
        if predicate.get('max_count', math.inf) < predicate.get('min_count', 1):
            raise ValueError('max_count must be at least min_count')
        if 'expected' in predicate and type(predicate['expected']) is not bool:
            raise ValueError('evidence expected must be boolean')
    if len(ids) != len(set(ids)):
        raise ValueError('success predicate IDs must be unique')


def _match(event, selector, scene):
    if event.get('analysis_selected') is False:
        return False
    if event.get('event_type') != selector['event_type']:
        return False
    if 'side' in selector and event.get('side') != selector['side']:
        return False
    if selector.get('exclude_initial') and (event.get('initial_phase') or event.get('initial_position')
            or event.get('is_initial') or any(reason in ('initial_pose', 'initial_time')
                                               for reason in event.get('exclusion_reasons', []))):
        return False
    if 'object' not in selector and 'role' not in selector:
        return True
    identification = event.get('identification') or {}
    if identification.get('status') in ('unknown', 'failed', 'pending', 'not_requested'):
        return None
    key = identification.get('candidate_key') or event.get('object')
    if key is None:
        return None
    if key not in scene.get('objects', {}):
        # A readable fixed-object label can identify a unique configured object.
        matches = [name for name, obj in scene.get('objects', {}).items() if obj.get('label') == key]
        if len(matches) != 1:
            return None
        key = matches[0]
    if 'object' in selector and key != selector['object']:
        return False
    return 'role' not in selector or selector['role'] in scene['objects'][key].get('roles', [])


def evaluate_success(events, scene, task, success, observations=None):
    """Return status and auditable per-predicate evidence for one episode.

    ``observations`` is a mapping from evidence key to an object containing
    boolean ``value`` and nonempty ``source``. Contact also requires
    ``kind: contact``. ``events_complete`` (boolean, default False) declares
    whether absence can be treated as failure instead of missing evidence.
    ``objects`` maps scene object IDs to measured ``pose_world`` / ``pose_base``
    and a nonempty ``source``; configured spawn poses are never evidence.
    """
    validate_evaluation_config(scene, task, success)
    observations = {} if observations is None else observations
    if not isinstance(observations, dict) or type(observations.get('events_complete', False)) is not bool:
        raise ValueError('observations must be an object with boolean events_complete')
    complete = observations.get('events_complete', False)
    matches = {name: [(event, _match(event, {'exclude_initial': (task.get('event_rules') or {}).get('exclude_initial', False),
                                          **selector}, scene)) for event in events]
               for name, selector in task.get('events', {}).items()}
    results = []
    for predicate in success.get('predicates', []):
        kind, status, evidence = predicate['kind'], 'unknown', {}
        if kind == 'event_count':
            selected = [event for event, match in matches[predicate['event']] if match is True]
            unknown = sum(match is None for _, match in matches[predicate['event']])
            count = len(selected)
            low, high = predicate.get('min_count', 1), predicate.get('max_count', math.inf)
            evidence = {'event_ids': [event.get('event_id') for event in selected], 'count': count,
                        'unresolved_candidates': unknown, 'events_complete': complete}
            if count > high:
                status = 'failed'
            elif complete and not unknown:
                status = 'passed' if low <= count <= high else 'failed'
            elif count >= low and high == math.inf:
                status = 'passed'
            elif complete and count + unknown < low:
                status = 'failed'
        elif kind == 'stage':
            stage = task['stages'][predicate['stage']]
            if 'success' in stage or predicate['stage'] in observations.get('stage_results', {}):
                result = observations.get('stage_results', {}).get(predicate['stage'], {})
                status = result.get('status', 'unknown')
                evidence = result
                results.append({'id': predicate['id'], 'kind': kind, 'status': status, 'evidence': evidence})
                continue
            selected, cursor, uncertain = [], -math.inf, False
            for name in stage['events']:
                candidates = [event for event, match in matches[name] if match is True]
                uncertain = uncertain or any(match is None for _, match in matches[name])
                if stage.get('ordered', True):
                    valid = [event for event in candidates if type(event.get('timestamp_s')) in (int, float)
                             and math.isfinite(event['timestamp_s']) and event['timestamp_s'] > cursor]
                    uncertain = uncertain or any(type(event.get('timestamp_s')) not in (int, float)
                                                   or not math.isfinite(event['timestamp_s']) for event in candidates)
                    chosen = min(valid, key=lambda event: event['timestamp_s']) if valid else None
                else:
                    chosen = next((event for event in candidates if event not in selected), None)
                if chosen is None:
                    break
                selected.append(chosen)
                if stage.get('ordered', True):
                    cursor = chosen['timestamp_s']
            status = ('passed' if len(selected) == len(stage['events']) else
                      'failed' if complete and not uncertain else 'unknown')
            evidence = {'event_ids': [event.get('event_id') for event in selected], 'events_complete': complete}
        elif kind == 'in_region':
            region = scene['regions'][predicate['region']]
            selected = [event for event, match in matches[predicate['event']] if match is True]
            evidence = {'event_ids': [event.get('event_id') for event in selected], 'region': predicate['region'],
                        'frame': region['frame'], 'positions': []}
            missing, outside = False, False
            for event in selected:
                from src.simulation_config.scene_profile import frame_coordinates
                position = frame_coordinates(event, scene, region['frame'])
                if position is None:
                    missing = True
                    continue
                evidence['positions'].append(position)
                outside |= any(v < lo or v > hi for v, lo, hi in zip(position, region['min'], region['max']))
            unknown = any(match is None for _, match in matches[predicate['event']])
            status = ('failed' if outside else 'passed' if selected and complete and not missing and not unknown
                      else 'failed' if not selected and complete and not unknown else 'unknown')
        elif kind == 'object_in_region':
            region = scene['regions'][predicate['region']]
            objects = observations.get('objects', {})
            state = objects.get(predicate['object']) if isinstance(objects, dict) else None
            evidence = {'object': predicate['object'], 'region': predicate['region'], 'frame': region['frame']}
            if isinstance(state, dict) and isinstance(state.get('source'), str) and state['source']:
                from src.simulation_config.scene_profile import frame_coordinates
                position = frame_coordinates({'tcp_pose_' + frame: state.get('pose_' + frame)
                                              for frame in ('world', 'base')}, scene, region['frame'])
                if position is not None:
                    inside = all(lo <= value <= hi for value, lo, hi in zip(position, region['min'], region['max']))
                    status = 'passed' if inside == predicate.get('expected', True) else 'failed'
                    evidence.update(position=position, source=state['source'])
        elif kind == 'object_relative_region':
            from src.simulation_config.poses import compose_pose, relative_pose, rotate
            objects = observations.get('objects', {})
            states = [objects.get(predicate[key]) for key in ('object', 'relative_to')] if isinstance(objects, dict) else []
            evidence = {'object': predicate['object'], 'relative_to': predicate['relative_to']}
            def measured(state):
                if not isinstance(state, dict) or not isinstance(state.get('source'), str) or not state['source']:
                    return False
                pose = state.get('pose_world')
                return (isinstance(pose, (list, tuple)) and len(pose) == 7
                        and all(type(v) in (int, float) and math.isfinite(v) for v in pose)
                        and abs(sum(v*v for v in pose[3:]) - 1.) < 1e-4)
            if len(states) == 2 and all(measured(state) for state in states):
                relative = relative_pose(states[1]['pose_world'], states[0]['pose_world'])
                point = compose_pose(relative, predicate['point'] + [1, 0, 0, 0])[:3]
                axis_dot = float(rotate(relative[3:], [0, 0, 1])[2])
                inside = all(lo <= v <= hi for v, lo, hi in zip(point, predicate['min'], predicate['max']))
                inside = inside and axis_dot >= predicate.get('min_axis_dot', -1.)
                status = 'passed' if inside == predicate.get('expected', True) else 'failed'
                evidence.update(point=point.tolist(), axis_dot=axis_dot, sources=[s['source'] for s in states])
        else:
            key = predicate['evidence']
            observation = observations.get(key)
            evidence = {'observation_key': key}
            if (isinstance(observation, dict) and type(observation.get('value')) is bool
                    and isinstance(observation.get('source'), str) and observation['source']
                    and (kind != 'contact' or observation.get('kind') == 'contact')):
                evidence.update(value=observation['value'], source=observation['source'])
                status = 'passed' if observation['value'] == predicate.get('expected', True) else 'failed'
        results.append({'id': predicate['id'], 'kind': kind, 'status': status, 'evidence': evidence})
    statuses = [item['status'] for item in results]
    if not statuses:
        overall = 'unknown'
    elif success.get('mode', 'all') == 'all':
        overall = 'failed' if 'failed' in statuses else 'unknown' if 'unknown' in statuses else 'passed'
    else:
        overall = 'passed' if 'passed' in statuses else 'unknown' if 'unknown' in statuses else 'failed'
    return {'status': overall, 'predicates': results,
            'reason': 'no success predicates configured' if not statuses else 'evaluated explicit evidence'}


def evaluate_task(events, scene, task, observations=None, completed_stages=None):
    """Evaluate ordered stages; the caller owns and clears episode history on reset.

    Completed stages retain their measured evidence. At most one new stage can
    finish per snapshot, so a final pose alone cannot prove an earlier pickup.
    """
    import copy
    if observations is not None and not isinstance(observations, dict):
        raise ValueError('observations must be an object')
    validate_evaluation_config(scene, task, task['success'])
    observations = copy.deepcopy(observations if observations is not None else {})
    observations.pop('stage_results', None)  # Stage progress is evaluator-owned.
    history = {} if completed_stages is None else completed_stages
    stage_results, active = {}, True
    for name in task.get('stage_order', []):
        if name in history:
            stage_results[name] = copy.deepcopy(history[name])
            continue
        stage = task['stages'][name]
        if not active:
            stage_results[name] = {'status': 'unknown', 'reason': 'waiting for preceding stage'}
            continue
        criteria = stage.get('success', {'predicates': [{'id': name, 'kind': 'stage', 'stage': name}]})
        result = evaluate_success(events, scene, task, criteria, observations)
        stage_results[name] = result
        if result['status'] == 'passed':
            history[name] = copy.deepcopy(result)
        active = False
    observations['stage_results'] = stage_results
    result = evaluate_success(events, scene, task, task['success'], observations)
    if stage_results:
        result['stages'] = stage_results
    return result
