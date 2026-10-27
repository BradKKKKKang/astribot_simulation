"""Offline, three-valued task predicates over explicit observations.

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
        _fields(stage, {'events', 'ordered'}, f'task stage {name}')
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
        _fields(predicate, {'id', 'kind', 'event', 'stage', 'min_count', 'max_count', 'region', 'evidence', 'expected'}, 'success predicate')
        if not isinstance(predicate.get('id'), str) or not predicate['id']:
            raise ValueError('success predicate id is required')
        ids.append(predicate['id'])
        kind = predicate.get('kind')
        if kind not in ('event_count', 'stage', 'in_region', 'evidence', 'contact'):
            raise ValueError('unknown success predicate kind')
        if kind in ('event_count', 'in_region') and predicate.get('event') not in events:
            raise ValueError('success predicate requires a declared task event')
        if kind == 'stage' and predicate.get('stage') not in stages:
            raise ValueError('success predicate requires a declared task stage')
        if kind == 'in_region' and predicate.get('region') not in scene.get('regions', {}):
            raise ValueError('success predicate references unknown region')
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
