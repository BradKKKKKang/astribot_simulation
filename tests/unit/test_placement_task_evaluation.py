"""Task-level semantics require actual evidence, independent of candidates."""
import json
import math
from pathlib import Path

import pytest
from src.evaluation.task_evaluation import evaluate_success

SCENE = {'objects': {'pen': {'label':'笔', 'roles':['handled']}},
         'regions': {'holder': {'frame':'base', 'min':[0,0,0], 'max':[1,1,1]}}}
TASK = {'events': {'pick':{'event_type':'grasp','role':'handled'},
                   'place':{'event_type':'release','object':'pen'}},
        'stages': {'transfer':{'events':['pick','place'],'ordered':True}}}


def event(kind, time, **kwargs):
    return {'event_id':str(time), 'event_type':kind, 'timestamp_s':time,
            'identification': {'candidate_key':'pen'}, **kwargs}


def evaluate(events, predicates, observations=None):
    return evaluate_success(events, SCENE, TASK, {'predicates':predicates}, observations)


def test_unconfigured_and_contact_without_evidence_are_unknown():
    assert evaluate([], [])['status'] == 'unknown'
    predicate = {'id':'contact','kind':'contact','evidence':'pen_contact'}
    events = [event('grasp', 1)]
    assert evaluate(events, [predicate])['status'] == 'unknown'
    assert evaluate(events, [predicate], {'pen_contact':{'value':True,'source':'JEV'}})['status'] == 'unknown'
    result = evaluate(events, [predicate], {'pen_contact':{'value':True,'source':'contact_sensor','kind':'contact'}})
    assert result['status'] == 'passed'
    assert result['predicates'][0]['evidence']['source'] == 'contact_sensor'


def test_ordered_task_stage_and_region_predicates():
    predicates = [{'id':'sequence','kind':'stage','stage':'transfer'},
                  {'id':'destination','kind':'in_region','event':'place','region':'holder'}]
    events = [event('grasp',1),event('release',2,tcp_pose_base=[.5,.5,.5,1,0,0,0])]
    assert evaluate(events,predicates,{'events_complete':True})['status'] == 'passed'
    events[1]['tcp_pose_base'][0] = 2
    assert evaluate(events,predicates,{'events_complete':True})['status'] == 'failed'
    reversed_events=[event('release',1),event('grasp',2)]
    assert evaluate(reversed_events,predicates[:1],{'events_complete':True})['status'] == 'failed'


def test_missing_label_pose_or_incomplete_stream_yields_unknown():
    count = {'id':'pick','kind':'event_count','event':'pick','min_count':1}
    assert evaluate([], [count])['status'] == 'unknown'
    assert evaluate([], [count], {'events_complete':True})['status'] == 'failed'
    unidentified = event('grasp',1); unidentified['identification'] = {}
    assert evaluate([unidentified], [count], {'events_complete':True})['status'] == 'unknown'
    region = {'id':'region','kind':'in_region','event':'place','region':'holder'}
    assert evaluate([event('release',1)], [region], {'events_complete':True})['status'] == 'unknown'


def test_independent_success_changes_keep_candidate_events_intact():
    events = [event('grasp',1)]
    one={'id':'count','kind':'event_count','event':'pick','min_count':1}
    two={**one,'min_count':2}
    assert evaluate(events,[one],{'events_complete':True})['status'] == 'passed'
    assert evaluate(events,[two],{'events_complete':True})['status'] == 'failed'
    assert events == [event('grasp',1)]


def test_duplicate_predicate_and_unknown_event_rejected():
    item={'id':'count','kind':'event_count','event':'pick'}
    with pytest.raises(ValueError, match='unique'):
        evaluate([], [item,item])
    with pytest.raises(ValueError, match='declared task event'):
        evaluate([], [{**item,'event':'typo'}])


def test_initial_and_uncertain_identification_do_not_satisfy_task():
    item = {'id':'count','kind':'event_count','event':'pick'}
    task = {**TASK, 'event_rules': {'exclude_initial':True}}
    initial = event('grasp',1,initial_phase=True)
    assert evaluate_success([initial],SCENE,task,{'predicates':[item]},
                            {'events_complete':True})['status'] == 'failed'
    uncertain = event('grasp',2)
    uncertain['identification']['status'] = 'unknown'
    assert evaluate([uncertain],[item],{'events_complete':True})['status'] == 'unknown'


def test_analysis_selection_mask_is_authoritative():
    event_row = event('grasp',1,analysis_selected=False)
    item = {'id':'count','kind':'event_count','event':'pick'}
    assert evaluate([event_row],[item],{'events_complete':True})['status'] == 'failed'


def test_dwell_selector_matches_actual_event_type():
    task = {'events': {'pause': {'event_type': 'dwell'}}}
    success = {'predicates': [{'id': 'paused', 'kind': 'event_count', 'event': 'pause'}]}
    result = evaluate_success([event('dwell', 1)], SCENE, task, success, {'events_complete': True})
    assert result['status'] == 'passed'
    with pytest.raises(ValueError, match='event_type'):
        evaluate_success([], SCENE, {'events': {'pause': {'event_type': 'stop'}}}, success)


def test_measured_object_region_and_missing_state_evidence():
    success = {'predicates': [{'id': 'placed', 'kind': 'object_in_region', 'object': 'pen', 'region': 'holder'}]}
    objects = {'pen': {'pose_base': [.5, .5, .5, 1, 0, 0, 0], 'source': 'rigid_body_state'}}
    evaluate_state = lambda evidence: evaluate_success([], SCENE, TASK, success, evidence)
    assert evaluate_state({'objects': objects})['status'] == 'passed'
    objects['pen']['pose_base'][0] = 2
    assert evaluate_state({'objects': objects})['status'] == 'failed'
    assert evaluate_state({})['status'] == 'unknown'
    objects['pen'].pop('source')
    assert evaluate_state({'objects': objects})['status'] == 'unknown'


@pytest.mark.parametrize('position,angle,expected', [
    ([0, 0, .068], 90, 'pen_across_holder_top'),
    ([.02, -.02, .068], -90, 'pen_across_holder_top'),
    ([0, 0, .068], 80, 'pen_across_holder_top'),
    ([0, 0, .068], 100, 'pen_across_holder_top'),
    ([0, 0, .03], 0, 'pen_inside_holder'),
    ([.06, 0, .068], 90, None),  # Beside the rim.
    ([0, 0, .10], 90, None),  # Above the rim.
    ([0, 0, -.05], 90, None),  # Flat near the bottom.
    ([0, 0, .068], 60, None),  # Too tilted for the top; tip outside insertion box.
    ([0, 0, .068], 120, None),
    ([0, 0, .16], 0, None),  # Upright, but not inserted.
    ([0, 0, .03], 180, None),  # Existing reversed insertion rejection.
])
def test_penholder_stage_two_alternatives_follow_measured_holder(position, angle, expected):
    from src.simulation_config.poses import compose_pose
    task = json.loads((Path(__file__).parents[2] / 'config/tasks/penholder.json').read_text())
    success = task['stages']['place_pen']['success']
    scene = {'objects': {'pen': {}, 'penholder': {}}}
    # Move and rotate the holder in all three axes, not just world yaw.
    holder = [.6, -.25, .811, .5, .5, .5, .5]
    half_angle = math.radians(angle) / 2
    pose = compose_pose(holder, position + [math.cos(half_angle), 0, math.sin(half_angle), 0]).tolist()
    objects = {'pen': {'pose_world': pose, 'source': 'test_physics'},
               'penholder': {'pose_world': holder, 'source': 'test_physics'}}
    result = evaluate_success([], scene, {}, success, {'objects': objects})
    assert result['status'] == ('passed' if expected else 'failed')
    assert [p['id'] for p in result['predicates'] if p['status'] == 'passed'] == ([expected] if expected else [])
    objects.pop('penholder')
    assert evaluate_success([], scene, {}, success, {'objects': objects})['status'] == 'unknown'


@pytest.mark.parametrize('bounds', [
    {'max_axis_dot': 1.1}, {'max_axis_dot': -1.1}, {'max_axis_dot': True},
    {'max_axis_dot': float('nan')}, {'max_axis_dot': float('inf')},
    {'max_axis_dot': '0.2'}, {'min_axis_dot': .3, 'max_axis_dot': .2},
])
def test_relative_region_rejects_invalid_axis_bounds(bounds):
    scene = {'objects': {'pen': {}, 'penholder': {}}}
    predicate = {'id': 'top', 'kind': 'object_relative_region', 'object': 'pen',
                 'relative_to': 'penholder', 'point': [0, 0, 0],
                 'min': [-1, -1, -1], 'max': [1, 1, 1], **bounds}
    with pytest.raises(ValueError, match='axis_dot'):
        evaluate_success([], scene, {}, {'predicates': [predicate]})
