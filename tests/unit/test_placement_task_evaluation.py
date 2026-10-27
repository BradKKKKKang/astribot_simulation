"""Task-level semantics require actual evidence, independent of candidates."""
import pytest
from src.placement_analysis.task_evaluation import evaluate_success

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
