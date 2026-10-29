"""Online evaluation uses measured state after reset/step without ROS or Isaac."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from threading import Event

from src.evaluation.online import OnlineEvaluator


def experiment(tmp_path):
    return {'kind': 'evaluation', 'fingerprints': {}, 'sections': {
        'name': 'measured', 'scene': {'objects': {'pen': {}},
        'regions': {'target': {'frame': 'world', 'min': [0, 0, 0], 'max': [1, 1, 1]}}},
        'task': {'name': 'move', 'events': {'press': {'event_type': 'press'}},
        'success': {'predicates': [{'id': 'position', 'kind': 'object_in_region', 'object': 'pen', 'region': 'target'}]}},
        'runtime': {'evaluation': {'max_steps': 2}}, 'output': {'root': str(tmp_path)}}}


def test_runtime_snapshot_changes_and_budget_do_not_fabricate_evidence(tmp_path):
    evaluator = OnlineEvaluator(experiment(tmp_path))
    state = {'objects': {'pen': {'pose_world': [2, 0, .5, 1, 0, 0, 0], 'source': 'physics'}},
             'events': [], 'observations': {}}
    env = SimpleNamespace(evaluation_state=lambda: state)
    assert evaluator.observe(env, reset=True) is False
    assert evaluator.result['status'] == 'failed'
    state['objects']['pen']['pose_world'][0] = .5
    assert evaluator.observe(env) is False
    assert evaluator.result['status'] == 'passed'
    state['objects'] = {}
    assert evaluator.observe(env) is True
    assert evaluator.result['status'] == 'unknown'
    path = evaluator.write_report()
    report = json.loads(path.read_text())
    assert report['steps'] == 2 and report['observations']['objects'] == {}
    assert 'events_complete' not in report['observations']
    assert report['result']['status'] == 'unknown'


def test_unsupported_runtime_state_keeps_contact_unknown(tmp_path):
    resolved = experiment(tmp_path)
    resolved['sections']['task']['success'] = {'predicates': [{'id': 'touch', 'kind': 'contact', 'evidence': 'pen_touch'}]}
    evaluator = OnlineEvaluator(resolved)
    evaluator.observe(SimpleNamespace(), reset=True)
    assert evaluator.result['status'] == 'unknown'
    evaluator.observe(SimpleNamespace(evaluation_state=lambda: {
        'observations': {'pen_touch': {'kind': 'contact', 'value': True, 'source': 'contact_sensor'}}}))
    assert evaluator.result['status'] == 'passed'


def test_penholder_stages_use_moving_holder_and_clear_history_on_reset():
    from src.simulation_config.experiment_config import resolve_experiment
    from src.simulation_config.poses import compose_pose
    from tests.unit.test_scene_assets import real_scene
    real_scene()
    root = Path(__file__).parents[2]
    resolved = resolve_experiment(root/'config/experiments/evaluation/penholder.json')
    evaluator = OnlineEvaluator(resolved)
    holder = [.45, .2, .811, 1, 0, 0, 0]
    state = {'objects': {name: {'pose_world': pose, 'source': 'test_physics'}
                        for name, pose in {'pen': [.3, -.2, .759, 1, 0, 0, 0], 'penholder': holder}.items()}}
    env = SimpleNamespace(evaluation_state=lambda: state)
    evaluator.observe(env, reset=True)
    assert evaluator.completed_stages == {}
    state['objects']['pen']['pose_world'][2] = .9
    evaluator.observe(env)
    assert set(evaluator.completed_stages) == {'pick_pen'}
    assert evaluator.result['stages']['place_pen']['status'] == 'unknown'
    # An insertion relative to the current holder works at an arbitrary XY/yaw.
    holder[:] = [.6, -.25, .811, .7071067811865476, 0, 0, .7071067811865476]
    inserted = compose_pose(holder, [0, 0, .03, 1, 0, 0, 0]).tolist()
    state['objects']['pen']['pose_world'] = inserted
    state['objects'].pop('penholder')
    evaluator.observe(env)
    assert evaluator.result['status'] == 'unknown'
    state['objects']['penholder'] = {'pose_world': holder, 'source': 'test_physics'}
    state['objects']['pen']['pose_world'] = compose_pose(holder, [.1, 0, .03, 1, 0, 0, 0]).tolist()
    evaluator.observe(env)
    assert evaluator.result['status'] == 'failed'
    state['objects']['pen']['pose_world'] = compose_pose(holder, [0, 0, .03, 0, 1, 0, 0]).tolist()
    evaluator.observe(env)
    assert evaluator.result['status'] == 'failed'
    state['objects']['pen']['pose_world'] = inserted
    evaluator.observe(env)
    assert evaluator.result['status'] == 'passed'
    state['objects']['pen']['pose_world'] = [.3, -.2, .759, 1, 0, 0, 0]
    evaluator.observe(env, reset=True)
    assert not evaluator.completed_stages
    assert evaluator.result['status'] != 'passed'


def test_factory_loop_consumes_actual_reset_and_step_snapshots(tmp_path):
    # Execute actual factory methods with a minimal non-ROS environment.
    source = Path(__file__).parents[2] / 'src/astribot_envs/astribot_envs_factory.py'
    cls = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef))
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in ('simu_env_loop', '_evaluate_state')]
    namespace = {}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), 'exec'), namespace)
    evaluator = OnlineEvaluator(experiment(tmp_path))
    samples = []
    state = {'objects': {}}
    def reset():
        samples.append('reset')
        return {}, {}
    def step(action):
        samples.append('step')
        state['objects'] = {'pen': {'pose_world': [.5, .5, .5, 1, 0, 0, 0], 'source': 'physics'}}
        return {}, 0, False, False, {}
    def read():
        samples.append('read')
        return state
    env = SimpleNamespace(reset=reset, step=step, action_space=SimpleNamespace(sample=lambda: None),
                          unwrapped=SimpleNamespace(evaluation_state=read))
    factory = SimpleNamespace(astribot_simu_env=env, evaluator=evaluator, stop_event=Event(),
                              running=lambda: True, rate=SimpleNamespace(sleep=lambda: None))
    factory._evaluate_state = lambda **kw: namespace['_evaluate_state'](factory, **kw)
    namespace['simu_env_loop'](factory)
    assert samples == ['reset', 'read', 'step', 'read', 'step', 'read']
    assert evaluator.steps == 2 and evaluator.result['status'] == 'passed'
