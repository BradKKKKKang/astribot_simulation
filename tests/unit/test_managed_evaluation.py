"""Deterministic protocol/physics boundary tests; no ROS or Isaac imports."""
import copy
import json
import threading
from types import SimpleNamespace

import pytest

from src.evaluation.managed import ManagedEvaluation, CAMERAS
from tests.unit.test_online_evaluation import experiment


class Environment:
    def __init__(self):
        self.steps = 0
        self.resets = []
        self.captures = []
        self.x = 2.
        self.lease = False
        self.pass_at = None

    def set_managed_lease(self, value):
        self.lease = value

    def reset(self, seed):
        self.resets.append(seed)
        self.x = 2.

    def managed_step(self, commands):
        assert self.lease
        self.steps += 1
        if self.steps == self.pass_at:
            self.x = .5

    def managed_publish_feedback(self):
        pass

    def evaluation_state(self):
        return {'objects': {'pen': {'pose_world': [self.x, .5, .5, 1., 0., 0., 0.], 'source': 'test'}}}

    def managed_snapshot(self, capture_images):
        self.captures.append((self.steps, capture_images))
        return {'joints': {name: {'joint_names': value['joint_names'],
                                 'position': [0.] * value['dof'], 'velocity': [0.] * value['dof']}
                           for name, value in self.parts.items()},
                'images': {name: {'encoding': 'jpeg', 'data_base64': 'test'} for name in CAMERAS} if capture_images else {}}


@pytest.fixture
def setup(tmp_path):
    resolved = experiment(tmp_path)
    resolved['sections']['runtime']['evaluation'].update(episodes=2, seed=42)
    resolved['sections']['robot'] = {'parts': [{'name': 'arm', 'feedback_joints': ['a', 'b']},
                                            {'name': 'gripper', 'feedback_joints': ['g']}]}
    controller = ManagedEvaluation(resolved)
    env = Environment()
    env.parts = controller.parts
    return controller, env


def request(controller, op, step=0, episode=1, **payload):
    return {'schema_version': 1, 'op': op, 'request_id': f'{op}-{episode}-{step}',
            'run_id': 'batch/test', 'episode_id': episode, 'step_id': step,
            'config_fingerprint': controller.config_fingerprint, 'payload': payload}


def batch(controller, step, **kw):
    commands = {name: {'joint_names': list(value['joint_names']), 'position': [0.] * value['dof'],
                       'velocity': [0.] * value['dof']} for name, value in controller.command_parts.items()}
    return request(controller, 'step', step, dt=.01, commands=commands, capture_images=False, **kw)


def complete(controller, env, value):
    assert controller.submit(value)['status'] == 'accepted'
    assert controller.process_one(env)
    result = controller.submit(value)
    assert result['status'] == 'completed', result
    return result['result']


def reset(controller, env, episode=1):
    return complete(controller, env, request(controller, 'reset', episode=episode, seed=41 + episode))['snapshot']


def test_paused_heartbeats_and_idempotent_steps(setup):
    controller, env = setup
    first = reset(controller, env)
    assert first['step_id'] == 0 and first['evaluation']['status'] == 'failed'
    for _ in range(20):
        assert not controller.process_one(env)
        assert controller.submit({'schema_version': 1, 'op': 'heartbeat', 'request_id': 'heart'})['status'] == 'completed'
    assert env.steps == 0
    value = batch(controller, 1)
    assert controller.submit(value)['status'] == 'accepted'
    assert controller.submit(value)['status'] == 'accepted'
    assert env.steps == 0
    controller.process_one(env)
    receipt = controller.submit(value)
    assert receipt == controller.submit(value)
    assert env.steps == 1
    snap = receipt['result']['snapshot']
    assert snap['step_id'] == 1 and snap['sim_time_s'] == .01 and snap['images'] == {}
    assert not snap['terminal']['done']  # failed is not terminal
    conflicting = copy.deepcopy(value)
    conflicting['payload']['commands']['arm']['position'][0] = 1.
    assert controller.submit(conflicting)['error']['code'] == 'request_conflict'
    assert env.steps == 1


def test_success_truncates_chunk_and_new_episode_resets_history(setup, tmp_path):
    controller, env = setup
    reset(controller, env)
    env.pass_at = 1
    snap = complete(controller, env, batch(controller, 1))['snapshot']
    assert snap['terminal'] == {'done': True, 'reason': 'success'}
    assert controller.submit(batch(controller, 2))['status'] == 'error'
    controller.evaluator.completed_stages['obsolete'] = {'status': 'passed'}
    new = reset(controller, env, 2)
    assert controller.evaluator.steps == 0 and controller.evaluator.completed_stages == {}
    assert new['snapshot_id'] != snap['snapshot_id']
    assert env.resets == [42, 43]
    stale = batch(controller, 1)
    stale['request_id'] = 'late-step'
    assert controller.submit(stale)['error']['code'] == 'stale_identity'
    complete(controller, env, batch(controller, 1, episode=2))
    terminal = complete(controller, env, batch(controller, 2, episode=2))['snapshot']
    assert terminal['terminal']['reason'] == 'timeout'
    assert env.steps == 3
    summaries = list(tmp_path.glob('run-*/summary.json'))
    summary = json.loads(summaries[0].read_text())
    assert summary['counts'] == {'success': 1, 'timeout': 1, 'aborted': 0, 'error': 0}
    assert len(list(summaries[0].parent.glob('episode-*.json'))) == 2
    report = json.loads(next(summaries[0].parent.glob('episode-000001.json')).read_text())
    assert report['seed'] == 42 and report['reset_layout']['pen']['pose_world'][0] == 2.


@pytest.mark.parametrize('mutate', [
    lambda r: r['payload']['commands'].pop('arm'),
    lambda r: r['payload']['commands']['arm']['position'].append(0),
    lambda r: r['payload']['commands']['arm']['position'].__setitem__(0, float('nan')),
    lambda r: r['payload']['commands']['arm']['joint_names'].reverse(),
    lambda r: r['payload'].__setitem__('dt', .03),
    lambda r: r.__setitem__('config_fingerprint', 'wrong'),
    lambda r: r.__setitem__('step_id', 2),
])
def test_invalid_batches_never_touch_physics(setup, mutate):
    controller, env = setup
    reset(controller, env)
    value = batch(controller, 1)
    mutate(value)
    assert controller.submit(value)['status'] == 'error'
    assert not controller.process_one(env)
    assert env.steps == 0


def test_snapshot_is_cached_at_completed_identity(setup):
    controller, env = setup
    first = reset(controller, env)
    value = request(controller, 'snapshot')
    snap = complete(controller, env, value)['snapshot']
    assert snap == first and len(env.captures) == 1
    step = complete(controller, env, batch(controller, 1))['snapshot']
    capture = complete(controller, env, request(controller, 'snapshot', 1))['snapshot']
    assert capture['snapshot_id'] == step['snapshot_id']
    assert set(capture['images']) == {'cam_high', 'cam_left_wrist', 'cam_right_wrist'}
    assert env.steps == 1


def test_stop_cancels_queued_step_and_reports_aborted(setup):
    controller, env = setup
    reset(controller, env)
    value = batch(controller, 1)
    assert controller.submit(value)['status'] == 'accepted'
    complete(controller, env, request(controller, 'stop'))
    assert controller.submit(value)['error']['code'] == 'cancelled'
    assert env.steps == 0 and not env.lease and controller.phase == 'stopped'
    assert controller.records[0]['terminal']['reason'] == 'aborted'


def test_stop_during_physics_cannot_schedule_another_step(setup):
    controller, env = setup
    reset(controller, env)
    entered, release = threading.Event(), threading.Event()
    original = env.managed_step
    def blocking(commands):
        entered.set()
        assert release.wait(2)
        original(commands)
    env.managed_step = blocking
    value = batch(controller, 1)
    assert controller.submit(value)['status'] == 'accepted'
    worker = threading.Thread(target=controller.process_one, args=(env,))
    worker.start()
    assert entered.wait(2)
    stop = request(controller, 'stop')
    assert controller.submit(stop)['status'] == 'accepted'
    release.set()
    worker.join(2)
    assert not worker.is_alive()
    assert controller.submit(value)['result']['snapshot']['terminal']['reason'] == 'aborted'
    controller.process_one(env)
    assert controller.phase == 'stopped' and env.steps == 1


def test_execution_failure_latches_error_and_blocks_continuation(setup):
    controller, env = setup
    reset(controller, env)
    def fail(commands):
        raise RuntimeError('physics failure')
    env.managed_step = fail
    value = batch(controller, 1)
    controller.submit(value)
    controller.process_one(env)
    result = controller.submit(value)
    assert result['error']['execution_uncertain'] is True
    assert controller.phase == 'error'
    assert controller.records[0]['terminal']['reason'] == 'error'
    assert controller.submit(batch(controller, 2))['status'] == 'error'


def test_replay_and_eval_leases_are_exclusive(setup):
    controller, env = setup
    acquire = request(controller, 'replay', enabled=True)
    acquire['run_id'] = 'replay-owner'
    complete(controller, env, acquire)
    assert controller.phase == 'replay' and not env.lease
    assert controller.submit(request(controller, 'reset', seed=42))['error']['code'] == 'lease_conflict'
    release = request(controller, 'replay', enabled=False)
    release['request_id'] = 'replay-release'
    release['run_id'] = 'replay-owner'
    complete(controller, env, release)
    reset(controller, env)
    acquire['request_id'] = 'replay-again'
    acquire['run_id'] = 'replay-second'
    assert controller.submit(acquire)['error']['code'] == 'lease_conflict'


def test_strict_json_and_nonfinite_rejected(setup):
    controller, _ = setup
    for value in ('{"schema_version":1,"schema_version":1}', '{"n":NaN}', '[]'):
        assert json.loads(controller.submit_json(value))['status'] == 'error'


def test_stop_cancels_reset_before_it_reaches_main_thread(setup):
    controller, env = setup
    value = request(controller, 'reset', seed=42)
    assert controller.submit(value)['status'] == 'accepted'
    complete(controller, env, request(controller, 'stop'))
    assert controller.submit(value)['error']['code'] == 'cancelled'
    assert env.resets == [] and env.steps == 0


def test_reset_failure_does_not_reuse_previous_episode_evidence(setup):
    controller, env = setup
    reset(controller, env)
    env.pass_at = 1
    complete(controller, env, batch(controller, 1))
    def fail(seed):
        raise RuntimeError('reset failed')
    env.reset = fail
    value = request(controller, 'reset', episode=2, seed=43)
    controller.submit(value)
    controller.process_one(env)
    assert controller.submit(value)['error']['execution_uncertain']
    last = controller.records[-1]
    assert last['episode_id'] == 2 and last['reset_layout'] == {}
    assert last['evidence_status'] == 'unknown' and last['terminal']['reason'] == 'error'


def test_isaac_managed_step_calls_exactly_one_physics_step_then_feedback():
    import ast
    from pathlib import Path
    path = Path(__file__).parents[2] / 'src/astribot_envs/astribot_isaaclab_env.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'AstribotIsaacLabEnv')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'managed_step')
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    calls = []
    env = SimpleNamespace(managed_evaluation=True, robot_joint_map={'arm': ['a', 'b']}, dt=.01,
        _apply_control=lambda: calls.append('apply'),
        sim=SimpleNamespace(is_playing=lambda: True, step=lambda **kw: calls.append(('physics', kw))),
        scene=SimpleNamespace(write_data_to_sim=lambda: calls.append('write'),
                              update=lambda **kw: calls.append(('update', kw))),
        managed_publish_feedback=lambda: calls.append('feedback'))
    namespace['managed_step'](env, {'arm': {'position': [1., 2.], 'velocity': [.1, .2]}})
    assert calls == ['apply', 'write', ('physics', {'render': True}), ('update', {'dt': .01}), 'feedback']
    assert env.joint_position_command_all == [1., 2.]
    calls.clear()
    env.sim.is_playing = lambda: False
    with pytest.raises(RuntimeError, match='paused'):
        namespace['managed_step'](env, {'arm': {'position': [1., 2.], 'velocity': [.1, .2]}})
    assert calls == []


def test_managed_compensation_never_depends_on_fps():
    import ast
    from pathlib import Path
    path = Path(__file__).parents[2] / 'src/astribot_envs/astribot_base_env.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'vel_compensation_ctrl')
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    for fps in (.1, 20., 100., 1000.):
        env = SimpleNamespace(managed_evaluation=True, real_time_fps=fps)
        assert namespace['vel_compensation_ctrl'](env, [1.], [10.], idx=0) == [1.]


def test_managed_launcher_preflight_is_cpu_only_and_keeps_budget(monkeypatch):
    import builtins
    from tests.unit.test_simulation_integrate import launcher
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in ('isaaclab', 'isaacsim', 'rclpy', 'gymnasium', 'omni'):
            raise AssertionError('unexpected runtime import')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    data = launcher.main(['--managed-evaluation', '--check-config'])
    assert data['managed_evaluation'] and data['dt'] == .01 and data['frame_skip'] == 1
    with pytest.raises(ValueError, match='cameras'):
        launcher.main(['--managed-evaluation', '--check-config', '--no-cameras'])


@pytest.mark.parametrize('evaluation', [{'episodes': True}, {'episodes': 0}, {'seed': -1}, {'seed': True}, {'max_steps': 0}])
def test_new_evaluation_config_rejects_invalid_numbers(evaluation):
    from src.simulation_config.experiment_config import _validate_evaluation_runtime
    with pytest.raises(ValueError):
        _validate_evaluation_runtime({'evaluation': evaluation})


def test_factory_idle_loop_does_not_run_free_physics(setup):
    import ast
    from pathlib import Path
    import time
    controller, env = setup
    path = Path(__file__).parents[2] / 'src/astribot_envs/astribot_envs_factory.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'managed_env_loop')
    namespace = {'time': time, 'ros_version': '2', 'rclpy': SimpleNamespace(ok=lambda: True)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    factory = SimpleNamespace(astribot_simu_env=SimpleNamespace(unwrapped=env),
                              managed_controller=controller, running=lambda: True,
                              stop_event=threading.Event())
    original = env.reset
    env.reset = lambda seed=None: original(seed)
    env.simulation_app = SimpleNamespace(is_running=lambda: True)
    env.multi_robot_ros_interface = SimpleNamespace(setup_evaluation_control=lambda owner: None)
    idle_steps = []
    def idle():
        idle_steps.append(env.steps)
        if len(idle_steps) == 1:
            assert controller.submit(batch(controller, 1))['status'] == 'accepted'
        elif len(idle_steps) == 3:
            assert controller.submit(request(controller, 'stop', 1))['status'] == 'accepted'
        elif len(idle_steps) == 4:
            factory.stop_event.set()
    env.managed_idle = idle
    assert controller.submit(request(controller, 'reset', seed=42))['status'] == 'accepted'
    namespace['managed_env_loop'](factory)
    assert idle_steps == [0, 1, 1, 1]
    assert env.resets == [None, 42]
    assert controller.records[0]['terminal']['reason'] == 'aborted'
    assert not env.lease


def test_ros_commands_are_blocked_only_during_eval_lease():
    import ast
    from pathlib import Path
    path = Path(__file__).parents[2] / 'src/simu_utils/robot_ros_interface.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'RobotRosInterface')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'joint_command_callback')
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    interface = SimpleNamespace(managed_evaluation=True, dof=1, joint_position_command=[1.],
                                joint_torque_command=[0.], is_two_wheel_chassis=lambda: False)
    message = SimpleNamespace(mode=1, command=[2.])
    namespace['joint_command_callback'](interface, message)
    assert interface.joint_position_command == [1.]
    interface.managed_evaluation = False
    namespace['joint_command_callback'](interface, message)
    assert interface.joint_position_command == [2.]


def test_factory_clears_evaluator_history_after_legacy_ros_reset():
    import ast
    from pathlib import Path
    path = Path(__file__).parents[2] / 'src/astribot_envs/astribot_envs_factory.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_evaluate_state')
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    calls = []
    env = SimpleNamespace(reset_generation=1)
    factory = SimpleNamespace(astribot_simu_env=SimpleNamespace(unwrapped=env),
        evaluator=SimpleNamespace(observe=lambda env, reset: calls.append(reset)), stop_event=threading.Event())
    namespace['_evaluate_state'](factory)
    namespace['_evaluate_state'](factory)
    env.reset_generation += 1
    namespace['_evaluate_state'](factory)
    assert calls == [True, False, True]


@pytest.mark.parametrize('operation', ['reset', 'step'])
def test_invalid_snapshot_cannot_be_reported_as_success(setup, operation):
    controller, env = setup
    if operation == 'step':
        reset(controller, env)
        env.pass_at = 1
        value = batch(controller, 1)
    else:
        env.evaluation_state = lambda: {'objects': {'pen': {
            'pose_world': [.5, .5, .5, 1., 0., 0., 0.], 'source': 'test'}}}
        value = request(controller, 'reset', seed=42)
    env.managed_snapshot = lambda capture_images: {'joints': {}, 'images': {}}
    assert controller.submit(value)['status'] == 'accepted'
    controller.process_one(env)
    assert controller.submit(value)['error']['execution_uncertain']
    assert controller.records[-1]['terminal']['reason'] == 'error'
    assert controller.phase == 'error'


def test_replay_lease_starts_with_measured_hold_and_clears_stale_reset():
    import ast
    from pathlib import Path
    path = Path(__file__).parents[2] / 'src/astribot_envs/astribot_isaaclab_env.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'AstribotIsaacLabEnv')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'set_managed_lease')
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    interface = SimpleNamespace(joint_position_command=[99.], joint_velocity_command=[5.],
                                joint_torque_command=[2.], controller_mode=3)
    multi = SimpleNamespace(reset_flag=True, pose_tuple_list=['stale object reset'])
    env = SimpleNamespace(robot_dict={'arm': interface}, robot_joint_map={'arm': ['a']},
                          multi_robot_ros_interface=multi, get_joint_positions=lambda names: [.25])
    namespace['set_managed_lease'](env, False)
    assert interface.joint_position_command == [.25]
    assert interface.joint_velocity_command == interface.joint_torque_command == [0.]
    assert interface.controller_mode == 1 and not interface.managed_evaluation
    assert not multi.reset_flag and not multi.pose_tuple_list
    namespace['set_managed_lease'](env, True)
    assert env.managed_evaluation and multi.managed_evaluation and interface.managed_evaluation


def test_heartbeat_pins_config_and_cancelled_reset_retires_its_owner(setup):
    controller, env = setup
    heartbeat = {'schema_version': 1, 'op': 'heartbeat', 'request_id': 'heartbeat',
                 'config_fingerprint': controller.config_fingerprint}
    result = controller.submit(heartbeat)['result']
    assert result['alive'] and result['config_fingerprint'] == controller.config_fingerprint
    heartbeat['config_fingerprint'] = 'another-config'
    assert controller.submit(heartbeat)['error']['code'] == 'config_mismatch'
    first = request(controller, 'reset', seed=42)
    assert controller.submit(first)['status'] == 'accepted'
    complete(controller, env, request(controller, 'stop'))
    first['request_id'] = 'late-retry-under-new-id'
    assert controller.submit(first)['error']['code'] == 'stale_identity'
    assert env.resets == []


def test_report_failure_replaces_success_with_execution_error(setup, monkeypatch, tmp_path):
    from src.evaluation import managed
    controller, env = setup
    reset(controller, env)
    env.pass_at = 1
    original = managed._write_json
    failed = []
    def fail_summary_once(path, document):
        if path.name == 'summary.json' and not failed:
            failed.append(True)
            raise OSError('injected summary write failure')
        return original(path, document)
    monkeypatch.setattr(managed, '_write_json', fail_summary_once)
    value = batch(controller, 1)
    controller.submit(value)
    controller.process_one(env)
    assert controller.submit(value)['error']['execution_uncertain']
    assert controller.terminal == {'done': True, 'reason': 'error'}
    assert len(controller.records) == 1 and controller.records[0]['terminal']['reason'] == 'error'
    report = json.loads(next(tmp_path.glob('run-*/episode-000001.json')).read_text())
    summary = json.loads(next(tmp_path.glob('run-*/summary.json')).read_text())
    assert report['terminal']['reason'] == 'error' and 'report write failed' in report['error']
    assert summary['counts']['success'] == 0 and summary['counts']['error'] == 1


def test_released_replay_identity_is_retired_but_cached_receipt_is_safe(setup):
    controller, env = setup
    acquire = request(controller, 'replay', enabled=True)
    original = complete(controller, env, acquire)
    release = request(controller, 'replay', enabled=False)
    release['request_id'] = 'release-owner'
    complete(controller, env, release)
    assert controller.phase == 'idle'
    assert controller.submit(acquire)['result'] == original
    assert not controller.process_one(env) and controller.phase == 'idle'
    stale = copy.deepcopy(acquire)
    stale['request_id'] = 'late-acquire'
    assert controller.submit(stale)['error']['code'] == 'stale_identity'
    assert controller.phase == 'idle'


@pytest.mark.parametrize('old_owner', ['replay', 'evaluation'])
def test_retired_cleanup_cannot_change_new_eval_owner(setup, old_owner):
    controller, env = setup
    if old_owner == 'replay':
        acquire = request(controller, 'replay', enabled=True)
        complete(controller, env, acquire)
        cleanup = request(controller, 'replay', enabled=False)
        cleanup['request_id'] = 'first-release'
    else:
        reset(controller, env)
        cleanup = request(controller, 'stop')
    complete(controller, env, cleanup)
    current = request(controller, 'reset', seed=42)
    current['run_id'] = 'new-eval-owner'
    current['request_id'] = 'new-eval-reset'
    complete(controller, env, current)
    before, steps = controller.state(), env.steps
    late = copy.deepcopy(cleanup)
    late['request_id'] = 'late-cleanup-new-request-id'
    result = controller.submit(late)
    assert result['status'] == 'completed'
    assert result['result']['retired_run_id'] == cleanup['run_id']
    assert controller.state() == before and env.steps == steps and env.lease
    assert not controller.process_one(env)
    late['request_id'] = 'wrong-fingerprint-cleanup'
    late['config_fingerprint'] = 'another-config'
    assert controller.submit(late)['error']['code'] == 'config_mismatch'
    assert controller.state() == before
