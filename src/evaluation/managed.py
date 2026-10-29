"""CPU-only queued, idempotent control of one externally scheduled episode.

ROS callbacks call submit(); only the simulation main thread calls process_one().
The evaluator remains three-valued. Terminal success is latched by this owner.
"""
from collections import OrderedDict
import copy
import hashlib
import json
import math
from pathlib import Path
import threading
import time

from .online import OnlineEvaluator
from src.simulation_config.experiment_config import fingerprint

CAMERAS = {'head_rgbd': 'cam_high', 'left_wrist_rgbd': 'cam_left_wrist',
           'right_wrist_rgbd': 'cam_right_wrist'}
IDENTITY = ('run_id', 'episode_id', 'step_id', 'config_fingerprint')


class ProtocolError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def validate_managed_config(resolved, data):
    runtime = resolved['sections'].get('runtime', {})
    evaluation = runtime.get('evaluation', {})
    if set(evaluation) != {'episodes', 'max_steps', 'seed'}:
        raise ValueError('managed evaluation requires explicit episodes, max_steps and seed')
    if runtime.get('simulation', {}).get('dt') != .01 or runtime.get('simulation', {}).get('frame_skip') != 1:
        raise ValueError('managed evaluation requires explicit runtime.simulation dt=0.01, frame_skip=1')
    if runtime['simulation'].get('decimation', 1) != 1:
        raise ValueError('managed evaluation requires decimation=1 when specified')
    task = resolved['sections']['task']
    if not task.get('initial_pose') or 'robot_placement' not in resolved['sections']:
        raise ValueError('managed evaluation requires task.initial_pose and robot_placement')
    if not task.get('success', {}).get('predicates'):
        raise ValueError('managed evaluation requires nonempty task success predicates')
    if set(data.get('camera_names', [])) != set(CAMERAS):
        raise ValueError('managed evaluation requires head and both wrist cameras')
    if data['simulator_type'] != 'IsaacLab':
        raise ValueError('managed evaluation supports IsaacLab only')
    data['managed_evaluation'] = True
    data['decimation'] = 1
    return data


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate request field: ' + key)
        result[key] = value
    return result


def _write_json(path, value):
    import os
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class ManagedEvaluation:
    """Single simulation owner; Atlas alone chooses the next episode."""

    def __init__(self, resolved):
        self.resolved = copy.deepcopy(resolved)
        self.sections = self.resolved['sections']
        self.config_fingerprint = fingerprint(self.sections)
        self.evaluation = copy.deepcopy(self.sections['runtime']['evaluation'])
        self.parts = {part['name']: {'joint_names': part['feedback_joints'],
                                   'dof': len(part['feedback_joints']),
                                   'position_unit': 'percent' if 'gripper' in part['name'] else 'rad'}
                      for part in self.sections['robot']['parts'] if 'chassis' not in part['name']}
        self.parts['astribot_chassis'] = {
            'joint_names': ['astribot_chassis_x', 'astribot_chassis_y', 'astribot_chassis_z_rot'],
            'dof': 3, 'position_unit': 'm,m,rad'}
        self.command_parts = {name: value for name, value in self.parts.items() if name != 'astribot_chassis'}
        self.evaluator = OnlineEvaluator(resolved)
        self.lock = threading.RLock()
        self.pending = None
        self.executing = None
        self.signatures = {}
        self.responses = OrderedDict()
        self.run_id = None
        self.retired_runs = set()
        self.episode_id = 0
        self.step_id = 0
        self.phase = 'idle'
        self.snapshot = None
        self.terminal = {'done': False, 'reason': None}
        self.records = []
        self.started = None
        self.seed = None
        self.reset_layout = {}
        self.stop_requested = False
        self.stop_reason = 'aborted'
        self.wake = threading.Event()

    def state(self):
        return {'run_id': self.run_id, 'episode_id': self.episode_id,
                'step_id': self.step_id, 'phase': self.phase}

    @staticmethod
    def response(request_id, status, **fields):
        return {'schema_version': 1, 'request_id': request_id, 'status': status, **fields}

    def _error(self, request_id, code, message, uncertain=False):
        return self.response(request_id, 'error', error={'code': code, 'message': message,
                                                       'execution_uncertain': uncertain})

    def _cache(self, request_id, response):
        self.responses[request_id] = response
        self.responses.move_to_end(request_id)
        while len(self.responses) > 128:
            self.responses.popitem(last=False)
        return copy.deepcopy(response)

    def submit_json(self, raw):
        try:
            request = json.loads(raw, object_pairs_hook=_pairs,
                                 parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
        except (ValueError, TypeError):
            return json.dumps(self._error(None, 'invalid_request', 'request must be strict finite JSON'))
        return json.dumps(self.submit(request), ensure_ascii=False, allow_nan=False)

    def submit(self, request):
        request_id = request.get('request_id') if isinstance(request, dict) else None
        try:
            if not isinstance(request, dict) or set(request) - {'schema_version', 'op', 'request_id', 'payload', *IDENTITY}:
                raise ProtocolError('invalid_request', 'unknown request fields')
            if request.get('schema_version') != 1 or type(request.get('schema_version')) is not int:
                raise ProtocolError('invalid_request', 'schema_version must be 1')
            if not isinstance(request_id, str) or not 1 <= len(request_id) <= 256:
                raise ProtocolError('invalid_request', 'request_id must be a nonempty string up to 256 characters')
            signature = fingerprint(request)
            with self.lock:
                if request_id in self.signatures:
                    if self.signatures[request_id] != signature:
                        raise ProtocolError('request_conflict', 'request_id was used with different content')
                    if request_id not in self.responses:
                        raise ProtocolError('expired_request', 'completed request is outside the receipt cache; it will not execute again')
                    return copy.deepcopy(self.responses[request_id])
                op = request.get('op')
                if op not in ('describe', 'heartbeat', 'reset', 'step', 'snapshot', 'stop', 'replay'):
                    raise ProtocolError('invalid_request', 'unsupported operation')
                if op in ('describe', 'heartbeat'):
                    if (op == 'heartbeat' and 'config_fingerprint' in request
                            and request['config_fingerprint'] != self.config_fingerprint):
                        raise ProtocolError('config_mismatch', 'configuration fingerprint differs')
                    result = {'state': self.state(), 'alive': True, 'monotonic_s': time.monotonic(),
                              'config_fingerprint': self.config_fingerprint}
                    if op == 'describe':
                        result.update(config_fingerprint=self.config_fingerprint, managed=True, dt=.01, frame_skip=1,
                                      evaluation=self.evaluation, parts=self.parts,
                                      task={**{k: self.sections['task'].get(k, '') for k in ('name', 'instruction')},
                                            'success_configured': bool(self.sections['task'].get('success', {}).get('predicates'))})
                    return self.response(request_id, 'completed', result=copy.deepcopy(result))
                self._validate_identity(request)
                if self._retired_cleanup(request):
                    self.signatures[request_id] = signature
                    return self._cache(request_id, self.response(request_id, 'completed',
                                       result={'state': self.state(), 'retired_run_id': request['run_id']}))
                self._validate_operation(request)
                if op == 'stop':
                    self.stop_requested = True
                    self.stop_reason = request.get('payload', {}).get('reason', 'aborted')
                    if self.pending is not None:
                        cancelled = self.pending
                        self._cache(cancelled['request_id'], self._error(cancelled['request_id'], 'cancelled', 'cancelled by stop'))
                        self.pending = None
                elif self.pending is not None or self.executing is not None:
                    raise ProtocolError('busy', 'another operation is pending')
                self.signatures[request_id] = signature
                self.pending = copy.deepcopy(request)
                self.wake.set()
                return self._cache(request_id, self.response(request_id, 'accepted'))
        except (ProtocolError, TypeError, ValueError) as error:
            return self._error(request_id, getattr(error, 'code', 'invalid_request'), str(error))

    def _validate_identity(self, request):
        if any(field not in request for field in IDENTITY):
            raise ProtocolError('invalid_identity', 'complete run/episode/step/config identity is required')
        if request['config_fingerprint'] != self.config_fingerprint:
            raise ProtocolError('config_mismatch', 'configuration fingerprint differs')
        if not isinstance(request['run_id'], str) or not 1 <= len(request['run_id']) <= 256:
            raise ProtocolError('invalid_identity', 'run_id must be a nonempty string up to 256 characters')
        if type(request['episode_id']) is not int or request['episode_id'] < 1:
            raise ProtocolError('invalid_identity', 'episode_id must start at 1')
        if type(request['step_id']) is not int or request['step_id'] < 0:
            raise ProtocolError('invalid_identity', 'step_id must be nonnegative')

    def _retired_cleanup(self, request):
        """A lost cleanup receipt is recoverable without touching a newer owner."""
        if request['run_id'] not in self.retired_runs:
            return False
        op, payload = request['op'], request.get('payload', {})
        if not isinstance(payload, dict):
            raise ProtocolError('invalid_request', 'payload must be an object')
        if op == 'stop':
            if set(payload) - {'reason'} or payload.get('reason', 'aborted') not in ('aborted', 'error'):
                raise ProtocolError('invalid_request', 'stop reason must be aborted or error')
            return True
        if op == 'replay' and payload.get('enabled') is False:
            if set(payload) != {'enabled'} or request['episode_id'] != 1 or request['step_id'] != 0:
                raise ProtocolError('invalid_identity', 'replay release requires episode_id=1 and step_id=0')
            return True
        return False

    def _validate_operation(self, request):
        op = request['op']
        payload = request.get('payload', {})
        if not isinstance(payload, dict):
            raise ProtocolError('invalid_request', 'payload must be an object')
        if op == 'replay':
            if request['episode_id'] != 1 or request['step_id'] != 0:
                raise ProtocolError('invalid_identity', 'replay lease requires episode_id=1 and step_id=0')
            if set(payload) != {'enabled'} or type(payload['enabled']) is not bool:
                raise ProtocolError('invalid_request', 'replay payload requires boolean enabled')
            if payload['enabled'] and request['run_id'] in self.retired_runs:
                raise ProtocolError('stale_identity', 'released run_id cannot acquire a new lease')
            if payload['enabled'] and self.phase not in ('idle', 'stopped'):
                raise ProtocolError('lease_conflict', 'evaluation must stop before replay acquires control')
            if not payload['enabled'] and (self.phase != 'replay' or request['run_id'] != self.run_id):
                raise ProtocolError('lease_conflict', 'only the replay owner can release control')
            return
        if self.phase == 'replay':
            raise ProtocolError('lease_conflict', 'replay owns simulation control')
        if op == 'reset':
            if self.phase == 'error':
                raise ProtocolError('execution_error', 'stop the faulted run before resetting')
            new_run = request['run_id'] != self.run_id or self.phase in ('idle', 'stopped')
            if new_run:
                if self.phase not in ('idle', 'stopped') or request['episode_id'] != 1 or request['run_id'] in self.retired_runs:
                    raise ProtocolError('stale_identity', 'new run requires idle/stopped phase and episode 1 with a fresh run_id')
            elif self.phase != 'terminal' or request['episode_id'] != self.episode_id + 1:
                raise ProtocolError('stale_identity', 'reset must follow the terminal episode')
            if request['episode_id'] > self.evaluation['episodes'] or request['step_id'] != 0:
                raise ProtocolError('invalid_identity', 'reset exceeds episode budget or has nonzero step_id')
            expected = self.evaluation['seed'] + request['episode_id'] - 1
            if set(payload) != {'seed'} or type(payload['seed']) is not int or payload['seed'] != expected:
                raise ProtocolError('invalid_seed', 'reset seed must be base seed + episode_id - 1')
            return
        cancelling_reset = (op == 'stop' and self.pending is not None and self.pending['op'] == 'reset'
                            and all(request[key] == self.pending[key] for key in IDENTITY))
        if not cancelling_reset and (request['run_id'] != self.run_id or request['episode_id'] != self.episode_id):
            raise ProtocolError('stale_identity', 'request belongs to another run or episode')
        if op == 'step':
            if self.phase != 'ready' or self.stop_requested or request['step_id'] != self.step_id + 1:
                raise ProtocolError('stale_step', 'step must follow the last completed step in a ready episode')
            if set(payload) != {'dt', 'commands', 'capture_images'} or type(payload['dt']) not in (int, float) or payload['dt'] != .01:
                raise ProtocolError('invalid_batch', 'step requires dt=0.01, commands and capture_images')
            if type(payload['capture_images']) is not bool:
                raise ProtocolError('invalid_batch', 'capture_images must be boolean')
            commands = payload['commands']
            if not isinstance(commands, dict) or set(commands) != set(self.command_parts):
                raise ProtocolError('invalid_batch', 'commands must contain every non-chassis part exactly once')
            for part, expected in self.command_parts.items():
                value = commands[part]
                if not isinstance(value, dict) or set(value) != {'joint_names', 'position', 'velocity'} or value['joint_names'] != expected['joint_names']:
                    raise ProtocolError('invalid_batch', 'joint names/order differ for ' + part)
                for key in ('position', 'velocity'):
                    vector = value[key]
                    if not isinstance(vector, list) or len(vector) != expected['dof'] or not all(type(x) in (int, float) and math.isfinite(x) for x in vector):
                        raise ProtocolError('invalid_batch', 'nonfinite or incomplete ' + key + ' for ' + part)
                if expected['position_unit'] == 'percent' and not all(0 <= x <= 100 for x in value['position']):
                    raise ProtocolError('invalid_batch', 'gripper position must be 0..100 percent')
        elif op == 'snapshot':
            if payload or request['step_id'] != self.step_id or self.snapshot is None or self.phase not in ('ready', 'terminal'):
                raise ProtocolError('stale_identity', 'snapshot must refer to the current completed step')
        elif op == 'stop':
            if set(payload) - {'reason'} or payload.get('reason', 'aborted') not in ('aborted', 'error'):
                raise ProtocolError('invalid_request', 'stop reason must be aborted or error')
            if not cancelling_reset and request['step_id'] not in (self.step_id, self.step_id + 1):
                raise ProtocolError('stale_identity', 'stop step must be current or in flight')

    def process_one(self, env):
        """Run at most one operation; called only from the simulation main thread."""
        with self.lock:
            if self.pending is None:
                self.wake.clear()
                return False
            request, self.pending = self.pending, None
            self.executing = request
        request_id, op = request['request_id'], request['op']
        try:
            if op == 'reset':
                with self.lock:
                    if request['run_id'] != self.run_id:
                        self.records = []
                    self.run_id, self.episode_id, self.step_id = request['run_id'], request['episode_id'], 0
                    self.phase = 'resetting'
                    self.seed = request['payload']['seed']
                    self.started = time.monotonic()
                    self.terminal = {'done': False, 'reason': None}
                    self.snapshot = None
                    self.reset_layout = {}
                    self.evaluator = OnlineEvaluator(self.resolved)
                env.set_managed_lease(True)
                env.reset(seed=self.seed)
                env.managed_publish_feedback()
                self.evaluator.observe(env, reset=True)
                self.reset_layout = copy.deepcopy(self.evaluator.observations.get('objects', {}))
                with self.lock:
                    self.phase = 'ready'
                result = {'snapshot': self._complete_boundary(env, True)}
            elif op == 'step':
                env.managed_step(request['payload']['commands'])
                with self.lock:
                    self.step_id = request['step_id']
                self.evaluator.observe(env)
                result = {'snapshot': self._complete_boundary(env, request['payload']['capture_images'])}
            elif op == 'snapshot':
                result = {'snapshot': self._snapshot(env, True)}
            elif op == 'stop':
                with self.lock:
                    if not self.terminal['done'] and self.started is not None:
                        self._finish(request.get('payload', {}).get('reason', 'aborted'))
                    self.phase = 'stopped'
                    self.retired_runs.add(request['run_id'])
                    self.stop_requested = False
                env.set_managed_lease(False)
                result = {'state': self.state(), 'terminal': copy.deepcopy(self.terminal)}
            else:  # Explicit Replay lease; the factory controls continuous steps.
                enabled = request['payload']['enabled']
                env.set_managed_lease(False)
                with self.lock:
                    if not enabled:
                        self.retired_runs.add(request['run_id'])
                    self.phase = 'replay' if enabled else 'idle'
                    self.run_id = request['run_id'] if enabled else None
                    self.episode_id, self.step_id = request['episode_id'], 0
                result = {'state': self.state()}
            response = self.response(request_id, 'completed', result=result)
        except Exception as error:
            with self.lock:
                self.phase = 'error'
                try:
                    if self.started is not None and not self.terminal['done']:
                        self._finish('error', str(error))
                except Exception as report_error:
                    error = RuntimeError(f'{error}; report failed: {report_error}')
            self.phase = 'error'
            response = self._error(request_id, 'execution_error', str(error), True)
        with self.lock:
            self.executing = None
            self._cache(request_id, response)
        return True

    def _complete_boundary(self, env, images):
        # Feedback/camera failures are execution errors even when the task
        # predicate passed. Validate the completion before recording success.
        snapshot = self._snapshot(env, images)
        with self.lock:
            self._check_terminal()
            snapshot['terminal'] = copy.deepcopy(self.terminal)
            self.snapshot = copy.deepcopy(snapshot)
        return snapshot

    def _check_terminal(self):
        if self.stop_requested:
            self._finish(self.stop_reason)
        elif self.evaluator.result['status'] == 'passed':
            self._finish('success')
        elif self.step_id >= self.evaluation['max_steps']:
            self._finish('timeout')

    def _snapshot(self, env, images):
        if (self.snapshot is not None and self.snapshot['run_id'] == self.run_id
                and self.snapshot['episode_id'] == self.episode_id
                and self.snapshot['step_id'] == self.step_id
                and (not images or self.snapshot['images'])):
            return copy.deepcopy(self.snapshot)
        capture = env.managed_snapshot(capture_images=images)
        joints = capture['joints']
        if not isinstance(joints, dict) or set(joints) != set(self.parts):
            raise RuntimeError('snapshot requires all seven feedback parts')
        for name, expected in self.parts.items():
            value = joints[name]
            if value.get('joint_names') != expected['joint_names']:
                raise RuntimeError('snapshot joint names/order differ: ' + name)
            for key in ('position', 'velocity'):
                vector = value.get(key)
                if (not isinstance(vector, list) or len(vector) != expected['dof']
                        or not all(type(x) in (int, float) and math.isfinite(x) for x in vector)):
                    raise RuntimeError('snapshot has invalid feedback: ' + name + '.' + key)
        encoded = {CAMERAS[name]: value for name, value in capture['images'].items()}
        if images and set(encoded) != set(CAMERAS.values()):
            raise RuntimeError('snapshot requires all three RGB cameras')
        snapshot = {'snapshot_id': fingerprint([self.config_fingerprint, self.run_id, self.episode_id, self.step_id]),
                    'config_fingerprint': self.config_fingerprint,
                    'run_id': self.run_id, 'episode_id': self.episode_id, 'step_id': self.step_id,
                    'sim_time_s': self.step_id * .01, 'joints': capture['joints'], 'images': encoded,
                    'reset_layout': copy.deepcopy(self.reset_layout),
                    'evaluation': copy.deepcopy(self.evaluator.result), 'terminal': copy.deepcopy(self.terminal)}
        with self.lock:
            self.snapshot = snapshot
        return copy.deepcopy(snapshot)

    def _finish(self, reason, error=None):
        if self.terminal['done']:
            return
        self.terminal = {'done': True, 'reason': reason}
        self.phase = 'terminal'
        report = self.evaluator.report()
        report.update(run_id=self.run_id, episode_id=self.episode_id, seed=self.seed,
                      steps=self.step_id, sim_time_s=self.step_id * .01,
                      wall_time_s=time.monotonic() - self.started,
                      reset_layout=copy.deepcopy(self.reset_layout), config_fingerprint=self.config_fingerprint,
                      terminal=copy.deepcopy(self.terminal),
                      evidence_status=self.evaluator.result['status'] if self.evaluator.result else 'unknown')
        if error:
            report['error'] = error
        self.records.append(report)
        try:
            self._persist_reports(report)
        except Exception as report_error:
            # A persisted task pass is not a completed evaluation if its result
            # cannot be committed. Replace this record rather than appending a
            # second outcome; retry both files in case the write fault cleared.
            self.terminal = {'done': True, 'reason': 'error'}
            self.phase = 'error'
            report['terminal'] = copy.deepcopy(self.terminal)
            report['error'] = 'report write failed: ' + str(report_error)
            try:
                self._persist_reports(report)
            except Exception as recovery_error:
                report['error'] += '; error report write failed: ' + str(recovery_error)
            raise RuntimeError(report['error']) from report_error

    def _persist_reports(self, report):
        root = self.sections.get('output', {}).get('root')
        if not root:
            return
        # The owner ID is opaque and must never become an unvalidated path.
        run_dir = Path(root).expanduser() / ('run-' + hashlib.sha256(self.run_id.encode()).hexdigest()[:24])
        _write_json(run_dir / f'episode-{self.episode_id:06d}.json', report)
        counts = {reason: sum(r['terminal']['reason'] == reason for r in self.records)
                  for reason in ('success', 'timeout', 'aborted', 'error')}
        _write_json(run_dir / 'summary.json', {'schema_version': 1, 'run_id': self.run_id,
                    'config_fingerprint': self.config_fingerprint, 'evaluation': self.evaluation,
                    'completed_episodes': len(self.records), 'counts': counts,
                    'episodes': [{key: r[key] for key in ('episode_id', 'seed', 'steps', 'sim_time_s',
                                 'wall_time_s', 'terminal', 'evidence_status')} for r in self.records]})

    def shutdown(self, env):
        """Finalize an interrupted episode before entrypoint closes Isaac."""
        with self.lock:
            if self.started is not None and not self.terminal['done']:
                self._finish('aborted')
            self.phase = 'stopped'
        env.set_managed_lease(False)
