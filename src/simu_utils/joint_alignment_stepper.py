"""Simulation-thread stepping after independently delivered SDK/ROS targets.

The service only verifies targets. It never assigns robot commands or states.
"""
import json
import queue
import threading
import time

import numpy as np


class JointAlignmentStepper:
    def __init__(self, node, dt):
        from astribot_msgs.srv import RawRequest
        from rclpy.callback_groups import ReentrantCallbackGroup
        self.dt = dt
        self.stepped = False
        self.steps = 0
        self.queue = queue.Queue(maxsize=1)
        self.group = ReentrantCallbackGroup()
        self.service = node.create_service(RawRequest, '/simulation/joint_alignment_step',
                                           self.submit, callback_group=self.group)

    def submit(self, request, response):
        item = {'done': threading.Event(), 'request': request.request, 'cancelled': False}
        try:
            self.queue.put_nowait(item)
            if not item['done'].wait(10):
                item['cancelled'] = True
                raise TimeoutError('simulation step request timed out')
            response.response = json.dumps(item['response'], allow_nan=False)
        except Exception as error:
            response.response = json.dumps({'error': str(error)})
        return response

    @staticmethod
    def feedback(env):
        result = {}
        for part in env.robot_list:
            names = env.robot_joint_map[part]
            result[part] = {'position': env.get_joint_positions(names),
                            'velocity': env.get_joint_velocities(names)}
        return result

    @staticmethod
    def targets_match(env, expected):
        if set(expected) != set(env.robot_list):
            raise ValueError('step must verify every configured robot part')
        for part, target in expected.items():
            interface = env.robot_dict[part]
            if len(target) != interface.dof or not np.isfinite(target).all():
                raise ValueError(f'{part}: invalid expected target')
            if interface.controller_mode != 1 or not np.allclose(
                    interface.joint_position_command, target, rtol=0, atol=1e-9):
                return False
        return True

    def process_one(self, env):
        try:
            item = self.queue.get(timeout=.001 if self.stepped else 0)
        except queue.Empty:
            return
        try:
            if item['cancelled']:
                return
            request = json.loads(item['request'])
            op = request['op']
            if op == 'begin':
                if self.stepped:
                    raise ValueError('alignment session already active')
                self.stepped, self.steps = True, 0
            elif op == 'end':
                self.stepped = False
            elif op == 'step':
                if not self.stepped or request['step'] != self.steps:
                    raise ValueError('alignment step sequence mismatch')
                deadline = time.monotonic() + 3
                while not self.targets_match(env, request['expected']):
                    if item['cancelled'] or time.monotonic() >= deadline:
                        raise TimeoutError('SDK targets were not received unchanged on ROS2')
                    time.sleep(.0005)
                env.step(None)
                self.steps += 1
            else:
                raise ValueError('unknown alignment operation')
            item['response'] = {'dt': self.dt, 'steps': self.steps,
                                'physics_time_s': self.steps * self.dt,
                                'observed_system_ns': time.time_ns(), 'joints': self.feedback(env)}
        except Exception as error:
            item['response'] = {'error': str(error)}
        finally:
            item['done'].set()
