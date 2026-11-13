"""Offline SDK query worker. No robot command/control APIs are used."""
import argparse
import os
from pathlib import Path
import threading
import time
import sys

import numpy as np

from .common import atomic_json, read_json

PARTS = [('astribot_chassis', 0, 3), ('astribot_torso', 3, 7),
         ('astribot_arm_left', 7, 14), ('astribot_gripper_left', 14, 15),
         ('astribot_arm_right', 15, 22), ('astribot_gripper_right', 22, 23), ('astribot_head', 23, 25)]
SOLVED = [('torso', 'astribot_torso', 3, 7), ('left', 'astribot_arm_left', 7, 14),
          ('right', 'astribot_arm_right', 15, 22)]


def initial_joints(pose):
    q = np.zeros(25)
    for name, start, end in PARTS:
        if name in pose['joint_targets_rad']:
            q[start:end] = pose['joint_targets_rad'][name]
    q[14], q[22] = pose['gripper_percent']['left'], pose['gripper_percent']['right']
    return q


def run(job, output):
    config = job['config']
    assert os.environ['ROS_LOCALHOST_ONLY'] == '1'
    assert os.environ['ROS_DOMAIN_ID'] == str(config['ros_domain_id'])
    from astribot_sdk.core.common.robotics_library_py import robotics_library_base as base
    base.robot_init()
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from astribot_msgs.msg import RobotJointState, RobotJointController
    from astribot_msgs.srv import RawRequest
    from astribot_sdk.core.astribot_api.astribot_client import Astribot

    initial = initial_joints(job['initial_pose'])
    current = [initial.copy()]
    feed = Node('eef_review_state_fixture')
    publishers = {name: feed.create_publisher(RobotJointState, '/' + name + '/joint_space_states', 10)
                  for name, _, _ in PARTS}
    commands = [0]

    def command_seen(_):
        commands[0] += 1

    subscriptions = [feed.create_subscription(RobotJointController, '/' + name + '/joint_space_command',
                                              command_seen, qos_profile_sensor_data) for name, _, _ in PARTS]

    def publish():
        q = current[0].copy()
        for name, start, end in PARTS:
            message = RobotJointState()
            message.header.stamp = feed.get_clock().now().to_msg()
            message.header.frame_id = 'simulation'
            message.mode = 1
            message.name = (['astribot_chassis_x', 'astribot_chassis_y', 'astribot_chassis_z_rot']
                            if name == 'astribot_chassis' else [name + f'_joint_{i + 1}' for i in range(end - start)])
            message.position = q[start:end].tolist()
            message.velocity = message.acceleration = message.torque = [0.] * (end - start)
            publishers[name].publish(message)

    def mode(_, response):
        response.response = 'simulation'
        return response

    feed.create_service(RawRequest, '/astribot_safe_mode/state', mode)
    feed.create_timer(.002, publish)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(feed)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    robot = Astribot(node_name='eef_review_offline_queries')
    names = [name for _, name, _, _ in SOLVED]

    def seed(q):
        current[0] = q.copy()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            observed = robot.get_current_joints_position(names)
            if all(np.allclose(observed[i], q[start:end], atol=1e-8, rtol=0)
                   for i, (_, _, start, end) in enumerate(SOLVED)):
                return
            time.sleep(.001)
        raise TimeoutError('SDK did not acknowledge continuous joint seed')

    def fk(q):
        return robot.get_forward_kinematics(names, [q[start:end].tolist() for _, _, start, end in SOLVED])

    seed(initial)
    initial_fk = fk(initial)
    limits = robot.get_joints_position_limit(names)
    checkpoint = Path(output) / 'sdk_checkpoint.json'
    previous = read_json(checkpoint) if checkpoint.exists() else {'frames': [], 'key': job['key']}
    if previous['key'] != job['key']:
        raise ValueError('checkpoint identity mismatch')
    results = previous['frames']
    q_previous = np.array(results[-1]['joints']) if results else initial.copy()
    try:
        for index in range(len(results), len(job['trajectory']['frames'])):
            frame = job['trajectory']['frames'][index][job['mode']]
            q = q_previous.copy()
            for side, slot in [('left', 14), ('right', 22)]:
                if frame['grippers'][side] is not None:
                    q[slot] = frame['grippers'][side]
            start = time.monotonic()
            seed(q)
            acknowledged = time.monotonic()
            held = fk(q)
            targets = [frame['targets']['torso'] or initial_fk['astribot_torso'],
                       frame['targets']['left'] or held['astribot_arm_left'],
                       frame['targets']['right'] or held['astribot_arm_right']]
            has_target = any(frame['targets'][side] is not None for side in ('left', 'right'))
            ok, result = robot.get_inverse_kinematics(names, targets) if has_target else (False, None)
            status = 'ok' if ok else ('ik_failed' if has_target else 'no_valid_targets')
            if ok:
                candidate = q.copy()
                for _, name, first, end in SOLVED:
                    candidate[first:end] = result[name]
                if not np.isfinite(candidate).all():
                    ok, status = False, 'invalid_solution'
                else:
                    q = candidate
            achieved = fk(q)
            violations = []
            for i, (_, name, first, end) in enumerate(SOLVED):
                for j, value in enumerate(q[first:end]):
                    if value < limits[0][i][j] - 1e-6 or value > limits[1][i][j] + 1e-6:
                        violations.append(name + f'_joint_{j + 1}')
            jump = float(np.max(np.abs(q[np.r_[3:14, 15:22]] - q_previous[np.r_[3:14, 15:22]])) * 180 / np.pi)
            results.append({'status': status, 'ik_success': bool(ok), 'joints': q.tolist(),
                            'sdk_fk': {side: achieved[name] for side, name, _, _ in SOLVED},
                            'held_pose': not bool(ok), 'joint_jump_deg': jump,
                            'limit_violations': violations,
                            'torso_drift_deg': float(np.max(np.abs(q[3:7] - initial[3:7])) * 180 / np.pi),
                            'seed_wait_s': acknowledged - start, 'solve_s': time.monotonic() - acknowledged})
            q_previous = q
            if (index + 1) % 32 == 0 or index + 1 == len(job['trajectory']['frames']):
                atomic_json(checkpoint, {'key': job['key'], 'frames': results})
                atomic_json(Path(output) / 'progress.json', {'status': 'running', 'stage': 'SDK IK',
                            'completed_frames': index + 1, 'total_frames': len(job['trajectory']['frames'])})
        atomic_json(Path(output) / 'sdk.json', {'frames': results, 'limits': limits,
                    'initial_fk': initial_fk, 'command_messages_observed': commands[0],
                    'seed_policy': 'previous valid solution, acknowledged through SDK current joint query'})
        if commands[0]:
            raise RuntimeError('unexpected command traffic in isolated query domain')
    finally:
        executor.shutdown(timeout_sec=2)
        feed.destroy_node()
    # The SDK registers overlapping Python/C++ atexit destructors. This dedicated
    # process has already fsynced its result and stopped the fixture; let the OS
    # release native ROS resources without running those destructors twice.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    args = parser.parse_args()
    output = Path(args.job).parent
    run(read_json(args.job), output)


if __name__ == '__main__':
    main()
