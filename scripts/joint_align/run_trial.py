#!/usr/bin/env python3
"""Own one isolated localhost simulator and SDK replay, preserving both logs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import yaml

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sdk-root', type=Path, default=ROOT.parent / 'astribot_sdk_ros2')
    parser.add_argument('--sim-python', type=Path, required=True)
    parser.add_argument('--sdk-python', type=Path, required=True)
    parser.add_argument('--domain', type=int, default=197)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--settle-steps', type=int, default=0)
    parser.add_argument('episodes', nargs='+', type=Path)
    args = parser.parse_args()
    if not 1 <= args.domain <= 232 or args.domain == 25:
        parser.error('choose an isolated ROS domain in 1..232, excluding shared domain 25')
    args.output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(args.config.read_text())
    if config.get('dt') != .004 or config.get('frame_skip') != 1:
        parser.error('trial config requires dt=.004 and frame_skip=1')
    stepped = config.get('joint_alignment_step_service', False)
    if args.settle_steps < 0 or (args.settle_steps and not stepped):
        parser.error('--settle-steps requires stepped simulation and a nonnegative count')
    snapshot = args.output / 'simulation_config.yaml'
    snapshot.write_text(args.config.read_text())
    runtime_env = os.environ.copy()
    runtime_env.update(ROS_LOG_DIR=str((args.output / 'ros_logs').resolve()),
                       PYTHONDONTWRITEBYTECODE='1')
    # Source first, then override the SDK/project defaults. Never publish on their shared domain.
    setup = '''set -e
source /opt/ros/humble/setup.zsh
source "$1/env.sh"
source "$2/third_party/software/local_setup.zsh"
source "$2/astribot_msgs/share/astribot_msgs/local_setup.zsh"
source "$2/third_party/third_pkg/local_setup.zsh"
export ASTRIBOT_SDK_ROOT="$2" ROBOT_TYPE=S1 ASTRIBOT_LOG=1
export PYTHONPATH="$2:$PYTHONPATH"
export LD_LIBRARY_PATH="$2/astribot_sdk/core/common/robotics_library_py:$2/astribot_sdk/core/common/whole_body_control/third_party:$2/third_party/drake/lib:${LD_LIBRARY_PATH:-}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID="$3" ROS_LOCALHOST_ONLY=1 ASTRIBOT_ISAAC_CONTROL_HZ=250
unset FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE ROS_DISCOVERY_SERVER
shift 3
exec "$@"
'''
    def command(python, *argv):
        return ['zsh', '-c', setup, 'joint-alignment', str(ROOT), str(args.sdk_root.resolve()),
                str(args.domain), str(python.resolve()), *map(str, argv)]
    sim = None
    started = time.time()
    result = {'config': str(args.config.resolve()), 'domain': args.domain,
              'config_sha256': hashlib.sha256(snapshot.read_bytes()).hexdigest(),
              'localhost_only': True, 'dt': .004, 'time_mode': 'physics_step' if stepped else 'wall_clock',
              'settle_steps': args.settle_steps,
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), ROOT / 'scripts/joint_align/replay_artifacts_sdk_100hz.py',
                                          ROOT / 'src/astribot_envs/astribot_isaaclab_env.py',
                                          ROOT / 'src/simulation_config/joint_filter.py']},
              'status': 'running'}
    try:
        with (args.output / 'simulation.log').open('w') as log:
            sim = subprocess.Popen(command(args.sim_python, '-u', ROOT / 'astribot_simulation.py',
                                           '--config', snapshot.resolve(), '--headless', '--no-cameras'),
                                   cwd=ROOT, env=runtime_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            with (args.output / 'sdk.log').open('w') as sdk_log:
                replay = subprocess.run(command(args.sdk_python, '-u', ROOT / 'scripts/joint_align/replay_artifacts_sdk_100hz.py',
                                                *[p.resolve() for p in args.episodes], '--control-hz', '250',
                                                '--connect-timeout', '180', '--preposition-timeout', '30',
                                                '--settle-steps', str(args.settle_steps),
                                                '--output-root', args.output.resolve() / 'replay',
                                                *(['--physics-step'] if stepped else [])),
                                        cwd=ROOT, env=runtime_env, stdout=sdk_log, stderr=subprocess.STDOUT, timeout=args.timeout)
            result['sdk_exit_code'] = replay.returncode
            if replay.returncode:
                raise RuntimeError(f'SDK replay failed ({replay.returncode}); see {args.output / "sdk.log"}')
            result['status'] = 'complete'
    except Exception as error:
        result.update(status='failed', error=str(error))
        raise
    finally:
        if sim is not None:
            if sim.poll() is None:
                sim.send_signal(signal.SIGINT)
                try:
                    sim.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    sim.kill()
                    sim.wait(timeout=10)
                    result['forced_simulator_stop'] = True
            result['simulator_exit_code'] = sim.returncode
            if sim.returncode != 0 or result.get('forced_simulator_stop'):
                result.update(status='failed', error='simulator did not shut down cleanly')
        result['duration_s'] = time.time() - started
        (args.output / 'trial.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    if result['status'] != 'complete':
        raise RuntimeError(result.get('error', 'trial did not complete'))


if __name__ == '__main__':
    main()
