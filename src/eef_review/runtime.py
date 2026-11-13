"""Construct isolated worker environments without loading ROS in the gateway."""
import os
from pathlib import Path
import subprocess

from .common import REPO, digest


def sdk_environment(config):
    root = Path(config['sdk_root'])
    setups = ['/opt/ros/humble/setup.bash', root / 'third_party/software/local_setup.bash',
              root / 'astribot_msgs/share/astribot_msgs/local_setup.bash',
              root / 'third_party/third_pkg/local_setup.bash']
    # Setup paths are arguments, never interpolated shell program text.
    command = ['bash', '-c', 'for setup in "$@"; do source "$setup" >/dev/null; done; env -0', 'eef-env',
               *map(str, setups)]
    output = subprocess.check_output(command, env=os.environ.copy())
    env = dict(item.split('=', 1) for item in output.decode().split('\0') if '=' in item)
    libraries = [root / 'astribot_sdk/core/common/robotics_library_py',
                 root / 'astribot_sdk/core/common/whole_body_control/third_party', root / 'third_party/drake/lib']
    env.update(ASTRIBOT_SDK_ROOT=str(root), ASTRIBOT_SIMU_ROOT=str(REPO), ROBOT_TYPE='S1',
               ROS_DOMAIN_ID=str(config['ros_domain_id']), ROS_LOCALHOST_ONLY='1',
               RMW_IMPLEMENTATION='rmw_fastrtps_cpp', ROS_LOG_DIR=str(Path(config['cache']) / 'ros_logs'),
               PYTHONDONTWRITEBYTECODE='1', ASTRIBOT_LOG='1')
    env['PYTHONPATH'] = ':'.join([str(root), str(REPO), env.get('PYTHONPATH', '')])
    env['LD_LIBRARY_PATH'] = ':'.join([*map(str, libraries), env.get('LD_LIBRARY_PATH', '')])
    for key in ('FASTRTPS_DEFAULT_PROFILES_FILE', 'FASTDDS_DEFAULT_PROFILES_FILE', 'ROS_DISCOVERY_SERVER'):
        env.pop(key, None)
    return env


def usd_environment(config):
    env = os.environ.copy()
    libraries = config['usd_libraries']
    env['PYTHONPATH'] = ':'.join([libraries, str(REPO), env.get('PYTHONPATH', '')])
    env['LD_LIBRARY_PATH'] = ':'.join([libraries + '/bin', libraries + '/bin/deps', env.get('LD_LIBRARY_PATH', '')])
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


def sdk_fingerprint(config):
    root = Path(config['sdk_root'])
    files = []
    for path in sorted((root / 'astribot_sdk').rglob('*')):
        if path.is_file() and path.suffix in ('.py', '.so', '.yaml', '.yml', '.urdf'):
            files.append([str(path.relative_to(root)), digest(path.read_bytes())])
    if not files:
        raise ValueError('SDK files not found')
    return digest(files)
