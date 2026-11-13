"""Own and restart only EEF child processes; survives a launching terminal."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .common import atomic_json, read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('role', choices=['local', 'cloud'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--state', required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    state = Path(args.state).resolve()
    state.mkdir(parents=True, exist_ok=True)
    lock = (state / (args.role + '.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic_json(state / (args.role + '.pid.json'), {'pid': os.getpid(), 'role': args.role})
    if args.role == 'cloud':
        commands = {'reader': [sys.executable, '-m', 'src.eef_review.cloud', '--root', config['remote_root'],
                               '--port', str(config['remote_port']), '--mode', config.get('source_mode', 'sample')]}
        if config.get('remote_index'):
            commands['reader'] += ['--index', config['remote_index']]
    else:
        commands = {
            'tunnel': ['ssh', '-N', '-L', f'127.0.0.1:{config["tunnel_port"]}:127.0.0.1:{config["remote_port"]}',
                       '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes', '-o', 'ConnectTimeout=10',
                       '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', config['ssh_host']],
            'gateway': [sys.executable, '-m', 'src.eef_review.gateway', '--config', str(Path(args.config).resolve())],
        }
    children = {}
    next_start = {name: 0 for name in commands}
    delay = {name: 1 for name in commands}
    stopped = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stopped.__setitem__(0, True))
    try:
        while not stopped[0]:
            for name, command in commands.items():
                process = children.get(name)
                if process and process.poll() is not None:
                    next_start[name] = time.monotonic() + delay[name]
                    delay[name] = min(30, delay[name] * 2)
                    del children[name]
                if name not in children and time.monotonic() >= next_start[name]:
                    with (state / (name + '.log')).open('ab') as log:
                        children[name] = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
                elif process and process.poll() is None and time.monotonic() - next_start[name] > 60:
                    delay[name] = 1
            time.sleep(.5)
    finally:
        for process in children.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in children.values():
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        (state / (args.role + '.pid.json')).unlink(missing_ok=True)


if __name__ == '__main__':
    main()
