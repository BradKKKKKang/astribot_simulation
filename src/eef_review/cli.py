"""Unified environment check, deployment, lifecycle and precompute commands."""
import argparse
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from .common import REPO, atomic_json, load_config, read_json
from .runtime import usd_environment


def json_url(url, body=None):
    request = Request(url, data=json.dumps(body).encode() if body is not None else None,
                      headers={'Content-Type': 'application/json'})
    deadline = time.monotonic() + 60
    while True:
        try:
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except (URLError, ConnectionError, TimeoutError) as error:
            if (isinstance(error, HTTPError) and error.code < 500) or time.monotonic() >= deadline:
                raise
            time.sleep(1)


def port_free(host, port):
    with socket.socket() as sock:
        # A just-stopped HTTP listener may leave TIME_WAIT connections. Match
        # the gateway's reuse semantics while still rejecting a live listener.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))


def prepare(config):
    cache = Path(config['cache'])
    cache.mkdir(parents=True, exist_ok=True)
    atomic_json(cache / 'runtime.json', config)
    subprocess.run([config['usd_python'], '-m', 'src.eef_review.fk_worker', '--prepare', str(cache / 'runtime.json')],
                   cwd=REPO, env=usd_environment(config), check=True)


def deploy_remote(config, start_reader=True):
    install = config['remote_install']
    package = io.BytesIO()
    with tarfile.open(fileobj=package, mode='w:gz') as archive:
        for path in (REPO / 'src/eef_review').glob('*.py'):
            archive.add(path, arcname='src/eef_review/' + path.name)
        content = json.dumps(config).encode()
        info = tarfile.TarInfo('runtime.json')
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    command = f'mkdir -p {shlex.quote(install)} && tar -xzf - -C {shlex.quote(install)}'
    subprocess.run(['ssh', '-o', 'BatchMode=yes', config['ssh_host'], command], input=package.getvalue(), check=True)
    if not start_reader:
        return
    # The bootstrap verifies ownership before stopping a previous reader supervisor.
    bootstrap = '''import json,os,pathlib,signal,subprocess,sys,time
p=pathlib.Path(sys.argv[1]).expanduser().resolve();state=p/'run';state.mkdir(exist_ok=True)
pidfile=state/'cloud.pid.json'
if pidfile.exists():
 pid=json.loads(pidfile.read_text())['pid'];cmd=pathlib.Path('/proc')/str(pid)/'cmdline'
 if cmd.exists() and b'src.eef_review.supervisor' in cmd.read_bytes():
  os.kill(pid,signal.SIGTERM)
  for _ in range(100):
   if not pidfile.exists():break
   time.sleep(.1)
with (state/'supervisor.log').open('ab') as log:
 subprocess.Popen([sys.executable,'-m','src.eef_review.supervisor','cloud','--config',str(p/'runtime.json'),'--state',str(state)],cwd=p,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
'''
    command = 'python3 -c ' + shlex.quote(bootstrap) + ' ' + shlex.quote(install)
    subprocess.run(['ssh', config['ssh_host'], command], check=True)


def stop(config):
    if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'astribot-eef-review.service']).returncode == 0:
        subprocess.run(['systemctl', '--user', 'stop', 'astribot-eef-review.service'], check=True)
        print('EEF local service stopped; remote reader retained for reuse')
        return
    state = Path(config['cache']) / 'run/local.pid.json'
    if not state.exists():
        print('EEF local service is stopped')
        return
    pid = read_json(state)['pid']
    proc = Path('/proc') / str(pid) / 'cmdline'
    if proc.exists() and b'src.eef_review.supervisor' in proc.read_bytes():
        os.kill(pid, signal.SIGTERM)
        for _ in range(100):
            if not state.exists():
                break
            time.sleep(.1)
    print('EEF local service stopped; remote reader retained for reuse')


def start(config):
    cache = Path(config['cache'])
    state = cache / 'run'
    state.mkdir(parents=True, exist_ok=True)
    if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'astribot-eef-review.service']).returncode == 0:
        raise RuntimeError('EEF supervisor already running; use status or restart')
    port_free(config['host'], config['port'])
    port_free('127.0.0.1', config['tunnel_port'])
    if not (Path(config['frontend']) / 'index.html').exists():
        raise RuntimeError('build web/eef-viewer first: npm ci && npm run build')
    prepare(config)
    deploy_remote(config)
    persistent_unit = cache / 'astribot-eef-review.service'
    fragment = subprocess.check_output(['systemctl', '--user', 'show', 'astribot-eef-review.service',
                                        '--property=FragmentPath', '--value'], text=True).strip()
    if persistent_unit.exists() and fragment and Path(fragment).resolve() == persistent_unit.resolve():
        subprocess.run(['systemctl', '--user', 'start', 'astribot-eef-review.service'], check=True)
    else:
        subprocess.run(['systemd-run', '--user', '--unit=astribot-eef-review', '--collect',
                        '--property=Restart=on-failure', '--property=RestartSec=3',
                        '--property=WorkingDirectory=' + str(REPO), sys.executable,
                        '-m', 'src.eef_review.supervisor', 'local', '--config', str(cache / 'runtime.json'),
                        '--state', str(state)], check=True)
    url = f'http://{config["host"]}:{config["port"]}'
    for _ in range(60):
        try:
            health = json_url(url + '/api/health')
            if health['cloud_connected']:
                print(url + '/')
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError('service did not become healthy; inspect ' + str(state))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('start', 'stop', 'restart', 'status', 'prepare-model', 'deploy-reader', 'build-index'):
        commands.add_parser(name)
    precompute = commands.add_parser('precompute')
    precompute.add_argument('--episode', action='append')
    precompute.add_argument('--per-group', type=int, default=1, choices=range(1, 6))
    precompute.add_argument('--mode', choices=['state', 'action'], default='state')
    clean = commands.add_parser('clean-cache')
    clean.add_argument('--older-than-days', type=float, default=30)
    clean.add_argument('--apply', action='store_true', help='default only lists expired complete results')
    args = parser.parse_args()
    config = load_config(args.config)
    url = f'http://{config["host"]}:{config["port"]}'
    if args.command == 'start':
        start(config)
    elif args.command == 'stop':
        stop(config)
    elif args.command == 'restart':
        stop(config)
        start(config)
    elif args.command == 'status':
        print(json.dumps(json_url(url + '/api/health'), indent=2))
    elif args.command == 'prepare-model':
        prepare(config)
    elif args.command == 'deploy-reader':
        deploy_remote(config)
    elif args.command == 'build-index':
        deploy_remote(config, start_reader=False)
        command = 'cd ' + shlex.quote(config['remote_install']) + ' && ' + shlex.join([
            'python3', '-m', 'src.eef_review.formal_index', '--root', config['remote_root'],
            '--sample-root', config['sample_root'], '--output', config['remote_index'], '--config', 'runtime.json'])
        subprocess.run(['ssh', config['ssh_host'], command], check=True)
    elif args.command == 'precompute':
        ids = args.episode or json_url(url + f'/api/sample?per_group={args.per_group}')['episode_ids']
        report = []
        for key in ids:
            metadata = json_url(url + '/api/episodes/' + key)
            if metadata['review']['status'] != 'admitted':
                item = {'episode_id': key, 'group': metadata['group'], 'status': metadata['review']['status'],
                        'reason': metadata['review']['reason']}
            else:
                job = json_url(url + f'/api/episodes/{key}/compute', {'mode': args.mode})
                while job['status'] in ('queued', 'running'):
                    time.sleep(1)
                    job = json_url(url + '/api/jobs/' + job['key'])
                item = {'episode_id': key, 'group': metadata['group'], **job}
                if job['status'] == 'complete':
                    item['summary'] = json_url(url + '/api/results/' + job['key'])['summary']
            report.append(item)
            prefix = 'acceptance-formal' if config.get('source_mode') == 'formal' else 'acceptance'
            atomic_json(Path(config['cache']) / f'{prefix}-{args.mode}-{args.per_group}.json', report)
            print(json.dumps({k: v for k, v in item.items() if k != 'summary'}, ensure_ascii=False), flush=True)
    elif args.command == 'clean-cache':
        if args.apply and subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'astribot-eef-review.service']).returncode == 0:
            raise RuntimeError('stop this service before applying trajectory cache cleanup')
        cutoff = time.time() - args.older_than_days * 86400
        for path in (Path(config['cache']) / 'results').glob('*'):
            if path.is_dir() and (path / 'result.json').is_file() and path.stat().st_mtime < cutoff:
                print(path)
                if args.apply:
                    shutil.rmtree(path)


if __name__ == '__main__':
    main()
