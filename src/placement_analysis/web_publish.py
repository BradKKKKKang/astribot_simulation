"""Publish a completed bundle into a directly discoverable Web data directory."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import shutil
import tempfile

DEFAULT_VIEWER = Path(__file__).resolve().parents[2] / 'web/placement-viewer'


def task_directory_name(value):
    name = value.strip()
    if not name or name.startswith('.') or Path(name).name != name or '\\' in name:
        raise ValueError('task name must be a single, non-hidden directory name')
    return name


def publish_bundle(output, *, task_name=None, viewer=DEFAULT_VIEWER):
    output, viewer = Path(output).resolve(), Path(viewer).resolve()
    placement = (output / 'placement.json').read_bytes()
    bundle = json.loads(placement)
    if bundle.get('metadata', {}).get('status') != 'complete':
        raise ValueError('only complete placement bundles can be published')
    if bundle.get('robot_model', 'robot.glb') != 'robot.glb':
        raise ValueError('published bundle must use its companion robot.glb')
    model = output / 'robot.glb'
    if not model.is_file():
        raise FileNotFoundError(model)
    name = task_directory_name(task_name if task_name is not None else output.name)
    public = viewer / 'public/data'
    public.mkdir(parents=True, exist_ok=True)
    target = public / name
    with (public / '.publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.is_symlink():
            raise ValueError('task directory must not be a symbolic link')
        if target.resolve() != output:
            with tempfile.TemporaryDirectory(dir=public, prefix='.publish-') as temporary:
                staged = Path(temporary) / 'next'
                previous = Path(temporary) / 'previous'
                if target.exists():
                    # Preserve other files owned by the user in the task directory.
                    shutil.copytree(target, staged)
                else:
                    staged.mkdir()
                (staged / 'placement.json').write_bytes(placement)
                shutil.copyfile(model, staged / 'robot.glb')
                if target.exists():
                    target.rename(previous)
                try:
                    staged.rename(target)
                except OSError:
                    if previous.exists():
                        previous.rename(target)
                    raise
    print(f'WEB_PUBLISHED: {name} ({target})', flush=True)
    return target
