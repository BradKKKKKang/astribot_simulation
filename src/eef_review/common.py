"""Small shared utilities; the remote reader needs only the standard library."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

REPO = Path(__file__).resolve().parents[2]


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(value).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def confined(root, relative):
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError('path outside registered root')
    return result


def load_config(path=None):
    config = read_json(path or REPO / 'config/eef_review/default.json')
    if config['source_base_to_chassis'] != [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]:
        raise ValueError('first release supports only the explicitly recorded identity base assumption')
    for key in ('cache', 'initial_pose', 'frontend'):
        config[key] = str(confined(REPO, config[key]))
    return config
