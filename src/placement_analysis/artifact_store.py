"""Content-addressed immutable local artifacts, independent of simulation libraries."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                       separators=(',', ':')) + '\n').encode()


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ArtifactStore:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()

    def _path(self, ref):
        if not isinstance(ref, dict) or not re.fullmatch(r'[a-f0-9]{64}', ref.get('sha256', '')):
            raise ValueError('invalid artifact digest')
        digest = ref['sha256']
        relative = ref.get('path', '')
        if not isinstance(relative, str) or not re.fullmatch(
                rf'objects/{digest[:2]}/{digest}\.[a-z0-9]+', relative):
            raise ValueError('invalid artifact path')
        path = self.root / relative
        if any(parent.is_symlink() for parent in [path, *path.parents] if parent != self.root.parent):
            raise ValueError('artifact paths must not contain symbolic links')
        if not path.resolve().is_relative_to(self.root):
            raise ValueError('artifact path escapes store')
        return path

    def resolve(self, ref):
        path = self._path(ref)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != ref['sha256'] or (
                'size_bytes' in ref and len(data) != ref['size_bytes']):
            raise ValueError('artifact digest or size mismatch')
        return path

    def read_bytes(self, ref):
        # Verify the bytes actually returned, including concurrent tampering.
        data = self._path(ref).read_bytes()
        if hashlib.sha256(data).hexdigest() != ref['sha256'] or (
                'size_bytes' in ref and len(data) != ref['size_bytes']):
            raise ValueError('artifact digest or size mismatch')
        return data

    def read_json(self, ref):
        return json.loads(self.read_bytes(ref))

    def put_bytes(self, data, suffix='bin'):
        if not re.fullmatch(r'[a-z0-9]+', suffix):
            raise ValueError('invalid artifact extension')
        data = bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        ref = {'path': f'objects/{digest[:2]}/{digest}.{suffix}',
               'sha256': digest, 'size_bytes': len(data)}
        path = self._path(ref)
        if path.exists():
            self.resolve(ref)
            return ref
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.artifact-', dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)  # create exclusively; never replace an immutable object
            except FileExistsError:
                self.resolve(ref)
        finally:
            os.unlink(temporary)
        return ref

    def put_json(self, value):
        return self.put_bytes(json_bytes(value), 'json')

    def put_file(self, path):
        path = Path(path)
        return self.put_bytes(path.read_bytes(), path.suffix.lstrip('.').lower() or 'bin')
