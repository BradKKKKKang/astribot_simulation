"""Explicit, hash-proven source relocation after dataset episode renumbering.

The original capture and its recorded input hash are never edited. A relocation
requires the exact original Parquet bytes plus a current file whose only data
changes are constant translations of episode_index and global index.
"""
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .recorder import write_json

MANIFEST = 'source_relocation.json'
ORIGINAL = 'source_original.parquet'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def renumbering_evidence(original, current):
    """Prove that every non-ID column, including time and row order, is identical."""
    old, new = pq.read_table(original), pq.read_table(current)
    identifiers = ('episode_index', 'index')
    if (not len(old) or len(old) != len(new) or old.column_names != new.column_names
            or not {'timestamp', 'frame_index', *identifiers} <= set(old.column_names)):
        raise ValueError('relocation requires identical row counts and columns')
    columns = [name for name in old.column_names if name not in identifiers]
    if not old.select(columns).equals(new.select(columns)):
        raise ValueError('relocation changes data beyond episode_index/index; trajectory or timestamps differ')
    offsets = {}
    for name in identifiers:
        before, after = old[name], new[name]
        if (not pa.types.is_integer(before.type) or before.type != after.type
                or before.null_count or after.null_count):
            raise ValueError('relocation identifiers must be non-null integers of the same type')
        differences = pc.unique(pc.subtract_checked(after, before)).to_pylist()
        if len(differences) != 1:
            raise ValueError(f'relocation requires a constant {name} offset')
        offsets[name] = differences[0]
    if len(pc.unique(old['episode_index'])) != 1 or len(pc.unique(new['episode_index'])) != 1:
        raise ValueError('relocation must contain exactly one episode')
    return {'rows': len(old), 'offsets': offsets,
            'original_episode_index': old['episode_index'][0].as_py(),
            'current_episode_index': new['episode_index'][0].as_py()}


def confined_file(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('relocation path must be relative')
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('relocation path escapes its root')
    return path


def resolve_source_relocation(capture, dataset_root, dataset_name, original_sha256):
    """Recheck proof and current hashes on every use; absent proof remains a failure."""
    capture = Path(capture)
    manifest = capture / MANIFEST
    entry = json.loads(manifest.read_text())
    if (entry.get('schema_version') != 1 or entry.get('dataset') != dataset_name
            or entry.get('original_sha256') != original_sha256
            or entry.get('capture_metadata_sha256') != sha256(capture / 'run_metadata.json')):
        raise ValueError('source relocation does not match the capture and task dataset')
    original = confined_file(capture, ORIGINAL)
    source = confined_file(dataset_root, entry['parquet'])
    if sha256(original) != original_sha256:
        raise ValueError('source relocation original Parquet SHA256 mismatch')
    if sha256(source) != entry.get('current_sha256'):
        raise ValueError('source relocation current Parquet SHA256 mismatch')
    if renumbering_evidence(original, source) != entry.get('renumbering'):
        raise ValueError('source relocation renumbering evidence mismatch')
    return source, {'source_relocation_file': str(manifest.resolve()),
                    'source_relocation_sha256': sha256(manifest)}


def register_source_relocation(capture, original, current, dataset_root, dataset_name):
    """Save verified proof alongside a capture; never replace capture metadata."""
    capture, original, current, dataset_root = [Path(p).resolve() for p in
                                               (capture, original, current, dataset_root)]
    metadata = json.loads((capture / 'run_metadata.json').read_text())
    expected = metadata['input_sha256']
    if metadata.get('status') != 'complete' or sha256(original) != expected:
        raise ValueError('relocation needs the exact original Parquet SHA256 from a complete capture')
    entry = {'schema_version': 1, 'dataset': dataset_name,
             'parquet': str(current.relative_to(dataset_root)), 'original_sha256': expected,
             'current_sha256': sha256(current),
             'capture_metadata_sha256': sha256(capture / 'run_metadata.json'),
             'renumbering': renumbering_evidence(original, current)}
    proof = capture / ORIGINAL
    if proof.exists() and sha256(proof) != expected:
        raise FileExistsError(f'refusing to overwrite different original source proof: {proof}')
    manifest = capture / MANIFEST
    if manifest.exists() and json.loads(manifest.read_text()) != entry:
        raise FileExistsError(f'refusing to overwrite different source relocation: {manifest}')
    if not proof.exists():
        temporary = proof.with_suffix('.parquet.tmp')
        temporary.write_bytes(original.read_bytes())
        if sha256(temporary) != expected:
            raise ValueError('original Parquet changed while registering relocation')
        temporary.replace(proof)
    write_json(manifest, entry)
    resolve_source_relocation(capture, dataset_root, dataset_name, expected)
    return entry
