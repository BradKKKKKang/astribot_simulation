"""Append-only evidence at the managed command/physics boundary."""
import hashlib
import json
from pathlib import Path


class EpisodeTrace:
    def __init__(self, root, run_id, episode_id, metadata):
        run = 'run-' + hashlib.sha256(run_id.encode()).hexdigest()[:24]
        self.directory = Path(root).expanduser().resolve() / run / f'episode-{episode_id:06d}'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'steps.jsonl'
        # Never mix a restarted process with an earlier episode's evidence.
        with self.path.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps({'event': 'metadata', 'schema_version': 1,
                                     'run_id': run_id, 'episode_id': episode_id,
                                     **metadata}, allow_nan=False) + '\n')

    @property
    def artifacts(self):
        return {'steps': str(self.path), 'directory': str(self.directory)}

    def append(self, event, **data):
        # Close after every record: aborts retain all completed boundaries.
        # A command without a matching state means execution is unconfirmed.
        line = json.dumps({'event': event, **data}, ensure_ascii=False, allow_nan=False)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')
