"""Consume measured simulation state through the shared success predicates."""
import copy
import json
import os
import tempfile
from pathlib import Path

from .task_evaluation import evaluate_task


class OnlineEvaluator:
    """One evaluation episode; event completeness must come from its producer."""

    def __init__(self, resolved):
        if resolved['kind'] != 'evaluation':
            raise ValueError('online evaluation requires an evaluation experiment')
        self.sections = copy.deepcopy(resolved['sections'])
        self.fingerprints = copy.deepcopy(resolved['fingerprints'])
        self.max_steps = self.sections.get('runtime', {}).get('evaluation', {}).get('max_steps')
        self.steps = 0
        self.events = []
        self.observations = {}
        self.result = None
        self.completed_stages = {}

    def observe(self, env, *, reset=False):
        if reset:
            self.steps, self.events = 0, []
            self.completed_stages.clear()
        else:
            self.steps += 1
        reader = getattr(env, 'evaluation_state', None)
        state = reader() if reader is not None else {}
        if not isinstance(state, dict) or set(state) - {'objects', 'events', 'observations'}:
            raise ValueError('evaluation_state must contain objects, events and observations')
        events, observations = state.get('events', []), state.get('observations', {})
        if not isinstance(events, list) or not all(isinstance(event, dict) for event in events):
            raise ValueError('evaluation_state.events must be an event list')
        if not isinstance(observations, dict) or 'objects' in observations:
            raise ValueError('evaluation_state.observations must be an evidence mapping without objects')
        self.events.extend(copy.deepcopy(events))
        # Replace the snapshot each step: disappearing observations must become unknown.
        self.observations = {**copy.deepcopy(observations), 'objects': copy.deepcopy(state.get('objects', {}))}
        self.result = evaluate_task(self.events, self.sections['scene'], self.sections['task'],
                                   self.observations, self.completed_stages)
        return self.max_steps is not None and self.steps >= self.max_steps

    def report(self):
        return {'schema_version': 1, 'kind': 'online_evaluation', 'name': self.sections['name'],
                'steps': self.steps, 'result': copy.deepcopy(self.result),
                'events': copy.deepcopy(self.events), 'observations': copy.deepcopy(self.observations),
                'config': {key: copy.deepcopy(self.sections[key]) for key in ('scene', 'task')},
                'fingerprints': self.fingerprints}

    def write_report(self):
        root = self.sections.get('output', {}).get('root')
        if root is None or self.result is None:
            return None
        directory = Path(root).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / 'evaluation.json'
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=directory, prefix='.evaluation-', suffix='.tmp',
                                             delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(self.report(), stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
            os.replace(temporary, target)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return target
