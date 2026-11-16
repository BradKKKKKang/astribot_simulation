"""YAML inputs retain strict experiment validation and reference semantics."""
import json
from pathlib import Path

import pytest
import yaml

from src.simulation_config.experiment_config import read_config, resolve_experiment


@pytest.mark.parametrize('suffix', ['.yaml', '.yml'])
def test_yaml_experiment_matches_json_and_resolves_relative_profile(tmp_path, suffix):
    root = Path(__file__).resolve().parents[2]
    experiment = json.loads((root / 'config/experiments/evaluation/penholder.json').read_text())
    for name in ('robot', 'scene', 'task'):
        experiment[name]['ref'] = str(
            (root / 'config/experiments/evaluation' / experiment[name]['ref']).resolve())
    # A YAML profile can refer to the existing JSON profiles.
    (tmp_path / 'task.yaml').write_text(yaml.safe_dump(experiment['task']))
    experiment['task'] = {'ref': 'task.yaml'}
    yaml_path = tmp_path / ('experiment' + suffix)
    json_path = tmp_path / 'experiment.json'
    yaml_path.write_text(yaml.safe_dump(experiment))
    json_path.write_text(json.dumps(experiment))
    actual, expected = resolve_experiment(yaml_path), resolve_experiment(json_path)
    assert actual['sections'] == expected['sections']
    assert actual['fingerprints'] == expected['fingerprints']


@pytest.mark.parametrize('document,match', [
    ('scene:\n  name: first\n  name: second\n', 'duplicate configuration field'),
    ('runtime: {dt: .nan}\n', 'finite JSON-compatible'),
    ('runtime: {dt: .inf}\n', 'finite JSON-compatible'),
    ('date: 2026-11-16\n', 'finite JSON-compatible'),
    ('1: value\n', 'field names must be strings'),
    ('cycle: &cycle [*cycle]\n', 'finite JSON-compatible'),
])
def test_yaml_rejects_ambiguous_or_non_json_data(tmp_path, document, match):
    path = tmp_path / 'invalid.yaml'
    path.write_text(document)
    with pytest.raises(ValueError, match=match):
        read_config(path)


def test_yaml_rejects_python_object_tags(tmp_path):
    path = tmp_path / 'unsafe.yaml'
    path.write_text('!!python/tuple [1, 2]\n')
    with pytest.raises(yaml.constructor.ConstructorError):
        read_config(path)
