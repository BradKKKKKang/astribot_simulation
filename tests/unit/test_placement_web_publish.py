"""Runtime Web publication checks; no ROS, USD or simulation."""
import json

import pytest

from src.placement_analysis.web_publish import publish_bundle


def make_bundle(path, *, status='complete', events=None):
    path.mkdir(parents=True)
    (path / 'placement.json').write_text(json.dumps({
        'metadata': {'status': status}, 'robot_model': 'robot.glb', 'events': events or [],
    }))
    (path / 'robot.glb').write_bytes(b'companion robot')
    return path


def test_publish_uses_named_directories_without_index_or_build(tmp_path):
    viewer = tmp_path / 'viewer'
    (viewer / 'dist').mkdir(parents=True)
    (viewer / 'dist/index.html').write_text('existing build')
    first = publish_bundle(make_bundle(tmp_path / 'first'), task_name='按压按钮', viewer=viewer)
    (first / 'notes.txt').write_text('user notes')
    other = publish_bundle(make_bundle(tmp_path / 'other'), viewer=viewer)
    latest = publish_bundle(make_bundle(tmp_path / 'second', events=[{'id': 1}]),
                            task_name='按压按钮', viewer=viewer)
    assert first == latest == viewer / 'public/data/按压按钮'
    assert json.loads((latest / 'placement.json').read_text())['events'] == [{'id': 1}]
    assert (latest / 'robot.glb').read_bytes() == b'companion robot'
    assert (latest / 'notes.txt').read_text() == 'user notes'
    assert other == viewer / 'public/data/other'
    assert not (viewer / 'public/data/tasks.json').exists()
    assert (viewer / 'dist/index.html').read_text() == 'existing build'
    assert not (viewer / 'dist/data').exists()  # preview serves public/data directly


def test_bundle_already_in_public_data_needs_no_copy(tmp_path):
    viewer = tmp_path / 'viewer'
    source = make_bundle(viewer / 'public/data/press_cube_olddemo')
    (source / 'analysis.txt').write_text('keep')
    original = (source / 'placement.json').stat().st_ino
    assert publish_bundle(source, viewer=viewer) == source
    assert (source / 'placement.json').stat().st_ino == original
    assert (source / 'analysis.txt').read_text() == 'keep'
    assert not (viewer / 'dist').exists()


@pytest.mark.parametrize('failure', ['partial', 'missing_model', 'wrong_model'])
def test_failed_bundle_keeps_previous_task(tmp_path, failure):
    viewer = tmp_path / 'viewer'
    source = make_bundle(tmp_path / 'task')
    target = publish_bundle(source, viewer=viewer)
    before = (target / 'placement.json').read_bytes()
    bundle = json.loads((source / 'placement.json').read_text())
    if failure == 'partial':
        bundle['metadata']['status'] = 'partial'
    elif failure == 'missing_model':
        (source / 'robot.glb').unlink()
    else:
        bundle['robot_model'] = '../robot.glb'
    (source / 'placement.json').write_text(json.dumps(bundle))
    with pytest.raises((ValueError, FileNotFoundError)):
        publish_bundle(source, viewer=viewer)
    assert (target / 'placement.json').read_bytes() == before


@pytest.mark.parametrize('name', ['', '.', '..', '../escape', '/absolute', 'a/b', '.hidden', 'a\\b'])
def test_task_name_is_one_visible_directory(tmp_path, name):
    with pytest.raises(ValueError, match='directory name'):
        publish_bundle(make_bundle(tmp_path / 'task'), task_name=name, viewer=tmp_path / 'viewer')
