"""Incremental CLI lifecycle with mock CPU children and mock replay only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
from threading import Event
from unittest.mock import Mock

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'incremental_replay_cli', Path(__file__).resolve().parents[2] / 'scripts/replay_extract_event.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def test_interim_flags_chosen_interpreter_coalescing_and_drain(tmp_path, monkeypatch):
    args = cli.parser().parse_args(['dataset', '--incremental', '--extract-python', '/cpu/python',
                                   '--output-root', str(tmp_path / 'captures'), '--output', str(tmp_path / 'bundle')])
    started, release = Event(), Event()
    commands, order = [], []
    def cpu(command, timeout, cancel_event):
        commands.append(command)
        assert timeout == 3600 and not cancel_event.is_set()
        assert command[0] == '/cpu/python'
        assert {'--incremental', '--resume-extraction', '--no-identify',
                '--no-web-publish'} <= set(command)
        assert command[command.index('--extract-only') + 1] == str(args.output_root)
        if len(commands) == 1:
            started.set()
            assert release.wait(3)
        order.append('CPU done')
        return 0
    def replay(args, report, on_capture):
        on_capture()
        assert started.wait(3)
        for _ in range(30):
            on_capture()
        order.append('replay cleanup')
        release.set()
    monkeypatch.setattr(cli, 'run_owned_cpu_process', cpu)
    monkeypatch.setattr(cli, 'run', replay)
    cli.replay_with_incremental_extraction(args, {})
    assert len(commands) == 2
    assert order == ['replay cleanup', 'CPU done', 'CPU done']
    assert args.resume_extraction
    assert not args.no_identify and not args.no_web_publish


def test_interruption_cancels_owned_cpu_before_propagating(tmp_path, monkeypatch):
    args = cli.parser().parse_args(['dataset', '--incremental'])
    started, finished = Event(), Event()
    def cpu(command, timeout, cancel_event):
        started.set()
        assert cancel_event.wait(3)
        finished.set()
        raise RuntimeError('cancelled fixture')
    def replay(args, report, on_capture):
        on_capture()
        assert started.wait(3)
        raise KeyboardInterrupt
    monkeypatch.setattr(cli, 'run_owned_cpu_process', cpu)
    monkeypatch.setattr(cli, 'run', replay)
    with pytest.raises(KeyboardInterrupt):
        cli.replay_with_incremental_extraction(args, {})
    assert finished.is_set()


def test_resumed_batch_callback_always_includes_later_completed_captures(tmp_path, monkeypatch):
    from src.placement_analysis import replay_capture
    args = cli.parser().parse_args(['dataset', '--resume', '--no-viewer', '--no-progress',
                                   '--output-root', str(tmp_path / 'captures')])
    sources = []
    for i in range(3):
        path = tmp_path / f'episode{i}.parquet'
        path.write_bytes(b'fixture')
        sources.append({'episode_id': f'episode{i}', 'input_parquet': str(path),
                        'input_sha256': hashlib.sha256(b'fixture').hexdigest(),
                        'metadata_file': None, 'frames': 2})
    robot, middleware = Mock(), Mock()
    monkeypatch.setenv('ROS_VERSION', '2')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('ROS_DOMAIN_ID', '197')
    monkeypatch.setattr(cli, 'load_sdk', lambda: (Mock(return_value=robot), middleware))
    monkeypatch.setattr(cli, 'wait_for_simulation', Mock())
    monkeypatch.setattr(cli, 'completed_capture', lambda path, *_: path.name in ('episode0', 'episode2'))
    monkeypatch.setattr(cli, 'load_segment', lambda *a, **k: (np.zeros((2, 25)), np.array([0, .01])))
    monkeypatch.setattr(cli, 'replay', Mock())
    capture = Mock()
    monkeypatch.setattr(replay_capture, 'ReplayCapture', Mock(return_value=capture))
    snapshots = []
    def completed():
        batch = json.loads((args.output_root / 'batch_manifest.json').read_text())
        snapshots.append(batch['completed'])
        if 'episode1' in batch['completed']:
            capture.save.assert_called_once()
            capture.close.assert_called_once()
    cli.run(args, report={'episodes': sources}, on_capture=completed)
    assert snapshots[:2] == [['episode0', 'episode2'], ['episode0', 'episode2']]
    assert snapshots[2:] == [['episode0', 'episode1', 'episode2']] * 2
    robot.stop_robot.assert_called_once()
    middleware.shutdown.assert_called_once()


def test_owned_cpu_cancellation_terminates_only_created_session(monkeypatch):
    cancelled = Event()
    process = Mock(pid=123456)
    process.poll.return_value = None
    calls = []
    def wait(timeout):
        calls.append(timeout)
        if len(calls) == 1:
            cancelled.set()
            raise subprocess.TimeoutExpired('fixture', timeout)
        return 0
    process.wait.side_effect = wait
    spawn = Mock(return_value=process)
    terminate = Mock()
    monkeypatch.setattr(cli.subprocess, 'Popen', spawn)
    monkeypatch.setattr(cli.os, 'killpg', terminate)
    with pytest.raises(RuntimeError, match='cancelled'):
        cli.run_owned_cpu_process(['/cpu/python', 'fixture'], 30, cancelled)
    spawn.assert_called_once_with(['/cpu/python', 'fixture'], start_new_session=True)
    terminate.assert_called_once_with(123456, signal.SIGTERM)
    assert calls[-1] == 3


def test_owned_cpu_timeout_escalates_cleanup_for_its_child_only(monkeypatch):
    process = Mock(pid=123457)
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired('fixture', 3), 0]
    monkeypatch.setattr(cli.subprocess, 'Popen', Mock(return_value=process))
    monkeypatch.setattr(cli.time, 'monotonic', Mock(side_effect=[0, 2]))
    terminate = Mock()
    monkeypatch.setattr(cli.os, 'killpg', terminate)
    with pytest.raises(subprocess.TimeoutExpired):
        cli.run_owned_cpu_process(['fixture'], 1, Event())
    assert [call.args for call in terminate.call_args_list] == [(123457, signal.SIGTERM), (123457, signal.SIGKILL)]
    assert [call.kwargs for call in process.wait.call_args_list] == [{'timeout': 3}, {'timeout': 3}]


def test_evaluation_is_an_explicit_separate_entrypoint(monkeypatch):
    from scripts import evaluate_placement
    from src.evaluation import placement
    calls = []
    monkeypatch.setattr(placement, 'evaluate_bundle',
                        lambda *a, **kw: calls.append((a, kw)) or {'summary': {'unknown': 1}})
    assert evaluate_placement.main(['bundle', '--experiment', 'evaluation.json', '--dry-run']) == 0
    assert calls == [((Path('bundle'),), {'task_config': Path('evaluation.json'),
                                         'observations': None, 'dry_run': True})]
    with pytest.raises(SystemExit):
        cli.parser().parse_args(['bundle', '--evaluate-only'])


def test_interim_extract_only_skips_evaluation_identification_and_sdk(tmp_path, monkeypatch):
    from src.placement_analysis import extraction, identification
    from src.evaluation import placement as pipeline_stages
    output = tmp_path / 'bundle'
    def extract(*args, **kwargs):
        output.mkdir()
        (output / 'placement.json').write_text('{}')
    monkeypatch.setattr(extraction, 'extract_events', extract)
    monkeypatch.setattr(identification, 'identify_bundle', lambda *a, **k: pytest.fail('interim identification'))
    monkeypatch.setattr(pipeline_stages, 'evaluate_bundle', lambda *a, **k: pytest.fail('interim evaluation'))
    monkeypatch.setattr(cli, 'load_sdk', lambda: pytest.fail('offline extraction imported SDK'))
    assert cli.main(['capture', '--extract-only', '--output', str(output), '--incremental',
                     '--resume-extraction', '--no-identify', '--no-web-publish']) == 0


def test_interruption_while_draining_also_cancels_and_joins_child(monkeypatch):
    from src.placement_analysis import incremental_extraction
    original = incremental_extraction.IncrementalExtraction
    started, finished = Event(), Event()
    instances = []
    class InterruptedClose(original):
        def __init__(self, callback):
            super().__init__(callback)
            self.interrupted = False
            instances.append(self)
        def close(self):
            if not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt
            return super().close()
    def cpu(command, timeout, cancel_event):
        started.set()
        assert cancel_event.wait(3)
        finished.set()
        raise RuntimeError('cancelled while draining')
    def replay(args, report, on_capture):
        on_capture()
        assert started.wait(3)
    monkeypatch.setattr(incremental_extraction, 'IncrementalExtraction', InterruptedClose)
    monkeypatch.setattr(cli, 'run_owned_cpu_process', cpu)
    monkeypatch.setattr(cli, 'run', replay)
    args = cli.parser().parse_args(['dataset', '--incremental'])
    with pytest.raises(KeyboardInterrupt):
        cli.replay_with_incremental_extraction(args, {})
    assert finished.is_set() and not instances[0]._thread.is_alive()
