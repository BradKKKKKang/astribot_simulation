"""Progress accounting, long-operation logs and cleanup without model/network calls."""
import io
import sys
import threading
from types import SimpleNamespace

import pytest

from src.placement_analysis.identification_progress import IdentificationProgress


class Terminal(io.StringIO):
    def isatty(self):
        return True


class Bar:
    def __init__(self, **kwargs):
        self.total, self.n, self.closed = kwargs['total'], 0, False
        self.postfix = {}

    def set_postfix(self, refresh=True, **kwargs):
        self.postfix = kwargs

    def update(self, count):
        self.n += count

    def close(self):
        self.closed = True

    def write(self, message, file):
        print(message, file=file)


def test_tqdm_counts_success_reuse_failure_and_closes(monkeypatch):
    bars = []
    def make_bar(**kwargs):
        bar = Bar(**kwargs)
        bars.append(bar)
        return bar
    monkeypatch.setitem(sys.modules, 'tqdm.auto', SimpleNamespace(tqdm=make_bar))
    counts = dict(success=0, failed=0, reused=0, skipped=10)
    with IdentificationProgress(3, stream=Terminal()) as progress:
        for number, kind in enumerate(('success', 'reused', 'failed')):
            with progress.event(f'episode:{number}', counts):
                progress.stage('processing')
                counts[kind] += 1
    assert bars[0].total == bars[0].n == 3
    assert bars[0].postfix == dict(ok=1, reuse=1, fail=1)
    assert bars[0].closed and not progress.worker.is_alive()


def test_interrupt_keeps_actual_progress_and_logs_resume(monkeypatch):
    bar = Bar(total=3)
    monkeypatch.setitem(sys.modules, 'tqdm.auto', SimpleNamespace(tqdm=lambda **_: bar))
    stream = Terminal()
    counts = dict(success=0, failed=0, reused=0)
    with pytest.raises(KeyboardInterrupt):
        with IdentificationProgress(3, stream=stream) as progress:
            with progress.event('first', counts):
                counts['success'] += 1
            with progress.event('second', counts):
                progress.stage('JEV request')
                raise KeyboardInterrupt
    assert bar.n == 1 and bar.closed
    assert '已完成 1/3' in stream.getvalue()
    assert '重新执行原命令可继续' in stream.getvalue()
    assert not progress.worker.is_alive()


def test_long_stage_heartbeat_redacts_secret_and_no_progress_keeps_logs(monkeypatch):
    waiting = threading.Event()
    class NotifyingTerminal(Terminal):
        def write(self, value):
            result = super().write(value)
            if '仍在处理中' in value:
                waiting.set()
            return result
    def forbidden(**kwargs):
        raise AssertionError('--no-progress must not create a bar')
    monkeypatch.setitem(sys.modules, 'tqdm.auto', SimpleNamespace(tqdm=forbidden))
    stream = NotifyingTerminal()
    with IdentificationProgress(1, enabled=False, stream=stream, heartbeat_interval=.01,
                                redact=lambda value: value.replace('test-secret', '[REDACTED]')) as progress:
        with progress.event('episode:0001', dict(success=0, failed=0, reused=0)):
            progress.stage('JEV request test-secret')
            assert waiting.wait(1), 'long operation did not report its status'
    assert 'JEV request [REDACTED]' in stream.getvalue()
    assert '仍在处理中' in stream.getvalue()
    assert 'test-secret' not in stream.getvalue()
    assert not progress.worker.is_alive()
