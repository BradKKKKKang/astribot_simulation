"""No subprocess/SDK: background coalescing and ownership contracts."""
from threading import Event, Thread

import pytest

from src.placement_analysis.incremental_extraction import IncrementalExtraction


def test_notifications_during_active_callback_coalesce_and_close_drains():
    started, release = Event(), Event()
    calls = []
    def callback():
        calls.append(len(calls))
        if len(calls) == 1:
            started.set()
            assert release.wait(3)
    coordinator = IncrementalExtraction(callback)
    coordinator.schedule()
    assert started.wait(3)
    for _ in range(100):
        coordinator.schedule()
    release.set()
    coordinator.close()
    assert calls == [0, 1]
    assert not coordinator._thread.is_alive()
    coordinator.close()
    with pytest.raises(RuntimeError, match='closed'):
        coordinator.schedule()


def test_callback_failure_propagates_and_discards_pending():
    started, release = Event(), Event()
    calls = []
    failure = ValueError('CPU subprocess exited 2')
    def callback():
        calls.append(1)
        started.set()
        assert release.wait(3)
        raise failure
    coordinator = IncrementalExtraction(callback)
    coordinator.schedule()
    assert started.wait(3)
    coordinator.schedule()
    release.set()
    with pytest.raises(RuntimeError, match='incremental extraction failed') as caught:
        coordinator.close()
    assert caught.value.__cause__ is failure
    assert calls == [1]
    assert not coordinator._thread.is_alive()
    with pytest.raises(RuntimeError, match='incremental extraction failed'):
        coordinator.schedule()


def test_close_without_schedule_creates_no_worker():
    coordinator = IncrementalExtraction(lambda: pytest.fail('unexpected callback'))
    coordinator.close()
    assert coordinator._thread is None


def test_close_waits_for_owned_active_callback():
    started, release, closed = Event(), Event(), Event()
    def callback():
        started.set()
        assert release.wait(3)
    coordinator = IncrementalExtraction(callback)
    coordinator.schedule()
    assert started.wait(3)
    closing = Thread(target=lambda: (coordinator.close(), closed.set()))
    closing.start()
    assert not closed.wait(.02)
    release.set()
    closing.join(3)
    assert closed.is_set() and not coordinator._thread.is_alive()
