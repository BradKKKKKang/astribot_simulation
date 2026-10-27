"""Coalesce completed-capture notifications into one background CPU callback.

The replay owner supplies a callback that launches its chosen extraction Python
with incremental/resume enabled and identification/publication disabled. This
module never imports the SDK or starts subprocesses itself. The callback owns
its subprocess timeout and cleanup; ``close`` drains and joins this owned thread
before the owner runs its final extraction/identification/publication phase.
"""
from threading import Condition, Thread


class IncrementalExtraction:
    """Run at most one callback, retaining only one pending notification.

    Call ``schedule`` after a completed episode and ``close`` in the replay
    owner's cleanup path. Failures are re-raised by the next schedule or close;
    a failed callback is never silently retried. Close is idempotent, drains the
    pending notification, and rejects any later schedule.
    """
    def __init__(self, callback):
        if not callable(callback):
            raise TypeError('incremental extraction callback must be callable')
        self._callback = callback
        self._condition = Condition()
        self._pending = False
        self._closed = False
        self._failure = None
        self._thread = None

    def schedule(self):
        with self._condition:
            self._raise_failure()
            if self._closed:
                raise RuntimeError('incremental extraction is closed')
            self._pending = True
            if self._thread is None:
                self._thread = Thread(target=self._run, name='placement-incremental-extraction', daemon=False)
                self._thread.start()
            self._condition.notify()

    def _raise_failure(self):
        if self._failure is not None:
            raise RuntimeError('incremental extraction failed') from self._failure

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending or self._closed)
                if not self._pending:
                    return
                self._pending = False
            try:
                self._callback()
            except BaseException as error:
                with self._condition:
                    self._failure = error
                    self._pending = False
                    self._closed = True
                    self._condition.notify_all()
                return

    def close(self):
        with self._condition:
            self._closed = True
            thread = self._thread
            self._condition.notify_all()
        if thread is not None:
            thread.join()
        with self._condition:
            self._raise_failure()
