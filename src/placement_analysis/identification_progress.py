"""Terminal progress and flushed stage logs for long-running identification."""
from contextlib import contextmanager
import sys
import threading
import time


class IdentificationProgress:
    def __init__(self, total, *, enabled=True, dry_run=False, stream=None, redact=None,
                 heartbeat_interval=10.0):
        self.total, self.completed = total, 0
        self.stream = stream if stream is not None else sys.stderr
        self.enabled, self.dry_run = enabled, dry_run
        self.redact = redact or (lambda message: message)
        self.heartbeat_interval = heartbeat_interval
        self.bar = None
        self.counts = dict(success=0, reused=0, failed=0)
        self.current = None
        self.phase = None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.worker = None
        self.started = time.monotonic()

    def log(self, message):
        line = self.redact(f'[{time.strftime("%H:%M:%S")}] {message}')
        if self.bar is not None:
            self.bar.write(line, file=self.stream)
            self.stream.flush()
        else:
            print(line, file=self.stream, flush=True)

    def __enter__(self):
        if self.enabled and self.stream.isatty():
            try:
                from tqdm.auto import tqdm
            except ImportError:
                self.log(f'tqdm 未安装，使用文本进度；可用 {sys.executable} -m pip install tqdm 安装')
            else:
                self.bar = tqdm(total=self.total, desc='Validate' if self.dry_run else 'JEV',
                                unit='event', file=self.stream, dynamic_ncols=True,
                                mininterval=.2, leave=True)
        self.worker = threading.Thread(target=self._heartbeat, name='identification-progress', daemon=True)
        self.worker.start()
        return self

    def _heartbeat(self):
        while not self.stop.wait(self.heartbeat_interval):
            with self.lock:
                phase = self.phase
            if phase is not None:
                label, started = phase
                elapsed = time.monotonic() - started
                if elapsed >= self.heartbeat_interval:
                    self.log(f'{label}；仍在处理中，已等待 {elapsed:.0f}s')

    def stage(self, message):
        label = self.redact(f'{self.current} | {message}' if self.current else message)
        with self.lock:
            self.phase = (label, time.monotonic())
        if self.bar is not None:
            # Keep counters readable in an 80-column terminal; the full stage
            # and event ID are in the adjacent log line and heartbeat.
            self.bar.set_postfix(ok=self.counts['success'], reuse=self.counts['reused'],
                                 fail=self.counts['failed'], refresh=True)
        self.log(label)

    @contextmanager
    def event(self, event_id, counts):
        self.counts = counts
        self.current = self.redact(f'[{self.completed + 1}/{self.total}] {event_id}')
        try:
            yield
        except BaseException:
            raise
        else:
            self.completed += 1
            if self.bar is not None:
                self.bar.set_postfix(ok=counts['success'], reuse=counts['reused'],
                                     fail=counts['failed'], refresh=False)
                self.bar.update(1)
        finally:
            with self.lock:
                self.phase = None
            self.current = None

    def __exit__(self, error_type, error, traceback):
        self.stop.set()
        if self.worker is not None:
            self.worker.join(timeout=1)
        if self.bar is not None:
            self.bar.close()
            self.bar = None
        if error_type is not None:
            self.log(f'识别中止：已完成 {self.completed}/{self.total} 个事件；'
                     '成功检查点已保留，重新执行原命令可继续')
        return False
