"""One-worker, versioned H.264 preview cache with timestamp validation."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import threading
import time

from .common import atomic_json, digest


def probe(path):
    process = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                              '-show_frames', '-show_streams', '-show_entries',
                              'frame=best_effort_timestamp_time:stream=width,height,codec_name',
                              '-of', 'json', str(path)], capture_output=True, check=True, timeout=600)
    result = json.loads(process.stdout)
    return result['streams'][0], [float(x['best_effort_timestamp_time']) for x in result['frames']]


def probe_interval(path, selection):
    start, end = max(0, selection['start'] - .1), selection['end'] + .1
    process = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                              '-read_intervals', f'{start:.9f}%{end:.9f}', '-show_frames',
                              '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(path)],
                             capture_output=True, check=True, timeout=600)
    timestamps = [float(row['best_effort_timestamp_time']) for row in json.loads(process.stdout)['frames']]
    return [t for t in timestamps if t >= selection['start'] - .0001][:selection['frames']]


class PreviewCache:
    def __init__(self, root, limit=2 * 1024**3):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.limit = limit
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.lock = threading.RLock()
        self.active = Counter()
        self.leases = {}
        self.jobs = {}

    @staticmethod
    def key(version, fps, selection=None):
        value = {'version': version, 'fps': fps, 'encoding': 'h264-crf23-gop1s-v1'}
        if selection is not None:
            value.update(selection=selection, clip_version=1)
        return digest(value)

    def request(self, path, version, fps, selection=None):
        key = self.key(version, fps, selection)
        with self.lock:
            output = self.root / (key + '.mp4')
            if output.exists():
                output.touch()
                return {'key': key, 'status': 'complete'}
            if key not in self.jobs or self.jobs[key]['status'] not in ('queued', 'running'):
                self.jobs[key] = {'key': key, 'status': 'queued'}
                self.executor.submit(self._convert, key, path, fps, selection)
            return dict(self.jobs[key])

    def _convert(self, key, path, fps, selection=None):
        temporary = self.root / (key + '.partial.mp4')
        started = time.monotonic()
        try:
            self.jobs[key] = {'key': key, 'status': 'running'}
            # Reserve before starting; enforce the hard limit before publication too.
            estimate = path.stat().st_size * 4 if selection is None else max(2 * 1024**2,
                int(selection['frames'] * selection['width'] * selection['height'] * .1))
            self.evict(min(estimate, self.limit))
            inputs, filters = ['-i', str(path)], []
            if selection is not None:
                # Keep absolute input PTS while seeking near the requested episode;
                # trim at its source timestamp, then reset the clip to time zero.
                inputs = ['-copyts', '-ss', str(max(0, selection['start'] - .1)), '-i', str(path)]
                filters = ['-vf', f'trim=start={max(0, selection["start"] - .0001):.9f},setpts=PTS-STARTPTS',
                           '-frames:v', str(selection['frames'])]
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', *inputs,
                            '-map', '0:v:0', '-an', '-c:v', 'libx264', '-preset', 'fast',
                            '-crf', '23', '-pix_fmt', 'yuv420p', '-g', str(max(1, round(fps))),
                            '-keyint_min', str(max(1, round(fps))), '-sc_threshold', '0',
                            *filters, '-vsync', '0', '-movflags', '+faststart', '-fs', str(self.limit),
                            str(temporary)], check=True, capture_output=True, timeout=1800)
            if selection is None:
                before, t0 = probe(path)
            else:
                before = {'width': selection['width'], 'height': selection['height']}
                source_pts = probe_interval(path, selection)
                if len(source_pts) != selection['frames'] or abs(source_pts[0] - selection['start']) > .001:
                    raise ValueError('source episode frame selection does not match metadata')
                t0 = [t - source_pts[0] for t in source_pts]
                if any(abs(t - index / fps) > .001 for index, t in enumerate(t0)):
                    raise ValueError('source video cadence differs from EEF timestamps; explicit resampling is required')
            after, t1 = probe(temporary)
            if len(t0) != len(t1) or not t0 or (before['width'], before['height']) != (after['width'], after['height']):
                raise ValueError('preview changed frame count or resolution')
            error = max(abs(a - b) for a, b in zip(t0, t1))
            if error > 0.001:
                raise ValueError('preview changed presentation timestamps')
            with self.lock:
                self.evict(temporary.stat().st_size)
                temporary.replace(self.root / (key + '.mp4'))
                report = {'key': key, 'status': 'complete', 'frames': len(t1),
                          'max_pts_error_s': error, 'elapsed_s': time.monotonic() - started}
                if selection is not None:
                    report.update(selection=selection, first_source_pts=source_pts[0], last_source_pts=source_pts[-1])
                atomic_json(self.root / (key + '.json'), report)
                self.jobs[key] = report
        except Exception as error:
            temporary.unlink(missing_ok=True)
            self.jobs[key] = {'key': key, 'status': 'failed', 'error': str(error)}

    def evict(self, needed=0):
        with self.lock:
            files = sorted((p for p in self.root.glob('*.mp4') if '.partial.' not in p.name),
                           key=lambda p: p.stat().st_mtime)
            used = sum(p.stat().st_size for p in files)
            for path in files:
                if used + needed <= self.limit:
                    break
                if self.active[path.stem] or self.leases.get(path.stem, 0) > time.monotonic():
                    continue
                used -= path.stat().st_size
                path.unlink()
                path.with_suffix('.json').unlink(missing_ok=True)
            if used + needed > self.limit:
                raise OSError('preview cache full; active videos are protected')

    @contextmanager
    def acquire(self, key):
        with self.lock:
            path = self.root / (key + '.mp4')
            if not path.is_file():
                raise FileNotFoundError('preview not ready')
            self.active[key] += 1
            self.leases[key] = time.monotonic() + 120
        try:
            yield path
        finally:
            with self.lock:
                self.active[key] -= 1

    def renew(self, key):
        with self.lock:
            if not (self.root / (key + '.mp4')).is_file():
                return False
            self.leases[key] = time.monotonic() + 120
            return True
