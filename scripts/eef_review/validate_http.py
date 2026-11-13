"""Reproducible live checks; saves actual Range bytes and source audit results."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://10.11.12.73:8780')
    parser.add_argument('--output', default='outputs/eef_review/http-validation.json')
    parser.add_argument('--per-group', type=int, default=5)
    parser.add_argument('--episode', action='append', help='validate only the specified episode IDs')
    args = parser.parse_args()

    def read(path, method='GET', headers=None):
        started = time.monotonic()
        try:
            response = urlopen(Request(args.url + path, method=method, headers=headers or {}), timeout=90)
        except HTTPError as error:
            response = error
        with response:
            content = response.read()
            return response.status, {k.lower(): v for k, v in response.headers.items()}, content, time.monotonic() - started

    def data(path):
        status, _, content, _ = read(path)
        assert status == 200, (path, status, content[:200])
        return json.loads(content)

    before = data('/api/health')
    ids = args.episode or data(f'/api/sample?per_group={args.per_group}')['episode_ids']
    model = data('/api/model')
    assert model['glb_url'].startswith('/artifacts/')
    status, headers, body, elapsed = read(model['glb_url'])
    assert status == 200 and body[:4] == b'glTF'
    glb = {'bytes': len(body), 'elapsed_s': elapsed}

    def check(key):
        metadata = data('/api/episodes/' + key)
        assert metadata['review']['status'] == 'admitted', metadata['review']
        source = data('/api/episodes/' + key + '/trajectory')
        assert source['frame_count'] == metadata['frames']
        assert source['frames'][0]['sample_timestamp'] == 0
        assert all(a['sample_timestamp'] < b['sample_timestamp'] for a, b in zip(source['frames'], source['frames'][1:]))
        for frame in source['frames']:
            for mode in ('state', 'action'):
                for side, first in [('left', 9), ('right', 19)]:
                    if not all(frame['mask'][first:first + 9]):
                        assert frame[mode]['targets'][side] is None
        media = []
        for camera in metadata['videos']:
            url = camera['url']
            assert url.startswith('/media/')
            deadline = time.monotonic() + 600
            while True:
                status, headers, body, _ = read(url, 'HEAD')
                if status != 202 or time.monotonic() >= deadline:
                    break
                time.sleep(1)
            assert status == 200 and not body
            length = int(headers['content-length'])
            for header in ['bytes=0-99', f'bytes={length // 2}-{length // 2 + 99}', 'bytes=-100', f'bytes={length}-']:
                code, h, chunk, latency = read(url, headers={'Range': header})
                assert code == (416 if header == f'bytes={length}-' else 206)
                assert len(chunk) == (0 if code == 416 else 100)
                media.append({'camera': camera['camera'], 'range': header, 'status': code, 'bytes': len(chunk), 'elapsed_s': latency})
            status, _, _, _ = read(url, headers={'If-None-Match': headers['etag']})
            assert status == 304
        return {'episode_id': key, 'group': metadata['group'], 'frames': source['frame_count'],
                'fps': metadata['fps'], 'duration_s': source['duration'], 'source_version': metadata['source_version'],
                'invalid_pose_frames': {mode: {side: sum(f[mode]['targets'][side] is None for f in source['frames'])
                                               for side in ('left', 'right')} for mode in ('state', 'action')},
                'video_checks': media}

    with ThreadPoolExecutor(max_workers=4) as pool:
        episodes = list(pool.map(check, ids))
    after = data('/api/health')
    values = sorted(item['elapsed_s'] for episode in episodes for item in episode['video_checks'] if item['status'] == 206)
    result = {'status': 'passed', 'url': args.url, 'episodes': episodes, 'total_frames': sum(e['frames'] for e in episodes),
              'range_latency_s': {'p50': statistics.median(values), 'p95': values[int(.95 * (len(values) - 1))]},
              'glb': glb, 'cloud_before': before['media'], 'cloud_after': after['media'],
              'note': 'Cloud byte counters include other simultaneous users; per-request delivered bytes are recorded separately.'}
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'episodes'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
