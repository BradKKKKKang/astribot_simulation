"""Same-origin LAN gateway, shared single-worker queue and immutable artifacts."""
import argparse
import asyncio
import fcntl
import io
import json
import logging
import os
from pathlib import Path
import re
import time

from aiohttp import ClientSession, ClientTimeout, web

from .catalog import Catalog, RemoteCatalog, admission
from .common import REPO, atomic_json, confined, digest, load_config, read_json
from .runtime import sdk_environment, sdk_fingerprint, usd_environment

LOG = logging.getLogger(__name__)
KEY = re.compile(r'^[0-9a-f]{64}$')


class ReviewService:
    def __init__(self, config):
        self.config = config
        self.cache = Path(config['cache'])
        self.cache.mkdir(parents=True, exist_ok=True)
        self.catalog = None
        self.jobs = {}
        self.episode_jobs = {}
        self.pending = asyncio.Queue()
        self.locks = {}
        self.process = None
        self.started = time.monotonic()

    async def start(self):
        self.lock_file = (self.cache / 'gateway.lock').open('a')
        fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.session = ClientSession(timeout=ClientTimeout(total=None, connect=10, sock_read=60), auto_decompress=False)
        self.sdk_env = await asyncio.to_thread(sdk_environment, self.config)
        self.sdk_version = await asyncio.to_thread(sdk_fingerprint, self.config)
        # Cache identity follows solver sources. UI, transport and lifecycle
        # changes do not invalidate unchanged numerical work. Retain a recorded
        # build identity when its complete solver-source manifest still matches.
        code_root = Path(__file__).parent
        core = digest([(name, digest((code_root / name).read_bytes())) for name in
                       ['common.py', 'fk_worker.py', 'runtime.py', 'sdk_worker.py', 'trajectory.py']])
        version_path = self.cache / 'compute-version.json'
        recorded = read_json(version_path) if version_path.exists() else {}
        self.implementation = recorded['implementation'] if recorded.get('core') == core else core
        atomic_json(version_path, {'core': core, 'implementation': self.implementation})
        self.model = read_json(self.cache / 'model.json')
        self.initial_pose = read_json(self.config['initial_pose'])
        await self.refresh_catalog()
        for path in (self.cache / 'results').glob('*/result.json'):
            result = read_json(path)
            provenance = result['provenance']
            metadata_path = self.cache / 'metadata' / (result['episode_id'] + '.json')
            if not metadata_path.exists():
                continue
            metadata = read_json(metadata_path)
            if (provenance.get('implementation') == self.implementation
                    and provenance.get('sdk') == self.sdk_version
                    and provenance.get('model') == self.model['kinematics_fingerprint']
                    and provenance.get('initial_pose') == digest(self.initial_pose)
                    and provenance.get('tool') == self.config['tool_offsets_m']
                    and provenance.get('thresholds') == self.config['thresholds']
                    and metadata.get('manifest') == self.catalog.version
                    and provenance.get('source') == metadata['value']['source_version']):
                self.episode_jobs[(result['episode_id'], result['mode'])] = {
                    'status': 'complete', 'key': result['key'],
                    'anomaly_count': len(result['summary']['anomaly_frames'])}
        self.worker = asyncio.create_task(self.work())
        for path in sorted((self.cache / 'work').glob('*/job.json')):
            job = read_json(path)
            if job.get('config', {}).get('source_mode', 'sample') != self.config.get('source_mode', 'sample'):
                continue
            if (job['provenance'].get('implementation') == self.implementation
                    and job['provenance'].get('sdk') == self.sdk_version
                    and job['model']['kinematics_fingerprint'] == self.model['kinematics_fingerprint']):
                self.jobs[job['key']] = {'key': job['key'], 'status': 'queued', 'recovered': True}
                self.episode_jobs[(job['episode_id'], job['mode'])] = self.jobs[job['key']]
                await self.pending.put(job)

    async def close(self):
        self.worker.cancel()
        process = self.process
        if process and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        await asyncio.gather(self.worker, return_exceptions=True)
        await self.session.close()
        self.lock_file.close()

    async def upstream_json(self, path):
        async with self.session.get(self.config['upstream'] + path) as response:
            if response.status != 200:
                raise web.HTTPBadGateway(text='cloud reader unavailable')
            return await response.json()

    async def refresh_catalog(self):
        path = self.cache / 'manifest.json'
        expected = self.config.get('source_mode', 'sample')
        def check_mode(value):
            if value.get('source_kind', 'sample') != expected:
                raise ValueError(f'catalog source mode differs from configured {expected}; wait for the matching reader')
        try:
            manifest = await self.upstream_json('/manifest')
            check_mode(manifest)
            atomic_json(path, manifest)
        except Exception:
            if not path.exists():
                raise
            manifest = read_json(path)
            check_mode(manifest)
        if manifest.get('source_kind') == 'formal':
            self.catalog = RemoteCatalog(manifest)
        else:
            self.catalog = Catalog(manifest, self.config['unit_evidence'])

    async def synchronize_catalog(self, version):
        """Replace a startup fallback once the reader exposes a newer snapshot."""
        if self.catalog.version == version:
            return
        async with self.locks.setdefault('catalog-refresh', asyncio.Lock()):
            if self.catalog.version != version:
                await self.refresh_catalog()
            if self.catalog.version != version:
                raise web.HTTPServiceUnavailable(text='catalog version changed; retry after the reader reconnects')

    async def episode(self, key):
        if not isinstance(self.catalog, RemoteCatalog) and key not in self.catalog.entries:
            raise web.HTTPNotFound()
        async with self.locks.setdefault('metadata:' + key, asyncio.Lock()):
            path = self.cache / 'metadata' / (key + '.json')
            # Manifest immutable snapshot identifies the metadata cache.
            if path.exists():
                cached = read_json(path)
                if cached.get('manifest') == self.catalog.version:
                    if isinstance(self.catalog, RemoteCatalog):
                        self.catalog.entries[key] = cached['value']['catalog_entry']
                    return cached['value']
            value = await self.upstream_json('/episodes/' + key + '/metadata')
            if isinstance(self.catalog, RemoteCatalog):
                await self.synchronize_catalog(value['catalog_entry']['source_version'])
                self.catalog.entries[key] = value['catalog_entry']
            atomic_json(path, {'manifest': self.catalog.version, 'value': value})
            return value

    def public_episode(self, key, metadata):
        row = {k: v for k, v in self.catalog.entries[key].items() if not k.startswith('_')}
        group = self.catalog.groups[row['group']]
        review = admission(group, metadata['robot_type'], self.config['unit_evidence'].get(row['group']))
        source = metadata['source']
        videos = []
        for video in source['videos']:
            output = video['output']
            videos.append({'camera': video['camera'], 'codec': output['codec'], 'fps': output['fps'],
                           'frames': output['frames'], 'width': output['width'], 'height': output['height'],
                           'version': digest(output), 'selection': video['selection'],
                           'kind': video.get('kind', 'sample_video'), 'requires_preview': video.get('requires_preview', False),
                           'url': f'/media/{key}/{video["camera"]}',
                           'preview_url': f'/preview/{key}/{video["camera"]}'})
        return {**row, 'review': review, 'source_version': metadata['source_version'],
                'robot_type': metadata['robot_type'], 'frames': source['frames'], 'fps': source['fps'],
                'task': ' / '.join(source.get('source_episode_metadata', {}).get('tasks', [])) or row['root'].split('/')[-1],
                'videos': videos, 'basis_warning': '源 base 暂按 SDK chassis；跨本体基座尚未对齐',
                'reference_frames': {'data_to_chassis': self.config['source_base_to_chassis'],
                                     'chassis_to_usd_translation_m': self.config['sdk_to_usd_translation_m'],
                                     'alignment_status': 'assumed_identity', 'alignment_verified': False},
                'sample_time_column': source['sample_time_column']}

    async def trajectory(self, key):
        from .trajectory import adapt
        import pyarrow.parquet as pq
        metadata = await self.episode(key)
        async with self.locks.setdefault('trajectory:' + key, asyncio.Lock()):
            adapter_version = digest(Path(__file__).with_name('trajectory.py').read_bytes())
            path = self.cache / 'trajectories' / (digest([key, metadata['source_version'], adapter_version]) + '.json')
            if path.exists():
                return read_json(path)
            async with self.session.get(self.config['upstream'] + f'/episodes/{key}/eef.parquet') as response:
                if response.status != 200:
                    raise web.HTTPBadGateway(text='trajectory unavailable')
                content = await response.read()
            value = await asyncio.to_thread(lambda: adapt(pq.read_table(io.BytesIO(content))))
            if value['frame_count'] != metadata['source']['frames']:
                raise ValueError('parquet frame count differs from source metadata')
            value['parquet_sha256'] = digest(content)
            atomic_json(path, value)
            return value

    async def enqueue(self, key, mode):
        if mode not in ('state', 'action'):
            raise web.HTTPBadRequest(text='mode must be state or action')
        metadata = await self.episode(key)
        episode = self.public_episode(key, metadata)
        if episode['review']['status'] != 'admitted':
            raise web.HTTPUnprocessableEntity(text=episode['review']['reason'])
        trajectory = await self.trajectory(key)
        provenance = {'schema': 1, 'source': metadata['source_version'], 'mapping': episode['mapping_version'],
                      'mode': mode, 'selection': 'all_frames', 'parquet': trajectory['parquet_sha256'],
                      'sdk': self.sdk_version, 'model': self.model['kinematics_fingerprint'],
                      'tool': self.config['tool_offsets_m'], 'initial_pose': digest(self.initial_pose),
                      'base_assumption': self.config['source_base_to_chassis'],
                      'sdk_to_usd': self.config['sdk_to_usd_translation_m'],
                      'strategy': 'acknowledged_continuous_seed_hold_initial_torso_v1',
                      'unit_evidence': episode['review'].get('unit_evidence'),
                      'thresholds': self.config['thresholds'], 'implementation': self.implementation}
        cache_key = digest(provenance)
        result_path = self.cache / 'results' / cache_key / 'result.json'
        if result_path.exists():
            self.episode_jobs[(key, mode)] = {'key': cache_key, 'status': 'complete',
                                            'anomaly_count': len(read_json(result_path)['summary']['anomaly_frames'])}
            return {'key': cache_key, 'status': 'complete', 'cache_hit': True}
        if cache_key in self.jobs and self.jobs[cache_key]['status'] in ('queued', 'running'):
            return {**self.jobs[cache_key], 'deduplicated': True}
        job = {'key': cache_key, 'episode_id': key, 'mode': mode, 'config': self.config,
               'initial_pose': self.initial_pose, 'model': self.model, 'provenance': provenance, 'trajectory': trajectory}
        atomic_json(self.cache / 'work' / cache_key / 'job.json', job)
        self.jobs[cache_key] = {'key': cache_key, 'status': 'queued'}
        self.episode_jobs[(key, mode)] = self.jobs[cache_key]
        await self.pending.put(job)
        return self.jobs[cache_key]

    async def work(self):
        while True:
            job = await self.pending.get()
            key = job['key']
            directory = self.cache / 'work' / key
            self.jobs[key] = {'key': key, 'status': 'running'}
            self.episode_jobs[(job['episode_id'], job['mode'])] = self.jobs[key]
            try:
                with (directory / 'worker.log').open('ab') as log:
                    for command, env in [(['/usr/bin/python3', '-m', 'src.eef_review.sdk_worker', str(directory / 'job.json')], self.sdk_env),
                                         ([self.config['usd_python'], '-m', 'src.eef_review.fk_worker', '--job', str(directory / 'job.json')], usd_environment(self.config))]:
                        self.process = await asyncio.create_subprocess_exec(*command, env=env, cwd=REPO, stdout=log, stderr=log)
                        try:
                            code = await asyncio.wait_for(self.process.wait(), 3600)
                        except asyncio.TimeoutError:
                            self.process.kill()
                            await self.process.wait()
                            raise RuntimeError('worker exceeded one-hour timeout')
                        if code:
                            raise RuntimeError(f'worker exited {code}; inspect outputs/eef_review/work/' + key + '/worker.log')
                target = self.cache / 'results' / key
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(directory, target)
                self.jobs[key] = {'key': key, 'status': 'complete', 'cache_hit': False,
                                  'anomaly_count': len(read_json(target / 'result.json')['summary']['anomaly_frames'])}
            except Exception as error:
                LOG.exception('compute failed: %s', key)
                self.jobs[key] = {'key': key, 'status': 'failed', 'error': str(error), 'resumable': True}
                atomic_json(directory / 'failure.json', self.jobs[key])
            finally:
                self.episode_jobs[(job['episode_id'], job['mode'])] = self.jobs[key]
                self.process = None
                self.pending.task_done()


def make_app(config):
    service = ReviewService(config)

    @web.middleware
    async def errors(request, handler):
        try:
            return await handler(request)
        except web.HTTPException:
            raise
        except (KeyError, FileNotFoundError):
            return web.json_response({'error': 'resource not found'}, status=404)
        except ValueError as error:
            return web.json_response({'error': str(error)}, status=422)
        except Exception:
            LOG.exception('request failed')
            return web.json_response({'error': 'service unavailable; retry after connection recovers'}, status=503)

    app = web.Application(middlewares=[errors], client_max_size=16384)
    app['review'] = service

    async def startup(_):
        await service.start()

    async def cleanup(_):
        await service.close()

    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)

    async def health(request):
        try:
            remote = await asyncio.wait_for(service.upstream_json('/health'), 3)
            if isinstance(service.catalog, RemoteCatalog):
                await service.synchronize_catalog(remote['source_version'])
            connected = True
        except Exception:
            remote, connected = {}, False
        return web.json_response({'status': 'ok', 'cloud_connected': connected, 'media': remote,
                                  'queue': service.pending.qsize(), 'uptime_s': time.monotonic() - service.started})

    async def catalog(request):
        query = request.query.get('q', '').lower()
        review = request.query.get('review', '')
        group = request.query.get('group', '')
        processing = request.query.get('processing', '')
        mode = request.query.get('mode', 'state')
        if isinstance(service.catalog, RemoteCatalog):
            arguments = dict(request.query)
            states = {key: state for (key, channel), state in service.episode_jobs.items() if channel == mode}
            if processing == 'uncomputed':
                arguments['exclude_ids'] = list(states)
            elif processing:
                arguments['ids'] = [key for key, state in states.items()
                                    if processing == state['status'] or (processing == 'anomaly' and state.get('anomaly_count', 0) > 0)]
            async with service.session.post(service.config['upstream'] + '/catalog/query', json=arguments) as response:
                if response.status != 200:
                    raise web.HTTPBadGateway(text='formal catalog unavailable')
                result = await response.json()
            await service.synchronize_catalog(result['source_version'])
            for row in result['episodes']:
                row['computation'] = states.get(row['id'], {'status': 'uncomputed'})
            return web.json_response(result)
        rows = service.catalog.public(request.query.get('all') == '1')
        rows = [{**row, 'computation': service.episode_jobs.get((row['id'], mode), {'status': 'uncomputed'})} for row in rows]
        rows = [r for r in rows if (not query or query in (r['root'] + r['dataset'] + r['profile']).lower())
                and (not review or review == r['review']['status']) and (not group or group == r['group'])
                and (not processing or processing == r['computation']['status']
                     or (processing == 'anomaly' and r['computation'].get('anomaly_count', 0) > 0))]
        offset = max(0, int(request.query.get('offset', 0)))
        limit = min(200, max(1, int(request.query.get('limit', 80))))
        return web.json_response({'total': len(rows), 'episodes': rows[offset:offset + limit],
                                  'source_version': service.catalog.version, 'groups': sorted(set(r['group'] for r in service.catalog.public()))})

    async def sample(request):
        if isinstance(service.catalog, RemoteCatalog):
            count = min(5, max(1, int(request.query.get('per_group', 1))))
            return web.json_response(await service.upstream_json(f'/sample?per_group={count}'))
        return web.json_response({'episode_ids': service.catalog.sample(min(5, max(1, int(request.query.get('per_group', 1)))))})

    async def episode(request):
        key = request.match_info['episode']
        return web.json_response(service.public_episode(key, await service.episode(key)))

    async def trajectory(request):
        return web.json_response(await service.trajectory(request.match_info['episode']))

    async def compute(request):
        body = await request.json()
        return web.json_response(await service.enqueue(request.match_info['episode'], body.get('mode', 'state')))

    def result_directory(request):
        key = request.match_info['key']
        if not KEY.fullmatch(key):
            raise web.HTTPNotFound()
        return service.cache / 'results' / key

    async def status(request):
        directory = result_directory(request)
        key = request.match_info['key']
        if (directory / 'result.json').exists():
            return web.json_response(service.jobs.get(key, {'key': key, 'status': 'complete', 'cache_hit': True}))
        state = service.jobs.get(key)
        if state is None:
            raise web.HTTPNotFound()
        progress = service.cache / 'work' / key / 'progress.json'
        return web.json_response({**(read_json(progress) if progress.exists() else {}), **state})

    async def result(request):
        return web.FileResponse(result_directory(request) / 'result.json')

    async def chunk(request):
        number = int(request.match_info['number'])
        manifest = read_json(result_directory(request) / 'result.json')
        if not 0 <= number < manifest['chunks']:
            raise web.HTTPNotFound()
        return web.FileResponse(result_directory(request) / f'chunk-{number:05d}.json')

    async def model_metadata(request):
        return web.json_response(service.model)

    async def model_file(request):
        if request.match_info['key'] != service.model['geometry_key']:
            raise web.HTTPNotFound()
        return web.FileResponse(service.cache / 'models' / service.model['geometry_key'] / 'robot.glb')

    async def proxy(request):
        key, camera = request.match_info['episode'], request.match_info['camera']
        metadata = await service.episode(key)
        if camera not in [v['camera'] for v in metadata['source']['videos']]:
            raise web.HTTPNotFound()
        from urllib.parse import quote
        route = 'lease' if request.path.startswith('/lease/') else 'preview' if request.path.startswith('/preview/') else 'media'
        headers = {k: request.headers[k] for k in ('Range', 'If-Range', 'If-None-Match', 'If-Modified-Since') if k in request.headers}
        async with service.session.request(request.method, service.config['upstream'] + f'/{route}/{key}/' + quote(camera, safe=''), headers=headers) as upstream:
            output_headers = {k: v for k, v in upstream.headers.items() if k.lower() in {
                'content-type', 'content-length', 'content-range', 'accept-ranges', 'etag', 'last-modified', 'cache-control'}}
            response = web.StreamResponse(status=upstream.status, headers=output_headers)
            await response.prepare(request)
            if request.method != 'HEAD':
                async for data in upstream.content.iter_chunked(65536):
                    await response.write(data)
            return response

    async def static(request):
        path = confined(config['frontend'], request.match_info.get('path', '') or 'index.html')
        if not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    app.router.add_get('/api/health', health)
    app.router.add_get('/api/catalog', catalog)
    app.router.add_get('/api/sample', sample)
    app.router.add_get('/api/model', model_metadata)
    app.router.add_get('/api/episodes/{episode}', episode)
    app.router.add_get('/api/episodes/{episode}/trajectory', trajectory)
    app.router.add_post('/api/episodes/{episode}/compute', compute)
    app.router.add_get('/api/jobs/{key}', status)
    app.router.add_get('/api/results/{key}', result)
    app.router.add_get('/api/results/{key}/chunks/{number}', chunk)
    app.router.add_get('/artifacts/models/{key}/robot.glb', model_file)
    app.router.add_get('/media/{episode}/{camera}', proxy)
    app.router.add_get('/preview/{episode}/{camera}', proxy)
    app.router.add_get('/lease/{episode}/{camera}', proxy)
    app.router.add_get('/{path:.*}', static)
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()
    config = load_config(args.config)
    logging.basicConfig(level=logging.INFO)
    web.run_app(make_app(config), host=config['host'], port=config['port'], handler_cancellation=True)


if __name__ == '__main__':
    main()
