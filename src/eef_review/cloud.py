"""Loopback-only dataset/media server, deployed independently of SDK and USD."""
import argparse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit, unquote, parse_qs

from .catalog import Catalog
from .common import confined, digest, read_json
from .media import send_file
from .preview import PreviewCache


class Reader:
    def __init__(self, root, cache):
        self.root = Path(root).resolve()
        self.catalog = Catalog(read_json(self.root / 'SAMPLE_MANIFEST.json'))
        self.group_digest = digest(read_json(self.root / 'MAPPING_GROUPS.json'))
        self.preview = PreviewCache(cache)
        self.stats = {'read_bytes': 0, 'requests': 0, 'disconnects': 0}

    def episode(self, key):
        row = self.catalog.entries[key]
        directory = confined(self.root, row['_directory'])
        return row, directory

    def metadata(self, key):
        row, directory = self.episode(key)
        source = read_json(directory / 'source.json')
        mapping = read_json(directory.parent / 'MAPPING.json')
        info = read_json(directory.parent / 'SOURCE_INFO.json')
        if (source['source_dataset'], source['source_root'], source['source_episode_index']) != (row['dataset'], row['root'], row['episode']):
            raise ValueError('source identity differs from manifest')
        if digest(mapping['contract']) != row['mapping_version']:
            raise ValueError('root mapping differs from manifest')
        return {'source': source, 'mapping': mapping, 'robot_type': info.get('robot_type', ''),
                'source_version': digest([self.catalog.version, mapping, source]),
                'mapping_groups_version': self.group_digest}

    def camera(self, key, camera):
        _, directory = self.episode(key)
        metadata = self.metadata(key)
        video = next(v for v in metadata['source']['videos'] if v['camera'] == camera)
        path = confined(directory.parent, video['output_name'])
        if not path.is_relative_to(directory):
            raise ValueError('camera outside episode')
        return path, video

    @contextmanager
    def data_file(self, key):
        _, directory = self.episode(key)
        yield directory / 'eef.parquet'


def handler_for(reader):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def json(self, value, status=200):
            content = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(content)

        def do_HEAD(self):
            self.do_GET()

        def do_POST(self):
            try:
                if urlsplit(self.path).path != '/catalog/query' or not hasattr(reader.catalog, 'query'):
                    raise KeyError('route')
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 16 * 1024**2:
                    raise ValueError('invalid catalog query length')
                self.json(reader.catalog.query(json.loads(self.rfile.read(length))))
            except (KeyError, ValueError) as error:
                self.json({'error': str(error)}, 400)

        def do_GET(self):
            reader.stats['requests'] += 1
            parts = [unquote(p) for p in urlsplit(self.path).path.strip('/').split('/')]
            try:
                if parts == ['health']:
                    self.json({'status': 'ok', 'source_version': reader.catalog.version, **reader.stats})
                elif parts == ['manifest']:
                    self.json(reader.catalog.manifest)
                elif parts == ['sample']:
                    query = parse_qs(urlsplit(self.path).query)
                    self.json({'episode_ids': reader.catalog.sample(min(5, max(1, int(query.get('per_group', ['1'])[0]))))})
                elif len(parts) == 3 and parts[0] == 'episodes':
                    key, resource = parts[1:]
                    if resource == 'metadata':
                        self.json(reader.metadata(key))
                    elif resource == 'eef.parquet':
                        with reader.data_file(key) as path:
                            send_file(self, path, stats=reader.stats)
                    else:
                        raise KeyError(resource)
                elif len(parts) == 3 and parts[0] == 'media':
                    path, video = reader.camera(*parts[1:])
                    if video.get('requires_preview'):
                        result = reader.preview.request(path, video['output'], video['output']['fps'], video.get('clip'))
                        if result['status'] == 'complete':
                            with reader.preview.acquire(result['key']) as output:
                                send_file(self, output, etag=result['key'], stats=reader.stats)
                        else:
                            self.json(result, 503 if result['status'] == 'failed' else 202)
                    else:
                        send_file(self, path, etag=digest(video['output']), stats=reader.stats)
                elif len(parts) == 3 and parts[0] == 'preview':
                    path, video = reader.camera(*parts[1:])
                    result = reader.preview.request(path, video['output'], video['output']['fps'], video.get('clip'))
                    if result['status'] == 'complete':
                        with reader.preview.acquire(result['key']) as output:
                            send_file(self, output, etag=result['key'], stats=reader.stats)
                    else:
                        self.json(result, 503 if result['status'] == 'failed' else 202)
                elif len(parts) == 3 and parts[0] == 'lease':
                    path, video = reader.camera(*parts[1:])
                    key = reader.preview.key(video['output'], video['output']['fps'], video.get('clip'))
                    self.json({'retained': reader.preview.renew(key)})
                else:
                    raise KeyError('route')
            except (BrokenPipeError, ConnectionResetError):
                reader.stats['disconnects'] += 1
            except (KeyError, StopIteration, FileNotFoundError):
                self.json({'error': 'registered resource not found'}, 404)
            except ValueError as error:
                self.json({'error': str(error)}, 422)
            except Exception:
                self.json({'error': 'reader unavailable; inspect remote service log'}, 503)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--cache', default=str(Path.home() / '.cache/astribot-eef-review/media'))
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--mode', choices=['sample', 'formal'], default='sample')
    parser.add_argument('--index', default=str(Path.home() / '.cache/astribot-eef-review/formal/index.sqlite'))
    args = parser.parse_args()
    if args.mode == 'formal':
        from .formal_reader import FormalReader
        reader = FormalReader(args.root, args.cache, args.index)
    else:
        reader = Reader(args.root, args.cache)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(reader))
    server.daemon_threads = True
    print(f'EEF reader ready: {len(reader.catalog.entries)} episodes', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
