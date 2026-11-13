"""Bounded file transfer and single-range HTTP semantics."""
from email.utils import formatdate, parsedate_to_datetime
import mimetypes
import re


class InvalidRange(ValueError):
    pass


def byte_range(header, size):
    if header is None:
        return 0, size - 1, 200
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', header.strip())
    if not match or not size or not any(match.groups()):
        raise InvalidRange('single byte range required')
    first, last = match.groups()
    if not first:
        count = int(last)
        if count <= 0:
            raise InvalidRange('empty suffix')
        start, end = max(0, size - count), size - 1
    else:
        start, end = int(first), min(int(last), size - 1) if last else size - 1
        if start >= size or start > end:
            raise InvalidRange('range outside resource')
    return start, end, 206


def send_file(handler, path, *, etag=None, stats=None):
    """Never read an entire video. On disconnect, close the file immediately."""
    stat = path.stat()
    etag = '"' + (etag or f'{stat.st_size:x}-{stat.st_mtime_ns:x}') + '"'
    modified = formatdate(stat.st_mtime, usegmt=True)
    headers = {'ETag': etag, 'Last-Modified': modified, 'Accept-Ranges': 'bytes',
               'Content-Type': mimetypes.guess_type(str(path))[0] or 'application/octet-stream',
               'Cache-Control': 'private, max-age=0, must-revalidate'}
    status = 200
    if handler.headers.get('If-None-Match') and etag in handler.headers['If-None-Match'].split(', '):
        status = 304
    elif not handler.headers.get('If-None-Match') and handler.headers.get('If-Modified-Since'):
        try:
            if int(stat.st_mtime) <= parsedate_to_datetime(handler.headers['If-Modified-Since']).timestamp():
                status = 304
        except (ValueError, TypeError, OverflowError):
            pass
    requested = handler.headers.get('Range')
    if handler.headers.get('If-Range') not in (None, etag, modified):
        requested = None
    try:
        start, end, ranged_status = byte_range(requested, stat.st_size)
        if status != 304:
            status = ranged_status
        if status == 206:
            headers['Content-Range'] = f'bytes {start}-{end}/{stat.st_size}'
        headers['Content-Length'] = str(end - start + 1)
    except InvalidRange:
        start, end, status = 0, -1, 416
        headers['Content-Range'] = f'bytes */{stat.st_size}'
        headers['Content-Length'] = '0'
    handler.send_response(status)
    for key, value in headers.items():
        if status != 304 or key != 'Content-Length':
            handler.send_header(key, value)
    handler.end_headers()
    if handler.command == 'HEAD' or status in (304, 416):
        return
    remaining = end - start + 1
    with path.open('rb') as stream:
        stream.seek(start)
        while remaining:
            chunk = stream.read(min(64 * 1024, remaining))
            if not chunk:
                break
            if stats is not None:
                stats['read_bytes'] += len(chunk)
            handler.wfile.write(chunk)
            remaining -= len(chunk)
