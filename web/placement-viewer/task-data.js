import { createReadStream } from 'node:fs';
import { lstat, readdir } from 'node:fs/promises';
import { join } from 'node:path';

async function taskFiles(root, name) {
  if (!name || name.startsWith('.') || name.includes('/') || name.includes('\\')) return null;
  const directory = join(root, name);
  try {
    if (!(await lstat(directory)).isDirectory()) return null;
    const files = await Promise.all(['placement.json', 'robot.glb'].map((file) =>
      lstat(join(directory, file), { bigint: true })));
    if (files.some((file) => !file.isFile())) return null;
    return { directory, revision: files.map((file) => `${file.ino}-${file.size}-${file.mtimeNs}`).join('_'),
      updated_at: Math.max(...files.map((file) => Number(file.mtimeMs))) };
  } catch (error) {
    if (error.code === 'ENOENT' || error.code === 'ENOTDIR') return null;
    throw error;
  }
}

export async function discoverTasks(root) {
  let entries;
  try { entries = await readdir(root, { withFileTypes: true }); }
  catch (error) {
    if (error.code === 'ENOENT') return [];
    throw error;
  }
  const tasks = await Promise.all(entries.filter((entry) => entry.isDirectory()).map(async (entry) => {
    const files = await taskFiles(root, entry.name);
    if (!files) return null;
    return { id: `folder:${entry.name}`, name: entry.name,
      bundle: `data/${encodeURIComponent(entry.name)}/placement.json`,
      revision: files.revision, updated_at: files.updated_at };
  }));
  return tasks.filter(Boolean).sort((a, b) => b.updated_at - a.updated_at || a.name.localeCompare(b.name));
}

export function taskDataMiddleware(root) {
  return async (req, res, next) => {
    const url = new URL(req.url, 'http://localhost');
    if (url.pathname !== '/api/tasks' && !url.pathname.startsWith('/data/')) return next();
    res.setHeader('Cache-Control', 'no-store');
    try {
      if (!['GET', 'HEAD'].includes(req.method)) { res.statusCode = 405; return res.end(); }
      if (url.pathname === '/api/tasks') {
        res.setHeader('Content-Type', 'application/json; charset=utf-8');
        const body = JSON.stringify({ tasks: await discoverTasks(root) });
        return res.end(req.method === 'HEAD' ? undefined : body);
      }
      const parts = url.pathname.slice('/data/'.length).split('/');
      const name = decodeURIComponent(parts[0]);
      const file = parts[1];
      const task = parts.length === 2 && ['placement.json', 'robot.glb'].includes(file)
        ? await taskFiles(root, name) : null;
      // Always own /data, including missing files: never fall back to stale dist copies.
      if (!task) { res.statusCode = 404; return res.end('Task data not found'); }
      if (url.searchParams.has('v') && url.searchParams.get('v') !== task.revision) {
        res.statusCode = 409;
        return res.end('Task changed; reload the current revision');
      }
      res.setHeader('Content-Type', file === 'robot.glb' ? 'model/gltf-binary' : 'application/json');
      if (req.method === 'HEAD') return res.end();
      const stream = createReadStream(join(task.directory, file));
      stream.on('error', (error) => {
        if (res.headersSent) res.destroy(error);
        else { res.statusCode = error.code === 'ENOENT' ? 404 : 500; res.end('Task data unavailable'); }
      });
      res.on('close', () => stream.destroy());
      stream.pipe(res);
    } catch (error) {
      res.statusCode = error instanceof URIError ? 400 : 500;
      res.end('Unable to read task data');
    }
  };
}

export function taskDataPlugin() {
  let root;
  return {
    name: 'placement-task-directories',
    configResolved(config) { root = join(config.publicDir, 'data'); },
    configureServer(server) { server.middlewares.use(taskDataMiddleware(root)); },
    configurePreviewServer(server) { server.middlewares.use(taskDataMiddleware(root)); },
  };
}
