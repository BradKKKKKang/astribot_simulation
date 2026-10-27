import { createReadStream } from 'node:fs';
import { lstat, readdir, readFile, realpath } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { join, resolve, dirname, relative } from 'node:path';

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

async function discoverLegacyTasks(root) {
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
    let decoded;
    try { decoded = decodeURIComponent(url.pathname); }
    catch { res.statusCode = 400; return res.end('Invalid URL'); }
    if (decoded.split('/').some((part) => part.startsWith('.placement-'))) {
      res.statusCode = 404; return res.end('Not found');
    }
    if (url.pathname !== '/api/tasks' && !url.pathname.startsWith('/data/')
        && !url.pathname.startsWith('/artifacts/') && !url.pathname.startsWith('/bundles/')) return next();
    res.setHeader('Cache-Control', 'no-store');
    try {
      if (!['GET', 'HEAD'].includes(req.method)) { res.statusCode = 405; return res.end(); }
      if (url.pathname.startsWith('/artifacts/') || url.pathname.startsWith('/bundles/')) {
        return await serveRegistered(root, url, req, res);
      }
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


async function readCatalog(root) {
  try {
    const catalog = JSON.parse(await readFile(join(root, '../../.placement-catalog.json'), 'utf8'));
    if (catalog.version !== 1 || !catalog.tasks || !catalog.stores || !catalog.revisions) throw new Error('Invalid catalog');
    return catalog;
  } catch (error) {
    if (error.code === 'ENOENT') return { tasks: {}, stores: {}, revisions: {} };
    throw error;
  }
}

export async function discoverTasks(root) {
  const [legacy, catalog] = await Promise.all([discoverLegacyTasks(root), readCatalog(root)]);
  const registered = Object.values(catalog.tasks).map((task) => ({
    id: `catalog:${task.name}`, name: task.name, revision: task.revision,
    updated_at: task.updated_at, bundle: `bundles/${task.revision}.json`,
  }));
  const names = new Set(registered.map((task) => task.name));
  return [...registered, ...legacy.filter((task) => !names.has(task.name))]
    .sort((a, b) => b.updated_at - a.updated_at || a.name.localeCompare(b.name));
}

async function artifactBytes(store, ref) {
  if (!ref || !/^[a-f0-9]{64}$/.test(ref.sha256)
      || !new RegExp(`^objects/${ref.sha256.slice(0, 2)}/${ref.sha256}\\.[a-z0-9]+$`).test(ref.path)) {
    throw new Error('Invalid artifact reference');
  }
  const root = resolve(store.root), path = resolve(root, ref.path);
  if (!relative(root, path) || relative(root, path).startsWith('..')) throw new Error('Invalid artifact path');
  // Do not follow links, even links that currently remain inside the allowed store.
  for (let candidate = path; candidate !== dirname(root); candidate = dirname(candidate)) {
    if ((await lstat(candidate)).isSymbolicLink()) throw new Error('Artifact symlink');
  }
  if (await realpath(path) !== path) throw new Error('Artifact path changed');
  const bytes = await readFile(path);
  if (createHash('sha256').update(bytes).digest('hex') !== ref.sha256
      || (ref.size_bytes !== undefined && bytes.length !== ref.size_bytes)) throw new Error('Artifact digest mismatch');
  return bytes;
}

async function serveRegistered(root, url, req, res) {
  const catalog = await readCatalog(root);
  let bytes, contentType;
  const bundleMatch = /^\/bundles\/([a-f0-9]{64})\.json$/.exec(url.pathname);
  const artifactMatch = /^\/artifacts\/([a-f0-9]{24})\/([a-f0-9]{64}\.[a-z0-9]+)$/.exec(url.pathname);
  if (bundleMatch && catalog.revisions[bundleMatch[1]]) {
    const revision = catalog.revisions[bundleMatch[1]], store = catalog.stores[revision.store_id];
    const manifest = JSON.parse(await artifactBytes(store, revision.manifest));
    manifest.artifact_base = `/artifacts/${revision.store_id}/`;
    manifest.robot_model = `${manifest.artifact_base}${manifest.robot_model.path.split('/').at(-1)}`;
    // Server-private local paths never become URLs or client-visible metadata.
    delete manifest.artifact_store;
    bytes = Buffer.from(JSON.stringify(manifest));
    contentType = 'application/json';
  } else if (artifactMatch && catalog.stores[artifactMatch[1]]?.artifacts[artifactMatch[2]]) {
    const store = catalog.stores[artifactMatch[1]];
    bytes = await artifactBytes(store, store.artifacts[artifactMatch[2]]);
    contentType = artifactMatch[2].endsWith('.glb') ? 'model/gltf-binary' : 'application/json';
  } else {
    res.statusCode = 404;
    return res.end('Artifact not registered');
  }
  res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
  res.setHeader('Content-Type', contentType);
  res.setHeader('X-Content-Type-Options', 'nosniff');
  return res.end(req.method === 'HEAD' ? undefined : bytes);
}
