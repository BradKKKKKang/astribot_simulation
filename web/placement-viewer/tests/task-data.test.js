import test from 'node:test';
import assert from 'node:assert/strict';
import { once } from 'node:events';
import { createServer } from 'node:http';
import { mkdtemp, mkdir, readFile, rename, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { discoverTasks, taskDataMiddleware } from '../task-data.js';

async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), 'placement-directories-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

async function bundle(root, name) {
  const directory = join(root, name);
  await mkdir(directory);
  await writeFile(join(directory, 'placement.json'), JSON.stringify({ metadata: { status: 'complete' }, events: [] }));
  await writeFile(join(directory, 'robot.glb'), 'model');
  return directory;
}

test('discover immediate complete file pairs, ignoring indexes, hidden folders and links', async (t) => {
  const root = await fixture(t);
  assert.deepEqual(await discoverTasks(join(root, 'absent')), []);
  const directory = await bundle(root, 'press_cube_olddemo');
  await bundle(root, '.staging');
  await mkdir(join(root, 'incomplete'));
  await writeFile(join(root, 'incomplete/placement.json'), '{}');
  await symlink(directory, join(root, 'linked'));
  await writeFile(join(root, 'tasks.json'), '{"tasks":[{"name":"stale"}]}');
  const [task] = await discoverTasks(root);
  assert.equal(task.name, 'press_cube_olddemo');
  assert.equal((await discoverTasks(root)).length, 1);
  await rename(directory, join(root, '按压 #1'));
  const [renamed] = await discoverTasks(root);
  assert.equal(renamed.name, '按压 #1');
  assert.notEqual(renamed.id, task.id);
  assert.equal(renamed.bundle, `data/${encodeURIComponent('按压 #1')}/placement.json`);
  await rm(join(root, '按压 #1'), { recursive: true });
  assert.deepEqual(await discoverTasks(root), []);
});

test('runtime HTTP reads current public folders, rejects stale revisions and never serves stale dist', async (t) => {
  const root = await fixture(t);
  const directory = await bundle(root, 'press cube');
  const middleware = taskDataMiddleware(root);
  const server = createServer((req, res) => middleware(req, res, () => { res.statusCode = 418; res.end('fallback'); }));
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(async () => { server.closeAllConnections(); await new Promise((resolve) => server.close(resolve)); });
  const origin = `http://127.0.0.1:${server.address().port}`;
  const response = await fetch(`${origin}/api/tasks`);
  assert.equal(response.headers.get('cache-control'), 'no-store');
  const { tasks: [task] } = await response.json();
  const modelUrl = `${origin}/data/press%20cube/robot.glb?v=${task.revision}`;
  assert.equal(await (await fetch(modelUrl)).text(), 'model');
  assert.equal(await (await fetch(`${origin}/${task.bundle}`)).text(), await readFile(join(directory, 'placement.json'), 'utf8'));
  await writeFile(join(directory, 'robot.glb'), 'updated model');
  assert.equal((await fetch(modelUrl)).status, 409);
  assert.notEqual((await (await fetch(`${origin}/api/tasks`)).json()).tasks[0].revision, task.revision);
  await rename(directory, join(root, 'renamed'));
  assert.equal((await fetch(`${origin}/${task.bundle}`)).status, 404);
  assert.equal((await (await fetch(`${origin}/api/tasks`)).json()).tasks[0].name, 'renamed');
  assert.equal((await fetch(`${origin}/data/%2e%2e%2fsecret/robot.glb`)).status, 404);
  assert.equal((await fetch(`${origin}/data/renamed/robot.glb`, { method: 'POST' })).status, 405);
  assert.equal((await fetch(`${origin}/unrelated`)).status, 418);
  assert.equal((await fetch(`${origin}/.placement-catalog.json`)).status, 404);
  assert.equal((await fetch(`${origin}/%2eplacement-catalog.json`)).status, 404);
  assert.equal((await fetch(`${origin}/@fs/tmp/.placement-catalog.json`)).status, 404);
});

test('Web serves only bundle and model, leaving recognition screenshots and responses local', async (t) => {
  const root = await fixture(t);
  const directory = await bundle(root, 'identified');
  await mkdir(join(directory, 'screenshots'));
  await writeFile(join(directory, 'screenshots/frame.png'), 'image bytes');
  await symlink(join(directory, 'placement.json'), join(directory, 'screenshots/link.png'));
  const middleware = taskDataMiddleware(root);
  const server = createServer((req, res) => middleware(req, res, () => res.end('fallback')));
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(async () => { server.closeAllConnections(); await new Promise((resolve) => server.close(resolve)); });
  const origin = `http://127.0.0.1:${server.address().port}`;
  const [task] = await discoverTasks(root);
  const response = await fetch(`${origin}/data/identified/screenshots/frame.png?v=${task.revision}`);
  assert.equal(response.status, 404);
  assert.equal((await fetch(`${origin}/data/identified/screenshots/link.png`)).status, 404);
  assert.equal((await fetch(`${origin}/data/identified/identification/raw.json`)).status, 404);
  assert.equal((await fetch(`${origin}/data/identified/screenshots/frame.png?v=stale`)).status, 404);
  assert.equal((await fetch(`${origin}/data/identified/screenshots/%2e%2e%2fplacement.json`)).status, 404);
});

test('catalog serves immutable permitted closures, retains old revisions and rejects arbitrary paths', async (t) => {
  const { createHash } = await import('node:crypto');
  const directory = await fixture(t), root = join(directory, 'public/data'), storeRoot = join(directory, 'store');
  await mkdir(root, { recursive: true });
  const storeId = 'a'.repeat(24), artifacts = {};
  const put = async (value, extension = 'json') => {
    const bytes = extension === 'json' ? JSON.stringify(value) : value;
    const sha256 = createHash('sha256').update(bytes).digest('hex');
    const ref = { path: `objects/${sha256.slice(0, 2)}/${sha256}.${extension}`, sha256, size_bytes: Buffer.byteLength(bytes) };
    await mkdir(join(storeRoot, 'objects', sha256.slice(0, 2)), { recursive: true });
    await writeFile(join(storeRoot, ref.path), bytes);
    artifacts[ref.path.split('/').at(-1)] = ref;
    return ref;
  };
  const model = await put('shared model', 'glb');
  const manifest = await put({ schema_version: 5, robot_model: model, episodes: [],
    artifact_store: { path: '/private/local/path' }, metadata: { status: 'complete' } });
  const catalog = { version: 1, tasks: { sample: { name: 'sample', revision: manifest.sha256, updated_at: 1 } },
    stores: { [storeId]: { root: storeRoot, artifacts } },
    revisions: { [manifest.sha256]: { store_id: storeId, manifest } } };
  const save = () => writeFile(join(directory, '.placement-catalog.json'), JSON.stringify(catalog));
  await save();
  const middleware = taskDataMiddleware(root);
  const server = createServer((req, res) => middleware(req, res, () => res.end('fallback')));
  server.listen(0, '127.0.0.1'); await once(server, 'listening');
  t.after(async () => { server.closeAllConnections(); await new Promise((done) => server.close(done)); });
  const origin = `http://127.0.0.1:${server.address().port}`;
  const [task] = await discoverTasks(root);
  const bundleResponse = await fetch(`${origin}/${task.bundle}`), body = await bundleResponse.json();
  assert.equal(body.artifact_store, undefined);
  assert.equal(body.robot_model, `/artifacts/${storeId}/${model.path.split('/').at(-1)}`);
  const response = await fetch(`${origin}${body.robot_model}`);
  assert.equal(await response.text(), 'shared model');
  assert.match(response.headers.get('cache-control'), /immutable/);
  catalog.tasks.sample.revision = 'b'.repeat(64); await save();
  assert.equal((await fetch(`${origin}/${task.bundle}`)).status, 200); // in-flight old revision remains coherent
  assert.equal((await fetch(`${origin}${body.robot_model}`, { method: 'DELETE' })).status, 405);
  assert.equal((await fetch(`${origin}/artifacts/${storeId}/unregistered.json`)).status, 404);
  assert.equal((await fetch(`${origin}/artifacts/${storeId}/%2e%2e%2fsecret`)).status, 404);
  await writeFile(join(storeRoot, model.path), 'tampered');
  assert.equal((await fetch(`${origin}${body.robot_model}`)).status, 500);
  await rm(join(storeRoot, model.path));
  await symlink(join(storeRoot, manifest.path), join(storeRoot, model.path));
  assert.equal((await fetch(`${origin}${body.robot_model}`)).status, 500);
});
