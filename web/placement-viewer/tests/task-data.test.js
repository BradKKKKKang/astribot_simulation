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
});
