import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { demoBundle } from '../../src/demo.js';
import { discoverTasks } from '../../task-data.js';

// Use a published robot asset; synthetic points exist only in test routes.
export async function viewerFixture() {
  const publicRoot = resolve(process.env.PLACEMENT_FIXTURE_PUBLIC || 'public');
  const task = (await discoverTasks(resolve(publicRoot, 'data'))).find((item) => item.id.startsWith('folder:'));
  if (!task) throw new Error('Browser tests require a published placement task');
  const source = JSON.parse(await readFile(resolve(publicRoot, task.bundle), 'utf8'));
  const bundle = demoBundle({ ...source.metadata, reference_snapshot: source.reference_snapshot });
  bundle.robot_model = `/${dirname(task.bundle)}/robot.glb`;
  return { bundle, modelPath: resolve(publicRoot, dirname(task.bundle), 'robot.glb') };
}

export async function installViewerFixture(page) {
  const { bundle, modelPath } = await viewerFixture();
  await page.route((url) => url.pathname === bundle.robot_model,
    (route) => route.fulfill({ path: modelPath, contentType: 'model/gltf-binary' }));
  await page.route('**/api/tasks', (route) => route.fulfill({ json: { tasks: [
    { id: 'fixture', name: '测试数据 · 合成点位', revision: '1', bundle: 'fixture.json' },
  ] } }));
  await page.route('**/fixture.json', (route) => route.fulfill({ json: bundle }));
}
