import { test, expect } from '@playwright/test';
import { createHash } from 'node:crypto';
import { viewerFixture } from './fixtures.js';

test('schema5 lazily fetches selected snapshots and keeps geometry on event-only revision updates', async ({ page }) => {
  const { bundle: source, modelPath } = await viewerFixture();
  const artifacts = new Map();
  const put = (value) => {
    const sha256 = createHash('sha256').update(JSON.stringify(value)).digest('hex');
    const path = `objects/${sha256.slice(0, 2)}/${sha256}.json`;
    artifacts.set(`${sha256}.json`, value);
    return { sha256, path };
  };
  const snapshot = structuredClone(source.reference_snapshot);
  snapshot.body_poses_world[1][0] += .01;
  const snapshotRef = put(snapshot), reference = put(source.reference_snapshot);
  const candidates = source.events.filter((event) => !event.initial_phase && event.event_type === 'grasp').slice(0, 2);
  const events = candidates.map((event, index) => ({ ...event, episode_id: 'fixture', event_id: `fixture:${index}`,
    object: index ? '蓝色方块' : '红色方块', identification: { status: 'unknown' },
    analysis_selected: true, analysis_exclusion_reasons: [], snapshot_ref: snapshotRef }));
  const modelUrl = `/artifacts/${'a'.repeat(24)}/${'b'.repeat(64)}.glb`;
  let revision = '1', count = 1, snapshots = 0, models = 0;
  await page.route('**/api/tasks', (route) => route.fulfill({ json: { tasks: [
    { id: 'schema5', name: '共享分析', revision, bundle: 'fixture-v5.json' },
  ] } }));
  await page.route('**/fixture-v5.json', (route) => route.fulfill({ json: {
    schema_version: 5, metadata: { ...source.metadata, demo: false, type: 'gripper' },
    display: { event_types: ['grasp'], default_event_types: ['grasp'], event_labels: { grasp: '自定义抓取' },
      operation_label: '配置动作', candidate_help: '来自分析配置' },
    robot_model: modelUrl, artifact_base: `/artifacts/${'a'.repeat(24)}/`, reference_snapshot: reference,
    episodes: [{ episode_id: 'fixture', events: put({ episode_id: 'fixture', events: events.slice(0, count) }) }],
  } }));
  await page.route('**/artifacts/**', async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === modelUrl) { models += 1; return route.fulfill({ path: modelPath, contentType: 'model/gltf-binary' }); }
    const name = url.pathname.split('/').at(-1);
    if (name.startsWith(snapshotRef.sha256)) snapshots += 1;
    return route.fulfill({ json: artifacts.get(name) });
  });
  await page.goto('/');
  await expect(page.locator('#scene-message')).toBeHidden({ timeout: 45000 });
  await expect(page.locator('#count-total')).toHaveText('1');
  expect(models).toBe(1);
  expect(snapshots).toBe(0);
  await expect(page.locator('#candidate-legend')).toContainText('自定义抓取');
  await expect(page.locator('#operation-type')).toHaveText('配置动作');
  await expect(page.locator('#ignore-seconds')).toBeDisabled();
  await page.locator('.event-row').first().click();
  await expect(page.locator('#detail-summary')).toContainText('已显示事件发生时的机器人姿态');
  expect(snapshots).toBe(1);
  await expect(page.locator('#detail-identification')).toContainText('未识别');
  count = 2; revision = '2';
  await expect(page.locator('#count-total')).toHaveText('2', { timeout: 15000 });
  expect(models).toBe(1);
  await page.locator('.event-row').last().click();
  await expect(page.locator('#detail-summary')).toContainText('已显示事件发生时的机器人姿态');
  expect(snapshots).toBe(1);
  await page.locator('#coordinate-frame').selectOption('world');
  await expect(page.locator('#frame-badge')).toHaveText('WORLD FRAME');
  expect(models).toBe(1);
});
