import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { installViewerFixture, viewerFixture } from './fixtures.js';

test.beforeEach(async ({ page }) => {
  await installViewerFixture(page);
});

test('published tasks appear, switch and refresh without rebuilding or reloading', async ({ page }) => {
  test.setTimeout(180000);
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  const { bundle: source } = await viewerFixture();
  const bundle = (count) => ({ ...source,
    metadata: { ...source.metadata, demo: false },
    events: source.events.filter((event) => !event.initial_phase && event.event_type === 'grasp').slice(0, count) });
  const first = { id: 'first', name: '抓取任务', revision: '1', bundle: 'runtime/first.json' };
  const second = { id: 'second', name: '放置任务', revision: '1', bundle: 'runtime/second.json' };
  const broken = { id: 'broken', name: '暂不可用任务', revision: '1', bundle: 'runtime/broken.json' };
  let catalog = [];
  let secondCount = 3;
  await page.route('**/api/tasks', (route) => route.fulfill({ json: { tasks: catalog } }));
  await page.route('**/runtime/first.json', (route) => route.fulfill({ json: bundle(2) }));
  await page.route('**/runtime/second.json', (route) => route.fulfill({ json: bundle(secondCount) }));
  await page.route('**/runtime/broken.json', (route) => route.fulfill({ status: 503, body: 'unavailable' }));
  await page.goto('/');
  await expect(page.locator('#scene-message')).toContainText('暂无已发布任务');
  await page.locator('#display-mode').selectOption('points');
  catalog = [first, second];
  await expect(page.locator('#task-select')).toHaveValue('first', { timeout: 45000 });
  await expect(page.locator('#count-total')).toHaveText('2');
  await page.locator('#task-select').selectOption('second');
  await expect(page.locator('#count-total')).toHaveText('3', { timeout: 45000 });
  secondCount = 4;
  catalog = [broken, first, { ...second, revision: '2' }];
  await expect(page.locator('#count-total')).toHaveText('4', { timeout: 45000 });
  await expect(page.locator('#task-select')).toHaveValue('second');
  await page.locator('#task-select').selectOption('broken');
  await expect(page.locator('#notice')).toContainText('HTTP 503');
  await expect(page.locator('#task-select')).toHaveValue('second');
  await expect(page.locator('#count-total')).toHaveText('4');
  await page.screenshot({ path: 'test-results/task-selector.png' });
  catalog = [{ ...first, id: 'renamed', name: 'renamed_folder' }];
  await expect(page.locator('#task-select option:checked')).toHaveText('renamed_folder', { timeout: 45000 });
  await expect(page.locator('#count-total')).toHaveText('2');
  catalog = [];
  await expect(page.locator('#scene-message')).toContainText('暂无已发布任务', { timeout: 15000 });
  await expect(page.locator('#count-total')).toHaveText('0');
  await expect(page.locator('.event-row')).toHaveCount(0);
  await expect(page.locator('#operation-type')).toHaveText('—');
  await expect(page.locator('#candidate-legend')).toBeEmpty();
  expect(errors).toEqual([]);
});

test('press task has a fixed type and shows only stroke candidates', async ({ page }) => {
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  const { bundle } = await viewerFixture();
  bundle.schema_version = 3;
  bundle.metadata.detector = { ...bundle.metadata.detector, type: 'press',
    press_min_depth_m: .02, press_min_retract_m: .015, press_max_lateral_m: .025 };
  bundle.events = bundle.events.filter((event) => event.event_type === 'dwell' && !event.initial_phase)
    .slice(0, 3).map((event) => ({ ...event, event_type: 'press', related_event_ids: [],
      press_depth_m: .021, press_retract_m: .016, press_lateral_m: .008 }));
  bundle.events.push({ ...bundle.events[0], event_id: 'unrelated-grasp', event_type: 'grasp', detection_sources: ['gripper'] });
  await page.route((url) => url.pathname === '/press-test.json', (route) => route.fulfill({ json: bundle }));
  await page.route('**/api/tasks', (route) => route.fulfill({ json: { tasks: [
    { id: 'press', name: '按压测试', revision: '1', bundle: 'press-test.json' },
    { id: 'gripper', name: '夹爪测试', revision: '1', bundle: 'fixture.json' },
  ] } }));
  await page.goto('/');
  await expect(page.locator('#scene-message')).toBeHidden({ timeout: 45000 });
  await expect(page.locator('#candidate-mode')).toHaveCount(0);
  await expect(page.locator('#operation-type')).toHaveText('按压');
  await expect(page.locator('#count-total')).toHaveText('3');
  await expect(page.locator('#candidate-legend')).toHaveText('按压候选点');
  await expect(page.locator('#candidate-help')).toContainText('下压 ≥ 20 mm');
  await page.locator('#display-mode').selectOption('points');
  await page.locator('.event-row').first().click();
  await expect(page.locator('#detail-evidence')).toContainText('下压 21.0 mm');
  await expect(page.locator('#detail-evidence')).toContainText('横向偏移 8.0 mm');
  await expect(page.locator('#candidate-legend input')).toHaveCount(0);
  await page.locator('#task-select').selectOption('gripper');
  await expect(page.locator('#operation-type')).toHaveText('夹爪开合');
  await expect(page.locator('#count-total')).toHaveText('96');
  await page.locator('.event-row').first().click();
  await expect(page.locator('#detail-evidence')).toHaveText('依据：夹爪开合');
  await page.locator('#task-select').selectOption('press');
  await expect(page.locator('#operation-type')).toHaveText('按压');
  await expect(page.locator('#count-total')).toHaveText('3');
  expect(errors).toEqual([]);
});

test('metric robot model, filters, event selection, views and mobile layout', async ({ page }) => {
  test.setTimeout(180000); // Full USD geometry and heat-field interactions use CPU SwiftShader.
  const errors = [];
  const requests = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
  page.on('request', (request) => requests.push({ method: request.method(), url: request.url() }));
  await page.goto('/');
  await expect(page).toHaveTitle('Simulation · Astribot');
  await expect(page.locator('.brand')).toContainText('SIMULATION');
  await expect(page.locator('#scene-message')).toBeHidden({ timeout: 45000 });
  await expect(page.locator('#task-select')).toHaveValue('fixture');
  await expect(page.locator('#task-select option:checked')).toContainText('测试数据');
  await expect(page.locator('#count-total')).toHaveText('96');
  await expect(page.locator('#display-mode option')).toHaveCount(3);
  await expect(page.getByLabel('排除初始位姿附近候选点', { exact: true })).toBeVisible();
  await expect(page.locator('#task-reference')).toContainText('待补充图片');
  await expect(page.locator('#episode-range-label')).toHaveText('1–24 / 24');
  // Both native handles occupy one physical track.
  const bounds = await page.locator('.dual-range input').evaluateAll((inputs) => inputs.map((input) => {
    const r = input.getBoundingClientRect(); return [r.x, r.y, r.width, r.height];
  }));
  expect(bounds[0]).toEqual(bounds[1]);
  // Real pointer drag of the lower handle along that shared track.
  const track = await page.locator('#episode-start-range').boundingBox();
  await page.mouse.move(track.x + 7, track.y + track.height / 2);
  await page.mouse.down();
  await page.mouse.move(track.x + track.width * .25, track.y + track.height / 2, { steps: 4 });
  await page.mouse.up();
  expect(Number(await page.locator('#episode-start').inputValue())).toBeGreaterThan(1);
  await page.locator('#episode-start').fill('2');
  await page.locator('#episode-start').press('Tab');
  await page.locator('#episode-end').fill('4');
  await page.locator('#episode-end').press('Tab');
  await expect(page.locator('#count-total')).toHaveText('12');
  await expect(page.locator('#episode-start-range')).toHaveValue('2');
  await expect(page.locator('#episode-end-range')).toHaveValue('4');
  await page.locator('#episode-start').fill('1');
  await page.locator('#episode-start').press('Tab');
  await page.locator('#episode-end').fill('24');
  await page.locator('#episode-end').press('Tab');
  await page.getByRole('button', { name: '左手', exact: true }).click();
  await expect(page.locator('#count-total')).toHaveText('48');
  await expect(page.locator('#candidate-mode')).toHaveCount(0);
  await expect(page.locator('#operation-type')).toHaveText('夹爪开合');
  await expect(page.locator('#candidate-legend input')).toHaveCount(2);
  await expect(page.locator('#candidate-legend')).toContainText('抓取点');
  await expect(page.locator('#candidate-legend')).toContainText('释放点');
  await expect(page.locator('#candidate-legend')).not.toContainText('停留点');
  await expect(page.locator('#viewport')).toHaveAttribute('data-density-kind', 'field');
  await expect(page.locator('#viewport')).toHaveAttribute('data-density-meshes', '1');
  await page.locator('#bandwidth').fill('8');
  await expect(page.locator('#bandwidth-label')).toHaveText('8 cm');
  await page.screenshot({ path: 'test-results/placement-field.png' });
  await page.locator('.event-row').first().click();
  await expect(page.locator('#detail-title')).toHaveText('抓取 · 左手');
  await expect(page.locator('#detail-summary')).toContainText('合成示例点');
  await page.locator('#coordinate-frame').selectOption('world');
  await expect(page.locator('#frame-badge')).toHaveText('WORLD FRAME');
  await page.getByRole('button', { name: '俯视', exact: true }).click();
  await page.locator('#display-mode').selectOption('field');
  await page.locator('#z-max').fill('0');
  await expect(page.locator('#count-total')).toHaveText('0');
  await expect(page.locator('#density-max')).toHaveText('0%');
  await expect(page.locator('#viewport')).toHaveAttribute('data-density-meshes', '0');
  await expect(page.locator('#event-list')).toContainText('没有操作候选点');
  await page.locator('#z-max').fill('3');
  await page.locator('#display-mode').selectOption('field');
  await expect(page.locator('#density-title')).toHaveText('相对点位密度');
  await page.getByRole('button', { name: '全部', exact: true }).click();
  await expect(page.locator('#count-total')).toHaveText('96');
  await page.locator('#show-release').uncheck();
  await expect(page.locator('#count-total')).toHaveText('48');
  await page.locator('#show-release').check();
  await page.locator('#include-excluded').check();
  await expect(page.locator('.event-row.excluded').first()).toBeVisible();
  await expect(page.locator('#count-total')).toHaveText('96'); // audit never contributes to density
  await expect(page.locator('#list-count')).toHaveText('144');
  await page.locator('.event-row.excluded').first().click();
  await expect(page.locator('#detail-evidence')).toContainText('初始位姿测试区');
  await page.locator('#include-excluded').uncheck();
  await page.locator('#ignore-seconds').fill('3');
  await expect(page.locator('#count-total')).toHaveText('48');
  await page.locator('#ignore-seconds').fill('1');
  await page.locator('#display-mode').selectOption('field-points');
  await page.getByRole('button', { name: '透视', exact: true }).click();
  await page.screenshot({ path: 'test-results/placement-desktop.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: 'test-results/placement-mobile.png', fullPage: true });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  expect(errors).toEqual([]);
  expect(requests.every((request) => request.method === 'GET')).toBe(true);
  expect(requests.every((request) => new URL(request.url).hostname === '127.0.0.1')).toBe(true);
});

test('local bundle import, robot snapshot, filtered export and malformed files', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#scene-message')).toBeHidden({ timeout: 45000 });
  const { bundle, modelPath } = await viewerFixture();
  bundle.metadata.demo = false;
  bundle.events = bundle.events.filter((e) => !e.initial_phase && e.event_type !== 'dwell').slice(0, 2).map((event) => ({ ...event,
    episode_id: '<script>not executable</script>', snapshot: bundle.reference_snapshot }));
  await page.locator('#bundle-file').setInputFiles({ name: 'placement.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(bundle)) });
  await page.locator('#model-file').setInputFiles(modelPath);
  await expect(page.locator('#task-select option:checked')).toHaveText('placement.json', { timeout: 45000 });
  await expect(page.locator('#count-total')).toHaveText('2');
  await page.locator('.event-row').first().click();
  await expect(page.locator('#detail-summary')).toContainText('已显示事件发生时的机器人姿态');
  await expect(page.locator('#event-list script')).toHaveCount(0);
  const downloadPromise = page.waitForEvent('download');
  await page.locator('#export-events').click();
  const download = await downloadPromise;
  const exported = JSON.parse(await readFile(await download.path(), 'utf8'));
  expect(exported.events).toHaveLength(2);
  expect(exported.metadata.view_filters.frame).toBe('base');
  expect(exported.metadata.view_filters.type).toBe('gripper');
  expect(exported.metadata.view_filters.excludeInitial).toBe(true);
  expect(exported.metadata.view_filters.episodeIds).toEqual(['<script>not executable</script>']);
  expect(exported.metadata.view_filters.eventTypes).toEqual(['grasp', 'release']);
  await expect(page.locator('#episode-start-range')).toBeDisabled();
  // Empty imports have a disabled interval and no stale heat field.
  const empty = { ...bundle, events: [], metadata: { ...bundle.metadata, episodes: [] } };
  await page.locator('#bundle-file').setInputFiles({ name: 'empty.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(empty)) });
  await page.locator('#model-file').setInputFiles(modelPath);
  await expect(page.locator('#count-total')).toHaveText('0');
  await expect(page.locator('#episode-range-label')).toHaveText('无 episode');
  await expect(page.locator('#episode-start')).toBeDisabled();
  await page.locator('#bundle-file').setInputFiles({ name: 'invalid.json', mimeType: 'application/json', buffer: Buffer.from('{"schema_version":99}') });
  await expect(page.locator('#notice')).toContainText('不支持的数据格式');
  await expect(page.locator('#count-total')).toHaveText('0');
});

test('objects color candidate icons and toggle points, counts and density without image requests', async ({ page }) => {
  const { bundle, modelPath } = await viewerFixture();
  bundle.schema_version = 4;
  bundle.events = bundle.events.filter((event) => !event.initial_phase && event.event_type === 'grasp').slice(0, 4);
  const objects = ['红色方块', '蓝色方块', '黄色方块', null];
  bundle.events.forEach((event, index) => {
    event.object = objects[index];
    event.identification = { status: index === 3 ? 'failed' : 'success', screenshot: 'screenshots/frame.png' };
  });
  bundle.robot_model = 'robot.glb';
  const requests = [];
  const errors = [];
  page.on('request', (request) => requests.push(request.url()));
  page.on('pageerror', (error) => errors.push(error.message));
  await page.route('**/api/tasks', (route) => route.fulfill({ json: { tasks: [
    { id: 'identified', name: '已识别任务', revision: '4', bundle: 'data/identified/placement.json' },
  ] } }));
  await page.route('**/data/identified/placement.json', (route) => route.fulfill({ json: bundle }));
  await page.route('**/data/identified/robot.glb*', (route) => route.fulfill({ path: modelPath, contentType: 'model/gltf-binary' }));
  await page.goto('/');
  await expect(page.locator('#scene-message')).toBeHidden({ timeout: 45000 });
  await expect(page.locator('#object-legend input')).toHaveCount(4);
  const red = page.locator('#object-legend').getByRole('checkbox', { name: '红色方块', exact: true });
  await expect(red).toBeChecked();
  await expect(page.locator('.event-row').first().locator('.dot')).toHaveCSS('background-color', 'rgb(236, 107, 101)');
  await page.locator('.event-row').first().click();
  await expect(page.locator('#detail-object')).toHaveText('交互物体：红色方块');
  await expect(page.locator('#detail-screenshot')).toHaveCount(0);
  await red.uncheck();
  await expect(page.locator('#count-total')).toHaveText('3');
  await expect(page.locator('#viewport')).toHaveAttribute('data-marker-count', '3');
  await expect(page.locator('#viewport')).toHaveAttribute('data-density-samples', '3');
  await expect(page.locator('#event-list')).not.toContainText('红色方块');
  await expect(page.locator('#detail-title')).toHaveText('选择一个操作候选点');
  const blueColor = await page.locator('.event-row').first().locator('.dot').evaluate((node) => node.style.backgroundColor);
  await page.locator('#display-mode').selectOption('points');
  await expect(red).not.toBeChecked();
  await expect(page.locator('.event-row').first().locator('.dot')).toHaveCSS('background-color', blueColor);
  for (const name of ['蓝色方块', '黄色方块', '未识别']) {
    await page.locator('#object-legend').getByRole('checkbox', { name, exact: true }).uncheck();
  }
  await expect(page.locator('#count-total')).toHaveText('0');
  await expect(page.locator('#viewport')).toHaveAttribute('data-density-meshes', '0');
  await expect(page.locator('#viewport')).toHaveAttribute('data-marker-count', '0');
  await page.locator('#object-legend').getByRole('checkbox', { name: '未识别', exact: true }).check();
  await expect(page.locator('#count-total')).toHaveText('1');
  await expect(page.locator('.event-row').first().locator('.dot')).toHaveCSS('background-color', 'rgb(137, 149, 163)');
  await red.check();
  await expect(page.locator('#count-total')).toHaveText('2');
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  expect(requests.some((url) => url.includes('/screenshots/'))).toBe(false);
  expect(errors).toEqual([]);
});
