import { test, expect } from '@playwright/test';
import { installViewerFixture } from './fixtures.js';

test('demo heat field updates with live episode filtering and clears empty results', async ({ page }) => {
  test.setTimeout(120000); // The demo always loads the full Astribot S1 GLB.
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
  await installViewerFixture(page);
  await page.goto('/');
  await expect(page.locator('#scene-message')).toBeHidden();
  await page.evaluate(async () => {
    const { PlacementScene } = await import('/src/scene.js');
    const THREE = await import('/node_modules/three/build/three.module.js');
    const renderEvents = PlacementScene.prototype.renderEvents;
    let scene;
    PlacementScene.prototype.renderEvents = function (...args) {
      renderEvents.apply(this, args);
      scene = this;
      scene.renderer.setAnimationLoop(null);
    };
    window.readField = (positions = []) => {
      const volume = scene.density.children[0];
      if (!volume) return null;
      const { data, width, height } = volume.material.uniforms.densityMap.value.image;
      const index = data.indexOf(1);
      const grid = [index % width, Math.floor(index / width) % height, Math.floor(index / (width * height))];
      const dimensions = volume.material.uniforms.dimensions.value.toArray();
      const peak = grid.map((v, i) => volume.position.getComponent(i)
        + (v / (dimensions[i] - 1) - .5) * volume.scale.getComponent(i));
      scene.robot.visible = scene.markers.visible = scene.grid.visible = scene.selected.visible = false;
      scene.scene.children.filter((child) => child.type === 'AxesHelper').forEach((child) => { child.visible = false; });
      scene.camera.up.set(0, 1, 0);
      scene.camera.position.set(.65, 0, 3);
      scene.camera.lookAt(.65, 0, .78);
      scene.camera.updateMatrixWorld(true);
      const target = new THREE.WebGLRenderTarget(320, 320);
      scene.renderer.setRenderTarget(target);
      scene.renderer.render(scene.scene, scene.camera);
      const colors = [peak, ...positions].map((position) => {
        const p = new THREE.Vector3(...position).project(scene.camera);
        const color = new Uint8Array(4);
        scene.renderer.readRenderTargetPixels(target, Math.round((p.x + 1) * 160), Math.round((p.y + 1) * 160), 1, 1, color);
        return [...color];
      });
      scene.renderer.setRenderTarget(null);
      target.dispose();
      return { peak, colors };
    };
  });
  await page.locator('#candidate-mode').selectOption('motion');
  await page.locator('#display-mode').selectOption('field');
  await expect(page.locator('#viewport')).toHaveAttribute('data-density-samples', '72');
  const all = await page.evaluate(() => window.readField());
  // Use the real range input event, without committing a number-field change.
  await page.locator('#episode-start-range').fill('23');
  await expect(page.locator('#episode-range-label')).toHaveText('23–24 / 24');
  await expect(page.locator('#viewport')).toHaveAttribute('data-density-samples', '6');
  const pair = await page.evaluate(() => window.readField());
  await page.locator('#episode-start-range').fill('24');
  await expect(page.locator('#viewport')).toHaveAttribute('data-density-samples', '2');
  const last = await page.evaluate((oldPeak) => window.readField([oldPeak]), all.peak);
  const distance = (a, b) => Math.hypot(...a.map((v, i) => v - b[i]));
  expect(distance(all.peak, pair.peak)).toBeGreaterThan(.025);
  expect(distance(pair.peak, last.peak)).toBeGreaterThan(.025);
  for (const { colors } of [all, pair, last]) expect(colors[0][0]).toBeGreaterThan(colors[0][1] * 1.4);
  expect(last.colors[0][0]).toBeGreaterThan(last.colors[1][0] * 2); // old hotspot cools
  await page.locator('#episode-start-range').fill('1');
  const restored = await page.evaluate(() => window.readField());
  expect(restored).toEqual(all);
  await page.locator('#z-max').fill('0');
  await expect(page.locator('#viewport')).toHaveAttribute('data-density-samples', '0');
  await expect(page.locator('#viewport')).toHaveAttribute('data-density-meshes', '0');
  expect(await page.evaluate(() => window.readField())).toBeNull();
  expect(errors).toEqual([]);
});

test('rendered heat colors follow local density rather than accumulated sparse depth', async ({ page }) => {
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
  await page.route('**/field-render-check', (route) => route.fulfill({
    contentType: 'text/html', body: '<div id="field" style="width:320px;height:320px"></div>',
  }));
  await page.goto('/field-render-check');
  const pixels = await page.evaluate(async () => {
    const { PlacementScene } = await import('/src/scene.js');
    const THREE = await import('/node_modules/three/build/three.module.js');
    const scene = new PlacementScene(document.querySelector('#field'), () => {});
    scene.renderer.setAnimationLoop(null);
    scene.grid.visible = false;
    scene.scene.children.filter((child) => child.type === 'AxesHelper').forEach((child) => { child.visible = false; });
    scene.camera.position.set(0, 0, 3);
    scene.controls.target.set(0, 0, 0);
    scene.controls.update();
    const event = (id, x, z) => ({ event_id: id, event_type: 'grasp', tcp_pose_base: [x, 0, z] });
    const events = [
      ...Array.from({ length: 12 }, (_, i) => event(`dense${i}`, -.3, 0)),
      ...[-.4, -.2, 0, .2, .4].flatMap((z, j) => Array.from({ length: 4 }, (_, i) => event(`sparse${j}:${i}`, .3, z))),
    ];
    scene.renderEvents(events, 'base', 'field', events, new Set(), .05);
    const target = new THREE.WebGLRenderTarget(320, 320);
    const sample = (x, z) => {
      const point = new THREE.Vector3(x, 0, z).project(scene.camera);
      const color = new Uint8Array(4);
      scene.renderer.readRenderTargetPixels(target, Math.round((point.x + 1) * 160), Math.round((point.y + 1) * 160), 1, 1, color);
      return [...color];
    };
    const result = [];
    for (const cameraZ of [3, -3]) {
      scene.camera.position.z = cameraZ;
      scene.camera.lookAt(0, 0, 0);
      scene.camera.updateMatrixWorld(true);
      scene.renderer.setRenderTarget(target);
      scene.renderer.render(scene.scene, scene.camera);
      result.push({ dense: sample(-.3, 0), sparse: sample(.3, 0) });
    }
    scene.renderer.setRenderTarget(null);
    target.dispose();
    scene.clear(scene.density);
    scene.renderer.dispose();
    scene.resizeObserver.disconnect();
    return result;
  });
  for (const { dense, sparse } of pixels) {
    expect(dense[0]).toBeGreaterThan(dense[1] * 1.4); // hot red
    expect(sparse[1]).toBeGreaterThan(sparse[0] * 1.4); // cool teal
    expect(dense[0]).toBeGreaterThan(sparse[0] * 2);
  }
  expect(errors).toEqual([]);
});
