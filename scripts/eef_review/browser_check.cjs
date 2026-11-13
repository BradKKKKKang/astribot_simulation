/* Run from this repository, or with EEF_PLAYWRIGHT/EEF_CHROME on another LAN host. */
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.EEF_PLAYWRIGHT || '../../web/eef-viewer/node_modules/playwright-core');
const base = process.env.EEF_URL || 'http://10.11.12.73:8780';
const output = process.env.EEF_QA_OUTPUT || path.resolve('outputs/eef_review/browser');
fs.mkdirSync(output, { recursive: true });

(async () => {
  const browser = await chromium.launch({ executablePath: process.env.EEF_CHROME || '/usr/bin/google-chrome', headless: true,
    args: ['--no-sandbox', '--no-proxy-server', '--disable-dev-shm-usage', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  const errors = [], requests = [], report = { base, host: require('node:os').hostname(), seek: [] };
  try {
    const contexts = await Promise.all([browser.newContext({ viewport: { width: 1550, height: 1050 } }), browser.newContext({ viewport: { width: 1440, height: 1000 } })]);
    const pages = await Promise.all(contexts.map(context => context.newPage()));
    for (const page of pages) {
      page.on('pageerror', error => { errors.push(error.message); console.error('PAGEERROR', error.message); });
      page.on('request', request => requests.push(request.url()));
      await page.goto(base, { waitUntil: 'domcontentloaded', timeout: 120000 });
      await page.waitForFunction(() => document.querySelector('#seek').max !== '0', {}, { timeout: 120000 });
      await page.waitForFunction(() => document.querySelector('#video').readyState >= 2, {}, { timeout: 60000 });
      if (page === pages[0]) report.coldFirstFrameMs = await page.evaluate(() => performance.now());
    }
    const page = pages[0];
    await page.locator('#compute').waitFor({ state: 'visible' });
    // Wait for a complete result rather than counting a held initial pose as replay.
    await page.waitForFunction(() => /缓存已就绪|计算已完成/.test(document.querySelector('#loading').textContent), {}, { timeout: 240000 });
    const frames = await page.locator('#seek').getAttribute('max');
    for (const fraction of [.8, .2, .55, 0, .95]) {
      const index = Math.round(Number(frames) * fraction), started = Date.now();
      await page.locator('#seek').evaluate((input, index) => { input.value = index; input.dispatchEvent(new Event('input', { bubbles: true })); }, index);
      await page.waitForFunction(index => {
        const text = document.querySelector('#time').textContent;
        return text.includes(`/ ${index + 1} ·`) && !document.querySelector('#sync').textContent.includes('等待');
      }, index, { timeout: 30000 });
      report.seek.push({ index, elapsed_ms: Date.now() - started, sync: await page.locator('#sync').textContent() });
    }
    const otherBefore = await pages[1].locator('#time').textContent();
    const computeBefore = requests.filter(url => url.endsWith('/compute')).length;
    await page.locator('#speed').selectOption('2'); await page.locator('#play').click();
    await page.waitForTimeout(1300); await page.locator('#play').click();
    report.independentPlayback = (await pages[1].locator('#time').textContent()) === otherBefore;
    report.playbackDidNotRecompute = requests.filter(url => url.endsWith('/compute')).length === computeBefore;
    await page.screenshot({ path: path.join(output, 'review.png'), fullPage: true });
    const buttons = pages[1].locator('#catalog button:not([disabled])');
    if (await buttons.count() > 1) {
      await buttons.nth(1).click();
      await pages[1].waitForFunction(() => document.querySelector('#video').readyState >= 2, {}, { timeout: 60000 });
      report.differentEpisode = (await pages[1].locator('#subtitle').textContent()) !== (await page.locator('#subtitle').textContent());
    }
    report.errors = errors;
    report.remoteUrlsOnly = requests.filter(url => url.startsWith('http')).every(url => new URL(url).origin === base);
    report.visibleError = await page.locator('#error').textContent();
    report.requests = requests.length;
    fs.writeFileSync(path.join(output, 'report.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
    if (errors.length || report.visibleError || !report.independentPlayback || !report.playbackDidNotRecompute || !report.remoteUrlsOnly || !report.differentEpisode) process.exitCode = 1;
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
