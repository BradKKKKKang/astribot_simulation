import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/browser',
  workers: 1,
  timeout: 60000,
  use: {
    baseURL: 'http://127.0.0.1:5173',
    headless: true,
    viewport: { width: 1440, height: 1080 },
    launchOptions: {
      executablePath: process.env.PLACEMENT_CHROME || '/usr/bin/google-chrome',
      // CPU software rendering only; does not access the hardware GPU.
      args: ['--disable-gpu', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
    },
  },
});
