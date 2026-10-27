import { defineConfig } from 'vite';
import { taskDataPlugin } from './task-data.js';

export default defineConfig({
  base: './',
  plugins: [taskDataPlugin()],
  server: { watch: { ignored: ['**/.placement-catalog.json', '**/.placement-catalog.lock',
    '**/.placement-artifacts/**', '**/public/data/**'] } },
  build: {
    copyPublicDir: false, // The plugin serves registered artifacts and legacy public/data at runtime.
    rollupOptions: { output: { manualChunks: { three: ['three',
      'three/addons/controls/OrbitControls.js', 'three/addons/loaders/GLTFLoader.js'] } } },
  },
});
