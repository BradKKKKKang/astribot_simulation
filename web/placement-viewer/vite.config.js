import { defineConfig } from 'vite';
import { taskDataPlugin } from './task-data.js';

export default defineConfig({
  base: './',
  plugins: [taskDataPlugin()],
  build: {
    copyPublicDir: false, // Runtime data is always served from public/data by the plugin.
    rollupOptions: { output: { manualChunks: { three: ['three',
      'three/addons/controls/OrbitControls.js', 'three/addons/loaders/GLTFLoader.js'] } } },
  },
});
