import { defineConfig } from 'vite';
export default defineConfig({ resolve: { dedupe: ['three'] }, server: { proxy: Object.fromEntries(['/api', '/media', '/preview', '/lease', '/artifacts'].map(path => [path, 'http://10.11.12.73:8780'])) } });
