import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  build: {
    outDir: 'dist',
    target: 'es2022',
  },
  server: {
    port: 3000,
    open: true,
  },
  optimizeDeps: {
    exclude: ['alice_physics'],
  },
});
