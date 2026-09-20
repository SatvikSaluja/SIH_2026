import path from 'path';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

import runtimeErrorOverlay from '@replit/vite-plugin-runtime-error-modal';

const rawPort = process.env.PORT || '5173';
const port = Number(rawPort);
const basePath = process.env.BASE_PATH || '/';

export default defineConfig({
  base: basePath,
  plugins: [
    react(),
    tailwindcss(),
    runtimeErrorOverlay(),
    ...(process.env.NODE_ENV !== 'production' &&
    process.env.REPL_ID !== undefined
      ? [
          await import('@replit/vite-plugin-cartographer').then((m) =>
            m.cartographer({
              root: path.resolve(import.meta.dirname, '..'),
            }),
          ),
          await import('@replit/vite-plugin-dev-banner').then((m) =>
            m.devBanner(),
          ),
        ]
      : []),
  ],
  resolve: {
    alias: {
      '@': path.resolve(import.meta.dirname, 'src'),
      '@assets': path.resolve(
        import.meta.dirname,
        '..',
        '..',
        'attached_assets',
      ),
    },
    dedupe: ['react', 'react-dom'],
  },
  root: path.resolve(import.meta.dirname),
  build: {
    outDir: path.resolve(import.meta.dirname, 'dist/public'),
    emptyOutDir: true,
  },
  server: {
    port,
    strictPort: false,
    host: '0.0.0.0',
    allowedHosts: true,
    fs: {
      strict: true,
    },
    // The frontend calls relative /api/* paths (setBaseUrl() is never
    // called, by design). These two rules mirror the production Netlify
    // redirects exactly, so a route that works here works deployed --
    // which was NOT true before: the translation routes used to live only
    // in the Express server, which production never routed through, so
    // /dashboard, /regions, /parcels, /changes, /processing/runs and
    // /exports all 404'd once deployed while passing locally.
    //
    // Order matters: Vite matches these in declaration order, so the
    // narrower /api/sentinel rule has to come first.
    proxy: {
      // Geo-VLM Sentinel is the one surface still served by the Express
      // app (its own store plus a Gemini client). Nothing in the Python
      // backend answers /sentinel/*, so it cannot go direct yet.
      '/api/sentinel': {
        target: process.env.SENTINEL_PROXY_TARGET || 'http://127.0.0.1:5001',
        changeOrigin: true,
      },
      // Everything else is the Python backend, path-rewritten the same
      // way the Netlify :splat redirect rewrites it.
      '/api': {
        target: process.env.API_PROXY_TARGET || 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (requestPath) => requestPath.replace(/^\/api/, ''),
      },
    },
  },
  preview: {
    port,
    host: '0.0.0.0',
    allowedHosts: true,
  },
});
