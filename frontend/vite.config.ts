import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/workspace": "http://127.0.0.1:8000",
      "/wards": "http://127.0.0.1:8000",
    },
  },
});
