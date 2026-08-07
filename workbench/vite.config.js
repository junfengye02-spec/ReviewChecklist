import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  return {
    plugins: [vue()],
    server: {
      proxy: {
        '/api/v1': {
          target: env.TENDER_REVIEW_BACKEND_URL || 'http://127.0.0.1:8000',
          changeOrigin: true,
        },
      },
    },
  }
})
