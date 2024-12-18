/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL: string
  readonly VITE_NEWS_API_KEY: string
  readonly VITE_ALPHA_VANTAGE_API_KEY: string
  readonly VITE_FINNHUB_API_KEY: string
  // Add more env variables types here if needed
}

interface ImportMeta {
  readonly env: ImportMetaEnv
} 