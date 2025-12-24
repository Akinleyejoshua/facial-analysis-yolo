// vite.config.ts

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  
  // --- FIX 1: Ensure WASM is recognized as a dependency/asset ---
  assetsInclude: ['**/*.wasm'], 

  // --- FIX 2: Ensure WASM dependencies are correctly handled ---
  optimizeDeps: {
    // This tells Vite how to handle the WASM files it finds
    exclude: ['onnxruntime-web'], 
  },
  
  // --- Optional: If the problem is persistent, use a strict build target ---
  // build: {
  //   target: 'es2020',
  // },

  server: {
    // --- FIX 3: Explicitly set MIME type for WASM files in the dev server ---
    // This often forces the correct header to be sent.
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  }
});