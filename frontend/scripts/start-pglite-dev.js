#!/usr/bin/env node

/**
 * Start development mode with PGLite enabled
 * This script sets up the environment and starts both React dev server and Electron
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const log = (message) => {
  console.log(`[PGLite Dev] ${message}`);
};

const error = (message) => {
  console.error(`[PGLite Dev] ❌ ${message}`);
};

const success = (message) => {
  console.log(`[PGLite Dev] ✅ ${message}`);
};

function main() {
  log('Starting Mirix in PGLite development mode...');
  
  // Set environment variables for PGLite mode
  const env = {
    ...process.env,
    MIRIX_USE_PGLITE: 'true',
    MIRIX_PGLITE_BRIDGE_URL: 'http://127.0.0.1:8001',
    MIRIX_PG_URI: '',
    MIRIX_DEBUG: 'true',
    MIRIX_LOG_LEVEL: 'DEBUG',
    DEBUG: 'true',
    // Add Node.js flags to help with WebAssembly (without experimental-wasm-threads)
    NODE_OPTIONS: '--max-old-space-size=4096'
  };
  
  log('Environment configured for PGLite mode');
  log('- MIRIX_USE_PGLITE: true');
  log('- MIRIX_PGLITE_BRIDGE_URL: http://127.0.0.1:8001');
  log('- MIRIX_PG_URI: (empty - SQLite fallback disabled)');
  
  // Start electron-dev with PGLite environment
  log('Starting Electron with React dev server...');
  
  const electronDev = spawn('npm', ['run', 'electron-dev'], {
    cwd: path.join(__dirname, '..'),
    env: env,
    stdio: 'inherit'
  });
  
  electronDev.on('close', (code) => {
    if (code === 0) {
      success('Application closed successfully');
    } else {
      error(`Application exited with code ${code}`);
    }
    process.exit(code);
  });
  
  electronDev.on('error', (err) => {
    error(`Failed to start application: ${err.message}`);
    process.exit(1);
  });
  
  // Handle Ctrl+C gracefully
  process.on('SIGINT', () => {
    log('Received SIGINT, shutting down...');
    electronDev.kill('SIGINT');
  });
  
  process.on('SIGTERM', () => {
    log('Received SIGTERM, shutting down...');
    electronDev.kill('SIGTERM');
  });
}

if (require.main === module) {
  main();
}

module.exports = { main }; 