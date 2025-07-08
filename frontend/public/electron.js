const { app, BrowserWindow, Menu, shell, ipcMain, dialog, systemPreferences } = require('electron');
const path = require('path');
const fs = require('fs');
const os = require('os');
const isDev = require('electron-is-dev');
const { spawn } = require('child_process');
const screenshot = require('screenshot-desktop');
const http = require('http');

// Add WebAssembly support flags for PGLite
app.commandLine.appendSwitch('enable-features', 'WebAssemblyThreads');
app.commandLine.appendSwitch('enable-experimental-web-platform-features');
app.commandLine.appendSwitch('js-flags', '--max-old-space-size=4096');

// Import PGLite bridge server
const { startPGLiteBridge, stopPGLiteBridge, closePGLite } = require('../src/pglite-bridge.js');

// Override isDev for packaged apps
const isPackaged = app.isPackaged || 
                  (process.mainModule && process.mainModule.filename && process.mainModule.filename.indexOf('app.asar') !== -1) ||
                  process.execPath.indexOf('MIRIX.app') !== -1 ||
                  __dirname.indexOf('app.asar') !== -1;
const actuallyDev = isDev && !isPackaged;

const safeLog = {
  log: (...args) => {
    if (actuallyDev) {
      console.log(...args);
    }
  },
  error: (...args) => {
    if (actuallyDev) {
      console.error(...args);
    }
  },
  warn: (...args) => {
    if (actuallyDev) {
      console.warn(...args);
    }
  }
};

let mainWindow;
let backendProcess = null;
const backendPort = 8000;
let backendLogFile = null;

// PGLite bridge server
let pgliteBridgeServer = null;
const pgliteBridgePort = 8001;

// Create screenshots directory
function ensureScreenshotDirectory() {
  const mirixDir = path.join(os.homedir(), '.mirix');
  const tmpDir = path.join(mirixDir, 'tmp');
  const imagesDir = path.join(tmpDir, 'images');
    
  if (!fs.existsSync(mirixDir)) {
    fs.mkdirSync(mirixDir, { recursive: true });
  }
  if (!fs.existsSync(tmpDir)) {
    fs.mkdirSync(tmpDir, { recursive: true });
  }
  if (!fs.existsSync(imagesDir)) {
    fs.mkdirSync(imagesDir, { recursive: true });
  }
  
  return imagesDir;
}

// Create backend log file
function createBackendLogFile() {
  const debugLogDir = path.join(os.homedir(), '.mirix', 'debug');
  if (!fs.existsSync(debugLogDir)) {
    fs.mkdirSync(debugLogDir, { recursive: true });
  }
  
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const logFileName = `backend-${timestamp}.log`;
  const logFilePath = path.join(debugLogDir, logFileName);
  
  // Create the log file with initial headers
  const initialLog = `=== MIRIX Backend Debug Log ===
Started: ${new Date().toISOString()}
Platform: ${process.platform}
Architecture: ${process.arch}
Node version: ${process.version}
Electron version: ${process.versions.electron}
Process execPath: ${process.execPath}
Process cwd: ${process.cwd()}
__dirname: ${__dirname}
Resources path: ${process.resourcesPath}
Is packaged: ${isPackaged}
Actually dev: ${actuallyDev}
========================================

`;
  
  fs.writeFileSync(logFilePath, initialLog);
  safeLog.log(`Created backend log file: ${logFilePath}`);
  
  return logFilePath;
}

// Helper function to log to backend log file
function logToBackendFile(message) {
  if (!backendLogFile) {
    backendLogFile = createBackendLogFile();
  }
  
  const timestamp = new Date().toISOString();
  const logMessage = `[${timestamp}] ${message}`;
  
  safeLog.log(logMessage);
  
  try {
    fs.appendFileSync(backendLogFile, logMessage + '\n');
  } catch (error) {
    safeLog.error('Failed to write to backend log file:', error);
  }
}

// Check if backend is running and healthy
async function isBackendHealthy() {
  try {
    const healthCheckResult = await checkBackendHealth();
    return true;
  } catch (error) {
    return false;
  }
}

// Ensure backend is running (start if not running)
async function ensureBackendRunning() {
  if (actuallyDev) {
    safeLog.log('Development mode: Backend should be running separately');
    return;
  }
  
  // Check if backend process is still running
  if (backendProcess && backendProcess.exitCode === null) {
    // Process is still running, check if it's healthy
    const isHealthy = await isBackendHealthy();
    if (isHealthy) {
      logToBackendFile('Backend is already running and healthy');
      return;
    } else {
      logToBackendFile('Backend process is running but not healthy, restarting...');
      stopBackendServer();
    }
  } else {
    logToBackendFile('Backend process is not running, starting...');
  }
  
  // Start the backend
  try {
    await startBackendServer();
    logToBackendFile('Backend started successfully');
  } catch (error) {
    logToBackendFile(`Failed to start backend: ${error.message}`);
    throw error;
  }
}

function startBackendServer() {
  if (actuallyDev) {
    safeLog.log('Development mode: Backend should be running separately');
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    try {
      const executableName = 'main';
      
      // Fix resourcesPath for packaged apps with detailed logging
      let actualResourcesPath = process.resourcesPath;
      logToBackendFile(`Initial resources path: ${actualResourcesPath}`);
      
      if (__dirname.indexOf('app.asar') !== -1) {
        const appAsarPath = __dirname.substring(0, __dirname.indexOf('app.asar'));
        actualResourcesPath = appAsarPath;
        logToBackendFile(`Adjusted resources path from asar: ${actualResourcesPath}`);
      }
      
      // Try multiple possible backend paths
      const possiblePaths = [
        path.join(actualResourcesPath, 'backend', executableName),
        path.join(actualResourcesPath, 'app', 'backend', executableName),
        path.join(actualResourcesPath, 'Contents', 'Resources', 'backend', executableName),
        path.join(actualResourcesPath, 'Contents', 'Resources', 'app', 'backend', executableName),
        path.join(process.resourcesPath, 'backend', executableName),
        path.join(process.resourcesPath, 'app', 'backend', executableName),
      ];
      
      logToBackendFile(`Searching for backend executable in ${possiblePaths.length} locations:`);
      
      let backendPath = null;
      for (const candidatePath of possiblePaths) {
        logToBackendFile(`  Checking: ${candidatePath}`);
        if (fs.existsSync(candidatePath)) {
          const stats = fs.statSync(candidatePath);
          logToBackendFile(`  ✅ Found! Size: ${stats.size} bytes, Modified: ${stats.mtime}`);
          logToBackendFile(`  File mode: ${stats.mode.toString(8)} (executable: ${(stats.mode & parseInt('111', 8)) !== 0})`);
          
          // Make sure it's executable
          if ((stats.mode & parseInt('111', 8)) === 0) {
            try {
              fs.chmodSync(candidatePath, '755');
              logToBackendFile(`  Made executable: ${candidatePath}`);
            } catch (chmodError) {
              logToBackendFile(`  Failed to make executable: ${chmodError.message}`);
            }
          }
          
          backendPath = candidatePath;
          break;
        } else {
          logToBackendFile(`  ❌ Not found`);
        }
      }
      
      if (!backendPath) {
        const error = `Backend executable not found in any of the expected locations:\n${possiblePaths.join('\n')}`;
        logToBackendFile(error);
        reject(new Error(error));
        return;
      }
      
      logToBackendFile(`Starting backend server on port ${backendPort}: ${backendPath}`);
      
      // Use user's .mirix directory as working directory (for .env files and SQLite database)
      const userMirixDir = path.join(os.homedir(), '.mirix');
      if (!fs.existsSync(userMirixDir)) {
        fs.mkdirSync(userMirixDir, { recursive: true });
        logToBackendFile(`Created working directory: ${userMirixDir}`);
      }
      const workingDir = userMirixDir;
      logToBackendFile(`Using working directory: ${workingDir}`);
      
      // Copy config files to working directory
      const configsDir = path.join(workingDir, 'configs');
      if (!fs.existsSync(configsDir)) {
        fs.mkdirSync(configsDir, { recursive: true });
        logToBackendFile(`Created configs directory: ${configsDir}`);
      }
      
      const sourceConfigsDir = path.join(actualResourcesPath, 'backend', 'configs');
      if (fs.existsSync(sourceConfigsDir)) {
        logToBackendFile(`Copying config files from: ${sourceConfigsDir}`);
        const configFiles = fs.readdirSync(sourceConfigsDir);
        for (const configFile of configFiles) {
          const sourcePath = path.join(sourceConfigsDir, configFile);
          const targetPath = path.join(configsDir, configFile);
          try {
            fs.copyFileSync(sourcePath, targetPath);
            logToBackendFile(`✅ Copied config: ${configFile}`);
          } catch (error) {
            logToBackendFile(`❌ Failed to copy config ${configFile}: ${error.message}`);
          }
        }
      } else {
        logToBackendFile(`❌ Source configs directory not found: ${sourceConfigsDir}`);
      }
      
      // Check if PGLite mode is enabled
      const usePGLite = process.env.MIRIX_USE_PGLITE === 'true';
      
      // Prepare environment variables
      const env = {
        ...process.env,
        PORT: backendPort.toString(),
        PYTHONPATH: workingDir,
        DEBUG: 'true',
        MIRIX_DEBUG: 'true',
        MIRIX_LOG_LEVEL: 'DEBUG'
      };
      
      if (usePGLite) {
        // Configure for PGLite mode
        env.MIRIX_USE_PGLITE = 'true';
        env.MIRIX_PGLITE_BRIDGE_URL = `http://127.0.0.1:${pgliteBridgePort}`;
        env.MIRIX_PG_URI = ''; // Clear PostgreSQL URI
        logToBackendFile('🔧 Configuring backend for PGLite mode');
      } else {
        // Configure for SQLite mode (default)
        env.MIRIX_PG_URI = ''; // Force SQLite fallback
        env.MIRIX_USE_PGLITE = 'false';
        logToBackendFile('🔧 Configuring backend for SQLite mode');
      }
      
      logToBackendFile(`Environment variables: PORT=${env.PORT}, PYTHONPATH=${env.PYTHONPATH}, MIRIX_PG_URI=${env.MIRIX_PG_URI}`);
      
      // Start backend with SQLite configuration
      backendProcess = spawn(backendPath, ['--host', '0.0.0.0', '--port', backendPort.toString()], {
        stdio: ['pipe', 'pipe', 'pipe'],
        detached: false,
        cwd: workingDir,
        env: env
      });

      let healthCheckStarted = false;

      backendProcess.stdout.on('data', (data) => {
        const output = data.toString().trim();
        logToBackendFile(`STDOUT: ${output}`);
        
        if (output.includes('Uvicorn running on') || 
            output.includes('Application startup complete') ||
            output.includes('Started server process')) {
          
          if (!healthCheckStarted) {
            healthCheckStarted = true;
            logToBackendFile('Backend server startup detected, starting health check...');
            setTimeout(() => {
              checkBackendHealth().then(() => {
                logToBackendFile('Backend health check passed, resolving startup');
                resolve();
              }).catch((healthError) => {
                logToBackendFile(`Backend health check failed: ${healthError.message}`);
                reject(healthError);
              });
            }, 3000);
          }
        }
      });

      backendProcess.stderr.on('data', (data) => {
        const output = data.toString();
        logToBackendFile(`STDERR: ${output}`);
        
        // Check stderr for startup messages too
        if (output.includes('Uvicorn running on') || 
            output.includes('Application startup complete') ||
            output.includes('Started server process')) {
          
          if (!healthCheckStarted) {
            healthCheckStarted = true;
            logToBackendFile('Backend server startup detected in stderr, starting health check...');
            setTimeout(() => {
              checkBackendHealth().then(() => {
                logToBackendFile('Backend health check passed, resolving startup');
                resolve();
              }).catch((healthError) => {
                logToBackendFile(`Backend health check failed: ${healthError.message}`);
                reject(healthError);
              });
            }, 3000);
          }
        }
      });

      backendProcess.on('close', (code) => {
        logToBackendFile(`Backend process exited with code ${code}`);
        if (code !== 0 && !healthCheckStarted) {
          reject(new Error(`Backend process exited with code ${code}`));
        }
      });

      backendProcess.on('error', (error) => {
        logToBackendFile(`Failed to start backend process: ${error.message}`);
        reject(error);
      });

      // Timeout fallback
      setTimeout(() => {
        if (backendProcess && backendProcess.exitCode === null && !healthCheckStarted) {
          logToBackendFile('Backend startup timeout, trying health check...');
          checkBackendHealth().then(() => {
            logToBackendFile('Health check passed despite timeout');
            resolve();
          }).catch((healthError) => {
            logToBackendFile(`Backend health check failed after timeout: ${healthError.message}`);
            reject(new Error(`Backend startup timeout: ${healthError.message}`));
          });
        }
      }, 30000);

      logToBackendFile('Backend server started');
    } catch (error) {
      safeLog.error('Failed to start backend server:', error);
      reject(error);
    }
  });
}

async function checkBackendHealth() {
  const maxRetries = 5;
  const retryDelay = 1000;
  
  for (let i = 0; i < maxRetries; i++) {
    try {
      logToBackendFile(`Health check attempt ${i + 1}/${maxRetries} - checking http://127.0.0.1:${backendPort}/health`);
      
      const healthCheckResult = await new Promise((resolve, reject) => {
        const req = http.get(`http://127.0.0.1:${backendPort}/health`, { timeout: 5000 }, (res) => {
          let data = '';
          
          res.on('data', chunk => {
            data += chunk;
          });
          
          res.on('end', () => {
            if (res.statusCode === 200) {
              logToBackendFile(`Health check response: ${data}`);
              resolve(data);
            } else {
              reject(new Error(`Health check failed with status: ${res.statusCode}, response: ${data}`));
            }
          });
        });
        
        req.on('error', (error) => {
          logToBackendFile(`Health check request error: ${error.message}`);
          reject(error);
        });
        
        req.setTimeout(5000, () => {
          req.destroy();
          reject(new Error('Health check timeout after 5 seconds'));
        });
      });
      
      logToBackendFile('✅ Backend health check passed');
      return healthCheckResult;
      
    } catch (error) {
      logToBackendFile(`❌ Health check attempt ${i + 1} failed: ${error.message} (code: ${error.code})`);
      
      if (i < maxRetries - 1) {
        logToBackendFile(`Retrying in ${retryDelay}ms...`);
        await new Promise(resolve => setTimeout(resolve, retryDelay));
      } else {
        logToBackendFile(`All health check attempts failed. Final error: ${error.message}`);
        throw error;
      }
    }
  }
}

function stopBackendServer() {
  if (backendProcess) {
    logToBackendFile('Stopping backend server...');
    backendProcess.kill();
    backendProcess = null;
    logToBackendFile('Backend server stopped');
  }
}

// PGLite bridge server management
async function startPGLiteBridgeServer() {
  try {
    if (pgliteBridgeServer) {
      logToBackendFile('PGLite bridge server already running');
      return;
    }
    
    logToBackendFile('Starting PGLite bridge server...');
    pgliteBridgeServer = await startPGLiteBridge(pgliteBridgePort);
    logToBackendFile(`✅ PGLite bridge server started on port ${pgliteBridgePort}`);
    
  } catch (error) {
    logToBackendFile(`❌ Failed to start PGLite bridge server: ${error.message}`);
    throw error;
  }
}

async function stopPGLiteBridgeServer() {
  if (pgliteBridgeServer) {
    logToBackendFile('Stopping PGLite bridge server...');
    stopPGLiteBridge();
    await closePGLite();
    pgliteBridgeServer = null;
    logToBackendFile('✅ PGLite bridge server stopped');
  }
}

async function isPGLiteBridgeHealthy() {
  try {
    const response = await new Promise((resolve, reject) => {
      const req = http.get(`http://127.0.0.1:${pgliteBridgePort}/health`, { timeout: 5000 }, (res) => {
        let data = '';
        res.on('data', chunk => data += chunk);
        res.on('end', () => {
          if (res.statusCode === 200) {
            resolve(data);
          } else {
            reject(new Error(`Health check failed: ${res.statusCode}`));
          }
        });
      });
      
      req.on('error', reject);
      req.setTimeout(5000, () => {
        req.destroy();
        reject(new Error('Health check timeout'));
      });
    });
    
    return true;
  } catch (error) {
    logToBackendFile(`PGLite bridge health check failed: ${error.message}`);
    return false;
  }
}

function createWindow() {
  ensureScreenshotDirectory();

  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false,
      preload: path.join(__dirname, 'preload.js')
    },
    icon: path.join(__dirname, 'icon.png'),
    titleBarStyle: 'default',
    show: false
  });

  const startUrl = actuallyDev 
    ? 'http://localhost:3000' 
    : `file://${path.join(__dirname, '../build/index.html')}`;
  
  mainWindow.loadURL(startUrl);

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    safeLog.log('MainWindow is ready to show');
    
    // Ensure backend is running when window is shown
    if (!actuallyDev) {
      ensureBackendRunning().catch((error) => {
        safeLog.error('Failed to ensure backend is running:', error);
      });
    }
  });

  // Listen for window show events
  mainWindow.on('show', () => {
    safeLog.log('Window shown - notifying renderer');
    mainWindow.webContents.send('window-show');
  });

  if (actuallyDev) {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
}

app.whenReady().then(async () => {
  safeLog.log('Electron ready - creating window immediately and starting backend in parallel...');
  
  createWindow();
  startBackendInBackground();
  
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    } else {
      // Window exists but user activated the app, ensure backend is running
      if (!actuallyDev) {
        ensureBackendRunning().catch((error) => {
          safeLog.error('Failed to ensure backend is running on activate:', error);
        });
      }
      
      // Notify renderer about app activation
      const focusedWindow = BrowserWindow.getFocusedWindow();
      if (focusedWindow) {
        focusedWindow.webContents.send('app-activate');
      }
    }
  });
});

async function cleanupOldTmpImages(maxAge = 7 * 24 * 60 * 60 * 1000) {
  try {
    const imagesDir = ensureScreenshotDirectory();
    const files = fs.readdirSync(imagesDir);
    const now = Date.now();
    let deletedCount = 0;

    for (const file of files) {
      if (!file.startsWith('screenshot-') && 
          (file.endsWith('.png') || file.endsWith('.jpg') || file.endsWith('.jpeg') || 
           file.endsWith('.gif') || file.endsWith('.bmp') || file.endsWith('.webp'))) {
        const filepath = path.join(imagesDir, file);
        const stats = fs.statSync(filepath);
        const age = now - stats.mtime.getTime();
        
        if (age > maxAge) {
          fs.unlinkSync(filepath);
          deletedCount++;
        }
      }
    }

    return {
      success: true,
      deletedCount: deletedCount
    };
  } catch (error) {
    safeLog.error('Failed to cleanup tmp images:', error);
    return {
      success: false,
      error: error.message
    };
  }
}

async function startBackendInBackground() {
  safeLog.log('Starting backend server in background...');
  
  try {
    logToBackendFile('Initial backend startup...');
    
    // Start PGLite bridge server if enabled
    const usePGLite = process.env.MIRIX_USE_PGLITE === 'true';
    if (usePGLite) {
      logToBackendFile('🔧 Starting PGLite bridge server...');
      await startPGLiteBridgeServer();
      logToBackendFile('✅ PGLite bridge server started');
    }
    
    await ensureBackendRunning();
    logToBackendFile('✅ Backend initialization complete');
    
    // Schedule cleanup of old tmp images after backend starts
    setTimeout(async () => {
      try {
        const result = await cleanupOldTmpImages();
        if (result.success && result.deletedCount > 0) {
          logToBackendFile(`Cleaned up ${result.deletedCount} old tmp images on startup`);
        }
      } catch (error) {
        logToBackendFile(`Failed to cleanup tmp images on startup: ${error.message}`);
      }
    }, 5000);
    
  } catch (error) {
    logToBackendFile(`❌ Backend initialization failed: ${error.message}`);
    logToBackendFile(`Error stack: ${error.stack}`);
    
    if (!actuallyDev) {
      let errorMessage = error.message || 'Unknown error';
      
      if (error.message && error.message.includes('ECONNREFUSED')) {
        errorMessage = 'Backend server failed to start - connection refused';
      } else if (error.message && error.message.includes('EADDRINUSE')) {
        errorMessage = 'Backend server failed to start - port already in use';
      } else if (error.message && error.message.includes('Backend process exited')) {
        errorMessage = 'Backend server crashed during startup';
      }
      
      const fullErrorMessage = `Failed to start the backend server: ${errorMessage}\n\nBackend log saved to: ${backendLogFile}`;
      
      dialog.showErrorBox(
        'Backend Startup Error', 
        fullErrorMessage
      );
    }
    
    safeLog.error(`Backend log saved to: ${backendLogFile}`);
  }
}

app.on('window-all-closed', async () => {
  stopBackendServer();
  await stopPGLiteBridgeServer();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', async () => {
  stopBackendServer();
  await stopPGLiteBridgeServer();
});

app.on('web-contents-created', (event, contents) => {
  contents.on('new-window', (event, navigationUrl) => {
    event.preventDefault();
    shell.openExternal(navigationUrl);
  });
});

// IPC handlers for file operations
ipcMain.handle('select-files', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile', 'multiSelections'],
    filters: [
      { name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'] },
      { name: 'All Files', extensions: ['*'] }
    ]
  });
  
  return result.filePaths;
});

ipcMain.handle('select-save-path', async (event, options = {}) => {
  const result = await dialog.showSaveDialog(mainWindow, {
    title: options.title || 'Save File',
    defaultPath: options.defaultName || 'memories_export.xlsx',
    filters: [
      { name: 'Excel Files', extensions: ['xlsx'] },
      { name: 'CSV Files', extensions: ['csv'] },
      { name: 'All Files', extensions: ['*'] }
    ]
  });
  
  return {
    canceled: result.canceled,
    filePath: result.filePath
  };
});

// IPC handler for taking screenshot
ipcMain.handle('take-screenshot', async () => {
  try {
    const imagesDir = ensureScreenshotDirectory();
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `screenshot-${timestamp}.png`;
    const filepath = path.join(imagesDir, filename);
    
    safeLog.log(`Taking screenshot: ${filepath}`);
    
    // Check if we're on macOS and ask for screen recording permissions
    if (process.platform === 'darwin') {
      // Check if we have screen recording permissions
      const hasScreenPermission = systemPreferences.getMediaAccessStatus('screen');
      safeLog.log(`Screen recording permission status: ${hasScreenPermission}`);
      
      if (hasScreenPermission !== 'granted') {
        // Request screen recording permissions
        const permissionGranted = await systemPreferences.askForMediaAccess('screen');
        safeLog.log(`Screen recording permission granted: ${permissionGranted}`);
        
        if (!permissionGranted) {
          throw new Error('Screen recording permission not granted. Please grant screen recording permissions in System Preferences > Security & Privacy > Screen Recording and restart the application.');
        }
      }
    }
    
    // Try to take screenshot with better error handling
    try {
      const imgBuffer = await screenshot();
      
      // Write the buffer to file
      fs.writeFileSync(filepath, imgBuffer);
      
      safeLog.log(`Screenshot buffer captured: ${imgBuffer.length} bytes`);
    } catch (screenshotError) {
      safeLog.error('Screenshot capture failed:', screenshotError);
      
      // Try alternative method if the first one fails
      try {
        safeLog.log('Trying alternative screenshot method...');
        await screenshot(filepath);
      } catch (altError) {
        safeLog.error('Alternative screenshot method also failed:', altError);
        
        // As a last resort, create a test image file for debugging
        if (actuallyDev) {
          safeLog.log('Creating test screenshot file for debugging...');
          try {
            const testBuffer = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==', 'base64');
            fs.writeFileSync(filepath, testBuffer);
            safeLog.log('Test screenshot file created successfully');
          } catch (testError) {
            safeLog.error('Failed to create test screenshot file:', testError);
            throw new Error(`Screenshot capture failed: ${screenshotError.message}. Alternative method error: ${altError.message}. Test file creation error: ${testError.message}`);
          }
        } else {
          throw new Error(`Screenshot capture failed: ${screenshotError.message}. Alternative method error: ${altError.message}`);
        }
      }
    }
    
    // Verify the file was created
    if (!fs.existsSync(filepath)) {
      throw new Error(`Screenshot file was not created: ${filepath}`);
    }
    
    const stats = fs.statSync(filepath);
    safeLog.log(`Screenshot saved successfully: ${filepath} (${stats.size} bytes)`);
    
    return {
      success: true,
      filepath: filepath,
      filename: filename,
      size: stats.size
    };
  } catch (error) {
    safeLog.error('Failed to take screenshot:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

// IPC handler for taking screenshot of specific display
ipcMain.handle('take-screenshot-display', async (event, displayId = 0) => {
  try {
    const imagesDir = ensureScreenshotDirectory();
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `screenshot-display-${displayId}-${timestamp}.png`;
    const filepath = path.join(imagesDir, filename);

    // Get list of displays and take screenshot of specific display
    const displays = await screenshot.listDisplays();
    if (displayId >= displays.length) {
      throw new Error(`Display ${displayId} not found. Available displays: ${displays.length}`);
    }

    const imgBuffer = await screenshot({ screen: displays[displayId].id });
    
    // Save screenshot
    fs.writeFileSync(filepath, imgBuffer);
    
    safeLog.log(`Screenshot of display ${displayId} saved: ${filepath}`);
    
    return {
      success: true,
      filepath: filepath,
      filename: filename,
      size: imgBuffer.length,
      displayId: displayId
    };
  } catch (error) {
    safeLog.error('Failed to take screenshot of display:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

// IPC handler for getting available displays
ipcMain.handle('list-displays', async () => {
  try {
    const displays = await screenshot.listDisplays();
    return {
      success: true,
      displays: displays.map((display, index) => ({
        id: display.id,
        index: index,
        name: display.name || `Display ${index + 1}`,
        bounds: display.bounds
      }))
    };
  } catch (error) {
    safeLog.error('Failed to list displays:', error);
    return {
      success: false,
      error: error.message,
      displays: []
    };
  }
});

// IPC handler for cleaning up old screenshots
ipcMain.handle('cleanup-screenshots', async (event, maxAge = 24 * 60 * 60 * 1000) => {
  try {
    const imagesDir = ensureScreenshotDirectory();
    const files = fs.readdirSync(imagesDir);
    const now = Date.now();
    let deletedCount = 0;

    for (const file of files) {
      if (file.startsWith('screenshot-') && file.endsWith('.png')) {
        const filepath = path.join(imagesDir, file);
        const stats = fs.statSync(filepath);
        const age = now - stats.mtime.getTime();
        
        if (age > maxAge) {
          fs.unlinkSync(filepath);
          deletedCount++;
        }
      }
    }

    return {
      success: true,
      deletedCount: deletedCount
    };
  } catch (error) {
    safeLog.error('Failed to cleanup screenshots:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

// IPC handler for reading image as base64 (for similarity comparison)
ipcMain.handle('read-image-base64', async (event, filepath) => {
  try {
    safeLog.log(`Reading image as base64: ${filepath}`);
    
    if (!fs.existsSync(filepath)) {
      throw new Error(`File does not exist: ${filepath}`);
    }

    const stats = fs.statSync(filepath);
    safeLog.log(`Image file exists, size: ${stats.size} bytes`);
    
    const imageBuffer = fs.readFileSync(filepath);
    const base64Data = imageBuffer.toString('base64');
    const mimeType = 'image/png'; // Assuming PNG format for screenshots
    const dataUrl = `data:${mimeType};base64,${base64Data}`;

    safeLog.log(`Successfully read image as base64: ${filepath} (${imageBuffer.length} bytes)`);

    return {
      success: true,
      dataUrl: dataUrl,
      base64: base64Data,
      size: imageBuffer.length
    };
  } catch (error) {
    safeLog.error('Failed to read image as base64:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

// IPC handler for deleting screenshot files (used when screenshots are too similar)
ipcMain.handle('delete-screenshot', async (event, filepath) => {
  try {
    safeLog.log(`Attempting to delete screenshot: ${filepath}`);
    
    if (!fs.existsSync(filepath)) {
      safeLog.warn(`Tried to delete non-existent file: ${filepath}`);
      return {
        success: true, // Consider it successful if file doesn't exist
        message: 'File does not exist'
      };
    }

    // Only allow deletion of files in the screenshots directory for security
    const imagesDir = ensureScreenshotDirectory();
    const normalizedFilepath = path.resolve(filepath);
    const normalizedImagesDir = path.resolve(imagesDir);
    
    if (!normalizedFilepath.startsWith(normalizedImagesDir)) {
      throw new Error('Can only delete files in the screenshots directory');
    }

    fs.unlinkSync(filepath);
    safeLog.log(`Screenshot deleted successfully: ${filepath}`);

    return {
      success: true,
      message: 'File deleted successfully'
    };
  } catch (error) {
    safeLog.error('Failed to delete screenshot:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

// IPC handler for saving image files to tmp directory
ipcMain.handle('save-image-to-tmp', async (event, sourcePath, filename) => {
  try {
    const imagesDir = ensureScreenshotDirectory();
    const targetPath = path.join(imagesDir, filename);

    // Check if source file exists
    if (!fs.existsSync(sourcePath)) {
      throw new Error(`Source file does not exist: ${sourcePath}`);
    }

    // Copy the file to the tmp directory
    fs.copyFileSync(sourcePath, targetPath);
    
    safeLog.log(`Image saved to tmp directory: ${targetPath}`);

    return targetPath;
  } catch (error) {
    safeLog.error('Failed to save image to tmp directory:', error);
    throw error;
  }
});

// IPC handler for saving image buffer to tmp directory
ipcMain.handle('save-image-buffer-to-tmp', async (event, arrayBuffer, filename) => {
  try {
    const imagesDir = ensureScreenshotDirectory();
    const targetPath = path.join(imagesDir, filename);

    // Convert ArrayBuffer to Buffer
    const buffer = Buffer.from(arrayBuffer);
    
    // Write the buffer to file
    fs.writeFileSync(targetPath, buffer);
    
    safeLog.log(`Image buffer saved to tmp directory: ${targetPath}`);

    return targetPath;
  } catch (error) {
    safeLog.error('Failed to save image buffer to tmp directory:', error);
    throw error;
  }
});

// IPC handler for cleaning up old tmp images
ipcMain.handle('cleanup-tmp-images', async (event, maxAge = 7 * 24 * 60 * 60 * 1000) => {
  try {
    const imagesDir = ensureScreenshotDirectory();
    const files = fs.readdirSync(imagesDir);
    const now = Date.now();
    let deletedCount = 0;

    for (const file of files) {
      // Clean up any image files older than maxAge, but skip screenshot files
      if (!file.startsWith('screenshot-') && 
          (file.endsWith('.png') || file.endsWith('.jpg') || file.endsWith('.jpeg') || 
           file.endsWith('.gif') || file.endsWith('.bmp') || file.endsWith('.webp'))) {
        const filepath = path.join(imagesDir, file);
        const stats = fs.statSync(filepath);
        const age = now - stats.mtime.getTime();
        
        if (age > maxAge) {
          fs.unlinkSync(filepath);
          deletedCount++;
        }
      }
    }

    safeLog.log(`Cleaned up ${deletedCount} old tmp images`);

    return {
      success: true,
      deletedCount: deletedCount
    };
  } catch (error) {
    safeLog.error('Failed to cleanup tmp images:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

// Handle app protocol for deep linking (optional)
if (process.defaultApp) {
  if (process.argv.length >= 2) {
    app.setAsDefaultProtocolClient('mirix', process.execPath, [path.resolve(process.argv[1])]);
  }
} else {
  app.setAsDefaultProtocolClient('mirix');
}

const createMenu = () => {
  const template = [
    {
      label: 'File',
      submenu: [
        {
          label: 'Quit',
          accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
          click: () => {
            app.quit();
          }
        }
      ]
    },
    {
      label: 'Edit',
      submenu: [
        { label: 'Undo', accelerator: 'CmdOrCtrl+Z', role: 'undo' },
        { label: 'Redo', accelerator: 'Shift+CmdOrCtrl+Z', role: 'redo' },
        { type: 'separator' },
        { label: 'Cut', accelerator: 'CmdOrCtrl+X', role: 'cut' },
        { label: 'Copy', accelerator: 'CmdOrCtrl+C', role: 'copy' },
        { label: 'Paste', accelerator: 'CmdOrCtrl+V', role: 'paste' }
      ]
    },
    {
      label: 'View',
      submenu: [
        { label: 'Reload', accelerator: 'CmdOrCtrl+R', role: 'reload' },
        { label: 'Force Reload', accelerator: 'CmdOrCtrl+Shift+R', role: 'forceReload' },
        { label: 'Toggle Developer Tools', accelerator: process.platform === 'darwin' ? 'Alt+Cmd+I' : 'Ctrl+Shift+I', role: 'toggleDevTools' },
        { type: 'separator' },
        { label: 'Actual Size', accelerator: 'CmdOrCtrl+0', role: 'resetZoom' },
        { label: 'Zoom In', accelerator: 'CmdOrCtrl+Plus', role: 'zoomIn' },
        { label: 'Zoom Out', accelerator: 'CmdOrCtrl+-', role: 'zoomOut' },
        { type: 'separator' },
        { label: 'Toggle Fullscreen', accelerator: process.platform === 'darwin' ? 'Ctrl+Cmd+F' : 'F11', role: 'togglefullscreen' }
      ]
    },
    {
      label: 'Window',
      submenu: [
        { label: 'Minimize', accelerator: 'CmdOrCtrl+M', role: 'minimize' },
        { label: 'Close', accelerator: 'CmdOrCtrl+W', role: 'close' }
      ]
    }
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
};

app.whenReady().then(() => {
  createMenu();
}); 