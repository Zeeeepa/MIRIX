const express = require('express');
const path = require('path');
const os = require('os');
const fs = require('fs');

/**
 * PGLite Bridge Server
 * 
 * This server runs in the Electron main process and provides an HTTP bridge
 * between the Python backend and PGLite database running in the frontend.
 */

let pgliteDb = null;
let bridgeServer = null;

async function initializePGLite() {
  try {
    console.log('🔧 Starting PGLite initialization...');
    console.log('🔧 Node.js version:', process.version);
    console.log('🔧 Process platform:', process.platform);
    console.log('🔧 Process arch:', process.arch);
    
    // Check WebAssembly support first
    console.log('🔧 WebAssembly support check:');
    console.log('  - typeof WebAssembly:', typeof WebAssembly);
    console.log('  - WebAssembly.instantiate:', typeof WebAssembly?.instantiate);
    console.log('  - WebAssembly.compile:', typeof WebAssembly?.compile);
    
    if (typeof WebAssembly === 'undefined') {
      throw new Error('WebAssembly is not supported in this environment');
    }
    
    // Dynamic import of PGLite (ES module)
    console.log('🔧 Importing PGLite module...');
    const { PGlite } = await import('@electric-sql/pglite');
    console.log('✅ PGLite module imported successfully');
    
    // Create data directory for PGLite
    const mirixDir = path.join(os.homedir(), '.mirix');
    const pgliteDir = path.join(mirixDir, 'pglite');
    
    if (!fs.existsSync(mirixDir)) {
      fs.mkdirSync(mirixDir, { recursive: true });
    }
    
    if (!fs.existsSync(pgliteDir)) {
      fs.mkdirSync(pgliteDir, { recursive: true });
    }
    
    // Try in-memory first with simplified options
    console.log(`🔧 Initializing PGLite database in memory...`);
    
    try {
      console.log('🔧 Creating PGLite instance with minimal config...');
      
      // Try with minimal configuration first
      pgliteDb = new PGlite();
      console.log('✅ PGLite instance created, waiting for ready...');
      
      // Add timeout to waitReady to avoid hanging
      const readyPromise = pgliteDb.waitReady;
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error('PGLite initialization timeout')), 30000);
      });
      
      await Promise.race([readyPromise, timeoutPromise]);
      
      console.log('✅ PGLite database initialized successfully (in-memory)');
      return pgliteDb;
      
    } catch (memoryError) {
      console.warn('⚠️ In-memory PGLite failed:', memoryError.message);
      console.warn('⚠️ Error stack:', memoryError.stack);
      
      // Don't try persistent storage if WebAssembly itself is failing
      if (memoryError.message.includes('Aborted') || memoryError.message.includes('WebAssembly')) {
        throw new Error(`PGLite WebAssembly initialization failed: ${memoryError.message}. This might be due to Electron version compatibility. Consider updating Electron or using regular PostgreSQL mode.`);
      }
      
      // Fallback to persistent storage
      const dbPath = path.join(pgliteDir, 'mirix.db');
      console.log(`🔧 Trying persistent PGLite database at: ${dbPath}`);
      
      pgliteDb = new PGlite(dbPath);
      await Promise.race([pgliteDb.waitReady, timeoutPromise]);
      
      console.log('✅ PGLite database initialized successfully (persistent)');
      return pgliteDb;
    }
    
  } catch (error) {
    console.error('❌ Failed to initialize PGLite:', error);
    console.error('❌ Error stack:', error.stack);
    console.error('❌ This might be due to WebAssembly compatibility issues in Electron');
    console.error('❌ Possible solutions:');
    console.error('  1. Update Electron to a newer version');
    console.error('  2. Use regular PostgreSQL mode (without MIRIX_USE_PGLITE)');
    console.error('  3. Check if your system supports WebAssembly threads');
    
    throw error;
  }
}

async function startPGLiteBridge(port = 8001) {
  try {
    // Initialize PGLite if not already done
    if (!pgliteDb) {
      await initializePGLite();
    }
    
    // Create Express app
    const app = express();
    app.use(express.json({ limit: '10mb' }));
    app.use(express.urlencoded({ extended: true, limit: '10mb' }));
    
    // CORS middleware
    app.use((req, res, next) => {
      res.header('Access-Control-Allow-Origin', '*');
      res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
      res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
      
      if (req.method === 'OPTIONS') {
        res.sendStatus(200);
      } else {
        next();
      }
    });
    
    // Health check endpoint
    app.get('/health', (req, res) => {
      res.json({ 
        status: 'ok', 
        database: pgliteDb ? 'connected' : 'not_connected',
        timestamp: new Date().toISOString()
      });
    });
    
    // Execute query endpoint
    app.post('/query', async (req, res) => {
      try {
        const { query, params = [] } = req.body;
        
        if (!query) {
          return res.status(400).json({ error: 'Query is required' });
        }
        
        console.log(`📊 Executing query: ${query.substring(0, 100)}...`);
        
        // Execute query with PGLite
        const result = await pgliteDb.query(query, params);
        
        // Format response to match expected structure
        const response = {
          rows: result.rows || [],
          rowCount: result.rowCount || 0,
          fields: result.fields || [],
          affectedRows: result.affectedRows || 0
        };
        
        res.json(response);
        
      } catch (error) {
        console.error('❌ Query execution error:', error);
        res.status(500).json({ 
          error: error.message,
          code: error.code || 'UNKNOWN_ERROR'
        });
      }
    });
    
    // Execute SQL statements endpoint
    app.post('/exec', async (req, res) => {
      try {
        const { sql } = req.body;
        
        if (!sql) {
          return res.status(400).json({ error: 'SQL is required' });
        }
        
        console.log(`📊 Executing SQL: ${sql.substring(0, 100)}...`);
        
        // Execute SQL with PGLite
        const result = await pgliteDb.exec(sql);
        
        // Format response
        const response = {
          success: true,
          rowCount: result.rowCount || 0,
          affectedRows: result.affectedRows || 0
        };
        
        res.json(response);
        
      } catch (error) {
        console.error('❌ SQL execution error:', error);
        res.status(500).json({ 
          error: error.message,
          code: error.code || 'UNKNOWN_ERROR'
        });
      }
    });
    
    // Transaction support endpoint
    app.post('/transaction', async (req, res) => {
      try {
        const { queries } = req.body;
        
        if (!queries || !Array.isArray(queries)) {
          return res.status(400).json({ error: 'Queries array is required' });
        }
        
        console.log(`📊 Executing transaction with ${queries.length} queries`);
        
        // Execute queries in transaction
        const results = [];
        
        for (const { query, params = [] } of queries) {
          const result = await pgliteDb.query(query, params);
          results.push({
            rows: result.rows || [],
            rowCount: result.rowCount || 0,
            affectedRows: result.affectedRows || 0
          });
        }
        
        res.json({ success: true, results });
        
      } catch (error) {
        console.error('❌ Transaction error:', error);
        res.status(500).json({ 
          error: error.message,
          code: error.code || 'UNKNOWN_ERROR'
        });
      }
    });
    
    // Database schema endpoint
    app.get('/schema', async (req, res) => {
      try {
        const result = await pgliteDb.query(`
          SELECT 
            table_name,
            column_name,
            data_type,
            is_nullable,
            column_default
          FROM information_schema.columns 
          WHERE table_schema = 'public'
          ORDER BY table_name, ordinal_position;
        `);
        
        res.json({
          tables: result.rows || [],
          success: true
        });
        
      } catch (error) {
        console.error('❌ Schema query error:', error);
        res.status(500).json({ 
          error: error.message,
          code: error.code || 'UNKNOWN_ERROR'
        });
      }
    });
    
    // Start server
    return new Promise((resolve, reject) => {
      bridgeServer = app.listen(port, '127.0.0.1', (err) => {
        if (err) {
          console.error(`❌ Failed to start PGLite bridge server: ${err.message}`);
          reject(err);
        } else {
          console.log(`🚀 PGLite bridge server running on http://127.0.0.1:${port}`);
          resolve(bridgeServer);
        }
      });
    });
    
  } catch (error) {
    console.error('❌ Failed to start PGLite bridge:', error);
    throw error;
  }
}

function stopPGLiteBridge() {
  if (bridgeServer) {
    console.log('🔄 Stopping PGLite bridge server...');
    bridgeServer.close();
    bridgeServer = null;
    console.log('✅ PGLite bridge server stopped');
  }
}

// Close database connection
async function closePGLite() {
  if (pgliteDb) {
    try {
      await pgliteDb.close();
      pgliteDb = null;
      console.log('✅ PGLite database closed');
    } catch (error) {
      console.error('❌ Error closing PGLite database:', error);
    }
  }
}

module.exports = {
  startPGLiteBridge,
  stopPGLiteBridge,
  closePGLite,
  initializePGLite
}; 