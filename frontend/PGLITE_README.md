# PGLite Integration for Mirix

This document explains how to use PGLite (PostgreSQL for the client-side) with the Mirix application instead of the default SQLite database.

## Overview

PGLite allows you to run PostgreSQL directly in the browser/Electron environment, providing better PostgreSQL compatibility while maintaining the benefits of a client-side database. The architecture consists of:

- **PGLite Database**: Runs in the Electron main process
- **PGLite Bridge Server**: HTTP server (port 8001) that bridges Python backend to PGLite
- **Python Backend**: Connects to PGLite via HTTP bridge instead of directly to SQLite

## Architecture

```
Python Backend (port 8000) → HTTP Bridge (port 8001) → PGLite Database (Electron process)
```

## Setup Instructions

### 1. Development Mode (electron-dev)

#### Option A: Using the convenience script
```bash
cd frontend
npm run pglite-dev
```

#### Option B: Manual setup
1. Copy the environment configuration:
   ```bash
   cp frontend/pglite.env.example frontend/.env
   ```

2. Start the Python backend separately:
   ```bash
   cd /path/to/mirix/backend
   python -m mirix.server --host 0.0.0.0 --port 8000
   ```

3. Start the frontend with PGLite environment:
   ```bash
   cd frontend
   MIRIX_USE_PGLITE=true npm run electron-dev
   ```

### 2. Production Mode (electron-pack)

```bash
cd frontend
npm run pglite-pack
```

This will build the application with PGLite enabled and create a distributable package.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MIRIX_USE_PGLITE` | Enable PGLite mode | `false` |
| `MIRIX_PGLITE_BRIDGE_URL` | PGLite bridge server URL | `http://127.0.0.1:8001` |
| `MIRIX_PG_URI` | PostgreSQL URI (should be empty for PGLite) | `` |
| `MIRIX_DEBUG` | Enable debug logging | `false` |
| `MIRIX_LOG_LEVEL` | Log level for debugging | `INFO` |

## File Structure

```
frontend/
├── src/
│   └── pglite-bridge.js          # PGLite bridge server implementation
├── public/
│   └── electron.js               # Modified to include PGLite support
├── scripts/
│   └── start-pglite-dev.js       # Development script for PGLite mode
├── pglite.env.example            # Example environment configuration
└── package.json                  # Updated with PGLite scripts
```

## Database Storage

- **Development**: `~/.mirix/pglite/mirix.db`
- **Production**: Same location, persisted across app restarts

## Troubleshooting

### Common Issues

1. **PGLite bridge server fails to start**
   - Check if port 8001 is available
   - Verify `@electric-sql/pglite` is properly installed
   - Check Electron console for error messages

2. **Backend can't connect to PGLite**
   - Ensure `MIRIX_USE_PGLITE=true` is set
   - Verify bridge server is running on port 8001
   - Check backend logs for connection errors

3. **Database schema issues**
   - PGLite uses PostgreSQL syntax, not SQLite
   - Some SQLite-specific features may not work
   - Check migration scripts for compatibility

### Debug Mode

Enable debug logging:
```bash
MIRIX_DEBUG=true MIRIX_LOG_LEVEL=DEBUG npm run pglite-dev
```

### Health Checks

The bridge server provides health endpoints:
- `GET http://127.0.0.1:8001/health` - Bridge server health
- `GET http://127.0.0.1:8001/schema` - Database schema info

## API Endpoints

The PGLite bridge server exposes the following endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/query` | Execute SQL query |
| POST | `/exec` | Execute SQL statements |
| POST | `/transaction` | Execute multiple queries in transaction |
| GET | `/schema` | Get database schema |

## Migration from SQLite

When switching from SQLite to PGLite:

1. **Backup your SQLite database**: `~/.mirix/sqlite.db`
2. **Start with PGLite mode**: Database will be recreated
3. **Migrate data if needed**: Use database export/import tools
4. **Test thoroughly**: Some SQL syntax may need adjustment

## Performance Considerations

- **Memory Usage**: PGLite uses more memory than SQLite
- **Startup Time**: Initial database creation may take longer
- **File Size**: PGLite database files are typically larger
- **Query Performance**: Generally similar to SQLite for small datasets

## Development Tips

1. **Use the convenience script**: `npm run pglite-dev` for easy development
2. **Check logs**: Both Electron and backend logs provide valuable debugging info
3. **Test incrementally**: Start with basic queries before complex operations
4. **Schema validation**: Use `/schema` endpoint to verify database structure

## Support

For issues specific to PGLite integration:
1. Check the Electron console for bridge server logs
2. Verify backend logs for connection issues
3. Test the bridge endpoints directly with curl/Postman
4. Ensure all environment variables are correctly set

For PGLite-specific issues, refer to the [PGLite documentation](https://github.com/electric-sql/pglite). 