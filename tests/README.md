# Mirix Tests

## Setup

```bash
# Install dependencies
pip install -e .

# Set API key
export GEMINI_API_KEY=your_api_key_here
# Or on Windows:
# set GEMINI_API_KEY=your_api_key_here
```

## Test Files Overview

| File | Tests | Type | Speed | Description |
|------|-------|------|-------|-------------|
| `test_memory_server.py` | 18 | Unit | Fast (~20s) | Direct `SyncServer()` calls, no network |
| `test_memory_integration.py` | 22 | Integration | Slow (~2-5min) | REST API via real server + client |

## Run Tests

```bash
# Run all tests (server + integration)
pytest -v

# Server-side tests only (fast, no real server needed)
pytest tests/test_memory_server.py -v

# Integration tests only (requires server startup)
pytest tests/test_memory_integration.py -v -m integration -s

# Skip integration tests (runs server tests only)
pytest -m "not integration" -v
```

## Test Coverage

Both test files cover all 5 memory types with comprehensive search testing:

### Memory Operations
- ✅ **Episodic Memory**: Insert events, search by summary/details (bm25, embedding)
- ✅ **Procedural Memory**: Insert procedures, search by summary/steps (bm25, embedding)
- ✅ **Resource Memory**: Insert resources, search by summary (bm25, embedding) / content (bm25 only)
- ✅ **Knowledge Vault**: Insert knowledge, search by caption (bm25, embedding) / secret_value (bm25)
- ✅ **Semantic Memory**: Insert items, search by name/summary/details (bm25, embedding)

### Search Methods
- **BM25**: Fast keyword-based ranking search
- **Embedding**: Vector similarity search for semantic matches
- **All Types**: Cross-memory search across all 5 memory types

## Prerequisites

**Important**: Server-side tests require existing agents in the database.

### Initial Setup (one-time)
```bash
# Terminal 1: Start server
python scripts/start_server.py

# Terminal 2: Start client (this creates demo-user and initializes agents)
python run_client.py
```

This creates the `demo-user` in `demo-org` organization with all required memory agents.

## Common Options

```bash
# Show print statements
pytest -v -s

# Run specific test
pytest tests/test_memory_server.py::TestDirectEpisodicMemory::test_insert_event -v

# With coverage
pytest --cov=mirix --cov-report=html

# Debug on failure
pytest --pdb
```
