# Mirix Tests

Simple testing setup using pytest.

## Structure

- `test_basic.py` - Unit tests (no API keys needed)
- `test_api.py` - API integration tests (requires API keys, auto-skipped if not available)
- `test_sdk.py` - Your existing SDK tests (optional)
- `test_memory.py` - Your existing memory tests (optional)

## Quick Start

### Install dependencies
```bash
pip install -e .
pip install pytest pytest-cov python-dotenv
```

### Run all tests
```bash
pytest
```

### Run specific tests
```bash
# Just unit tests (fast, no API key needed)
pytest tests/test_basic.py

# Just API tests (needs API key)
pytest tests/test_api.py

# With verbose output
pytest -v

# With coverage
pytest --cov=mirix --cov-report=html
```

## Setup for API Tests

Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_api_key_here
```

Tests that need API keys will automatically skip if keys aren't available.

## GitHub Actions

Tests run automatically on every push. The workflow:
1. Runs on Ubuntu, Windows, and macOS
2. Tests Python 3.10, 3.11, and 3.12
3. Runs all tests (skips API tests if no keys configured)

To enable API tests in CI/CD:
1. Go to GitHub: Settings → Secrets and variables → Actions
2. Add secret: `GEMINI_API_KEY`

## Writing Tests

Use pytest fixtures and standard patterns:

```python
import pytest
from mirix import Mirix

class TestMyFeature:
    """Test my feature."""
    
    def test_something(self):
        """Test something specific."""
        result = some_function()
        assert result == expected
```

That's it! Keep it simple.
