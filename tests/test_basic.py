"""
Basic unit tests for Mirix - no API keys required.
These tests run quickly and validate core functionality.
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestImports:
    """Test that core modules can be imported."""
    
    def test_import_main_package(self):
        """Test importing main Mirix package."""
        from mirix import Mirix
        assert Mirix is not None


class TestHelpers:
    """Test helper functions."""
    
    def test_datetime_helpers(self):
        """Test datetime helper functions."""
        from datetime import datetime
        from mirix.helpers.datetime_helpers import get_utc_time
        
        utc_time = get_utc_time()
        assert isinstance(utc_time, datetime)
    
    def test_json_helpers(self):
        """Test JSON helpers."""
        from mirix.helpers.json_helpers import json_loads, json_dumps
        
        # Test json_loads
        result = json_loads('{"key": "value"}')
        assert result == {"key": "value"}
        
        # Test json_dumps
        data = {"test": 123}
        json_str = json_dumps(data)
        assert "test" in json_str
        assert "123" in json_str


class TestConfiguration:
    """Test configuration loading."""
    
    def test_default_config_exists(self):
        """Test that default config file exists."""
        config_path = Path("mirix/configs/mirix.yaml")
        assert config_path.exists()
    
    def test_example_configs_exist(self):
        """Test that example configs exist."""
        examples_dir = Path("mirix/configs/examples")
        assert examples_dir.exists()
        assert list(examples_dir.glob("*.yaml"))


class TestProjectStructure:
    """Test project structure."""
    
    def test_required_directories_exist(self):
        """Test that required directories exist."""
        assert Path("mirix").exists()
        assert Path("mirix/agent").exists()
        assert Path("mirix/schemas").exists()
        assert Path("tests").exists()
    
    def test_readme_exists(self):
        """Test that README exists."""
        assert Path("README.md").exists()
    
    def test_license_exists(self):
        """Test that LICENSE exists."""
        assert Path("LICENSE").exists()

