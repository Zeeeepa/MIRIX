"""
API integration tests - requires API keys.
These tests will be skipped if API keys are not available.
"""

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set"
)


class TestMirixBasic:
    """Basic Mirix SDK tests."""
    
    @pytest.fixture
    def mirix_agent(self):
        """Create a Mirix agent for testing."""
        from mirix import Mirix
        # Use the Gemini config file which has all the proper settings
        return Mirix(
            api_key=os.getenv("GEMINI_API_KEY"),
            config_path="mirix/configs/examples/mirix_gemini.yaml"
        )
    
    def test_initialization(self, mirix_agent):
        """Test that Mirix initializes correctly."""
        assert mirix_agent is not None
    
    def test_add_memory(self, mirix_agent):
        """Test adding a simple memory."""
        # Just verify it runs without errors (add() returns None)
        mirix_agent.add("Test memory: Python is a programming language")
    
    def test_list_users(self, mirix_agent):
        """Test listing users."""
        users = mirix_agent.list_users()
        assert len(users) > 0
        assert users[0].name == "default_user"
    
    def test_create_user(self, mirix_agent):
        """Test creating a new user."""
        user = mirix_agent.create_user(user_name="test_user")
        assert user.name == "test_user"
    

class TestMemoryOperations:
    """Test memory operations."""
    
    @pytest.fixture
    def mirix_agent(self):
        """Create a Mirix agent for testing."""
        from mirix import Mirix
        # Use the Gemini config file which has all the proper settings
        return Mirix(
            api_key=os.getenv("GEMINI_API_KEY"),
            config_path="mirix/configs/examples/mirix_gemini.yaml"
        )
    
    # def test_user_specific_memory(self, mirix_agent):
    #     """Test user-specific memory."""
    #     user = mirix_agent.create_user(user_name="alice")
        
    #     # Add memory for specific user
    #     mirix_agent.add("Alice likes chocolate", user_id=user.id)
        
    #     # Query for that user
    #     response = mirix_agent.chat("What do I like?", user_id=user.id)
    #     assert response is not None

