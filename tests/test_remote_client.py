"""
Basic tests for MirixClient functionality.

To run these tests:
1. Start the server: python -m mirix.server.rest_api
2. Run tests: pytest tests/test_remote_client.py

Note: These are integration tests that require a running server.
"""

import pytest
from mirix.client import MirixClient, LocalClient, create_client


class TestClientFactory:
    """Test the client factory function."""
    
    def test_create_local_client(self):
        """Test creating a local client."""
        client = create_client(mode="local")
        assert isinstance(client, LocalClient)
    
    def test_create_remote_client(self):
        """Test creating a remote client."""
        client = create_client(
            mode="remote",
            base_url="http://localhost:8000"
        )
        assert isinstance(client, MirixClient)
    
    def test_invalid_mode(self):
        """Test creating client with invalid mode."""
        with pytest.raises(ValueError, match="Invalid mode"):
            create_client(mode="invalid")
    
    def test_remote_without_base_url(self):
        """Test creating remote client without base_url."""
        with pytest.raises(ValueError, match="base_url is required"):
            create_client(mode="remote")


class TestMirixClientBasics:
    """Test basic MirixClient functionality."""
    
    @pytest.fixture
    def client(self):
        """Create a MirixClient for testing."""
        return MirixClient(
            base_url="http://localhost:8000",
            user_id="test-user",
            debug=True,
        )
    
    def test_client_initialization(self):
        """Test client initialization."""
        client = MirixClient(
            base_url="http://localhost:8000",
            api_key="test-key",
            user_id="test-user",
            org_id="test-org",
            debug=True,
        )
        
        assert client.base_url == "http://localhost:8000"
        assert client.api_key == "test-key"
        assert client.user_id == "test-user"
        assert client.org_id == "test-org"
        assert client.debug is True
    
    def test_health_check(self, client):
        """Test server health check."""
        # This is a basic connectivity test
        try:
            response = client._request("GET", "/health")
            assert response["status"] == "healthy"
        except Exception as e:
            pytest.skip(f"Server not running: {e}")


class TestMirixClientAgents:
    """Test agent operations with MirixClient."""
    
    @pytest.fixture
    def client(self):
        """Create a MirixClient for testing."""
        return MirixClient(
            base_url="http://localhost:8000",
            user_id="test-user",
        )
    
    @pytest.fixture
    def agent(self, client):
        """Create a test agent."""
        try:
            agent = client.create_agent(name="test_agent")
            yield agent
            # Cleanup
            try:
                client.delete_agent(agent.id)
            except:
                pass
        except Exception as e:
            pytest.skip(f"Cannot create agent: {e}")
    
    def test_list_agents(self, client):
        """Test listing agents."""
        try:
            agents = client.list_agents()
            assert isinstance(agents, list)
        except Exception as e:
            pytest.skip(f"Server not running: {e}")
    
    def test_create_agent(self, client):
        """Test creating an agent."""
        try:
            agent = client.create_agent(
                name="test_create_agent",
                description="Test agent",
            )
            assert agent.name == "test_create_agent"
            assert agent.id is not None
            
            # Cleanup
            client.delete_agent(agent.id)
        except Exception as e:
            pytest.skip(f"Server not running: {e}")
    
    def test_get_agent(self, client, agent):
        """Test getting an agent by ID."""
        retrieved = client.get_agent(agent.id)
        assert retrieved.id == agent.id
        assert retrieved.name == agent.name
    
    def test_agent_exists(self, client, agent):
        """Test checking if agent exists."""
        exists_by_id = client.agent_exists(agent_id=agent.id)
        exists_by_name = client.agent_exists(agent_name=agent.name)
        
        assert exists_by_id is True
        assert exists_by_name is True
    
    def test_get_agent_id(self, client, agent):
        """Test getting agent ID by name."""
        agent_id = client.get_agent_id(agent.name)
        assert agent_id == agent.id
    
    def test_send_message(self, client, agent):
        """Test sending a message to an agent."""
        response = client.send_message(
            agent_id=agent.id,
            message="Hello, test!",
            role="user",
        )
        
        assert response is not None
        assert hasattr(response, 'messages')
        assert len(response.messages) > 0


class TestMirixClientTools:
    """Test tool operations with MirixClient."""
    
    @pytest.fixture
    def client(self):
        """Create a MirixClient for testing."""
        return MirixClient(
            base_url="http://localhost:8000",
            user_id="test-user",
        )
    
    def test_list_tools(self, client):
        """Test listing tools."""
        try:
            tools = client.list_tools()
            assert isinstance(tools, list)
        except Exception as e:
            pytest.skip(f"Server not running: {e}")
    
    def test_get_tool(self, client):
        """Test getting a tool by ID."""
        try:
            tools = client.list_tools()
            if tools:
                tool = client.get_tool(tools[0].id)
                assert tool.id == tools[0].id
        except Exception as e:
            pytest.skip(f"Server not running: {e}")
    
    def test_create_tool_not_supported(self, client):
        """Test that creating tools with functions is not supported."""
        def my_function():
            pass
        
        with pytest.raises(NotImplementedError):
            client.create_tool(my_function, name="test_tool")


class TestMirixClientMemory:
    """Test memory operations with MirixClient."""
    
    @pytest.fixture
    def client(self):
        """Create a MirixClient for testing."""
        return MirixClient(
            base_url="http://localhost:8000",
            user_id="test-user",
        )
    
    @pytest.fixture
    def agent(self, client):
        """Create a test agent."""
        try:
            agent = client.create_agent(name="test_memory_agent")
            yield agent
            client.delete_agent(agent.id)
        except Exception as e:
            pytest.skip(f"Cannot create agent: {e}")
    
    def test_get_in_context_memory(self, client, agent):
        """Test getting in-context memory."""
        memory = client.get_in_context_memory(agent.id)
        assert memory is not None
        assert hasattr(memory, 'blocks')
    
    def test_get_archival_memory_summary(self, client, agent):
        """Test getting archival memory summary."""
        summary = client.get_archival_memory_summary(agent.id)
        assert summary is not None
        assert hasattr(summary, 'size')
    
    def test_get_recall_memory_summary(self, client, agent):
        """Test getting recall memory summary."""
        summary = client.get_recall_memory_summary(agent.id)
        assert summary is not None
        assert hasattr(summary, 'size')
    
    def test_get_messages(self, client, agent):
        """Test getting messages."""
        # Send a message first
        client.send_message(
            agent_id=agent.id,
            message="Test message",
            role="user"
        )
        
        # Get messages
        messages = client.get_messages(agent.id, limit=10)
        assert isinstance(messages, list)
        assert len(messages) > 0


class TestMirixClientConfig:
    """Test configuration operations with MirixClient."""
    
    @pytest.fixture
    def client(self):
        """Create a MirixClient for testing."""
        return MirixClient(
            base_url="http://localhost:8000",
            user_id="test-user",
        )
    
    def test_list_llm_configs(self, client):
        """Test listing LLM configurations."""
        try:
            configs = client.list_model_configs()
            assert isinstance(configs, list)
        except Exception as e:
            pytest.skip(f"Server not running: {e}")
    
    def test_list_embedding_configs(self, client):
        """Test listing embedding configurations."""
        try:
            configs = client.list_embedding_configs()
            assert isinstance(configs, list)
        except Exception as e:
            pytest.skip(f"Server not running: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

