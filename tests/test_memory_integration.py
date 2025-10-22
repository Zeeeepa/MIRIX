"""
Memory System Integration Tests for Mirix

Tests memory operations with real server-client interaction via REST API.
Comprehensive coverage of all memory types and operations:

ADD MEMORY:
- Send messages to sub agents (episodic, procedural, resource, knowledge, semantic)
- Send messages to meta agent for automatic memory routing

READ FROM MEMORY:
- Retrieve with conversation context
- Retrieve by topic
- Search in memory with all methods and fields:
  * Episodic: summary, details (bm25, embedding)
  * Procedural: summary, steps (bm25, embedding)
  * Resource: summary (bm25, embedding), content (bm25 only)
  * Knowledge Vault: secret_value, caption (bm25, embedding)
  * Semantic: name, summary, details (bm25, embedding)
  * All memories: cross-memory search

Usage:
    pytest tests/test_memory_integration.py -v -m integration -s
"""

import os
import sys
import time
import subprocess
import requests
from pathlib import Path

import pytest
import yaml
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mirix import EmbeddingConfig, LLMConfig, create_client
from mirix.client import MirixClient
from mirix.schemas.agent import AgentType

# Mark all tests as integration tests
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set"
    )
]


def load_config() -> dict:
    """Load configuration from mirix_gemini.yaml file."""
    config_path = Path("mirix/configs/examples/mirix_gemini.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def server_process():
    """Start the server as a background process."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    process = subprocess.Popen(
        [sys.executable, "scripts/start_server.py", "--port", "8899"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )
    
    # Wait for server to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8899/health", timeout=1)
            if response.status_code == 200:
                print(f"\n✓ Server ready after {i+1} attempts")
                break
        except (requests.ConnectionError, requests.Timeout):
            if i == max_retries - 1:
                process.terminate()
                stdout, stderr = process.communicate(timeout=5)
                pytest.fail(
                    f"Server failed to start.\n"
                    f"STDOUT: {stdout}\n"
                    f"STDERR: {stderr}"
                )
            time.sleep(0.5)
    
    yield process
    
    # Cleanup
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    print("\n✓ Server stopped")


@pytest.fixture(scope="module")
def client(server_process):
    """Create a client connected to the test server."""
    config = load_config()
    
    client = MirixClient(
        base_url="http://localhost:8899",
        user_id="test-user",
        debug=True,
    )
    
    # Store configs for later use when creating agents
    # MirixClient doesn't have set_default_* methods like LocalClient
    llm_config_data = config.get("llm_config")
    if llm_config_data and isinstance(llm_config_data, dict):
        client._llm_config = LLMConfig(**llm_config_data)
    else:
        client._llm_config = LLMConfig.default_config("gpt-4o-mini")
    
    embedding_config_data = config.get("embedding_config")
    if embedding_config_data and isinstance(embedding_config_data, dict):
        client._embedding_config = EmbeddingConfig(**embedding_config_data)
    else:
        client._embedding_config = EmbeddingConfig.default_config("text-embedding-004")
    
    return client


@pytest.fixture(scope="module")
def meta_agent(client):
    """Get or initialize meta agent for tests."""
    # Use initialize_meta_agent which handles checking if agents already exist
    # This will return existing agent if found, or create new one if needed
    meta_agent = client.initialize_meta_agent(
        config_path="mirix/configs/examples/mirix_gemini.yaml",
        update_agents=False  # Don't update if already exists
    )
    
    return meta_agent


def get_sub_agent(client, meta_agent, agent_type):
    """Helper to get sub agent by type."""
    sub_agents = client.list_agents(parent_id=meta_agent.id)
    
    for agent in sub_agents:
        if agent.agent_type == agent_type:
            return agent
    
    raise ValueError(f"Sub agent of type {agent_type} not found")


# =================================================================
# MESSAGE-BASED MEMORY OPERATIONS (Sub Agents)
# =================================================================

class TestMessageToSubAgents:
    """Test adding memories by sending messages to sub agents."""
    
    def test_message_to_episodic_agent(self, client, meta_agent):
        """Test sending message to episodic memory agent."""
        episodic_agent = get_sub_agent(client, meta_agent, AgentType.episodic_memory_agent)
        
        response = client.send_message(
            agent_id=episodic_agent.id,
            message="Remember: I attended a product launch event today. It was very successful with over 200 attendees.",
            role="user"
        )
        
        assert response is not None
        assert hasattr(response, 'messages')
        print(f"✅ Message sent to episodic agent, got {len(response.messages)} messages back")
    
    def test_message_to_procedural_agent(self, client, meta_agent):
        """Test sending message to procedural memory agent."""
        procedural_agent = get_sub_agent(client, meta_agent, AgentType.procedural_memory_agent)
        
        response = client.send_message(
            agent_id=procedural_agent.id,
            message="Learn this process: To backup database - 1) Stop services 2) Export data 3) Compress files 4) Upload to cloud 5) Verify integrity",
            role="user"
        )
        
        assert response is not None
        assert hasattr(response, 'messages')
        print(f"✅ Message sent to procedural agent")
    
    def test_message_to_resource_agent(self, client, meta_agent):
        """Test sending message to resource memory agent."""
        resource_agent = get_sub_agent(client, meta_agent, AgentType.resource_memory_agent)
        
        response = client.send_message(
            agent_id=resource_agent.id,
            message="Store this resource: Docker Compose Tutorial - A comprehensive guide on using docker-compose for multi-container applications",
            role="user"
        )
        
        assert response is not None
        assert hasattr(response, 'messages')
        print(f"✅ Message sent to resource agent")
    
    def test_message_to_knowledge_vault_agent(self, client, meta_agent):
        """Test sending message to knowledge vault agent."""
        knowledge_vault_agent = get_sub_agent(client, meta_agent, AgentType.knowledge_vault_agent)
        
        response = client.send_message(
            agent_id=knowledge_vault_agent.id,
            message="Save this credential: AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE (for staging environment)",
            role="user"
        )
        
        assert response is not None
        assert hasattr(response, 'messages')
        print(f"✅ Message sent to knowledge vault agent")
    
    def test_message_to_semantic_agent(self, client, meta_agent):
        """Test sending message to semantic memory agent."""
        semantic_agent = get_sub_agent(client, meta_agent, AgentType.semantic_memory_agent)
        
        response = client.send_message(
            agent_id=semantic_agent.id,
            message="Remember this concept: REST API follows principles of statelessness, client-server architecture, and uniform interface",
            role="user"
        )
        
        assert response is not None
        assert hasattr(response, 'messages')
        print(f"✅ Message sent to semantic agent")


# =================================================================
# MESSAGE-BASED MEMORY OPERATIONS (Meta Agent)
# =================================================================

class TestMessageToMetaAgent:
    """Test adding memories by sending messages to meta agent."""
    
    def test_meta_agent_creates_episodic_memory(self, client, meta_agent):
        """Test meta agent creating episodic memory from conversation."""
        response = client.send_message(
            agent_id=meta_agent.id,
            message="Today I had a great brainstorming session with the team about the new feature. We came up with some innovative ideas.",
            role="user"
        )
        
        assert response is not None
        assert hasattr(response, 'messages')
        print(f"✅ Meta agent processed message for episodic memory")
    
    def test_meta_agent_creates_procedural_memory(self, client, meta_agent):
        """Test meta agent creating procedural memory from conversation."""
        response = client.send_message(
            agent_id=meta_agent.id,
            message="Can you remember this workflow? To review code: first check syntax, then verify logic, test edge cases, review documentation, and finally approve or request changes.",
            role="user"
        )
        
        assert response is not None
        print(f"✅ Meta agent processed workflow")
    
    def test_meta_agent_creates_resource_memory(self, client, meta_agent):
        """Test meta agent creating resource memory from conversation."""
        response = client.send_message(
            agent_id=meta_agent.id,
            message="I found this useful article about microservices architecture patterns. It covers service discovery, API gateway, and circuit breakers.",
            role="user"
        )
        
        assert response is not None
        print(f"✅ Meta agent processed resource information")


# =================================================================
# RETRIEVE FROM MEMORY TESTS
# =================================================================

class TestRetrieveWithConversation:
    """Test retrieving memories through conversation."""
    
    def test_retrieve_episodic_via_conversation(self, client, meta_agent):
        """Test retrieving episodic memories through conversation."""
        # First add a memory
        client.send_message(
            agent_id=meta_agent.id,
            message="Remember: I completed the database migration project yesterday.",
            role="user"
        )
        
        # Wait a bit for processing
        time.sleep(1)
        
        # Try to retrieve
        response = client.send_message(
            agent_id=meta_agent.id,
            message="What did I work on yesterday?",
            role="user"
        )
        
        assert response is not None
        print(f"✅ Retrieved memory via conversation")
    
    def test_retrieve_procedural_via_conversation(self, client, meta_agent):
        """Test retrieving procedural memory through conversation."""
        # First add a procedure
        client.send_message(
            agent_id=meta_agent.id,
            message="Remember this: To restart the server, first drain connections, stop the service, clear cache, and start the service.",
            role="user"
        )
        
        time.sleep(1)
        
        # Try to retrieve
        response = client.send_message(
            agent_id=meta_agent.id,
            message="How do I restart the server?",
            role="user"
        )
        
        assert response is not None
        print(f"✅ Retrieved procedure via conversation")


class TestRetrieveWithTopic:
    """Test retrieving memories by topic."""
    
    def test_retrieve_by_topic_deployment(self, client, meta_agent):
        """Test retrieving memories related to deployment topic."""
        response = client.send_message(
            agent_id=meta_agent.id,
            message="Tell me everything you know about deployment processes.",
            role="user"
        )
        
        assert response is not None
        print(f"✅ Retrieved memories by topic: deployment")
    
    def test_retrieve_by_topic_meetings(self, client, meta_agent):
        """Test retrieving memories related to meetings topic."""
        response = client.send_message(
            agent_id=meta_agent.id,
            message="What meetings have I attended recently?",
            role="user"
        )
        
        assert response is not None
        print(f"✅ Retrieved memories by topic: meetings")


# =================================================================
# SEARCH IN MEMORY TESTS (All Methods & All Types)
# =================================================================

class TestSearchInMemory:
    """Test search functionality across all memory types and methods via REST API."""
    
    def test_episodic_search_methods_and_fields(self, client, meta_agent):
        """Test all search methods and fields on episodic memory via API."""
        print("\n🔍 Testing episodic memory search...")
        
        base_url = client.base_url
        user_id = client.user_id
        
        # Test different fields: summary, details
        test_cases = [
            ("meeting", "summary", "bm25"),
            ("meeting", "summary", "embedding"),
            ("team", "details", "bm25"),
            ("team", "details", "embedding"),
        ]
        
        for query, field, method in test_cases:
            response = requests.get(
                f"{base_url}/memory/search",
                params={
                    "user_id": user_id,
                    "query": query,
                    "memory_type": "episodic",
                    "search_field": field,
                    "search_method": method,
                    "limit": 10,
                }
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["results"]) > 0, f"{method} on '{field}' should find results for '{query}'"
            print(f"  [OK] Episodic {method} on '{field}': {len(data['results'])} results")
    
    def test_procedural_search_methods_and_fields(self, client, meta_agent):
        """Test all search methods and fields on procedural memory via API."""
        print("\n🔍 Testing procedural memory search...")
        
        base_url = client.base_url
        user_id = client.user_id
        
        # Test different fields: summary, steps
        test_cases = [
            ("deploy", "summary", "bm25"),
            ("deploy", "summary", "embedding"),
            ("production", "steps", "bm25"),
            ("production", "steps", "embedding"),
        ]
        
        for query, field, method in test_cases:
            response = requests.get(
                f"{base_url}/memory/search",
                params={
                    "user_id": user_id,
                    "query": query,
                    "memory_type": "procedural",
                    "search_field": field,
                    "search_method": method,
                    "limit": 10,
                }
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["results"]) > 0, f"{method} on '{field}' should find results for '{query}'"
            print(f"  [OK] Procedural {method} on '{field}': {len(data['results'])} results")
    
    def test_resource_search_methods_and_fields(self, client, meta_agent):
        """Test all search methods and fields on resource memory via API."""
        print("\n🔍 Testing resource memory search...")
        
        base_url = client.base_url
        user_id = client.user_id
        
        # Test different fields: summary (embedding + bm25), content (bm25 only)
        test_cases = [
            ("Python", "summary", "bm25"),
            ("Python", "summary", "embedding"),
            ("Docker", "content", "bm25"),  # embedding NOT supported for content
        ]
        
        for query, field, method in test_cases:
            response = requests.get(
                f"{base_url}/memory/search",
                params={
                    "user_id": user_id,
                    "query": query,
                    "memory_type": "resource",
                    "search_field": field,
                    "search_method": method,
                    "limit": 10,
                }
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["results"]) > 0, f"{method} on '{field}' should find results for '{query}'"
            print(f"  [OK] Resource {method} on '{field}': {len(data['results'])} results")
    
    def test_knowledge_vault_search_methods_and_fields(self, client, meta_agent):
        """Test all search methods and fields on knowledge vault via API."""
        print("\n🔍 Testing knowledge vault search...")
        
        base_url = client.base_url
        user_id = client.user_id
        
        # Test different fields: secret_value, caption
        test_cases = [
            ("AWS", "secret_value", "bm25"),
            ("credential", "caption", "bm25"),
            ("credential", "caption", "embedding"),
        ]
        
        for query, field, method in test_cases:
            response = requests.get(
                f"{base_url}/memory/search",
                params={
                    "user_id": user_id,
                    "query": query,
                    "memory_type": "knowledge_vault",
                    "search_field": field,
                    "search_method": method,
                    "limit": 10,
                }
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["results"]) > 0, f"{method} on '{field}' should find results for '{query}'"
            print(f"  [OK] Knowledge Vault {method} on '{field}': {len(data['results'])} results")
    
    def test_semantic_search_methods_and_fields(self, client, meta_agent):
        """Test all search methods and fields on semantic memory via API."""
        print("\n🔍 Testing semantic memory search...")
        
        base_url = client.base_url
        user_id = client.user_id
        
        # Test different fields: name, summary, details
        test_cases = [
            ("REST API", "name", "bm25"),
            ("REST API", "name", "embedding"),
            ("stateless", "summary", "bm25"),
            ("stateless", "summary", "embedding"),
            ("architecture", "details", "bm25"),
            ("architecture", "details", "embedding"),
        ]
        
        for query, field, method in test_cases:
            response = requests.get(
                f"{base_url}/memory/search",
                params={
                    "user_id": user_id,
                    "query": query,
                    "memory_type": "semantic",
                    "search_field": field,
                    "search_method": method,
                    "limit": 10,
                }
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["results"]) > 0, f"{method} on '{field}' should find results for '{query}'"
            print(f"  [OK] Semantic {method} on '{field}': {len(data['results'])} results")
    
    def test_search_all_memory_types(self, client, meta_agent):
        """Test search with memory_type='all' across all memories."""
        print("\n🔍 Testing search across all memory types...")
        
        base_url = client.base_url
        user_id = client.user_id
        
        # Search all memory types at once
        response = requests.get(
            f"{base_url}/memory/search",
            params={
                "user_id": user_id,
                "query": "server",
                "memory_type": "all",
                "search_field": "null",  # For "all", field should be "null"
                "search_method": "bm25",
                "limit": 10,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["results"]) > 0, "Should find results across all memory types"
        print(f"  [OK] Search all memory types: {len(data['results'])} total results")
        
        # Verify results from multiple memory types
        memory_types_found = set(result["memory_type"] for result in data["results"])
        print(f"  [OK] Found results from {len(memory_types_found)} memory type(s): {memory_types_found}")


# =================================================================
# COMPREHENSIVE INTEGRATION TESTS
# =================================================================

class TestComprehensiveMemoryFlow:
    """Test complete memory workflow from creation to retrieval."""
    
    def test_full_memory_lifecycle(self, client, meta_agent):
        """Test complete lifecycle: add -> search -> retrieve -> update."""
        print("\n🔄 Testing full memory lifecycle...")
        
        # 1. Add memory via meta agent
        add_response = client.send_message(
            agent_id=meta_agent.id,
            message="I just completed a security audit of our authentication system. Found and fixed 3 vulnerabilities.",
            role="user"
        )
        assert add_response is not None
        print("  ✅ Step 1: Memory added")
        
        time.sleep(2)  # Wait for processing
        
        # 2. Retrieve via conversation
        retrieve_response = client.send_message(
            agent_id=meta_agent.id,
            message="What security work did I do recently?",
            role="user"
        )
        assert retrieve_response is not None
        print("  ✅ Step 2: Memory retrieved")
        
        # 3. Search by topic
        search_response = client.send_message(
            agent_id=meta_agent.id,
            message="Tell me about security-related activities.",
            role="user"
        )
        assert search_response is not None
        print("  ✅ Step 3: Memory searched")
        
        print("✅ Full lifecycle completed")
    
    def test_cross_memory_type_retrieval(self, client, meta_agent):
        """Test retrieving information that spans multiple memory types."""
        print("\n🔗 Testing cross-memory type retrieval...")
        
        # Add different types of memories
        memories = [
            "I learned about Docker containerization today.",  # Could be episodic + semantic
            "To deploy with Docker: build image, push to registry, pull on server, run container.",  # Procedural
            "Here's a useful Docker reference guide I found: https://docs.docker.com",  # Resource
        ]
        
        for memory in memories:
            client.send_message(
                agent_id=meta_agent.id,
                message=memory,
                role="user"
            )
            time.sleep(0.5)
        
        # Now try to retrieve all Docker-related information
        response = client.send_message(
            agent_id=meta_agent.id,
            message="Tell me everything you know about Docker.",
            role="user"
        )
        
        assert response is not None
        print("✅ Cross-memory type retrieval completed")


# =================================================================
# PERFORMANCE AND STRESS TESTS
# =================================================================

class TestMemoryPerformance:
    """Test performance and scalability of memory operations."""
    
    def test_concurrent_memory_additions(self, client, meta_agent):
        """Test adding multiple memories in succession."""
        print("\n⚡ Testing concurrent memory additions...")
        
        messages = [
            "Event 1: Morning standup meeting",
            "Event 2: Code review session",
            "Event 3: Lunch with team",
            "Event 4: Afternoon debugging",
            "Event 5: Evening documentation update",
        ]
        
        for i, message in enumerate(messages, 1):
            response = client.send_message(
                agent_id=meta_agent.id,
                message=message,
                role="user"
            )
            assert response is not None
            print(f"  ✅ Added memory {i}/5")
        
        print("✅ Concurrent additions completed")
    
    def test_large_scale_search(self, client, meta_agent):
        """Test searching across potentially large memory store."""
        print("\n🔎 Testing large-scale search...")
        
        # Perform multiple searches
        queries = [
            "meeting",
            "deployment",
            "documentation",
            "API",
            "security",
        ]
        
        for query in queries:
            response = client.send_message(
                agent_id=meta_agent.id,
                message=f"Search for: {query}",
                role="user"
            )
            assert response is not None
            print(f"  ✅ Searched: {query}")
        
        print("✅ Large-scale search completed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "integration"])

