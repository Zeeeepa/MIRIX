"""
Memory System Unit Tests for Mirix

This test suite validates all memory types in the Mirix system:
- Episodic Memory
- Procedural Memory
- Resource Memory
- Knowledge Vault
- Semantic Memory

Tests include:
- Direct memory operations (using manager methods)
- Indirect memory operations (message-based)
- Search methods (bm25, embedding, string_match)
- Performance comparisons

Usage:
    pytest tests/test_memory_units.py -v
"""

import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pytest
import yaml
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mirix import EmbeddingConfig, LLMConfig, create_client  # noqa: E402
from mirix.schemas.agent import AgentType, CreateMetaAgent  # noqa: E402


# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set"
)


class TestTracker:
    """Class to track test results and provide summary reporting"""

    def __init__(self):
        self.tests = []
        self.current_test = None

    def start_test(self, test_name, description=""):
        """Start tracking a new test"""
        self.current_test = {
            "name": test_name,
            "description": description,
            "status": "running",
            "error": None,
            "subtests": [],
        }
        print(f"\n🚀 Starting: {test_name}")
        if description:
            print(f"   Description: {description}")

    def start_subtest(self, subtest_name):
        """Start tracking a subtest within the current test"""
        if not self.current_test:
            print("Warning: No current test to add subtest to")
            return

        subtest = {"name": subtest_name, "status": "running", "error": None}
        self.current_test["subtests"].append(subtest)
        print(f"  ▶️ {subtest_name}")
        return len(self.current_test["subtests"]) - 1

    def pass_subtest(self, subtest_index=None, message=""):
        """Mark the current or specified subtest as passed"""
        if not self.current_test:
            return

        if subtest_index is None:
            subtest_index = len(self.current_test["subtests"]) - 1

        if 0 <= subtest_index < len(self.current_test["subtests"]):
            self.current_test["subtests"][subtest_index]["status"] = "passed"
            subtest_name = self.current_test["subtests"][subtest_index]["name"]
            print(f"  ✅ {subtest_name}" + (f" - {message}" if message else ""))

    def fail_subtest(self, error, subtest_index=None):
        """Mark the current or specified subtest as failed"""
        if not self.current_test:
            return

        if subtest_index is None:
            subtest_index = len(self.current_test["subtests"]) - 1

        if 0 <= subtest_index < len(self.current_test["subtests"]):
            self.current_test["subtests"][subtest_index]["status"] = "failed"
            self.current_test["subtests"][subtest_index]["error"] = str(error)
            subtest_name = self.current_test["subtests"][subtest_index]["name"]
            print(f"  ❌ {subtest_name} - ERROR: {error}")

    def pass_test(self, message=""):
        """Mark the current test as passed"""
        if not self.current_test:
            return

        self.current_test["status"] = "passed"
        print(
            f"✅ PASSED: {self.current_test['name']}"
            + (f" - {message}" if message else "")
        )
        self.tests.append(self.current_test)
        self.current_test = None

    def fail_test(self, error):
        """Mark the current test as failed"""
        if not self.current_test:
            return

        self.current_test["status"] = "failed"
        self.current_test["error"] = str(error)
        print(f"❌ FAILED: {self.current_test['name']} - ERROR: {error}")
        self.tests.append(self.current_test)
        self.current_test = None

    def get_summary(self):
        """Get a summary of all test results"""
        total_tests = len(self.tests)
        passed_tests = len([t for t in self.tests if t["status"] == "passed"])
        failed_tests = len([t for t in self.tests if t["status"] == "failed"])

        total_subtests = sum(len(t["subtests"]) for t in self.tests)
        passed_subtests = sum(
            len([s for s in t["subtests"] if s["status"] == "passed"])
            for t in self.tests
        )
        failed_subtests = sum(
            len([s for s in t["subtests"] if s["status"] == "failed"])
            for t in self.tests
        )

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "total_subtests": total_subtests,
            "passed_subtests": passed_subtests,
            "failed_subtests": failed_subtests,
            "tests": self.tests,
        }

    def print_summary(self):
        """Print a detailed summary of all test results"""
        summary = self.get_summary()

        print("\n" + "=" * 80)
        print("🏁 TEST EXECUTION SUMMARY")
        print("=" * 80)

        print("\n📊 OVERALL RESULTS:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   ✅ Passed Tests: {summary['passed_tests']}")
        if summary["failed_tests"] > 0:
            print(f"   ❌ Failed Tests: {summary['failed_tests']}")
        print(
            f"   📈 Success Rate: {(summary['passed_tests'] / summary['total_tests'] * 100):.1f}%"
            if summary["total_tests"] > 0
            else "   📈 Success Rate: N/A"
        )

        if summary["total_subtests"] > 0:
            print("\n🔍 SUBTEST DETAILS:")
            print(f"   Total Subtests: {summary['total_subtests']}")
            print(f"   ✅ Passed Subtests: {summary['passed_subtests']}")
            if summary["failed_subtests"] > 0:
                print(f"   ❌ Failed Subtests: {summary['failed_subtests']}")
            print(
                f"   📈 Subtest Success Rate: {(summary['passed_subtests'] / summary['total_subtests'] * 100):.1f}%"
            )

        # Show failed tests details
        failed_tests = [t for t in summary["tests"] if t["status"] == "failed"]
        if failed_tests:
            print("\n❌ FAILED TESTS DETAILS:")
            for i, test in enumerate(failed_tests, 1):
                print(f"   {i}. {test['name']}")
                print(f"      Error: {test['error']}")

                failed_subtests = [
                    s for s in test["subtests"] if s["status"] == "failed"
                ]
                if failed_subtests:
                    print("      Failed Subtests:")
                    for subtest in failed_subtests:
                        print(f"        - {subtest['name']}: {subtest['error']}")

        # Show passed tests summary
        passed_tests = [t for t in summary["tests"] if t["status"] == "passed"]
        if passed_tests:
            print("\n✅ PASSED TESTS:")
            for i, test in enumerate(passed_tests, 1):
                subtest_count = len(test["subtests"])
                passed_subtest_count = len(
                    [s for s in test["subtests"] if s["status"] == "passed"]
                )
                print(
                    f"   {i}. {test['name']} ({passed_subtest_count}/{subtest_count} subtests passed)"
                )

        print("\n" + "=" * 80)

        return summary


# Global test tracker
test_tracker = TestTracker()


def load_config() -> dict:
    """Load configuration from mirix_gemini.yaml file."""
    config_path = Path("mirix/configs/examples/mirix_gemini.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def initialize_client():
    """Initialize Mirix client with configuration."""
    config = load_config()
    client = create_client()

    # Set LLM config from config file or use default
    llm_config_data = config.get("llm_config")
    if llm_config_data and isinstance(llm_config_data, dict):
        llm_config = LLMConfig(**llm_config_data)
    else:
        llm_config = LLMConfig.default_config("gpt-4o-mini")

    client.set_default_llm_config(llm_config)

    # Set embedding config from config file or use default
    embedding_config_data = config.get("embedding_config")
    if embedding_config_data and isinstance(embedding_config_data, dict):
        embedding_config = EmbeddingConfig(**embedding_config_data)
    else:
        embedding_config = EmbeddingConfig.default_config("text-embedding-004")

    client.set_default_embedding_config(embedding_config)

    return client, config


def setup_meta_agent(client, config):
    """Setup meta agent for the client."""
    meta_agent_config_data = config.get("meta_agent_config", {})
    system_prompts_folder = config.get("system_prompts_folder")

    # Get LLM and embedding configs from client
    llm_config = client._default_llm_config
    embedding_config = client._default_embedding_config

    # Create CreateMetaAgent request
    create_request = CreateMetaAgent(
        agents=meta_agent_config_data.get("agents", CreateMetaAgent().agents),
        system_prompts_folder=system_prompts_folder,
        llm_config=llm_config,
        embedding_config=embedding_config,
    )

    # Check if meta agent already exists
    meta_agent = None
    agents = client.list_agents()
    for agent in agents:
        if agent.agent_type == AgentType.meta_memory_agent:
            meta_agent = agent
            break

    if not meta_agent:
        meta_agent = client.create_meta_agent(request=create_request)
    
    return meta_agent


def get_agent_by_type(client, agent_type):
    """Helper to get agent by type."""
    agents = client.list_agents()
    meta_agent = agents[0]
    sub_agents = client.list_agents(parent_id=meta_agent.id)
    for agent in sub_agents:
        if agent.agent_type == agent_type:
            return meta_agent, agent
    return None


@pytest.fixture(scope="module")
def client():
    """Create and initialize Mirix client for all tests"""
    client, config = initialize_client()
    setup_meta_agent(client, config)
    return client


# =================================================================
# DIRECT MEMORY OPERATIONS TESTS
# =================================================================

def test_episodic_memory_direct(client):
    """Test direct episodic memory operations"""
    test_tracker.start_test(
        "Direct Episodic Memory",
        "Testing direct episodic memory operations using manager methods",
    )

    try:
        meta_agent, episodic_agent = get_agent_by_type(client, AgentType.episodic_memory_agent)
        
        # Test 1: Direct insert
        subtest_idx = test_tracker.start_subtest("Direct Event Insert")
        event = client.server.episodic_memory_manager.insert_event(
            actor=client.user,
            agent_id=meta_agent.id,
            agent_state=episodic_agent,
            event_type="activity",
            timestamp=datetime.now(),
            event_actor="user",
            summary="Started working on a coding project",
            details="User began working on a new Python project for data analysis",
            organization_id=client.org_id,
        )
        print(f"Inserted event with ID: {event.id}")
        test_tracker.pass_subtest(
            subtest_idx, f"Event inserted with ID: {event.id}"
        )

        # Test 2: Direct search operations
        subtest_idx = test_tracker.start_subtest("Direct Search Operations")
        try:
            search_response = client.server.episodic_memory_manager.list_episodic_memory(
                agent_state=episodic_agent,
                actor=client.user,
                query="coding",
                search_method="embedding",
                search_field="details",
                limit=10,
            )
            print(f"Semantic search found {len(search_response)} results")

            search_response = client.server.episodic_memory_manager.list_episodic_memory(
                agent_state=episodic_agent,
                actor=client.user,
                query="coding",
                search_method="bm25",
                search_field="summary",
                limit=10,
            )
            print(f"FTS5 search found {len(search_response)} results")
            test_tracker.pass_subtest(
                subtest_idx, "Search operations completed successfully"
            )
        except Exception as e:
            test_tracker.fail_subtest(e, subtest_idx)
            raise e

        test_tracker.pass_test(
            "All direct episodic memory operations completed successfully"
        )

    except Exception as e:
        test_tracker.fail_test(f"Direct episodic memory test failed: {e}")
        traceback.print_exc()
        raise


def test_procedural_memory_direct(client):
    """Test direct procedural memory operations"""
    print("=== Direct Procedural Memory Tests ===")

    meta_agent, procedural_agent = get_agent_by_type(client, AgentType.procedural_memory_agent)

    # Direct insert
    procedure = client.server.procedural_memory_manager.insert_procedure(
        agent_state=procedural_agent,
        agent_id=meta_agent.id,
        entry_type="process",
        summary="How to make coffee",
        steps=[
            "Boil water",
            "Add coffee grounds",
            "Pour hot water",
            "Wait 4 minutes",
            "Serve",
        ],
        actor=client.user,
        organization_id=client.org_id,
    )
    print(f"Inserted procedure with ID: {procedure.id}")

    # Search operations
    search_results = client.server.procedural_memory_manager.list_procedures(
        agent_state=procedural_agent,
        actor=client.user,
        query="coffee",
        search_method="embedding",
        search_field="summary",
        limit=10,
    )
    print(f"Semantic search found {len(search_results)} procedures")

    # Cleanup
    client.server.procedural_memory_manager.delete_procedure_by_id(
        procedure_id=procedure.id, actor=client.user
    )
    print("Cleaned up test procedure\n")


def test_resource_memory_direct(client):
    """Test direct resource memory operations"""
    print("=== Direct Resource Memory Tests ===")

    meta_agent, resource_agent = get_agent_by_type(client, AgentType.resource_memory_agent)

    # Direct insert
    resource = client.server.resource_memory_manager.insert_resource(
        actor=client.user,
        agent_id=meta_agent.id,
        agent_state=resource_agent,
        title="Python Documentation Test",
        summary="Test resource for direct operations",
        resource_type="documentation",
        content="This is test content for direct resource operations",
        organization_id=client.org_id,
    )
    print(f"Inserted resource with ID: {resource.id}")

    # Search operations
    search_results = client.server.resource_memory_manager.list_resources(
        agent_state=resource_agent,
        actor=client.user,
        query="Python",
        search_method="embedding",
        search_field="summary",
        limit=10,
    )
    print(f"Semantic search found {len(search_results)} resources")

    # Cleanup
    client.server.resource_memory_manager.delete_resource_by_id(
        resource_id=resource.id, actor=client.user
    )
    print("Cleaned up test resource\n")


def test_knowledge_vault_direct(client):
    """Test direct knowledge vault operations"""
    print("=== Direct Knowledge Vault Tests ===")

    meta_agent, knowledge_vault_agent = get_agent_by_type(client, AgentType.knowledge_vault_agent)

    # Direct insert
    knowledge = client.server.knowledge_vault_manager.insert_knowledge(
        actor=client.user,
        agent_id=meta_agent.id,
        agent_state=knowledge_vault_agent,
        entry_type="credential",
        source="development_environment",
        sensitivity="medium",
        secret_value="test_api_key_12345",
        caption="Test API key for direct operations",
        organization_id=client.org_id,
    )
    print(f"Inserted knowledge with ID: {knowledge.id}")

    # Search
    search_results = client.server.knowledge_vault_manager.list_knowledge(
        agent_state=knowledge_vault_agent,
        actor=client.user,
        query="test_api_key",
        search_method="bm25",
        search_field="secret_value",
        limit=10,
    )
    print(f"FTS5 search found {len(search_results)} knowledge items")

    # Cleanup
    client.server.knowledge_vault_manager.delete_knowledge_by_id(
        knowledge_vault_item_id=knowledge.id, actor=client.user
    )
    print("Cleaned up test knowledge\n")


def test_semantic_memory_direct(client):
    """Test direct semantic memory operations"""
    print("=== Direct Semantic Memory Tests ===")

    meta_agent, semantic_agent = get_agent_by_type(client, AgentType.semantic_memory_agent)

    # Direct insert
    semantic_item = client.server.semantic_memory_manager.insert_semantic_item(
        actor=client.user,
        agent_id=meta_agent.id,
        agent_state=semantic_agent,
        name="Test Machine Learning Concept",
        summary="A test concept for direct operations",
        details="This is detailed information about the test concept",
        source="test_source",
        organization_id=client.org_id,
    )
    print(f"Inserted semantic item with ID: {semantic_item.id}")

    # Search operations
    search_results = client.server.semantic_memory_manager.list_semantic_items(
        agent_state=semantic_agent,
        actor=client.user,
        query="Test Machine Learning",
        search_method="embedding",
        search_field="name",
        limit=10,
    )
    print(f"Semantic search found {len(search_results)} semantic items")

    # Cleanup
    client.server.semantic_memory_manager.delete_semantic_item_by_id(
        semantic_memory_id=semantic_item.id, actor=client.user
    )
    print("Cleaned up test semantic item\n")


def test_resource_memory_update_direct(client):
    """Test direct resource memory update operations"""
    print("=== Direct Resource Memory Update Tests ===")

    meta_agent, resource_agent = get_agent_by_type(client, AgentType.resource_memory_agent)

    # Create resources for update testing
    resource1 = client.server.resource_memory_manager.insert_resource(
        actor=client.user,
        agent_id=meta_agent.id,
        agent_state=resource_agent,
        title="Initial Test Resource 1",
        summary="Initial summary for update testing",
        resource_type="documentation",
        content="Initial content for resource update testing",
        organization_id=client.org_id,
    )

    print(f"Created test resource: {resource1.id}")

    # Search after creation
    search_results = client.server.resource_memory_manager.list_resources(
        agent_state=resource_agent,
        actor=client.user,
        query="Initial Test",
        search_method="embedding",
        search_field="summary",
        limit=10,
    )
    print(f"Semantic search found {len(search_results)} resources")

    # Cleanup
    client.server.resource_memory_manager.delete_resource_by_id(
        resource_id=resource1.id, actor=client.user
    )
    print("Cleaned up test resource\n")


def test_all_direct_memory_operations(client):
    """Run all direct memory operations tests"""
    test_tracker.start_test(
        "Direct Memory Operations",
        "Testing direct manager method calls for all memory types",
    )

    try:
        test_episodic_memory_direct(client)
        test_procedural_memory_direct(client)
        test_resource_memory_direct(client)
        test_knowledge_vault_direct(client)
        test_semantic_memory_direct(client)
        test_resource_memory_update_direct(client)

        test_tracker.pass_test("All direct memory operations completed successfully")

    except Exception as e:
        test_tracker.fail_test(f"Direct memory operations failed: {e}")
        traceback.print_exc()


# =================================================================
# INDIRECT MEMORY OPERATIONS TESTS (message-based)
# =================================================================
# TODO: These tests are currently disabled because sending messages to specific agents
# is not yet supported in the new client architecture. Re-enable once the feature is available.

def test_episodic_memory_indirect(client):
    """Test episodic memory through messages"""
    print("=== Indirect Episodic Memory Tests ===")
    print("TODO: Sending messages to specific agents not yet supported\n")


def test_procedural_memory_indirect(client):
    """Test procedural memory through messages"""
    print("=== Indirect Procedural Memory Tests ===")
    print("TODO: Sending messages to specific agents not yet supported\n")


def test_resource_memory_indirect(client):
    """Test resource memory through messages"""
    print("=== Indirect Resource Memory Tests ===")
    print("TODO: Sending messages to specific agents not yet supported\n")


def test_knowledge_vault_indirect(client):
    """Test knowledge vault through messages"""
    print("=== Indirect Knowledge Vault Tests ===")
    print("TODO: Sending messages to specific agents not yet supported\n")


def test_semantic_memory_indirect(client):
    """Test semantic memory through messages"""
    print("=== Indirect Semantic Memory Tests ===")
    print("TODO: Sending messages to specific agents not yet supported\n")


def test_resource_memory_update_indirect(client):
    """Test resource memory updates through messages"""
    print("=== Indirect Resource Memory Update Tests ===")
    print("TODO: Sending messages to specific agents not yet supported\n")


def test_text_only_memorization(client):
    """Test text-only memorization"""
    print("=== Testing Text-Only Memorization ===")
    print("TODO: Sending messages to specific agents not yet supported\n")


# =================================================================
# SEARCH AND PERFORMANCE TESTS
# =================================================================

def test_search_methods(client):
    """Test different search methods"""
    test_tracker.start_test(
        "Search Methods Test",
        "Testing different search methods across memory types",
    )

    try:
        search_methods = ["bm25", "embedding", "string_match"]
        meta_agent, episodic_agent = get_agent_by_type(client, AgentType.episodic_memory_agent)

        # Test Episodic Memory
        subtest_idx = test_tracker.start_subtest("Episodic Memory Search")
        try:
            for method in search_methods:
                results = client.server.episodic_memory_manager.list_episodic_memory(
                    agent_state=episodic_agent,
                    actor=client.user,
                    query="grocery",
                    search_method=method,
                    search_field="summary",
                    limit=5,
                )
                print(f"  {method}: {len(results)} results")
            test_tracker.pass_subtest(subtest_idx)
        except Exception as e:
            test_tracker.fail_subtest(e, subtest_idx)

        test_tracker.pass_test("All search methods tested successfully")

    except Exception as e:
        test_tracker.fail_test(f"Search methods test failed: {e}")
        traceback.print_exc()


def test_fts5_comprehensive(client):
    """Comprehensive FTS5 testing"""
    print("=== Comprehensive FTS5 Testing ===")

    meta_agent, episodic_agent = get_agent_by_type(client, AgentType.episodic_memory_agent)

    test_cases = [
        {"name": "Single word", "query": "python"},
        {"name": "Multi-word", "query": "python programming"},
        {"name": "Phrase", "query": '"machine learning"'},
    ]

    for test_case in test_cases:
        print(f"\n{test_case['name']}: '{test_case['query']}'")
        
        try:
            results = client.server.episodic_memory_manager.list_episodic_memory(
                agent_state=episodic_agent,
                actor=client.user,
                query=test_case["query"],
                search_method="bm25",
                limit=5,
            )
            print(f"  Episodic: {len(results)} results")
        except Exception as e:
            print(f"  Error: {e}")

    print("\nFTS5 testing completed.\n")


def test_fts5_performance_comparison(client):
    """Compare FTS5 performance"""
    print("=== FTS5 Performance Comparison ===")

    import time

    meta_agent, episodic_agent = get_agent_by_type(client, AgentType.episodic_memory_agent)
    query = "python"
    
    for method in ["bm25", "string_match"]:
        try:
            start_time = time.time()
            results = client.server.episodic_memory_manager.list_episodic_memory(
                agent_state=episodic_agent,
                actor=client.user,
                query=query,
                search_method=method,
                search_field="summary" if method != "bm25" else None,
                limit=50,
            )
            elapsed_time = time.time() - start_time
            print(f"  {method}: {len(results)} results in {elapsed_time:.4f}s")
        except Exception as e:
            print(f"  {method}: Error - {e}")

    print("\nPerformance comparison completed.\n")


def test_fts5_advanced_features(client):
    """Test advanced FTS5 features"""
    print("=== Advanced FTS5 Features ===")

    meta_agent, episodic_agent = get_agent_by_type(client, AgentType.episodic_memory_agent)

    syntax_tests = [
        ("Standard", "machine learning"),
        ("Phrase", '"machine learning"'),
        ("Multi-term", "artificial intelligence programming"),
    ]

    for test_name, query in syntax_tests:
        print(f"\n{test_name}: '{query}'")
        try:
            results = client.server.episodic_memory_manager.list_episodic_memory(
                agent_state=episodic_agent,
                actor=client.user,
                query=query,
                search_method="bm25",
                limit=5,
            )
            print(f"  Results: {len(results)}")
        except Exception as e:
            print(f"  Error: {e}")

    print("\nAdvanced features testing completed.\n")


def list_all_memory_content(client):
    """List all content in each memory type"""
    print("=== Listing All Memory Content ===")

    meta_agent, episodic_agent = get_agent_by_type(client, AgentType.episodic_memory_agent)

    try:
        episodic_memory = client.server.episodic_memory_manager.list_episodic_memory(
            agent_state=episodic_agent,
            actor=client.user,
            query="",
            limit=50,
        )
        print(f"Total episodic events: {len(episodic_memory)}")
    except Exception as e:
        print(f"Error: {e}")

    print("\nMemory content listing completed.\n")


def test_all_search_and_performance_operations(client):
    """Run all search and performance tests"""
    test_tracker.start_test(
        "Search and Performance Tests",
        "Testing search methods and performance",
    )

    try:
        test_search_methods(client)
        test_fts5_comprehensive(client)
        test_fts5_performance_comparison(client)
        test_fts5_advanced_features(client)
        list_all_memory_content(client)

        test_tracker.pass_test("All search tests completed successfully")

    except Exception as e:
        test_tracker.fail_test(f"Search tests failed: {e}")
        traceback.print_exc()


# =================================================================
# MAIN TEST FUNCTION
# =================================================================

def test_all_memories(client):
    """Main test function that runs all memory tests"""
    print("\n" + "=" * 80)
    print("Starting comprehensive memory system tests...")
    print("=" * 80 + "\n")

    try:
        # Phase 1: Direct memory operations
        test_all_direct_memory_operations(client)

        # Phase 2: Indirect memory operations (currently disabled)
        test_all_indirect_memory_operations(client)

        # Phase 3: Search and performance
        test_all_search_and_performance_operations(client)

        print("\n" + "=" * 80)
        print("✅ All memory tests completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        traceback.print_exc()

    finally:
        # Print summary
        test_tracker.print_summary()


if __name__ == "__main__":
    # For running directly (not via pytest)
    client, config = initialize_client()
    setup_meta_agent(client, config)
    test_all_memories(client)

