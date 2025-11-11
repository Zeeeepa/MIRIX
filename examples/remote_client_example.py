"""
Example script demonstrating how to use MirixClient to connect to a cloud-hosted Mirix server.

This example shows:
1. How to connect to a remote Mirix server
2. How to create and interact with agents
3. How to manage tools and memory
4. Error handling for remote connections

Prerequisites:
- A Mirix server running and accessible at the specified URL
- Valid API credentials (API key or user_id)
"""

from mirix import MirixClient, create_client
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.memory import ChatMemory

# ============================================================================
# Example 1: Basic Connection and Agent Creation
# ============================================================================


def example_basic_usage():
    """Basic example of connecting to remote server and creating an agent."""
    print("=" * 80)
    print("Example 1: Basic Remote Client Usage")
    print("=" * 80)
    
    # Create a remote client
    client = MirixClient(
        base_url="http://localhost:8000",  # Change to your server URL
        api_key="your-api-key-here",  # Optional: API key for authentication
        # user_id="user-123",  # Optional: For development
        debug=True,  # Enable debug logging
    )
    
    # Or use the factory function
    # client = create_client(
    #     mode="remote",
    #     base_url="http://localhost:8000",
    #     api_key="your-api-key-here"
    # )
    
    print("\n1. Listing available LLM configs...")
    llm_configs = client.list_model_configs()
    print(f"   Found {len(llm_configs)} LLM configurations")
    
    print("\n2. Creating an agent...")
    agent = client.create_agent(
        name="remote_assistant",
        description="An AI assistant accessed via REST API",
    )
    print(f"   Created agent: {agent.name} (ID: {agent.id})")
    
    print("\n3. Sending a message...")
    response = client.send_message(
        agent_id=agent.id,
        message="Hello! What can you help me with?",
        role="user",
    )
    print(f"   Agent responded with {len(response.messages)} messages")
    for msg in response.messages:
        print(f"   - {msg.role}: {msg.text[:100]}...")
    
    print("\n4. Cleaning up...")
    client.delete_agent(agent.id)
    print(f"   Deleted agent {agent.id}")


# ============================================================================
# Example 2: Managing Multiple Agents
# ============================================================================


def example_multiple_agents():
    """Example showing how to manage multiple agents."""
    print("\n" + "=" * 80)
    print("Example 2: Managing Multiple Agents")
    print("=" * 80)
    
    client = MirixClient(
        base_url="http://localhost:8000",
        user_id="demo-user",
        debug=False,
    )
    
    print("\n1. Creating multiple agents...")
    agents = []
    for i in range(3):
        agent = client.create_agent(
            name=f"agent_{i}",
            description=f"Agent number {i}",
            tags=[f"batch-{i // 2}", "demo"],
        )
        agents.append(agent)
        print(f"   Created: {agent.name}")
    
    print("\n2. Listing all agents...")
    all_agents = client.list_agents()
    print(f"   Total agents: {len(all_agents)}")
    
    print("\n3. Filtering agents by tag...")
    tagged_agents = client.list_agents(tags=["demo"])
    print(f"   Agents with 'demo' tag: {len(tagged_agents)}")
    
    print("\n4. Checking if agent exists...")
    exists = client.agent_exists(agent_name="agent_0")
    print(f"   Agent 'agent_0' exists: {exists}")
    
    print("\n5. Cleaning up...")
    for agent in agents:
        client.delete_agent(agent.id)
    print(f"   Deleted {len(agents)} agents")


# ============================================================================
# Example 3: Working with Memory
# ============================================================================


def example_memory_management():
    """Example showing memory management operations."""
    print("\n" + "=" * 80)
    print("Example 3: Memory Management")
    print("=" * 80)
    
    client = MirixClient(
        base_url="http://localhost:8000",
        user_id="demo-user",
    )
    
    print("\n1. Creating agent with custom memory...")
    memory = ChatMemory(
        human="I am a data scientist working on ML projects",
        persona="I am a helpful AI assistant specialized in Python and ML",
    )
    
    agent = client.create_agent(
        name="memory_agent",
        memory=memory,
    )
    print(f"   Created agent: {agent.name}")
    
    print("\n2. Getting in-context memory...")
    current_memory = client.get_in_context_memory(agent.id)
    print(f"   Memory blocks: {[b.label for b in current_memory.blocks]}")
    
    print("\n3. Getting memory summaries...")
    archival_summary = client.get_archival_memory_summary(agent.id)
    recall_summary = client.get_recall_memory_summary(agent.id)
    print(f"   Archival memory size: {archival_summary.size}")
    print(f"   Recall memory size: {recall_summary.size}")
    
    print("\n4. Sending messages and checking recall...")
    for i in range(3):
        client.send_message(
            agent_id=agent.id,
            message=f"Test message {i}",
            role="user",
        )
    
    messages = client.get_messages(agent.id, limit=10)
    print(f"   Retrieved {len(messages)} messages")
    
    print("\n5. Cleaning up...")
    client.delete_agent(agent.id)


# ============================================================================
# Example 4: Working with Tools
# ============================================================================


def example_tool_management():
    """Example showing tool management operations."""
    print("\n" + "=" * 80)
    print("Example 4: Tool Management")
    print("=" * 80)
    
    client = MirixClient(
        base_url="http://localhost:8000",
        user_id="demo-user",
    )
    
    print("\n1. Listing available tools...")
    tools = client.list_tools()
    print(f"   Found {len(tools)} tools")
    for tool in tools[:5]:  # Show first 5
        print(f"   - {tool.name}: {tool.description[:50]}...")
    
    print("\n2. Getting a specific tool...")
    if tools:
        tool = client.get_tool(tools[0].id)
        print(f"   Tool: {tool.name}")
        print(f"   Type: {tool.source_type}")
    
    print("\n3. Creating agent with specific tools...")
    if tools:
        tool_ids = [tools[0].id]
        agent = client.create_agent(
            name="tool_agent",
            tool_ids=tool_ids,
            include_base_tools=True,
        )
        print(f"   Created agent with {len(agent.tools)} tools")
        
        print("\n4. Getting tools from agent...")
        agent_tools = client.get_tools_from_agent(agent.id)
        print(f"   Agent has {len(agent_tools)} tools")
        
        print("\n5. Cleaning up...")
        client.delete_agent(agent.id)


# ============================================================================
# Example 5: Error Handling
# ============================================================================


def example_error_handling():
    """Example showing error handling for remote connections."""
    print("\n" + "=" * 80)
    print("Example 5: Error Handling")
    print("=" * 80)
    
    client = MirixClient(
        base_url="http://localhost:8000",
        user_id="demo-user",
        timeout=5,  # Short timeout for demo
        max_retries=2,
    )
    
    print("\n1. Handling missing agent...")
    try:
        agent = client.get_agent("non-existent-id")
        print(f"   Found agent: {agent.name}")
    except Exception as e:
        print(f"   ✓ Expected error: {type(e).__name__}")
    
    print("\n2. Handling invalid operations...")
    try:
        # Try to create tool with function (not supported in remote)
        def my_function():
            pass
        
        client.create_tool(my_function, name="test_tool")
    except NotImplementedError as e:
        print(f"   ✓ Expected error: {e}")
    
    print("\n3. Testing connection timeout...")
    try:
        # This will work if server is running
        client.list_agents()
        print("   ✓ Connection successful")
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")


# ============================================================================
# Example 6: Comparison with LocalClient
# ============================================================================


def example_local_vs_remote():
    """Example comparing LocalClient and MirixClient usage."""
    print("\n" + "=" * 80)
    print("Example 6: LocalClient vs MirixClient")
    print("=" * 80)

    print("\n1. Creating LocalClient (embedded server)...")
    from mirix import LocalClient

    local_client = LocalClient()
    print("   ✓ LocalClient created (server runs in-process)")

    print("\n2. Creating MirixClient (cloud server)...")
    remote_client = MirixClient(
        base_url="http://localhost:8000",
        user_id="demo-user",
    )
    print("   ✓ MirixClient created (connects to remote server)")

    print("\n3. Both clients have the same interface!")
    print("   - local_client.create_agent(...)")
    print("   - remote_client.create_agent(...)")
    print("   - Both return AgentState objects")

    print("\n4. Key differences:")
    print("   LocalClient:")
    print("     + No network latency")
    print("     + Can create tools with Python functions")
    print("     + Full access to server internals")
    print("     - Server runs in same process")
    print("   MirixClient:")
    print("     + Server runs separately (scalable)")
    print("     + Multiple clients can connect")
    print("     + Suitable for production deployment")
    print("     - Network latency")
    print("     - Cannot create tools with functions (must be on server)")


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "MIRIX REMOTE CLIENT EXAMPLES" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    
    try:
        # Run examples
        example_basic_usage()
        example_multiple_agents()
        example_memory_management()
        example_tool_management()
        example_error_handling()
        example_local_vs_remote()
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n\n⚠️  Error running examples: {e}")
        print("\nMake sure:")
        print("1. The Mirix server is running at http://localhost:8000")
        print("2. You have the correct API credentials")
        print("3. The server is accessible from your network")
        print("\nTo start the server, run:")
        print("   python -m mirix.server.rest_api")
        print("   # or")
        print("   uvicorn mirix.server.rest_api:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
