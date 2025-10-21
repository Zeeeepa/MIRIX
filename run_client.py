#!/usr/bin/env python3
"""
Simple Mirix MirixClient script to connect to a remote server,
set up meta agent, and demonstrate memory operations (add and retrieve).

Prerequisites:
- Start the server first: python scripts/start_server.py --reload
- Or set MIRIX_API_URL environment variable to your server URL
"""

import logging
import os

import yaml

from mirix.schemas.agent import AgentType
from mirix.client import MirixClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load configuration from mirix_openai.yaml file."""
    with open("mirix/configs/examples/mirix_openai.yaml", "r") as f:
        return yaml.safe_load(f)

def add_memory(client, memory_type="meta_memory", content=""):
    """
    Add a memory by sending a message to a specific memory agent.

    Args:
        client: Mirix client instance
        memory_type: Type of memory ("meta_memory", "episodic", "semantic", etc.)
        content: Memory content to add

    Returns:
        Response from the agent
    """
    # Map memory_type to agent names
    memory_type_to_agent = {
        "meta_memory": "meta_memory_agent",
        "episodic": "episodic_memory_agent",
        "semantic": "semantic_memory_agent",
        "procedural": "procedural_memory_agent",
        "resource": "resource_memory_agent",
        "knowledge_vault": "knowledge_vault_agent",
        "core": "core_memory_agent",
    }

    agent_name = memory_type_to_agent.get(memory_type, "meta_memory_agent")

    # Get the target agent
    agents = client.list_agents()
    target_agent = next((a for a in agents if a.name == agent_name), None)

    if not target_agent:
        logger.error(f"Agent {agent_name} not found")
        return None

    logger.info(f"Adding memory to {agent_name}: {content[:100]}...")

    # Send message to appropriate memory agent via client
    response = client.send_message(
        agent_id=target_agent.id, message=content, role="user"
    )

    logger.info("Memory added successfully")
    return response


def retrieve_memory(client, query, memory_type="meta_memory"):
    """
    Retrieve memories by querying a specific memory agent.

    Args:
        client: Mirix client instance
        query: Query string to retrieve relevant memories
        memory_type: Type of memory to query

    Returns:
        Response from the agent
    """
    # Map memory_type to agent names
    memory_type_to_agent = {
        "meta_memory": "meta_memory_agent",
        "episodic": "episodic_memory_agent",
        "semantic": "semantic_memory_agent",
        "procedural": "procedural_memory_agent",
        "resource": "resource_memory_agent",
        "knowledge_vault": "knowledge_vault_agent",
        "core": "core_memory_agent",
    }

    agent_name = memory_type_to_agent.get(memory_type, "meta_memory_agent")

    # Get the target agent
    agents = client.list_agents()
    target_agent = next((a for a in agents if a.name == agent_name), None)

    if not target_agent:
        logger.error(f"Agent {agent_name} not found")
        return None

    logger.info(f"Retrieving memories from {agent_name} with query: {query[:100]}...")

    # Send query to appropriate memory agent via client
    response = client.send_message(
        agent_id=target_agent.id,
        message=f"Retrieve relevant information about: {query}",
        role="user",
    )

    logger.info("Memory retrieval completed")
    return response


def main():
    
    
    # Create MirixClient (connects to server via REST API)
    user_id = 'demo-user'
    org_id = 'demo-org'
    
    client = MirixClient(
        api_key=None, # TODO: add authentication later
        user_id=user_id,
        org_id=org_id,
        debug=True,
    )

    config = load_config()

    client.initialize_meta_agent(config=config)

    result = client.add(
        user_id=user_id,
        messages=[
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "I just had a meeting with Sarah from the design team at 2 PM today. We discussed the new UI mockups and she showed me three different color schemes."
                }]
            },
            {
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": "I've recorded this meeting: You met with Sarah from the design team at 2 PM today, reviewed three color schemes for UI mockups, selected the blue theme, and scheduled a follow-up for next Wednesday."
                }]
            }
        ]
    )
    print(f"[OK] Memory added successfully: {result.get('success', False)}")

    # 4. Example: Retrieve memories using new API
    print("Step 4: Retrieving memories with conversation context...")
    print("-" * 70)
    try:
        memories = client.retrieve_with_conversation(
            user_id=user_id,
            messages=[
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": "What did I discuss in my meeting with Sarah?"
                    }]
                }
            ]
        )
        
        print(f"[OK] Retrieved memories successfully")
        if memories.get("memories"):
            for memory_type, items in memories["memories"].items():
                if items:
                    print(f"\n  {memory_type.upper()} memories ({len(items)}):")
                    for item in items[:3]:  # Show first 3
                        summary = item.get('summary', item.get('name', 'N/A'))
                        print(f"    - {summary[:100]}...")
        else:
            print("  No memories found yet (may need more time to process)")
    except Exception as e:
        print(f"[ERROR] Error retrieving memories: {e}")
        import traceback
        traceback.print_exc()
    print()

    # 5. Example: Search memories
    print("Step 5: Searching memories...")
    print("-" * 70)
    try:
        results = client.search(
            user_id=user_id,
            query="meeting with Sarah design team",
            limit=5
        )
        print(f"[OK] Search completed: {results.get('success', False)}")
        print(f"  Found {results.get('count', 0)} results")
    except Exception as e:
        print(f"[ERROR] Error searching: {e}")
        import traceback
        traceback.print_exc()
    print()

    # 6. Example: Retrieve by topic
    print("Step 6: Retrieving by topic...")
    print("-" * 70)
    try:
        topic_memories = client.retrieve_with_topic(
            user_id=user_id,
            topic="design"
        )
        print(f"[OK] Topic retrieval completed: {topic_memories.get('success', False)}")
    except Exception as e:
        print(f"[ERROR] Error retrieving by topic: {e}")
        import traceback
        traceback.print_exc()
    print()

    print("=" * 70)
    print("Demo completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
