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

from mirix.schemas.agent import AgentType
from mirix import MirixClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

    client.initialize_meta_agent(
        # config_path="mirix/configs/examples/mirix_gemini.yaml",
        config_path="mirix/configs/examples/mirix_openai.yaml",
        update_agents=False
    )

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
        ],
        chaining=False
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
            ],
            limit=10  # Retrieve up to 10 items per memory type
        )

        print(f"[OK] Retrieved memories successfully")
        if memories.get("memories"):
            for memory_type, items in memories["memories"].items():
                if items:
                    print(f"\n  {memory_type.upper()} memories ({items['total_count']}):")
        else:
            print("  No memories found yet (may need more time to process)")
    except Exception as e:
        print(f"[ERROR] Error retrieving memories: {e}")
        import traceback
        traceback.print_exc()
    print()

    # 5. Example: Retrieve by topic
    print("Step 5: Retrieving by topic...")
    print("-" * 70)
    try:
        topic_memories = client.retrieve_with_topic(
            user_id=user_id,
            topic="design",
            limit=5  # Retrieve up to 5 items per memory type
        )
        print(f"[OK] Topic retrieval completed: {topic_memories.get('success', False)}")
        if topic_memories.get("memories"):
            print(f"  Topics searched: {topic_memories.get('topic')}")
            for memory_type, items in topic_memories["memories"].items():
                if items and items.get('total_count', 0) > 0:
                    print(f"  {memory_type.upper()}: {items['total_count']} total, {len(items.get('items', items.get('recent', [])))} retrieved")
    except Exception as e:
        print(f"[ERROR] Error retrieving by topic: {e}")
        import traceback
        traceback.print_exc()
    print()

    # 6. Example: Search memories - returns flat list of results
    print("Step 6: Searching memories...")
    print("-" * 70)
    try:
        # Example 1: Search all memory types
        results = client.search(
            user_id=user_id,
            query="meeting design",
            memory_type="all",
            limit=5
        )
        print(f"[OK] Search completed: {results.get('success', False)}")
        print(f"  Found {results.get('count', 0)} total results across all memory types")
        
        # Display sample results
        if results.get("results"):
            print(f"  Sample results:")
            for i, item in enumerate(results["results"][:3], 1):
                mem_type = item.get("memory_type", "unknown").upper()
                summary = item.get("summary", item.get("caption", item.get("name", "N/A")))
                print(f"    {i}. [{mem_type}] {summary[:60]}...")
        
        # Example 2: Search only episodic memories
        print("\n  Searching only episodic memories in details field...")
        episodic_results = client.search(
            user_id=user_id,
            query="Sarah",
            memory_type="episodic",
            search_field="details",
            search_method='bm25',
            limit=5
        )
        print(f"  Found {episodic_results.get('count', 0)} episodic results")
        
    except Exception as e:
        print(f"[ERROR] Error searching: {e}")
        import traceback
        traceback.print_exc()
    print()

    print("=" * 70)
    print("Demo completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
