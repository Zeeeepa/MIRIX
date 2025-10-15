#!/usr/bin/env python3
"""
Simple Mirix client script to initialize client, set up meta agent,
and demonstrate memory operations (add and retrieve).
"""

import logging
import os
from pathlib import Path

import yaml
from mirix.schemas.agent import CreateMetaAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load configuration from mirix_gemini.yaml file."""
    with open("mirix/configs/examples/mirix_gemini.yaml", "r") as f:
        return yaml.safe_load(f)

def initialize_client():
    """Initialize Mirix client with configuration."""
    print("DEBUG: About to import mirix modules...")
    from mirix import create_client, LLMConfig, EmbeddingConfig
    print("DEBUG: Imports complete")
    
    # Load configuration
    print("DEBUG: Loading config...")
    config = load_config()
    print("DEBUG: Config loaded")
    
    # Initialize Mirix client
    print("DEBUG: About to create client...")
    client = create_client()
    print("DEBUG: Client created")
    logger.info("Mirix client initialized")
    
    # Set LLM config from config file or use default
    llm_config_data = config.get("llm_config")
    if llm_config_data and isinstance(llm_config_data, dict):
        llm_config = LLMConfig(**llm_config_data)
        logger.info(f"Using LLM config from file: {llm_config.model}")
    else:
        llm_config = LLMConfig.default_config("gpt-4o-mini")
        logger.info("Using default LLM config: gpt-4o-mini")
    
    client.set_default_llm_config(llm_config)

    # Set embedding config from config file or use default
    embedding_config_data = config.get("embedding_config")
    if embedding_config_data and isinstance(embedding_config_data, dict):
        embedding_config = EmbeddingConfig(**embedding_config_data)
        logger.info(f"Using embedding config from file: {embedding_config.embedding_model}")
    else:
        embedding_config = EmbeddingConfig.default_config("text-embedding-004")
        logger.info("Using default embedding config: text-embedding-004")
    
    client.set_default_embedding_config(embedding_config)
    
    return client, config

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
        agent_id=target_agent.id,
        message=content,
        role="user"
    )
    
    logger.info(f"Memory added successfully")
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
        role="user"
    )
    
    logger.info(f"Memory retrieval completed")
    return response


def main():
    """Main function to demonstrate memory operations."""
    print("\n" + "="*70)
    print("Mirix Memory Client Demo")
    print("="*70 + "\n")
    
    # 1. Initialize client
    print("Step 1: Initializing client...")
    client, config = initialize_client()
    print("✓ Client initialized\n")
    
    # 2. Setup meta agent
    print("Step 2: Setting up MetaAgent...")
    
    # Get configuration from config file
    meta_agent_config_data = config.get("meta_agent_config", {})
    system_prompts_folder = config.get("system_prompts_folder")
    
    # Get LLM and embedding configs from client
    llm_config = client._default_llm_config
    embedding_config = client._default_embedding_config
    
    # Create CreateMetaAgent request (will auto-load system prompts from folder)
    create_request = CreateMetaAgent(
        agents=meta_agent_config_data.get("agents", CreateMetaAgent().agents),
        system_prompts_folder=system_prompts_folder,
        llm_config=llm_config,
        embedding_config=embedding_config,
    )
    
    # Initialize MetaAgent using client method
    meta_agent = client.create_meta_agent(request=create_request)
    print("✓ MetaAgent set up\n")

    # 3. Example: Add memories
    print("Step 3: Adding memories...")
    print("-" * 70)
    response = client.send_message(
        role='user',
        agent_id=meta_agent.id,
        message='[User] I need to remember that my project deadline is November 15th, 2025. The client prefers communication via email and their budget is $50,000.\n\n[Assistant] I\'ve noted the key information: Project deadline is November 15th, 2025, client prefers email communication, and the budget is $50,000. I\'ll help you keep track of these important details.'
    )

    # # 4. Example: Retrieve memories
    # print("Step 4: Retrieving memories...")
    # print("-" * 70)
    
    # # Retrieve using meta_memory_agent (queries across all memory types)
    # query = "What do you know about Python programming?"
    # print(f"Query: {query}")
    # retrieve_response = retrieve_memory(client, query=query, memory_type="meta_memory")
    # if retrieve_response and retrieve_response.messages:
    #     print(f"Retrieved: {retrieve_response.messages[-1].content[:500]}...")
    # print()
    
    # query2 = "What meetings did I have recently?"
    # print(f"Query: {query2}")
    # retrieve_response = retrieve_memory(client, query=query2, memory_type="episodic")
    # if retrieve_response and retrieve_response.messages:
    #     print(f"Retrieved: {retrieve_response.messages[-1].content[:500]}...")
    # print()
    
    # print("✓ Memory retrieval completed\n")
    
    # print("="*70)
    # print("Demo completed successfully!")
    # print("="*70 + "\n")


if __name__ == "__main__":
    main()
