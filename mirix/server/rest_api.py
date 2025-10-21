"""
FastAPI REST API server for Mirix.
This provides HTTP endpoints that wrap the SyncServer functionality,
allowing MirixClient instances to communicate with a cloud-hosted server.
"""

import traceback
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Header, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from mirix.log import get_logger
from mirix.schemas.agent import AgentState, AgentType, CreateAgent
from mirix.schemas.block import Block, BlockUpdate, CreateBlock, Human, Persona
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.environment_variables import (
    SandboxEnvironmentVariable,
    SandboxEnvironmentVariableCreate,
    SandboxEnvironmentVariableUpdate,
)
from mirix.schemas.file import FileMetadata
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.memory import ArchivalMemorySummary, Memory, RecallMemorySummary
from mirix.schemas.message import Message, MessageCreate
from mirix.schemas.mirix_response import MirixResponse
from mirix.schemas.organization import Organization
from mirix.schemas.sandbox_config import (
    E2BSandboxConfig,
    LocalSandboxConfig,
    SandboxConfig,
    SandboxConfigCreate,
    SandboxConfigUpdate,
)
from mirix.schemas.tool import Tool, ToolCreate, ToolUpdate
from mirix.schemas.tool_rule import BaseToolRule
from mirix.schemas.user import User
from mirix.server.server import SyncServer
from mirix.utils import convert_message_to_mirix_message

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Mirix API",
    description="REST API for Mirix - Memory-augmented AI Agent System",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize server (single instance shared across all requests)
_server: Optional[SyncServer] = None


def get_server() -> SyncServer:
    """Get or create the singleton SyncServer instance."""
    global _server
    if _server is None:
        _server = SyncServer()
    return _server


# ============================================================================
# Helper Functions
# ============================================================================


def get_user_and_org(
    x_user_id: Optional[str] = None,
    x_org_id: Optional[str] = None,
) -> tuple[str, str]:
    """
    Get user_id and org_id from headers or use defaults.
    
    Returns:
        tuple[str, str]: (user_id, org_id)
    """
    server = get_server()
    
    if x_user_id:
        user_id = x_user_id
        org_id = x_org_id or server.organization_manager.DEFAULT_ORG_ID
    else:
        user_id = server.user_manager.DEFAULT_USER_ID
        org_id = server.organization_manager.DEFAULT_ORG_ID
    
    return user_id, org_id


# ============================================================================
# Error Handling
# ============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for all unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "type": type(exc).__name__,
        },
    )


# ============================================================================
# Health Check
# ============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mirix-api"}


# ============================================================================
# Agent Endpoints
# ============================================================================


@app.get("/agents", response_model=List[AgentState])
async def list_agents(
    query_text: Optional[str] = None,
    tags: Optional[str] = None,  # Comma-separated
    limit: int = 100,
    cursor: Optional[str] = None,
    parent_id: Optional[str] = None,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """List all agents for the authenticated user."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    
    tags_list = tags.split(",") if tags else None
    
    return server.agent_manager.list_agents(
        actor=user,
        tags=tags_list,
        query_text=query_text,
        limit=limit,
        cursor=cursor,
        parent_id=parent_id,
    )


class CreateAgentRequest(BaseModel):
    """Request model for creating an agent."""

    name: Optional[str] = None
    agent_type: Optional[AgentType] = AgentType.chat_agent
    embedding_config: Optional[EmbeddingConfig] = None
    llm_config: Optional[LLMConfig] = None
    memory: Optional[Memory] = None
    block_ids: Optional[List[str]] = None
    system: Optional[str] = None
    tool_ids: Optional[List[str]] = None
    tool_rules: Optional[List[BaseToolRule]] = None
    include_base_tools: Optional[bool] = True
    include_meta_memory_tools: Optional[bool] = False
    metadata: Optional[Dict] = None
    description: Optional[str] = None
    initial_message_sequence: Optional[List[Message]] = None
    tags: Optional[List[str]] = None


@app.post("/agents", response_model=AgentState)
async def create_agent(
    request: CreateAgentRequest,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Create a new agent."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    
    # Create memory blocks if provided
    if request.memory:
        for block in request.memory.get_blocks():
            server.block_manager.create_or_update_block(block, actor=user)
    
    # Prepare block IDs
    block_ids = request.block_ids or []
    if request.memory:
        block_ids.extend([b.id for b in request.memory.get_blocks()])
    
    # Create agent request
    create_params = {
        "description": request.description,
        "metadata_": request.metadata,
        "memory_blocks": [],
        "block_ids": block_ids,
        "tool_ids": request.tool_ids or [],
        "tool_rules": request.tool_rules,
        "include_base_tools": request.include_base_tools,
        "system": request.system,
        "agent_type": request.agent_type,
        "llm_config": request.llm_config,
        "embedding_config": request.embedding_config,
        "initial_message_sequence": request.initial_message_sequence,
        "tags": request.tags,
    }
    
    if request.name:
        create_params["name"] = request.name
    
    agent_state = server.create_agent(CreateAgent(**create_params), actor=user)
    
    return server.agent_manager.get_agent_by_id(agent_state.id, actor=user)


@app.get("/agents/{agent_id}", response_model=AgentState)
async def get_agent(
    agent_id: str,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get an agent by ID."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    return server.agent_manager.get_agent_by_id(agent_id, actor=user)


@app.delete("/agents/{agent_id}")
async def delete_agent(
    agent_id: str,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Delete an agent."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    server.agent_manager.delete_agent(agent_id, actor=user)
    return {"status": "success", "message": f"Agent {agent_id} deleted"}


class UpdateAgentRequest(BaseModel):
    """Request model for updating an agent."""

    name: Optional[str] = None
    description: Optional[str] = None
    system: Optional[str] = None
    tool_ids: Optional[List[str]] = None
    metadata: Optional[Dict] = None
    llm_config: Optional[LLMConfig] = None
    embedding_config: Optional[EmbeddingConfig] = None
    message_ids: Optional[List[str]] = None
    memory: Optional[Memory] = None
    tags: Optional[List[str]] = None


@app.patch("/agents/{agent_id}", response_model=AgentState)
async def update_agent(
    agent_id: str,
    request: UpdateAgentRequest,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Update an agent."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    
    # TODO: Implement update_agent in server
    raise HTTPException(status_code=501, detail="Update agent not yet implemented")


# ============================================================================
# Agent Interaction Endpoints
# ============================================================================


class SendMessageRequest(BaseModel):
    """Request model for sending a message to an agent."""

    message: Union[str, List[Dict]]
    role: str = "user"
    name: Optional[str] = None
    stream_steps: bool = False
    stream_tokens: bool = False
    force_response: bool = False
    existing_file_uris: Optional[List[str]] = None
    extra_messages: Optional[List[Dict]] = None


@app.post("/agents/{agent_id}/messages", response_model=MirixResponse)
async def send_message(
    agent_id: str,
    request: SendMessageRequest,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Send a message to an agent."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    
    # Import here to avoid circular imports
    from mirix.interface import QueuingInterface
    from mirix.schemas.enums import MessageRole
    from mirix.schemas.mirix_message_content import TextContent
    
    # Create interface for this request
    interface = QueuingInterface()
    
    # Prepare input messages
    if isinstance(request.message, str):
        content = [TextContent(text=request.message)]
        input_messages = [
            MessageCreate(role=MessageRole(request.role), content=content, name=request.name)
        ]
    else:
        # Handle complex message types (images, files, etc.)
        # For now, simplified implementation
        raise HTTPException(
            status_code=501,
            detail="Complex message types not yet implemented in REST API",
        )
    
    # Send messages
    usage = server.send_messages(
        actor=user,
        agent_id=agent_id,
        input_messages=input_messages,
        force_response=request.force_response,
    )
    
    # Get messages from interface
    messages = interface.to_list()
    mirix_messages = []
    for m in messages:
        mirix_messages.extend(m.to_mirix_message())
    
    return MirixResponse(messages=mirix_messages, usage=usage)


# ============================================================================
# Memory Endpoints
# ============================================================================


@app.get("/agents/{agent_id}/memory", response_model=Memory)
async def get_agent_memory(
    agent_id: str,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get an agent's in-context memory."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    return server.get_agent_memory(agent_id=agent_id, actor=user)


@app.get("/agents/{agent_id}/memory/archival", response_model=ArchivalMemorySummary)
async def get_archival_memory_summary(
    agent_id: str,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get archival memory summary."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    return server.get_archival_memory_summary(agent_id=agent_id, actor=user)


@app.get("/agents/{agent_id}/memory/recall", response_model=RecallMemorySummary)
async def get_recall_memory_summary(
    agent_id: str,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get recall memory summary."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    return server.get_recall_memory_summary(agent_id=agent_id, actor=user)


@app.get("/agents/{agent_id}/messages", response_model=List[Message])
async def get_agent_messages(
    agent_id: str,
    cursor: Optional[str] = None,
    limit: int = 1000,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get messages from an agent."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    return server.get_agent_recall_cursor(
        user_id=user_id,
        agent_id=agent_id,
        before=cursor,
        limit=limit,
        reverse=True,
    )


# ============================================================================
# Tool Endpoints
# ============================================================================


@app.get("/tools", response_model=List[Tool])
async def list_tools(
    cursor: Optional[str] = None,
    limit: int = 50,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """List all tools."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    return server.tool_manager.list_tools(cursor=cursor, limit=limit, actor=user)


@app.get("/tools/{tool_id}", response_model=Tool)
async def get_tool(
    tool_id: str,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get a tool by ID."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    return server.tool_manager.get_tool_by_id(tool_id, actor=user)


@app.post("/tools", response_model=Tool)
async def create_tool(
    tool: Tool,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Create a new tool."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    return server.tool_manager.create_tool(tool, actor=user)


@app.delete("/tools/{tool_id}")
async def delete_tool(
    tool_id: str,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Delete a tool."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    server.tool_manager.delete_tool_by_id(tool_id, actor=user)
    return {"status": "success", "message": f"Tool {tool_id} deleted"}


# ============================================================================
# Block Endpoints
# ============================================================================


@app.get("/blocks", response_model=List[Block])
async def list_blocks(
    label: Optional[str] = None,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """List all blocks."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    return server.block_manager.get_blocks(actor=user, label=label)


@app.get("/blocks/{block_id}", response_model=Block)
async def get_block(
    block_id: str,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get a block by ID."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    return server.block_manager.get_block_by_id(block_id, actor=user)


@app.post("/blocks", response_model=Block)
async def create_block(
    block: Block,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Create a block."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    return server.block_manager.create_or_update_block(block, actor=user)


@app.delete("/blocks/{block_id}")
async def delete_block(
    block_id: str,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Delete a block."""
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    server.block_manager.delete_block(block_id, actor=user)
    return {"status": "success", "message": f"Block {block_id} deleted"}


# ============================================================================
# Configuration Endpoints
# ============================================================================


@app.get("/config/llm", response_model=List[LLMConfig])
async def list_llm_configs(
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """List available LLM configurations."""
    server = get_server()
    return server.list_llm_models()


@app.get("/config/embedding", response_model=List[EmbeddingConfig])
async def list_embedding_configs(
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """List available embedding configurations."""
    server = get_server()
    return server.list_embedding_models()


# ============================================================================
# Organization Endpoints
# ============================================================================


@app.get("/organizations", response_model=List[Organization])
async def list_organizations(
    cursor: Optional[str] = None,
    limit: int = 50,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """List organizations."""
    server = get_server()
    return server.organization_manager.list_organizations(cursor=cursor, limit=limit)


@app.post("/organizations", response_model=Organization)
async def create_organization(
    name: Optional[str] = None,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Create an organization."""
    server = get_server()
    return server.organization_manager.create_organization(
        pydantic_org=Organization(name=name)
    )


@app.get("/organizations/{org_id}", response_model=Organization)
async def get_organization(
    org_id: str,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get an organization by ID."""
    server = get_server()
    try:
        return server.organization_manager.get_organization_by_id(org_id)
    except Exception:
        # If organization doesn't exist, return default or create it
        return server.get_organization_or_default(org_id)


class CreateOrGetOrganizationRequest(BaseModel):
    """Request model for creating or getting an organization."""
    
    org_id: Optional[str] = None
    name: Optional[str] = None


@app.post("/organizations/create_or_get", response_model=Organization)
async def create_or_get_organization(
    request: CreateOrGetOrganizationRequest,
):
    """
    Create organization if it doesn't exist, or get existing one.
    This endpoint doesn't require authentication as it's used during client initialization.
    
    If org_id is not provided, a random ID will be generated.
    If org_id is provided, it will be used as-is (no prefix constraint).
    """
    server = get_server()
    from mirix.schemas.organization import OrganizationCreate
    
    # Use provided org_id or generate a new one
    if request.org_id:
        org_id = request.org_id
    else:
        # Generate a random org ID
        import uuid
        org_id = f"org-{uuid.uuid4().hex[:8]}"
    
    try:
        # Try to get existing organization
        org = server.organization_manager.get_organization_by_id(org_id)
        if org:
            return org
    except Exception:
        pass
    
    # Create new organization if it doesn't exist
    org_create = OrganizationCreate(
        id=org_id,
        name=request.name or org_id
    )
    org = server.organization_manager.create_organization(
        pydantic_org=Organization(**org_create.model_dump())
    )
    logger.info(f"Created new organization: {org_id}")
    return org


# ============================================================================
# User Endpoints
# ============================================================================


@app.get("/users/{user_id}", response_model=User)
async def get_user(
    user_id: str,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """Get a user by ID."""
    server = get_server()
    return server.user_manager.get_user_by_id(user_id)


class CreateOrGetUserRequest(BaseModel):
    """Request model for creating or getting a user."""
    
    user_id: Optional[str] = None
    name: Optional[str] = None
    org_id: Optional[str] = None


@app.post("/users/create_or_get", response_model=User)
async def create_or_get_user(
    request: CreateOrGetUserRequest,
):
    """
    Create user if it doesn't exist, or get existing one.
    This endpoint doesn't require authentication as it's used during client initialization.
    
    If user_id is not provided, a random ID will be generated.
    If user_id is provided, it will be used as-is (no prefix constraint).
    """
    server = get_server()
    
    # Use provided user_id or generate a new one
    if request.user_id:
        user_id = request.user_id
    else:
        # Generate a random user ID
        import uuid
        user_id = f"user-{uuid.uuid4().hex[:8]}"
    
    org_id = request.org_id
    if not org_id:
        org_id = server.organization_manager.DEFAULT_ORG_ID
    
    try:
        # Try to get existing user
        user = server.user_manager.get_user_by_id(user_id)
        if user:
            return user
    except Exception:
        pass
    
    from mirix.schemas.user import User as PydanticUser
    
    # Create a User object with all required fields
    user = server.user_manager.create_user(
        pydantic_user=PydanticUser(
            id=user_id,
            name=request.name or user_id,
            organization_id=org_id,
            timezone=server.user_manager.DEFAULT_TIME_ZONE,
            status="active"
        )
    )
    logger.info(f"Created new user: {user_id}")
    return user

# ============================================================================
# Memory API Endpoints (New)
# ============================================================================


class InitializeMetaAgentRequest(BaseModel):
    """Request model for initializing a meta agent."""

    config: Dict[str, Any]
    project: Optional[str] = None


@app.post("/agents/meta/initialize", response_model=AgentState)
async def initialize_meta_agent(
    request: InitializeMetaAgentRequest,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """
    Initialize a meta agent with configuration.
    
    This creates a meta memory agent that manages specialized memory agents.
    """
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    
    # Extract config components
    config = request.config
    llm_config = None
    embedding_config = None
    system_prompts = None
    agents_config = None

    # Build create_params by flattening meta_agent_config
    create_params = {
        "llm_config": LLMConfig(**config["llm_config"]),
        "embedding_config": EmbeddingConfig(**config["embedding_config"]),
    }
    
    # Flatten meta_agent_config fields into create_params
    if "meta_agent_config" in config and config["meta_agent_config"]:
        meta_config = config["meta_agent_config"]
        # Add fields from meta_agent_config directly
        if "agents" in meta_config:
            create_params["agents"] = meta_config["agents"]
        if "system_prompts" in meta_config:
            create_params["system_prompts"] = meta_config["system_prompts"]

    # Check if meta agent already exists for this project
    existing_meta_agents = server.agent_manager.list_agents(actor=user, limit=1000)

    assert len(existing_meta_agents) <= 1, "Only one meta agent can be created for a project"

    if len(existing_meta_agents) == 1:
        meta_agent = existing_meta_agents[0]
    else:
        from mirix.schemas.agent import CreateMetaAgent
        meta_agent = server.agent_manager.create_meta_agent(meta_agent_create=CreateMetaAgent(**create_params), actor=user)

    return meta_agent

class AddMemoryRequest(BaseModel):
    """Request model for adding memory."""

    user_id: str
    meta_agent_id: str
    messages: List[Dict[str, Any]]
    verbose: bool = False


@app.post("/memory/add")
async def add_memory(
    request: AddMemoryRequest,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """
    Add conversation turns to memory.
    
    Processes conversation messages and stores them in appropriate memory systems.
    """
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    
    # Get the meta agent by ID
    meta_agent = server.agent_manager.get_agent_by_id(request.meta_agent_id, actor=user)

    message = request.messages

    if isinstance(message, list) and "role" in message[0].keys():
        # This means the input is in the format of [{"role": "user", "content": [{"type": "text", "text": "..."}]}, {"role": "assistant", "content": [{"type": "text", "text": "..."}]}]

        # We need to convert the message to the format in "content"
        new_message = []
        for msg in message:
            new_message.append({'type': "text", "text": "[USER]" if msg["role"] == "user" else "[ASSISTANT]"})
            new_message.extend(msg["content"])
        message = new_message

    input_messages = convert_message_to_mirix_message(message)

    usage = await server.send_messages(
        actor=user,
        agent_id=meta_agent.id,
        input_messages=input_messages,
        # chaining=chaining, TODO: add back later
        verbose=request.verbose,
    )
    
    return {
        "success": True,
        "message": "Memory added successfully",
        "usage": usage.model_dump() if usage else None,
    }


class RetrieveMemoryRequest(BaseModel):
    """Request model for retrieving memory."""

    user_id: str
    messages: List[Dict[str, Any]]


@app.post("/memory/retrieve/conversation")
async def retrieve_memory_with_conversation(
    request: RetrieveMemoryRequest,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """
    Retrieve relevant memories based on conversation context.
    """
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    
    # Extract query from last user message
    last_message = request.messages[-1]
    query_text = ""
    
    if isinstance(last_message["content"], str):
        query_text = last_message["content"]
    else:
        for item in last_message["content"]:
            if item["type"] == "text":
                query_text = item["text"]
                break
    
    # Retrieve memories using various memory managers
    memories = {}
    
    # Get all agents for this user
    all_agents = server.agent_manager.list_agents(actor=user, limit=1000)
    
    # Get episodic memories
    try:
        episodic_manager = server.episodic_memory_manager
        episodic_agents = [
            agent for agent in all_agents 
            if agent.agent_type == AgentType.episodic_memory_agent
        ]
        
        episodic_memories = []
        for episodic_agent in episodic_agents:
            events = episodic_manager.list_episodic_memory(
                agent_state=episodic_agent,
                actor=user,
                limit=10,
            )
            episodic_memories.extend([
                {
                    "timestamp": event.occurred_at.isoformat() if event.occurred_at else None,
                    "summary": event.summary,
                    "details": event.details,
                }
                for event in events
            ])
        
        memories["episodic"] = episodic_memories
    except Exception:
        memories["episodic"] = []
    
    # Get semantic memories
    try:
        semantic_manager = server.semantic_memory_manager
        semantic_agents = [
            agent for agent in all_agents 
            if agent.agent_type == AgentType.semantic_memory_agent
        ]
        
        semantic_memories = []
        for semantic_agent in semantic_agents:
            items = semantic_manager.list_semantic_items(
                agent_state=semantic_agent,
                actor=user,
                limit=10,
            )
            semantic_memories.extend([
                {
                    "name": item.name,
                    "summary": item.summary,
                    "details": item.details,
                }
                for item in items
            ])
        
        memories["semantic"] = semantic_memories
    except Exception:
        memories["semantic"] = []
    
    return {
        "success": True,
        "query": query_text,
        "memories": memories,
    }


@app.get("/memory/retrieve/topic")
async def retrieve_memory_with_topic(
    user_id: str,
    topic: str,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """
    Retrieve relevant memories based on a topic.
    """
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    
    # Search memories by topic
    memories = {}
    
    # Use semantic search across memory types
    # This is a simplified implementation - you may want to add more sophisticated search
    
    return {
        "success": True,
        "topic": topic,
        "memories": memories,
    }


@app.get("/memory/search")
async def search_memory(
    user_id: str,
    query: str,
    limit: int = 10,
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
):
    """
    Search for memories using semantic search.
    """
    server = get_server()
    user_id, org_id = get_user_and_org(x_user_id, x_org_id)
    user = server.user_manager.get_user_by_id(user_id)
    
    # Perform semantic search across all memory types
    results = []
    
    # This is a placeholder - implement actual semantic search logic
    
    return {
        "success": True,
        "query": query,
        "results": results,
        "count": len(results),
    }


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

