"""
MCP Client implementation for Mirix - adapted from Letta's structure
"""

from .base_client import BaseAsyncMCPClient, BaseMCPClient
from .exceptions import MCPConnectionError, MCPNotInitializedError, MCPTimeoutError
from .gmail_client import GmailMCPClient
from .manager import (
    AsyncMCPClientManager,
    MCPClientManager,
    get_async_mcp_client_manager,
    get_mcp_client_manager,
)
from .stdio_client import AsyncStdioMCPClient, StdioMCPClient
from .types import (
    BaseServerConfig,
    GmailServerConfig,
    MCPServerType,
    MCPTool,
    SSEServerConfig,
    StdioServerConfig,
)

__all__ = [
    "MCPTimeoutError",
    "MCPConnectionError",
    "MCPNotInitializedError",
    "MCPTool",
    "BaseServerConfig",
    "StdioServerConfig",
    "SSEServerConfig",
    "GmailServerConfig",
    "MCPServerType",
    "BaseMCPClient",
    "BaseAsyncMCPClient",
    "StdioMCPClient",
    "AsyncStdioMCPClient",
    "GmailMCPClient",
    "MCPClientManager",
    "AsyncMCPClientManager",
    "get_mcp_client_manager",
    "get_async_mcp_client_manager",
]
