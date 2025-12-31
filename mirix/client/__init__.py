"""
Mirix Client Module

This module provides client implementations for interacting with Mirix agents:
- AbstractClient: Base interface for all clients
- MirixClient: For cloud/remote deployments (server accessed via REST API)

For embedded deployments, use the Mirix SDK directly.
"""

from mirix.client.remote_client import MirixClient
from mirix.client.remote_client_terminal import MirixTerminalClient

__all__ = ["MirixClient", "MirixTerminalClient"]
