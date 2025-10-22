"""
Server Registry - Maps server IDs to server instances for message routing.

This allows multiple clients to share a single queue while ensuring each message
is processed by the correct server.
"""
import logging
from typing import Dict, Optional, TYPE_CHECKING
import threading

if TYPE_CHECKING:
    from mirix.server.server import SyncServer

logger = logging.getLogger(__name__)


class ServerRegistry:
    """
    Thread-safe registry that maps server IDs to SyncServer instances.
    
    This enables message routing in a shared queue architecture where multiple
    clients use the same queue but each message must be processed by a specific server.
    """
    
    def __init__(self):
        """Initialize the server registry with thread-safe storage."""
        self._servers: Dict[str, 'SyncServer'] = {}
        self._lock = threading.RLock()
    
    def register(self, server_id: str, server: 'SyncServer') -> None:
        """
        Register a server instance with a unique ID.
        
        Args:
            server_id: Unique identifier for this server
            server: SyncServer instance to register
            
        Raises:
            ValueError: If server_id is already registered
        """
        with self._lock:
            if server_id in self._servers:
                logger.warning(f"Server {server_id} is already registered, overwriting")
            
            self._servers[server_id] = server
            logger.debug(f"Registered server: {server_id}")
    
    def unregister(self, server_id: str) -> None:
        """
        Unregister a server instance.
        
        Args:
            server_id: ID of the server to unregister
        """
        with self._lock:
            if server_id in self._servers:
                del self._servers[server_id]
                logger.debug(f"Unregistered server: {server_id}")
            else:
                logger.warning(f"Attempted to unregister unknown server: {server_id}")
    
    def get(self, server_id: str) -> Optional['SyncServer']:
        """
        Get a server instance by ID.
        
        Args:
            server_id: ID of the server to retrieve
            
        Returns:
            SyncServer instance if found, None otherwise
        """
        with self._lock:
            return self._servers.get(server_id)
    
    def has_server(self, server_id: str) -> bool:
        """
        Check if a server is registered.
        
        Args:
            server_id: ID to check
            
        Returns:
            True if server is registered, False otherwise
        """
        with self._lock:
            return server_id in self._servers
    
    def get_server_count(self) -> int:
        """
        Get the number of registered servers.
        
        Returns:
            Number of registered servers
        """
        with self._lock:
            return len(self._servers)
    
    def clear(self) -> None:
        """Clear all registered servers."""
        with self._lock:
            self._servers.clear()
            logger.debug("Cleared all servers from registry")


# Global singleton instance
_registry = ServerRegistry()


def get_registry() -> ServerRegistry:
    """
    Get the global server registry instance.
    
    Returns:
        ServerRegistry singleton instance
    """
    return _registry

