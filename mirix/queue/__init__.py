"""
Mirix Queue - A lightweight queue-based messaging system

This module provides asynchronous message processing for the Mirix library.
It automatically starts a background worker when imported and uses a singleton
pattern to ensure only one queue instance runs per application.

Features:
- Auto-initialization on import (safe to import multiple times)
- In-memory queue (default) or Kafka (via QUEUE_TYPE env var)
- Server routing via server_id for multi-client scenarios
- Thread-safe background worker

Usage:
    >>> from mirix.queue import save, QueueMessage
    >>> msg = QueueMessage()
    >>> msg.agent_id = "agent-123"
    >>> save(msg)  # Message will be processed asynchronously

The queue is automatically initialized when the mirix library is imported.
This happens in mirix/__init__.py to ensure the worker is always available.
"""
import logging

from mirix.queue.manager import get_manager
from mirix.queue.message_pb2 import QueueMessage
from mirix.queue.server_registry import get_registry

logger = logging.getLogger(__name__)

# Version
__version__ = '0.1.0'

# Get the global manager instance (singleton)
_manager = get_manager()

# Initialize queue on import (idempotent - safe to call multiple times)
# This starts the background worker thread for processing messages
logger.debug("Initializing mirix.queue module")
if not _manager.is_initialized:
    _manager.initialize()
    logger.info("Mirix queue worker started successfully")
else:
    logger.debug("Mirix queue already initialized")


def save(message: QueueMessage) -> None:
    """
    Add a message to the queue
    
    The message will be automatically processed by the background worker
    and logged to the configured log file.
    
    Args:
        message: QueueMessage protobuf message to add to the queue
        
    Raises:
        RuntimeError: If the queue is not initialized
        
    Example:
        >>> import mirix.queue as queue
        >>> from mirix.queue.message_pb2 import QueueMessage
        >>> msg = QueueMessage()
        >>> msg.agent_id = "agent-123"
        >>> queue.save(msg)
    """
    _manager.save(message)


def register_server(server) -> None:
    """
    Register a server instance with the queue for message processing
    
    This allows the queue worker to process messages by calling
    server.send_messages() instead of just logging them.
    
    Args:
        server: SyncServer instance to register
        
    Example:
        >>> import mirix.queue as queue
        >>> from mirix.server.server import SyncServer
        >>> server = SyncServer()
        >>> queue.register_server(server)
    """
    _manager.register_server(server)


# Export public API
__all__ = ['save', 'register_server', 'get_registry', 'QueueMessage']


