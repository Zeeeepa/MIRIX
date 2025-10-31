"""
In-memory queue implementation using Python's queue.Queue
Thread-safe and suitable for single-process applications
"""
import logging
import queue
from typing import Optional

from mirix.queue.queue_interface import QueueInterface
from mirix.queue.message_pb2 import QueueMessage

logger = logging.getLogger(__name__)

class MemoryQueue(QueueInterface):
    """Thread-safe in-memory queue implementation"""
    
    def __init__(self):
        """Initialize the in-memory queue"""
        self._queue = queue.Queue()
    
    def put(self, message: QueueMessage) -> None:
        """
        Add a message to the in-memory queue
        
        Args:
            message: QueueMessage protobuf message to add
        """
        logger.debug("Adding message to queue: agent_id=%s", message.agent_id)
        self._queue.put(message)
    
    def get(self, timeout: Optional[float] = None) -> QueueMessage:
        """
        Retrieve a message from the queue
        
        Args:
            timeout: Optional timeout in seconds (None = block indefinitely)
            
        Returns:
            QueueMessage protobuf message from the queue
            
        Raises:
            queue.Empty: If no message available within timeout
        """
        message = self._queue.get(timeout=timeout)
        logger.debug("Retrieved message from queue: agent_id=%s", message.agent_id)
        return message
    
    def close(self) -> None:
        """
        Clean up resources
        For in-memory queue, no cleanup is needed
        """
        pass

