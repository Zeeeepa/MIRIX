"""
Queue Manager - Handles queue initialization and lifecycle management
Separates implementation logic from the public API
"""
import atexit
import logging
from typing import Optional, Any

from mirix.queue import config
from mirix.queue.queue_interface import QueueInterface
from mirix.queue.memory_queue import MemoryQueue
from mirix.queue.worker import QueueWorker
from mirix.queue.message_pb2 import QueueMessage

logger = logging.getLogger(__name__)


class QueueManager:
    """
    Manages queue lifecycle and worker coordination
    Singleton pattern to ensure only one instance per application
    """
    
    def __init__(self):
        """Initialize the queue manager"""
        self._queue: Optional[QueueInterface] = None
        self._worker: Optional[QueueWorker] = None
        self._server: Optional[Any] = None
        self._initialized = False
    
    def initialize(self, server: Optional[Any] = None) -> None:
        """
        Initialize the queue and start the background worker
        Creates appropriate queue type based on configuration
        
        This method is idempotent and thread-safe - calling it multiple times
        will only initialize once. The queue uses a singleton pattern.
        
        Args:
            server: Optional server instance for worker to invoke APIs on
        """
        if self._initialized:
            logger.debug("Queue manager already initialized - skipping duplicate initialization")
            # Allow updating server if provided
            if server:
                logger.info("Updating queue manager with server instance")
                self._server = server
                if self._worker:
                    self._worker.set_server(server)
            return  # Already initialized
        
        logger.info("Initializing queue manager with type: %s, server=%s", config.QUEUE_TYPE, 'provided' if server else 'None')
        
        self._server = server
        
        # Create appropriate queue based on configuration
        self._queue = self._create_queue()
        
        # Create and start the background worker
        self._worker = QueueWorker(self._queue, server=self._server)
        self._worker.start()
        
        # Register cleanup function to stop worker on exit
        atexit.register(self.cleanup)
        
        self._initialized = True
        logger.debug("Queue manager initialized successfully")
    
    def _create_queue(self) -> QueueInterface:
        """
        Factory method to create the appropriate queue implementation
        
        Returns:
            QueueInterface instance (MemoryQueue or KafkaQueue)
            
        Raises:
            ImportError: If required dependencies are not installed
        """
        if config.QUEUE_TYPE == 'kafka':
            # Import Kafka queue (lazy import to avoid unnecessary dependency)
            try:
                from .kafka_queue import KafkaQueue
                return KafkaQueue(
                    bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
                    topic=config.KAFKA_TOPIC,
                    group_id=config.KAFKA_GROUP_ID
                )
            except ImportError as e:
                raise ImportError(
                    f"Kafka queue requested but dependencies not installed: {e}\n"
                    "Install with: pip install queue-sample[kafka]"
                ) from e
        else:
            # Default to in-memory queue (no external dependencies)
            return MemoryQueue()
    
    def save(self, message: QueueMessage) -> None:
        """
        Add a message to the queue
        
        Args:
            message: QueueMessage protobuf message to add to the queue
            
        Raises:
            RuntimeError: If the queue is not initialized
        """
        if self._queue is None:
            logger.error("Attempted to save message to uninitialized queue")
            raise RuntimeError(
                "Queue not initialized. This should not happen - "
                "please report this as a bug."
            )
        
        logger.debug("Saving message to queue: agent_id=%s, user_id=%s", message.agent_id, message.user_id if message.HasField('user_id') else 'None')
        
        # Delegate to the queue implementation
        self._queue.put(message)
    
    def cleanup(self) -> None:
        """
        Cleanup function called when the program exits
        Stops the worker and closes queue connections gracefully
        """
        logger.debug("Cleaning up queue manager")
        
        if self._worker:
            self._worker.stop()
            self._worker = None
        
        self._queue = None
        self._initialized = False
        
        logger.debug("Queue manager cleanup complete")
    
    @property
    def is_initialized(self) -> bool:
        """Check if the manager is initialized"""
        return self._initialized
    
    @property
    def queue_type(self) -> str:
        """Get the current queue type (memory or kafka)"""
        return config.QUEUE_TYPE


# Global singleton instance
_manager = QueueManager()


def get_manager() -> QueueManager:
    """
    Get the global queue manager instance
    
    Returns:
        QueueManager singleton instance
    """
    return _manager

