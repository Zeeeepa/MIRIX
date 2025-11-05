"""
Background worker that consumes messages from the queue
Runs in a daemon thread and processes messages through the server
"""
import logging
import threading
from typing import TYPE_CHECKING, Optional, Any
from datetime import datetime

from mirix.queue.message_pb2 import QueueMessage

if TYPE_CHECKING:
    from .queue_interface import QueueInterface
    from mirix.schemas.user import User
    from mirix.schemas.message import MessageCreate

logger = logging.getLogger(__name__)

class QueueWorker:
    """Background worker that processes messages from the queue"""
    
    def __init__(self, queue: 'QueueInterface', server: Optional[Any] = None):
        """
        Initialize the queue worker
        
        Args:
            queue: Queue implementation to consume from
            server: Optional server instance to invoke APIs on
        """
        logger.debug("Initializing queue worker with server=%s", 'provided' if server else 'None')
        
        self.queue = queue
        self._server = server
        self._running = False
        self._thread = None
        self._lock = threading.RLock()
    
    def _convert_proto_user_to_pydantic(self, proto_user) -> 'User':
        """
        Convert protobuf User to Pydantic User
        
        Args:
            proto_user: Protobuf User message
            
        Returns:
            Pydantic User object
        """
        # Lazy import to avoid circular dependency
        from mirix.schemas.user import User
        
        return User(
            id=proto_user.id,
            organization_id=proto_user.organization_id if proto_user.organization_id else None,
            name=proto_user.name,
            status=proto_user.status,
            timezone=proto_user.timezone,
            created_at=proto_user.created_at.ToDatetime() if proto_user.HasField('created_at') else datetime.now(),
            updated_at=proto_user.updated_at.ToDatetime() if proto_user.HasField('updated_at') else datetime.now(),
            is_deleted=proto_user.is_deleted
        )
    
    def _convert_proto_message_to_pydantic(self, proto_msg) -> 'MessageCreate':
        """
        Convert protobuf MessageCreate to Pydantic MessageCreate
        
        Args:
            proto_msg: Protobuf MessageCreate message
            
        Returns:
            Pydantic MessageCreate object
        """
        # Lazy import to avoid circular dependency
        from mirix.schemas.message import MessageCreate
        from mirix.schemas.enums import MessageRole
        
        # Map role
        if proto_msg.role == proto_msg.ROLE_USER:
            role = MessageRole.user
        elif proto_msg.role == proto_msg.ROLE_SYSTEM:
            role = MessageRole.system
        else:
            role = MessageRole.user  # Default
        
        # Get content (currently only supporting text_content)
        content = proto_msg.text_content if proto_msg.HasField('text_content') else ""
        
        return MessageCreate(
            role=role,
            content=content,
            name=proto_msg.name if proto_msg.HasField('name') else None,
            otid=proto_msg.otid if proto_msg.HasField('otid') else None,
            sender_id=proto_msg.sender_id if proto_msg.HasField('sender_id') else None,
            group_id=proto_msg.group_id if proto_msg.HasField('group_id') else None
        )
    
    def set_server(self, server: Any) -> None:
        """
        Set or update the server instance.
        
        Args:
            server: Server instance to invoke APIs on
        """
        with self._lock:
            self._server = server
            logger.info("Updated worker server instance")
    
    def _process_message(self, message: QueueMessage) -> None:
        """
        Process a queue message by calling server.send_messages()
        
        Args:
            message: QueueMessage protobuf to process
        """
        # Check if server is available
        with self._lock:
            server = self._server
        
        if server is None:
            log_msg = (
                f"No server available - skipping message: "
                f"agent_id={message.agent_id}, "
                f"input_messages_count={len(message.input_messages)}"
            )
            logger.warning(log_msg)
            return
        
        try:
            # Convert protobuf to Pydantic objects
            actor = self._convert_proto_user_to_pydantic(message.actor)
            input_messages = [
                self._convert_proto_message_to_pydantic(msg) 
                for msg in message.input_messages
            ]
            
            # Extract optional parameters
            chaining = message.chaining if message.HasField('chaining') else True
            
            # Extract filter_tags from protobuf Struct
            filter_tags = None
            if message.HasField('filter_tags') and message.filter_tags:
                filter_tags = dict(message.filter_tags)
            
            # Extract use_cache
            use_cache = message.use_cache if message.HasField('use_cache') else True
            
            # Log the processing
            log_msg = (
                f"Processing message via server: "
                f"agent_id={message.agent_id}, "
                f"input_messages_count={len(input_messages)}, "
                f"use_cache={use_cache}, "
                f"filter_tags={filter_tags}"
            )
            logger.info(log_msg)
            
            # Call server.send_messages()
            usage = server.send_messages(
                actor=actor,
                agent_id=message.agent_id,
                input_messages=input_messages,
                chaining=chaining,
                filter_tags=filter_tags,
                use_cache=use_cache,
            )
            
            # Log successful processing
            success_msg = (
                f"Successfully processed message: agent_id={message.agent_id}, "
                f"usage={usage.model_dump() if usage else 'None'}"
            )
            logger.debug(success_msg)
            
        except Exception as e:
            error_msg = f"Error processing message for agent_id={message.agent_id}: {e}"
            logger.error(error_msg, exc_info=True)
    
    def _consume_messages(self) -> None:
        """
        Main worker loop - continuously consume and process messages
        Runs in a separate thread
        """
        logger.debug("Queue worker started")
        
        while self._running:
            try:
                # Get message from queue (with timeout to allow graceful shutdown)
                message: QueueMessage = self.queue.get(timeout=1.0)
                
                # Log receipt of message
                log_msg = (
                    f"Received message: agent_id={message.agent_id}, "
                    f"user_id={message.user_id if message.HasField('user_id') else 'None'}, "
                    f"input_messages_count={len(message.input_messages)}"
                )
                logger.debug(log_msg)
                
                # Process the message through the server
                self._process_message(message)
                
            except Exception as e:
                # Handle timeout and other exceptions
                # For queue.Empty or StopIteration, just continue
                if type(e).__name__ in ['Empty', 'StopIteration']:
                    continue
                else:
                    error_msg = f"Error in message consumption loop: {e}"
                    logger.error(error_msg, exc_info=True)
        
        logger.info("Queue worker stopped")
    
    def start(self) -> None:
        """Start the background worker thread"""
        if self._running:
            logger.warning("Queue worker already running")
            return  # Already running
        
        logger.debug("Starting queue worker thread")
        self._running = True
        
        # Create and start daemon thread
        # Daemon threads automatically stop when the main program exits
        self._thread = threading.Thread(target=self._consume_messages, daemon=True)
        self._thread.start()
        
        logger.debug("Queue worker thread started successfully")
    
    def stop(self) -> None:
        """Stop the background worker thread"""
        if not self._running:
            logger.warning("Queue worker not running, nothing to stop")
            return  # Not running
        
        logger.debug("Stopping queue worker")
        self._running = False
        
        # Wait for thread to finish
        if self._thread:
            logger.debug("Waiting for worker thread to finish")
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Worker thread did not finish within timeout")
            else:
                logger.debug("Worker thread finished successfully")
        
        # Close queue resources
        logger.debug("Closing queue resources")
        self.queue.close()
        
        logger.debug("Queue worker stopped")


