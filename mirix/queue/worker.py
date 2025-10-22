"""
Background worker that consumes messages from the queue
Runs in a daemon thread and processes messages through the server

Uses server registry to route messages to the correct server based on server_id
"""
import logging
import threading
from typing import TYPE_CHECKING, Optional, List
from datetime import datetime

from mirix.queue.message_pb2 import QueueMessage
from mirix.queue.server_registry import get_registry

if TYPE_CHECKING:
    from .queue_interface import QueueInterface
    from mirix.server.server import SyncServer
    from mirix.schemas.user import User
    from mirix.schemas.message import MessageCreate

logger = logging.getLogger(__name__)

class QueueWorker:
    """Background worker that processes messages from the queue"""
    
    def __init__(self, queue: 'QueueInterface', server: Optional['SyncServer'] = None):
        """
        Initialize the queue worker
        
        Args:
            queue: Queue implementation to consume from
            server: Optional SyncServer instance to process messages through
        """
        logger.debug(f"Initializing queue worker with server={'provided' if server else 'None'}")
        
        self.queue = queue
        self.server = server
        self._running = False
        self._thread = None
    
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
    
    def _process_message(self, message: QueueMessage) -> None:
        """
        Process a queue message by calling server.send_messages()
        
        Routes the message to the correct server using the server_id field.
        Falls back to self.server if server_id is not set or server not found in registry.
        
        Args:
            message: QueueMessage protobuf to process
        """
        # Get the server registry
        registry = get_registry()
        
        # Determine which server should process this message
        target_server = None
        if message.server_id:
            # Try to get server from registry using server_id
            target_server = registry.get(message.server_id)
            if target_server is None:
                logger.warning(
                    f"Server {message.server_id} not found in registry, "
                    f"falling back to worker's default server"
                )
        
        # Fall back to worker's default server if no server_id or not found
        if target_server is None:
            target_server = self.server
        
        # If still no server available, just log
        if target_server is None:
            log_msg = (
                f"No server available - logging only: server_id={message.server_id}, "
                f"agent_id={message.agent_id}, "
                f"user_id={message.user_id if message.HasField('user_id') else 'None'}, "
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
            user_id = message.user_id if message.HasField('user_id') else None
            verbose = message.verbose if message.HasField('verbose') else None
            
            # Convert filter_tags from Struct to dict
            filter_tags = None
            if message.HasField('filter_tags'):
                filter_tags = dict(message.filter_tags)
            
            # Log the processing
            log_msg = (
                f"Processing message: server_id={message.server_id}, "
                f"agent_id={message.agent_id}, "
                f"user_id={user_id or 'None'}, "
                f"input_messages_count={len(input_messages)}"
            )
            logger.info(log_msg)
            
            # Call server.send_messages() on the target server
            usage = target_server.send_messages(
                actor=actor,
                agent_id=message.agent_id,
                input_messages=input_messages,
                chaining=chaining,
                user_id=user_id,
                verbose=verbose,
                filter_tags=filter_tags
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


