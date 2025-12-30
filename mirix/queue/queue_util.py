import logging
from typing import List, Optional

from mirix.schemas.client import Client
from mirix.schemas.message import MessageCreate
from mirix.schemas.enums import MessageRole
from mirix.schemas.mirix_message_content import (
    TextContent,
    ImageContent,
    FileContent,
    CloudFileContent,
)

from mirix.queue.message_pb2 import User as ProtoUser
from mirix.queue.message_pb2 import MessageCreate as ProtoMessageCreate
from mirix.queue.message_pb2 import QueueMessage
from mirix.queue.message_pb2 import (
    MessageContentList,
    MessageContentPart,
    TextContent as ProtoTextContent,
    ImageContent as ProtoImageContent,
    FileContent as ProtoFileContent,
    CloudFileContent as ProtoCloudFileContent,
)
import mirix.queue as queue
from mirix.services.memory_queue_trace_manager import MemoryQueueTraceManager

logger = logging.getLogger(__name__)

def put_messages(
        actor: Client,
        agent_id: str,
        input_messages: List[MessageCreate],
        chaining: Optional[bool] = True,
        user_id: Optional[str] = None,
        verbose: Optional[bool] = None,
        filter_tags: Optional[dict] = None,
        use_cache: bool = True,
        occurred_at: Optional[str] = None,
    ):
        """
        Create QueueMessage protobuf and send to queue.
        
        Args:
            actor: The Client performing the action (for auth/write operations)
                   Client ID is derived from actor.id
            agent_id: ID of the agent to send message to
            input_messages: List of messages to send
            chaining: Enable/disable chaining
            user_id: Optional user ID (end-user ID)
            verbose: Enable verbose logging
            filter_tags: Filter tags dictionary
            use_cache: Control Redis cache behavior
            occurred_at: Optional ISO 8601 timestamp string for episodic memory
        
        Returns:
            Optional[str]: Queue trace ID if tracing was recorded.
        """
        logger.debug("Creating queue message for agent_id=%s, actor=%s (client_id derived from actor)", agent_id, actor.id)
        
        if not actor or not actor.id:
            raise ValueError(
                f"Cannot queue message: actor is None or has no ID. "
                f"actor={actor}, actor.id={actor.id if actor else 'N/A'}"
            )
        
        # Create queue trace record (best-effort, do not block enqueue)
        trace_id = None
        try:
            trace_manager = MemoryQueueTraceManager()
            trace = trace_manager.create_trace(
                actor=actor,
                agent_id=agent_id,
                user_id=user_id,
                message_count=len(input_messages),
            )
            trace_id = trace.id
        except Exception as exc:
            logger.warning("Failed to create queue trace: %s", exc)

        # Convert Pydantic Client to protobuf User (protobuf schema still uses "User")
        proto_user = ProtoUser()
        proto_user.id = actor.id
        proto_user.organization_id = actor.organization_id or ""
        proto_user.name = actor.name
        proto_user.status = actor.status
        # Client doesn't have timezone - use default "UTC"
        proto_user.timezone = getattr(actor, 'timezone', 'UTC')
        if actor.created_at:
            proto_user.created_at.FromDatetime(actor.created_at)
        if actor.updated_at:
            proto_user.updated_at.FromDatetime(actor.updated_at)
        proto_user.is_deleted = actor.is_deleted
        
        # Convert Pydantic MessageCreate list to protobuf MessageCreate list
        proto_input_messages = []
        for msg in input_messages:
            proto_msg = ProtoMessageCreate()
            # Map role
            if msg.role == MessageRole.user:
                proto_msg.role = ProtoMessageCreate.ROLE_USER
            elif msg.role == MessageRole.system:
                proto_msg.role = ProtoMessageCreate.ROLE_SYSTEM
            else:
                proto_msg.role = ProtoMessageCreate.ROLE_UNSPECIFIED
            
            # Handle content (can be string or list)
            if isinstance(msg.content, str):
                proto_msg.text_content = msg.content
            elif isinstance(msg.content, list):
                # Convert list of content to structured_content
                proto_content_list = MessageContentList()
                for content_part in msg.content:
                    proto_part = MessageContentPart()
                    
                    if isinstance(content_part, TextContent):
                        proto_text = ProtoTextContent()
                        proto_text.text = content_part.text
                        proto_part.text.CopyFrom(proto_text)
                    elif isinstance(content_part, ImageContent):
                        proto_image = ProtoImageContent()
                        proto_image.image_id = content_part.image_id
                        if content_part.detail:
                            proto_image.detail = content_part.detail
                        proto_part.image.CopyFrom(proto_image)
                    elif isinstance(content_part, FileContent):
                        proto_file = ProtoFileContent()
                        proto_file.file_id = content_part.file_id
                        proto_part.file.CopyFrom(proto_file)
                    elif isinstance(content_part, CloudFileContent):
                        proto_cloud_file = ProtoCloudFileContent()
                        proto_cloud_file.cloud_file_uri = content_part.cloud_file_uri
                        proto_part.cloud_file.CopyFrom(proto_cloud_file)
                    else:
                        logger.warning("Unknown content type: %s, skipping", type(content_part))
                        continue
                    
                    proto_content_list.parts.append(proto_part)
                
                proto_msg.structured_content.CopyFrom(proto_content_list)
            
            # Optional fields
            if msg.name:
                proto_msg.name = msg.name
            if msg.otid:
                proto_msg.otid = msg.otid
            if msg.sender_id:
                proto_msg.sender_id = msg.sender_id
            if msg.group_id:
                proto_msg.group_id = msg.group_id
            
            proto_input_messages.append(proto_msg)
        
        # Build the QueueMessage
        queue_msg = QueueMessage()
        
        queue_msg.actor.CopyFrom(proto_user)
            
        queue_msg.agent_id = agent_id
        queue_msg.input_messages.extend(proto_input_messages)
        
        # Optional fields
        if chaining is not None:
            queue_msg.chaining = chaining
        if user_id:
            queue_msg.user_id = user_id
        if verbose is not None:
            queue_msg.verbose = verbose
        
        # Convert dict to Struct for filter_tags
        if filter_tags is not None:
            filter_tags = dict(filter_tags)
        else:
            filter_tags = {}
        if trace_id:
            filter_tags["__queue_trace_id"] = trace_id
        if filter_tags:
            queue_msg.filter_tags.update(filter_tags)
        
        # Set use_cache
        queue_msg.use_cache = use_cache
        
        # Set occurred_at if provided
        if occurred_at is not None:
            queue_msg.occurred_at = occurred_at
        
        # Send to queue
        logger.debug("Sending message to queue: agent_id=%s, input_messages_count=%s, occurred_at=%s", 
                    agent_id, len(input_messages), occurred_at)
        queue.save(queue_msg)
        logger.debug("Message successfully sent to queue")
        return trace_id
