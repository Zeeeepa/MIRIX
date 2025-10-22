import logging
from typing import List, Optional

from mirix.schemas.user import User
from mirix.schemas.message import MessageCreate
from mirix.schemas.message import MessageCreate as PydanticMessageCreate
from mirix.schemas.enums import MessageRole
from mirix.schemas.mirix_message_content import TextContent

from mirix.queue.message_pb2 import User as ProtoUser
from mirix.queue.message_pb2 import MessageCreate as ProtoMessageCreate
from mirix.queue.message_pb2 import QueueMessage
import mirix.queue as queue

logger = logging.getLogger(__name__)

def put_messages(
        actor: User,
        agent_id: str,
        input_messages: List[MessageCreate],
        server_id: Optional[str] = None,
        chaining: Optional[bool] = True,
        user_id: Optional[str] = None,
        verbose: Optional[bool] = None,
        filter_tags: Optional[dict] = None,
    ):
        """
        Create QueueMessage protobuf and send to queue.
        
        Args:
            actor: The user/actor sending the message
            agent_id: ID of the agent to send message to
            input_messages: List of messages to send
            server_id: Optional server ID for routing (ensures correct server processes the message)
            chaining: Enable/disable chaining
            user_id: Optional user ID
            verbose: Enable verbose logging
            filter_tags: Filter tags dictionary
        """
        logger.debug(f"Creating queue message for server_id={server_id}, agent_id={agent_id}, user={actor.id}")
        
        # Convert Pydantic User to protobuf User
        proto_user = ProtoUser()
        proto_user.id = actor.id
        proto_user.organization_id = actor.organization_id or ""
        proto_user.name = actor.name
        proto_user.status = actor.status
        proto_user.timezone = actor.timezone
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
            # For list content, we'd need to convert to structured_content
            # but for now, just convert to string representation
            elif isinstance(msg.content, list):
                # Convert list of content to string for now
                text_parts = []
                for content_part in msg.content:
                    if isinstance(content_part, TextContent):
                        text_parts.append(content_part.text)
                proto_msg.text_content = "\n".join(text_parts)
            
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
        
        # Set server_id for routing (required for multi-client scenarios)
        if server_id:
            queue_msg.server_id = server_id
        
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
        if filter_tags:
            queue_msg.filter_tags.update(filter_tags)
        
        # Send to queue
        logger.debug(f"Sending message to queue: agent_id={agent_id}, input_messages_count={len(input_messages)}")
        queue.save(queue_msg)
        logger.debug("Message successfully sent to queue")
