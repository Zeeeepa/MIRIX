import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import List, Optional

from mirix.agent import Agent, AgentState
from mirix.schemas.episodic_memory import EpisodicEventForLLM
from mirix.schemas.knowledge import KnowledgeItemBase
from mirix.schemas.mirix_message_content import TextContent
from mirix.schemas.procedural_memory import ProceduralMemoryItemBase
from mirix.schemas.resource_memory import ResourceMemoryItemBase
from mirix.schemas.semantic_memory import SemanticMemoryItemBase


def core_memory_append(
    self: "Agent", agent_state: "AgentState", label: str, content: str
) -> Optional[str]:  # type: ignore
    """
    Append to the contents of core memory. The content will be appended to the end of the block with the given label. If you hit the limit, you can use `core_memory_rewrite` to rewrite the entire block to shorten the content. Note that "Line n:" is only for your visualization of the memory, and you should not include it in the content.

    Args:
        label (str): Section of the memory to be edited (persona or human).
        content (str): Content to write to the memory. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is returned on success, or an error message string if the operation would exceed the limit.
    """
    # check if the content starts with something like "Line n:" (here n is a number) using regex
    if re.match(r"^Line \d+:", content):
        raise ValueError(
            "You should not include 'Line n:' (here n is a number) in the content."
        )

    # Get the current block and its limit
    current_block = agent_state.memory.get_block(label)
    current_value = str(current_block.value)
    limit = current_block.limit
    
    # Calculate the new value and its length
    new_value = (current_value + "\n" + str(content)).strip()
    new_length = len(new_value)
    
    # Check if the new value would exceed the limit
    if new_length > limit:
        # Return a descriptive error message instead of raising an exception
        # This allows the agent to see the error and adapt its behavior
        error_msg = (
            f"ERROR: Cannot append - would exceed {limit} character limit "
            f"(current: {len(current_value)}, adding: {len(content)}, "
            f"total would be: {new_length}). "
            f"Use core_memory_rewrite to condense the '{label}' block first, "
            f"targeting around {int(limit * 0.5)} characters (~50% capacity)."
        )
        return error_msg
    
    # If within limit, perform the append
    agent_state.memory.update_block_value(label=label, value=new_value)
    return None


def core_memory_rewrite(
    self: "Agent", agent_state: "AgentState", label: str, content: str
) -> Optional[str]:  # type: ignore
    """
    Rewrite the entire content of block <label> in core memory. The entire content in that block will be replaced with the new content. If the old content is full, and you have to rewrite the entire content, make sure to be extremely concise and make it shorter than 20% of the limit.

    Args:
        label (str): Section of the memory to be edited (persona or human).
        content (str): Content to write to the memory. All unicode (including emojis) are supported.
    Returns:
        Optional[str]: None is returned on success, or an error message string if the new content exceeds the limit.
    """
    # Get the current block and its limit
    current_block = agent_state.memory.get_block(label)
    current_value = str(current_block.value)
    limit = current_block.limit
    new_value = content.strip()
    new_length = len(new_value)
    
    # Check if the new value exceeds the limit
    if new_length > limit:
        error_msg = (
            f"ERROR: Rewrite failed - new content exceeds {limit} character limit "
            f"(provided: {new_length} characters). "
            f"Please condense further, targeting around {int(limit * 0.5)} characters (~50% capacity)."
        )
        return error_msg
    
    # Only update if the content actually changed
    if current_value != new_value:
        agent_state.memory.update_block_value(label=label, value=new_value)
        # Provide feedback on the operation
        percentage = int((new_length / limit) * 100)
        return f"Successfully rewrote '{label}' block: {new_length}/{limit} characters ({percentage}% full)."
    
    return None


def episodic_memory_insert(self: "Agent", items: List[EpisodicEventForLLM]):
    """
    The tool to update episodic memory. The item being inserted into the episodic memory is an event either happened on the user or the assistant.

    Args:
        items (array): List of episodic memory items to insert.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    agent_id = (
        self.agent_state.parent_id
        if self.agent_state.parent_id is not None
        else self.agent_state.id
    )
    
    # Get filter_tags, use_cache, client_id, user_id, and occurred_at from agent instance
    filter_tags = getattr(self, 'filter_tags', None)
    use_cache = getattr(self, 'use_cache', True)
    user_id = getattr(self, 'user_id', None)
    occurred_at_override = getattr(self, 'occurred_at', None)  # Optional timestamp override from API

    for item in items:
        # Use occurred_at_override if provided, otherwise use LLM-extracted timestamp
        timestamp = occurred_at_override if occurred_at_override else item["occurred_at"]
        
        # Convert string to datetime if needed
        if isinstance(timestamp, str):
            from datetime import datetime
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        self.episodic_memory_manager.insert_event(
            actor=self.actor,
            agent_state=self.agent_state,
            agent_id=agent_id,
            timestamp=timestamp,  # Use potentially overridden timestamp
            event_type=item["event_type"],
            event_actor=item["actor"],
            summary=item["summary"],
            details=item["details"],
            organization_id=self.actor.organization_id,
            filter_tags=filter_tags if filter_tags else None,
            use_cache=use_cache,
            user_id=user_id,
        )
    response = "Events inserted! Now you need to check if there are repeated events shown in the system prompt."
    return response


def episodic_memory_merge(
    self: "Agent",
    event_id: str,
    combined_summary: str = None,
    combined_details: str = None,
):
    """
    The tool to merge the new episodic event into the selected episodic event by event_id, should be used when the user is continuing doing the same thing with more details. The combined_summary and combined_details will overwrite the old summary and details of the selected episodic event. Thus DO NOT use "User continues xxx" as the combined_summary because the old one WILL BE OVERWRITTEN and then we can only see "User continus xxx" without the old event.

    Args:
        event_id (str): This is the id of which episodic event to append to.
        combined_summary (str): The updated summary. Note that it will overwrite the old summary so make sure to include the information from the old summary. The new summary needs to be only slightly different from the old summary.
        combined_details (str): The new details to add into the details of the selected episodic event.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """

    episodic_memory = self.episodic_memory_manager.update_event(
        event_id=event_id,
        new_summary=combined_summary,
        new_details=combined_details,
        actor=self.actor,
    )
    response = (
        "These are the `summary` and the `details` of the updated event:\n",
        str(
            {
                "event_id": episodic_memory.id,
                "summary": episodic_memory.summary,
                "details": episodic_memory.details,
            }
        )
        + "\nIf the `details` are too verbose, or the `summary` cannot cover the information in the `details`, call episodic_memory_replace to update this event.",
    )
    return response


def episodic_memory_replace(
    self: "Agent", event_ids: List[str], new_items: List[EpisodicEventForLLM]
):
    """
    The tool to replace or delete items in the episodic memory. To replace the memory, set the event_ids to be the ids of the events that needs to be replaced and new_items as the updated events. Note that the number of new items does not need to be the same as the number of event_ids as it is not a one-to-one mapping. To delete the memory, set the event_ids to be the ids of the events that needs to be deleted and new_items as an empty list. To insert new events, use episodic_memory_insert function.

    Args:
        event_ids (str): The ids of the episodic events to be deleted (or replaced).
        new_items (array): List of new episodic memory items to insert. If this is an empty list, then it means that the items are being deleted.
    """
    agent_id = (
        self.agent_state.parent_id
        if self.agent_state.parent_id is not None
        else self.agent_state.id
    )
    
    # Get filter_tags, use_cache, client_id, user_id, and occurred_at from agent instance
    filter_tags = getattr(self, 'filter_tags', None)
    use_cache = getattr(self, 'use_cache', True)
    occurred_at_override = getattr(self, 'occurred_at', None)  # Optional timestamp override from API
    user_id = self.user_id

    valid_event_ids = []
    for event_id in event_ids:
        # Check if the event exists and is accessible
        event = self.episodic_memory_manager.get_episodic_memory_by_id(
            event_id, actor=self.actor, user_id=user_id
        )
        if event is not None:
            valid_event_ids.append(event_id)

    for event_id in valid_event_ids:
        self.episodic_memory_manager.delete_event_by_id(event_id, actor=self.actor)

    for new_item in new_items:
        # Use occurred_at_override if provided, otherwise use LLM-extracted timestamp
        timestamp = occurred_at_override if occurred_at_override else new_item["occurred_at"]
        
        # Convert string to datetime if needed
        if isinstance(timestamp, str):
            from datetime import datetime
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        self.episodic_memory_manager.insert_event(
            actor=self.actor,
            agent_state=self.agent_state,
            agent_id=agent_id,
            timestamp=timestamp,  # Use potentially overridden timestamp
            event_type=new_item["event_type"],
            event_actor=new_item["actor"],
            summary=new_item["summary"],
            details=new_item["details"],
            organization_id=self.actor.organization_id,
            filter_tags=filter_tags if filter_tags else None,
            use_cache=use_cache,
            user_id=user_id,
        )


def check_episodic_memory(
    self: "Agent", event_ids: List[str], timezone_str: str
) -> List[EpisodicEventForLLM]:
    """
    The tool to check the episodic memory. This function will return the episodic events with the given event_ids.

    Args:
        event_ids (str): The ids of the episodic events to be checked.

    Returns:
        List[EpisodicEventForLLM]: List of episodic events with the given event_ids.
    """
    episodic_memory = [
        self.episodic_memory_manager.get_episodic_memory_by_id(
            event_id, timezone_str=timezone_str, actor=self.actor, user_id=self.user_id
        )
        for event_id in event_ids
    ]

    formatted_results = [
        {
            "event_id": x.id,
            "timestamp": x.occurred_at,
            "event_type": x.event_type,
            "actor": x.actor,
            "summary": x.summary,
            "details": x.details,
        }
        for x in episodic_memory
        if x is not None
    ]

    return formatted_results


def resource_memory_insert(self: "Agent", items: List[ResourceMemoryItemBase]):
    """
    The tool to insert new items into resource memory.

    Args:
        items (array): List of resource memory items to insert.

    Returns:
        Optional[str]: Message about insertion results including any duplicates detected.
    """
    # No imports needed - using agent instance attributes
    
    agent_id = (
        self.agent_state.parent_id
        if self.agent_state.parent_id is not None
        else self.agent_state.id
    )
    
    # Get filter_tags, use_cache, client_id, and user_id from agent instance
    filter_tags = getattr(self, 'filter_tags', None)
    use_cache = getattr(self, 'use_cache', True)
    user_id = getattr(self, 'user_id', None)

    inserted_count = 0
    skipped_count = 0
    skipped_titles = []

    for item in items:
        # Check for existing similar resources (by title, summary, and filter_tags)
        existing_resources = self.resource_memory_manager.list_resources(
            agent_state=self.agent_state,
            user=self.user,  # User for read operations (data filtering)
            query="",  # Get all resources
            limit=1000,  # Get enough to check for duplicates
            filter_tags=filter_tags if filter_tags else None,
            use_cache=use_cache,
        )
        
        # Check if this resource already exists
        is_duplicate = False
        for existing in existing_resources:
            if (existing.title == item["title"] and 
                existing.summary == item["summary"] and
                existing.content == item["content"]):
                is_duplicate = True
                skipped_count += 1
                skipped_titles.append(item["title"])
                break
        
        if not is_duplicate:
            self.resource_memory_manager.insert_resource(
                actor=self.actor,
                agent_state=self.agent_state,
                agent_id=agent_id,
                title=item["title"],
                summary=item["summary"],
                resource_type=item["resource_type"],
                content=item["content"],
                organization_id=self.actor.organization_id,
                filter_tags=filter_tags if filter_tags else None,
                use_cache=use_cache,
                user_id=user_id,
            )
            inserted_count += 1
    
    # Return feedback message
    if skipped_count > 0:
        skipped_list = ", ".join(f"'{t}'" for t in skipped_titles[:3])
        if len(skipped_titles) > 3:
            skipped_list += f" and {len(skipped_titles) - 3} more"
        return f"Inserted {inserted_count} new resource(s). Skipped {skipped_count} duplicate(s): {skipped_list}."
    elif inserted_count > 0:
        return f"Successfully inserted {inserted_count} new resource(s)."
    else:
        return "No resources were inserted."


def resource_memory_update(
    self: "Agent", old_ids: List[str], new_items: List[ResourceMemoryItemBase]
):
    """
    The tool to update and delete items in the resource memory. To update the memory, set the old_ids to be the ids of the items that needs to be updated and new_items as the updated items. Note that the number of new items does not need to be the same as the number of old ids as it is not a one-to-one mapping. To delete the memory, set the old_ids to be the ids of the items that needs to be deleted and new_items as an empty list.

    Args:
        old_ids (array): List of ids of the items to be deleted (or updated).
        new_items (array): List of new resource memory items to insert. If this is an empty list, then it means that the items are being deleted.
    """
    agent_id = (
        self.agent_state.parent_id
        if self.agent_state.parent_id is not None
        else self.agent_state.id
    )
    
    # Get filter_tags, use_cache, client_id, and user_id from agent instance
    filter_tags = getattr(self, 'filter_tags', None)
    use_cache = getattr(self, 'use_cache', True)
    user_id = getattr(self, 'user_id', None)

    for old_id in old_ids:
        self.resource_memory_manager.delete_resource_by_id(
            resource_id=old_id, actor=self.actor
        )

    for item in new_items:
        self.resource_memory_manager.insert_resource(
            actor=self.actor,
            agent_state=self.agent_state,
            agent_id=agent_id,
            title=item["title"],
            summary=item["summary"],
            resource_type=item["resource_type"],
            content=item["content"],
            organization_id=self.actor.organization_id,
            filter_tags=filter_tags if filter_tags else None,
            use_cache=use_cache,
            user_id=user_id,
        )


def procedural_memory_insert(self: "Agent", items: List[ProceduralMemoryItemBase]):
    """
    The tool to insert new procedures into procedural memory. Note that the `summary` should not be a general term such as "guide" or "workflow" but rather a more informative description of the procedure.

    Args:
        items (array): List of procedural memory items to insert.

    Returns:
        Optional[str]: Message about insertion results including any duplicates detected.
    """
    agent_id = (
        self.agent_state.parent_id
        if self.agent_state.parent_id is not None
        else self.agent_state.id
    )
    
    # Get filter_tags, use_cache, client_id, and user_id from agent instance
    filter_tags = getattr(self, 'filter_tags', None)
    use_cache = getattr(self, 'use_cache', True)
    user_id = getattr(self, 'user_id', None)

    inserted_count = 0
    skipped_count = 0
    skipped_summaries = []

    for item in items:
        # Check for existing similar procedures (by summary and filter_tags)
        existing_procedures = self.procedural_memory_manager.list_procedures(
            agent_state=self.agent_state,
            user=self.user,  # User for read operations (data filtering)
            query="",  # Get all procedures
            limit=1000,  # Get enough to check for duplicates
            filter_tags=filter_tags if filter_tags else None,
            use_cache=use_cache,
        )
        
        # Check if this procedure already exists
        is_duplicate = False
        for existing in existing_procedures:
            if (existing.summary == item["summary"] and 
                existing.steps == item["steps"]):
                is_duplicate = True
                skipped_count += 1
                skipped_summaries.append(item["summary"])
                break
        
        if not is_duplicate:
            self.procedural_memory_manager.insert_procedure(
                agent_state=self.agent_state,
                agent_id=agent_id,
                entry_type=item["entry_type"],
                summary=item["summary"],
                steps=item["steps"],
                actor=self.actor,
                organization_id=self.user.organization_id,
                filter_tags=filter_tags if filter_tags else None,
                use_cache=use_cache,
                user_id=user_id,
            )
            inserted_count += 1
    
    # Return feedback message
    if skipped_count > 0:
        skipped_list = ", ".join(f"'{s}'" for s in skipped_summaries[:3])
        if len(skipped_summaries) > 3:
            skipped_list += f" and {len(skipped_summaries) - 3} more"
        return f"Inserted {inserted_count} new procedure(s). Skipped {skipped_count} duplicate(s): {skipped_list}."
    elif inserted_count > 0:
        return f"Successfully inserted {inserted_count} new procedure(s)."
    else:
        return "No procedures were inserted."


def procedural_memory_update(
    self: "Agent", old_ids: List[str], new_items: List[ProceduralMemoryItemBase]
):
    """
    The tool to update/delete items in the procedural memory. To update the memory, set the old_ids to be the ids of the items that needs to be updated and new_items as the updated items. Note that the number of new items does not need to be the same as the number of old ids as it is not a one-to-one mapping. To delete the memory, set the old_ids to be the ids of the items that needs to be deleted and new_items as an empty list.

    Args:
        old_ids (array): List of ids of the items to be deleted (or updated).
        new_items (array): List of new procedural memory items to insert. If this is an empty list, then it means that the items are being deleted.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    agent_id = (
        self.agent_state.parent_id
        if self.agent_state.parent_id is not None
        else self.agent_state.id
    )
    
    # Get filter_tags, use_cache, client_id, and user_id from agent instance
    filter_tags = getattr(self, 'filter_tags', None)
    use_cache = getattr(self, 'use_cache', True)
    user_id = getattr(self, 'user_id', None)

    for old_id in old_ids:
        self.procedural_memory_manager.delete_procedure_by_id(
            procedure_id=old_id, actor=self.actor
        )

    for item in new_items:
        self.procedural_memory_manager.insert_procedure(
            agent_state=self.agent_state,
            agent_id=agent_id,
            entry_type=item["entry_type"],
            summary=item["summary"],
            steps=item["steps"],
            actor=self.actor,
            organization_id=self.actor.organization_id,
            filter_tags=filter_tags if filter_tags else None,
            use_cache=use_cache,
            user_id=user_id,
        )


def check_semantic_memory(
    self: "Agent", semantic_item_ids: List[str], timezone_str: str
) -> List[SemanticMemoryItemBase]:
    """
    The tool to check the semantic memory. This function will return the semantic memory items with the given ids.

    Args:
        semantic_item_ids (str): The ids of the semantic memory items to be checked.

    Returns:
        List[SemanticMemoryItemBase]: List of semantic memory items with the given ids.
    """
    semantic_memory = [
        self.semantic_memory_manager.get_semantic_item_by_id(
            semantic_memory_id=id, timezone_str=timezone_str, actor=self.actor
        )
        for id in semantic_item_ids
    ]

    formatted_results = [
        {
            "semantic_item_id": x.id,
            "name": x.name,
            "summary": x.summary,
            "details": x.details,
            "source": x.source,
        }
        for x in semantic_memory
    ]

    return formatted_results


def semantic_memory_insert(self: "Agent", items: List[SemanticMemoryItemBase]):
    """
    The tool to insert items into semantic memory.

    Args:
        items (array): List of semantic memory items to insert.

    Returns:
        Optional[str]: Message about insertion results including any duplicates detected.
    """
    agent_id = (
        self.agent_state.parent_id
        if self.agent_state.parent_id is not None
        else self.agent_state.id
    )
    
    # Get filter_tags, use_cache, client_id, and user_id from agent instance
    filter_tags = getattr(self, 'filter_tags', None)
    use_cache = getattr(self, 'use_cache', True)
    user_id = getattr(self, 'user_id', None)

    inserted_count = 0
    skipped_count = 0
    skipped_names = []

    for item in items:
        # Check for existing similar semantic items (by name, summary, and filter_tags)
        existing_items = self.semantic_memory_manager.list_semantic_items(
            agent_state=self.agent_state,
            user=self.user,  # User for read operations (data filtering)
            query="",  # Get all items
            limit=1000,  # Get enough to check for duplicates
            filter_tags=filter_tags if filter_tags else None,
            use_cache=use_cache,
        )
        
        # Check if this semantic item already exists
        is_duplicate = False
        for existing in existing_items:
            if (existing.name == item["name"] and 
                existing.summary == item["summary"] and
                existing.details == item["details"]):
                is_duplicate = True
                skipped_count += 1
                skipped_names.append(item["name"])
                break
        
        if not is_duplicate:
            self.semantic_memory_manager.insert_semantic_item(
                agent_state=self.agent_state,
                agent_id=agent_id,
                name=item["name"],
                summary=item["summary"],
                details=item["details"],
                source=item["source"],
                organization_id=self.actor.organization_id,
                actor=self.actor,  # Client for write operations
                filter_tags=filter_tags if filter_tags else None,
                use_cache=use_cache,
                user_id=user_id,
            )
            inserted_count += 1
    
    # Return feedback message
    if skipped_count > 0:
        skipped_list = ", ".join(f"'{n}'" for n in skipped_names[:3])
        if len(skipped_names) > 3:
            skipped_list += f" and {len(skipped_names) - 3} more"
        return f"Inserted {inserted_count} new semantic item(s). Skipped {skipped_count} duplicate(s): {skipped_list}."
    elif inserted_count > 0:
        return f"Successfully inserted {inserted_count} new semantic item(s)."
    else:
        return "No semantic items were inserted."


def semantic_memory_update(
    self: "Agent",
    old_semantic_item_ids: List[str],
    new_items: List[SemanticMemoryItemBase],
):
    """
    The tool to update/delete items in the semantic memory. To update the memory, set the old_ids to be the ids of the items that needs to be updated and new_items as the updated items. Note that the number of new items does not need to be the same as the number of old ids as it is not a one-to-one mapping. To delete the memory, set the old_ids to be the ids of the items that needs to be deleted and new_items as an empty list.

    Args:
        old_semantic_item_ids (array): List of ids of the items to be deleted (or updated).
        new_items (array): List of new semantic memory items to insert. If this is an empty list, then it means that the items are being deleted.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    agent_id = (
        self.agent_state.parent_id
        if self.agent_state.parent_id is not None
        else self.agent_state.id
    )
    
    # Get filter_tags, use_cache, client_id, and user_id from agent instance
    filter_tags = getattr(self, 'filter_tags', None)
    use_cache = getattr(self, 'use_cache', True)
    user_id = getattr(self, 'user_id', None)
    
    for old_id in old_semantic_item_ids:
        self.semantic_memory_manager.delete_semantic_item_by_id(
            semantic_memory_id=old_id, actor=self.actor
        )

    new_ids = []
    for item in new_items:
        inserted_item = self.semantic_memory_manager.insert_semantic_item(
            agent_state=self.agent_state,
            agent_id=agent_id,
            name=item["name"],
            summary=item["summary"],
            details=item["details"],
            source=item["source"],
            actor=self.actor,
            organization_id=self.actor.organization_id,
            filter_tags=filter_tags if filter_tags else None,
            use_cache=use_cache,
            user_id=user_id,
        )
        new_ids.append(inserted_item.id)

    message_to_return = (
        "Semantic memory with the following ids have been deleted: "
        + str(old_semantic_item_ids)
        + f". New semantic memory items are created: {str(new_ids)}"
    )
    return message_to_return


def knowledge_insert(self: "Agent", items: List[KnowledgeItemBase]):
    """
    The tool to update knowledge.

    Args:
        items (array): List of knowledge items to insert.

    Returns:
        Optional[str]: Message about insertion results including any duplicates detected.
    """
    agent_id = (
        self.agent_state.parent_id
        if self.agent_state.parent_id is not None
        else self.agent_state.id
    )
    
    # Get filter_tags, use_cache, and user_id from agent instance
    filter_tags = getattr(self, 'filter_tags', None)
    use_cache = getattr(self, 'use_cache', True)
    user_id = getattr(self, 'user_id', None)

    inserted_count = 0
    skipped_count = 0
    skipped_captions = []

    for item in items:
        # Check for existing similar knowledge items (by caption, source, and filter_tags)
        existing_items = self.knowledge_memory_manager.list_knowledge(
            agent_state=self.agent_state,
            user=self.user,  # User for read operations (data filtering)
            query="",  # Get all items
            limit=1000,  # Get enough to check for duplicates
            filter_tags=filter_tags if filter_tags else None,
            use_cache=use_cache,
        )
        
        # Check if this knowledge item already exists
        is_duplicate = False
        for existing in existing_items:
            if (existing.caption == item["caption"] and 
                existing.source == item["source"] and
                existing.secret_value == item["secret_value"]):
                is_duplicate = True
                skipped_count += 1
                skipped_captions.append(item["caption"])
                break
        
        if not is_duplicate:
            self.knowledge_memory_manager.insert_knowledge(
                actor=self.actor,
                agent_state=self.agent_state,
                agent_id=agent_id,
                entry_type=item["entry_type"],
                source=item["source"],
                sensitivity=item["sensitivity"],
                secret_value=item["secret_value"],
                caption=item["caption"],
                organization_id=self.actor.organization_id,
                filter_tags=filter_tags if filter_tags else None,
                use_cache=use_cache,
                user_id=user_id,
            )
            inserted_count += 1
    
    # Return feedback message
    if skipped_count > 0:
        skipped_list = ", ".join(f"'{c}'" for c in skipped_captions[:3])
        if len(skipped_captions) > 3:
            skipped_list += f" and {len(skipped_captions) - 3} more"
        return f"Inserted {inserted_count} new knowledge item(s). Skipped {skipped_count} duplicate(s): {skipped_list}."
    elif inserted_count > 0:
        return f"Successfully inserted {inserted_count} new knowledge item(s)."
    else:
        return "No knowledge items were inserted."


def knowledge_update(
    self: "Agent", old_ids: List[str], new_items: List[KnowledgeItemBase]
):
    """
    The tool to update/delete items in the knowledge. To update the knowledge, set the old_ids to be the ids of the items that needs to be updated and new_items as the updated items. Note that the number of new items does not need to be the same as the number of old ids as it is not a one-to-one mapping. To delete the memory, set the old_ids to be the ids of the items that needs to be deleted and new_items as an empty list.

    Args:
        old_ids (array): List of ids of the items to be deleted (or updated).
        new_items (array): List of new knowledge items to insert. If this is an empty list, then it means that the items are being deleted.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response
    """
    agent_id = (
        self.agent_state.parent_id
        if self.agent_state.parent_id is not None
        else self.agent_state.id
    )
    
    # Get filter_tags, use_cache, and user_id from agent instance
    filter_tags = getattr(self, 'filter_tags', None)
    use_cache = getattr(self, 'use_cache', True)
    user_id = getattr(self, 'user_id', None)

    for old_id in old_ids:
        self.knowledge_memory_manager.delete_knowledge_by_id(
            knowledge_item_id=old_id, actor=self.actor
        )

    for item in new_items:
        self.knowledge_memory_manager.insert_knowledge(
            actor=self.actor,
            agent_state=self.agent_state,
            agent_id=agent_id,
            entry_type=item["entry_type"],
            source=item["source"],
            sensitivity=item["sensitivity"],
            secret_value=item["secret_value"],
            caption=item["caption"],
            organization_id=self.actor.organization_id,
            filter_tags=filter_tags if filter_tags else None,
            use_cache=use_cache,
            user_id=user_id,
        )


def trigger_memory_update_with_instruction(
    self: "Agent", user_message: object, instruction: str, memory_type: str
) -> Optional[str]:
    """
    Choose which memory to update. The function will trigger one specific memory agent with the instruction telling the agent what to do.

    Args:
        instruction (str): The instruction to the memory agent.
        memory_type (str): The type of memory to update. It should be chosen from the following: "core", "episodic", "resource", "procedural", "knowledge", "semantic". For instance, ['episodic', 'resource'].

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """

    from mirix.interface import QueuingInterface
    from mirix.server.server import SyncServer
    from mirix.services.queue_trace_context import (
        get_agent_trace_id,
        reset_parent_agent_trace_id,
        set_parent_agent_trace_id,
    )

    # Initialize server directly
    interface = QueuingInterface(debug=False)
    server = SyncServer(default_interface_factory=lambda: interface)
    
    # Get default client for actor
    client_id = server.client_manager.DEFAULT_CLIENT_ID
    org_id = server.organization_manager.DEFAULT_ORG_ID
    client = server.client_manager.get_client_or_default(client_id, org_id)
    
    agents = server.agent_manager.list_agents(actor=client)

    # Validate that user_message is a dictionary
    if not isinstance(user_message, dict):
        raise TypeError(
            f"user_message must be a dictionary, got {type(user_message).__name__}: {user_message}"
        )

    # Fallback to sequential processing for backward compatibility
    response = ""

    if memory_type == "core":
        agent_type = "core_memory_agent"
    elif memory_type == "episodic":
        agent_type = "episodic_memory_agent"
    elif memory_type == "resource":
        agent_type = "resource_memory_agent"
    elif memory_type == "procedural":
        agent_type = "procedural_memory_agent"
    elif memory_type == "knowledge":
        agent_type = "knowledge_memory_agent"
    elif memory_type == "semantic":
        agent_type = "semantic_memory_agent"
    else:
        raise ValueError(
            f"Memory type '{memory_type}' is not supported. Please choose from 'core', 'episodic', 'resource', 'procedural', 'knowledge', 'semantic'."
        )

    matching_agent = None
    for agent in agents:
        if agent.agent_type == agent_type:
            matching_agent = agent
            break

    if matching_agent is None:
        raise ValueError(f"No agent found with type '{agent_type}'")

    parent_trace_id = get_agent_trace_id()
    parent_token = None
    if parent_trace_id:
        parent_token = set_parent_agent_trace_id(parent_trace_id)
    try:
        from mirix.schemas.message import MessageCreate
        from mirix.schemas.enums import MessageRole
        from mirix.schemas.mirix_message_content import MessageContentType, TextContent
        
        # Build message
        message_text = "[Message from Chat Agent (Now you are allowed to make multiple function calls sequentially)] " + instruction
        message_content = [TextContent(type=MessageContentType.text, text=message_text)]
        input_messages = [MessageCreate(role=MessageRole.user, content=message_content)]
        
        # Send messages via server
        interface.clear()
        server.send_messages(
            actor=client,
            agent_id=matching_agent.id,
            input_messages=input_messages,
            user=self.user,
        )
    finally:
        if parent_token:
            reset_parent_agent_trace_id(parent_token)
    response += (
        "[System Message] Agent "
        + matching_agent.name
        + " has been triggered to update the memory.\n"
    )

    return response.strip()


def trigger_memory_update(
    self: "Agent", user_message: object, memory_types: List[str]
) -> Optional[str]:
    """
    Choose which memory to update. This function will trigger another memory agent which is specifically in charge of handling the corresponding memory to update its memory. Trigger all necessary memory updates at once.

    Args:
        memory_types (List[str]): The types of memory to update. It should be chosen from the following: "core", "episodic", "resource", "procedural", "knowledge", "semantic". For instance, ['episodic', 'resource'].

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """

    from mirix.agent import (
        CoreMemoryAgent,
        EpisodicMemoryAgent,
        KnowledgeMemoryAgent,
        ProceduralMemoryAgent,
        ResourceMemoryAgent,
        SemanticMemoryAgent,
    )
    from mirix.services.memory_agent_trace_manager import MemoryAgentTraceManager
    from mirix.services.queue_trace_context import (
        get_agent_trace_id,
        get_memory_update_counts,
        get_queue_trace_id,
        init_memory_update_counts,
        reset_agent_trace_id,
        reset_memory_update_counts,
        reset_parent_agent_trace_id,
        reset_queue_trace_id,
        set_agent_trace_id,
        set_parent_agent_trace_id,
        set_queue_trace_id,
    )

    # Validate that user_message is a dictionary
    if not isinstance(user_message, dict):
        raise TypeError(
            f"user_message must be a dictionary, got {type(user_message).__name__}: {user_message}"
        )

    # Map memory types to agent classes
    memory_type_to_agent_class = {
        "core": CoreMemoryAgent,
        "episodic": EpisodicMemoryAgent,
        "resource": ResourceMemoryAgent,
        "procedural": ProceduralMemoryAgent,
        "knowledge": KnowledgeMemoryAgent,
        "semantic": SemanticMemoryAgent,
    }

    # Validate memory types
    for memory_type in memory_types:
        if memory_type not in memory_type_to_agent_class:
            raise ValueError(
                f"Memory type '{memory_type}' is not supported. Please choose from 'core', 'episodic', 'resource', 'procedural', 'knowledge', 'semantic'."
            )

    # Get child agents
    child_agent_states = self.agent_manager.list_agents(parent_id=self.agent_state.id, actor=self.actor)

    # Map agent types to agent states
    agent_type_to_state = {
        agent_state.agent_type: agent_state for agent_state in child_agent_states
    }

    parent_trace_id = get_agent_trace_id()
    queue_trace_id = get_queue_trace_id()

    def _run_single_memory_update(memory_type: str) -> str:
        queue_token = None
        parent_token = None
        agent_trace_token = None
        counts_token = None
        agent_trace = None
        trace_success = False
        trace_error = None
        if queue_trace_id:
            queue_token = set_queue_trace_id(queue_trace_id)
        if parent_trace_id:
            parent_token = set_parent_agent_trace_id(parent_trace_id)
        agent_class = memory_type_to_agent_class[memory_type]
        agent_type_str = f"{memory_type}_memory_agent"

        try:
            agent_state = agent_type_to_state.get(agent_type_str)
            if agent_state is None:
                raise ValueError(f"No agent found with type '{agent_type_str}'")

            # Get filter_tags, use_cache, client_id, user_id, and occurred_at from parent agent instance
            # Deep copy filter_tags to ensure complete isolation between child agents
            parent_filter_tags = getattr(self, 'filter_tags', None)
            # Don't use 'or {}' because empty dict {} is valid and different from None
            filter_tags = deepcopy(parent_filter_tags) if parent_filter_tags is not None else None
            use_cache = getattr(self, 'use_cache', True)
            actor = getattr(self, 'actor', None)
            user = getattr(self, 'user', None)
            occurred_at = getattr(self, 'occurred_at', None)  # Get occurred_at from parent agent
            
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"🏷️  Creating {memory_type} agent with filter_tags={filter_tags}, client_id={actor.id if actor else None}, user_id={user.id if user else None}, occurred_at={occurred_at}")
            
            if queue_trace_id:
                trace_manager = MemoryAgentTraceManager()
                agent_trace = trace_manager.start_trace(
                    queue_trace_id=queue_trace_id,
                    parent_trace_id=parent_trace_id,
                    agent_state=agent_state,
                    actor=actor,
                )
                agent_trace_token = set_agent_trace_id(agent_trace.id)
                counts_token = init_memory_update_counts()

            memory_agent = agent_class(
                agent_state=agent_state,
                interface=self.interface,
                actor=actor,
                user=user,
                filter_tags=filter_tags,
                use_cache=use_cache,
            )
            
            # Set occurred_at on the child agent so it can use it during memory operations
            if occurred_at is not None:
                memory_agent.occurred_at = occurred_at

            # Work on a copy of the user message so parallel updates do not interfere
            if "message" not in user_message:
                raise KeyError("user_message must contain a 'message' field")

            if hasattr(user_message["message"], "model_copy"):
                message_copy = user_message["message"].model_copy(deep=True)  # type: ignore[attr-defined]
            else:
                message_copy = deepcopy(user_message["message"])

            system_msg = "[System Message] According to the instructions, the retrieved memories and the above content, update the corresponding memory."
            
            if not user_message["chaining"]:
                system_msg += " You are only being called ONCE. Please call all memory update tools IMMEDIATELY in the current response. Do not call search tools, update the memory directly according to the memory in the system prompt."

            system_msg = TextContent(text=system_msg)

            if isinstance(message_copy.content, str):
                message_copy.content = [TextContent(text=message_copy.content), system_msg]
            elif isinstance(message_copy.content, list):
                message_copy.content = list(message_copy.content) + [system_msg]
            else:
                message_copy.content = [system_msg]

            # Pass actor (Client) and user (User) to memory agent
            # actor is needed for write operations, user is needed for read operations
            memory_agent.step(
                input_messages=message_copy,
                chaining=user_message.get("chaining", False),
                actor=actor,  # Client for write operations
                user=user,  # User for read operations
            )

            trace_success = True
            return (
                f"[System Message] Agent {agent_state.name} has been triggered to update the memory.\n"
            )
        except Exception as exc:
            trace_error = str(exc)
            raise
        finally:
            if agent_trace:
                counts = get_memory_update_counts()
                trace_manager.finish_trace(
                    agent_trace.id,
                    success=trace_success,
                    error_message=trace_error,
                    memory_update_counts=counts,
                    actor=actor,
                )
            if counts_token:
                reset_memory_update_counts(counts_token)
            if agent_trace_token:
                reset_agent_trace_id(agent_trace_token)
            if parent_token:
                reset_parent_agent_trace_id(parent_token)
            if queue_token:
                reset_queue_trace_id(queue_token)

    max_workers = min(len(memory_types), max(os.cpu_count() or 1, 1))
    responses: dict[int, str] = {}

    if not memory_types:
        return ""

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_run_single_memory_update, memory_type): index
            for index, memory_type in enumerate(memory_types)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            memory_type = memory_types[index]
            try:
                responses[index] = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to trigger memory update for '{memory_type}'"
                ) from exc

    ordered_responses = [responses[i] for i in range(len(memory_types)) if i in responses]
    return "".join(ordered_responses).strip()


def finish_memory_update(self: "Agent"):
    """
    Finish the memory update process. This function should be called after the Memory is updated.
    
    Note: This function takes no parameters. Call it without any arguments.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    return None
