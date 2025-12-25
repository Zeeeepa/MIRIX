from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, TypeVar

from sqlalchemy import and_, desc, select

from mirix.log import get_logger
from mirix.orm.block import Block as BlockModel
from mirix.orm.episodic_memory import EpisodicEvent
from mirix.orm.knowledge import KnowledgeItem
from mirix.orm.procedural_memory import ProceduralMemoryItem
from mirix.orm.resource_memory import ResourceMemoryItem
from mirix.orm.semantic_memory import SemanticMemoryItem

logger = get_logger(__name__)

T = TypeVar("T")


def retrieve_memories_by_updated_at_range(
    *,
    server: Any,
    user: Any,
    start_time: Optional[datetime],
    end_time: datetime,
    limit: int = 100,
) -> Dict[str, Dict[str, Any]]:
    """
    Retrieve memories updated between start_time and end_time (inclusive of end_time).

    Notes:
    - Uses DB-level `updated_at` timestamps (not `last_modify` JSON timestamps).
    - Applies `limit` per memory type to avoid building unbounded prompts.
    """
    if end_time.tzinfo is None:
        raise ValueError("end_time must be timezone-aware")
    if start_time is not None and start_time.tzinfo is None:
        raise ValueError("start_time must be timezone-aware when provided")

    if limit <= 0:
        raise ValueError("limit must be a positive integer")

    core = _list_updated_between(
        session_maker=server.block_manager.session_maker,
        model=BlockModel,
        user=user,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
    )
    episodic = _list_updated_between(
        session_maker=server.episodic_memory_manager.session_maker,
        model=EpisodicEvent,
        user=user,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
    )
    semantic = _list_updated_between(
        session_maker=server.semantic_memory_manager.session_maker,
        model=SemanticMemoryItem,
        user=user,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
    )
    resource = _list_updated_between(
        session_maker=server.resource_memory_manager.session_maker,
        model=ResourceMemoryItem,
        user=user,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
    )
    knowledge = _list_updated_between(
        session_maker=server.knowledge_memory_manager.session_maker,
        model=KnowledgeItem,
        user=user,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
    )
    procedural = _list_updated_between(
        session_maker=server.procedural_memory_manager.session_maker,
        model=ProceduralMemoryItem,
        user=user,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
    )

    return {
        "core": {"total_count": len(core), "items": [_serialize_core_block(b) for b in core]},
        "episodic": {
            "total_count": len(episodic),
            "items": [_serialize_episodic_event(e) for e in episodic],
        },
        "semantic": {
            "total_count": len(semantic),
            "items": [_serialize_semantic_item(i) for i in semantic],
        },
        "resource": {
            "total_count": len(resource),
            "items": [_serialize_resource_item(i) for i in resource],
        },
        "knowledge": {
            "total_count": len(knowledge),
            "items": [_serialize_knowledge_item(i) for i in knowledge],
        },
        "procedural": {
            "total_count": len(procedural),
            "items": [_serialize_procedural_item(i) for i in procedural],
        },
    }


def retrieve_specific_memories_by_ids(
    *,
    server: Any,
    user: Any,
    core_ids: Optional[List[str]] = None,
    episodic_ids: Optional[List[str]] = None,
    semantic_ids: Optional[List[str]] = None,
    resource_ids: Optional[List[str]] = None,
    knowledge_ids: Optional[List[str]] = None,
    procedural_ids: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Retrieve only the memories specified by ID lists.

    Missing IDs are ignored (no exception raised) to keep the endpoint ergonomic.
    """
    core_models = _list_by_ids(
        session_maker=server.block_manager.session_maker,
        model=BlockModel,
        user=user,
        ids=_dedupe_preserve_order(core_ids or []),
    )
    episodic_models = _list_by_ids(
        session_maker=server.episodic_memory_manager.session_maker,
        model=EpisodicEvent,
        user=user,
        ids=_dedupe_preserve_order(episodic_ids or []),
    )
    semantic_models = _list_by_ids(
        session_maker=server.semantic_memory_manager.session_maker,
        model=SemanticMemoryItem,
        user=user,
        ids=_dedupe_preserve_order(semantic_ids or []),
    )
    resource_models = _list_by_ids(
        session_maker=server.resource_memory_manager.session_maker,
        model=ResourceMemoryItem,
        user=user,
        ids=_dedupe_preserve_order(resource_ids or []),
    )
    knowledge_models = _list_by_ids(
        session_maker=server.knowledge_memory_manager.session_maker,
        model=KnowledgeItem,
        user=user,
        ids=_dedupe_preserve_order(knowledge_ids or []),
    )
    procedural_models = _list_by_ids(
        session_maker=server.procedural_memory_manager.session_maker,
        model=ProceduralMemoryItem,
        user=user,
        ids=_dedupe_preserve_order(procedural_ids or []),
    )

    return {
        "core": {
            "total_count": len(core_models),
            "items": [_serialize_core_block(b) for b in core_models],
        },
        "episodic": {
            "total_count": len(episodic_models),
            "items": [_serialize_episodic_event(e) for e in episodic_models],
        },
        "semantic": {
            "total_count": len(semantic_models),
            "items": [_serialize_semantic_item(i) for i in semantic_models],
        },
        "resource": {
            "total_count": len(resource_models),
            "items": [_serialize_resource_item(i) for i in resource_models],
        },
        "knowledge": {
            "total_count": len(knowledge_models),
            "items": [_serialize_knowledge_item(i) for i in knowledge_models],
        },
        "procedural": {
            "total_count": len(procedural_models),
            "items": [_serialize_procedural_item(i) for i in procedural_models],
        },
    }


def build_self_reflection_prompt(
    memories: Dict[str, Dict[str, Any]],
    *,
    is_targeted: bool = False,
) -> str:
    """
    Build a single prompt for the Reflexion Agent with memories in a fixed order:
    core, episodic, semantic, resource, knowledge, procedural.
    """
    header_lines = [
        "SELF-REFLECTION INPUT (Memory Optimization)",
        f"Mode: {'targeted' if is_targeted else 'time-based'}",
        "",
        "Instructions:",
        "- Review the memories below grouped by type.",
        "- Remove duplicates, merge redundancies, and fix inconsistencies using available memory tools.",
        "- Do not invent new facts; only reorganize/clean up what's provided.",
        "- If you need full text for truncated fields, use memory search/retrieval tools.",
        "- When finished, call `finish_memory_update`.",
    ]

    sections: List[str] = []
    for memory_type in [
        "core",
        "episodic",
        "semantic",
        "resource",
        "knowledge",
        "procedural",
    ]:
        data = memories.get(memory_type, {}) or {}
        items = data.get("items", []) or []
        sections.append(_format_section(memory_type=memory_type, items=items))

    return "\n".join(header_lines + [""] + sections).strip() + "\n"


# -------------------------
# Internal helpers
# -------------------------


def _dedupe_preserve_order(ids: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item_id in ids:
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        out.append(item_id)
    return out


def _dt_iso(ts: Optional[datetime]) -> Optional[str]:
    return ts.isoformat() if ts else None


def _truncate(text: Optional[str], max_chars: int) -> Optional[str]:
    if text is None:
        return None
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated, original_length={len(text)}]"


def _sort_key_updated_then_id(obj: Any) -> Tuple[datetime, str]:
    updated_at = getattr(obj, "updated_at", None) or getattr(obj, "created_at", None)
    if updated_at is None:
        updated_at = datetime.min.replace(tzinfo=timezone.utc)
    obj_id = getattr(obj, "id", "") or ""
    return (updated_at, obj_id)


def _list_updated_between(
    *,
    session_maker: Any,
    model: Type[T],
    user: Any,
    start_time: Optional[datetime],
    end_time: datetime,
    limit: int,
) -> List[T]:
    conditions = [model.user_id == user.id]
    if hasattr(model, "organization_id") and getattr(user, "organization_id", None):
        conditions.append(model.organization_id == user.organization_id)
    if start_time is not None:
        conditions.append(model.updated_at > start_time)
    conditions.append(model.updated_at <= end_time)

    query = (
        select(model)
        .where(and_(*conditions))
        .order_by(desc(model.updated_at), desc(model.created_at), desc(model.id))
        .limit(limit)
    )

    with session_maker() as session:
        results = session.execute(query).scalars().all()

    # Readability: chronological order in the prompt
    results.sort(key=_sort_key_updated_then_id)
    return results


def _list_by_ids(
    *,
    session_maker: Any,
    model: Type[T],
    user: Any,
    ids: Sequence[str],
) -> List[T]:
    if not ids:
        return []

    conditions = [model.id.in_(list(ids)), model.user_id == user.id]
    if hasattr(model, "organization_id") and getattr(user, "organization_id", None):
        conditions.append(model.organization_id == user.organization_id)

    with session_maker() as session:
        rows = session.execute(select(model).where(and_(*conditions))).scalars().all()

    by_id: Dict[str, T] = {getattr(row, "id"): row for row in rows}
    missing = [item_id for item_id in ids if item_id not in by_id]
    if missing:
        logger.info("Self-reflection: %s missing IDs for %s", len(missing), model.__name__)
    ordered = [by_id[item_id] for item_id in ids if item_id in by_id]
    ordered.sort(key=_sort_key_updated_then_id)
    return ordered


def _serialize_core_block(block: BlockModel) -> Dict[str, Any]:
    return {
        "id": block.id,
        "label": getattr(block, "label", None),
        "value": getattr(block, "value", None),
        "updated_at": _dt_iso(getattr(block, "updated_at", None)),
    }


def _serialize_episodic_event(event: EpisodicEvent) -> Dict[str, Any]:
    return {
        "id": event.id,
        "occurred_at": _dt_iso(getattr(event, "occurred_at", None)),
        "updated_at": _dt_iso(getattr(event, "updated_at", None)),
        "actor": getattr(event, "actor", None),
        "event_type": getattr(event, "event_type", None),
        "summary": getattr(event, "summary", None),
        "details": getattr(event, "details", None),
        "last_modify": getattr(event, "last_modify", None),
    }


def _serialize_semantic_item(item: SemanticMemoryItem) -> Dict[str, Any]:
    return {
        "id": item.id,
        "updated_at": _dt_iso(getattr(item, "updated_at", None)),
        "name": getattr(item, "name", None),
        "summary": getattr(item, "summary", None),
        "details": getattr(item, "details", None),
        "source": getattr(item, "source", None),
        "last_modify": getattr(item, "last_modify", None),
    }


def _serialize_resource_item(item: ResourceMemoryItem) -> Dict[str, Any]:
    return {
        "id": item.id,
        "updated_at": _dt_iso(getattr(item, "updated_at", None)),
        "title": getattr(item, "title", None),
        "summary": getattr(item, "summary", None),
        "resource_type": getattr(item, "resource_type", None),
        "content": getattr(item, "content", None),
        "last_modify": getattr(item, "last_modify", None),
    }


def _serialize_knowledge_item(item: KnowledgeItem) -> Dict[str, Any]:
    return {
        "id": item.id,
        "updated_at": _dt_iso(getattr(item, "updated_at", None)),
        "entry_type": getattr(item, "entry_type", None),
        "caption": getattr(item, "caption", None),
        "source": getattr(item, "source", None),
        "sensitivity": getattr(item, "sensitivity", None),
        "last_modify": getattr(item, "last_modify", None),
    }


def _serialize_procedural_item(item: ProceduralMemoryItem) -> Dict[str, Any]:
    return {
        "id": item.id,
        "updated_at": _dt_iso(getattr(item, "updated_at", None)),
        "entry_type": getattr(item, "entry_type", None),
        "summary": getattr(item, "summary", None),
        "steps": getattr(item, "steps", None),
        "last_modify": getattr(item, "last_modify", None),
    }


def _format_section(*, memory_type: str, items: Sequence[Dict[str, Any]]) -> str:
    title = memory_type.replace("_", " ").title()
    lines: List[str] = [f"=== {title} ({len(items)} items) ==="]
    if not items:
        lines.append("(none)")
        return "\n".join(lines)

    # Keep the prompt from exploding (especially resource content / long details).
    limits = {
        "core": {"value": 4000},
        "episodic": {"details": 3000, "summary": 500},
        "semantic": {"details": 3000, "summary": 800},
        "resource": {"content": 3000, "summary": 800},
        "knowledge": {"caption": 800},
        "procedural": {"summary": 800},
    }
    per_type_limits = limits.get(memory_type, {})

    for item in items:
        item_id = item.get("id")
        lines.append(f"- id: {item_id}")

        if item.get("updated_at"):
            lines.append(f"  updated_at: {item.get('updated_at')}")

        if memory_type == "core":
            lines.append(f"  label: {item.get('label')}")
            value = _truncate(item.get("value"), per_type_limits.get("value", 4000))
            lines.append("  value:")
            for vline in (value or "").splitlines() or [""]:
                lines.append(f"    {vline}")

        elif memory_type == "episodic":
            lines.append(f"  occurred_at: {item.get('occurred_at')}")
            lines.append(f"  actor: {item.get('actor')}")
            lines.append(f"  event_type: {item.get('event_type')}")
            summary = _truncate(item.get("summary"), per_type_limits.get("summary", 500))
            details = _truncate(item.get("details"), per_type_limits.get("details", 3000))
            lines.append(f"  summary: {summary}")
            lines.append("  details:")
            for dline in (details or "").splitlines() or [""]:
                lines.append(f"    {dline}")

        elif memory_type == "semantic":
            lines.append(f"  name: {item.get('name')}")
            lines.append(f"  source: {item.get('source')}")
            summary = _truncate(item.get("summary"), per_type_limits.get("summary", 800))
            details = _truncate(item.get("details"), per_type_limits.get("details", 3000))
            lines.append(f"  summary: {summary}")
            lines.append("  details:")
            for dline in (details or "").splitlines() or [""]:
                lines.append(f"    {dline}")

        elif memory_type == "resource":
            lines.append(f"  title: {item.get('title')}")
            lines.append(f"  resource_type: {item.get('resource_type')}")
            summary = _truncate(item.get("summary"), per_type_limits.get("summary", 800))
            content = _truncate(item.get("content"), per_type_limits.get("content", 3000))
            lines.append(f"  summary: {summary}")
            lines.append("  content:")
            for cline in (content or "").splitlines() or [""]:
                lines.append(f"    {cline}")

        elif memory_type == "knowledge":
            lines.append(f"  entry_type: {item.get('entry_type')}")
            lines.append(f"  sensitivity: {item.get('sensitivity')}")
            lines.append(f"  source: {item.get('source')}")
            caption = _truncate(item.get("caption"), per_type_limits.get("caption", 800))
            lines.append(f"  caption: {caption}")

        elif memory_type == "procedural":
            lines.append(f"  entry_type: {item.get('entry_type')}")
            summary = _truncate(item.get("summary"), per_type_limits.get("summary", 800))
            lines.append(f"  summary: {summary}")
            steps = item.get("steps") or []
            if isinstance(steps, list):
                lines.append("  steps:")
                for step in steps[:50]:
                    lines.append(f"    - {_truncate(str(step), 300)}")
                if len(steps) > 50:
                    lines.append(f"    - ...[truncated, total_steps={len(steps)}]")
            else:
                lines.append(f"  steps: {steps}")

        lines.append("")

    return "\n".join(lines).rstrip()

