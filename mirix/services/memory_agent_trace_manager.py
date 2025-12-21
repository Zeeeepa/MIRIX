import datetime as dt
from datetime import datetime
from typing import Dict, List, Optional

from mirix.orm.memory_agent_trace import MemoryAgentTrace
from mirix.schemas.agent import AgentState
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.memory_agent_trace import MemoryAgentTrace as PydanticMemoryAgentTrace
from mirix.utils import generate_unique_short_id


class MemoryAgentTraceManager:
    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    def start_trace(
        self,
        queue_trace_id: Optional[str],
        parent_trace_id: Optional[str],
        agent_state: AgentState,
        actor: Optional[PydanticClient],
    ) -> PydanticMemoryAgentTrace:
        trace_id = generate_unique_short_id(
            self.session_maker, MemoryAgentTrace, "mat"
        )
        trace = MemoryAgentTrace(
            id=trace_id,
            queue_trace_id=queue_trace_id,
            parent_trace_id=parent_trace_id,
            agent_id=agent_state.id,
            agent_type=agent_state.agent_type.value
            if hasattr(agent_state.agent_type, "value")
            else str(agent_state.agent_type),
            agent_name=agent_state.name,
            organization_id=actor.organization_id if actor else None,
            status="running",
            started_at=datetime.now(dt.timezone.utc),
        )
        with self.session_maker() as session:
            trace.create(session, actor=actor)
            return trace.to_pydantic()

    def finish_trace(
        self,
        trace_id: str,
        success: bool,
        error_message: Optional[str] = None,
        memory_update_counts: Optional[Dict[str, Dict[str, int]]] = None,
        actor: Optional[PydanticClient] = None,
    ) -> None:
        with self.session_maker() as session:
            trace = session.get(MemoryAgentTrace, trace_id)
            if not trace:
                return
            trace.status = "completed" if success else "failed"
            trace.success = success
            trace.error_message = error_message
            trace.completed_at = datetime.now(dt.timezone.utc)
            if memory_update_counts is not None:
                trace.memory_update_counts = memory_update_counts
            trace.update(session, actor=actor)

    def append_assistant_message(
        self,
        trace_id: str,
        content: Optional[str],
        reasoning_content: Optional[str],
        tool_calls: Optional[List[str]] = None,
        actor: Optional[PydanticClient] = None,
    ) -> None:
        if not content and not reasoning_content:
            return
        message_entry = {
            "timestamp": datetime.now(dt.timezone.utc).isoformat(),
            "content": content,
            "reasoning_content": reasoning_content,
            "tool_calls": tool_calls or [],
        }
        with self.session_maker() as session:
            trace = session.get(MemoryAgentTrace, trace_id)
            if not trace:
                return
            assistant_messages = trace.assistant_messages or []
            assistant_messages.append(message_entry)
            trace.assistant_messages = assistant_messages
            trace.update(session, actor=actor)

    def set_triggered_memory_types(
        self,
        trace_id: str,
        memory_types: Optional[List[str]],
        actor: Optional[PydanticClient] = None,
    ) -> None:
        if not memory_types:
            return
        with self.session_maker() as session:
            trace = session.get(MemoryAgentTrace, trace_id)
            if not trace:
                return
            trace.triggered_memory_types = list(memory_types)
            trace.update(session, actor=actor)
