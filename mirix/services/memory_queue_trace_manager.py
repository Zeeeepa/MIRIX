import datetime as dt
from datetime import datetime
from typing import Dict, Optional

from mirix.orm.memory_agent_trace import MemoryAgentTrace
from mirix.orm.memory_queue_trace import MemoryQueueTrace
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.memory_queue_trace import MemoryQueueTrace as PydanticMemoryQueueTrace
from mirix.utils import generate_unique_short_id


class MemoryQueueTraceManager:
    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    def create_trace(
        self,
        actor: Optional[PydanticClient],
        agent_id: Optional[str],
        user_id: Optional[str],
        message_count: int,
    ) -> PydanticMemoryQueueTrace:
        trace_id = generate_unique_short_id(
            self.session_maker, MemoryQueueTrace, "mqt"
        )
        trace = MemoryQueueTrace(
            id=trace_id,
            organization_id=actor.organization_id if actor else None,
            client_id=actor.id if actor else None,
            user_id=user_id,
            agent_id=agent_id,
            status="queued",
            queued_at=datetime.now(dt.timezone.utc),
            message_count=message_count,
        )
        with self.session_maker() as session:
            trace.create(session, actor=actor)
            return trace.to_pydantic()

    def mark_started(self, trace_id: str, actor: Optional[PydanticClient] = None) -> None:
        with self.session_maker() as session:
            trace = session.get(MemoryQueueTrace, trace_id)
            if not trace:
                return
            trace.status = "processing"
            trace.started_at = datetime.now(dt.timezone.utc)
            trace.update(session, actor=actor)

    def mark_completed(
        self,
        trace_id: str,
        success: bool,
        error_message: Optional[str] = None,
        memory_update_counts: Optional[Dict[str, Dict[str, int]]] = None,
        actor: Optional[PydanticClient] = None,
    ) -> None:
        with self.session_maker() as session:
            trace = session.get(MemoryQueueTrace, trace_id)
            if not trace:
                return
            trace.status = "completed" if success else "failed"
            trace.success = success
            trace.error_message = error_message
            trace.completed_at = datetime.now(dt.timezone.utc)
            if memory_update_counts is not None:
                trace.memory_update_counts = memory_update_counts
            trace.update(session, actor=actor)

    def set_meta_agent_output(
        self,
        trace_id: str,
        output_text: Optional[str],
        actor: Optional[PydanticClient] = None,
    ) -> None:
        if not output_text:
            return
        with self.session_maker() as session:
            trace = session.get(MemoryQueueTrace, trace_id)
            if not trace:
                return
            trace.meta_agent_output = output_text
            trace.update(session, actor=actor)

    def set_triggered_memory_types(
        self,
        trace_id: str,
        memory_types: Optional[list],
        actor: Optional[PydanticClient] = None,
    ) -> None:
        if not memory_types:
            return
        with self.session_maker() as session:
            trace = session.get(MemoryQueueTrace, trace_id)
            if not trace:
                return
            trace.triggered_memory_types = list(memory_types)
            trace.update(session, actor=actor)

    def aggregate_memory_update_counts(
        self, trace_id: str
    ) -> Dict[str, Dict[str, int]]:
        aggregated: Dict[str, Dict[str, int]] = {}
        with self.session_maker() as session:
            traces = session.query(MemoryAgentTrace).filter(
                MemoryAgentTrace.queue_trace_id == trace_id
            )
            for trace in traces:
                counts = trace.memory_update_counts or {}
                for memory_type, ops in counts.items():
                    agg_ops = aggregated.setdefault(memory_type, {})
                    for op_name, op_count in ops.items():
                        agg_ops[op_name] = agg_ops.get(op_name, 0) + op_count
        return aggregated
