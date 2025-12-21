import datetime as dt
from datetime import datetime
from typing import Optional

from mirix.orm.memory_agent_tool_call import MemoryAgentToolCall
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.memory_agent_tool_call import (
    MemoryAgentToolCall as PydanticMemoryAgentToolCall,
)
from mirix.utils import generate_unique_short_id


class MemoryAgentToolCallTraceManager:
    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    def start_tool_call(
        self,
        agent_trace_id: str,
        function_name: str,
        function_args: Optional[dict],
        tool_call_id: Optional[str] = None,
        actor: Optional[PydanticClient] = None,
    ) -> PydanticMemoryAgentToolCall:
        trace_id = generate_unique_short_id(
            self.session_maker, MemoryAgentToolCall, "mtc"
        )
        trace = MemoryAgentToolCall(
            id=trace_id,
            agent_trace_id=agent_trace_id,
            tool_call_id=tool_call_id,
            function_name=function_name,
            function_args=function_args,
            status="running",
            started_at=datetime.now(dt.timezone.utc),
        )
        with self.session_maker() as session:
            trace.create(session, actor=actor)
            return trace.to_pydantic()

    def finish_tool_call(
        self,
        trace_id: str,
        success: bool,
        response_text: Optional[str] = None,
        error_message: Optional[str] = None,
        actor: Optional[PydanticClient] = None,
    ) -> None:
        with self.session_maker() as session:
            trace = session.get(MemoryAgentToolCall, trace_id)
            if not trace:
                return
            trace.status = "completed" if success else "failed"
            trace.success = success
            trace.response_text = response_text
            trace.error_message = error_message
            trace.completed_at = datetime.now(dt.timezone.utc)
            trace.update(session, actor=actor)
