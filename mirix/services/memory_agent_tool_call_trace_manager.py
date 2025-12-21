import datetime as dt
from datetime import datetime
from typing import Optional

from sqlalchemy import text

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
        self._ensure_usage_columns()

    def _ensure_usage_columns(self) -> None:
        with self.session_maker() as session:
            bind = session.get_bind()
            if not bind or bind.dialect.name != "sqlite":
                return
            results = session.execute(text("PRAGMA table_info(memory_agent_tool_calls)"))
            existing = {row[1] for row in results.fetchall()}
            statements = []
            if "llm_call_id" not in existing:
                statements.append(
                    "ALTER TABLE memory_agent_tool_calls ADD COLUMN llm_call_id VARCHAR"
                )
            if "prompt_tokens" not in existing:
                statements.append(
                    "ALTER TABLE memory_agent_tool_calls ADD COLUMN prompt_tokens INTEGER"
                )
            if "completion_tokens" not in existing:
                statements.append(
                    "ALTER TABLE memory_agent_tool_calls ADD COLUMN completion_tokens INTEGER"
                )
            if "cached_tokens" not in existing:
                statements.append(
                    "ALTER TABLE memory_agent_tool_calls ADD COLUMN cached_tokens INTEGER"
                )
            if "total_tokens" not in existing:
                statements.append(
                    "ALTER TABLE memory_agent_tool_calls ADD COLUMN total_tokens INTEGER"
                )
            if "credit_cost" not in existing:
                statements.append(
                    "ALTER TABLE memory_agent_tool_calls ADD COLUMN credit_cost FLOAT"
                )
            if statements:
                for statement in statements:
                    session.execute(text(statement))
                session.commit()

    def start_tool_call(
        self,
        agent_trace_id: str,
        function_name: str,
        function_args: Optional[dict],
        tool_call_id: Optional[str] = None,
        llm_call_id: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        cached_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        credit_cost: Optional[float] = None,
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
            llm_call_id=llm_call_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            total_tokens=total_tokens,
            credit_cost=credit_cost,
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
        llm_call_id: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        cached_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        credit_cost: Optional[float] = None,
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
            if llm_call_id is not None:
                trace.llm_call_id = llm_call_id
            if prompt_tokens is not None:
                trace.prompt_tokens = prompt_tokens
            if completion_tokens is not None:
                trace.completion_tokens = completion_tokens
            if cached_tokens is not None:
                trace.cached_tokens = cached_tokens
            if total_tokens is not None:
                trace.total_tokens = total_tokens
            if credit_cost is not None:
                trace.credit_cost = credit_cost
            trace.completed_at = datetime.now(dt.timezone.utc)
            trace.update(session, actor=actor)
