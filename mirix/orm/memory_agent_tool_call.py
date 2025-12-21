import datetime as dt
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.memory_agent_tool_call import (
    MemoryAgentToolCall as PydanticMemoryAgentToolCall,
)


class MemoryAgentToolCall(SqlalchemyBase):
    """
    Tracks tool/function calls executed by an agent run.
    """

    __tablename__ = "memory_agent_tool_calls"
    __pydantic_model__ = PydanticMemoryAgentToolCall

    id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_trace_id: Mapped[str] = mapped_column(
        String, ForeignKey("memory_agent_traces.id", ondelete="CASCADE")
    )

    tool_call_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    function_name: Mapped[str] = mapped_column(String)
    function_args: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    status: Mapped[str] = mapped_column(
        String, default="running", doc="running|completed|failed"
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(dt.timezone.utc)
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    success: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    response_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
