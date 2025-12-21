import datetime as dt
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from mirix.orm.mixins import OrganizationMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.memory_agent_trace import MemoryAgentTrace as PydanticMemoryAgentTrace


class MemoryAgentTrace(SqlalchemyBase, OrganizationMixin):
    """
    Tracks a single agent run for a queued memory update request.
    """

    __tablename__ = "memory_agent_traces"
    __pydantic_model__ = PydanticMemoryAgentTrace

    id: Mapped[str] = mapped_column(String, primary_key=True)

    queue_trace_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("memory_queue_traces.id", ondelete="CASCADE"), nullable=True
    )
    parent_trace_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("memory_agent_traces.id", ondelete="SET NULL"), nullable=True
    )

    agent_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("agents.id", ondelete="SET NULL"), nullable=True
    )
    agent_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    agent_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)

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
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    assistant_messages: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    triggered_memory_types: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    memory_update_counts: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
