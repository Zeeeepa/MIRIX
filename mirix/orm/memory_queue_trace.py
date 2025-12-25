import datetime as dt
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from mirix.orm.mixins import OrganizationMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.memory_queue_trace import MemoryQueueTrace as PydanticMemoryQueueTrace


class MemoryQueueTrace(SqlalchemyBase, OrganizationMixin):
    """
    Tracks a queued memory update request and its lifecycle.
    """

    __tablename__ = "memory_queue_traces"
    __pydantic_model__ = PydanticMemoryQueueTrace

    id: Mapped[str] = mapped_column(String, primary_key=True)

    client_id: Mapped[Optional[str]] = mapped_column(
        String, nullable=True
    )
    user_id: Mapped[Optional[str]] = mapped_column(
        String, nullable=True
    )
    agent_id: Mapped[Optional[str]] = mapped_column(
        String, nullable=True
    )

    status: Mapped[str] = mapped_column(
        String, default="queued", doc="queued|processing|completed|failed"
    )
    queued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(dt.timezone.utc)
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    interrupt_requested_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    interrupt_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    message_count: Mapped[int] = mapped_column(Integer, default=0)
    success: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    meta_agent_output: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    triggered_memory_types: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    memory_update_counts: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
