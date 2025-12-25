from datetime import datetime
from typing import Dict, List, Optional

from pydantic import Field

from mirix.schemas.mirix_base import MirixBase


class MemoryQueueTraceBase(MirixBase):
    __id_prefix__ = "mqt"


class MemoryQueueTrace(MemoryQueueTraceBase):
    id: str = Field(..., description="Queue trace ID")

    organization_id: Optional[str] = Field(
        None, description="Organization that owns this trace"
    )
    client_id: Optional[str] = Field(
        None, description="Client that initiated the request"
    )
    user_id: Optional[str] = Field(
        None, description="End-user associated with the request"
    )
    agent_id: Optional[str] = Field(
        None, description="Agent that received the queued request"
    )

    status: str = Field(..., description="queued|processing|completed|failed")
    queued_at: datetime = Field(..., description="When the request was queued")
    started_at: Optional[datetime] = Field(
        None, description="When processing started"
    )
    completed_at: Optional[datetime] = Field(
        None, description="When processing completed"
    )
    interrupt_requested_at: Optional[datetime] = Field(
        None, description="When an interrupt was requested"
    )
    interrupt_reason: Optional[str] = Field(
        None, description="Reason for interruption request"
    )

    message_count: int = Field(0, description="Number of input messages queued")
    success: Optional[bool] = Field(
        None, description="Whether processing succeeded"
    )
    error_message: Optional[str] = Field(
        None, description="Error message if processing failed"
    )

    meta_agent_output: Optional[str] = Field(
        None, description="Assistant output from meta memory agent"
    )
    triggered_memory_types: Optional[List[str]] = Field(
        None, description="Memory manager types triggered by the meta agent"
    )
    memory_update_counts: Optional[Dict[str, Dict[str, int]]] = Field(
        None, description="Aggregated memory update counts per type and operation"
    )
