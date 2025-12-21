from datetime import datetime
from typing import Dict, List, Optional

from pydantic import Field

from mirix.schemas.mirix_base import MirixBase


class MemoryAgentTraceBase(MirixBase):
    __id_prefix__ = "mat"


class AgentAssistantMessage(MirixBase):
    timestamp: str = Field(..., description="ISO timestamp for the assistant output")
    content: Optional[str] = Field(None, description="Assistant message content")
    reasoning_content: Optional[str] = Field(
        None, description="Model reasoning content if available"
    )
    tool_calls: List[str] = Field(
        default_factory=list, description="Tool call names in this step"
    )


class MemoryAgentTrace(MemoryAgentTraceBase):
    id: str = Field(..., description="Agent trace ID")

    queue_trace_id: Optional[str] = Field(
        None, description="Parent queue trace ID"
    )
    parent_trace_id: Optional[str] = Field(
        None, description="Parent agent trace ID (meta agent)"
    )

    agent_id: Optional[str] = Field(None, description="Agent ID")
    agent_type: Optional[str] = Field(None, description="Agent type string")
    agent_name: Optional[str] = Field(None, description="Agent name")

    status: str = Field(..., description="running|completed|failed")
    started_at: datetime = Field(..., description="When agent run started")
    completed_at: Optional[datetime] = Field(
        None, description="When agent run completed"
    )
    success: Optional[bool] = Field(
        None, description="Whether agent run succeeded"
    )
    error_message: Optional[str] = Field(
        None, description="Error message if agent run failed"
    )

    assistant_messages: Optional[List[AgentAssistantMessage]] = Field(
        None, description="Assistant outputs captured during the run"
    )
    triggered_memory_types: Optional[List[str]] = Field(
        None, description="Memory managers triggered by this agent run"
    )
    memory_update_counts: Optional[Dict[str, Dict[str, int]]] = Field(
        None, description="Memory update counts per type and operation"
    )
