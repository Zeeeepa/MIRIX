from datetime import datetime
from typing import Dict, Optional

from pydantic import Field

from mirix.schemas.mirix_base import MirixBase


class MemoryAgentToolCallBase(MirixBase):
    __id_prefix__ = "mtc"


class MemoryAgentToolCall(MemoryAgentToolCallBase):
    id: str = Field(..., description="Tool call trace ID")
    agent_trace_id: str = Field(..., description="Associated agent trace ID")

    tool_call_id: Optional[str] = Field(
        None, description="LLM tool_call_id (if provided)"
    )
    function_name: str = Field(..., description="Executed function/tool name")
    function_args: Optional[Dict] = Field(
        None, description="Arguments passed to the tool"
    )

    status: str = Field(..., description="running|completed|failed")
    started_at: datetime = Field(..., description="When tool execution started")
    completed_at: Optional[datetime] = Field(
        None, description="When tool execution completed"
    )
    success: Optional[bool] = Field(
        None, description="Whether the tool execution succeeded"
    )
    response_text: Optional[str] = Field(
        None, description="Returned response text from the tool"
    )
    error_message: Optional[str] = Field(
        None, description="Error message if tool execution failed"
    )
