import contextvars
from typing import Dict, Optional

queue_trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "queue_trace_id", default=None
)
agent_trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "agent_trace_id", default=None
)
parent_agent_trace_id_var: contextvars.ContextVar[Optional[str]] = (
    contextvars.ContextVar("parent_agent_trace_id", default=None)
)
memory_update_counts_var: contextvars.ContextVar[Optional[Dict[str, Dict[str, int]]]] = (
    contextvars.ContextVar("memory_update_counts", default=None)
)


def set_queue_trace_id(trace_id: Optional[str]):
    return queue_trace_id_var.set(trace_id)


def reset_queue_trace_id(token):
    queue_trace_id_var.reset(token)


def get_queue_trace_id() -> Optional[str]:
    return queue_trace_id_var.get()


def set_agent_trace_id(trace_id: Optional[str]):
    return agent_trace_id_var.set(trace_id)


def reset_agent_trace_id(token):
    agent_trace_id_var.reset(token)


def get_agent_trace_id() -> Optional[str]:
    return agent_trace_id_var.get()


def set_parent_agent_trace_id(trace_id: Optional[str]):
    return parent_agent_trace_id_var.set(trace_id)


def reset_parent_agent_trace_id(token):
    parent_agent_trace_id_var.reset(token)


def get_parent_agent_trace_id() -> Optional[str]:
    return parent_agent_trace_id_var.get()


def init_memory_update_counts():
    return memory_update_counts_var.set({})


def reset_memory_update_counts(token):
    memory_update_counts_var.reset(token)


def get_memory_update_counts() -> Dict[str, Dict[str, int]]:
    return memory_update_counts_var.get() or {}


def increment_memory_update_count(
    memory_type: str, operation: str, count: int = 1
) -> None:
    counts = memory_update_counts_var.get()
    if counts is None:
        return
    type_counts = counts.setdefault(memory_type, {})
    type_counts[operation] = type_counts.get(operation, 0) + count
    memory_update_counts_var.set(counts)
