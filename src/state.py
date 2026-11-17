"""State and trace dataclasses for the controller pipeline."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineTraceStep:
    """Record of a single processing step in the agentic pipeline."""

    step_name: str
    input_summary: str
    output_summary: str
    latency_ms: float = 0.0
    model: str = ""
    key_label: str = "primary"
    fallback_used: bool = False
    error: str | None = None


@dataclass
class PipelineState:
    """Mutable pipeline state passing through controller functions."""

    trace_id: str
    session_id: str
    question: str
    rewritten: str = ""
    intent: str = "data_question"
    schema_ctx: list[Any] = field(default_factory=list)
    sql: str = ""
    rows: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    attempts: int = 0
    answer: str = ""
    trace: list[PipelineTraceStep] = field(default_factory=list)


@dataclass
class Response:
    """Final user-facing response object."""

    answer: str
    sql: str
    rows: list[dict[str, Any]]
    intent: str
    rewritten_question: str
    trace: list[PipelineTraceStep]
    trace_id: str
    fallback_used: bool = False
    error: str | None = None
    schema_ctx: list[Any] = field(default_factory=list)
