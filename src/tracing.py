"""JSONL execution trace logging for auditing, debugging, and evaluation."""

import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from src.state import PipelineTraceStep

logger = logging.getLogger(__name__)

# Log directory
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


def log_trace_to_file(trace_id: str, session_id: str, steps: list[PipelineTraceStep]) -> None:
    """Appends full pipeline trace to JSONL file for evaluation and audit."""
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOGS_DIR / "traces.jsonl"
        record = {
            "trace_id": trace_id,
            "session_id": session_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "steps": [asdict(s) for s in steps],
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning("Failed to write trace to file: %s", exc)
