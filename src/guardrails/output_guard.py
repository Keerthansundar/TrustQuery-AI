"""Output guardrail: Masks PII in row sets, sanitizes responses for leaked secrets, and caps displayed rows."""

import logging
import re
from typing import Any

from src.config import settings
from src.guardrails.input_guard import mask_pii

logger = logging.getLogger(__name__)

# Patterns for leaked sensitive information in LLM text responses
LEAK_PATTERNS = [
    (re.compile(r"AIza[0-9A-Za-z-_]{35}"), "[REDACTED_API_KEY]"),
    (re.compile(r"postgres(?:ql)?://[^\s]+", re.IGNORECASE), "[REDACTED_DATABASE_DSN]"),
    (re.compile(r"password\s*=\s*['\"][^'\"]+['\"]", re.IGNORECASE), "password=[REDACTED]"),
    (
        re.compile(r"You are a senior PostgreSQL 16 database engineer", re.IGNORECASE),
        "[REDACTED_PROMPT]",
    ),
    (re.compile(r"Strict PostgreSQL Rules:", re.IGNORECASE), "[REDACTED_PROMPT]"),
]


def sanitize_rows(
    rows: list[dict[str, Any]],
    allow_pii: bool | None = None,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Masks PII values in row dictionaries unless ALLOW_PII is enabled, and caps row count."""
    pii_allowed = allow_pii if allow_pii is not None else settings.ALLOW_PII
    row_cap = max_rows or settings.MAX_ROWS

    capped_rows = rows[:row_cap]
    if pii_allowed:
        return capped_rows

    sanitized = []
    for row in capped_rows:
        clean_row = {}
        for key, val in row.items():
            k_lower = key.lower()
            if isinstance(val, str) and ("email" in k_lower or "phone" in k_lower):
                clean_row[key] = "[PII_PROTECTED]"
            elif isinstance(val, str):
                clean_row[key] = mask_pii(val)
            else:
                clean_row[key] = val
        sanitized.append(clean_row)

    return sanitized


def sanitize_answer(answer_text: str) -> str:
    """Scans and sanitizes final LLM explanation for leaked system prompts, keys, or DSNs."""
    if not answer_text:
        return ""

    sanitized = answer_text
    for pattern, replacement in LEAK_PATTERNS:
        if pattern.search(sanitized):
            logger.warning("Redacted sensitive content matching pattern '%s'", pattern.pattern)
            sanitized = pattern.sub(replacement, sanitized)

    return sanitized
