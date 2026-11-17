"""Responder agent: Plain language explanations, chitchat replies, clarification prompts, and result sanity checks."""

import json
import logging
from typing import Any

from src.agents.planner import format_history_for_prompt
from src.config import settings
from src.llm.llm_client import openai_client
from src.prompts import (
    CHITCHAT_SYSTEM_PROMPT,
    CLARIFYING_SYSTEM_PROMPT,
    EXPLAINER_SYSTEM_PROMPT,
    EXPLAINER_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)


def explain(
    standalone_question: str,
    sql: str,
    rows_summary: dict[str, Any] | list[Any],
) -> str:
    """Generates an executive, plain-language explanation of database results.

    States assumptions (e.g., net revenue, USD, as-of reference date) and never hallucinates numbers.
    """
    summary_text = json.dumps(rows_summary, default=str, indent=2)
    user_prompt = EXPLAINER_USER_TEMPLATE.format(
        question=standalone_question,
        sql=sql,
        results_summary=summary_text,
    )

    try:
        resp = openai_client.generate(
            prompt=user_prompt,
            model=settings.GEN_MODEL,
            temperature=0.2,
            system=EXPLAINER_SYSTEM_PROMPT,
        )
        return resp.text.strip()
    except Exception as exc:
        logger.warning("Explainer call failed (%s); providing deterministic summary.", exc)
        row_count = (
            rows_summary.get("row_count", 0)
            if isinstance(rows_summary, dict)
            else len(rows_summary)
        )
        return f"Executed query successfully. Found {row_count} matching records."


def reply_directly(question: str, history: list[Any] | None = None) -> str:
    """Replies directly to greetings and conversational chit-chat without database retrieval."""
    hist_text = format_history_for_prompt(history)
    user_prompt = f"Conversation History:\n{hist_text}\n\nUser Input: {question}"

    try:
        resp = openai_client.generate(
            prompt=user_prompt,
            model=settings.FAST_MODEL,
            temperature=0.4,
            system=CHITCHAT_SYSTEM_PROMPT,
        )
        return resp.text.strip()
    except Exception as exc:
        logger.warning("Chitchat reply failed (%s); returning default greeting.", exc)
        return "Hello! I am your AI Text-to-SQL Analyst. Ask me any business question regarding sales, customers, products, orders, payments, or shipments!"


def ask_clarifying_question(question: str, history: list[Any] | None = None) -> str:
    """Generates a concise clarifying question with 2-3 interpretations for ambiguous queries."""
    hist_text = format_history_for_prompt(history)
    user_prompt = f"Conversation History:\n{hist_text}\n\nUnclear User Query:\n{question}"

    try:
        resp = openai_client.generate(
            prompt=user_prompt,
            model=settings.FAST_MODEL,
            temperature=0.1,
            system=CLARIFYING_SYSTEM_PROMPT,
        )
        return resp.text.strip()
    except Exception as exc:
        logger.warning("Clarification generation failed (%s); returning default.", exc)
        return (
            "Could you clarify your question? For example, are you interested in "
            "total sales revenue, order volumes, or specific customer segments?"
        )


def sanity_check(rows: list[dict[str, Any]] | list[tuple[Any, ...]]) -> tuple[bool, str]:
    """Sanity checks database query results.

    Detects:
    - Empty results
    - All NULL columns in aggregates
    - Implausible values: negative revenue, percentages > 100

    Returns (is_sane, hint_message).
    """
    if not rows:
        return False, "Query returned 0 rows. Please verify join conditions and date filters."

    # Check if first row is dictionary or tuple
    first_row = rows[0]
    if isinstance(first_row, dict):
        # 1. Check all NULLs in aggregate
        if all(v is None for v in first_row.values()):
            return (
                False,
                "Query returned only NULL values. Check table joins and column aggregations.",
            )

        # 2. Check for negative monetary or revenue values
        for key, val in first_row.items():
            k_lower = key.lower()
            if any(term in k_lower for term in ("revenue", "sales", "amount", "price", "profit")):
                if isinstance(val, (int, float)) and val < 0 and "profit" not in k_lower:
                    return (
                        False,
                        f"Detected negative revenue/sales value ({val}) in column '{key}', which is implausible.",
                    )
            if "pct" in k_lower or "percent" in k_lower or "rate" in k_lower:
                if isinstance(val, (int, float)) and val > 100.0:
                    return False, f"Detected percentage greater than 100 ({val}) in column '{key}'."

    return True, ""
