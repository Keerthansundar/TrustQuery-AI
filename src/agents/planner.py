"""Planner agent: Classifies user query intent using FAST_MODEL."""

import logging
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from src.config import settings
from src.llm.llm_client import openai_client
from src.prompts import (
    COMBINED_PLAN_REWRITE_SYSTEM,
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)


class Intent(StrEnum):
    DATA_QUESTION = "data_question"
    FOLLOWUP = "followup"
    CHITCHAT = "chitchat"
    UNCLEAR = "unclear"


class PlanResult(BaseModel):
    intent: Intent = Field(
        description="Classified user intent: data_question, followup, chitchat, or unclear"
    )


class CombinedPlanRewriteResult(BaseModel):
    intent: Intent
    standalone_question: str = Field(
        min_length=1,
        description="A complete standalone question preserving all relevant constraints from conversation history."
    )


def extract_intent(parsed: Any) -> Intent:
    raw_intent = (
        parsed.get("intent", Intent.UNCLEAR)
        if isinstance(parsed, dict)
        else getattr(parsed, "intent", Intent.UNCLEAR)
    )

    try:
        return Intent(raw_intent)
    except (ValueError, TypeError):
        return Intent.UNCLEAR


def format_history_for_prompt(history: list[Any] | None) -> str:
    """Formats recent turns into readable dialogue context."""
    if not history:
        return "(No prior conversation turns)"

    lines = []
    for turn in history[-5:]:
        # Supports dict or object turn
        q = getattr(turn, "question", None) or (
            turn.get("question") if isinstance(turn, dict) else ""
        )
        ans = getattr(turn, "answer", None) or (
            turn.get("answer") if isinstance(turn, dict) else ""
        )
        sql = getattr(turn, "sql", None) or (turn.get("sql") if isinstance(turn, dict) else "")
        turn_str = f"User: {q}"
        if sql:
            turn_str += f"\nAssistant SQL: {sql}"
        if ans:
            turn_str += f"\nAssistant Answer: {ans[:150]}"
        lines.append(turn_str)
    return "\n\n".join(lines)


def plan(question: str, history: list[Any] | None = None) -> Intent:
    """Classifies user intent into data_question, followup, chitchat, or unclear."""

    hist_text = format_history_for_prompt(history)
    user_prompt = PLANNER_USER_TEMPLATE.format(
        history=hist_text,
        question=question,
    )

    try:
        parsed, _ = openai_client.generate_json(
            prompt=user_prompt,
            schema=PlanResult,
            model=settings.FAST_MODEL,
            temperature=0.0,
            system=PLANNER_SYSTEM_PROMPT,
        )

        return extract_intent(parsed)

    except Exception as exc:
        logger.warning(
            "Planner failed (%s); classifying as unclear.",
            exc,
        )
        return Intent.UNCLEAR


def plan_and_rewrite(
    question: str,
    history: list[Any] | None = None,
) -> tuple[Intent, str]:
    """Classifies intent and rewrites follow-ups into standalone questions."""

    hist_text = format_history_for_prompt(history)

    user_prompt = (
        f"Conversation History:\n{hist_text}\n\n"
        f"User Question:\n{question}"
    )

    try:
        parsed, _ = openai_client.generate_json(
            prompt=user_prompt,
            schema=CombinedPlanRewriteResult,
            model=settings.FAST_MODEL,
            temperature=0.0,
            system=COMBINED_PLAN_REWRITE_SYSTEM,
        )

        intent = extract_intent(parsed)

        standalone = (
            parsed.get("standalone_question", "")
            if isinstance(parsed, dict)
            else getattr(parsed, "standalone_question", "")
        )

        standalone = standalone.strip() if isinstance(standalone, str) else ""

        if not standalone:
            logger.warning(
                "Planner returned an empty standalone question."
            )
            return Intent.UNCLEAR, question

        logger.info(
            "Planner result | intent=%s | original=%s | standalone=%s",
            intent.value,
            question,
            standalone,
        )

        return intent, standalone

    except Exception as exc:
        logger.warning(
            "Combined plan and rewrite failed (%s); classifying as unclear.",
            exc,
        )
        return Intent.UNCLEAR, question
