"""Rewriter agent: Resolves conversational references and expands query for retrieval."""

import logging
from typing import Any

from src.agents.planner import format_history_for_prompt
from src.config import settings
from src.llm.llm_client import openai_client
from src.prompts import (
    RETRIEVAL_EXPANSION_TEMPLATE,
    REWRITER_SYSTEM_PROMPT,
    REWRITER_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)

# Heuristics to detect if question is already self-contained
FOLLOWUP_TRIGGERS = (
    "what about",
    "how about",
    "and for",
    "and by",
    "what if",
    "filter by",
    "show them",
    "same for",
    "last month",
    "last quarter",
    "that",
    "those",
    "them",
    "these",
    "instead",
    "also",
)


def is_self_contained(question: str, history: list[Any] | None) -> bool:
    """Heuristic check: if there is no history or no followup markers, it is self-contained."""
    if not history:
        return True

    q_lower = question.strip().lower()
    # If the question contains typical followup reference keywords, it is not self-contained
    for trigger in FOLLOWUP_TRIGGERS:
        if trigger in q_lower:
            return False

    # If it is a long question (> 12 words) and starts with interrogatives, likely self-contained
    words = q_lower.split()
    if len(words) >= 12 and words[0] in (
        "what",
        "which",
        "how",
        "list",
        "show",
        "calculate",
        "find",
    ):
        return True

    return False


def rewrite(question: str, history: list[Any] | None = None) -> str:
    """Rewrites user query using prior conversation history into a standalone query.

    Skips rewriting if the query is already self-contained or if no history exists.
    """
    if is_self_contained(question, history):
        logger.info("Question is already self-contained; skipping rewrite.")
        return question.strip()

    hist_text = format_history_for_prompt(history)
    user_prompt = REWRITER_USER_TEMPLATE.format(history=hist_text, question=question)

    try:
        resp = openai_client.generate(
            prompt=user_prompt,
            model=settings.FAST_MODEL,
            temperature=0.0,
            system=REWRITER_SYSTEM_PROMPT,
        )
        rewritten = resp.text.strip().strip('"').strip("'")
        if rewritten:
            logger.info("Rewrote question from '%s' to '%s'", question, rewritten)
            return rewritten
    except Exception as exc:
        logger.warning("Rewrite call failed (%s); using original question.", exc)

    return question.strip()


def rewrite_for_retrieval(standalone_question: str) -> str:
    """Expands standalone question with database schema keywords and glossary terms for retrieval."""
    prompt = RETRIEVAL_EXPANSION_TEMPLATE.format(question=standalone_question)
    try:
        resp = openai_client.generate(
            prompt=prompt,
            model=settings.FAST_MODEL,
            temperature=0.0,
        )
        expanded = resp.text.strip()
        if expanded:
            return expanded
    except Exception as exc:
        logger.warning("Retrieval expansion failed (%s); using standalone question.", exc)

    return standalone_question
