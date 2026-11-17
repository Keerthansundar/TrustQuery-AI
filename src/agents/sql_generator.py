"""SQL Generator agent: Generates PostgreSQL 16 queries using schema, glossary, and rules."""

import logging
import re
from typing import Any

from src.config import settings
from src.llm.llm_client import LLMResponse, openai_client
from src.prompts import (
    SQL_ERROR_RETRY_TEMPLATE,
    SQL_GENERATOR_SYSTEM_PROMPT,
    SQL_GENERATOR_USER_TEMPLATE,
)
from src.retrieval.indexer import DocumentChunk

logger = logging.getLogger(__name__)


def clean_sql_output(raw_sql: str) -> str:
    """Strips markdown code fences (```sql or ```), leading/trailing prose,
    and returns clean PostgreSQL query text.
    """
    text = raw_sql.strip()

    # 1. Search for any markdown fenced block: ```sql ... ``` or ``` ... ```
    fence_pattern = re.compile(
        r"```(?:sql|postgresql|postgres)?\s*(.*?)\s*```",
        flags=re.DOTALL | re.IGNORECASE,
    )
    fence_match = fence_pattern.search(text)
    if fence_match:
        text = fence_match.group(1).strip()
    else:
        # If there's an unclosed opening or trailing fence
        text = re.sub(r"^```(?:sql|postgresql|postgres)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    # 2. Extract starting from the root SELECT or WITH keyword
    root_match = re.search(r"\b(SELECT|WITH)\b", text, re.IGNORECASE)
    if root_match:
        text = text[root_match.start() :]

    # 3. If there is trailing prose after a terminating semicolon, truncate to the statement
    semicolon_pos = text.find(";")
    if semicolon_pos != -1:
        remainder = text[semicolon_pos + 1 :].strip()
        if remainder and not re.search(
            r"\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b", remainder, re.IGNORECASE
        ):
            text = text[: semicolon_pos + 1].strip()

    return text.strip()




def generate_sql(
    standalone_question: str,
    context_chunks: list[DocumentChunk],
    history: list[Any] | None = None,
    error: str | None = None,
    previous_sql: str | None = None,
) -> tuple[str, LLMResponse]:
    """Generates PostgreSQL 16 query for the standalone question given schema context."""
    # Build formatted context
    table_docs = [c.content for c in context_chunks if c.doc_type == "table"]
    glossary_docs = [c.content for c in context_chunks if c.doc_type == "glossary"]
    example_docs = [c.content for c in context_chunks if c.doc_type == "example"]

    context_parts = []
    if table_docs:
        context_parts.append("### Tables & Columns:\n" + "\n\n".join(table_docs))
    if glossary_docs:
        context_parts.append("### Business Glossary & Metrics:\n" + "\n\n".join(glossary_docs))
    if example_docs:
        context_parts.append("### Few-Shot Query Examples:\n" + "\n\n".join(example_docs[:4]))

    formatted_context = "\n\n".join(context_parts)

    # Format error retry context if this is a correction loop
    error_context = ""
    if error and previous_sql:
        error_context = SQL_ERROR_RETRY_TEMPLATE.format(
            previous_sql=previous_sql,
            error=error,
        )

    system_prompt = SQL_GENERATOR_SYSTEM_PROMPT.format(as_of_date=settings.AS_OF_DATE)
    user_prompt = SQL_GENERATOR_USER_TEMPLATE.format(
        context=formatted_context,
        as_of_date=settings.AS_OF_DATE,
        question=standalone_question,
        error_context=error_context,
    )

    resp = openai_client.generate(
        prompt=user_prompt,
        model=settings.GEN_MODEL,
        temperature=0.0,
        system=system_prompt,
    )

    cleaned_sql = clean_sql_output(resp.text)
    return cleaned_sql, resp
