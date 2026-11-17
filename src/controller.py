"""Central controller orchestrating Text-to-SQL Agentic RAG pipeline in plain Python."""

import logging
import time
import uuid
from typing import Any

from src.agents.planner import Intent, plan, plan_and_rewrite
from src.agents.responder import (
    ask_clarifying_question,
    explain,
    reply_directly,
    sanity_check,
)
from src.agents.rewriter import rewrite
from src.agents.sql_generator import generate_sql
from src.config import settings
from src.db import db_manager
from src.guardrails.input_guard import validate_input
from src.guardrails.output_guard import sanitize_answer, sanitize_rows
from src.guardrails.sql_guard import sql_guard
from src.llm.llm_client import LLMUnavailableError
from src.memory import load_history, log_query, save_turn, seed_history, summarize_rows
from src.retrieval.retriever import retriever
from src.state import PipelineState, PipelineTraceStep, Response
from src.tracing import log_trace_to_file

logger = logging.getLogger(__name__)


class Controller:
    """Orchestrates query classification, rewriting, retrieval, SQL generation,

    guardrails, and database execution with bounded retry loops.
    """

    def seed_history(self, session_id: str, history: list[dict[str, Any]]) -> None:
        """Seeds conversation history for a session before running evaluation turns."""
        seed_history(session_id, history)

    def handle(self, session_id: str, question: str) -> Response:
        """Processes a single conversational turn."""
        trace_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()

        state = PipelineState(
            trace_id=trace_id,
            session_id=session_id,
            question=question,
        )

        def add_trace(
            step_name: str,
            in_summary: str,
            out_summary: str,
            latency: float,
            model: str = "",
            key_label: str = "primary",
            fb_used: bool = False,
            err: str | None = None,
        ) -> None:
            step = PipelineTraceStep(
                step_name=step_name,
                input_summary=in_summary[:200],
                output_summary=out_summary[:200],
                latency_ms=round(latency, 2),
                model=model,
                key_label=key_label,
                fallback_used=fb_used,
                error=err,
            )
            state.trace.append(step)

        # 1. Load memory history (last 5 turns)
        t0 = time.perf_counter()
        history = load_history(session_id, n=5)
        add_trace(
            "load_history",
            f"session_id={session_id}",
            f"{len(history)} turns loaded",
            (time.perf_counter() - t0) * 1000,
        )

        # 2. Input Guardrail: check length, empty, prompt-injection regex
        t0 = time.perf_counter()
        guard_res = validate_input(question)
        add_trace(
            "input_guard", question, f"safe={guard_res.is_safe}", (time.perf_counter() - t0) * 1000
        )

        if not guard_res.is_safe:
            log_trace_to_file(trace_id, session_id, state.trace)
            return Response(
                answer=guard_res.error_message or "Invalid input.",
                sql="",
                rows=[],
                intent="security_blocked",
                rewritten_question=question,
                trace=state.trace,
                trace_id=trace_id,
                error=guard_res.error_message,
            )

        sanitized_question = guard_res.masked_text

        # 3. Intent Planning (and optional combined rewrite)
        t0 = time.perf_counter()
        try:
            if settings.COMBINE_PLAN_REWRITE:
                intent, standalone = plan_and_rewrite(sanitized_question, history)
            else:
                intent = plan(sanitized_question, history)
                standalone = rewrite(sanitized_question, history)

            # Debug planner and follow-up rewriting
            logger.info("ORIGINAL QUESTION: %s", sanitized_question)
            logger.info("PLANNER INTENT: %s", intent.value)
            logger.info("REWRITTEN QUESTION: %s", standalone)
            logger.info("HISTORY PROVIDED TO PLANNER: %s", len(history))

            
            state.intent = intent.value
            state.rewritten = standalone
            add_trace(
                "planner_rewriter",
                sanitized_question,
                f"intent={intent}, standalone={standalone}",
                (time.perf_counter() - t0) * 1000,
                model=settings.FAST_MODEL,
            )
        except LLMUnavailableError as e:
            add_trace(
                "planner",
                sanitized_question,
                "LLM unavailable",
                (time.perf_counter() - t0) * 1000,
                err=str(e),
            )
            return Response(
                answer="OpenAI API is currently unavailable or the API key is not configured. Please verify your OPENAI_API_KEY.",
                sql="",
                rows=[],
                intent="system_error",
                rewritten_question=sanitized_question,
                trace=state.trace,
                trace_id=trace_id,
                error=str(e),
            )

        # 4. Handle Chitchat (ZERO retrieval calls)
        if intent == Intent.CHITCHAT:
            t0 = time.perf_counter()
            chat_reply = reply_directly(sanitized_question, history)
            add_trace(
                "reply_directly",
                sanitized_question,
                chat_reply,
                (time.perf_counter() - t0) * 1000,
                model=settings.FAST_MODEL,
            )
            save_turn(session_id, question, sanitized_question, "", [], chat_reply)
            log_query(
                trace_id,
                session_id,
                "chitchat",
                sanitized_question,
                "",
                True,
                0,
                (time.perf_counter() - start_time) * 1000,
                False,
            )
            log_trace_to_file(trace_id, session_id, state.trace)
            return Response(
                answer=chat_reply,
                sql="",
                rows=[],
                intent=intent.value,
                rewritten_question=sanitized_question,
                trace=state.trace,
                trace_id=trace_id,
            )

        # 5. Handle Unclear intent
        if intent == Intent.UNCLEAR:
            t0 = time.perf_counter()
            clarify_msg = ask_clarifying_question(sanitized_question, history)
            add_trace(
                "ask_clarifying_question",
                sanitized_question,
                clarify_msg,
                (time.perf_counter() - t0) * 1000,
                model=settings.FAST_MODEL,
            )
            save_turn(session_id, question, sanitized_question, "", [], clarify_msg)
            log_query(
                trace_id,
                session_id,
                "unclear",
                sanitized_question,
                "",
                True,
                0,
                (time.perf_counter() - start_time) * 1000,
                False,
            )
            log_trace_to_file(trace_id, session_id, state.trace)
            return Response(
                answer=clarify_msg,
                sql="",
                rows=[],
                intent=intent.value,
                rewritten_question=sanitized_question,
                trace=state.trace,
                trace_id=trace_id,
            )

        # 6. Retrieve relevant schema, glossary, and few-shot examples
        t0 = time.perf_counter()
        context_chunks = retriever.retrieve(standalone)
        state.schema_ctx = context_chunks
        add_trace(
            "retrieval",
            standalone,
            f"{len(context_chunks)} chunks retrieved",
            (time.perf_counter() - t0) * 1000,
        )

        # 7. Generation and Self-Correction Loop (Bounded to MAX_ATTEMPTS)
        error: str | None = None
        current_sql = ""
        rows: list[dict[str, Any]] = []
        fb_used_overall = False

        for attempt in range(1, settings.MAX_ATTEMPTS + 1):
            state.attempts = attempt
            t0 = time.perf_counter()

            try:
                gen_sql, llm_resp = generate_sql(
                    standalone_question=standalone,
                    context_chunks=context_chunks,
                    history=history,
                    error=error,
                    previous_sql=current_sql if error else None,
                )
                current_sql = gen_sql
                fb_used_overall = fb_used_overall or llm_resp.fallback_used
                add_trace(
                    f"generate_sql_attempt_{attempt}",
                    f"error={error}",
                    current_sql,
                    llm_resp.latency_ms,
                    model=llm_resp.model,
                    key_label=llm_resp.key_label,
                    fb_used=llm_resp.fallback_used,
                )
            except LLMUnavailableError as e:
                add_trace(
                    f"generate_sql_attempt_{attempt}",
                    "generate",
                    "LLM unavailable",
                    (time.perf_counter() - t0) * 1000,
                    err=str(e),
                )
                log_trace_to_file(trace_id, session_id, state.trace)
                return Response(
                    answer="OpenAI API is currently unavailable or requests failed after retries.",
                    sql="",
                    rows=[],
                    intent=state.intent,
                    rewritten_question=standalone,
                    trace=state.trace,
                    trace_id=trace_id,
                    error=str(e),
                )

            # SQL Guardrail verification
            t_guard = time.perf_counter()
            ok, code, reason, sanitized_sql = sql_guard(current_sql)
            add_trace(
                f"sql_guard_attempt_{attempt}",
                current_sql,
                f"ok={ok}, code={code}",
                (time.perf_counter() - t_guard) * 1000,
            )

            if not ok:
                if code == "WRITE_ATTEMPT":
                    log_query(
                        trace_id,
                        session_id,
                        state.intent,
                        standalone,
                        current_sql,
                        False,
                        attempt,
                        (time.perf_counter() - start_time) * 1000,
                        fb_used_overall,
                        reason,
                    )
                    log_trace_to_file(trace_id, session_id, state.trace)
                    return Response(
                        answer="I can only run read-only queries.",
                        sql=current_sql,
                        rows=[],
                        intent=state.intent,
                        rewritten_question=standalone,
                        trace=state.trace,
                        trace_id=trace_id,
                        error="WRITE_ATTEMPT",
                    )
                # Fixable error (UNKNOWN_TABLE, UNKNOWN_COLUMN, FORBIDDEN_FUNCTION, etc.) -> retry
                error = reason
                continue

            current_sql = sanitized_sql
            state.sql = current_sql

            # Database execution with safety checks
            t_exec = time.perf_counter()
            try:
                rows = db_manager.run_query(current_sql)
                add_trace(
                    f"db_execution_attempt_{attempt}",
                    current_sql,
                    f"{len(rows)} rows returned",
                    (time.perf_counter() - t_exec) * 1000,
                )

                # Result sanity check
                is_sane, hint = sanity_check(rows)
                if not is_sane and attempt < settings.MAX_ATTEMPTS:
                    add_trace(
                        f"sanity_check_attempt_{attempt}",
                        f"{len(rows)} rows",
                        f"insane: {hint}",
                        1.0,
                    )
                    error = hint
                    continue

                # Query succeeded and passed sanity checks
                error = None
                break
            except Exception as db_exc:
                err_str = str(db_exc)
                add_trace(
                    f"db_execution_attempt_{attempt}",
                    current_sql,
                    "execution error",
                    (time.perf_counter() - t_exec) * 1000,
                    err=err_str,
                )
                error = err_str

        # If all attempts failed
        if error is not None:
            log_query(
                trace_id,
                session_id,
                state.intent,
                standalone,
                current_sql,
                False,
                state.attempts,
                (time.perf_counter() - start_time) * 1000,
                fb_used_overall,
                error,
            )
            log_trace_to_file(trace_id, session_id, state.trace)
            return Response(
                answer=(
                    f"I attempted to query the database {state.attempts} times, but encountered execution errors: {error}. "
                    "Please review the generated SQL or rephrase your question."
                ),
                sql=current_sql,
                rows=[],
                intent=state.intent,
                rewritten_question=standalone,
                trace=state.trace,
                trace_id=trace_id,
                error=error,
                schema_ctx=list(state.schema_ctx),
            )

        # 8. Output Guardrail: mask row PII and cap displayed rows
        t0 = time.perf_counter()
        safe_rows = sanitize_rows(rows)
        state.rows = safe_rows
        add_trace(
            "output_guard",
            f"{len(rows)} rows",
            f"{len(safe_rows)} safe rows",
            (time.perf_counter() - t0) * 1000,
        )

        # 9. Natural Language Explanation
        t0 = time.perf_counter()
        row_summary = summarize_rows(safe_rows)
        raw_answer = explain(standalone, current_sql, row_summary)
        safe_answer = sanitize_answer(raw_answer)
        state.answer = safe_answer
        add_trace(
            "explainer",
            current_sql,
            safe_answer[:100],
            (time.perf_counter() - t0) * 1000,
            model=settings.GEN_MODEL,
        )

        # 10. Persist to memory and audit log
        save_turn(session_id, question, standalone, current_sql, safe_rows, safe_answer)
        total_latency = (time.perf_counter() - start_time) * 1000
        log_query(
            trace_id,
            session_id,
            state.intent,
            standalone,
            current_sql,
            True,
            state.attempts,
            total_latency,
            fb_used_overall,
        )
        log_trace_to_file(trace_id, session_id, state.trace)

        return Response(
            answer=safe_answer,
            sql=current_sql,
            rows=safe_rows,
            intent=state.intent,
            rewritten_question=standalone,
            trace=state.trace,
            trace_id=trace_id,
            fallback_used=fb_used_overall,
            schema_ctx=list(state.schema_ctx),
        )


# Global controller instance
controller = Controller()
