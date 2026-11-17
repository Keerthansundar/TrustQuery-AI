"""Session memory and query audit logging backed by PostgreSQL schema 'app'."""

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import psycopg

from src.config import settings
from src.db import get_conninfo
from src.llm.llm_client import openai_client

logger = logging.getLogger(__name__)


@dataclass
class SessionTurn:
    session_id: str
    turn_idx: int
    ts: datetime
    question: str
    standalone: str
    sql: str
    result_summary: dict[str, Any]
    answer: str


# In-memory session store fallback when database is offline during unit testing
_IN_MEMORY_SESSIONS: dict[str, list[SessionTurn]] = {}
_IN_MEMORY_QUERY_LOG: list[dict[str, Any]] = []


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Generates a compact summary of query results: column list, row count, and first 3 rows."""
    if not rows:
        return {"columns": [], "row_count": 0, "sample_rows": []}

    columns = list(rows[0].keys())
    return {
        "columns": columns,
        "row_count": len(rows),
        "sample_rows": rows[:3],
    }


def load_history(session_id: str, n: int = 5) -> list[SessionTurn]:
    """Loads up to N most recent turns for a session from app.sessions.

    Beyond N turns, generates a rolling summary if needed and returns the last N turns.
    """
    try:
        conninfo = get_conninfo("app_rw")
        with psycopg.connect(conninfo, autocommit=True, timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT session_id, turn_idx, ts, question, standalone, sql, result_summary, answer
                    FROM app.sessions
                    WHERE session_id = %s
                    ORDER BY turn_idx ASC;
                    """,
                    (session_id,),
                )
                raw_rows = cur.fetchall()
                turns = []
                for r in raw_rows:
                    summary = r[6] if isinstance(r[6], dict) else (json.loads(r[6]) if r[6] else {})
                    turns.append(
                        SessionTurn(
                            session_id=r[0],
                            turn_idx=r[1],
                            ts=r[2],
                            question=r[3],
                            standalone=r[4] or "",
                            sql=r[5] or "",
                            result_summary=summary,
                            answer=r[7] or "",
                        )
                    )
                _IN_MEMORY_SESSIONS[session_id] = turns
                return turns[-n:] if len(turns) > n else turns
    except Exception as exc:
        logger.debug("Database unreachable for session load (%s); using in-memory store.", exc)
        turns = _IN_MEMORY_SESSIONS.get(session_id, [])
        return turns[-n:] if len(turns) > n else turns


def save_turn(
    session_id: str,
    question: str,
    standalone: str,
    sql: str,
    rows: list[dict[str, Any]],
    answer: str,
) -> SessionTurn:
    """Persists a new turn into app.sessions and updates in-memory fallback."""
    existing = _IN_MEMORY_SESSIONS.get(session_id, [])
    turn_idx = len(existing) + 1
    now = datetime.now(UTC)
    summary = summarize_rows(rows)

    turn = SessionTurn(
        session_id=session_id,
        turn_idx=turn_idx,
        ts=now,
        question=question,
        standalone=standalone,
        sql=sql,
        result_summary=summary,
        answer=answer,
    )

    if session_id not in _IN_MEMORY_SESSIONS:
        _IN_MEMORY_SESSIONS[session_id] = []
    _IN_MEMORY_SESSIONS[session_id].append(turn)

    try:
        conninfo = get_conninfo("app_rw")
        with psycopg.connect(conninfo, autocommit=True, timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO app.sessions (session_id, turn_idx, ts, question, standalone, sql, result_summary, answer)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (session_id, turn_idx) DO UPDATE SET
                        question = EXCLUDED.question,
                        standalone = EXCLUDED.standalone,
                        sql = EXCLUDED.sql,
                        result_summary = EXCLUDED.result_summary,
                        answer = EXCLUDED.answer;
                    """,
                    (
                        session_id,
                        turn_idx,
                        now,
                        question,
                        standalone,
                        sql,
                        json.dumps(summary),
                        answer,
                    ),
                )
    except Exception as exc:
        logger.debug("Database unreachable for session save (%s); stored in-memory.", exc)

    return turn


def clear_history(session_id: str) -> None:
    """Clears history for a session from both app.sessions and memory."""
    _IN_MEMORY_SESSIONS.pop(session_id, None)
    try:
        conninfo = get_conninfo("app_rw")
        with psycopg.connect(conninfo, autocommit=True, timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM app.sessions WHERE session_id = %s;", (session_id,))
    except Exception as exc:
        logger.debug("Database unreachable for session clear (%s); cleared in-memory.", exc)


def seed_history(session_id: str, history: list[dict[str, Any]]) -> None:
    """Seeds conversation history for a session with prior turns."""
    for item in history:
        save_turn(
            session_id=session_id,
            question=item.get("question", ""),
            standalone=item.get("question", ""),
            sql=item.get("sql", ""),
            rows=[],
            answer=item.get("answer", "Prior query executed."),
        )


def generate_rolling_summary(turns: list[SessionTurn]) -> str:
    """Generates rolling summary from FAST_MODEL for sessions exceeding 5 turns."""
    if len(turns) <= 5:
        return ""

    older_turns = turns[:-5]
    dialogue = "\n".join(
        f"Turn {t.turn_idx} Q: {t.question}\nSQL: {t.sql}\nAnswer: {t.answer[:120]}"
        for t in older_turns
    )
    prompt = (
        "Summarize the analytical context, queried entities, and findings from these earlier conversation turns:\n\n"
        f"{dialogue}\n\n"
        "Provide a concise bulleted summary of key filters, customer segments, product categories, or timeframes analyzed:"
    )
    try:
        resp = openai_client.generate(prompt, model=settings.FAST_MODEL, temperature=0.0)
        return resp.text.strip()
    except Exception as exc:
        logger.warning("Rolling summary generation failed (%s).", exc)
        return ""


def log_query(
    trace_id: str,
    session_id: str,
    intent: str,
    standalone: str,
    sql: str,
    ok: bool,
    attempts: int,
    latency_ms: float,
    fallback_used: bool,
    error: str | None = None,
) -> None:
    """Persists an execution audit record into app.query_log."""
    now = datetime.now(UTC)
    entry = {
        "trace_id": trace_id,
        "session_id": session_id,
        "ts": now,
        "intent": intent,
        "standalone": standalone,
        "sql": sql,
        "ok": ok,
        "attempts": attempts,
        "latency_ms": latency_ms,
        "fallback_used": fallback_used,
        "error": error,
    }
    _IN_MEMORY_QUERY_LOG.append(entry)

    try:
        conninfo = get_conninfo("app_rw")
        with psycopg.connect(conninfo, autocommit=True, timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO app.query_log (
                        trace_id, session_id, ts, intent, standalone, sql, ok, attempts, latency_ms, fallback_used, error
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (trace_id) DO NOTHING;
                    """,
                    (
                        trace_id,
                        session_id,
                        now,
                        intent,
                        standalone,
                        sql,
                        ok,
                        attempts,
                        latency_ms,
                        fallback_used,
                        error,
                    ),
                )
    except Exception as exc:
        logger.debug("Database unreachable for query_log insert (%s); stored in memory.", exc)
