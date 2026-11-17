"""Streamlit Web Application for Text-to-SQL Analyst."""

import glob
import json
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from src.config import settings
from src.controller import controller
from src.db import db_manager
from src.llm.llm_client import get_token_usage_stats
from src.memory import clear_history

# Configure Streamlit page
st.set_page_config(
    page_title="Text-to-SQL Analyst | Agentic RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_schema_docs():
    schema_path = Path(__file__).resolve().parent / "data" / "schema_docs.yaml"
    if schema_path.exists():
        with open(schema_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


# Initialize Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Controls
with st.sidebar:
    st.title("🤖 Text-to-SQL Analyst")
    st.caption("Agentic RAG in Plain Python | PostgreSQL 16 + pgvector")

    # Status indicators
    st.subheader("System Status")
    db_healthy = db_manager.check_health()
    if db_healthy:
        st.success("🟢 PostgreSQL: Connected (127.0.0.1:5432)")
    else:
        st.warning("🟡 PostgreSQL: Offline (Mock/In-Memory Mode)")

    has_key = bool(
        settings.OPENAI_API_KEY
        and not settings.OPENAI_API_KEY.startswith("change_")
        and not settings.OPENAI_API_KEY.startswith("your-key")
    )
    if has_key:
        st.info(f"🔑 OpenAI Model: **{settings.GEN_MODEL}**")
    else:
        st.warning("⚠️ OPENAI_API_KEY is not configured")

    usage = get_token_usage_stats()
    if usage.total_calls > 0:
        st.caption(f"Spend: ~${usage.estimated_cost_usd:.4f} ({usage.total_tokens:,} tokens)")

    # Runtime Toggles
    st.subheader("Runtime Configuration")
    allow_pii = st.toggle("Allow PII Access (email, phone)", value=settings.ALLOW_PII)
    settings.ALLOW_PII = allow_pii

    schema_mode = st.selectbox(
        "Schema Injection Mode",
        options=["rag", "full"],
        index=0 if settings.SCHEMA_MODE == "rag" else 1,
        help="rag: 2-stage hybrid retrieval; full: injects full schema docs bypassing retrieval",
    )
    settings.SCHEMA_MODE = schema_mode

    rerank_enabled = st.toggle("LLM 2nd-Stage Rerank", value=settings.RERANK_ENABLED)
    settings.RERANK_ENABLED = rerank_enabled

    grading_enabled = st.toggle("Retrieval Grading", value=settings.GRADING_ENABLED)
    settings.GRADING_ENABLED = grading_enabled

    # Model and Business Info
    with st.expander("ℹ️ Models & Reference Dates"):
        st.write(f"**Gen Model:** `{settings.GEN_MODEL}`")
        st.write(f"**Fast Model:** `{settings.FAST_MODEL}`")
        st.write(f"**Embedding:** `{settings.EMBEDDING_MODEL}`")
        st.write(f"**As-Of Reference Date:** `{settings.AS_OF_DATE}`")

    # Sample Questions
    st.subheader("💡 Sample Questions")
    sample_queries = [
        "Top 5 products by revenue last quarter",
        "What was total net revenue by channel?",
        "Which customers have never placed any orders?",
        "What is the return rate for each customer segment?",
        "Calculate total profit by product brand",
        "How much cash has been collected by each payment method?",
    ]
    for sq in sample_queries:
        if st.button(sq, key=f"btn_{sq}", use_container_width=True):
            st.session_state.pending_query = sq

    # Clear Memory Button
    if st.button("🧹 Clear Conversation Memory", use_container_width=True):
        clear_history(st.session_state.session_id)
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    # Schema Browser
    st.subheader("📚 Schema Browser")
    schema_data = get_schema_docs()
    tables = schema_data.get("tables", {})
    for tname, tinfo in tables.items():
        with st.expander(f"Table: public.{tname}"):
            st.caption(tinfo.get("description", ""))
            cols = tinfo.get("columns", {})
            for c, cdesc in cols.items():
                st.markdown(f"- **`{c}`**: {cdesc}")

# Main Tabs Layout
tab_chat, tab_eval = st.tabs(["💬 Analyst Chat", "📊 Evaluation & Benchmarks"])

# ------------------------------------------------------------------------------
# TAB 1: Chat Interface
# ------------------------------------------------------------------------------
with tab_chat:
    col_hdr1, col_hdr2 = st.columns([4, 1])
    with col_hdr1:
        st.subheader("Conversational Business Intelligence")
    with col_hdr2:
        if st.button("➕ New Chat", use_container_width=True):
            clear_history(st.session_state.session_id)
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.rerun()

    # Display message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sql"):
                with st.expander("🔍 View Generated PostgreSQL", expanded=False):
                    st.code(msg["sql"], language="sql")
            if msg.get("rows"):
                df = pd.DataFrame(msg["rows"])
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name=f"query_result_{st.session_state.session_id}.csv",
                    mime="text/csv",
                    key=f"dl_{msg.get('trace_id', uuid.uuid4())}",
                )
            if msg.get("trace"):
                with st.expander("⚙️ How I got this (Agent Execution Trace)", expanded=False):
                    st.write(f"**Intent Classified:** `{msg.get('intent')}`")
                    st.write(f"**Rewritten Standalone Question:** {msg.get('rewritten')}")
                    trace_df = pd.DataFrame(
                        [
                            {
                                "Step": t.get("step_name") if isinstance(t, dict) else t.step_name,
                                "Latency (ms)": t.get("latency_ms")
                                if isinstance(t, dict)
                                else t.latency_ms,
                                "Model": t.get("model") if isinstance(t, dict) else t.model,
                                "Key": t.get("key_label") if isinstance(t, dict) else t.key_label,
                                "Output Summary": (
                                    t.get("output_summary")
                                    if isinstance(t, dict)
                                    else t.output_summary
                                )[:100],
                            }
                            for t in msg["trace"]
                        ]
                    )
                    st.dataframe(trace_df, use_container_width=True)

    # Handle incoming user query
    user_query = st.chat_input(
        "Ask a question about sales, products, customers, orders, or shipments..."
    )
    if "pending_query" in st.session_state:
        user_query = st.session_state.pop("pending_query")

    if user_query:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Execute controller pipeline
        with st.chat_message("assistant"):
            with st.spinner("Analyzing question, retrieving context, and querying database..."):
                resp = controller.handle(
                    session_id=st.session_state.session_id,
                    question=user_query,
                )

            st.markdown(resp.answer)

            if resp.sql:
                with st.expander("🔍 View Generated PostgreSQL", expanded=False):
                    st.code(resp.sql, language="sql")

            if resp.rows:
                df = pd.DataFrame(resp.rows)
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name=f"query_result_{resp.trace_id}.csv",
                    mime="text/csv",
                    key=f"dl_new_{resp.trace_id}",
                )

            with st.expander("⚙️ How I got this (Agent Execution Trace)", expanded=False):
                st.write(f"**Intent Classified:** `{resp.intent}`")
                st.write(f"**Rewritten Standalone Question:** {resp.rewritten_question}")
                st.write(f"**Fallback Model Used:** `{resp.fallback_used}`")
                trace_df = pd.DataFrame(
                    [
                        {
                            "Step": t.step_name,
                            "Latency (ms)": t.latency_ms,
                            "Model": t.model,
                            "Key": t.key_label,
                            "Output Summary": t.output_summary[:120],
                        }
                        for t in resp.trace
                    ]
                )
                st.dataframe(trace_df, use_container_width=True)

            # Store in session state
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": resp.answer,
                    "sql": resp.sql,
                    "rows": resp.rows,
                    "intent": resp.intent,
                    "rewritten": resp.rewritten_question,
                    "trace_id": resp.trace_id,
                    "trace": [
                        {
                            "step_name": t.step_name,
                            "latency_ms": t.latency_ms,
                            "model": t.model,
                            "key_label": t.key_label,
                            "output_summary": t.output_summary,
                        }
                        for t in resp.trace
                    ],
                }
            )

# ------------------------------------------------------------------------------
# TAB 2: Evaluation & Benchmarks
# ------------------------------------------------------------------------------
with tab_eval:
    st.subheader("DeepEval & Execution Accuracy Evaluation Reports")
    reports_dir = Path(__file__).resolve().parent / "evaluation" / "reports"
    report_files = sorted(glob.glob(str(reports_dir / "*.json")), reverse=True)

    if report_files:
        selected_report = st.selectbox("Select Evaluation Report", options=report_files)
        if selected_report:
            with open(selected_report, encoding="utf-8") as rf:
                report_data = json.load(rf)

            st.write(f"**Report Generated:** `{report_data.get('timestamp')}`")
            metrics = report_data.get("metrics", {})

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Execution Accuracy", f"{metrics.get('execution_accuracy', 0) * 100:.1f}%")
            col2.metric("Valid SQL Rate", f"{metrics.get('valid_sql_rate', 0) * 100:.1f}%")
            col3.metric(
                "Guardrail Block Rate", f"{metrics.get('guardrail_block_rate', 0) * 100:.1f}%"
            )
            col4.metric("Avg Latency", f"{metrics.get('avg_latency_ms', 0):.0f} ms")

            # Ablation Table
            if "ablations" in report_data:
                st.subheader("Retrieval & Pipeline Ablations")
                st.dataframe(pd.DataFrame(report_data["ablations"]), use_container_width=True)

            # Detailed test cases
            if "cases" in report_data:
                st.subheader("Evaluated Test Cases")
                st.dataframe(pd.DataFrame(report_data["cases"]), use_container_width=True)
    else:
        st.info(
            "No evaluation reports generated yet. Run `make eval` or `python -m evaluation.run_eval --limit 10`."
        )
