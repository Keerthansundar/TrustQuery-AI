# Text-to-SQL Analyst (Agentic RAG in Plain Python with OpenAI)

An end-to-end, production-minded **Text-to-SQL Analyst** built using Agentic RAG in **plain Python**, backed by **PostgreSQL 16 with pgvector in Docker**, **Streamlit**, and the official **OpenAI Python SDK** (`gpt-4o-mini` & `text-embedding-3-small`).

---

## Architecture Overview

```mermaid
flowchart TD
    User([User Question]) --> IG[1. Input Guardrail\nLength, Injection Regex, PII Mask]
    IG -->|Blocked| BlockedResp[Polite Refusal Response]
    IG -->|Safe| Plan[2. Intent Planner\ngpt-4o-mini: data_question | followup | chitchat | unclear]
    
    Plan -->|Chitchat| ReplyDirect[Reply Directly\n0 Retrieval Calls]
    Plan -->|Unclear| Clarify[Ask Clarifying Question\nSuggest Interpretations]
    Plan -->|Data / Followup| Rewrite[3. Query Rewriter\nResolve Pronouns & Carry Context]
    
    Rewrite --> Retr[4. Two-Stage Hybrid Retrieval\npgvector Cosine Top-8 1536-dim + BM25 Top-8\nRRF Fusion + LLM Reranking + FK/Glossary Expansion]
    Retr --> Grade{Retrieval Grading\nContext Sufficient?}
    Grade -->|No| ReRetrieve[Single Retrieval Query Rewrite] --> GenSQL
    Grade -->|Yes| GenSQL
    
    subgraph Bounded Retry Loop [Self-Correction Loop: Up to 3 Attempts]
        GenSQL[5. SQL Generator\nPostgreSQL 16, Net Revenue Rules, AS_OF Date]
        GenSQL --> ASTGuard[6. SQL AST Guardrail\nsqlglot: SELECT-only, Table/Func Allowlists, Star Expansion]
        ASTGuard -->|Write Attempt| ShortCircuit[Immediate Refusal:\nRead-only Queries Only]
        ASTGuard -->|Fixable Error| GenSQL
        ASTGuard -->|Safe SQL| ExecDB[(7. PostgreSQL Execution\nanalyst_ro, READ ONLY Tx, Timeout, Cost Guard)]
        ExecDB -->|Cost/DB Error| GenSQL
        ExecDB --> Sanity{Sanity Check\nEmpty / Negative / >100%}
        Sanity -->|Insane| GenSQL
        Sanity -->|Valid Data| OutGuard[8. Output Guardrail\nRow PII Mask & Secret Redaction]
    end
    
    OutGuard --> Explain[9. Explainer Agent\nPlain English, Net Assumptions, USD]
    Explain --> Mem[(10. Session Memory & Audit Log\napp.sessions & app.query_log)]
    Mem --> UI([Streamlit UI Response & Trace Expander])
```

---

## Key Architectural Principles

### 1. "Why no LangGraph / LangChain?"
- **Linear Flow with Single Branch & Bounded Loop:** The agentic flow is fundamentally a deterministic linear pipeline with one classification branch (`chitchat` / `unclear` vs `data_question`) and a bounded retry loop (up to 3 attempts). A graph framework introduces heavy dependency weight, complex abstractions, hard-to-trace stack frames, and runtime overhead without adding capability.
- **Maintainability & Testability:** Plain Python functions and dataclasses (`PipelineState`, `PipelineTraceStep`, `Response`) can be unit tested, mocked, and debugged with standard Python tooling (`pytest`, `pdb`, `unittest.mock`) in milliseconds.
- **When We Would Adopt a Graph Framework:** Graph orchestration engines become worthwhile only when systems require parallel asynchronous branches, durable checkpoint/resume across distributed workers, human-in-the-loop suspension interrupts, or multi-agent swarms with dynamic handoffs.

### 2. "Is this agentic?"
Yes. The system is agentic because an LLM dynamically:
1. **Plans the execution path:** Classifies user intent and routes execution (short-circuiting chitchat with 0 retrieval calls).
2. **Rewrites and disambiguates:** Resolves pronouns, carries forward SQL entities from conversational history, and reformulates queries for retrieval.
3. **Grades its own context:** Evaluates whether retrieved chunks are sufficient to answer the prompt, autonomously triggering a single re-retrieval if necessary.
4. **Self-corrects from database feedback:** Captures syntax errors, missing columns, or database execution exceptions and iteratively repairs SQL inside a bounded loop.

---

## Defense-in-Depth Guardrail Matrix

Security and database integrity are enforced across 5 layered boundaries:

| Layer | Mechanism | Scope & Purpose |
| :--- | :--- | :--- |
| **1. Input Guardrail** | Regex patterns & heuristic validation | Blocks prompt injection (`"ignore previous instructions"`, `"reveal system prompt"`, `"you are now"`), rejects oversized questions, and masks user PII before logging. |
| **2. SQL AST Guardrail** | `sqlglot` deterministic AST parser | Guarantees single-statement `SELECT` root only; rejects `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, `GRANT`, `COPY`, `SELECT INTO`, `FOR UPDATE`; restricts access to the 7 business tables in schema `public`; enforces function allowlists (blocking `pg_sleep`, `pg_read_file`, `dblink`); expands `SELECT *`; clamps `LIMIT 100`. |
| **3. Database Security Roles** | PostgreSQL role privileges (`04_roles.sh`) | `analyst_ro` is granted `SELECT` only on business tables, with column-level restrictions that **strictly exclude `email` and `phone`** on `customers`. Schema `rag` and `app` are locked to `app_rw`. |
| **4. Database Runtime Limits & Cost Guard** | Transaction settings & `EXPLAIN` cost guard | Transactions run with `SET LOCAL default_transaction_read_only = on`, `statement_timeout = '5s'`, and `lock_timeout = '2s'`. Pre-flight `EXPLAIN (FORMAT JSON)` blocks queries whose total estimated plan cost exceeds `MAX_PLAN_COST` (50,000), preventing cartesian joins from exhausting server CPU/RAM. |
| **5. Output Guardrail** | Regex sanitization & row masking | Masks residual customer emails/phones from dataframe results unless `ALLOW_PII=true`, truncates rows to `MAX_ROWS` (100), and strips any accidentally leaked API keys or database DSNs from generated explanations. |

---

## Quickstart & Setup Guide

### 1. Prerequisites
- Python 3.11+ (recommended Python 3.12 or 3.13)
- Docker & Docker Compose
- An OpenAI API Key from [platform.openai.com](https://platform.openai.com/)

### 2. Environment Configuration
Create your `.env` file from the provided `.env.example` template:
```bash
cp .env.example .env
```
Edit `.env` and fill in your OpenAI API key:
```ini
OPENAI_API_KEY=sk-proj-yourActualKeyHere
```

Optional runtime environment configurations (can be overridden in `.env`):
```ini
# Optional model fallback on the same key during 429 or transient outages
OPENAI_MODEL_FALLBACK=gpt-4.1-mini

# Models (default cost-efficient choices)
GEN_MODEL=gpt-4o-mini
FAST_MODEL=gpt-4o-mini
JUDGE_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536

# Timeouts & Retries
OPENAI_TIMEOUT_SECONDS=30.0
OPENAI_MAX_RETRIES=3
```

### 3. Execution Workflow via Makefile / CLI

```bash
# 1. Start PostgreSQL 16 with pgvector in Docker
make up

# 2. Verify schema loading, counts, and security roles
make verify

# 3. Run the credential smoke test (verifies 1 embedding + 1 SQL completion call offline/online)
python scripts/smoke_test.py

# 4. Migrate and rebuild the vector store with 1536-dim OpenAI embeddings
python db/reindex_embeddings.py
# (or: make index)

# 5. Launch the Streamlit Web Application
make run
# (or: streamlit run app.py)
```
Access the application at: `http://localhost:8501`

---

## Vector Store Migration & Re-Indexing

OpenAI's `text-embedding-3-small` generates 1536-dimensional vectors (replacing legacy 768-dimensional embeddings). 

To ensure complete index compatibility, run the migration script:
```bash
python db/reindex_embeddings.py
```
This script:
1. Connects to PostgreSQL and updates `rag.documents.embedding` column type from `vector(768)` to `vector(1536)`.
2. Clears legacy incompatible embeddings.
3. Batch-embeds all schema docs, glossary definitions, and few-shot examples using `text-embedding-3-small`.

---

## Evaluation & Benchmarks

The project includes an evaluation suite combining deterministic execution accuracy and DeepEval LLM-judge metrics:

```bash
# Run deterministic evaluation on 5 golden test cases (saves tokens)
python -m evaluation.run_eval --limit 5 --no-llm-judge

# Run evaluation with DeepEval OpenAI judge (gpt-4o-mini)
python -m evaluation.run_eval --limit 5

# Run evaluation on specific subset of questions
python -m evaluation.run_eval --subset "q1,q3,q7"

# Run full evaluation on all 42 golden test cases
python -m evaluation.run_eval --limit 42
```

### DeepEval Evaluation Summary Output Example
Evaluation prints a detailed summary table at the conclusion of each run:
| Case ID | Metric Name | Score | Threshold | Status | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- |
| q1 | contextual_relevancy | 1.000 | 0.70 | ✅ PASS | Retrieved context directly covers the required tables. |
| q1 | contextual_recall | 0.950 | 0.70 | ✅ PASS | Key column definitions for order revenue were present. |
| q1 | faithfulness | 1.000 | 0.70 | ✅ PASS | Explanations accurately describe the net revenue formula. |
| q1 | answer_relevancy | 0.920 | 0.70 | ✅ PASS | Answers the user's specific temporal query directly. |
| q1 | geval_sql_correctness | 0.980 | 0.70 | ✅ PASS | Correct PostgreSQL 16 syntax adhering to read-only constraints. |

### Token Cost Estimates for Evaluation (gpt-4o-mini)
- **Per test case:** ~6,400 input tokens, ~850 output tokens = **~$0.00147** (< 0.2 cents)
- **Subset run (5 cases):** ~36,000 tokens = **~$0.007** (< 1 cent)
- **Full benchmark (all 42 cases):** ~305,000 tokens = **~$0.062** (~6 cents total)

---

## Offline Verification & Unit Testing

The test suite runs 100% offline without requiring an active OpenAI key:
```bash
pytest -m "not integration" -v
```
All 81 unit tests pass in `< 2 seconds`, verifying:
- OpenAI client initialization, early missing-key failure checks, and timeout handling
- Exponential backoff retry on HTTP 429 rate-limit errors and transient connection failures
- Optional model fallback on the same key
- Token usage tracking and spend calculation
- Input, AST, and Output guardrails
- Intent planning, standalone query rewriting, and bounded retry loops

---

## Technical Interview Q&A

### Q1: Why plain Python instead of LangGraph, LangChain, or LlamaIndex?
**Answer:** Frameworks like LangGraph excel in dynamic multi-agent topologies with cyclical state machines, durable checkpoints, and parallel graph edges. For a text-to-SQL analyst, the workflow is fundamentally a deterministic linear pipeline with one routing branch and one bounded retry loop. Implementing this in plain Python functions with dataclasses eliminates thousands of transitive dependencies, prevents vendor lock-in, ensures zero hidden prompt overhead, makes unit testing instantaneous (`<2s` for 79 tests), and produces transparent stack traces that can be debugged with standard Python tools.

### Q2: Why use hybrid retrieval (BM25 + pgvector) instead of vector-only?
**Answer:** Vector embeddings excel at semantic matching (e.g. associating "how much cash came in" with "payments" and "amount"), but frequently perform poorly on exact token matches such as specific database column names (`order_item_id`, `shipping_fee`), status enums (`'cancelled'`, `'in_transit'`), or domain terminology. BM25 provides lexical precision for exact identifiers. By combining both with Reciprocal Rank Fusion (RRF), we achieve high recall on both domain concepts and exact database identifiers.

### Q3: Why bother with RAG when schemas can fit inside an LLM's context window?
**Answer:** While modern LLMs have large context windows, feeding the entire database schema, constraint catalog, business glossary, and 25 few-shot examples into every single prompt:
1. Significantly increases time-to-first-token (TTFT) and inference latency.
2. Increases token consumption and cost.
3. Causes attention degradation on complex queries involving subtle business rules.
4. Breaks down when the schema scales from 7 tables to hundreds of enterprise tables. RAG modularizes context injection and keeps prompts focused and minimal.

### Q4: How is LLM reliability and fallback handled with the OpenAI API?
**Answer:** All LLM and embedding calls route through a single provider module (`src/llm/llm_client.py`). The client enforces request timeouts (`OPENAI_TIMEOUT_SECONDS = 30.0`), client-side token bucket rate-limiting, and exponential backoff retry with jitter on `RateLimitError` (HTTP 429) and transient connection/server errors. Additionally, an optional model fallback (`OPENAI_MODEL_FALLBACK`) can be configured: if the primary model (`gpt-4o-mini`) encounters persistent rate limits or outages, it automatically falls back to an alternate model using the same key without dropping the query.

### Q5: How is destructive or injected SQL prevented (Defense-in-Depth)?
**Answer:** We enforce 5 layers of defense:
1. *Input Guardrail:* Blocks prompt-injection patterns (`"ignore previous instructions"`, `"drop table"`).
2. *Deterministic AST Parser (`sqlglot`):* Rejects any query containing multiple statements, non-SELECT roots, comments hiding statements, CTE-wrapped writes (`WITH x AS (DELETE ...)`), or non-whitelisted functions (`pg_sleep`, `pg_read_file`, `dblink`).
3. *Database Role Isolation:* Queries run under the `analyst_ro` role, which possesses only `SELECT` privileges on the 7 business tables.
4. *Read-Only Transactions:* Every transaction executes with `SET LOCAL default_transaction_read_only = on`.
5. *Statement Timeouts & Cost Guard:* Sessions are capped at a 5-second `statement_timeout` and pre-flight `EXPLAIN (FORMAT JSON)` rejects plans exceeding `MAX_PLAN_COST`.

### Q6: How is sensitive Personally Identifiable Information (PII) protected at the database level?
**Answer:** We enforce column-level security in PostgreSQL (`04_roles.sh`). The default role `analyst_ro` is granted `SELECT` only on non-PII columns of the `customers` table (`customer_id`, `first_name`, `last_name`, `city`, `state`, `country`, `signup_date`, `segment`). The columns `email` and `phone` are **explicitly denied**. Even if the LLM attempts to generate `SELECT email FROM customers`, PostgreSQL throws a SQL authorization error `permission denied for column email`. An elevated role `analyst_pii_ro` is accessed only when the configuration flag `ALLOW_PII=true` is deliberately enabled.

### Q7: How do you evaluate Text-to-SQL systems, and how do you prevent test leakage?
**Answer:** Evaluating text-to-SQL purely on generated SQL string matching (e.g. BLEU or exact string match) is fundamentally flawed because different SQL statements (different join orders, subqueries vs CTEs, aliases) can produce identical correct results. Instead, we evaluate using:
1. *Deterministic Execution Accuracy:* Executing both generated and gold SQL against the database and comparing resulting record multisets order-insensitively with numeric rounding tolerance.
2. *DeepEval LLM Judge:* Measuring Contextual Relevancy, Faithfulness, and SQL Correctness.
3. *Leakage Prevention:* We enforce a programmatic unit test (`tests/test_golden.py`) that computes string and fuzzy SequenceMatcher similarity between every golden test case and all few-shot examples in `examples.yaml`, failing if similarity exceeds `0.88`.

### Q8: How does conversational follow-up memory work across turns?
**Answer:** Chat history is stored in Postgres table `app.sessions`. When a user submits a follow-up (e.g., `"what about by category?"` after asking `"top products by revenue"`), the `rewriter` agent inspects the user question and the last 5 turns. It extracts previously applied filters (date ranges, customer segments, status conditions) and standalone entities, producing an expanded, self-contained question (`"top categories by revenue last quarter"`) before retrieval and generation occur.

### Q9: How do you manage cost on the OpenAI API?
**Answer:**
1. Using `gpt-4o-mini` ($0.15/1M input, $0.60/1M output), keeping inference cost under $0.0002 per query.
2. Embedding with `text-embedding-3-small` ($0.02/1M tokens).
3. Intent short-circuiting: chitchat skips retrieval and SQL steps completely.
4. Standalone heuristics: questions that are already self-contained skip conversational rewriting calls.
5. Content-hash caching: embeddings are generated once and re-computed only if underlying documentation changes.

### Q10: What are the primary failure modes of this architecture and what would you build next?
**Answer:**
- *Primary Failure Modes:*
  1. Ambiguous business metrics where the user's mental model differs from company rules (e.g. gross vs net revenue).
  2. Extremely complex multi-table joins that produce slow execution plans.
  3. API quota or transient outages on the provider end.
- *Next Steps in Production:*
  1. *Semantic Layer / Metric Repository (e.g. Cube or dbt Semantic Layer):* Pre-defining certified metrics (e.g. `net_revenue`, `churn_rate`) so the LLM queries standardized views rather than assembling raw math.
  2. *HNSW Vector Indexing:* Replacing exact cosine scan with pgvector HNSW indexes for sub-millisecond retrieval at scale.
  3. *Query Cache:* Caching deterministic SQL outputs for frequent questions with TTL invalidation.
