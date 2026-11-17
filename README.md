# TrustQuery AI

An agentic **text-to-SQL analyst** that turns plain-English business questions into safe, read-only PostgreSQL queries.

Built in plain Python with schema-aware RAG (PostgreSQL 16 + pgvector), SQL guardrails, and a DeepEval golden-set harness. Uses the OpenAI API (`gpt-4o-mini`, `text-embedding-3-small`) and a Streamlit UI.

## How it works


1. **Input guardrail** blocks prompt injection and masks PII.
2. **Intent planner** routes chitchat, unclear questions, follow-ups and data questions.
3. **Query rewriter** resolves follow-ups ("what about FedEx?") into standalone questions.
4. **Hybrid retrieval** finds relevant schema, glossary and example queries (pgvector + BM25, fused and reranked).
5. **SQL generator** writes PostgreSQL, with up to 3 self-correction attempts.
6. **SQL guardrail** (sqlglot) allows only a single `SELECT` on allowlisted tables and functions.
7. **Read-only execution** runs under a restricted role with timeouts and a query cost limit.
8. **Output guardrail and explainer** mask leaked PII or secrets and explain the result in plain English.

## Safety layers

| Layer | What it does |
| :--- | :--- |
| Input guardrail | Blocks injection patterns, oversized input, masks PII |
| SQL AST guardrail | `SELECT` only, table and function allowlists, `LIMIT` clamp |
| Database roles | `analyst_ro` has no access to `email` and `phone` columns |
| Runtime limits | Read-only transaction, 5 s statement timeout, `EXPLAIN` cost cap |
| Output guardrail | Masks PII, redacts keys, caps rows at 100 |

## Quickstart

**Prerequisites:** Python 3.11+, Docker with Docker Compose, an [OpenAI API key](https://platform.openai.com/).

```bash
# 1. Clone and create a virtual environment
git clone https://github.com/Keerthansundar/TrustQuery-AI.git
cd TrustQuery-AI
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env             # Windows: copy .env.example .env
# Edit .env: set OPENAI_API_KEY and change the placeholder database passwords

# 3. Start PostgreSQL 16 + pgvector
make up                          # or: docker compose up -d

# 4. Verify schema, data and roles
make verify

# 5. Build the vector index
make index                       # or: python db/reindex_embeddings.py

# 6. Launch the app
make run                         # or: streamlit run app.py
```

Open **http://localhost:8501**.

Optional credential check before indexing: `python scripts/smoke_test.py`

## Configuration

All settings live in `.env` (see `.env.example` for the full list). The most useful ones:

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `OPENAI_API_KEY` | none | Required |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | SQL generation, planning, explanation |
| `JUDGE_MODEL` | `gpt-4o-mini` | DeepEval judge |
| `AS_OF_DATE` | set in `.env.example` | Fixed "today" so date answers are reproducible |
| `ALLOW_PII` | `false` | Enables the PII-capable role when `true` |
| `MAX_ROWS` / `MAX_PLAN_COST` | `100` / `50000` | Result and query cost limits |

## Testing

Unit tests run offline and need no API key:

```bash
pytest -m "not integration" -v
```

## Evaluation

Runs a 42-case golden set (30 answerable questions plus chitchat, clarification, injection, destructive-write and PII cases). Execution accuracy compares result sets order-insensitively.

```bash
python -m evaluation.run_eval --limit 5 --no-llm-judge   # quick, no judge cost
python -m evaluation.run_eval --limit 42                 # full run with DeepEval
```

A full run costs under $0.10.

### Latest results

| Metric | Result |
| :--- | :--- |
| Execution accuracy (30 answerable questions) | 80% (24/30) |
| Valid SQL rate | 100% |
| Guardrail cases passed (injection, writes, PII) | 12/12 |
| Intent accuracy | 100% |
| DeepEval faithfulness | 0.94 |
| DeepEval answer relevancy | 0.96 |
| DeepEval contextual relevancy | 0.77 |
| G-Eval SQL correctness | 0.70 |

Seen and unseen query patterns score the same (80%), and a unit test (`tests/test_golden.py`) fails if any golden question is too similar to a few-shot example.

### Known limitations

- Follow-up questions retrieve on short fragments, which lowers contextual relevancy.
- Two failures are real logic errors (wrong filter column, missing leaf-category filter).
- G-Eval penalises alias and column-order differences, so execution accuracy is the primary metric.

## Tech stack

Python, OpenAI API, PostgreSQL 16, pgvector, sqlglot, Pydantic, DeepEval, Streamlit, Docker.