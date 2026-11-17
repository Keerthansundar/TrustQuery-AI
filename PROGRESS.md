# Project Progress & Implementation Status

**Last Updated:** 2026-11-15  
**Status Overview:** Complete migration from Google Gemini to official OpenAI Python SDK (`gpt-4o-mini`, `text-embedding-3-small`, and `OpenAIJudge`). All 81 unit tests pass offline (`0.00` Ruff errors). Integration tests, re-indexing script, and smoke test ready for live OpenAI key execution.

---

## Phase Status Summary Table

| Phase | Description | Status | Evidence | Exact Remaining Tasks |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | Docker + PostgreSQL 16 + pgvector | **Done** (Ready for Docker up) | DDL (`01_schema.sql`), load (`02_load.sql`), app/rag (`03_rag_app.sql` updated to 1536-dim), roles (`04_roles.sh`), `docker-compose.yml`, `Dockerfile`, `Makefile`, and `db/verify.py`. | Start Docker daemon and run `make up && make verify`. |
| **Phase 2** | OpenAI LLM layer with backoff & token tracking | **Done** | `src/llm/llm_client.py` unified client. `tests/test_llm.py` passes 8/8 tests offline (missing key check, primary success, retry on 429, model fallback, unrecoverable failure, json generation, embeddings, token tracking). | Supply live `OPENAI_API_KEY` in `.env` for live queries. |
| **Phase 3** | Two-stage retrieval engine (OpenAI embeddings) | **Done** | `text-embedding-3-small` (1536-dim). `db/reindex_embeddings.py` created. `tests/test_retrieval.py` passes 5/5 tests (load chunks, BM25, RRF merge, FK expansions, schema mode full). | Run `python db/reindex_embeddings.py` on live DB. |
| **Phase 4** | Plain function agents | **Done** | `tests/test_agents.py` passes 4/4 tests (`test_is_self_contained_heuristics`, `test_clean_sql_output`, `test_responder_sanity_check`, `test_intent_enum`). Centralized prompts in `src/prompts.py`. | None. |
| **Phase 5** | Guardrails (Input, SQL AST, DB, Output) | **Done** | `tests/test_guardrails.py` passes 53/53 tests covering SQL injection, AST validation, table/function allowlists, column denylists, limit injection, PII masking, and output redaction. | None. |
| **Phase 6** | Memory (`app.sessions`, `app.query_log`) | **Done** | `src/memory.py` implemented with Postgres backing and in-memory test fallback. `tests/test_controller.py` verifies session loading, rolling summary, and row summarization. | None. |
| **Phase 7** | Controller pipeline & tracing | **Done** | `src/controller.py`, `src/state.py`, `src/tracing.py` implemented. `tests/test_controller.py` passes 7/7 tests (routing, chitchat 0-retrieval, unclear questions, write attempt refusal, 3-attempt retry loop). | None. |
| **Phase 8** | Streamlit UI (`app.py`) | **Done** | `app.py` updated with OpenAI model status indicator, live session token & cost tracking, chat interface, trace expander, and evaluation tab. | Run `make run` to launch UI. |
| **Phase 9** | Content files & DeepEval Evaluation | **Done** | `data/schema_docs.yaml`, `data/examples.yaml`, and `data/golden_set.json` complete. Leakage test passes. `evaluation/openai_judge.py`, `evaluation/metrics.py`, and `evaluation/run_eval.py` updated with readable markdown summary table, reasons, subset run option, and token cost estimation. | Execute `python -m evaluation.run_eval --limit 5` on live stack. |
| **Phase 10** | Tests and CI Workflow | **Done** | 81 unit tests passing in `<2s`. `tests/test_integration.py` created with `@pytest.mark.integration`. GitHub Actions CI workflow updated with `OPENAI_API_KEY` in `.github/workflows/ci.yml`. | None. |
| **Phase 11** | README and Final Documentation | **Done** | `README.md` complete with updated Mermaid architecture diagram, OpenAI setup guide, re-indexing guide, smoke-test guide, cost breakdown, and interview Q&A. | None. |

---

## Detailed Task Checklist

- [x] Phase 1: Database DDL updated to 1536-dim vector, load, roles, verification script, docker-compose, and Makefile
- [x] Phase 2: OpenAIClient with exponential backoff, request timeout, model fallback, token tracking, spend calculation
- [x] Phase 3: Document chunking, hybrid RRF, BM25 store, pgvector 1536-dim store, retriever, reindex script
- [x] Phase 4: Intent planner, conversational rewriter, SQL generator, explainer, prompts
- [x] Phase 5: Input guardrail, SQL AST guardrail (53 tests), Output guardrail, DB execution safety
- [x] Phase 6: Session memory, rolling summarization, query audit logging
- [x] Phase 7: Controller orchestration, pipeline state, JSONL tracing
- [x] Phase 8: Streamlit web UI with chat, trace drawer, and spend accounting
- [x] Phase 9: Golden test set validation (`test_golden.py`), `openai_judge.py`, `metrics.py`, `run_eval.py`
- [x] Phase 10: Integration tests (`test_integration.py`), GitHub Actions CI (`.github/workflows/ci.yml`)
- [x] Phase 11: Production `README.md` with interview Q&A and architecture documentation
- [ ] Live Verification: User supplies `OPENAI_API_KEY` in `.env`, runs `python scripts/smoke_test.py`, runs `python db/reindex_embeddings.py`, and runs `streamlit run app.py`.
- [x] Evaluation Phase 1 (Golden Set Audit): Completed scripts/audit_golden_set.py with single SELECT/WITH assertion, AST validation, and tie checking on LIMIT clauses.
- [x] Evaluation Phase 2 (Fix the Golden Set): Updated 42 cases with tags, removed arbitrary LIMITs (gold_10, gold_22), fixed empty queries (gold_14, gold_17, gold_40), clarified glossary alignment (gold_04, gold_06, gold_22, gold_25, gold_02, gold_42), neutralized destructive queries (gold_35, gold_36), set gold_39 to safe_handling, and documented all edits in data/GOLDEN_SET_CHANGES.md.
- [x] Evaluation Phase 3 (Runner Fixes): Implemented history seeding, table row count mutation safety snapshot, per-behavior pass-rate table, answer-only execution accuracy, stratified sampling (--limit 8), real spend accounting from token usage, and golden set SHA256 hashing in evaluation/run_eval.py.
- [x] Evaluation Phase 4 (Metric Fixes): Implemented float normalization, bipartite row matching with superset multisets, metric_values_only compare mode, deterministic schema_recall with sqlglot, dedicated behavior scorers, and G-Eval actual_output=resp.sql in evaluation/metrics.py.
- [x] Evaluation Phase 5 (Few-Shot Overlap): Implemented evaluation/overlap.py with token Jaccard and SQL difflib matching (threshold 0.8), tagging seen_pattern cases and reporting seen vs unseen accuracy split.
- [x] Evaluation Phase 6 (OpenAI Judge & Tests): Enhanced OpenAIJudge schema retry-and-raise logic, created tests/test_judge_smoke.py marked @pytest.mark.integration without gold_sql leakage, added 7 comprehensive offline unit tests in tests/test_evaluation.py (normalization, superset bipartite matching, metric_values_only, schema_recall, behavior scoring, single SELECT/WITH assertion, history seeding), ran full offline test suite (88 passed in 1.22s), and re-ran final audit pass.
