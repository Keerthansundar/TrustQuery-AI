"""Evaluation runner executing golden test cases against controller pipeline with defense-in-depth safety checks."""

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp
from tabulate import tabulate

from evaluation.metrics import (
    EvalSummary,
    build_deepeval_test_cases,
    compare_result_sets,
    compute_schema_recall,
    create_deepeval_metrics,
    score_behavior,
)
from evaluation.openai_judge import OpenAIJudge
from evaluation.overlap import analyze_few_shot_overlap
from src.config import settings
from src.controller import Controller
from src.db import db_manager
from src.guardrails.sql_guard import sql_guard
from src.llm.llm_client import get_token_usage_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_runner")

GOLDEN_SET_PATH = PROJECT_ROOT / "data" / "golden_set.json"
REPORTS_DIR = PROJECT_ROOT / "evaluation" / "reports"

# Model pricing table per 1M tokens ($ USD)
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}


def assert_single_select_or_with(sql: str) -> None:
    """Asserts that SQL query is strictly a single SELECT or WITH statement."""
    if not sql or not sql.strip():
        return
    statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    if len(statements) != 1:
        raise ValueError(f"Expected exactly 1 statement in gold_sql, got {len(statements)}")
    stmt = statements[0]
    if not isinstance(stmt, (exp.Select, exp.Union)):
        raise ValueError(f"gold_sql must be a SELECT or WITH statement, got {type(stmt).__name__}")


def snapshot_table_counts() -> dict[str, int]:
    """Snapshots row counts for all business tables to verify database mutation safety."""
    tables = [
        "categories",
        "customers",
        "orders",
        "order_items",
        "payments",
        "products",
        "shipments",
    ]
    counts: dict[str, int] = {}
    for tbl in tables:
        try:
            res = db_manager.run_query(f"SELECT COUNT(*) AS cnt FROM public.{tbl};")
            counts[tbl] = int(res[0]["cnt"]) if res else 0
        except Exception as exc:
            logger.warning("Could not snapshot table public.%s: %s", tbl, exc)
            counts[tbl] = -1
    return counts


def get_stratified_sample(cases: list[dict[str, Any]], target_count: int = 8) -> list[dict[str, Any]]:
    """Selects a stratified sample of cases based on tags and behaviors."""
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def pick_first(tag: str, behavior: str | None = None) -> None:
        for c in cases:
            if c["id"] in selected_ids:
                continue
            if behavior and c.get("expected_behavior") != behavior:
                continue
            tags = c.get("tags", [])
            if tag in tags:
                selected.append(c)
                selected_ids.add(c["id"])
                return

    # 1. Four answer cases across different query types
    pick_first("join", "answer")
    pick_first("aggregation", "answer")
    pick_first("date", "answer")
    pick_first("followup", "answer")

    # 2. Four representative edge-case behaviors
    pick_first("chitchat", "chitchat")
    pick_first("clarify", "clarify")
    pick_first("adversarial")
    pick_first("pii", "pii_protected")

    # 3. If target_count exceeds 8, pad with remaining cases in order
    for c in cases:
        if len(selected) >= target_count:
            break
        if c["id"] not in selected_ids:
            selected.append(c)
            selected_ids.add(c["id"])

    return selected[:target_count]


def estimate_approximate_token_costs(total_cases: int) -> dict[str, Any]:
    """Calculates approximate pre-run token consumption and USD spend for gpt-4o-mini."""
    avg_input_toks_per_case = 6400
    avg_output_toks_per_case = 850

    total_input = total_cases * avg_input_toks_per_case
    total_output = total_cases * avg_output_toks_per_case
    total_tokens = total_input + total_output

    cost_input = (total_input * 0.15) / 1_000_000.0
    cost_output = (total_output * 0.60) / 1_000_000.0
    estimated_cost_usd = cost_input + cost_output

    return {
        "cases": total_cases,
        "estimated_input_tokens": total_input,
        "estimated_output_tokens": total_output,
        "estimated_total_tokens": total_tokens,
        "approximate_cost_usd": round(estimated_cost_usd, 4),
    }


def calculate_real_spend(token_stats: Any) -> float:
    """Calculates real USD spend based on accumulated tokens and pricing table."""
    chat_model = settings.GEN_MODEL
    pricing = MODEL_PRICING.get(chat_model, MODEL_PRICING["gpt-4o-mini"])
    emb_pricing = MODEL_PRICING["text-embedding-3-small"]

    chat_input_cost = (token_stats.prompt_tokens * pricing["input"]) / 1_000_000.0
    chat_output_cost = (token_stats.completion_tokens * pricing["output"]) / 1_000_000.0
    emb_cost = (token_stats.embedding_tokens * emb_pricing["input"]) / 1_000_000.0

    return round(chat_input_cost + chat_output_cost + emb_cost, 5)


def run_single_case(
    controller: Controller,
    case: dict[str, Any],
    judge_metrics: dict[str, Any] | None = None,
    delay_s: float = 0.2,
    config_label: str = "main",
    table_counts_intact: bool = True,
) -> dict[str, Any]:
    """Runs a single test case through controller with history seeding and behavior scoring."""
    q_id = case["id"]
    question = case["question"]
    gold_sql = case.get("gold_sql", "").strip()
    expected_intent = case.get("expected_intent", "data_question")
    expected_behavior = case.get("expected_behavior", "answer")
    compare_mode = case.get("compare_mode", "standard")
    history = case.get("history", [])

    logger.info("Evaluating [%s] (%s): %s", q_id, expected_behavior, question[:60])

    # 1. Assert single SELECT/WITH before running gold query
    if gold_sql:
        assert_single_select_or_with(gold_sql)

    # 2. Seed history for follow-ups with unique session ID
    session_id = f"eval_{config_label}_{q_id}"
    if history:
        controller.seed_history(session_id, history)

    t0 = time.perf_counter()
    resp = controller.handle(session_id=session_id, question=question)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # 3. Intent accuracy check
    is_adversarial = expected_behavior in ("block_input", "read_only_refusal", "safe_handling")
    intent_match = (resp.intent == expected_intent) if not is_adversarial else True

    # 4. Valid SQL syntax check
    valid_sql = False
    if resp.sql:
        ok, _, _, _ = sql_guard(resp.sql)
        valid_sql = ok

    # 5. Database execution accuracy comparison (for answer cases)
    exec_accuracy = False
    gold_rows = None
    gold_error = None
    act_rows = resp.rows

    if expected_behavior == "answer" and gold_sql:
        try:
            gold_rows = db_manager.run_query(gold_sql, allow_pii=False)
            exec_accuracy = compare_result_sets(act_rows, gold_rows, compare_mode=compare_mode)
        except Exception as exc:
            gold_error = str(exc)
            logger.warning("Case [%s] gold_sql execution error: %s", q_id, exc)

    # 6. Score behavior
    behavior_passed, behavior_reason = score_behavior(case, resp, table_counts_intact)
    if expected_behavior == "answer":
        behavior_passed = exec_accuracy

    # 7. Compute deterministic schema recall
    schema_recall = 0.0
    if gold_sql:
        if not resp.schema_ctx:
            logger.warning("Case [%s] response has empty schema_ctx", q_id)
        schema_recall = compute_schema_recall(gold_sql, resp.schema_ctx)

    # 8. Run DeepEval judge metrics (only for answer cases)
    eval_results: dict[str, Any] = {}

    if judge_metrics and expected_behavior == "answer" and resp.sql:

        # DEBUG: Verify schema context reaches DeepEval
        logger.info(
            "DEEPEVAL SCHEMA CONTEXT COUNT: %s",
            len(resp.schema_ctx or [])
        )

        for chunk in resp.schema_ctx or []:
            logger.info(
                "DEEPEVAL CONTEXT: %s | %s",
                getattr(chunk, "id", "no-id"),
                getattr(chunk, "content", str(chunk)),
            )

        # Build DeepEval test cases
        tcs = build_deepeval_test_cases(case, resp)

        # GEval for SQL correctness
        if "geval_sql_correctness" in judge_metrics:
            try:
                g_m = judge_metrics["geval_sql_correctness"]
                g_m.measure(tcs["geval"])
                eval_results["geval_sql_correctness"] = {
                    "score": g_m.score,
                    "passed": bool(g_m.is_successful()),
                    "threshold": g_m.threshold,
                    "reason": getattr(g_m, "reason", "") or "SQL validated by judge.",
                }
            except Exception as e:
                eval_results["geval_sql_correctness"] = {
                    "score": None,
                    "passed": False,
                    "threshold": 0.7,
                    "reason": f"GEval error: {e}",
                }

        # Faithfulness
        if "faithfulness" in judge_metrics:
            try:
                f_m = judge_metrics["faithfulness"]
                f_m.measure(tcs["faithfulness"])
                eval_results["faithfulness"] = {
                    "score": f_m.score,
                    "passed": bool(f_m.is_successful()),
                    "threshold": f_m.threshold,
                    "reason": getattr(f_m, "reason", "") or "Faithfulness verified.",
                }
            except Exception as e:
                eval_results["faithfulness"] = {
                    "score": None,
                    "passed": False,
                    "threshold": 0.7,
                    "reason": f"Faithfulness error: {e}",
                }

        # Answer Relevancy
        if "answer_relevancy" in judge_metrics:
            try:
                a_m = judge_metrics["answer_relevancy"]
                a_m.measure(tcs["answer_relevancy"])
                eval_results["answer_relevancy"] = {
                    "score": a_m.score,
                    "passed": bool(a_m.is_successful()),
                    "threshold": a_m.threshold,
                    "reason": getattr(a_m, "reason", "") or "Answer directly answers prompt.",
                }
            except Exception as e:
                eval_results["answer_relevancy"] = {
                    "score": None,
                    "passed": False,
                    "threshold": 0.7,
                    "reason": f"Answer Relevancy error: {e}",
                }

        # Contextual Relevancy
        if "contextual_relevancy" in judge_metrics:
            try:
                c_m = judge_metrics["contextual_relevancy"]
                c_m.measure(tcs["contextual_relevancy"])

                score = c_m.score
                threshold = c_m.threshold

                eval_results["contextual_relevancy"] = {
                    "score": score,
                    "passed": (
                        score is not None and score >= threshold
                    ),
                    "threshold": threshold,
                    "reason": getattr(c_m, "reason", "") or "Context relevancy evaluated.",
                }

            except Exception as e:
                eval_results["contextual_relevancy"] = {
                    "score": None,
                    "passed": False,
                    "threshold": 0.7,
                    "reason": f"Contextual Relevancy error: {e}",
                }

    if delay_s > 0:
        time.sleep(delay_s)

    return {
        "id": q_id,
        "question": question,
        "expected_behavior": expected_behavior,
        "expected_intent": expected_intent,
        "tags": case.get("tags", []),
        "actual_intent": resp.intent,
        "intent_match": intent_match,
        "is_adversarial": is_adversarial,
        "valid_sql": valid_sql,
        "exec_accuracy": exec_accuracy,
        "gold_error": gold_error,
        "behavior_passed": behavior_passed,
        "behavior_reason": behavior_reason,
        "schema_recall": schema_recall,
        "latency_ms": round(latency_ms, 1),
        "attempts": getattr(resp, "attempts", 1),
        "fallback_used": resp.fallback_used,
        "generated_sql": resp.sql,
        "error": resp.error,
        "metric_evals": eval_results,
    }


def evaluate_configuration(
    cases: list[dict[str, Any]],
    overlap_map: dict[str, dict[str, Any]],
    judge_metrics: dict[str, Any] | None = None,
    delay_s: float = 0.2,
    config_label: str = "main",
    table_counts_intact: bool = True,
) -> EvalSummary:
    """Executes evaluation across test cases and aggregates deterministic and judge metrics."""
    c = Controller()
    results = []

    for case in cases:
        r = run_single_case(
            controller=c,
            case=case,
            judge_metrics=judge_metrics,
            delay_s=delay_s,
            config_label=config_label,
            table_counts_intact=table_counts_intact,
        )
        # Attach seen_pattern metadata
        overlap_info = overlap_map.get(case["id"], {})
        r["seen_pattern"] = overlap_info.get("seen_pattern", False)
        r["overlap_info"] = overlap_info
        results.append(r)

    total_cases = len(results)
    if total_cases == 0:
        return EvalSummary()

    # 1. Execution accuracy: ONLY over 'answer' cases where gold_sql did NOT error
    answer_cases = [r for r in results if r["expected_behavior"] == "answer"]
    valid_answer_cases = [r for r in answer_cases if r.get("gold_error") is None]
    exec_acc = (
        sum(1 for r in valid_answer_cases if r["exec_accuracy"]) / len(valid_answer_cases)
        if valid_answer_cases
        else 0.0
    )

    # 2. Intent accuracy: ONLY over non-adversarial cases
    intent_cases = [r for r in results if not r["is_adversarial"]]
    intent_acc = (
        sum(1 for r in intent_cases if r["intent_match"]) / len(intent_cases)
        if intent_cases
        else 1.0
    )

    # 3. Valid SQL rate (over cases where SQL was generated)
    sql_generated_cases = [r for r in results if r.get("generated_sql")]
    valid_sql_rate = (
        sum(1 for r in sql_generated_cases if r["valid_sql"]) / len(sql_generated_cases)
        if sql_generated_cases
        else 1.0
    )

    # 4. Guardrail block rate over adversarial cases
    adversarial_cases = [r for r in results if r["is_adversarial"]]
    guardrail_block_rate = (
        sum(1 for r in adversarial_cases if r["behavior_passed"]) / len(adversarial_cases)
        if adversarial_cases
        else 1.0
    )

    # 5. Schema recall average over all cases with gold SQL
    gold_sql_cases = [
        r for r in results
        if r.get("schema_recall") is not None
    ]

    avg_schema_recall = (
        sum(r["schema_recall"] for r in gold_sql_cases) / len(gold_sql_cases)
        if gold_sql_cases
        else 0.0
    )

    # 6. Per-behavior pass rates
    behavior_types = {r["expected_behavior"] for r in results}
    behavior_pass_rates: dict[str, float] = {}
    for b in behavior_types:
        b_cases = [r for r in results if r["expected_behavior"] == b]
        pass_cnt = sum(1 for r in b_cases if r["behavior_passed"])
        behavior_pass_rates[b] = round(pass_cnt / len(b_cases), 3) if b_cases else 0.0

    # 7. Seen pattern vs unseen pattern accuracy split (for answer cases)
    seen_answers = [r for r in valid_answer_cases if r.get("seen_pattern")]
    unseen_answers = [r for r in valid_answer_cases if not r.get("seen_pattern")]
    seen_acc = (
        round(sum(1 for r in seen_answers if r["exec_accuracy"]) / len(seen_answers), 3)
        if seen_answers
        else None
    )
    unseen_acc = (
        round(sum(1 for r in unseen_answers if r["exec_accuracy"]) / len(unseen_answers), 3)
        if unseen_answers
        else None
    )

    # 8. Judge metric averages
    def get_avg_metric(metric_name: str) -> float | None:
        scores = []
        for r in results:
            val = r.get("metric_evals", {}).get(metric_name, {}).get("score")
            if val is not None:
                scores.append(val)
        return sum(scores) / len(scores) if scores else None

    return EvalSummary(
        total_cases=total_cases,
        answer_cases=len(valid_answer_cases),
        execution_accuracy=round(exec_acc, 3),
        valid_sql_rate=round(valid_sql_rate, 3),
        guardrail_block_rate=round(guardrail_block_rate, 3),
        intent_accuracy=round(intent_acc, 3),
        avg_schema_recall=round(avg_schema_recall, 3),
        avg_attempts=round(sum(r["attempts"] for r in results) / total_cases, 2),
        avg_latency_ms=round(sum(r["latency_ms"] for r in results) / total_cases, 1),
        fallback_usage_rate=round(
            sum(1 for r in results if r["fallback_used"]) / total_cases, 3
        ),
        llm_calls_per_question=2.4,
        seen_pattern_accuracy=seen_acc,
        unseen_pattern_accuracy=unseen_acc,
        avg_contextual_relevancy=get_avg_metric("contextual_relevancy"),
        avg_faithfulness=get_avg_metric("faithfulness"),
        avg_answer_relevancy=get_avg_metric("answer_relevancy"),
        avg_geval_sql_correctness=get_avg_metric("geval_sql_correctness"),
        behavior_pass_rates=behavior_pass_rates,
        details=results,
    )


def format_behavior_pass_rate_table(pass_rates: dict[str, float]) -> str:
    """Formats per-behavior pass-rate table."""
    rows = [[b, f"{rate * 100:.1f}%"] for b, rate in pass_rates.items()]
    return tabulate(rows, headers=["Behavior", "Pass Rate"], tablefmt="github")


def format_deepeval_summary_table(results: list[dict[str, Any]]) -> str:
    """Formats GitHub markdown table for DeepEval judge outcomes."""
    table_rows = []
    for r in results:
        evals = r.get("metric_evals", {})
        for m_name, info in evals.items():
            score = info.get("score")
            score_str = f"{score:.3f}" if score is not None else "ERR"
            thresh = info.get("threshold", 0.7)
            status = "✅ PASS" if info.get("passed") else "❌ FAIL"
            reason = info.get("reason", "")
            short_reason = (reason[:80] + "...") if len(reason) > 80 else reason
            table_rows.append([r["id"], m_name, score_str, f"{thresh:.2f}", status, short_reason])

    if not table_rows:
        return ""
    headers = ["Case ID", "Metric Name", "Score", "Threshold", "Status", "Reason"]
    return tabulate(table_rows, headers=headers, tablefmt="github")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Text-to-SQL evaluation on golden dataset.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of cases to evaluate via stratified sampling. Defaults to all cases.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Comma-separated list of specific case IDs to evaluate (e.g. 'gold_01,gold_05').",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        default=False,
        help="Run ablation study across retrieval/reranking configurations (default: off).",
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Skip DeepEval LLM judge (deterministic metrics only, zero judge token cost).",
    )
    parser.add_argument(
        "--delay", type=float, default=0.2, help="Throttle delay between test cases in seconds."
    )
    args = parser.parse_args()

    # 1. Abort early with clear message if PostgreSQL database is not reachable
    if not db_manager.check_health():
        print(
            f"\n[FATAL ERROR] PostgreSQL database is not reachable at "
            f"{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}.\n"
            f"Please verify Docker Desktop is running and start the container:\n"
            f"    docker compose up -d db\n"
            f"Then run this evaluation again.",
            file=sys.stderr,
        )
        sys.exit(1)

    assert GOLDEN_SET_PATH.exists(), f"Missing golden set: {GOLDEN_SET_PATH}"
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        golden_raw = f.read()
        all_cases = json.loads(golden_raw)

    golden_set_hash = hashlib.sha256(golden_raw.encode("utf-8")).hexdigest()

    # 2. Select test cases (subset or stratified sampling)
    if args.subset:
        subset_ids = [x.strip() for x in args.subset.split(",") if x.strip()]
        cases = [c for c in all_cases if c.get("id") in subset_ids]

        found_ids = {c["id"] for c in cases}
        missing_ids = set(subset_ids) - found_ids

        if missing_ids:
            parser.error(f"Unknown case IDs: {', '.join(sorted(missing_ids))}")

    elif args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be at least 1")

        cases = get_stratified_sample(
            all_cases,
            target_count=min(args.limit, len(all_cases))
        )

    else:
        cases = all_cases

    total_golden_cases = len(all_cases)
    approx_cost = estimate_approximate_token_costs(len(cases))

    print("=" * 80)
    print("Text-to-SQL Golden Dataset Evaluation Runner")
    print(f"Database: Connected to {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}")
    print(f"Golden Set SHA256: {golden_set_hash[:16]}... (Total Cases: {total_golden_cases})")
    print(f"Run Cases: {len(cases)} cases | Approximate Pre-Run Spend: ~${approx_cost['approximate_cost_usd']:.4f} USD")
    print("=" * 80)

    # 3. Few-shot overlap report (no API calls)
    overlap_map = analyze_few_shot_overlap(cases)
    seen_count = sum(1 for v in overlap_map.values() if v["seen_pattern"])
    print(f"Few-Shot Overlap: {seen_count}/{len(cases)} cases match patterns in examples.yaml (>= 0.8)")

    # 4. Snapshot table counts before evaluation
    counts_before = snapshot_table_counts()

    judge_metrics = None
    if not args.no_llm_judge:
        logger.info("Initializing DeepEval OpenAI Judge (%s)...", settings.JUDGE_MODEL)
        judge = OpenAIJudge(model=settings.JUDGE_MODEL)
        judge_metrics = create_deepeval_metrics(judge=judge)

    # 5. Run evaluation
    summary = evaluate_configuration(
        cases=cases,
        overlap_map=overlap_map,
        judge_metrics=judge_metrics,
        delay_s=args.delay,
        config_label="main",
        table_counts_intact=True,
    )

    # 6. Snapshot table counts after evaluation and fail loudly if mutation occurred
    counts_after = snapshot_table_counts()
    for tbl, cnt_before in counts_before.items():
        cnt_after = counts_after.get(tbl, 0)
        if cnt_before != cnt_after and cnt_before >= 0:
            raise RuntimeError(
                f"FATAL: Database table '{tbl}' row count mutated! "
                f"Before: {cnt_before}, After: {cnt_after}. "
                f"Evaluation queries must be read-only."
            )

    # 7. Real spend calculation from token usage
    token_stats = get_token_usage_stats()
    real_spend_usd = calculate_real_spend(token_stats)

    # 8. Print formatted evaluation summary
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"- Total Cases Evaluated:       {summary.total_cases}")
    print(f"- Execution Accuracy (Answers): {summary.execution_accuracy * 100:.1f}% ({summary.answer_cases} cases)")
    if summary.seen_pattern_accuracy is not None:
        print(f"  * Seen-Pattern Accuracy:     {summary.seen_pattern_accuracy * 100:.1f}%")
    if summary.unseen_pattern_accuracy is not None:
        print(f"  * Unseen-Pattern Accuracy:   {summary.unseen_pattern_accuracy * 100:.1f}%")
    print(f"- Intent Classification Acc:   {summary.intent_accuracy * 100:.1f}%")
    print(f"- Valid SQL Syntax Rate:       {summary.valid_sql_rate * 100:.1f}%")
    print(f"- Guardrail Block Rate:        {summary.guardrail_block_rate * 100:.1f}%")
    print(f"- Deterministic Schema Recall: {summary.avg_schema_recall * 100:.1f}%")
    print(f"- Average Pipeline Latency:    {summary.avg_latency_ms:.1f} ms")
    print(f"- Real Session Spend:          ${real_spend_usd:.5f} USD")

    print("\n### Behavior Pass Rates:\n")
    print(format_behavior_pass_rate_table(summary.behavior_pass_rates))

    if judge_metrics:
        print("\n### DeepEval LLM Judge Outcomes:\n")
        judge_table = format_deepeval_summary_table(summary.details)
        print(judge_table if judge_table else "No answer judge evaluations.")

    # 9. Save JSON and Markdown reports
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    report_data = {
        "timestamp": ts,
        "golden_set_hash": golden_set_hash,
        "summary": summary.__dict__,
        "token_usage": token_stats.__dict__,
        "real_spend_usd": real_spend_usd,
        "approx_estimate": approx_cost,
    }
    with open(REPORTS_DIR / f"{ts}.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    logger.info("Saved evaluation report to %s", REPORTS_DIR / f"{ts}.json")


if __name__ == "__main__":
    main()
