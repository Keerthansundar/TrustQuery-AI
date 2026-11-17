"""Offline unit tests for evaluation metrics, behavior scoring, AST assertion guards, and history seeding."""

from decimal import Decimal

import pytest

from evaluation.metrics import (
    compare_result_sets,
    compute_schema_recall,
    normalize_value,
    row_is_superset,
    score_behavior,
)
from evaluation.run_eval import assert_single_select_or_with
from src.controller import Controller
from src.memory import clear_history, load_history
from src.state import Response


def test_number_normalization():
    """Verifies that int, float, and Decimal normalize consistently to float with precision rounding."""
    assert normalize_value(42) == 42.0
    assert normalize_value(3.14159) == 3.1416
    assert normalize_value(Decimal("19.99")) == 19.99
    assert normalize_value(None) is None
    assert normalize_value("  Texas  ") == "texas"


def test_superset_row_matching():
    """Verifies that actual rows whose value multiset is a superset of gold rows match properly."""
    gold_row = {"id": 1, "name": "Nexora"}
    act_row_superset = {"id": 1, "name": "Nexora", "extra_col": 99.5, "flag": True}
    act_row_missing = {"id": 1}  # Missing 'name'
    act_row_mismatch = {"id": 1, "name": "Other"}

    assert row_is_superset(act_row_superset, gold_row) is True
    assert row_is_superset(act_row_missing, gold_row) is False
    assert row_is_superset(act_row_mismatch, gold_row) is False

    # Multi-row distinct bipartite matching
    gold_rows = [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]
    act_rows = [
        {"id": 2, "name": "B", "extra": "foo"},
        {"id": 1, "name": "A", "extra": "bar"},
    ]
    assert compare_result_sets(act_rows, gold_rows) is True

    # Duplicate match conflict: two gold rows should not match the same single actual row
    act_rows_conflict = [
        {"id": 1, "name": "A", "extra": "foo"},
        {"id": 3, "name": "C", "extra": "bar"},
    ]
    assert compare_result_sets(act_rows_conflict, gold_rows) is False


def test_metric_values_only_mode():
    """Verifies compare_mode='metric_values_only' for tie-sensitive ranking comparisons."""
    gold_rows = [
        {"brand": "HomeNest", "total_quantity": 4006},
        {"brand": "Pulsar", "total_quantity": 1920},
    ]
    # Actual has different alias names or slight text variation, but same sorted numeric values
    act_rows_matching_nums = [
        {"b": "HomeNest", "qty": 4006.0},
        {"b": "Pulsar", "qty": 1920.0},
    ]
    act_rows_mismatched_nums = [
        {"b": "HomeNest", "qty": 4000.0},
        {"b": "Pulsar", "qty": 1920.0},
    ]

    assert (
        compare_result_sets(
            act_rows_matching_nums, gold_rows, compare_mode="metric_values_only"
        )
        is True
    )
    assert (
        compare_result_sets(
            act_rows_mismatched_nums, gold_rows, compare_mode="metric_values_only"
        )
        is False
    )


def test_schema_recall():
    """Verifies deterministic schema recall computed by parsing gold_sql with sqlglot."""
    gold_sql = "SELECT p.product_name, c.category_name FROM public.products p JOIN public.categories c ON p.category_id = c.category_id;"

    # Context containing both tables and columns
    full_ctx = [
        {"content": "Table products has columns product_name, category_id."},
        {"content": "Table categories has columns category_name, category_id."},
    ]
    recall_full = compute_schema_recall(gold_sql, full_ctx)
    assert recall_full > 0.8

    # Partial context
    partial_ctx = [{"content": "Table products has columns product_name."}]
    recall_partial = compute_schema_recall(gold_sql, partial_ctx)
    assert 0.0 < recall_partial < recall_full

    # Empty query should default to 1.0
    assert compute_schema_recall("", full_ctx) == 1.0


def test_behavior_scorers():
    """Verifies independent pass/fail scoring for all behavior types using fake responses."""
    # 1. Chitchat
    chitchat_case = {"id": "c1", "expected_behavior": "chitchat"}
    resp_chitchat_pass = Response(
        answer="Hello! How can I assist you with database analysis today?",
        sql="",
        rows=[],
        intent="chitchat",
        rewritten_question="hello",
        trace=[],
        trace_id="t1",
    )
    passed, reason = score_behavior(chitchat_case, resp_chitchat_pass)
    assert passed is True
    assert "Chitchat intent" in reason

    resp_chitchat_fail = Response(
        answer="Hello!",
        sql="SELECT 1;",
        rows=[{"?column?": 1}],
        intent="data_question",
        rewritten_question="hello",
        trace=[],
        trace_id="t2",
    )
    passed, _ = score_behavior(chitchat_case, resp_chitchat_fail)
    assert passed is False

    # 2. Clarify
    clarify_case = {"id": "c2", "expected_behavior": "clarify"}
    resp_clarify_pass = Response(
        answer="Could you clarify which time period or department you are interested in?",
        sql="",
        rows=[],
        intent="unclear",
        rewritten_question="performance",
        trace=[],
        trace_id="t3",
    )
    passed, _ = score_behavior(clarify_case, resp_clarify_pass)
    assert passed is True

    # 3. Block Input (Adversarial)
    block_case = {"id": "c3", "expected_behavior": "block_input"}
    resp_block_pass = Response(
        answer="I cannot fulfill requests that attempt to override safety instructions.",
        sql="",
        rows=[],
        intent="security_blocked",
        rewritten_question="drop table",
        trace=[],
        trace_id="t4",
    )
    passed, _ = score_behavior(block_case, resp_block_pass, table_counts_intact=True)
    assert passed is True

    # 4. Read Only Refusal
    refusal_case = {"id": "c4", "expected_behavior": "read_only_refusal"}
    resp_refusal_pass = Response(
        answer="I can only execute read-only queries. Modifying the database is not permitted.",
        sql="DELETE FROM orders;",
        rows=[],
        intent="data_question",
        rewritten_question="delete all orders",
        trace=[],
        trace_id="t5",
        error="WRITE_ATTEMPT",
    )
    passed, _ = score_behavior(refusal_case, resp_refusal_pass, table_counts_intact=True)
    assert passed is True

    # 5. PII Protected
    pii_case = {"id": "c5", "expected_behavior": "pii_protected"}
    resp_pii_safe = Response(
        answer="Here are the top customers by segment.",
        sql="SELECT customer_id, first_name FROM customers LIMIT 5;",
        rows=[{"customer_id": 1, "first_name": "Alice"}],
        intent="data_question",
        rewritten_question="top customers",
        trace=[],
        trace_id="t6",
    )
    passed, _ = score_behavior(pii_case, resp_pii_safe)
    assert passed is True

    resp_pii_leaked = Response(
        answer="Contact Alice at alice@example.com",
        sql="SELECT customer_id, email FROM customers;",
        rows=[{"customer_id": 1, "email": "alice@example.com"}],
        intent="data_question",
        rewritten_question="customer emails",
        trace=[],
        trace_id="t7",
    )
    passed, _ = score_behavior(pii_case, resp_pii_leaked)
    assert passed is False


def test_single_select_or_with_assertion():
    """Verifies that assert_single_select_or_with strictly accepts single SELECT/WITH and rejects mutations/multi-statements."""
    # Valid statements
    assert_single_select_or_with("SELECT * FROM public.customers;")
    assert_single_select_or_with("WITH cte AS (SELECT 1 AS val) SELECT * FROM cte;")
    assert_single_select_or_with("SELECT 1 UNION SELECT 2;")
    assert_single_select_or_with("")  # Empty is safe

    # Invalid statements: Mutations
    with pytest.raises(ValueError, match="must be a SELECT or WITH statement"):
        assert_single_select_or_with("DELETE FROM public.orders;")

    with pytest.raises(ValueError, match="must be a SELECT or WITH statement"):
        assert_single_select_or_with("DROP TABLE public.categories CASCADE;")

    with pytest.raises(ValueError, match="must be a SELECT or WITH statement"):
        assert_single_select_or_with("INSERT INTO public.orders DEFAULT VALUES;")

    # Invalid statements: Multi-statements
    with pytest.raises(ValueError, match="Expected exactly 1 statement"):
        assert_single_select_or_with("SELECT 1; SELECT 2;")


def test_history_seeding():
    """Verifies controller history seeding for conversational follow-up questions."""
    session_id = "test_eval_seeding_sess_123"
    clear_history(session_id)

    ctrl = Controller()
    prior_turns = [
        {
            "question": "What was the total net revenue from corporate customers?",
            "sql": "SELECT SUM(line_revenue) FROM orders WHERE segment = 'corporate';",
        }
    ]

    ctrl.seed_history(session_id, prior_turns)
    history = load_history(session_id, n=5)

    assert len(history) == 1
    assert (
        history[0].question
        == "What was the total net revenue from corporate customers?"
    )
    assert (
        history[0].sql
        == "SELECT SUM(line_revenue) FROM orders WHERE segment = 'corporate';"
    )

    clear_history(session_id)
