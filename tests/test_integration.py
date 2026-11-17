"""Integration tests requiring a live PostgreSQL 16 + pgvector database.

Marked with @pytest.mark.integration so they can run in CI with the Postgres service
or locally when the Docker database container is running.
"""

import json
from pathlib import Path

import pytest
import yaml

from src.config import settings
from src.db import QueryTooExpensiveError, db_manager
from src.retrieval.retriever import retriever

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_PATH = PROJECT_ROOT / "data" / "examples.yaml"
GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_set.json"


@pytest.fixture(autouse=True)
def check_db_connectivity():
    """Skips integration tests if the database container is not reachable."""
    if not db_manager.check_health():
        pytest.skip(
            f"PostgreSQL database not reachable on {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}. "
            "Start the Docker db service (`make up`) to run integration tests."
        )


def test_db_read_only_enforcement():
    """Verifies that the analyst role operates in a read-only transaction."""
    with pytest.raises(Exception) as exc_info:
        db_manager.run_query(
            "INSERT INTO public.categories (category_id, category_name) VALUES (9999, 'Temp');"
        )
    err = str(exc_info.value).lower()
    assert "read-only" in err or "permission denied" in err


def test_analyst_ro_pii_denial_at_db_level():
    """Verifies that analyst_ro cannot access customers.email at the database grant level."""
    if settings.ALLOW_PII:
        pytest.skip("ALLOW_PII is enabled, skipping analyst_ro denial test")

    with pytest.raises(Exception) as exc_info:
        db_manager.run_query("SELECT email FROM public.customers LIMIT 1;")
    err = str(exc_info.value).lower()
    assert "permission denied" in err or "cannot access" in err


def test_explain_cost_guard():
    """Verifies that an excessively expensive query exceeds MAX_PLAN_COST and is blocked."""
    # A massive 5-way cartesian join
    cartesian_sql = """
        SELECT count(*)
        FROM public.orders o1
        CROSS JOIN public.orders o2
        CROSS JOIN public.order_items oi1
        CROSS JOIN public.order_items oi2;
    """
    with pytest.raises(QueryTooExpensiveError):
        db_manager.run_query(cartesian_sql)


def test_all_examples_sql_execute_successfully():
    """Asserts that all 25 SQL examples from examples.yaml execute without errors on live DB."""
    assert EXAMPLES_PATH.exists()
    with open(EXAMPLES_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    examples = data.get("examples", [])
    assert len(examples) == 25, f"Expected exactly 25 examples, found {len(examples)}"

    for idx, ex in enumerate(examples, start=1):
        sql = ex["sql"]
        try:
            rows = db_manager.run_query(sql)
            assert isinstance(rows, list), f"Example #{idx} did not return list of rows"
        except Exception as exc:
            pytest.fail(f"Example #{idx} ('{ex['question']}') failed with: {exc}\nSQL:\n{sql}")


def test_all_golden_answer_sql_execute_successfully():
    """Asserts that all golden SQL queries in golden_set.json execute successfully on live DB."""
    assert GOLDEN_PATH.exists()
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        golden_cases = json.load(f)

    answer_cases = [
        c for c in golden_cases if c.get("expected_behavior") == "answer" and c.get("gold_sql")
    ]
    assert len(answer_cases) >= 30, (
        f"Expected at least 30 answer test cases, got {len(answer_cases)}"
    )

    for case in answer_cases:
        sql = case["gold_sql"]
        try:
            rows = db_manager.run_query(sql)
            assert isinstance(rows, list), f"Golden case {case['id']} did not return list of rows"
        except Exception as exc:
            pytest.fail(
                f"Golden case {case['id']} ('{case['question']}') failed with: {exc}\nSQL:\n{sql}"
            )


def test_retrieval_end_to_end_on_live_db():
    """Verifies that retriever retrieves relevant schema and example chunks."""
    chunks = retriever.retrieve("What is total net sales by customer segment?")
    assert len(chunks) > 0, "Retriever returned 0 chunks"
    chunk_titles = [c.title.lower() for c in chunks]
    # Should include customers or orders or revenue
    assert any("customer" in t or "order" in t or "revenue" in t for t in chunk_titles)
