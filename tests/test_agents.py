"""Unit tests for agents: planner, rewriter, SQL cleaner, and responder sanity checks."""

from src.agents.planner import Intent
from src.agents.responder import sanity_check
from src.agents.rewriter import is_self_contained
from src.agents.sql_generator import clean_sql_output


def test_is_self_contained_heuristics():
    # Standalone queries
    assert is_self_contained("What is our total net revenue for the past year?", history=[])
    assert is_self_contained(
        "Show top 5 products by revenue across all categories of products sold",
        history=[{"q": "hi"}],
    )

    # Followup queries with references
    assert not is_self_contained("What about last month?", history=[{"question": "Top products"}])
    assert not is_self_contained("And by category?", history=[{"question": "Top products"}])
    assert not is_self_contained("Show them instead", history=[{"question": "Top products"}])


def test_clean_sql_output():
    # 1. Fenced block with ```sql
    fenced_sql = "```sql\nSELECT product_id, product_name FROM public.products LIMIT 5;\n```"
    assert (
        clean_sql_output(fenced_sql)
        == "SELECT product_id, product_name FROM public.products LIMIT 5;"
    )

    # 2. Plain fenced block with ```
    plain_fenced = "```\nSELECT count(*) FROM public.orders;\n```"
    assert clean_sql_output(plain_fenced) == "SELECT count(*) FROM public.orders;"

    # 3. Extra text before and after fenced block
    extra_fenced = (
        "Here is the SQL query you requested for your analysis:\n\n"
        "```sql\nSELECT customer_id, count(*) FROM public.orders GROUP BY customer_id;\n```\n\n"
        "Hope this answers your question! Let me know if you need any adjustments."
    )
    assert (
        clean_sql_output(extra_fenced)
        == "SELECT customer_id, count(*) FROM public.orders GROUP BY customer_id;"
    )

    # 4. Extra text before and after without fences
    extra_bare = "Sure! You can run this: SELECT * FROM public.categories; which returns categories."
    assert clean_sql_output(extra_bare) == "SELECT * FROM public.categories;"

    # 5. Bare block
    bare_sql = "SELECT category_id, category_name FROM public.categories WHERE parent_category_id IS NULL;"
    assert (
        clean_sql_output(bare_sql)
        == "SELECT category_id, category_name FROM public.categories WHERE parent_category_id IS NULL;"
    )

    # 6. Already-clean SQL
    clean_sql = "SELECT order_id, shipping_fee FROM public.orders WHERE status = 'delivered';"
    assert clean_sql_output(clean_sql) == clean_sql

    # 7. CTE query with postgresql fence
    cte_sql = "```postgresql\nWITH rev AS (SELECT order_id, amount FROM public.payments) SELECT * FROM rev;\n```"
    assert (
        clean_sql_output(cte_sql)
        == "WITH rev AS (SELECT order_id, amount FROM public.payments) SELECT * FROM rev;"
    )


def test_responder_sanity_check():
    # Normal rows pass
    good_rows = [{"product_name": "Laptop", "total_revenue": 15000.50, "return_rate_pct": 5.2}]
    is_sane, hint = sanity_check(good_rows)
    assert is_sane
    assert hint == ""

    # Empty rows fail
    is_sane_empty, hint_empty = sanity_check([])
    assert not is_sane_empty
    assert "0 rows" in hint_empty

    # Negative revenue fails
    bad_rev_rows = [{"product_name": "Laptop", "total_revenue": -450.0}]
    is_sane_bad_rev, hint_bad_rev = sanity_check(bad_rev_rows)
    assert not is_sane_bad_rev
    assert "negative revenue" in hint_bad_rev

    # Percentage > 100 fails
    bad_pct_rows = [{"segment": "corporate", "return_rate_pct": 145.0}]
    is_sane_bad_pct, hint_bad_pct = sanity_check(bad_pct_rows)
    assert not is_sane_bad_pct
    assert "greater than 100" in hint_bad_pct


def test_intent_enum():
    assert Intent.DATA_QUESTION == "data_question"
    assert Intent.CHITCHAT == "chitchat"
    assert Intent.FOLLOWUP == "followup"
    assert Intent.UNCLEAR == "unclear"
