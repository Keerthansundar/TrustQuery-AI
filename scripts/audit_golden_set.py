"""Audit golden dataset against PostgreSQL database using read-only analyst role.

Reports for every expected_behavior == 'answer' case:
- Execution error
- Row count
- Empty result
- All-NULL result
- Rows exceeding MAX_ROWS (default: 100)
- Whether LIMIT boundary has ties (re-run with LIMIT n+1 using sqlglot, compare row n and n+1 ORDER BY key)
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp
from tabulate import tabulate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.db import db_manager

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_golden_set")

GOLDEN_SET_PATH = PROJECT_ROOT / "data" / "golden_set.json"


def assert_single_select_or_with(sql: str) -> None:
    """Asserts that SQL query is strictly a single SELECT or WITH statement."""
    if not sql or not sql.strip():
        return
    statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    if len(statements) != 1:
        raise ValueError(f"Expected exactly 1 statement, got {len(statements)}")
    stmt = statements[0]
    if not isinstance(stmt, (exp.Select, exp.Union)):
        raise ValueError(f"Statement must be SELECT or WITH, got {type(stmt).__name__}")


def check_limit_ties(sql: str, run_query_fn) -> tuple[bool, str]:
    """Checks whether a query with LIMIT n has ties between row n and row n+1."""
    try:
        parsed = sqlglot.parse_one(sql, read="postgres")
    except Exception as exc:
        return False, f"AST Parse Error: {exc}"

    limit_node = parsed.find(exp.Limit)
    if not limit_node or not limit_node.expression:
        return False, "No LIMIT"

    try:
        n = int(str(limit_node.expression.this))
    except (ValueError, TypeError):
        return False, "Non-integer LIMIT"

    order_node = parsed.find(exp.Order)
    if not order_node:
        return True, f"LIMIT {n} without ORDER BY (Arbitrary Cutoff)"

    order_keys: list[str] = []
    for o_exp in order_node.expressions:
        if isinstance(o_exp, exp.Ordered):
            col_target = o_exp.this
            if isinstance(col_target, exp.Column):
                order_keys.append(col_target.name)
            elif isinstance(col_target, exp.Literal):
                order_keys.append(str(col_target.this))
            else:
                order_keys.append(col_target.sql(dialect="postgres"))
        else:
            order_keys.append(o_exp.sql(dialect="postgres"))

    modified_parsed = parsed.copy()
    mod_limit = modified_parsed.find(exp.Limit)
    mod_limit.set("expression", exp.Literal.number(n + 1))
    sql_n_plus_1 = modified_parsed.sql(dialect="postgres")

    try:
        rows_n_plus_1 = run_query_fn(sql_n_plus_1, max_rows=n + 1)
    except Exception as exc:
        return False, f"LIMIT n+1 exec error: {exc}"

    if len(rows_n_plus_1) <= n:
        return False, f"Under limit ({len(rows_n_plus_1)} <= {n})"

    row_n = rows_n_plus_1[n - 1]
    row_n_plus_1 = rows_n_plus_1[n]

    tied_keys = []
    for key in order_keys:
        val_n = row_n.get(key)
        val_next = row_n_plus_1.get(key)
        if val_n is None or val_next is None:
            clean_k = key.split(".")[-1].strip('"').lower()
            for rk, rv in row_n.items():
                if rk.lower() == clean_k:
                    val_n = rv
                    break
            for rk, rv in row_n_plus_1.items():
                if rk.lower() == clean_k:
                    val_next = rv
                    break

        if val_n is not None and val_next is not None and val_n == val_next:
            tied_keys.append(f"{key}={val_n}")

    if tied_keys:
        return True, f"TIE at rank {n}/{n+1} on {', '.join(tied_keys)}"
    return False, f"No tie (row {n} vs {n+1} distinct)"


def audit_golden_set() -> dict[str, Any]:
    """Runs audit on all golden set answer cases."""
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        cases = json.load(f)

    db_up = db_manager.check_health()
    print("=" * 110)
    print(f"Golden Set Audit Runner | Total Cases: {len(cases)}")
    print(
        f"Database Health: {'ONLINE (Connected to ' + settings.POSTGRES_HOST + ':' + str(settings.POSTGRES_PORT) + ')' if db_up else 'OFFLINE (PostgreSQL not reachable at ' + settings.POSTGRES_HOST + ':' + str(settings.POSTGRES_PORT) + ')'}"
    )
    print("=" * 110)

    results = []
    for c in cases:
        cid = c["id"]
        behavior = c.get("expected_behavior", "answer")
        gold_sql = c.get("gold_sql", "").strip()
        tags = c.get("tags", [])

        if behavior != "answer":
            results.append(
                {
                    "id": cid,
                    "behavior": behavior,
                    "tags": ", ".join(tags),
                    "ast_valid": "PASS" if not gold_sql else "PASS",
                    "rows": 0,
                    "status": "SKIP (Non-answer)",
                    "empty": "No",
                    "all_null": "No",
                    "exceeds_max": "No",
                    "limit_tie": "No",
                    "notes": f"Behavior: {behavior}",
                }
            )
            continue

        ast_valid = "PASS"
        ast_err = ""
        try:
            assert_single_select_or_with(gold_sql)
        except Exception as exc:
            ast_valid = "FAIL"
            ast_err = str(exc)

        if not db_up:
            has_limit = "LIMIT" in gold_sql.upper()
            results.append(
                {
                    "id": cid,
                    "behavior": behavior,
                    "tags": ", ".join(tags),
                    "ast_valid": ast_valid,
                    "rows": "N/A (DB down)",
                    "status": f"AST {ast_valid}" if ast_valid == "PASS" else f"ERR: {ast_err}",
                    "empty": "N/A",
                    "all_null": "N/A",
                    "exceeds_max": "N/A",
                    "limit_tie": "Has LIMIT" if has_limit else "No LIMIT",
                    "notes": "DB Offline - start Docker to run live SQL",
                }
            )
            continue

        rows: list[dict[str, Any]] = []
        exec_err = ""
        try:
            rows = db_manager.run_query(gold_sql, allow_pii=False, max_rows=settings.MAX_ROWS + 5)
        except Exception as exc:
            exec_err = str(exc)

        row_count = len(rows)
        is_empty = row_count == 0 if not exec_err else False
        exceeds_max = row_count > settings.MAX_ROWS

        is_all_null = False
        if row_count > 0:
            all_vals = [v for r in rows for v in r.values()]
            is_all_null = all(v is None for v in all_vals)

        has_ties, tie_details = check_limit_ties(gold_sql, db_manager.run_query)

        status_str = "OK"
        notes = []
        if exec_err:
            status_str = "EXEC_ERROR"
            notes.append(exec_err[:60])
        if is_empty:
            status_str = "EMPTY"
            notes.append("0 rows")
        if is_all_null:
            status_str = "ALL_NULL"
            notes.append("All columns NULL")
        if exceeds_max:
            status_str = "EXCEEDS_MAX"
            notes.append(f"{row_count} > {settings.MAX_ROWS}")
        if has_ties:
            notes.append(tie_details)

        results.append(
            {
                "id": cid,
                "behavior": behavior,
                "tags": ", ".join(tags),
                "ast_valid": ast_valid,
                "rows": row_count if not exec_err else 0,
                "status": status_str,
                "empty": "YES" if is_empty else "No",
                "all_null": "YES" if is_all_null else "No",
                "exceeds_max": "YES" if exceeds_max else "No",
                "limit_tie": "YES" if has_ties else "No",
                "notes": "; ".join(notes) if notes else "Clean",
            }
        )

    table_headers = [
        "ID",
        "Behavior",
        "Tags",
        "AST",
        "Rows",
        "Status",
        "Empty?",
        "All Null?",
        "> MAX_ROWS?",
        "Limit Tie?",
        "Notes",
    ]
    rows_data = [
        [
            r["id"],
            r["behavior"],
            r["tags"],
            r["ast_valid"],
            r["rows"],
            r["status"],
            r["empty"],
            r["all_null"],
            r["exceeds_max"],
            r["limit_tie"],
            r["notes"],
        ]
        for r in results
    ]

    print("\n" + tabulate(rows_data, headers=table_headers, tablefmt="github"))
    return {"db_up": db_up, "results": results}


if __name__ == "__main__":
    audit_golden_set()

