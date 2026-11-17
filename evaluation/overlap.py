"""Few-shot example overlap analysis comparing golden test cases with examples.yaml."""

import difflib
import re
from pathlib import Path
from typing import Any

import sqlglot
import yaml

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXAMPLES_PATH = DATA_DIR / "examples.yaml"


def tokenize_question(q: str) -> set[str]:
    """Extracts alphanumeric word tokens from a question string."""
    tokens = re.findall(r"\b\w+\b", q.lower())
    return set(tokens)


def compute_jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Computes Jaccard similarity between two token sets."""
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return round(intersection / union, 4) if union > 0 else 0.0


def normalize_sql_for_diff(sql: str) -> str:
    """Normalizes SQL query for whitespace- and case-tolerant difflib comparison."""
    if not sql or not sql.strip():
        return ""
    try:
        parsed = sqlglot.parse_one(sql, read="postgres")
        return parsed.sql(dialect="postgres").lower()
    except Exception:
        # Fallback to regex whitespace collapse
        return re.sub(r"\s+", " ", sql.strip().lower())


def analyze_few_shot_overlap(
    cases: list[dict[str, Any]], examples_path: Path | None = None
) -> dict[str, dict[str, Any]]:
    """Compares each golden case question and gold_sql with all few-shot examples.

    Flags cases as 'seen_pattern' if:
    - Question token Jaccard similarity >= 0.8, OR
    - SQL difflib ratio >= 0.8, OR
    - SQL is identical.
    """
    path = examples_path or EXAMPLES_PATH
    examples: list[dict[str, Any]] = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            examples = data.get("examples", [])

    overlap_map: dict[str, dict[str, Any]] = {}

    for c in cases:
        cid = c["id"]
        q_tokens = tokenize_question(c.get("question", ""))
        c_sql_norm = normalize_sql_for_diff(c.get("gold_sql", ""))

        max_jaccard = 0.0
        max_sql_ratio = 0.0
        identical_sql = False
        matched_ex_id = ""

        for ex in examples:
            ex_id = ex.get("id", "")
            ex_tokens = tokenize_question(ex.get("question", ""))
            jaccard = compute_jaccard_similarity(q_tokens, ex_tokens)

            ex_sql_norm = normalize_sql_for_diff(ex.get("sql", ""))
            is_ident = bool(c_sql_norm and ex_sql_norm and c_sql_norm == ex_sql_norm)
            if is_ident:
                identical_sql = True

            sql_ratio = (
                difflib.SequenceMatcher(None, c_sql_norm, ex_sql_norm).ratio()
                if c_sql_norm and ex_sql_norm
                else 0.0
            )

            if (jaccard > max_jaccard) or (sql_ratio > max_sql_ratio) or is_ident:
                if jaccard > max_jaccard:
                    max_jaccard = jaccard
                if sql_ratio > max_sql_ratio:
                    max_sql_ratio = sql_ratio
                matched_ex_id = ex_id

        seen = bool((max_jaccard >= 0.8) or (max_sql_ratio >= 0.8) or identical_sql)
        overlap_map[cid] = {
            "seen_pattern": seen,
            "max_question_jaccard": round(max_jaccard, 3),
            "max_sql_ratio": round(max_sql_ratio, 3),
            "identical_sql": identical_sql,
            "matched_example_id": matched_ex_id,
        }

    return overlap_map
