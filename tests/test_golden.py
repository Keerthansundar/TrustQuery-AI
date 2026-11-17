"""Tests for golden dataset integrity, schema conformity, and no-leakage against examples.yaml."""

import difflib
import json
import re
from pathlib import Path

import yaml

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def normalize_text(text: str) -> str:
    """Lowercase and remove non-alphanumeric characters for fuzzy matching."""
    text = text.lower()
    return re.sub(r"[^a-z0-9\s]", "", text).strip()


def test_golden_set_schema_and_count():
    golden_path = DATA_DIR / "golden_set.json"
    assert golden_path.exists(), "data/golden_set.json must exist"

    with open(golden_path, encoding="utf-8") as f:
        golden_cases = json.load(f)

    assert len(golden_cases) >= 40, f"Expected at least 40 eval cases, got {len(golden_cases)}"

    valid_behaviors = {
        "answer",
        "block_input",
        "read_only_refusal",
        "clarify",
        "chitchat",
        "pii_protected",
        "safe_handling",
    }
    valid_intents = {"data_question", "followup", "chitchat", "unclear"}

    ids = set()
    for case in golden_cases:
        assert "id" in case
        assert case["id"] not in ids, f"Duplicate ID: {case['id']}"
        ids.add(case["id"])

        assert "question" in case and case["question"].strip()
        assert "gold_sql" in case
        assert "expected_intent" in case
        assert case["expected_intent"] in valid_intents
        assert "expected_behavior" in case
        assert case["expected_behavior"] in valid_behaviors


def test_no_leakage_between_golden_and_examples():
    """Ensures no golden question duplicates or near-duplicates an example in examples.yaml."""
    golden_path = DATA_DIR / "golden_set.json"
    examples_path = DATA_DIR / "examples.yaml"

    assert golden_path.exists()
    assert examples_path.exists()

    with open(golden_path, encoding="utf-8") as f:
        golden_cases = json.load(f)

    with open(examples_path, encoding="utf-8") as f:
        examples_data = yaml.safe_load(f)

    example_questions = [normalize_text(ex["question"]) for ex in examples_data.get("examples", [])]

    for case in golden_cases:
        q_norm = normalize_text(case["question"])
        for ex_norm in example_questions:
            # Check exact match
            assert q_norm != ex_norm, (
                f"Exact leakage detected between golden and example: '{case['question']}'"
            )

            # Check fuzzy similarity ratio using difflib
            similarity = difflib.SequenceMatcher(None, q_norm, ex_norm).ratio()
            assert similarity < 0.88, (
                f"Fuzzy leakage detected (similarity {similarity:.2f} >= 0.88) between "
                f"Golden: '{case['question']}' and Example: '{ex_norm}'"
            )
