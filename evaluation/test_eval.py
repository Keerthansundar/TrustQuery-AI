"""DeepEval test runner compatible pytest file for Text-to-SQL evaluation."""

import json
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from evaluation.metrics import create_deepeval_metrics
from evaluation.openai_judge import OpenAIJudge
from src.config import settings

GOLDEN_SET_PATH = Path(__file__).resolve().parent.parent / "data" / "golden_set.json"


@pytest.fixture(scope="module")
def golden_cases():
    if not GOLDEN_SET_PATH.exists():
        pytest.skip("golden_set.json not found")
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        cases = json.load(f)
    return [c for c in cases if c.get("expected_behavior") == "answer"][:3]


@pytest.fixture(scope="module")
def deepeval_judge():
    if (
        not settings.OPENAI_API_KEY
        or settings.OPENAI_API_KEY.startswith("change_")
        or settings.OPENAI_API_KEY.startswith("your-key")
    ):
        pytest.skip("Valid OPENAI_API_KEY required for live DeepEval test run")
    return OpenAIJudge()


@pytest.mark.integration
@pytest.mark.skipif(
    not settings.OPENAI_API_KEY
    or settings.OPENAI_API_KEY.startswith("change_")
    or settings.OPENAI_API_KEY.startswith("your-key"),
    reason="Live OpenAI API key required for DeepEval judge tests",
)
def test_deepeval_case(golden_cases, deepeval_judge):
    """Executes DeepEval assertion test on a sample golden case."""
    if not golden_cases:
        pytest.skip("No golden answer cases found")

    case = golden_cases[0]
    metrics_map = create_deepeval_metrics(deepeval_judge)
    metrics_to_test = [
        metrics_map["contextual_relevancy"],
        metrics_map["answer_relevancy"],
    ]

    test_case = LLMTestCase(
        input=case["question"],
        actual_output="The query successfully retrieved and aggregated the requested business metrics.",
        expected_output="Detailed explanation of the business metric results.",
        retrieval_context=["public.orders", "public.order_items"],
    )

    assert_test(test_case, metrics_to_test)
