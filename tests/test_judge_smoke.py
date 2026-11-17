"""DeepEval judge live smoke test (requires live OpenAI key and PostgreSQL)."""

from pathlib import Path

import pytest
from deepeval.test_case import LLMTestCase

from evaluation.metrics import create_deepeval_metrics
from evaluation.openai_judge import OpenAIJudge
from src.config import settings

GOLDEN_SET_PATH = Path(__file__).resolve().parent.parent / "data" / "golden_set.json"


@pytest.fixture(scope="module")
def deepeval_judge():
    if (
        not settings.OPENAI_API_KEY
        or settings.OPENAI_API_KEY.startswith("change_")
        or settings.OPENAI_API_KEY.startswith("your-key")
    ):
        pytest.skip("Valid OPENAI_API_KEY required for live DeepEval judge smoke test")
    return OpenAIJudge()


@pytest.mark.integration
def test_judge_smoke(deepeval_judge):
    """Smoke tests the OpenAIJudge with an answer relevancy test case."""
    metrics_map = create_deepeval_metrics(deepeval_judge)
    answer_metric = metrics_map["answer_relevancy"]

    sample_tc = LLMTestCase(
        input="What is the total net sales generated in Texas?",
        actual_output="The total net sales generated across all qualifying orders in Texas is $14,250.50.",
        retrieval_context=["public.orders", "public.order_items", "public.customers"],
    )

    answer_metric.measure(sample_tc)
    assert answer_metric.score is not None
    assert answer_metric.score >= 0.0
