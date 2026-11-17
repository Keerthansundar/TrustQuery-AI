"""Unit tests for OpenAI LLM client, retry with backoff, token tracking, and model fallback."""

from unittest.mock import MagicMock, patch

import openai
import pytest
from pydantic import BaseModel

from src.config import settings
from src.llm.llm_client import (
    LLMUnavailableError,
    OpenAIClient,
    RateLimiter,
    get_token_usage_stats,
    reset_token_usage_stats,
)


@pytest.fixture(autouse=True)
def clean_token_stats():
    reset_token_usage_stats()
    yield
    reset_token_usage_stats()


def test_openai_client_missing_key_fails_early():
    client = OpenAIClient(api_key="")
    with patch.object(settings, "OPENAI_API_KEY", ""):
        with pytest.raises(LLMUnavailableError) as exc_info:
            client.generate("Hello")
        assert "OPENAI_API_KEY is not configured" in str(exc_info.value)


def test_openai_client_placeholder_key_fails_early():
    client = OpenAIClient(api_key="your-key-here")
    with pytest.raises(LLMUnavailableError) as exc_info:
        client.generate("Hello")
    assert "OPENAI_API_KEY is not configured" in str(exc_info.value)


def test_openai_client_primary_success():
    rate_limiter = RateLimiter(requests_per_minute=6000)
    client = OpenAIClient(
        api_key="sk-test-mock-key-1234567890",
        rate_limiter=rate_limiter,
        max_retries=2,
    )

    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "SELECT * FROM orders;"
    mock_resp.choices = [mock_choice]
    mock_resp.usage.prompt_tokens = 120
    mock_resp.usage.completion_tokens = 25
    mock_resp.usage.total_tokens = 145

    mock_sdk_client = MagicMock()
    mock_sdk_client.chat.completions.create.return_value = mock_resp
    client._client = mock_sdk_client

    resp = client.generate("Generate SQL query", model="gpt-4o-mini")

    assert resp.text == "SELECT * FROM orders;"
    assert resp.model == "gpt-4o-mini"
    assert resp.fallback_used is False
    assert resp.prompt_tokens == 120
    assert resp.completion_tokens == 25
    assert resp.total_tokens == 145
    assert resp.latency_ms >= 0.0

    # Verify cumulative token tracking
    stats = get_token_usage_stats()
    assert stats.total_calls == 1
    assert stats.prompt_tokens == 120
    assert stats.completion_tokens == 25
    assert stats.total_tokens == 145
    assert stats.estimated_cost_usd > 0.0


def test_openai_client_retry_on_rate_limit_success():
    rate_limiter = RateLimiter(requests_per_minute=6000)
    client = OpenAIClient(
        api_key="sk-test-mock-key-1234567890",
        rate_limiter=rate_limiter,
        max_retries=3,
    )

    # First attempt raises RateLimitError, second attempt succeeds
    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Recovered response after 429"
    mock_resp.choices = [mock_choice]
    mock_resp.usage.prompt_tokens = 50
    mock_resp.usage.completion_tokens = 10
    mock_resp.usage.total_tokens = 60

    rate_limit_err = openai.RateLimitError(
        message="Rate limit exceeded",
        response=MagicMock(status_code=429, headers={}),
        body=None,
    )

    mock_sdk_client = MagicMock()
    mock_sdk_client.chat.completions.create.side_effect = [rate_limit_err, mock_resp]
    client._client = mock_sdk_client

    with patch("time.sleep", return_value=None):
        resp = client.generate("Test prompt")

    assert resp.text == "Recovered response after 429"
    assert resp.fallback_used is False
    assert mock_sdk_client.chat.completions.create.call_count == 2


def test_openai_client_model_fallback():
    rate_limiter = RateLimiter(requests_per_minute=6000)
    client = OpenAIClient(
        api_key="sk-test-mock-key-1234567890",
        rate_limiter=rate_limiter,
        max_retries=2,
    )

    rate_limit_err = openai.RateLimitError(
        message="Model rate limit reached",
        response=MagicMock(status_code=429, headers={}),
        body=None,
    )

    fallback_resp = MagicMock()
    fallback_choice = MagicMock()
    fallback_choice.message.content = "Fallback model response"
    fallback_resp.choices = [fallback_choice]
    fallback_resp.usage.prompt_tokens = 30
    fallback_resp.usage.completion_tokens = 5
    fallback_resp.usage.total_tokens = 35

    def mock_create(**kwargs):
        if kwargs.get("model") == "gpt-4o-mini":
            raise rate_limit_err
        if kwargs.get("model") == "gpt-4.1-mini":
            return fallback_resp
        raise RuntimeError("Unknown model")

    mock_sdk_client = MagicMock()
    mock_sdk_client.chat.completions.create.side_effect = mock_create
    client._client = mock_sdk_client

    with patch.object(settings, "OPENAI_MODEL_FALLBACK", "gpt-4.1-mini"):
        with patch("time.sleep", return_value=None):
            resp = client.generate("Hello", model="gpt-4o-mini")

    assert resp.text == "Fallback model response"
    assert resp.model == "gpt-4.1-mini"
    assert resp.fallback_used is True


def test_openai_client_all_retries_fail_raises_unavailable():
    rate_limiter = RateLimiter(requests_per_minute=6000)
    client = OpenAIClient(
        api_key="sk-test-mock-key-1234567890",
        rate_limiter=rate_limiter,
        max_retries=2,
    )

    conn_err = openai.APIConnectionError(request=MagicMock())
    mock_sdk_client = MagicMock()
    mock_sdk_client.chat.completions.create.side_effect = conn_err
    client._client = mock_sdk_client

    with patch.object(settings, "OPENAI_MODEL_FALLBACK", ""):
        with patch("time.sleep", return_value=None):
            with pytest.raises(LLMUnavailableError):
                client.generate("Test prompt")


def test_openai_client_structured_json():
    client = OpenAIClient(api_key="sk-test-mock-key-1234567890", max_retries=1)

    class SampleSchema(BaseModel):
        intent: str
        confidence: float

    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"intent": "data_question", "confidence": 0.99}'
    mock_resp.choices = [mock_choice]
    mock_resp.usage.prompt_tokens = 80
    mock_resp.usage.completion_tokens = 15
    mock_resp.usage.total_tokens = 95

    mock_sdk_client = MagicMock()
    mock_sdk_client.chat.completions.create.return_value = mock_resp
    client._client = mock_sdk_client

    parsed, meta = client.generate_json(
        prompt="Classify intent",
        schema=SampleSchema,
    )

    assert isinstance(parsed, SampleSchema)
    assert parsed.intent == "data_question"
    assert parsed.confidence == 0.99
    assert meta.prompt_tokens == 80


def test_openai_client_generate_json_adds_missing_json_instruction():
    """Confirms generate_json injects 'JSON' into system message when omitted."""
    client = OpenAIClient(api_key="sk-test-mock-key-1234567890", max_retries=1)

    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"status": "ok"}'
    mock_resp.choices = [mock_choice]
    mock_resp.usage.prompt_tokens = 20
    mock_resp.usage.completion_tokens = 5
    mock_resp.usage.total_tokens = 25

    mock_sdk_client = MagicMock()
    mock_sdk_client.chat.completions.create.return_value = mock_resp
    client._client = mock_sdk_client

    # Call with system message that deliberately lacks the word 'json'
    parsed, _ = client.generate_json(
        prompt="Check status",
        system="You are a status reporting agent.",
    )

    assert parsed == {"status": "ok"}
    create_call = mock_sdk_client.chat.completions.create.call_args[1]
    assert create_call["response_format"] == {"type": "json_object"}
    system_msg = create_call["messages"][0]["content"]
    assert "json" in system_msg.lower()
    assert system_msg == "You are a status reporting agent.\nYou must output valid JSON."


def test_openai_client_generate_json_preserves_existing_json_instruction():
    """Confirms generate_json preserves instructions when 'json' is already present."""
    client = OpenAIClient(api_key="sk-test-mock-key-1234567890", max_retries=1)

    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"result": 42}'
    mock_resp.choices = [mock_choice]
    mock_resp.usage.prompt_tokens = 20
    mock_resp.usage.completion_tokens = 5
    mock_resp.usage.total_tokens = 25

    mock_sdk_client = MagicMock()
    mock_sdk_client.chat.completions.create.return_value = mock_resp
    client._client = mock_sdk_client

    # Call with system message that already includes 'json'
    parsed, _ = client.generate_json(
        prompt="Compute value",
        system="Produce a valid JSON object with the calculated result.",
    )

    assert parsed == {"result": 42}
    create_call = mock_sdk_client.chat.completions.create.call_args[1]
    assert create_call["response_format"] == {"type": "json_object"}
    system_msg = create_call["messages"][0]["content"]
    assert system_msg == "Produce a valid JSON object with the calculated result."


def test_openai_client_embed():
    client = OpenAIClient(api_key="sk-test-mock-key-1234567890", max_retries=1)

    mock_resp = MagicMock()
    item1 = MagicMock()
    item1.index = 0
    item1.embedding = [0.1] * 1536
    item2 = MagicMock()
    item2.index = 1
    item2.embedding = [0.2] * 1536
    mock_resp.data = [item2, item1]  # out of order to verify sorting
    mock_resp.usage.prompt_tokens = 40

    mock_sdk_client = MagicMock()
    mock_sdk_client.embeddings.create.return_value = mock_resp
    client._client = mock_sdk_client

    embeddings = client.embed(["first", "second"])

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 1536
    assert embeddings[0][0] == 0.1
    assert embeddings[1][0] == 0.2

    stats = get_token_usage_stats()
    assert stats.embedding_tokens == 40
