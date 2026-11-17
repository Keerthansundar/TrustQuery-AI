"""OpenAI LLM & Embedding provider module with exponential backoff, model fallback, and token tracking."""

import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Any

import openai
from openai import OpenAI
from pydantic import BaseModel

from src.config import settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base exception for LLM errors."""


class LLMUnavailableError(LLMError):
    """Raised when the OpenAI API key is missing or calls fail after retries."""


@dataclass
class LLMResponse:
    """Metadata and content for LLM completions."""

    text: str
    model: str
    latency_ms: float
    fallback_used: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    key_label: str = "openai"  # Retained for trace and UI schema compatibility


@dataclass
class TokenUsageStats:
    """Session cumulative token usage and spend tracking."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    embedding_tokens: int = 0
    total_calls: int = 0
    estimated_cost_usd: float = 0.0


# Module-level cumulative token tracker
_usage_stats = TokenUsageStats()


def get_token_usage_stats() -> TokenUsageStats:
    """Returns current cumulative session token usage statistics."""
    return _usage_stats


def reset_token_usage_stats() -> None:
    """Resets session cumulative token usage counters."""
    global _usage_stats
    _usage_stats = TokenUsageStats()


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculates approximate cost based on official OpenAI pricing per 1M tokens."""
    m = model.lower()
    if "gpt-4o-mini" in m:
        # $0.15/1M input, $0.60/1M output
        return (prompt_tokens * 0.15 + completion_tokens * 0.60) / 1_000_000.0
    if "gpt-4.1-mini" in m:
        # $0.40/1M input, $1.60/1M output
        return (prompt_tokens * 0.40 + completion_tokens * 1.60) / 1_000_000.0
    if "text-embedding-3-small" in m:
        # $0.02/1M tokens
        return (prompt_tokens * 0.02) / 1_000_000.0
    # Generic conservative default ($0.50/1M input, $2.00/1M output)
    return (prompt_tokens * 0.50 + completion_tokens * 2.00) / 1_000_000.0


class RateLimiter:
    """Client-side token bucket rate limiter to prevent bursts."""

    def __init__(self, requests_per_minute: int = 60) -> None:
        self.rpm = max(1, requests_per_minute)
        self.interval = 60.0 / self.rpm
        self.last_call: float = 0.0

    def wait(self) -> None:
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.interval:
            sleep_time = self.interval - elapsed
            time.sleep(sleep_time)
        self.last_call = time.time()


class OpenAIClient:
    """Unified client for OpenAI Chat Completions and Embeddings.

    Features:
    - Early missing key validation with actionable error message
    - Exponential backoff with random jitter on rate-limit (429) & transient errors
    - Request timeout protection
    - Optional single-key model fallback on rate-limit/unavailability
    - Token tracking and spend accounting
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._api_key = api_key
        self.timeout = timeout if timeout is not None else settings.OPENAI_TIMEOUT_SECONDS
        self.max_retries = max_retries if max_retries is not None else settings.OPENAI_MAX_RETRIES
        self.rate_limiter = rate_limiter or RateLimiter(settings.RATE_LIMIT_RPM)
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        """Returns initialized OpenAI client or raises LLMUnavailableError if key is missing."""
        key = self._api_key or settings.OPENAI_API_KEY
        if not key or key.strip() == "" or key.startswith("change_") or key.startswith("your-key"):
            raise LLMUnavailableError(
                "OPENAI_API_KEY is not configured. Please supply a valid OPENAI_API_KEY in your environment or .env file."
            )
        if self._client is None:
            self._client = OpenAI(api_key=key, timeout=self.timeout)
        return self._client

    def _execute_with_retry(self, call_fn, target_model: str) -> tuple[Any, str, bool]:
        """Executes API call with exponential backoff and optional model fallback.

        Returns (result, final_model_name, fallback_used).
        """
        client = self._get_client()
        retry_exceptions = (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
        )

        # 1. Attempt with target_model
        last_exception: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                self.rate_limiter.wait()
                result = call_fn(client, target_model)
                return result, target_model, False
            except retry_exceptions as exc:
                last_exception = exc
                wait_time = (1.5**attempt) + random.uniform(0.2, 1.0)
                logger.warning(
                    "OpenAI request to %s failed (attempt %d/%d: %s). Retrying in %.2fs...",
                    target_model,
                    attempt + 1,
                    self.max_retries,
                    exc,
                    wait_time,
                )
                time.sleep(wait_time)
            except Exception as unrecoverable:
                logger.error(
                    "OpenAI request to %s encountered unrecoverable error: %s",
                    target_model,
                    unrecoverable,
                )
                raise LLMUnavailableError(
                    f"OpenAI API call failed: {unrecoverable}"
                ) from unrecoverable

        # 2. Check if model fallback is configured on the same key
        fallback_model = settings.OPENAI_MODEL_FALLBACK.strip()
        if fallback_model and fallback_model != target_model:
            logger.info(
                "Primary model %s exhausted retries. Failing over to configured fallback model %s...",
                target_model,
                fallback_model,
            )
            for attempt in range(self.max_retries):
                try:
                    self.rate_limiter.wait()
                    result = call_fn(client, fallback_model)
                    return result, fallback_model, True
                except retry_exceptions as exc:
                    last_exception = exc
                    wait_time = (1.5**attempt) + random.uniform(0.2, 1.0)
                    logger.warning(
                        "Fallback model %s failed (attempt %d/%d: %s). Retrying in %.2fs...",
                        fallback_model,
                        attempt + 1,
                        self.max_retries,
                        exc,
                        wait_time,
                    )
                    time.sleep(wait_time)
                except Exception as unrecoverable:
                    raise LLMUnavailableError(
                        f"Fallback model {fallback_model} call failed: {unrecoverable}"
                    ) from unrecoverable

        raise LLMUnavailableError(
            f"OpenAI API request failed after retries: {last_exception}"
        ) from last_exception

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.0,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generates text completion using OpenAI Chat Completions API."""
        target_model = model or settings.GEN_MODEL
        start_time = time.perf_counter()

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        def _call(client: OpenAI, active_model: str):
            kwargs: dict[str, Any] = {
                "model": active_model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            return client.chat.completions.create(**kwargs)

        resp, used_model, fallback_used = self._execute_with_retry(_call, target_model)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        content = ""
        if resp.choices and resp.choices[0].message:
            content = resp.choices[0].message.content or ""

        p_tokens = getattr(resp.usage, "prompt_tokens", 0) or 0
        c_tokens = getattr(resp.usage, "completion_tokens", 0) or 0
        t_tokens = getattr(resp.usage, "total_tokens", p_tokens + c_tokens) or (p_tokens + c_tokens)

        # Update cumulative tracker
        cost = _estimate_cost(used_model, p_tokens, c_tokens)
        _usage_stats.prompt_tokens += p_tokens
        _usage_stats.completion_tokens += c_tokens
        _usage_stats.total_tokens += t_tokens
        _usage_stats.total_calls += 1
        _usage_stats.estimated_cost_usd += cost

        logger.info(
            "OpenAI chat completion [%s]: %d prompt, %d comp toks, ~$%.5f (latency: %.1fms)",
            used_model,
            p_tokens,
            c_tokens,
            cost,
            latency_ms,
        )

        return LLMResponse(
            text=content,
            model=used_model,
            fallback_used=fallback_used,
            latency_ms=round(latency_ms, 2),
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=t_tokens,
        )

    def generate_json(
        self,
        prompt: str,
        schema: Any = None,
        model: str | None = None,
        temperature: float = 0.0,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> tuple[dict | Any, LLMResponse]:
        """Generates structured JSON completion using response_format={"type": "json_object"}.

        Parses JSON and optionally instantiates a Pydantic schema model.
        """
        target_model = model or settings.FAST_MODEL
        start_time = time.perf_counter()

        sys_msg = system or "You are a helpful structured JSON assistant."
        if "json" not in sys_msg.lower():
            sys_msg += "\nYou must output valid JSON."

        messages: list[dict[str, str]] = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ]

        def _call(client: OpenAI, active_model: str):
            kwargs: dict[str, Any] = {
                "model": active_model,
                "messages": messages,
                "temperature": temperature,
                "response_format": {"type": "json_object"},
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            return client.chat.completions.create(**kwargs)

        resp, used_model, fallback_used = self._execute_with_retry(_call, target_model)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        raw_text = ""
        if resp.choices and resp.choices[0].message:
            raw_text = resp.choices[0].message.content or "{}"

        # Clean markdown code fences if model returned ```json ... ```
        cleaned = (
            raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        )
        try:
            parsed = json.loads(cleaned)
        except Exception as exc:
            logger.warning("JSON decode failed (%s). Returning empty dict. Raw: %s", exc, raw_text)
            parsed = {}

        # Validate with Pydantic schema if provided
        if schema is not None:
            if (
                isinstance(schema, type)
                and issubclass(schema, BaseModel)
                and isinstance(parsed, dict)
            ):
                try:
                    parsed = schema(**parsed)
                except Exception as pydantic_exc:
                    logger.warning(
                        "Pydantic validation error (%s). Returning raw dict.", pydantic_exc
                    )

        p_tokens = getattr(resp.usage, "prompt_tokens", 0) or 0
        c_tokens = getattr(resp.usage, "completion_tokens", 0) or 0
        t_tokens = getattr(resp.usage, "total_tokens", p_tokens + c_tokens) or (p_tokens + c_tokens)

        cost = _estimate_cost(used_model, p_tokens, c_tokens)
        _usage_stats.prompt_tokens += p_tokens
        _usage_stats.completion_tokens += c_tokens
        _usage_stats.total_tokens += t_tokens
        _usage_stats.total_calls += 1
        _usage_stats.estimated_cost_usd += cost

        meta = LLMResponse(
            text=raw_text,
            model=used_model,
            fallback_used=fallback_used,
            latency_ms=round(latency_ms, 2),
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=t_tokens,
        )
        return parsed, meta

    def embed(
        self,
        texts: list[str],
        model: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 100,
    ) -> list[list[float]]:
        """Generates vector embeddings for a list of texts using OpenAI embeddings API."""
        if not texts:
            return []

        target_model = model or settings.EMBEDDING_MODEL
        target_dimensions = dimensions or settings.EMBEDDING_DIMENSIONS
        embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            def _call(client: OpenAI, active_model: str, current_batch=batch):
                return client.embeddings.create(
                    model=active_model,
                    input=current_batch,
                    dimensions=target_dimensions,
                )

            resp, used_model, _ = self._execute_with_retry(_call, target_model)

            # Sort by index to maintain ordering guarantees
            sorted_items = sorted(resp.data, key=lambda item: item.index)
            for item in sorted_items:
                embeddings.append(list(item.embedding))

            prompt_toks = getattr(getattr(resp, "usage", None), "prompt_tokens", 0) or 0
            cost = _estimate_cost(used_model, prompt_toks, 0)
            _usage_stats.embedding_tokens += prompt_toks
            _usage_stats.total_tokens += prompt_toks
            _usage_stats.total_calls += 1
            _usage_stats.estimated_cost_usd += cost

            logger.info(
                "OpenAI embeddings [%s]: %d texts, %d tokens, ~$%.6f",
                used_model,
                len(batch),
                prompt_toks,
                cost,
            )

        return embeddings


# Singleton global client instance
openai_client = OpenAIClient()
