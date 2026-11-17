"""OpenAI custom LLM judge for DeepEval metrics with configurable model selection."""

import asyncio
import json
import logging
from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import BaseModel

from src.config import settings
from src.llm.llm_client import openai_client

logger = logging.getLogger(__name__)


class OpenAIJudge(DeepEvalBaseLLM):
    """DeepEval custom LLM judge powered by OpenAI API using configurable cost-efficient models.

    Defaults explicitly to settings.JUDGE_MODEL (e.g., gpt-4o-mini) and routes all calls
    through the unified OpenAIClient for consistent retry backoff, timeouts, and token tracking.
    """

    def __init__(self, model: str | None = None, *args: Any, **kwargs: Any) -> None:
        self._model_name = model or settings.JUDGE_MODEL
        super().__init__(self._model_name, *args, **kwargs)

    def load_model(self, *args: Any, **kwargs: Any) -> Any:
        """Returns the configured OpenAI provider client."""
        return openai_client

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        """Returns the active judge model name."""
        return self._model_name

    def generate(self, prompt: str, schema: Any = None, *args: Any, **kwargs: Any) -> Any:
        """Synchronously generates an evaluation response or structured JSON output."""
        if schema is not None:
            last_err = None
            for attempt in range(2):
                try:
                    parsed_data, _ = openai_client.generate_json(
                        prompt=prompt,
                        schema=schema,
                        model=self._model_name,
                    )
                    if isinstance(parsed_data, BaseModel):
                        return parsed_data
                    if isinstance(parsed_data, dict):
                        if isinstance(schema, type) and issubclass(schema, BaseModel):
                            return schema(**parsed_data)
                        return json.dumps(parsed_data)
                except Exception as exc:
                    last_err = exc
                    logger.warning(
                        "OpenAIJudge generate_json attempt %d failed: %s", attempt + 1, exc
                    )
            raise RuntimeError(
                f"OpenAIJudge generate_json failed after 2 attempts: {last_err}"
            ) from last_err

        resp = openai_client.generate(
            prompt=prompt,
            model=self._model_name,
            temperature=0.0,
        )
        return resp.text

    async def a_generate(self, prompt: str, schema: Any = None, *args: Any, **kwargs: Any) -> Any:
        """Asynchronously generates evaluation judgment via worker thread."""
        return await asyncio.to_thread(self.generate, prompt, schema, *args, **kwargs)
