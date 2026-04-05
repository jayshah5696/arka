from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from arka.config.models import LLMConfig
from arka.llm.models import LLMOutput, TokenUsage

CAPABILITIES: dict[str, dict[str, bool]] = {
    "openai": {
        "structured_output": True,
        "logprobs": True,
        "batch_api": True,
    }
}

Message = dict[str, str]
ClientFactory = Callable[[LLMConfig], OpenAI | Any]
SleepFn = Callable[[float], None]


class LLMClientError(RuntimeError):
    """Raised when the LLM client cannot complete a request."""


class LLMClient:
    def __init__(
        self,
        config: LLMConfig,
        client_factory: ClientFactory | None = None,
        sleep: SleepFn | None = None,
    ) -> None:
        self.config = config
        self._client_factory = client_factory or self._build_openai_client
        self._sleep = sleep or time.sleep
        self._client = self._client_factory(config)

    def complete(self, messages: Sequence[Message]) -> LLMOutput:
        started_at = time.perf_counter()

        @retry(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_fixed(1),
            retry=retry_if_exception_type(
                (
                    RateLimitError,
                    APIConnectionError,
                    APITimeoutError,
                    InternalServerError,
                )
            ),
            reraise=True,
            sleep=self._sleep,
        )
        def create_completion() -> Any:
            return self._client.chat.completions.create(
                model=self.config.model,
                messages=list(messages),
            )

        try:
            response = create_completion()
        except AuthenticationError as exc:
            raise LLMClientError(f"auth error: {exc}") from exc
        except BadRequestError as exc:
            raise LLMClientError(f"bad_request error: {exc}") from exc
        except (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
        ) as exc:
            raise LLMClientError(f"retryable_api_error: {exc}") from exc

        return self._to_output(
            response=response,
            latency_ms=self._latency_ms(started_at),
        )

    def complete_structured(
        self,
        messages: Sequence[Message],
        schema: type[BaseModel],
    ) -> LLMOutput:
        output = self.complete(messages=messages)
        if output.text is None:
            raise LLMClientError("structured output expected text but received none")
        try:
            parsed = schema.model_validate(
                json.loads(self._extract_json_text(output.text))
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LLMClientError(f"parse error: {exc}") from exc
        except ValueError as exc:
            raise LLMClientError(f"parse error: {exc}") from exc
        return output.model_copy(update={"parsed": parsed})

    def _extract_json_text(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
            if match is None:
                raise ValueError("Could not extract JSON from code fence")
            return match.group(1)
        json_object_match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if json_object_match is not None:
            return json_object_match.group(0)
        return stripped

    def complete_batch(
        self,
        batch: Sequence[Sequence[Message]],
        max_workers: int | None = None,
    ) -> list[LLMOutput]:
        worker_count = max_workers or len(batch) or 1
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(self.complete, list(messages)) for messages in batch
            ]
            return [future.result() for future in futures]

    def _build_openai_client(self, config: LLMConfig) -> OpenAI:
        default_headers: dict[str, str] = {}
        if config.openai_compatible is not None:
            if config.openai_compatible.referer is not None:
                default_headers["HTTP-Referer"] = str(config.openai_compatible.referer)
            if config.openai_compatible.title is not None:
                default_headers["X-Title"] = config.openai_compatible.title
        return OpenAI(
            api_key=config.api_key,
            base_url=str(config.base_url),
            timeout=config.timeout_seconds,
            max_retries=0,
            default_headers=default_headers or None,
        )

    def _to_output(self, response: Any, latency_ms: int) -> LLMOutput:
        usage = self._usage_from_response(response)
        choice = response.choices[0]
        content = choice.message.content
        return LLMOutput(
            text=content,
            parsed=None,
            usage=usage,
            finish_reason=choice.finish_reason,
            model=response.model,
            provider=self.config.provider,
            request_id=getattr(response, "id", None),
            latency_ms=latency_ms,
            error=None,
        )

    def _usage_from_response(self, response: Any) -> TokenUsage:
        usage = getattr(response, "usage", None)
        if usage is None:
            return TokenUsage()
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    def _latency_ms(self, started_at: float) -> int:
        return int((time.perf_counter() - started_at) * 1000)
