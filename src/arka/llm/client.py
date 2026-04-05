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
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from arka.common.concurrency import bounded_worker_count
from arka.config.models import LLMConfig
from arka.llm.models import LLMOutput, TokenUsage
from arka.llm.openai_client import build_openai_client

CAPABILITIES: dict[str, dict[str, bool | tuple[str, ...]]] = {
    "openai": {
        "structured_output": True,
        "structured_output_strategies": (
            "openai_compatible_json_schema",
            "openai_native_parse",
            "prompt_parse_fallback",
        ),
        "logprobs": True,
        "batch_api": True,
    }
}

Message = dict[str, str]
ClientFactory = Callable[[LLMConfig], OpenAI | Any]
SleepFn = Callable[[float], None]


class LLMClientError(RuntimeError):
    """Raised when the LLM client cannot complete a request."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class StructuredOutputStrategy:
    name: str

    def is_applicable(self, client: LLMClient) -> bool:
        return True

    def complete(
        self,
        client: LLMClient,
        messages: Sequence[Message],
        schema: type[BaseModel],
    ) -> LLMOutput | None:
        raise NotImplementedError


class OpenAICompatibleJsonSchemaStrategy(StructuredOutputStrategy):
    name = "openai_compatible_json_schema"

    def is_applicable(self, client: LLMClient) -> bool:
        flag = client.config.supports_json_schema
        if flag is not None:
            return flag
        # Legacy fallback: auto-detect known providers by URL
        return "openrouter.ai" in str(client.config.base_url)

    def complete(
        self,
        client: LLMClient,
        messages: Sequence[Message],
        schema: type[BaseModel],
    ) -> LLMOutput | None:
        started_at = time.perf_counter()

        @retry(
            stop=stop_after_attempt(client.config.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(
                (
                    RateLimitError,
                    APIConnectionError,
                    APITimeoutError,
                    InternalServerError,
                )
            ),
            reraise=True,
            sleep=client._sleep,
        )
        def create_completion() -> Any:
            return client._client.chat.completions.create(
                model=client.config.model,
                messages=list(messages),
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "strict": True,
                        "schema": schema.model_json_schema(),
                    },
                },
            )

        try:
            response = create_completion()
        except BadRequestError:
            return None
        except AuthenticationError as exc:
            raise LLMClientError("auth_error", str(exc)) from exc
        except (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
        ) as exc:
            raise LLMClientError("retryable_api_error", str(exc)) from exc

        output = client._to_output(
            response=response,
            latency_ms=client._latency_ms(started_at),
        )
        if output.text is None:
            raise LLMClientError(
                "invalid_structured_response",
                "structured output expected text but received none",
            )
        try:
            parsed = schema.model_validate(
                json.loads(client._extract_json_text(output.text))
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LLMClientError("parse_error", str(exc)) from exc
        except ValueError as exc:
            raise LLMClientError("parse_error", str(exc)) from exc
        return output.model_copy(update={"parsed": parsed})


class OpenAINativeParseStrategy(StructuredOutputStrategy):
    name = "openai_native_parse"

    def complete(
        self,
        client: LLMClient,
        messages: Sequence[Message],
        schema: type[BaseModel],
    ) -> LLMOutput | None:
        parse_api = getattr(
            getattr(getattr(client._client, "beta", None), "chat", None),
            "completions",
            None,
        )
        if parse_api is None or not hasattr(parse_api, "parse"):
            return None

        started_at = time.perf_counter()

        @retry(
            stop=stop_after_attempt(client.config.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(
                (
                    RateLimitError,
                    APIConnectionError,
                    APITimeoutError,
                    InternalServerError,
                )
            ),
            reraise=True,
            sleep=client._sleep,
        )
        def create_completion() -> Any:
            return parse_api.parse(
                model=client.config.model,
                messages=list(messages),
                response_format=schema,
            )

        try:
            response = create_completion()
        except BadRequestError:
            return None
        except AuthenticationError as exc:
            raise LLMClientError("auth_error", str(exc)) from exc
        except (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
        ) as exc:
            raise LLMClientError("retryable_api_error", str(exc)) from exc

        output = client._to_output(
            response=response,
            latency_ms=client._latency_ms(started_at),
        )
        parsed = getattr(response.choices[0].message, "parsed", None)
        if not isinstance(parsed, schema):
            raise LLMClientError(
                "invalid_structured_response",
                "provider returned no parsed object",
            )
        return output.model_copy(update={"parsed": parsed})


class PromptParseFallbackStrategy(StructuredOutputStrategy):
    name = "prompt_parse_fallback"

    def complete(
        self,
        client: LLMClient,
        messages: Sequence[Message],
        schema: type[BaseModel],
    ) -> LLMOutput | None:
        output = client.complete(messages=messages)
        if output.text is None:
            raise LLMClientError(
                "invalid_structured_response",
                "structured output expected text but received none",
            )
        try:
            parsed = schema.model_validate(
                json.loads(client._extract_json_text(output.text))
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LLMClientError("parse_error", str(exc)) from exc
        except ValueError as exc:
            raise LLMClientError("parse_error", str(exc)) from exc
        return output.model_copy(update={"parsed": parsed})


class LLMClient:
    def __init__(
        self,
        config: LLMConfig,
        client_factory: ClientFactory | None = None,
        sleep: SleepFn | None = None,
    ) -> None:
        self.config = config
        self._client_factory = client_factory or build_openai_client
        self._sleep = sleep or time.sleep
        self._client = self._client_factory(config)
        self._structured_output_strategies: tuple[StructuredOutputStrategy, ...] = (
            OpenAICompatibleJsonSchemaStrategy(),
            OpenAINativeParseStrategy(),
            PromptParseFallbackStrategy(),
        )

    def complete(self, messages: Sequence[Message]) -> LLMOutput:
        started_at = time.perf_counter()

        @retry(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=8),
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
            raise LLMClientError("auth_error", str(exc)) from exc
        except BadRequestError as exc:
            raise LLMClientError("bad_request_error", str(exc)) from exc
        except (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
        ) as exc:
            raise LLMClientError("retryable_api_error", str(exc)) from exc

        return self._to_output(
            response=response,
            latency_ms=self._latency_ms(started_at),
        )

    def complete_structured(
        self,
        messages: Sequence[Message],
        schema: type[BaseModel],
    ) -> LLMOutput:
        for strategy in self._structured_output_strategies:
            if not strategy.is_applicable(self):
                continue
            output = strategy.complete(self, messages=messages, schema=schema)
            if output is not None:
                return output
        raise LLMClientError(
            "invalid_structured_response",
            "no structured output strategy produced a result",
        )

    def _extract_json_text(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
            if match is None:
                raise ValueError("Could not extract JSON from code fence")
            return match.group(1)
        # Greedy regex: matches from first '{' to last '}' in the text.
        # Known limitation: breaks on responses with multiple JSON objects
        # or significant text containing curly braces after the JSON block.
        # This is a last-resort fallback; provider-native strategies above
        # handle the common case without regex extraction.
        json_object_match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if json_object_match is not None:
            return json_object_match.group(0)
        return stripped

    def complete_batch(
        self,
        batch: Sequence[Sequence[Message]],
        max_workers: int | None = None,
    ) -> list[LLMOutput]:
        worker_count = bounded_worker_count(len(batch), max_workers)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(self.complete, list(messages)) for messages in batch
            ]
            return [future.result() for future in futures]

    def _to_output(self, response: Any, latency_ms: int) -> LLMOutput:
        usage = self._usage_from_response(response)
        choice = response.choices[0]
        content = getattr(choice.message, "content", None)
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
        # Some providers (e.g. OpenRouter) include cost in the usage object.
        raw_cost = getattr(usage, "total_cost", None)
        cost_usd = float(raw_cost) if raw_cost is not None else None
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
        )

    def _latency_ms(self, started_at: float) -> int:
        return int((time.perf_counter() - started_at) * 1000)
