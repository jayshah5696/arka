from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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
from arka.llm.models import LLMOutput, SequenceScore, TokenUsage
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
        "response_scoring": True,
        "batch_api": True,
    }
}

Message = dict[str, str]
ClientFactory = Callable[[LLMConfig], OpenAI | Any]
SleepFn = Callable[[float], None]


def provider_supports_sequence_scoring(config: LLMConfig) -> bool:
    provider_capabilities = CAPABILITIES.get(config.provider, {})
    return bool(provider_capabilities.get("response_scoring", False))


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
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
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
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
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
            request_kwargs: dict[str, Any] = {
                "model": client.config.model,
                "messages": list(messages),
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "strict": True,
                        "schema": schema.model_json_schema(),
                    },
                },
            }
            if temperature is not None:
                request_kwargs["temperature"] = temperature
            if max_tokens is not None:
                request_kwargs["max_tokens"] = max_tokens
            return client._client.chat.completions.create(**request_kwargs)

        try:
            response = create_completion()
        except BadRequestError:
            return None
        except AuthenticationError as exc:
            # SECURITY: Drop original exception context to prevent leaking HTTP request context (headers, API keys) in tracebacks
            raise LLMClientError("auth_error", str(exc)) from None
        except (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
        ) as exc:
            # SECURITY: Drop original exception context to prevent leaking HTTP request context (headers, API keys) in tracebacks
            raise LLMClientError("retryable_api_error", str(exc)) from None

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
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
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
            request_kwargs: dict[str, Any] = {
                "model": client.config.model,
                "messages": list(messages),
                "response_format": schema,
            }
            if temperature is not None:
                request_kwargs["temperature"] = temperature
            if max_tokens is not None:
                request_kwargs["max_tokens"] = max_tokens
            return parse_api.parse(**request_kwargs)

        try:
            response = create_completion()
        except BadRequestError:
            return None
        except AuthenticationError as exc:
            # SECURITY: Drop original exception context to prevent leaking HTTP request context (headers, API keys) in tracebacks
            raise LLMClientError("auth_error", str(exc)) from None
        except (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
        ) as exc:
            # SECURITY: Drop original exception context to prevent leaking HTTP request context (headers, API keys) in tracebacks
            raise LLMClientError("retryable_api_error", str(exc)) from None

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
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMOutput | None:
        output = client.complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
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


def extract_json_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
        if match is None:
            raise ValueError("Could not extract JSON from code fence")
        return match.group(1)
    first_brace = stripped.find("{")
    if first_brace != -1:
        brace_count = 0
        in_string = False
        escape = False
        for i in range(first_brace, len(stripped)):
            ch = stripped[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string:
                if ch == "{":
                    brace_count += 1
                elif ch == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        return stripped[first_brace : i + 1]
    json_object_match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if json_object_match is not None:
        return json_object_match.group(0)
    return stripped


class LLMClient:
    def __init__(
        self,
        config: LLMConfig,
        client_factory: ClientFactory | None = None,
        sleep: SleepFn | None = None,
        cache_db_path: Path | None = None,
        read_cache: bool = True,
    ) -> None:
        self.config = config
        self._client_factory = client_factory or build_openai_client
        self._sleep = sleep or time.sleep
        self._client = self._client_factory(config)
        self.cache_db_path = cache_db_path
        self.read_cache = read_cache
        self.cached_calls = 0
        self._cache_lock = threading.Lock()
        self._cache_key_counts: dict[str, int] = {}
        self._structured_output_strategies: tuple[StructuredOutputStrategy, ...] = (
            OpenAICompatibleJsonSchemaStrategy(),
            OpenAINativeParseStrategy(),
            PromptParseFallbackStrategy(),
        )

    _extract_json_text = staticmethod(extract_json_text)

    def _compute_cache_key(
        self,
        messages: Sequence[Message],
        *,
        schema: type[BaseModel] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        payload = {
            "provider": self.config.provider,
            "model": self.config.model,
            "base_url": str(self.config.base_url),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": schema.model_json_schema()
            if schema is not None
            else None,
            "messages": list(messages),
        }
        content = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _next_cache_key(self, base_key: str) -> str:
        with self._cache_lock:
            occurrence = self._cache_key_counts.get(base_key, 0)
            self._cache_key_counts[base_key] = occurrence + 1
        return base_key if occurrence == 0 else f"{base_key}:{occurrence}"

    def _get_cached_llm(self, cache_key: str) -> str | None:
        if (
            not self.read_cache
            or not self.cache_db_path
            or not self.cache_db_path.exists()
        ):
            return None
        try:
            with sqlite3.connect(self.cache_db_path, check_same_thread=False) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT response_json FROM llm_cache WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()
                if row:
                    return row["response_json"]
        except Exception:
            return None
        return None

    def _save_cached_llm(self, cache_key: str, response_json: str) -> None:
        if not self.cache_db_path:
            return
        try:
            with sqlite3.connect(self.cache_db_path, check_same_thread=False) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS llm_cache (cache_key TEXT PRIMARY KEY, response_json TEXT NOT NULL)"
                )
                conn.execute(
                    "INSERT OR REPLACE INTO llm_cache (cache_key, response_json) VALUES (?, ?)",
                    (cache_key, response_json),
                )
        except Exception:
            pass

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMOutput:
        cache_key: str | None = None
        if self.cache_db_path is not None:
            base_key = self._compute_cache_key(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            cache_key = self._next_cache_key(base_key)
            if self.read_cache and self.cache_db_path.exists():
                cached_json = self._get_cached_llm(cache_key)
                if cached_json is not None:
                    self.cached_calls += 1
                    out = LLMOutput.model_validate_json(cached_json)
                    return out.model_copy(update={"latency_ms": 0})
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
            request_kwargs: dict[str, Any] = {
                "model": self.config.model,
                "messages": list(messages),
            }
            if temperature is not None:
                request_kwargs["temperature"] = temperature
            if max_tokens is not None:
                request_kwargs["max_tokens"] = max_tokens
            return self._client.chat.completions.create(**request_kwargs)

        try:
            response = create_completion()
        except AuthenticationError as exc:
            # SECURITY: Drop original exception context to prevent leaking HTTP request context (headers, API keys) in tracebacks
            raise LLMClientError("auth_error", str(exc)) from None
        except BadRequestError as exc:
            # SECURITY: Drop original exception context to prevent leaking HTTP request context (headers, API keys) in tracebacks
            raise LLMClientError("bad_request_error", str(exc)) from None
        except (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
        ) as exc:
            # SECURITY: Drop original exception context to prevent leaking HTTP request context (headers, API keys) in tracebacks
            raise LLMClientError("retryable_api_error", str(exc)) from None

        output = self._to_output(
            response=response,
            latency_ms=self._latency_ms(started_at),
        )
        if cache_key is not None:
            self._save_cached_llm(cache_key, output.model_dump_json(exclude={"parsed"}))
        return output

    def complete_structured(
        self,
        messages: Sequence[Message],
        schema: type[BaseModel],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMOutput:
        cache_key: str | None = None
        if self.cache_db_path is not None:
            base_key = self._compute_cache_key(
                messages,
                schema=schema,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            cache_key = self._next_cache_key(base_key)
            if self.read_cache and self.cache_db_path.exists():
                cached_json = self._get_cached_llm(cache_key)
                if cached_json is not None:
                    self.cached_calls += 1
                    out = LLMOutput.model_validate_json(cached_json)
                    parsed = None
                    if out.text is not None:
                        try:
                            parsed = schema.model_validate(
                                json.loads(self._extract_json_text(out.text))
                            )
                        except Exception:
                            pass
                    return out.model_copy(update={"latency_ms": 0, "parsed": parsed})
        for strategy in self._structured_output_strategies:
            if not strategy.is_applicable(self):
                continue
            output = strategy.complete(
                self,
                messages=messages,
                schema=schema,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if output is not None:
                if cache_key is not None:
                    serialized_output = output
                    if serialized_output.text is None and isinstance(
                        serialized_output.parsed, BaseModel
                    ):
                        serialized_output = serialized_output.model_copy(
                            update={"text": serialized_output.parsed.model_dump_json()}
                        )
                    self._save_cached_llm(
                        cache_key,
                        serialized_output.model_dump_json(exclude={"parsed"}),
                    )
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
        # Balanced brace counting to extract the first complete JSON object
        first_brace = stripped.find("{")
        if first_brace != -1:
            brace_count = 0
            in_string = False
            escape = False
            for i in range(first_brace, len(stripped)):
                ch = stripped[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if not in_string:
                    if ch == "{":
                        brace_count += 1
                    elif ch == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            return stripped[first_brace : i + 1]

        # Greedy fallback if balanced matching fails
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

    def supports_sequence_scoring(self) -> bool:
        return provider_supports_sequence_scoring(self.config)

    def score_response(
        self,
        *,
        messages: Sequence[Message],
        target_text: str,
    ) -> SequenceScore:
        if not self.supports_sequence_scoring():
            raise LLMClientError(
                "unsupported_capability",
                (
                    f"provider {self.config.provider!r} / model {self.config.model!r} "
                    "does not support response scoring"
                ),
            )
        scoring_messages = [
            *list(messages),
            {"role": "assistant", "content": target_text},
        ]
        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=scoring_messages,
                max_tokens=1,
                logprobs=True,
                echo=True,
            )
        except BadRequestError:
            # Fallback: some providers don't support echo — use plain completion
            # with logprobs on a re-generated response as an approximation.
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=list(messages),
                max_tokens=max(len(target_text.split()) * 2, 64),
                logprobs=True,
            )
        except AuthenticationError as exc:
            # SECURITY: Drop original exception context to prevent leaking HTTP request context (headers, API keys) in tracebacks
            raise LLMClientError("auth_error", str(exc)) from None

        logprobs_list = self._extract_logprobs(response)
        if not logprobs_list:
            raise LLMClientError(
                "unsupported_capability",
                "Provider returned no logprobs for scoring",
            )
        total_logprob = sum(logprobs_list)
        mean_logprob = total_logprob / len(logprobs_list)
        return SequenceScore(
            token_count=len(logprobs_list),
            mean_logprob=round(mean_logprob, 6),
            total_logprob=round(total_logprob, 6),
            provider=self.config.provider,
            model=response.model if hasattr(response, "model") else self.config.model,
        )

    def _extract_logprobs(self, response: Any) -> list[float]:
        """Extract per-token logprobs from an OpenAI-compatible response."""
        choice = response.choices[0] if response.choices else None
        if choice is None:
            return []
        logprobs_obj = getattr(choice, "logprobs", None)
        if logprobs_obj is None:
            return []
        # OpenAI v1 format: logprobs.content is a list of TokenLogprob objects
        content_logprobs = getattr(logprobs_obj, "content", None)
        if content_logprobs is not None:
            return [
                token_lp.logprob
                for token_lp in content_logprobs
                if hasattr(token_lp, "logprob")
            ]
        # Legacy format: logprobs.token_logprobs is a list of floats
        token_logprobs = getattr(logprobs_obj, "token_logprobs", None)
        if token_logprobs is not None:
            return [lp for lp in token_logprobs if lp is not None]
        return []

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
