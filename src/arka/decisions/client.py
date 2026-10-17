from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import TYPE_CHECKING

import httpx
from pydantic import TypeAdapter, ValidationError
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from arka.common.concurrency import bounded_worker_count
from arka.decisions.backends import BACKENDS
from arka.decisions.models import (
    DecisionAnswer,
    DecisionConfig,
    DecisionQuestion,
    DecisionRequest,
    DecisionResult,
    ScoreQuestion,
    YesNoQuestion,
)
from arka.llm.models import TokenUsage

if TYPE_CHECKING:
    from arka.pipeline.checkpoint import CheckpointManager

ANSWER = TypeAdapter(DecisionAnswer)


class DecisionClientError(RuntimeError):
    """A safe error that never contains provider response bodies or credentials."""


def _retryable(exc: BaseException) -> bool:
    return isinstance(exc, httpx.TransportError) or (
        isinstance(exc, httpx.HTTPStatusError)
        and (exc.response.status_code in (429, 529) or exc.response.status_code >= 500)
    )


class DecisionClient:
    def __init__(
        self,
        config: DecisionConfig,
        *,
        checkpoint: CheckpointManager | None = None,
        executor: ThreadPoolExecutor | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if config.provider not in BACKENDS:
            raise ValueError(f"Unknown decision provider: {config.provider}")
        self.config = config
        self._backend = BACKENDS[config.provider]
        self._checkpoint = checkpoint
        self._executor = executor
        self._sleep = sleep
        self._lock = Lock()
        self._cache: dict[str, str] = {}
        self._http = httpx.Client(timeout=config.timeout_seconds, transport=transport)

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _key(self, state: str, question: DecisionQuestion) -> str:
        content = [
            self.config.provider,
            self.config.model,
            str(self.config.base_url),
            state,
            question.model_dump(exclude={"name", "exclude_from_panel"}),
        ]
        return hashlib.sha256(
            json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def decide(
        self, state: str, questions: Sequence[DecisionQuestion]
    ) -> DecisionResult:
        request = DecisionRequest(state=state, questions=list(questions))
        state = request.state[: self.config.max_state_chars]
        answers, missing, keys = {}, [], {}
        for question in request.questions:
            key = self._key(state, question)
            keys[question.name] = key
            if self._checkpoint is not None:
                cached = self._checkpoint.load_decision(key)
            else:
                with self._lock:
                    cached = self._cache.get(key)
            if cached is None:
                missing.append(question)
            else:
                answers[question.name] = ANSWER.validate_json(cached)
        result = DecisionResult(
            answers={},
            model=self.config.model,
            provider=self.config.provider,
            cache_hits=len(answers),
        )
        if missing:
            started = time.perf_counter()
            try:
                endpoint = (
                    str(self.config.base_url).rstrip("/")
                    + "/"
                    + self._backend.endpoint(self.config.model)
                )
                for attempt in Retrying(
                    stop=stop_after_attempt(self.config.max_retries),
                    wait=wait_random_exponential(multiplier=1, max=8),
                    retry=retry_if_exception(_retryable),
                    sleep=self._sleep,
                    reraise=True,
                ):
                    with attempt:
                        response = self._http.post(
                            endpoint,
                            json=self._backend.request(
                                state, missing, self.config.model
                            ),
                            headers={
                                "Authorization": f"Bearer {self.config.api_key.get_secret_value()}"
                            },
                        )
                        response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("response must be an object")
                if self.config.provider == "clef" and "result" in payload:
                    if payload.get("success") is not True:
                        raise ValueError("Clef request failed")
                    payload = payload["result"]
                parsed = self._backend.parse(payload)
                if set(parsed) != {q.name for q in missing}:
                    raise ValueError("answer names do not match questions")
                validated = {
                    name: ANSWER.validate_python(answer)
                    for name, answer in parsed.items()
                }
                for q in missing:
                    answer = validated[q.name]
                    if answer.type != q.type:
                        raise ValueError("answer type does not match question")
                    if not isinstance(q, YesNoQuestion):
                        expected = (
                            set(range(len(q.levels)))
                            if isinstance(q, ScoreQuestion)
                            else set(q.choices)
                        )
                        if set(answer.probabilities) != expected:
                            raise ValueError(
                                "probability options do not match question"
                            )
                usage = payload.get("usage")
                if not isinstance(usage, dict):
                    raise ValueError("response must include token usage")
                tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
                output_tokens = usage.get(
                    "output_tokens", usage.get("completion_tokens", 0)
                )
                if (
                    not isinstance(tokens, int)
                    or tokens < 0
                    or not isinstance(output_tokens, int)
                    or output_tokens < 0
                ):
                    raise ValueError("token usage must contain nonnegative integers")
                price = self.config.input_cost_per_million
                cost = usage.get("cost")
                if cost is not None and (
                    isinstance(cost, bool)
                    or not isinstance(cost, (int, float))
                    or not math.isfinite(cost)
                    or cost < 0
                ):
                    raise ValueError("reported cost must be finite and nonnegative")
                if cost is None and price is not None:
                    cost = tokens * price / 1_000_000
                result.usage = TokenUsage(
                    prompt_tokens=tokens,
                    completion_tokens=output_tokens,
                    total_tokens=tokens + output_tokens,
                    cost_usd=cost,
                )
                result.latency_ms = round((time.perf_counter() - started) * 1000)
                result.requests_sent = 1
                result.model = payload.get("model", self.config.model)
                for name, answer in validated.items():
                    serialized = answer.model_dump_json()
                    if self._checkpoint is not None:
                        self._checkpoint.save_decision(keys[name], serialized)
                    else:
                        with self._lock:
                            self._cache[keys[name]] = serialized
                answers.update(validated)
            except httpx.HTTPStatusError as exc:
                raise DecisionClientError(
                    f"decision provider returned HTTP {exc.response.status_code}"
                ) from None
            except httpx.TransportError:
                raise DecisionClientError(
                    "decision provider connection failed"
                ) from None
            except (KeyError, TypeError, ValueError, ValidationError):
                raise DecisionClientError(
                    "decision provider returned invalid or refused answers"
                ) from None
        result.answers = {q.name: answers[q.name] for q in request.questions}
        return result

    def decide_many(
        self, items: Sequence[DecisionRequest], max_workers: int | None = None
    ) -> list[DecisionResult]:
        workers = bounded_worker_count(len(items), max_workers)
        if self._executor is not None:
            # Submit in bounded groups so a larger shared pool respects this limit.
            results = []
            for start in range(0, len(items), workers):
                futures = [
                    self._executor.submit(self.decide, item.state, item.questions)
                    for item in items[start : start + workers]
                ]
                results.extend(f.result() for f in futures)
            return results
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(self.decide, item.state, item.questions)
                for item in items
            ]
            return [f.result() for f in futures]
