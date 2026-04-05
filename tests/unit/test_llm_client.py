from __future__ import annotations

import httpx
import pytest
from openai import APITimeoutError, AuthenticationError, BadRequestError, RateLimitError
from pydantic import BaseModel

from arka.config.models import LLMConfig
from arka.llm.client import LLMClient, LLMClientError, LLMOutput


class GreetingResponse(BaseModel):
    greeting: str


class FakeChatCompletionsAPI:
    def __init__(self, responses: list[object]) -> None:
        self._responses = responses
        self.calls = 0
        self.last_kwargs: dict[str, object] | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        response = self._responses[self.calls]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self.chat = type("ChatNamespace", (), {})()
        self.chat.completions = FakeChatCompletionsAPI(responses)


class FakeStructuredChatCompletionsAPI:
    def __init__(self, responses: list[object]) -> None:
        self._responses = responses
        self.calls = 0
        self.last_kwargs: dict[str, object] | None = None

    def parse(self, **kwargs):
        self.last_kwargs = kwargs
        response = self._responses[self.calls]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return response


class FakeStructuredClient:
    def __init__(
        self,
        structured_responses: list[object],
        fallback_responses: list[object] | None = None,
    ) -> None:
        self.beta = type("BetaNamespace", (), {})()
        self.beta.chat = type("BetaChatNamespace", (), {})()
        self.beta.chat.completions = FakeStructuredChatCompletionsAPI(
            structured_responses
        )
        self.chat = type("ChatNamespace", (), {})()
        self.chat.completions = FakeChatCompletionsAPI(fallback_responses or [])


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.message = FakeMessage(content)
        self.finish_reason = finish_reason


class FakeStructuredMessage:
    def __init__(self, parsed: BaseModel, content: str | None = None) -> None:
        self.parsed = parsed
        self.content = content


class FakeStructuredChoice:
    def __init__(self, parsed: BaseModel, finish_reason: str = "stop") -> None:
        self.message = FakeStructuredMessage(parsed=parsed, content=None)
        self.finish_reason = finish_reason


class FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class FakeResponse:
    def __init__(
        self,
        content: str,
        prompt_tokens: int = 5,
        completion_tokens: int = 7,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.choices = [FakeChoice(content)]
        self.usage = FakeUsage(prompt_tokens, completion_tokens)
        self.model = model
        self.id = "req_123"


class FakeStructuredResponse:
    def __init__(
        self,
        parsed: BaseModel,
        prompt_tokens: int = 5,
        completion_tokens: int = 7,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.choices = [FakeStructuredChoice(parsed)]
        self.usage = FakeUsage(prompt_tokens, completion_tokens)
        self.model = model
        self.id = "req_structured_123"


def build_config(base_url: str = "https://api.openai.com/v1") -> LLMConfig:
    return LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key="test-key",
        base_url=base_url,
    )


def test_complete_returns_text_and_usage() -> None:
    client = LLMClient(
        config=build_config(),
        client_factory=lambda _: FakeClient([FakeResponse("hello world")]),
        sleep=lambda _: None,
    )

    output = client.complete(
        messages=[{"role": "user", "content": "Say hello"}],
    )

    assert isinstance(output, LLMOutput)
    assert output.text == "hello world"
    assert output.provider == "openai"
    assert output.model == "gpt-4o-mini"
    assert output.usage.prompt_tokens == 5
    assert output.usage.completion_tokens == 7


def test_complete_retries_rate_limits_then_succeeds() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(429, request=request)
    rate_limit_error = RateLimitError("rate limited", response=response, body=None)
    fake_client = FakeClient([rate_limit_error, FakeResponse("ok")])
    sleep_calls: list[float] = []

    client = LLMClient(
        config=build_config(),
        client_factory=lambda _: fake_client,
        sleep=sleep_calls.append,
    )

    output = client.complete(messages=[{"role": "user", "content": "test"}])

    assert output.text == "ok"
    assert fake_client.chat.completions.calls == 2
    assert sleep_calls == [1.0]


def test_complete_fails_fast_on_auth_error() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(401, request=request)
    auth_error = AuthenticationError("bad key", response=response, body=None)
    fake_client = FakeClient([auth_error])

    client = LLMClient(
        config=build_config(),
        client_factory=lambda _: fake_client,
        sleep=lambda _: None,
    )

    with pytest.raises(LLMClientError, match="auth_error") as exc_info:
        client.complete(messages=[{"role": "user", "content": "test"}])

    assert exc_info.value.code == "auth_error"
    assert fake_client.chat.completions.calls == 1


def test_complete_retries_timeout_then_succeeds() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    timeout_error = APITimeoutError(request=request)
    fake_client = FakeClient([timeout_error, FakeResponse("ok")])
    sleep_calls: list[float] = []

    client = LLMClient(
        config=build_config(),
        client_factory=lambda _: fake_client,
        sleep=sleep_calls.append,
    )

    output = client.complete(messages=[{"role": "user", "content": "test"}])

    assert output.text == "ok"
    assert fake_client.chat.completions.calls == 2
    assert sleep_calls == [1.0]


def test_complete_uses_exponential_backoff_across_multiple_retries() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(429, request=request)
    rate_limit_error = RateLimitError("rate limited", response=response, body=None)
    fake_client = FakeClient([rate_limit_error, rate_limit_error, FakeResponse("ok")])
    sleep_calls: list[float] = []

    client = LLMClient(
        config=build_config(),
        client_factory=lambda _: fake_client,
        sleep=sleep_calls.append,
    )

    output = client.complete(messages=[{"role": "user", "content": "test"}])

    assert output.text == "ok"
    assert fake_client.chat.completions.calls == 3
    assert sleep_calls == [1.0, 2.0]


def test_complete_does_not_retry_bad_request() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(400, request=request)
    bad_request_error = BadRequestError("bad request", response=response, body=None)
    fake_client = FakeClient([bad_request_error])

    client = LLMClient(
        config=build_config(),
        client_factory=lambda _: fake_client,
        sleep=lambda _: None,
    )

    with pytest.raises(LLMClientError, match="bad_request_error") as exc_info:
        client.complete(messages=[{"role": "user", "content": "test"}])

    assert exc_info.value.code == "bad_request_error"
    assert fake_client.chat.completions.calls == 1


def test_complete_structured_parses_pydantic_model() -> None:
    client = LLMClient(
        config=build_config(),
        client_factory=lambda _: FakeClient([FakeResponse('{"greeting":"hello"}')]),
        sleep=lambda _: None,
    )

    output = client.complete_structured(
        messages=[{"role": "user", "content": "Return greeting JSON"}],
        schema=GreetingResponse,
    )

    assert output.text == '{"greeting":"hello"}'
    assert output.parsed == GreetingResponse(greeting="hello")


def test_complete_structured_prefers_native_parse_when_available() -> None:
    fake_client = FakeStructuredClient(
        structured_responses=[
            FakeStructuredResponse(GreetingResponse(greeting="hello"))
        ]
    )
    client = LLMClient(
        config=build_config(),
        client_factory=lambda _: fake_client,
        sleep=lambda _: None,
    )

    output = client.complete_structured(
        messages=[{"role": "user", "content": "Return greeting JSON"}],
        schema=GreetingResponse,
    )

    assert output.parsed == GreetingResponse(greeting="hello")
    assert fake_client.beta.chat.completions.calls == 1
    assert fake_client.chat.completions.calls == 0


def test_complete_structured_falls_back_when_native_parse_is_rejected() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(400, request=request)
    bad_request_error = BadRequestError(
        "unsupported response_format", response=response, body=None
    )
    fake_client = FakeStructuredClient(
        structured_responses=[bad_request_error],
        fallback_responses=[FakeResponse('{"greeting":"hello"}')],
    )
    client = LLMClient(
        config=build_config(),
        client_factory=lambda _: fake_client,
        sleep=lambda _: None,
    )

    output = client.complete_structured(
        messages=[{"role": "user", "content": "Return greeting JSON"}],
        schema=GreetingResponse,
    )

    assert output.parsed == GreetingResponse(greeting="hello")
    assert fake_client.beta.chat.completions.calls == 1
    assert fake_client.chat.completions.calls == 1


def test_complete_structured_uses_json_schema_strategy_when_config_flag_set() -> None:
    """supports_json_schema=True in config triggers json_schema strategy
    regardless of base_url."""
    fake_client = FakeStructuredClient(
        structured_responses=[
            FakeStructuredResponse(GreetingResponse(greeting="from-native-parse"))
        ],
        fallback_responses=[FakeResponse('{"greeting":"from-json-schema"}')],
    )
    config = build_config(base_url="https://api.together.xyz/v1")
    config = LLMConfig(
        provider=config.provider,
        model=config.model,
        api_key=config.api_key,
        base_url="https://api.together.xyz/v1",
        supports_json_schema=True,
    )
    client = LLMClient(
        config=config,
        client_factory=lambda _: fake_client,
        sleep=lambda _: None,
    )

    output = client.complete_structured(
        messages=[{"role": "user", "content": "Return greeting JSON"}],
        schema=GreetingResponse,
    )

    assert output.parsed == GreetingResponse(greeting="from-json-schema")
    assert fake_client.chat.completions.calls == 1
    assert fake_client.beta.chat.completions.calls == 0


def test_complete_structured_skips_json_schema_when_flag_false() -> None:
    """supports_json_schema=False skips json_schema even for openrouter."""
    fake_client = FakeStructuredClient(
        structured_responses=[
            FakeStructuredResponse(GreetingResponse(greeting="from-native-parse"))
        ],
    )
    config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        supports_json_schema=False,
    )
    client = LLMClient(
        config=config,
        client_factory=lambda _: fake_client,
        sleep=lambda _: None,
    )

    output = client.complete_structured(
        messages=[{"role": "user", "content": "Return greeting JSON"}],
        schema=GreetingResponse,
    )

    # Should skip json_schema and use native parse instead
    assert output.parsed == GreetingResponse(greeting="from-native-parse")
    assert fake_client.beta.chat.completions.calls == 1
    assert fake_client.chat.completions.calls == 0


def test_complete_structured_uses_openrouter_json_schema_strategy_first() -> None:
    fake_client = FakeStructuredClient(
        structured_responses=[
            FakeStructuredResponse(GreetingResponse(greeting="from-native-parse"))
        ],
        fallback_responses=[FakeResponse('{"greeting":"from-json-schema"}')],
    )
    client = LLMClient(
        config=build_config(base_url="https://openrouter.ai/api/v1"),
        client_factory=lambda _: fake_client,
        sleep=lambda _: None,
    )

    output = client.complete_structured(
        messages=[{"role": "user", "content": "Return greeting JSON"}],
        schema=GreetingResponse,
    )

    assert output.parsed == GreetingResponse(greeting="from-json-schema")
    assert fake_client.chat.completions.calls == 1
    assert fake_client.beta.chat.completions.calls == 0
    assert fake_client.chat.completions.last_kwargs is not None
    response_format = fake_client.chat.completions.last_kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["name"] == "GreetingResponse"


def test_complete_structured_falls_back_from_openrouter_json_schema_to_native_parse() -> (
    None
):
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(400, request=request)
    bad_request_error = BadRequestError(
        "unsupported json_schema", response=response, body=None
    )
    fake_client = FakeStructuredClient(
        structured_responses=[
            FakeStructuredResponse(GreetingResponse(greeting="from-native-parse"))
        ],
        fallback_responses=[bad_request_error],
    )
    client = LLMClient(
        config=build_config(base_url="https://openrouter.ai/api/v1"),
        client_factory=lambda _: fake_client,
        sleep=lambda _: None,
    )

    output = client.complete_structured(
        messages=[{"role": "user", "content": "Return greeting JSON"}],
        schema=GreetingResponse,
    )

    assert output.parsed == GreetingResponse(greeting="from-native-parse")
    assert fake_client.chat.completions.calls == 1
    assert fake_client.beta.chat.completions.calls == 1


def test_complete_structured_extracts_json_from_code_fence() -> None:
    client = LLMClient(
        config=build_config(),
        client_factory=lambda _: FakeClient(
            [FakeResponse('```json\n{"greeting":"hello"}\n```')]
        ),
        sleep=lambda _: None,
    )

    output = client.complete_structured(
        messages=[{"role": "user", "content": "Return greeting JSON"}],
        schema=GreetingResponse,
    )

    assert output.parsed == GreetingResponse(greeting="hello")


def test_complete_structured_extracts_json_from_labeled_lines() -> None:
    client = LLMClient(
        config=build_config(),
        client_factory=lambda _: FakeClient(
            [FakeResponse('Scores: {"greeting":"hello"}\nReasoning: ignored')]
        ),
        sleep=lambda _: None,
    )

    output = client.complete_structured(
        messages=[{"role": "user", "content": "Return greeting JSON"}],
        schema=GreetingResponse,
    )

    assert output.parsed == GreetingResponse(greeting="hello")


def test_complete_uses_custom_openai_compatible_base_url() -> None:
    captured_config: list[LLMConfig] = []

    def capture_factory(config: LLMConfig) -> FakeClient:
        captured_config.append(config)
        return FakeClient([FakeResponse("ok")])

    client = LLMClient(
        config=build_config(base_url="https://openrouter.ai/api/v1"),
        client_factory=capture_factory,
        sleep=lambda _: None,
    )

    output = client.complete(messages=[{"role": "user", "content": "test"}])

    assert output.text == "ok"
    assert str(captured_config[0].base_url) == "https://openrouter.ai/api/v1"


def test_complete_batch_returns_outputs_in_input_order() -> None:
    client = LLMClient(
        config=build_config(),
        client_factory=lambda _: FakeClient([]),
        sleep=lambda _: None,
    )

    messages_seen: list[str] = []

    def fake_complete(messages: list[dict[str, str]]) -> LLMOutput:
        content = messages[0]["content"]
        messages_seen.append(content)
        return LLMOutput(
            text=content.upper(),
            parsed=None,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            finish_reason="stop",
            model="gpt-4o-mini",
            provider="openai",
            request_id="req_batch",
            latency_ms=1,
            error=None,
        )

    client.complete = fake_complete  # type: ignore[method-assign]

    outputs = client.complete_batch(
        batch=[
            [{"role": "user", "content": "alpha"}],
            [{"role": "user", "content": "beta"}],
            [{"role": "user", "content": "gamma"}],
        ],
        max_workers=3,
    )

    assert [output.text for output in outputs] == ["ALPHA", "BETA", "GAMMA"]
    assert messages_seen == ["alpha", "beta", "gamma"]


def test_complete_batch_caps_default_worker_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LLMClient(
        config=build_config(),
        client_factory=lambda _: FakeClient([]),
        sleep=lambda _: None,
    )
    captured_workers: list[int] = []

    class FakeFuture:
        def __init__(self, value: LLMOutput) -> None:
            self._value = value

        def result(self) -> LLMOutput:
            return self._value

    class FakeExecutor:
        def __init__(self, max_workers: int) -> None:
            captured_workers.append(max_workers)

        def __enter__(self) -> FakeExecutor:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def submit(self, fn, *args):
            return FakeFuture(fn(*args))

    monkeypatch.setattr("arka.llm.client.ThreadPoolExecutor", FakeExecutor)

    client.complete = lambda messages: LLMOutput(  # type: ignore[method-assign]
        text=messages[0]["content"],
        parsed=None,
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        finish_reason="stop",
        model="gpt-4o-mini",
        provider="openai",
        request_id="req_batch",
        latency_ms=1,
        error=None,
    )

    batch = [[{"role": "user", "content": f"item-{index}"}] for index in range(20)]
    outputs = client.complete_batch(batch=batch)

    assert len(outputs) == 20
    assert captured_workers == [8]


# --- _extract_json_text edge-case tests ---


def _make_client_for_extraction() -> LLMClient:
    config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
    )
    return LLMClient(config=config)


def test_extract_json_text_bare_json() -> None:
    client = _make_client_for_extraction()
    result = client._extract_json_text('{"greeting": "hello"}')
    assert result == '{"greeting": "hello"}'


def test_extract_json_text_with_surrounding_text() -> None:
    client = _make_client_for_extraction()
    text = 'Here is the result: {"greeting": "hello"} Hope that helps!'
    result = client._extract_json_text(text)
    assert result == '{"greeting": "hello"}'


def test_extract_json_text_code_fence() -> None:
    client = _make_client_for_extraction()
    text = '```json\n{"greeting": "hello"}\n```'
    result = client._extract_json_text(text)
    assert result == '{"greeting": "hello"}'


def test_extract_json_text_code_fence_no_lang() -> None:
    client = _make_client_for_extraction()
    text = '```\n{"greeting": "hello"}\n```'
    result = client._extract_json_text(text)
    assert result == '{"greeting": "hello"}'


def test_extract_json_text_nested_braces() -> None:
    """Nested braces in a single JSON object should be extracted correctly."""
    client = _make_client_for_extraction()
    text = '{"outer": {"inner": "value"}}'
    result = client._extract_json_text(text)
    assert result == '{"outer": {"inner": "value"}}'


def test_extract_json_text_greedy_limitation_two_objects() -> None:
    """Known limitation: greedy regex grabs from first { to last },
    which produces invalid JSON when two objects are present."""
    client = _make_client_for_extraction()
    text = '{"a": 1} some text {"b": 2}'
    result = client._extract_json_text(text)
    # The greedy match spans both objects — this is the documented limitation.
    assert result == '{"a": 1} some text {"b": 2}'


def test_extract_json_text_no_braces_returns_stripped() -> None:
    client = _make_client_for_extraction()
    result = client._extract_json_text("  just plain text  ")
    assert result == "just plain text"


def test_extract_json_text_unclosed_code_fence_raises() -> None:
    client = _make_client_for_extraction()
    with pytest.raises(ValueError, match="Could not extract JSON from code fence"):
        client._extract_json_text('```json\n{"a": 1}')


def test_extract_json_text_whitespace_padding() -> None:
    client = _make_client_for_extraction()
    text = '\n  \n  {"greeting": "hello"}  \n  '
    result = client._extract_json_text(text)
    assert result == '{"greeting": "hello"}'


# --- cost_usd extraction tests ---


class FakeUsageWithCost:
    def __init__(
        self, prompt_tokens: int, completion_tokens: int, total_cost: float
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        self.total_cost = total_cost


class FakeResponseWithCost:
    def __init__(self, content: str, total_cost: float) -> None:
        self.choices = [FakeChoice(content)]
        self.usage = FakeUsageWithCost(5, 7, total_cost)
        self.model = "gpt-4o-mini"
        self.id = "req_cost_123"


def test_usage_from_response_extracts_cost_when_present() -> None:
    client = _make_client_for_extraction()
    response = FakeResponseWithCost("hello", total_cost=0.00042)
    usage = client._usage_from_response(response)
    assert usage.cost_usd == 0.00042


def test_usage_from_response_leaves_cost_none_when_absent() -> None:
    client = _make_client_for_extraction()
    response = FakeResponse("hello")
    usage = client._usage_from_response(response)
    assert usage.cost_usd is None
