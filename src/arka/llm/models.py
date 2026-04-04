from __future__ import annotations

from pydantic import BaseModel


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None


class LLMError(BaseModel):
    type: str
    message: str
    retryable: bool


class LLMOutput(BaseModel):
    text: str | None = None
    parsed: BaseModel | None = None
    usage: TokenUsage
    finish_reason: str | None = None
    model: str
    provider: str
    request_id: str | None = None
    latency_ms: int
    error: LLMError | None = None
