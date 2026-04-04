from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from arka.config.models import LLMConfig


@dataclass(frozen=True)
class OpenAIClientFactory:
    config: LLMConfig

    def build(self) -> OpenAI:
        default_headers: dict[str, str] = {}
        if self.config.openai_compatible is not None:
            if self.config.openai_compatible.referer is not None:
                default_headers["HTTP-Referer"] = str(
                    self.config.openai_compatible.referer
                )
            if self.config.openai_compatible.title is not None:
                default_headers["X-Title"] = self.config.openai_compatible.title
        return OpenAI(
            api_key=self.config.api_key,
            base_url=str(self.config.base_url),
            timeout=self.config.timeout_seconds,
            max_retries=0,
            default_headers=default_headers or None,
        )
