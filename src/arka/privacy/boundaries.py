from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from arka.llm.client import LLMClientError, StructuredResponseError
from arka.llm.models import LLMOutput
from arka.privacy.models import PIISummary
from arka.privacy.policy import PIIPolicy, PrivacyError


class ProtectedLLMClient:
    """Apply the same local policy before requests and before returning output.

    Keep the underlying client's configuration and scoring capabilities. Its
    exceptions can contain provider text, so never carry them across this seam.
    """

    def __init__(
        self, client: Any, policy: PIIPolicy, *, retain_rejected_output: bool = False
    ) -> None:
        self.client = client
        self.policy = policy
        # Generators can persist a safe rejection plus usage. Judges retain the
        # client's exception contract so their drop/skip handling still applies.
        self.retain_rejected_output = retain_rejected_output

    def __getattr__(self, name):
        return getattr(self.client, name)

    def _output(
        self, output: LLMOutput, *, structured: bool, schema: type[BaseModel] | None
    ) -> LLMOutput:
        safe_error = (
            output.error.model_copy(
                update={
                    "message": "Model returned an error while PII protection was enabled"
                }
            )
            if output.error
            else None
        )
        if structured:
            try:
                # Validate the schema before retaining any raw JSON. Unexpected
                # field names can contain sensitive text too.
                original = (
                    output.parsed.model_dump(mode="json")
                    if output.parsed is not None
                    else json.loads(output.text or "")
                )
                original = schema.model_validate(original)
            except (ValueError, TypeError):
                if not self.retain_rejected_output:
                    raise LLMClientError(
                        "parse_error", "Model output did not match the requested schema"
                    ) from None
                return output.model_copy(
                    update={
                        "text": "[PII_UNPARSEABLE_RESPONSE]",
                        "parsed": None,
                        "error": safe_error,
                        "privacy": PIISummary(drop_reason="pii_unparseable_response"),
                    }
                )
            result = self.policy.protect(original.model_dump(mode="json"))
            parsed = schema.model_validate(result.value)
            # A single canonical representation prevents a second raw copy or
            # ignored provider fields from entering generator artifacts.
            text = parsed.model_dump_json()
        else:
            result = self.policy.protect(output.text or "")
            text, parsed = result.value, None
        return output.model_copy(
            update={
                "text": text,
                "parsed": parsed,
                "error": safe_error,
                "privacy": PIISummary(
                    matched_bytes=result.matched_bytes,
                    total_bytes=result.total_bytes,
                    categories=result.categories,
                    drop_reason="pii_provider_error"
                    if safe_error
                    else result.drop_reason,
                ),
            }
        )

    def _messages(self, messages):
        return [
            {**message, "content": self.policy.protect(message["content"]).value}
            for message in messages
        ]

    def _call(self, method, *, messages, structured=False, **kwargs):
        safe_messages = self._messages(messages)
        try:
            output = method(messages=safe_messages, **kwargs)
        except StructuredResponseError as exc:
            if not self.retain_rejected_output:
                raise LLMClientError(
                    exc.code, "Model output did not match the requested schema"
                ) from None
            output = exc.output
        except LLMClientError as exc:
            raise LLMClientError(
                exc.code, "Model request failed while PII protection was enabled"
            ) from None
        except Exception:
            raise PrivacyError(
                "Model request failed while PII protection was enabled"
            ) from None
        try:
            return self._output(
                output, structured=structured, schema=kwargs.get("schema")
            )
        except (PrivacyError, LLMClientError):
            raise
        except Exception:
            raise PrivacyError("Model output could not pass PII protection") from None

    def complete(self, messages, **kwargs):
        return self._call(self.client.complete, messages=messages, **kwargs)

    def complete_structured(self, messages, schema: type[BaseModel], **kwargs):
        return self._call(
            self.client.complete_structured,
            messages=messages,
            schema=schema,
            structured=True,
            **kwargs,
        )

    def score_response(self, *, messages, target_text):
        safe_messages = self._messages(messages)
        safe_target = self.policy.protect(target_text).value
        try:
            return self.client.score_response(
                messages=safe_messages, target_text=safe_target
            )
        except LLMClientError as exc:
            raise LLMClientError(
                exc.code, "Model scoring failed while PII protection was enabled"
            ) from None
        except Exception:
            raise PrivacyError(
                "Model scoring failed while PII protection was enabled"
            ) from None
