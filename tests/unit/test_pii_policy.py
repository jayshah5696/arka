from __future__ import annotations

import json

import pytest

from arka.privacy.policy import PIIPolicy, PrivacyError, Span


class FixedDetector:
    def __init__(self, spans: list[Span]) -> None:
        self.spans = spans

    def detect(self, text: str) -> list[Span]:
        return self.spans


def test_unicode_overlaps_count_original_utf8_bytes_once() -> None:
    text = "é Alice!"
    policy = PIIPolicy(
        FixedDetector([Span(0, 7, "address"), Span(2, 7, "person")]),
        max_redacted_ratio=0.8,
    )

    result = policy.protect(text)

    assert result.value == "[address]!"
    assert result.matched_bytes == 8
    assert result.total_bytes == 9
    assert result.drop_reason == "pii_ratio"
    assert result.categories == {"address": 1, "person": 1}


def test_repeated_values_use_returned_offsets() -> None:
    result = PIIPolicy(FixedDetector([Span(8, 13, "person")])).protect("Alice / Alice")

    assert result.value == "Alice / [person]"


def test_nested_json_preserves_keys_types_and_escaped_values() -> None:
    class EmailDetector:
        def detect(self, text: str) -> list[Span]:
            needle = "a@example.com"
            return (
                [Span(text.index(needle), text.index(needle) + len(needle), "email")]
                if needle in text
                else []
            )

    original = {
        "email": "a@example.com",
        "turns": [{"content": "a@example.com"}],
        "n": 3,
    }
    result = PIIPolicy(EmailDetector()).protect(original)

    assert result.value == {
        "email": "[email]",
        "turns": [{"content": "[email]"}],
        "n": 3,
    }
    assert original["email"] == "a@example.com"
    assert result.matched_bytes == 26
    assert result.total_bytes == 26


def test_blocked_category_takes_precedence_over_ratio() -> None:
    result = PIIPolicy(
        FixedDetector([Span(0, 1, "secret")]),
        blocked_entities={"secret"},
        max_redacted_ratio=0,
    ).protect("x")

    assert result.value == "[secret]"
    assert result.drop_reason == "pii_blocked_entity"


def test_ratio_is_strict_and_has_no_default_threshold() -> None:
    detector = FixedDetector([Span(0, 1, "person")])

    assert PIIPolicy(detector).protect("x").drop_reason is None
    assert PIIPolicy(detector, max_redacted_ratio=0.5).protect("xy").drop_reason is None
    assert PIIPolicy(FixedDetector([])).protect("").total_bytes == 0


@pytest.mark.parametrize(
    "span",
    [
        Span(-1, 1, "person"),
        Span(0, 99, "person"),
        Span(1, 1, "person"),
        Span(0.1, 2, "person"),
    ],
)
def test_invalid_detector_offsets_fail_without_payload(span: Span) -> None:
    with pytest.raises(PrivacyError) as caught:
        PIIPolicy(FixedDetector([span])).protect("private")

    assert "private" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_detector_exception_does_not_disclose_input() -> None:
    class BrokenDetector:
        def detect(self, text: str) -> list[Span]:
            raise RuntimeError(f"cannot parse {text}")

    with pytest.raises(PrivacyError) as caught:
        PIIPolicy(BrokenDetector()).protect("private")

    assert "private" not in str(caught.value)
    assert caught.value.__suppress_context__


def test_generated_json_redacts_decoded_values_and_keeps_schema() -> None:
    class EmailDetector:
        def detect(self, text: str) -> list[Span]:
            return [Span(0, len(text), "email")] if "@" in text else []

    policy = PIIPolicy(EmailDetector())
    protected = policy.protect_response('{"response":"a\\u0040example.com","n":1}')

    assert json.loads(protected.value) == {"response": "[email]", "n": 1}


def test_malformed_generated_json_is_redacted_before_diagnostics() -> None:
    policy = PIIPolicy(FixedDetector([Span(9, 14, "person")]))

    assert policy.protect_response('{"text":"Alice').value == '{"text":"[person]'


def test_contextual_detector_cannot_redact_inside_generated_placeholders() -> None:
    policy = PIIPolicy(FixedDetector([Span(1, 7, "person")]))
    assert policy.protect("[person]").value == "[person]"
