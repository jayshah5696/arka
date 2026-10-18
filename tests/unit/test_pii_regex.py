from __future__ import annotations

import pytest

from arka.privacy.detectors import RegexDetector
from arka.privacy.policy import PIIPolicy


@pytest.mark.parametrize(
    ("entity", "text", "expected"),
    [
        (
            "email",
            "Contact alice@example.com twice alice@example.com.",
            "Contact [email] twice [email].",
        ),
        ("phone_number", "Call +1 (415) 555-2671.", "Call [phone_number]."),
        ("payment_card", "Card 4111 1111 1111 1111.", "Card [payment_card]."),
        ("payment_card", "Card 4111 1111 1111 1112.", "Card 4111 1111 1111 1112."),
        ("iban", "Account GB82 WEST 1234 5698 7654 32.", "Account [iban]."),
        (
            "iban",
            "Account GB82 WEST 1234 5698 7654 33.",
            "Account GB82 WEST 1234 5698 7654 33.",
        ),
        (
            "ip_address",
            "Hosts 192.0.2.7 and 2001:db8::1.",
            "Hosts [ip_address] and [ip_address].",
        ),
        ("ip_address", "Version 999.0.0.1.", "Version 999.0.0.1."),
    ],
)
def test_identifier_formats_and_validators(
    entity: str, text: str, expected: str
) -> None:
    assert PIIPolicy(RegexDetector({entity})).protect(text).value == expected


def test_unselected_entities_are_preserved_and_long_suffix_is_scanned() -> None:
    text = "x " * 10000 + "alice@example.com +1 (415) 555-2671"

    result = PIIPolicy(RegexDetector({"email"})).protect(text)

    assert result.value == "x " * 10000 + "[email] +1 (415) 555-2671"
    assert result.categories == {"email": 1}


def test_regex_does_not_claim_contextual_person_detection() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        RegexDetector({"person"})


def test_regex_redaction_is_idempotent() -> None:
    policy = PIIPolicy(RegexDetector({"email", "payment_card"}))
    first = policy.protect("alice@example.com / 4111111111111111")

    assert policy.protect(first.value).value == first.value
