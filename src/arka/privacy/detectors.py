from __future__ import annotations

import ipaddress
import re

from arka.privacy.policy import Span

REGEX_ENTITIES = frozenset(
    {"email", "phone_number", "payment_card", "iban", "ip_address"}
)

_PATTERNS = {
    "email": re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.UNICODE),
    # Only explicit international numbers. Local numbering plans are not inferred.
    "phone_number": re.compile(r"(?<!\w)\+\d(?:[\d ()-]{5,}\d)"),
    "payment_card": re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}\b"),
    "ip_address": re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
        r"|(?<![\w:])(?:[a-fA-F0-9]{0,4}:){2,}[a-fA-F0-9]{0,4}(?![\w:])"
    ),
}


class RegexDetector:
    """Detect a limited set of identifier formats without a model runtime."""

    def __init__(self, entities: set[str]) -> None:
        if not entities or entities - REGEX_ENTITIES:
            raise ValueError("Regex detector requested unsupported entities")
        self.entities = sorted(entities)

    def detect(self, text: str) -> list[Span]:
        return [
            Span(match.start(), match.end(), entity)
            for entity in self.entities
            for match in _PATTERNS[entity].finditer(text)
            if self._valid(entity, match.group())
        ]

    def _valid(self, entity: str, value: str) -> bool:
        if entity == "payment_card":
            digits = [int(c) for c in value if c.isascii() and c.isdigit()]
            if len(digits) < 13 or len(digits) > 19 or len(set(digits)) == 1:
                return False
            total = 0
            for index, digit in enumerate(reversed(digits)):
                if index % 2:
                    digit *= 2
                    digit = digit - 9 if digit > 9 else digit
                total += digit
            return total % 10 == 0
        if entity == "iban":
            compact = value.replace(" ", "")
            if not 15 <= len(compact) <= 34:
                return False
            rearranged = compact[4:] + compact[:4]
            number = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged)
            return int(number) % 97 == 1
        if entity == "phone_number":
            return 7 <= sum(c.isdigit() for c in value) <= 15
        if entity == "ip_address":
            try:
                ipaddress.ip_address(value)
            except ValueError:
                return False
        return True
