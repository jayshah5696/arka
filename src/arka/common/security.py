from __future__ import annotations

import re

_TAG_PATTERN = re.compile(r"(?i)</?text>")


def sanitize_for_prompt(text: str) -> str:
    """
    SECURITY: Sanitize untrusted input by stripping XML-like <text> tags.
    Uses regex substitution in a loop to prevent evasion via nested tags (e.g. <te<text>xt>).
    """
    while True:
        new_text = _TAG_PATTERN.sub("", text)
        if new_text == text:
            break
        text = new_text
    return text
