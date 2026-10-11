"""Conservative defaults for opt-in generation-artifact filtering."""

DEFAULT_ARTIFACT_GROUPS = {
    "openers": [
        r"\A[ \t]*(?:Certainly!|Sure!|Of course!)[ \t]*(?:\r?\n[ \t]*)?",
    ],
    "closers": [
        r"(?:[ \t]*\r?\n)?[ \t]*(?:I hope this helps[!.]?|Would you like me to[^?\n]*\?)[ \t]*\Z",
    ],
    "self_reference": [
        r"\bAs an AI language model\b",
        r"\bas of my last update\b",
        r"\bmy knowledge cutoff\b",
    ],
    "placeholders": [
        r"\[(?:your (?:name|email|address|company)|link here|insert (?:text|name|link) here|placeholder)\]",
    ],
    "tool_fingerprints": [
        r"\bciteturn\d+(?:search|view|fetch)\d+\b",
        r":?contentReference\[oaicite:\d+\]",
        r"cite[^]*",
    ],
}
