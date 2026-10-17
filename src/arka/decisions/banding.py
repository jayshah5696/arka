from typing import Literal


def band(
    probability: float, *, accept_at: float, reject_at: float
) -> Literal["accept", "reject", "uncertain"]:
    """Accept at the upper boundary; reject strictly below the lower boundary."""
    if not 0 <= reject_at < accept_at <= 1 or not 0 <= probability <= 1:
        raise ValueError(
            "require 0 <= reject_at < accept_at <= 1 and a valid probability"
        )
    if probability >= accept_at:
        return "accept"
    if probability < reject_at:
        return "reject"
    return "uncertain"
