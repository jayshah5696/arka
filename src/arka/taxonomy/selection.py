"""Reserve records for explicit minimums, without inferring taxonomy labels."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CoverageSelection:
    indices: tuple[int, ...]
    status: Literal["satisfied", "infeasible", "search_limit"]


def reserve_coverage(
    memberships: list[frozenset[int]],
    minimums: tuple[int, ...],
    target: int,
    search_limit: int,
) -> CoverageSelection:
    """Reserve from score-ordered records, then search if greedy cannot fit.

    Records with identical quota memberships are interchangeable for feasibility.
    Grouping them avoids branching on every record in a large generated pool.
    Only the highest-scoring records of each group need to be considered.
    """
    by_membership: dict[frozenset[int], list[int]] = {}
    for i, membership in enumerate(memberships):
        if membership:
            by_membership.setdefault(membership, []).append(i)
    patterns = list(by_membership)
    groups = list(by_membership.values())
    capacities = [
        min(len(group), max(minimums[q] for q in pattern))
        for pattern, group in zip(patterns, groups, strict=True)
    ]

    def indices(used: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            sorted(
                i
                for group, count in zip(groups, used, strict=True)
                for i in group[:count]
            )
        )

    def choices(needs: tuple[int, ...], used: tuple[int, ...]) -> list[int]:
        return sorted(
            (
                g
                for g, pattern in enumerate(patterns)
                if used[g] < capacities[g] and any(needs[q] for q in pattern)
            ),
            key=lambda g: (-sum(needs[q] > 0 for q in patterns[g]), groups[g][used[g]]),
        )

    def take(
        needs: tuple[int, ...], used: tuple[int, ...], g: int
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (
            tuple(max(0, n - (q in patterns[g])) for q, n in enumerate(needs)),
            tuple(count + (i == g) for i, count in enumerate(used)),
        )

    needs, used = minimums, (0,) * len(groups)
    while any(needs) and sum(used) < target:
        candidates = choices(needs, used)
        if not candidates:
            break
        needs, used = take(needs, used, candidates[0])
    greedy = indices(used)
    if not any(needs):
        return CoverageSelection(greedy, "satisfied")

    # Explicit stack keeps large quota counts independent of Python's recursion limit.
    stack = [(minimums, (0,) * len(groups))]
    visited: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    while stack:
        needs, used = stack.pop()
        state = (needs, used)
        if state in visited:
            continue
        if len(visited) >= search_limit:
            return CoverageSelection(greedy, "search_limit")
        visited.add(state)
        if not any(needs):
            return CoverageSelection(indices(used), "satisfied")
        slots = target - sum(used)
        available = [
            sum(
                capacities[g] - used[g]
                for g, pattern in enumerate(patterns)
                if q in pattern
            )
            for q in range(len(minimums))
        ]
        if slots < max(needs) or any(n > available[q] for q, n in enumerate(needs)):
            continue
        quota = min(
            (q for q, n in enumerate(needs) if n),
            key=lambda q: (available[q] - needs[q], q),
        )
        for g in reversed(choices(needs, used)):
            if quota in patterns[g]:
                stack.append(take(needs, used, g))
    return CoverageSelection(greedy, "infeasible")
