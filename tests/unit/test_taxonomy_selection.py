from itertools import combinations

from arka.taxonomy.selection import reserve_coverage


def test_coverage_feasibility_matches_exhaustive_small_pools() -> None:
    # Compare the selector to all subsets, including overlapping requirements.
    patterns = [
        frozenset(i for i in range(3) if mask & (1 << i)) for mask in range(1, 8)
    ]
    for pool in combinations(patterns, 4):
        for target in (1, 2, 3):
            for minimums in ((1, 1, 1), (2, 1, 1)):
                feasible = any(
                    all(
                        sum(q in pool[i] for i in chosen) >= minimums[q]
                        for q in range(3)
                    )
                    for chosen in combinations(range(len(pool)), target)
                )
                result = reserve_coverage(list(pool), minimums, target, 10000)
                assert (result.status == "satisfied") == feasible
                assert len(result.indices) <= target
                if feasible:
                    assert all(
                        sum(q in pool[i] for i in result.indices) >= minimums[q]
                        for q in range(3)
                    )
