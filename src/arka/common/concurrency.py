from __future__ import annotations

DEFAULT_MAX_WORKERS = 8


def bounded_worker_count(
    item_count: int,
    requested_max_workers: int | None,
    *,
    default_max_workers: int = DEFAULT_MAX_WORKERS,
) -> int:
    if item_count <= 0:
        return 1
    if requested_max_workers is not None and requested_max_workers > 0:
        return min(requested_max_workers, item_count)
    return min(default_max_workers, item_count)
