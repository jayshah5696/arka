"""Find repeated byte blocks across distinct pages of one PDF."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class BoilerplateSpan:
    page: int
    start: int
    end: int
    block_hash: str
    occurrences: int


def strip_boilerplate(
    pages: list[tuple[int, str]], min_block_bytes: int, min_doc_frequency: int
) -> tuple[list[tuple[int, str]], list[BoilerplateSpan]]:
    """Cut repeated aligned blocks, extending through their exact shared edges.

    Offsets refer to UTF-8 bytes in the original extracted, trimmed page text.
    Extension removes an entire repeated header even when its length is not a
    multiple of the block size. Alignment can still miss unanchored repeats.
    """
    raw = {page: text.encode("utf-8") for page, text in pages}
    occurrences: dict[bytes, list[tuple[int, int, int]]] = defaultdict(list)
    for page, data in raw.items():
        for start in range(0, len(data) - min_block_bytes + 1, min_block_bytes):
            end = start + min_block_bytes
            occurrences[data[start:end]].append((page, start, end))

    cuts: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for locations in occurrences.values():
        if len({page for page, _, _ in locations}) < min_doc_frequency:
            continue
        left = right = 0
        while all(start > left for _, start, _ in locations):
            if len({raw[page][start - left - 1] for page, start, _ in locations}) != 1:
                break
            left += 1
        while all(end + right < len(raw[page]) for page, _, end in locations):
            if len({raw[page][end + right] for page, _, end in locations}) != 1:
                break
            right += 1
        for page, start, end in locations:
            start, end = start - left, end + right
            # Never leave half of a UTF-8 character behind.
            while start < end and raw[page][start] & 0xC0 == 0x80:
                start += 1
            while (
                end > start and end < len(raw[page]) and raw[page][end] & 0xC0 == 0x80
            ):
                end -= 1
            if start < end:
                cuts[page].append((start, end))

    merged: dict[int, list[tuple[int, int]]] = {}
    for page, ranges in cuts.items():
        spans: list[tuple[int, int]] = []
        for start, end in sorted(ranges):
            if spans and start <= spans[-1][1]:
                spans[-1] = (spans[-1][0], max(end, spans[-1][1]))
            else:
                spans.append((start, end))
        merged[page] = spans

    removed_pages: dict[bytes, set[int]] = defaultdict(set)
    for page, ranges in merged.items():
        for start, end in ranges:
            removed_pages[raw[page][start:end]].add(page)
    reports: list[BoilerplateSpan] = []
    cleaned: list[tuple[int, str]] = []
    for page, _ in pages:
        data = raw[page]
        ranges = merged.get(page, [])
        for start, end in ranges:
            removed = data[start:end]
            reports.append(
                BoilerplateSpan(
                    page,
                    start,
                    end,
                    hashlib.blake2b(removed).hexdigest(),
                    len(removed_pages[removed]),
                )
            )
        for start, end in reversed(ranges):
            data = data[:start] + data[end:]
        cleaned.append((page, data.decode("utf-8").strip()))
    return cleaned, reports
