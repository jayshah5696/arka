import hashlib

import pytest

from arka.common.pdf_boilerplate import strip_boilerplate
from arka.config.models import PDFSourceConfig


def test_repeated_header_removed_and_spans_match_original_bytes():
    header = "Header" * 100  # 600 bytes, deliberately not a multiple of 500.
    pages = [(i + 1, header + chr(65 + i) * 151) for i in range(10)]
    cleaned, spans = strip_boilerplate(pages, 500, 2)
    assert cleaned == [(i + 1, chr(65 + i) * 151) for i in range(10)]
    assert len(spans) == 10
    for span in spans:
        raw = pages[span.page - 1][1].encode()
        assert raw[span.start : span.end] == header.encode()
        assert span.block_hash == hashlib.blake2b(header.encode()).hexdigest()
        assert span.occurrences == 10


def test_frequency_counts_distinct_pages_and_preserves_unique_blocks():
    repeated = "x" * 500
    pages = [(1, repeated * 3), (2, "y" * 600), (3, "")]
    assert strip_boilerplate(pages, 500, 2) == (pages, [])
    pages = [(1, repeated + "a"), (2, repeated + "b")]
    assert strip_boilerplate(pages, 500, 3) == (pages, [])


def test_utf8_removals_preserve_valid_text_and_original_page_numbers():
    header = "é" * 301
    pages = [(1, header + "A unique body"), (2, ""), (3, header + "B unique body")]
    cleaned, spans = strip_boilerplate(pages, 500, 2)
    assert cleaned == [(1, "A unique body"), (2, ""), (3, "B unique body")]
    assert [(s.page, s.start, s.end) for s in spans] == [(1, 0, 602), (3, 0, 602)]


@pytest.mark.parametrize(
    "options",
    [
        {"min_block_bytes": 0},
        {"min_doc_frequency": 1},
        {"min_remaining_bytes": -1},
    ],
)
def test_boilerplate_options_are_validated(options):
    with pytest.raises(ValueError):
        PDFSourceConfig(path="sample.pdf", **options)
