import json

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject

from arka.cli import main
from arka.pipeline.output import OutputWriter


def test_pdf_cleanup_exports_clean_chunks_and_resumes(tmp_path):
    writer = PdfWriter()
    for body in ("A unique body.", "B different body."):
        page = writer.add_blank_page(width=300, height=300)
        stream = DecodedStreamObject()
        stream.set_data(f"BT ({'Header' * 100}{body}) Tj ET".encode())
        page.replace_contents(stream)
    writer.write(tmp_path / "sample.pdf")
    base = {
        "version": "1",
        "llm": {
            "provider": "openai",
            "model": "local",
            "api_key": "fixture",
            "base_url": "http://127.0.0.1:1/v1",
        },
        "executor": {"mode": "threadpool", "max_workers": 1},
        "pipeline": [
            {
                "type": "pdf_source",
                "path": "sample.pdf",
                "chunk_size_chars": 300,
                "chunk_overlap_chars": 0,
            }
        ],
        "output": {"format": "jsonl", "path": "dataset.jsonl"},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(base))
    main(["--config", str(path), "--run-id", "original"])
    original = (tmp_path / "dataset.jsonl").read_bytes()
    base["pipeline"][0].update(boilerplate_strip=True, min_remaining_bytes=1)
    path.write_text(json.dumps(base))
    main(["--config", str(path), "--run-id", "cleaned"])
    clean = (tmp_path / "dataset.jsonl").read_bytes()
    rows = [json.loads(line) for line in clean.splitlines()]
    assert [row["text"] for row in rows] == ["A unique body.\n\nB different body."]
    assert len(clean.splitlines()) < len(original.splitlines())
    directory = tmp_path / "runs/cleaned/stages/01_source"
    records = OutputWriter().read_parquet(directory / "data.parquet")
    assert (records[0].source.page_start, records[0].source.page_end) == (1, 2)
    assert (
        json.loads((directory / "stats.json").read_text())["boilerplate_bytes_removed"]
        == 1200
    )
    main(["--config", str(path), "--run-id", "cleaned", "--resume"])
    assert (tmp_path / "dataset.jsonl").read_bytes() == clean
