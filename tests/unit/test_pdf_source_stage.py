from __future__ import annotations

from pathlib import Path

import pytest

from arka.config.loader import ConfigLoader
from arka.pipeline.models import StageContext
from arka.pipeline.source_stages import PDFSourceStage
from arka.records.models import GroundedChunkRecord

_MINIMAL_TEXT_PDF = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
72 72 Td
(Hello PDF world) Tj
ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f 
0000000010 00000 n 
0000000063 00000 n 
0000000122 00000 n 
0000000248 00000 n 
0000000342 00000 n 
trailer
<< /Root 1 0 R /Size 6 >>
startxref
412
%%EOF
"""


def _ctx(tmp_path: Path, relative_pdf_path: str) -> StageContext:
    config = ConfigLoader().load_dict(
        {
            "version": "1",
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "test-key",
                "base_url": "https://api.openai.com/v1",
            },
            "executor": {"mode": "threadpool", "max_workers": 1},
            "data_source": {
                "type": "pdf",
                "path": relative_pdf_path,
                "chunk_strategy": "fixed",
                "chunk_size_chars": 20,
                "chunk_overlap_chars": 5,
            },
            "generator": {
                "type": "prompt_based",
                "target_count": 1,
                "generation_multiplier": 1,
            },
            
            "filters": {"target_count": 1},
            "embeddings": {"provider": "huggingface", "model": "all-MiniLM-L6-v2"},
            "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
        }
    )
    work_dir = tmp_path / "runs" / "pdf-run" / "stages" / "01_source"
    work_dir.mkdir(parents=True, exist_ok=True)
    return StageContext(
        run_id="pdf-run",
        stage_name="01_source",
        work_dir=work_dir,
        config=config,
        executor_mode=config.executor.mode,
        max_workers=config.executor.max_workers,
    )


def test_pdf_source_stage_emits_grounded_chunks_with_provenance(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(_MINIMAL_TEXT_PDF)
    ctx = _ctx(tmp_path, "./sample.pdf")

    records = PDFSourceStage(project_root=tmp_path).run([], ctx)

    assert records
    assert all(isinstance(record, GroundedChunkRecord) for record in records)
    assert records[0].payload.text.startswith("Hello PDF")
    assert records[0].payload.doc_id == "sample"
    assert records[0].source.doc_id == "sample"
    assert records[0].source.page_start == 1
    assert records[0].source.page_end == 1
    assert records[0].source.source_hash is not None


def test_pdf_source_stage_raises_for_empty_pdf_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
    ctx = _ctx(tmp_path, "./empty.pdf")

    with pytest.raises(
        ValueError,
        match="PDF extraction produced no text|EOF marker not found|Cannot find Root object",
    ):
        PDFSourceStage(project_root=tmp_path).run([], ctx)
