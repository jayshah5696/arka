"""Exercise the CLI and actual SDK against a loopback OpenAI fixture."""

import argparse
import atexit
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from arka.pipeline.output import OutputWriter

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
ROOT = args.output.resolve()
ROOT.mkdir(parents=True, exist_ok=False)
CLI = str(Path(sys.executable).with_name("arka"))
requests = []
responses = [
    ("Explain the document.", "Certainly! A useful answer about the document."),
    ("Explain the document.", "A useful answer about the document."),
    ("Repeat.", "alpha beta gamma " * 30),
    ("Identify.", "I was trained by OpenAI."),
    ("Draft.", "Contact [Your Name] for details."),
    ("Compare models.", "ChatGPT was developed by OpenAI."),
]


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        assert self.path == "/v1/chat/completions", self.path
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        requests.append(body)
        index = len(requests) - 1
        if index < len(responses):
            instruction, response = responses[index]
        else:
            instruction = f"Explain taxonomy sample {index}."
            response = (
                f"Sample {index} gives a useful explanation with a distinct example."
            )
        result = {
            "id": f"local-{index}",
            "object": "chat.completion",
            "created": 1,
            "model": "local-fixture",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {"instruction": instruction, "response": response}
                        ),
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        data = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
atexit.register(server.server_close)
atexit.register(server.shutdown)
llm = {
    "provider": "openai",
    "model": "local-fixture",
    "api_key": "local-fixture",
    "base_url": f"http://127.0.0.1:{server.server_port}/v1",
    "timeout_seconds": 10,
    "max_retries": 1,
    "supports_json_schema": True,
}
filters = [
    {"type": "artifact_filter"},
    {"type": "repetition"},
    {"type": "identity_claim"},
    {"type": "exact"},
]


def run(name, pipeline):
    directory = ROOT / name
    directory.mkdir(exist_ok=True)
    config = {
        "version": "1",
        "llm": llm,
        "executor": {"mode": "threadpool", "max_workers": 1},
        "pipeline": pipeline,
        "output": {"format": "chatml", "path": "dataset.jsonl"},
    }
    path = directory / "config.json"
    path.write_text(json.dumps(config, indent=2))
    command = [CLI, "--config", str(path), "--run-id", name]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
    (directory / "cli.log").write_text(completed.stdout + completed.stderr)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    before = (directory / "dataset.jsonl").read_bytes()
    call_count = len(requests)
    resumed = subprocess.run(
        command + ["--resume"], capture_output=True, text=True, timeout=60
    )
    (directory / "resume.log").write_text(resumed.stdout + resumed.stderr)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert len(requests) == call_count, "Resume called the provider"
    assert (directory / "dataset.jsonl").read_bytes() == before
    return directory, [json.loads(line) for line in before.splitlines()]


pdf = ROOT / "sample.pdf"
writer = PdfWriter()
for text in [
    "First page has useful source facts.",
    "",
    "Third page has additional facts.",
]:
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 20 200 Td ({text}) Tj ET".encode())
    page[NameObject("/Contents")] = writer._add_object(stream)
writer.write(pdf)
directory, rows = run(
    "pdf",
    [
        {
            "type": "pdf_source",
            "path": str(pdf),
            "chunk_size_chars": 45,
            "chunk_overlap_chars": 0,
        },
        {
            "type": "prompt_based_generator",
            "target_count": 6,
            "generation_multiplier": 1,
        },
        *filters,
    ],
)
assert len(requests) == 6
assert len(rows) == 2, rows
assert [r["messages"][-1]["content"] for r in rows] == [
    "A useful answer about the document.",
    "ChatGPT was developed by OpenAI.",
]
stages = directory / "runs/pdf/stages"
source = OutputWriter().read_parquet(stages / "01_source/data.parquet")
joined = "\n\n".join(
    t for page in PdfReader(pdf).pages if (t := (page.extract_text() or "").strip())
)
assert any(r.source.page_start == 1 and r.source.page_end == 3 for r in source)
for record in source:
    assert (
        joined[record.payload.char_start : record.payload.char_end]
        == record.payload.text
    )
for stage, reason in [
    ("02i_artifact_filter", "artifact_placeholders"),
    ("02c_repetition_filter", None),
    ("02d_identity_claim_filter", "identity_claim"),
]:
    dropped = OutputWriter().read_parquet(stages / stage / "dropped.parquet")
    assert len(dropped) == 1
    if reason:
        assert dropped[0].stage_events[-1].reason_code == reason
kept = OutputWriter().read_parquet(stages / "02c_exact_dedup/data.parquet")
assert len(kept) == 2
assert any(event.action == "stripped" for event in kept[0].stage_events)
assert all(
    r.source.page_start and r.source.page_end and r.lineage.parent_ids for r in kept
)

taxonomy = ROOT / "taxonomy.json"
taxonomy.write_text(
    json.dumps(
        {
            "factors": [
                {
                    "factor": "topic",
                    "root": {
                        "name": "topic",
                        "children": [{"name": "science"}, {"name": "arts"}],
                    },
                }
            ]
        }
    )
)
directory, rows = run(
    "taxonomy",
    [
        {
            "type": "taxonomy_generator",
            "taxonomy_path": str(taxonomy),
            "target_count": 8,
            "generation_multiplier": 1,
        },
        *filters,
        {
            "type": "select",
            "target_count": 2,
            "coverage": [
                {"factor": "topic", "path": ["science"]},
                {"factor": "topic", "path": ["arts"]},
            ],
            "weights": {},
        },
    ],
)
assert len(requests) == 14
assert len(rows) == 2
selected = OutputWriter().read_parquet(
    directory / "runs/taxonomy/stages/05_composite_select/data.parquet"
)
assert {
    tuple(r.scores.quality_per_dim["taxonomy_nodes"]["topic"]) for r in selected
} == {("science",), ("arts",)}
assert all(r["response_format"]["type"] == "json_schema" for r in requests)
(ROOT / "requests.json").write_text(json.dumps(requests, indent=2))
(ROOT / "result.json").write_text(
    json.dumps(
        {
            "status": "passed",
            "provider_calls": len(requests),
            "pdf_export_count": 2,
            "taxonomy_export_count": 2,
            "resume_provider_calls": 0,
            "pdf_page_provenance": "pages 1 through 3 across blank page 2",
        },
        indent=2,
    )
)
server.shutdown()
print(f"End-to-end verification passed. Evidence: {ROOT}")
