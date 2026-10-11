"""Verify the new stack through actual CLI, HTTP, fastText, and HF formats."""

import argparse
import atexit
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject

from arka.pipeline.output import OutputWriter

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
ROOT = args.output.resolve()
ROOT.mkdir(parents=True, exist_ok=False)
model_path = Path(os.environ["ARKA_TEST_LID_MODEL"]).resolve()
if not model_path.is_file():
    parser.error("ARKA_TEST_LID_MODEL must name an existing local model")
german = "Das ist eine deutsche Erklärung über die Lösung dieser Aufgabe."
english = "This is an English explanation about the solution to this problem."
instruction = "Erkläre bitte die wichtigsten Eigenschaften dieser wissenschaftlichen Untersuchung."
answer = "Die Untersuchung erklärt den Zusammenhang zwischen den Beobachtungen und den Ergebnissen."
question = "In welchem Jahr wurde der Eiffelturm in Paris gebaut?"
outputs = [
    {
        "instruction": instruction,
        "response": "Certainly! " + answer,
        "reasoning": german,
    },
    {"instruction": instruction, "response": answer, "reasoning": english},
    {
        "instruction": question,
        "response": "Der Eiffelturm wurde im Jahr 1889 gebaut.",
        "reasoning": german,
    },
    {"instruction": instruction, "response": "I am ChatGPT.", "reasoning": german},
    {
        "instruction": instruction,
        "response": "Wiederhole diese Worte " * 40,
        "reasoning": german,
    },
    {
        "instruction": instruction,
        "response": "Bitte fragen Sie [Your Name].",
        "reasoning": german,
    },
    {"instruction": instruction, "response": answer},
]
requests = []


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        assert self.path == "/v1/chat/completions"
        requests.append(
            json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        )
        payload = outputs[(len(requests) - 1) % len(outputs)]
        data = json.dumps(
            {
                "id": f"fixture-{len(requests)}",
                "object": "chat.completion",
                "created": 1,
                "model": "local-fixture",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(payload),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            }
        ).encode()
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
writer = PdfWriter()
for body in [
    "A source page with unique scientific facts.",
    "B source page with different experimental results.",
]:
    page = writer.add_blank_page(width=300, height=300)
    stream = DecodedStreamObject()
    stream.set_data(f"BT ({'Header' * 100}{body}) Tj ET".encode())
    page.replace_contents(stream)
writer.write(ROOT / "sample.pdf")
(ROOT / "eval.jsonl").write_text(json.dumps({"question": question}))
config = {
    "version": "1",
    "llm": {
        "provider": "openai",
        "model": "local-fixture",
        "api_key": "local-fixture",
        "base_url": f"http://127.0.0.1:{server.server_port}/v1",
        "supports_json_schema": True,
        "max_retries": 1,
    },
    "executor": {"mode": "threadpool", "max_workers": 1},
    "pipeline": [
        {
            "type": "pdf_source",
            "path": "sample.pdf",
            "boilerplate_strip": True,
            "min_remaining_bytes": 1,
            "chunk_size_chars": 80,
            "chunk_overlap_chars": 0,
        },
        {
            "type": "prompt_based_generator",
            "target_count": 7,
            "generation_multiplier": 1,
        },
        {"type": "artifact_filter"},
        {"type": "repetition"},
        {"type": "identity_claim"},
        {"type": "reasoning_language_gate", "lid_model_path": str(model_path)},
        {"type": "decontaminate", "eval_paths": ["eval.jsonl"], "threshold": 0.9},
        {"type": "exact"},
    ],
    "output": {"format": "chatml", "path": "dataset.jsonl"},
}
path = ROOT / "config.json"
path.write_text(json.dumps(config, indent=2))
command = [
    str(Path(sys.executable).with_name("arka")),
    "--config",
    str(path),
    "--run-id",
    "stack",
]
completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
(ROOT / "cli.log").write_text(completed.stdout + completed.stderr)
assert completed.returncode == 0, completed.stdout + completed.stderr
assert len(requests) == 7
before = (ROOT / "dataset.jsonl").read_bytes()
rows = [json.loads(line) for line in before.splitlines()]
assert [r["messages"][-1]["content"] for r in rows] == [
    f"<think>\n{german}\n</think>\n\n{answer}",
    answer,
]
stages = ROOT / "runs/stack/stages"


def stats(stage):
    return json.loads((stages / stage / "stats.json").read_text())


assert stats("01_source")["boilerplate_bytes_removed"] == 1200
source = OutputWriter().read_parquet(stages / "01_source/data.parquet")
assert all("Header" not in r.payload.text for r in source)
assert any(r.source.page_start == 1 and r.source.page_end == 2 for r in source)
assert stats("02j_reasoning_language_gate")["stripped_count"] == 1
assert stats("02k_decontaminate")["dropped_count"] == 1
kept = OutputWriter().read_parquet(stages / "02c_exact_dedup/data.parquet")
assert [r.payload.reasoning for r in kept] == [german, None]
assert all(r.source.page_start and r.lineage.parent_ids for r in kept)
for fmt in ["jsonl", "alpaca"]:
    OutputWriter().write_jsonl(kept, ROOT / f"{fmt}.jsonl", output_format=fmt)
resumed = subprocess.run(
    command + ["--resume"], capture_output=True, text=True, timeout=60
)
(ROOT / "resume.log").write_text(resumed.stdout + resumed.stderr)
assert resumed.returncode == 0, resumed.stdout + resumed.stderr
assert len(requests) == 7 and (ROOT / "dataset.jsonl").read_bytes() == before
(ROOT / "requests.json").write_text(json.dumps(requests, indent=2))
server.shutdown()
(ROOT / "result.json").write_text(
    json.dumps(
        {
            "status": "passed",
            "provider_calls": 7,
            "resume_provider_calls": 0,
            "export_count": 2,
            "boilerplate_bytes_removed": 1200,
            "traces_stripped": 1,
            "benchmark_leaks_dropped": 1,
            "real_fasttext_model": model_path.name,
        },
        indent=2,
    )
)
print(f"Full-stack end-to-end verification passed: {ROOT}")
