import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(
    description="Load exports and check Qwen3 reasoning tokens"
)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = args.output.resolve()
dataset = load_dataset("json", data_files=str(root / "dataset.jsonl"), split="train")
canonical = load_dataset("json", data_files=str(root / "jsonl.jsonl"), split="train")
alpaca = load_dataset("json", data_files=str(root / "alpaca.jsonl"), split="train")
assert len(dataset) == len(canonical) == len(alpaca) == 2
assert canonical[0]["reasoning"] and canonical[1]["reasoning"] is None
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
rendered = tokenizer.apply_chat_template(
    dataset[0]["messages"], tokenize=False, add_generation_prompt=False
)
assert canonical[0]["reasoning"] in rendered and canonical[0]["response"] in rendered
assert rendered.count("<think>") == rendered.count("</think>") == 1
tokenized = tokenizer.apply_chat_template(
    dataset[0]["messages"],
    tokenize=True,
    add_generation_prompt=False,
    return_dict=False,
)
assert len(tokenized) > 20
decoded = tokenizer.decode(tokenized)
assert canonical[0]["reasoning"] in decoded and canonical[0]["response"] in decoded
assert canonical[0]["reasoning"] in alpaca[0]["output"]
(root / "hf-rendered.txt").write_text(rendered)
result = {
    "status": "passed",
    "dataset_rows": len(dataset),
    "formats_loaded": ["chatml", "jsonl", "alpaca"],
    "tokenizer": "Qwen/Qwen3-0.6B",
    "retained_reasoning_and_answer": True,
    "think_blocks": 1,
    "training_tokens": len(tokenized),
}
(root / "hf-result.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result))
