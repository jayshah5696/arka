import sys
from unittest.mock import patch
from pydantic import BaseModel
import arka.cli
import logging
logging.basicConfig(level=logging.DEBUG)

def patched_complete_structured(self, *args, **kwargs):
    from arka.llm.models import LLMOutput, TokenUsage
    class ResponseInner(BaseModel):
        instruction: str
        response: str
    return LLMOutput(
        text='{"instruction": "Tell me about the project.", "response": "We are working on SUPER_SECRET_PROJECT_X which is very cool."}',
        parsed=ResponseInner(instruction="Tell me about the project.", response="We are working on SUPER_SECRET_PROJECT_X which is very cool."),
        model="mock-model",
        provider="openai",
        latency_ms=100.0,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=10, cost_usd=None),
        finish_reason="stop"
    )

def patched_embed(self, *args, **kwargs):
    import numpy as np
    texts = kwargs.get('texts')
    return np.array([[0.1, 0.2, 0.3] for _ in texts], dtype=float)

def patched_runner_embed(self, *args, **kwargs):
    import numpy as np
    texts = kwargs.get('texts')
    return np.array([[0.1, 0.2, 0.3] for _ in texts], dtype=float)


def main():
    from arka.llm.client import LLMClient
    from arka.pipeline.filter_stages import SemanticSimilarityFilterStage
    from arka.pipeline.runner import PipelineRunner

    with patch.object(LLMClient, 'complete_structured', new=patched_complete_structured):
        with patch.object(SemanticSimilarityFilterStage, '_embed_texts', new=patched_embed):
            with patch.object(PipelineRunner, '_embed_texts', new=patched_runner_embed):
                try:
                    arka.cli.main()
                except SystemExit as e:
                    print(f"Exited with {e}")

if __name__ == '__main__':
    sys.argv = ['arka', '--config', 'examples/08-privacy-guardrails.yaml', '--run-id', 'test-08']
    main()
