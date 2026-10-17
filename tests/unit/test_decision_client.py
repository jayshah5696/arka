"""Decision failures: bad schemas, partial answers, cache collisions, retries,
secret leaks, and unbounded concurrency must not change a decision silently.
"""

import json
import threading
import time

import httpx
import pytest
from pydantic import ValidationError

from arka.decisions.banding import band
from arka.decisions.client import DecisionClient, DecisionClientError
from arka.decisions.models import (
    ChoiceQuestion,
    DecisionConfig,
    DecisionRequest,
    ScoreQuestion,
    YesNoQuestion,
)
from arka.pipeline.checkpoint import CheckpointManager


def config(provider="openai", **kwargs):
    return DecisionConfig(
        provider=provider,
        model="test",
        api_key="secret-test",
        base_url="https://example.test/v1",
        **kwargs,
    )


def questions():
    return [
        YesNoQuestion(name="usable", instructions="Usable?", yes="content", no="junk"),
        ChoiceQuestion(
            name="domain", instructions="Domain?", choices={"a": "Alpha", "b": None}
        ),
        ScoreQuestion(name="quality", instructions="Quality?", levels=["Poor", "Good"]),
    ]


def reply(provider, names):
    answers = {
        "usable": {"type": "predicate", "probability": 0.9},
        "domain": {
            "type": "choice",
            "choice": "a",
            "probabilities": {"a": 0.8, "b": 0.2},
            "confidence": 0.7,
        },
        "quality": {
            "type": "score",
            "score": 0.75,
            "probabilities": {"0": 0.25, "1": 0.75},
            "confidence": 0.5,
        },
    }
    if provider == "openai":
        result = []
        for name in names:
            answer = {"name": name, **answers[name]}
            if "probabilities" in answer:
                answer["probabilities"] = [
                    {"value": int(k) if name == "quality" else k, "probability": v}
                    for k, v in answer["probabilities"].items()
                ]
            result.append(answer)
    else:
        answers["usable"] = {"type": "noul", "noul": 0.9}
        result = {name: answers[name] for name in names}
    data = {
        "model": "test",
        "answers": result,
        "usage": {"input_tokens": 12, "output_tokens": 0},
    }
    return {"success": True, "result": data} if provider == "clef" else data


@pytest.mark.parametrize(
    "provider,endpoint",
    [
        ("openai", "/v1/decisions"),
        ("typesafe", "/v1/systemone"),
        ("clef", "/v1/@cf/cloudflare/clef"),
    ],
)
def test_wire_formats_and_typed_answers(provider, endpoint):
    captured = []

    def handle(request):
        captured.append(request)
        return httpx.Response(200, json=reply(provider, [q.name for q in questions()]))

    cfg = config(provider)
    if provider == "clef":
        cfg = cfg.model_copy(update={"model": "clef"})
    with DecisionClient(cfg, transport=httpx.MockTransport(handle)) as client:
        result = client.decide("evidence", questions())
    payload = json.loads(captured[0].content)
    assert captured[0].url.path == endpoint
    assert payload["input" if provider == "openai" else "state"] == "evidence"
    wire_questions = payload["questions"]
    if provider == "openai":
        assert wire_questions[0]["type"] == "predicate"
        assert wire_questions[1]["choices"] == [
            {"value": "a", "description": "Alpha"},
            {"value": "b"},
        ]
        assert wire_questions[2]["levels"][1]["description"] == "Good"
    else:
        assert wire_questions["usable"]["criteria"] == {
            "true": "content",
            "false": "junk",
        }
        assert wire_questions["quality"]["criteria"] == ["Poor", "Good"]
    assert result.answers["usable"].type == "yes_no"
    assert result.answers["usable"].probability == 0.9
    assert result.answers["quality"].probabilities == {0: 0.25, 1: 0.75}
    assert result.usage.prompt_tokens == 12


@pytest.mark.parametrize(
    "status,retries", [(429, 3), (529, 3), (503, 3), (401, 1), (422, 1)]
)
def test_retry_policy_and_safe_errors(status, retries):
    calls, delays = [], []

    def handle(request):
        calls.append(request)
        return httpx.Response(status, text="secret-test")

    with DecisionClient(
        config(max_retries=3),
        transport=httpx.MockTransport(handle),
        sleep=delays.append,
    ) as client:
        with pytest.raises(DecisionClientError) as exc:
            client.decide("evidence", questions())
    assert len(calls) == retries
    assert len(delays) == retries - 1
    assert "secret-test" not in str(exc.value)


def test_answer_cache_survives_restart_and_sends_only_missing_questions(tmp_path):
    sent = []

    def handle(request):
        names = [q["name"] for q in json.loads(request.content)["questions"]]
        sent.append(names)
        return httpx.Response(200, json=reply("openai", names))

    checkpoint = CheckpointManager(tmp_path / "cache.sqlite")
    with DecisionClient(
        config(), checkpoint=checkpoint, transport=httpx.MockTransport(handle)
    ) as client:
        client.decide("evidence", questions()[:1])
        result = client.decide("evidence", questions())
        assert result.cache_hits == 1
    checkpoint.close()
    checkpoint = CheckpointManager(tmp_path / "cache.sqlite")
    with DecisionClient(
        config(), checkpoint=checkpoint, transport=httpx.MockTransport(handle)
    ) as client:
        assert client.decide("evidence", questions()).requests_sent == 0
        client.decide("changed evidence", questions()[:1])
        client.decide(
            "evidence", [questions()[0].model_copy(update={"instructions": "Changed?"})]
        )
    checkpoint.close()
    assert sent == [["usable"], ["domain", "quality"], ["usable"], ["usable"]]


@pytest.mark.parametrize(
    "data",
    [
        {"answers": []},
        {"answers": [{"name": "usable", "type": "refusal"}]},
        {"answers": [{"name": "usable", "type": "predicate", "probability": 1.1}]},
        {
            "answers": [
                {
                    "name": "usable",
                    "type": "score",
                    "score": 0,
                    "probabilities": [],
                    "confidence": 1,
                }
            ]
        },
    ],
)
def test_invalid_or_refused_answers_fail_without_fallback(data):
    with DecisionClient(
        config(),
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=data)),
    ) as client:
        with pytest.raises(DecisionClientError):
            client.decide("evidence", questions()[:1])


def test_decide_many_bounds_concurrency_and_preserves_order():
    active = peak = 0
    lock = threading.Lock()

    def handle(request):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return httpx.Response(200, json=reply("openai", ["usable"]))

    with DecisionClient(config(), transport=httpx.MockTransport(handle)) as client:
        results = client.decide_many(
            [
                DecisionRequest(state=str(i), questions=questions()[:1])
                for i in range(8)
            ],
            max_workers=2,
        )
    assert len(results) == 8
    assert peak == 2


def test_schema_limits_and_secrets():
    with pytest.raises(ValidationError):
        ChoiceQuestion(
            name="x", instructions="?", choices={str(i): None for i in range(256)}
        )
    for levels in [["one"], [str(i) for i in range(11)]]:
        with pytest.raises(ValidationError):
            ScoreQuestion(name="x", instructions="?", levels=levels)
    with pytest.raises(ValidationError):
        DecisionRequest(state="x", questions=[questions()[0], questions()[0]])
    assert "secret-test" not in config().model_dump_json()


@pytest.mark.parametrize(
    "p,expected",
    [(0.85, "accept"), (0.20, "uncertain"), (0.19, "reject"), (0.5, "uncertain")],
)
def test_banding_boundaries(p, expected):
    assert band(p, accept_at=0.85, reject_at=0.20) == expected
