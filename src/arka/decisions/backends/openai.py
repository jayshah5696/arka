from arka.decisions.models import DecisionQuestion, ScoreQuestion, YesNoQuestion


class OpenAIBackend:
    def endpoint(self, model: str) -> str:
        return "decisions"

    def request(
        self, state: str, questions: list[DecisionQuestion], model: str
    ) -> dict:
        wire = []
        for q in questions:
            item = {"name": q.name, "instructions": q.instructions, "type": q.type}
            if isinstance(q, YesNoQuestion):
                item["type"] = "predicate"
                criteria = [
                    f"{name}: {value}"
                    for name, value in [("Yes", q.yes), ("No", q.no)]
                    if value is not None
                ]
                if criteria:
                    item["instructions"] += "\n" + "\n".join(criteria)
            elif isinstance(q, ScoreQuestion):
                item["levels"] = [
                    {"label": str(i), "description": text}
                    for i, text in enumerate(q.levels)
                ]
            else:
                item["choices"] = [
                    {
                        "value": value,
                        **({"description": text} if text is not None else {}),
                    }
                    for value, text in q.choices.items()
                ]
            wire.append(item)
        return {"model": model, "input": state, "questions": wire}

    def parse(self, payload: dict) -> dict:
        answers = {}
        for raw in payload["answers"]:
            name = raw["name"]
            if name in answers:
                raise ValueError("duplicate answer name")
            if raw["type"] == "predicate":
                answer = {"type": "yes_no", "probability": raw["probability"]}
            elif raw["type"] in ("choice", "score"):
                entries = raw["probabilities"]
                probabilities = {
                    entry["value"]: entry["probability"] for entry in entries
                }
                if len(entries) != len(probabilities):
                    raise ValueError("duplicate probability option")
                answer = {
                    "type": raw["type"],
                    raw["type"]: raw[raw["type"]],
                    "probabilities": probabilities,
                    "confidence": raw["confidence"],
                }
            else:
                raise ValueError("provider refused or returned an unknown answer type")
            answers[name] = answer
        return answers
