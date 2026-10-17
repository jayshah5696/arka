from arka.decisions.models import DecisionQuestion, ScoreQuestion, YesNoQuestion


class TypeSafeBackend:
    def endpoint(self, model: str) -> str:
        return "systemone"

    def request(
        self, state: str, questions: list[DecisionQuestion], model: str
    ) -> dict:
        wire = {}
        for q in questions:
            item = {"type": q.type, "instructions": q.instructions}
            if isinstance(q, YesNoQuestion):
                item["type"] = "noul"
                criteria = {
                    k: v for k, v in [("true", q.yes), ("false", q.no)] if v is not None
                }
                if criteria:
                    item["criteria"] = criteria
            else:
                item["criteria"] = (
                    q.levels if isinstance(q, ScoreQuestion) else q.choices
                )
            wire[q.name] = item
        return {"model": model, "state": state, "questions": wire}

    def parse(self, payload: dict) -> dict:
        answers = {}
        if not isinstance(payload["answers"], dict):
            raise ValueError("answers must be an object")
        for name, raw in payload["answers"].items():
            if raw["type"] == "noul":
                answer = {"type": "yes_no", "probability": raw["noul"]}
            elif raw["type"] in ("choice", "score"):
                answer = {
                    key: raw[key]
                    for key in ("type", raw["type"], "probabilities", "confidence")
                }
            else:
                raise ValueError("unknown answer type")
            answers[name] = answer
        return answers


class ClefBackend(TypeSafeBackend):
    def endpoint(self, model: str) -> str:
        return f"@cf/cloudflare/{model}"
