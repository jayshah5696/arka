from arka.decisions.backends.typesafe import TypeSafeBackend


class OpenRouterBackend(TypeSafeBackend):
    """OpenRouter routes the Jev question map through its alpha Decisions API."""

    def endpoint(self, model: str) -> str:
        return "decisions"
