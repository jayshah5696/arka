from arka.decisions.backends.openai import OpenAIBackend
from arka.decisions.backends.openrouter import OpenRouterBackend
from arka.decisions.backends.typesafe import ClefBackend, TypeSafeBackend

# An adapter implements endpoint(model), request(state, questions, model),
# and parse(payload). Stages do not know which wire format it uses.
BACKENDS = {
    "openai": OpenAIBackend(),
    "openrouter": OpenRouterBackend(),
    "typesafe": TypeSafeBackend(),
    "clef": ClefBackend(),
}
