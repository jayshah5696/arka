from arka.decisions.backends.openai import OpenAIBackend
from arka.decisions.backends.typesafe import ClefBackend, TypeSafeBackend

# An adapter implements endpoint(model), request(state, questions, model),
# and parse(payload). Stages do not know which wire format it uses.
BACKENDS = {
    "openai": OpenAIBackend(),
    "typesafe": TypeSafeBackend(),
    "clef": ClefBackend(),
}
