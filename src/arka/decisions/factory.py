from collections.abc import Callable

from arka.decisions.client import DecisionClient
from arka.decisions.models import DecisionConfig, DecisionOverride

DecisionClientFactory = Callable[[DecisionConfig], DecisionClient]


def resolve_decision_config(
    base: DecisionConfig, override: DecisionOverride | None
) -> DecisionConfig:
    if override is None:
        return base
    # model_dump excludes credentials. Preserve the SecretStr during merging.
    values = {name: getattr(base, name) for name in DecisionConfig.model_fields}
    values.update(
        {
            name: getattr(override, name)
            for name in type(override).model_fields
            if getattr(override, name) is not None
        }
    )
    return DecisionConfig.model_validate(values)


def build_decision_client(
    config: DecisionConfig,
    override: DecisionOverride | None = None,
    factory: DecisionClientFactory | None = None,
    **kwargs,
) -> DecisionClient:
    effective = resolve_decision_config(config, override)
    return (
        factory(effective)
        if factory is not None
        else DecisionClient(effective, **kwargs)
    )
