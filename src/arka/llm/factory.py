"""LLMClientFactory: the single seam for constructing LLMClients inside Stages.

Before this module existed, every Stage that needed to talk to a Provider
wrote some variant of::

    llm_client = self._llm_client or LLMClient(config=ctx.config.llm)
    # or with an override:
    effective = resolve_llm_override(ctx.config.llm, cfg.llm_override)
    llm_client = self._llm_client or LLMClient(config=effective)

Seven near-identical lines across generator, scoring, ifd, evol, filter
stages -- every Stage knew about the LLMClient constructor and the override
resolver. ADR-0001 explicitly listed "deduplicate OpenAI client construction"
as deferred cleanup; this factory makes it real.

Stages now ask the StageContext for a client::

    llm_client = self._llm_client or ctx.llm_client(override=cfg.llm_override)

The ``self._llm_client`` constructor seam is preserved for tests (per the
agreed migration path) -- existing fakes injected into Stage __init__
still work.
"""

from __future__ import annotations

from typing import Protocol

from arka.config.models import LLMConfig, StageLLMOverride, resolve_llm_override

# Re-exported here so tests can ``monkeypatch.setattr('arka.llm.factory.LLMClient', Fake)``
# to swap the production client without injecting a factory by hand. This is
# the canonical patch target for tests that previously patched the per-stage
# imports (e.g. ``arka.pipeline.generator_stages.LLMClient``).
from arka.llm.client import LLMClient


class LLMClientFactory(Protocol):
    """Anything callable that turns a resolved LLMConfig into an LLMClient.

    Tests can pass a fake factory by hand; the default implementation just
    constructs a real LLMClient.
    """

    def __call__(self, config: LLMConfig) -> LLMClient: ...


def default_factory(config: LLMConfig) -> LLMClient:
    """The production LLMClientFactory. Builds a real LLMClient.

    Looks up ``LLMClient`` on this module rather than capturing it at import
    time so that ``monkeypatch.setattr('arka.llm.factory.LLMClient', Fake)``
    works for tests.
    """
    import arka.llm.factory as factory_module

    return factory_module.LLMClient(config=config)


def build_client(
    *,
    base_config: LLMConfig,
    override: StageLLMOverride | None = None,
    factory: LLMClientFactory | None = None,
) -> LLMClient:
    """Resolve any per-Stage override against the base LLMConfig and build a client.

    The single canonical entry point used by ``StageContext.llm_client``.
    ``factory`` defaults to :func:`default_factory`; tests inject their own.
    """
    effective = resolve_llm_override(base_config, override)
    return (factory or default_factory)(effective)
