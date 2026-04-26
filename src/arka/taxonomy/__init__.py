"""Simula taxonomy support: factors of variation modelled as trees of nodes.

Slice 3 ships only the schema + YAML loader and a sampling-aware generator.
M3-driven Best-of-N taxonomy expansion (Simula §2.1) is intentionally deferred.
"""

from arka.taxonomy.models import (
    SamplingStrategy,
    Taxonomy,
    TaxonomyBundle,
    TaxonomyNode,
)

__all__ = [
    "SamplingStrategy",
    "Taxonomy",
    "TaxonomyBundle",
    "TaxonomyNode",
]
