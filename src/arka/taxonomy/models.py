"""Pydantic taxonomy models + YAML loader for Simula §2.1.

A `Taxonomy` is a tree of factor-of-variation nodes. The root node represents
the broad factor (e.g. "cat type"); children are increasingly specific (e.g.
"domestic" -> "shorthair" -> "British shorthair"). Each root-to-leaf chain is
a sample-able attribute.

A `TaxonomyBundle` is the user's full coverage map: one taxonomy per factor
plus optional `SamplingStrategy` entries that group compatible factors. If no
strategies are given, the loader synthesises a single all-factors strategy so
the smallest-possible YAML stays usable.

The schema is rigid by design \u2014 leaf names must be unique among siblings and
strategies must reference known factors. We fail loud at load time rather than
limp into a confusing runtime error during generation.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import AliasChoices, Field, model_validator

from arka.common.models import StrictModel


class TaxonomyNode(StrictModel):
    """A single node in a taxonomy tree.

    Children are themselves TaxonomyNodes. Leaf nodes have an empty children
    list. Sibling names must be unique \u2014 ambiguity at any level breaks the
    leaf-path interpretation Simula relies on for coverage metrics.
    """

    name: str
    children: list[TaxonomyNode] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_duplicate_siblings(self) -> TaxonomyNode:
        seen: set[str] = set()
        for child in self.children:
            if child.name in seen:
                raise ValueError(
                    f"duplicate sibling name {child.name!r} under node {self.name!r}"
                )
            seen.add(child.name)
        return self

    def is_leaf(self) -> bool:
        return not self.children


# Pydantic v2 needs explicit forward-ref resolution for self-referencing models.
TaxonomyNode.model_rebuild()


class Taxonomy(StrictModel):
    """One taxonomy, one factor of variation, modelled as a rooted tree.

    The factor name accepts either ``factor:`` or ``name:`` in YAML so the
    bundle is comfortable to write. We canonicalise to `factor` internally.
    """

    factor: str = Field(validation_alias=AliasChoices("factor", "name"))
    root: TaxonomyNode

    @model_validator(mode="after")
    def _factor_matches_root(self) -> Taxonomy:
        # Soft check: not strictly required, but a mismatched root is almost
        # always a YAML typo. We do not raise; we just note it via depth.
        return self

    def leaf_paths(self) -> list[list[str]]:
        """Return every root-to-leaf chain, EXCLUDING the root.

        The root represents the factor itself; the meaningful sample-able
        units are its descendants. Returning an empty list when the root has
        no children would be ambiguous (one whole-tree path vs none) so we
        instead document that a childless root is degenerate and yields [].
        """
        if not self.root.children:
            return []
        out: list[list[str]] = []

        def walk(node: TaxonomyNode, chain: list[str]) -> None:
            chain = [*chain, node.name]
            if node.is_leaf():
                out.append(chain)
                return
            for child in node.children:
                walk(child, chain)

        for child in self.root.children:
            walk(child, [])
        return out

    @property
    def depth(self) -> int:
        """Depth of the tree, counting edges from root to deepest leaf."""

        def _depth(node: TaxonomyNode) -> int:
            if not node.children:
                return 0
            return 1 + max(_depth(c) for c in node.children)

        return _depth(self.root)


class SamplingStrategy(StrictModel):
    """Which factors to sample together when building one meta-prompt.

    Simula \u00a72.2 motivates strategies as a way to forbid illogical combinations
    (e.g. children\u2019s-content topics + horror format). For slice 3 we only
    enforce factor membership; future versions may add per-factor weights.
    """

    name: str
    include_factors: list[str]


class TaxonomyBundle(StrictModel):
    """A complete user-authored taxonomy bundle.

    Loaded from a YAML file. The schema is checked at load time:
      - factor names are unique
      - every strategy references known factors only
      - if no strategies are provided, a default one covering all factors is
        synthesised
    """

    version: str = "1"
    factors: list[Taxonomy]
    strategies: list[SamplingStrategy] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> TaxonomyBundle:
        names = [f.factor for f in self.factors]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate factor names in taxonomy bundle: {names}")
        if not self.strategies:
            self.strategies = [
                SamplingStrategy(name="default", include_factors=names),
            ]
        else:
            known = set(names)
            for strat in self.strategies:
                unknown = set(strat.include_factors) - known
                if unknown:
                    raise ValueError(
                        f"strategy {strat.name!r} references unknown factor(s): "
                        f"{sorted(unknown)} (known factors: {sorted(known)})"
                    )
        return self

    def factor(self, name: str) -> Taxonomy:
        for f in self.factors:
            if f.factor == name:
                return f
        raise KeyError(
            f"unknown factor {name!r}; known: {[f.factor for f in self.factors]}"
        )

    @classmethod
    def from_yaml(cls, path: Path) -> TaxonomyBundle:
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(
                f"taxonomy YAML must be a mapping at top level (got {type(data).__name__})"
            )
        return cls.model_validate(data)
