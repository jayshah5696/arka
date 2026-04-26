"""Slice 3 \u2014 Pydantic taxonomy schema + YAML loader.

Covers Simula \u00a72.1: a taxonomy is a tree of factor-of-variation nodes. A
TaxonomyBundle is a set of taxonomies (one per factor) plus optional sampling
strategies that group factors together (e.g. \"children's content\": pick from
the topic + format taxonomies but skip mature themes).

Tests:
1. A `TaxonomyNode` validates and supports recursive children.
2. A `Taxonomy` exposes `leaf_paths()` returning all root-to-leaf chains.
3. A `TaxonomyBundle` loads from YAML.
4. `TaxonomyBundle.strategies` defaults to one all-factors strategy if absent.
5. A node with the same name as a sibling raises during validation.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_taxonomy_node_supports_children() -> None:
    from arka.taxonomy.models import TaxonomyNode

    node = TaxonomyNode(
        name="cat",
        children=[
            TaxonomyNode(name="domestic", children=[TaxonomyNode(name="shorthair")]),
            TaxonomyNode(name="wild"),
        ],
    )
    assert node.name == "cat"
    assert len(node.children) == 2
    assert node.children[0].children[0].name == "shorthair"


def test_taxonomy_leaf_paths_returns_root_to_leaf_chains() -> None:
    from arka.taxonomy.models import Taxonomy, TaxonomyNode

    tax = Taxonomy(
        factor="cat_type",
        root=TaxonomyNode(
            name="cat",
            children=[
                TaxonomyNode(
                    name="domestic", children=[TaxonomyNode(name="shorthair")]
                ),
                TaxonomyNode(name="wild"),
            ],
        ),
    )
    paths = tax.leaf_paths()
    # Each leaf path is the chain from (excluding root) to leaf.
    # We expect: ['domestic', 'shorthair'] and ['wild'].
    path_strs = [" / ".join(p) for p in paths]
    assert "domestic / shorthair" in path_strs
    assert "wild" in path_strs
    assert len(paths) == 2


def test_taxonomy_node_rejects_duplicate_sibling_names() -> None:
    from arka.taxonomy.models import TaxonomyNode

    with pytest.raises(ValueError, match="duplicate"):
        TaxonomyNode(
            name="root",
            children=[
                TaxonomyNode(name="x"),
                TaxonomyNode(name="x"),
            ],
        )


def test_taxonomy_bundle_loads_from_yaml(tmp_path: Path) -> None:
    from arka.taxonomy.models import TaxonomyBundle

    yaml_path = tmp_path / "tax.yaml"
    yaml_path.write_text(
        """
version: "1"
factors:
  - name: domain
    root:
      name: domain_root
      children:
        - name: email
        - name: blog
        - name: technical
  - name: tone
    root:
      name: tone_root
      children:
        - name: casual
        - name: professional
strategies:
  - name: default
    include_factors: [domain, tone]
""".strip()
    )
    bundle = TaxonomyBundle.from_yaml(yaml_path)
    assert len(bundle.factors) == 2
    assert bundle.factor("domain").root.children[0].name == "email"
    assert len(bundle.strategies) == 1
    assert bundle.strategies[0].include_factors == ["domain", "tone"]


def test_taxonomy_bundle_defaults_to_all_factors_strategy(tmp_path: Path) -> None:
    """If no strategies are listed, sample across all factors uniformly.\n
    This keeps the smallest-possible YAML usable."""
    from arka.taxonomy.models import TaxonomyBundle

    yaml_path = tmp_path / "tax.yaml"
    yaml_path.write_text(
        """
version: "1"
factors:
  - name: domain
    root:
      name: domain_root
      children:
        - name: email
        - name: blog
""".strip()
    )
    bundle = TaxonomyBundle.from_yaml(yaml_path)
    assert len(bundle.strategies) == 1
    assert set(bundle.strategies[0].include_factors) == {"domain"}


def test_taxonomy_bundle_rejects_strategy_referencing_unknown_factor(
    tmp_path: Path,
) -> None:
    from arka.taxonomy.models import TaxonomyBundle

    yaml_path = tmp_path / "tax.yaml"
    yaml_path.write_text(
        """
version: "1"
factors:
  - name: domain
    root:
      name: domain_root
      children:
        - name: email
strategies:
  - name: oops
    include_factors: [domain, nonexistent]
""".strip()
    )
    with pytest.raises(ValueError, match="unknown factor"):
        TaxonomyBundle.from_yaml(yaml_path)


def test_taxonomy_bundle_factor_lookup_raises_for_missing(tmp_path: Path) -> None:
    from arka.taxonomy.models import TaxonomyBundle

    yaml_path = tmp_path / "tax.yaml"
    yaml_path.write_text(
        """
version: "1"
factors:
  - name: domain
    root:
      name: domain_root
      children:
        - name: email
""".strip()
    )
    bundle = TaxonomyBundle.from_yaml(yaml_path)
    with pytest.raises(KeyError):
        bundle.factor("missing")
