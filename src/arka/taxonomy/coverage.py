"""Slice 4 \u2014 Simula \u00a72.3 Level-Ratio Taxonomic Coverage.

For each factor and each depth in that factor's taxonomy, the level-ratio
coverage is::

    coverage(F, L) = |unique nodes at depth L hit by the dataset|
                     -------------------------------------------
                     |total nodes at depth L in the taxonomy|

The function works directly off the per-record audit trail
``record.scores.quality_per_dim['taxonomy_nodes']`` that
:mod:`arka.pipeline.taxonomy_generator` writes. M3-driven taxonomy assignment
for unlabelled datasets is a separate slice (4.5).

Failure modes are explicit, not silent:
- A record with ``None`` or empty ``taxonomy_nodes`` contributes zero \u2014 the
  coverage drops, the call does not raise.
- A factor name that doesn't exist in the bundle is recorded under
  ``unknown_factors`` and ignored.
- A path containing a node that doesn't exist at its depth in the taxonomy is
  recorded under ``unknown_nodes`` and is NOT counted toward coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arka.taxonomy.models import Taxonomy, TaxonomyBundle, TaxonomyNode

# A "sampled record" is the per-record taxonomy assignment from
# scores.quality_per_dim['taxonomy_nodes']. Shape:
#   { factor_name: [chain_of_node_names_excluding_the_root], ... }
SampledRecord = dict[str, list[str]] | None


@dataclass(frozen=True)
class CoverageReport:
    """Result of :func:`level_ratio_coverage`."""

    by_level: dict[int, float]
    """Aggregate coverage ratio per depth, averaged across all factors."""

    by_factor: dict[str, dict[int, float]]
    """Per-factor coverage ratio per depth."""

    unknown_factors: set[str] = field(default_factory=set)
    """Factor names that appeared in records but not in the bundle."""

    unknown_nodes: list[str] = field(default_factory=list)
    """Human-readable strings noting node-paths the bundle didn't recognise.

    Each entry looks like ``factor_name:level:node_name``. Useful for
    debugging when datasets and taxonomies drift.
    """


def _nodes_by_depth(tax: Taxonomy) -> dict[int, set[str]]:
    """Return {depth: {node_name, ...}} for every depth that exists in `tax`.

    Depth 1 = root's children (the first sample-able level).
    Depth 2 = grandchildren. Depth 0 (the root) is excluded by convention,
    matching how ``Taxonomy.leaf_paths`` reports chains.
    """
    out: dict[int, set[str]] = {}

    def walk(node: TaxonomyNode, depth: int) -> None:
        if depth >= 1:
            out.setdefault(depth, set()).add(node.name)
        for child in node.children:
            walk(child, depth + 1)

    walk(tax.root, 0)
    return out


def _record_paths_against_taxonomy(
    tax: Taxonomy,
    record_chain: list[str],
    factor_name: str,
    unknown_nodes_out: list[str],
) -> dict[int, str]:
    """Return ``{depth: node_name}`` for the prefix of ``record_chain`` that
    actually exists in ``tax``. Stops at the first node that doesn't exist
    at its expected depth and notes it in ``unknown_nodes_out``.
    """
    found: dict[int, str] = {}
    cursor: TaxonomyNode | None = tax.root
    for depth_one_indexed, node_name in enumerate(record_chain, start=1):
        if cursor is None:
            unknown_nodes_out.append(
                f"{factor_name}:{depth_one_indexed}:{node_name} (no parent)"
            )
            break
        next_node = next(
            (c for c in cursor.children if c.name == node_name),
            None,
        )
        if next_node is None:
            unknown_nodes_out.append(f"{factor_name}:{depth_one_indexed}:{node_name}")
            # Don't bail \u2014 still try to record what's valid, but stop
            # descending. The deeper levels become uncountable for this
            # record.
            break
        found[depth_one_indexed] = next_node.name
        cursor = next_node
    return found


def level_ratio_coverage(
    bundle: TaxonomyBundle,
    sampled_per_record: list[SampledRecord],
) -> CoverageReport:
    """Compute per-level / per-factor coverage from a list of per-record samples.

    Returns:
        CoverageReport with `by_level`, `by_factor`, `unknown_factors`,
        `unknown_nodes`. The `by_level` ratio at depth L is the average over
        factors of the per-factor coverage at L (factors without that depth
        are skipped).
    """
    factor_index = {f.factor: f for f in bundle.factors}
    nodes_by_depth: dict[str, dict[int, set[str]]] = {
        f.factor: _nodes_by_depth(f) for f in bundle.factors
    }
    covered: dict[str, dict[int, set[str]]] = {
        f.factor: {depth: set() for depth in nodes_by_depth[f.factor]}
        for f in bundle.factors
    }
    unknown_factors: set[str] = set()
    unknown_nodes: list[str] = []

    for record in sampled_per_record:
        if not record:
            continue
        for factor_name, chain in record.items():
            if factor_name not in factor_index:
                unknown_factors.add(factor_name)
                continue
            tax = factor_index[factor_name]
            found = _record_paths_against_taxonomy(
                tax, chain, factor_name, unknown_nodes
            )
            for depth, node_name in found.items():
                covered[factor_name][depth].add(node_name)

    # Per-factor coverage map.
    by_factor: dict[str, dict[int, float]] = {}
    for factor_name, depth_to_nodes in nodes_by_depth.items():
        by_factor[factor_name] = {}
        for depth, total_nodes in depth_to_nodes.items():
            if not total_nodes:
                by_factor[factor_name][depth] = 0.0
            else:
                by_factor[factor_name][depth] = len(covered[factor_name][depth]) / len(
                    total_nodes
                )

    # Aggregate by level: average coverage across factors that have that depth.
    all_depths: set[int] = set()
    for d_map in nodes_by_depth.values():
        all_depths.update(d_map.keys())

    by_level: dict[int, float] = {}
    for depth in sorted(all_depths):
        contributors = [
            by_factor[f.factor][depth]
            for f in bundle.factors
            if depth in nodes_by_depth[f.factor]
        ]
        by_level[depth] = sum(contributors) / len(contributors) if contributors else 0.0

    return CoverageReport(
        by_level=by_level,
        by_factor=by_factor,
        unknown_factors=unknown_factors,
        unknown_nodes=unknown_nodes,
    )


def extract_sampled_from_record(record: Any) -> SampledRecord:
    """Helper for the eval harness: pull taxonomy_nodes out of a Record's scores.

    Returns ``None`` for records without taxonomy info. Defensive about both
    Pydantic models and raw parquet/json dicts.
    """
    if record is None:
        return None
    scores = getattr(record, "scores", None)
    if scores is None and isinstance(record, dict):
        scores = record.get("scores")
    if scores is None:
        return None
    qpd = getattr(scores, "quality_per_dim", None)
    if qpd is None and isinstance(scores, dict):
        qpd = scores.get("quality_per_dim")
    if not qpd:
        return None
    nodes = qpd.get("taxonomy_nodes") if isinstance(qpd, dict) else None
    return nodes if isinstance(nodes, dict) else None
