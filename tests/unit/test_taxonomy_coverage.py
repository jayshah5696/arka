"""Slice 4 \u2014 Simula \u00a72.3 Level-Ratio Taxonomic Coverage.

For each factor F and each depth L of F's taxonomy, the level-ratio coverage
is:

    coverage(F, L) = |unique nodes at depth L hit by the dataset|
                     -------------------------------------------
                     |total nodes at depth L in the taxonomy|

Aggregated across factors we report a per-level ratio (and a per-factor
breakdown). Slice 4 ONLY ships the metric \u2014 not the assignment-of-arbitrary-
records to taxonomy nodes (that requires M3 calls and lives in slice 4.5).
We work directly off the `record.scores.quality_per_dim['taxonomy_nodes']`
audit trail that slice 3 already writes.
"""

from __future__ import annotations

from pathlib import Path

from arka.taxonomy.models import TaxonomyBundle


def _bundle(tmp_path: Path) -> TaxonomyBundle:
    p = tmp_path / "tax.yaml"
    p.write_text(
        """
version: "1"
factors:
  - name: domain
    root:
      name: domain
      children:
        - name: email
          children:
            - name: work
            - name: personal
        - name: blog
          children:
            - name: opinion
            - name: short
        - name: essay
""".strip()
    )
    return TaxonomyBundle.from_yaml(p)


def test_empty_dataset_yields_zero_coverage(tmp_path: Path) -> None:
    from arka.taxonomy.coverage import level_ratio_coverage

    bundle = _bundle(tmp_path)
    cov = level_ratio_coverage(bundle, sampled_per_record=[])
    # Two depths exist in the 'domain' taxonomy: 1 (email/blog/essay) and 2
    # (work/personal/opinion/short).
    assert cov.by_level == {1: 0.0, 2: 0.0}
    assert cov.by_factor["domain"][1] == 0.0
    assert cov.by_factor["domain"][2] == 0.0


def test_full_coverage_yields_one(tmp_path: Path) -> None:
    from arka.taxonomy.coverage import level_ratio_coverage

    bundle = _bundle(tmp_path)
    sampled = [
        {"domain": ["email", "work"]},
        {"domain": ["email", "personal"]},
        {"domain": ["blog", "opinion"]},
        {"domain": ["blog", "short"]},
        {"domain": ["essay"]},
    ]
    cov = level_ratio_coverage(bundle, sampled_per_record=sampled)
    assert cov.by_level[1] == 1.0  # email + blog + essay = 3/3
    assert cov.by_level[2] == 1.0  # 4/4 leaves at depth 2


def test_partial_coverage_reports_expected_ratios(tmp_path: Path) -> None:
    from arka.taxonomy.coverage import level_ratio_coverage

    bundle = _bundle(tmp_path)
    # Hit only email/work. That covers 1/3 at depth 1 (only email) and 1/4
    # at depth 2 (only the email->work branch).
    sampled = [{"domain": ["email", "work"]}]
    cov = level_ratio_coverage(bundle, sampled_per_record=sampled)
    assert abs(cov.by_level[1] - (1 / 3)) < 1e-9
    assert abs(cov.by_level[2] - (1 / 4)) < 1e-9


def test_handles_records_missing_taxonomy_nodes(tmp_path: Path) -> None:
    """If some records lack a taxonomy_nodes entry, treat them as
    contributing zero \u2014 do NOT crash. Common when comparing slice-3 data
    against a slice-0 baseline that has no taxonomy assignment.
    """
    from arka.taxonomy.coverage import level_ratio_coverage

    bundle = _bundle(tmp_path)
    sampled = [
        {"domain": ["email", "work"]},
        None,  # record had no taxonomy info
        {},  # record had an empty assignment dict
    ]
    cov = level_ratio_coverage(bundle, sampled_per_record=sampled)
    # One real assignment -> 1/3 at depth 1, 1/4 at depth 2.
    assert abs(cov.by_level[1] - (1 / 3)) < 1e-9
    assert abs(cov.by_level[2] - (1 / 4)) < 1e-9


def test_unknown_factor_is_ignored(tmp_path: Path) -> None:
    """Records that name a factor not in the bundle should be ignored, with a
    debug-level note. Don't crash \u2014 datasets evolve, taxonomies evolve."""
    from arka.taxonomy.coverage import level_ratio_coverage

    bundle = _bundle(tmp_path)
    sampled = [
        {"domain": ["email", "work"], "tone": ["casual"]},
    ]
    cov = level_ratio_coverage(bundle, sampled_per_record=sampled)
    # 'tone' is not in the bundle; it should appear in 'unknown_factors'.
    assert "tone" in cov.unknown_factors
    assert cov.by_factor.get("tone") is None  # not in the bundle


def test_unknown_node_in_known_factor_is_recorded_but_does_not_inflate(
    tmp_path: Path,
) -> None:
    """A path to a node that doesn't exist in the taxonomy must be reported
    as an 'unknown_node' but must NOT count toward coverage."""
    from arka.taxonomy.coverage import level_ratio_coverage

    bundle = _bundle(tmp_path)
    sampled = [
        {"domain": ["email", "work"]},
        {"domain": ["email", "doesnt_exist"]},  # email is real, leaf is fake
    ]
    cov = level_ratio_coverage(bundle, sampled_per_record=sampled)
    # Depth 1: only 'email' covered -> 1/3.
    # Depth 2: only 'work' is a real node we hit -> 1/4.
    assert abs(cov.by_level[1] - (1 / 3)) < 1e-9
    assert abs(cov.by_level[2] - (1 / 4)) < 1e-9
    assert cov.unknown_nodes  # at least one entry recorded
