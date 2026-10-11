"""Indexed, windowed benchmark overlap and deterministic calibration."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from arka.config.models import DecontaminateConfig
from arka.pipeline.artifacts import StageArtifacts, StageReport
from arka.pipeline.models import StageContext
from arka.pipeline.stages import Stage
from arka.records.models import ConversationRecord, DecontaminationMatch, Record

Gram = tuple[str, ...]


def _words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.casefold())


def _grams(words: list[str], n: int) -> list[Gram]:
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


@dataclass(frozen=True)
class EvalItem:
    benchmark: str
    line: int
    text: str
    word_count: int
    grams: frozenset[Gram]


class BenchmarkIndex:
    def __init__(
        self,
        paths: list[str],
        text_field: str,
        n: int,
        idf: Literal["per_benchmark", "global"],
    ) -> None:
        self.n = n
        self.items: list[EvalItem] = []
        self.short_items_skipped = 0
        names = Counter(Path(path).name for path in paths)
        if len(set(paths)) != len(paths):
            raise ValueError("eval_paths must not repeat a benchmark file")
        for path_string in paths:
            path = Path(path_string)
            benchmark = path.name if names[path.name] == 1 else str(path)
            before = len(self.items)
            with path.open(encoding="utf-8") as handle:
                for line, raw in enumerate(handle, 1):
                    if not raw.strip():
                        continue
                    try:
                        row = json.loads(raw)
                        text = row[text_field]
                        if not isinstance(text, str) or not text.strip():
                            raise ValueError(
                                "evaluation text must be a non-empty string"
                            )
                    except (ValueError, KeyError, TypeError) as exc:
                        raise ValueError(
                            f"{path}:{line}: invalid evaluation text: {exc}"
                        ) from exc
                    words = _words(text)
                    grams = frozenset(_grams(words, n))
                    if not grams:
                        self.short_items_skipped += 1
                        continue
                    self.items.append(
                        EvalItem(benchmark, line, text, len(words), grams)
                    )
            if len(self.items) == before:
                raise ValueError(f"{path}: no evaluation items contain {n} words")

        frequencies: dict[str, Counter[Gram]] = defaultdict(Counter)
        counts: Counter[str] = Counter()
        for item in self.items:
            scope = item.benchmark if idf == "per_benchmark" else "global"
            counts[scope] += 1
            frequencies[scope].update(item.grams)
        self.weights: list[dict[Gram, float]] = []
        self.denominators: list[float] = []
        self.index: dict[Gram, list[int]] = defaultdict(list)
        for i, item in enumerate(self.items):
            scope = item.benchmark if idf == "per_benchmark" else "global"
            total = counts[scope]
            weights = {
                gram: (
                    1.0
                    if total == 1
                    else math.log((total + 1) / (frequencies[scope][gram] + 1))
                )
                for gram in item.grams
            }
            self.weights.append(weights)
            self.denominators.append(sum(weights.values()))
            for gram, weight in weights.items():
                if weight > 0:
                    self.index[gram].append(i)

    def match(
        self, text: str, field: str, window_factor: int
    ) -> DecontaminationMatch | None:
        hits: dict[int, list[tuple[int, Gram]]] = defaultdict(list)
        for pos, gram in enumerate(_grams(_words(text), self.n)):
            for item_id in self.index.get(gram, ()):
                hits[item_id].append((pos, gram))
        best: DecontaminationMatch | None = None
        for item_id, positions in hits.items():
            item = self.items[item_id]
            weights = self.weights[item_id]
            width = window_factor * item.word_count
            present: Counter[Gram] = Counter()
            left = 0
            matched_weight = 0.0
            for pos, gram in positions:
                if not present[gram]:
                    matched_weight += weights[gram]
                present[gram] += 1
                while pos + self.n - positions[left][0] > width:
                    old = positions[left][1]
                    present[old] -= 1
                    if not present[old]:
                        matched_weight -= weights[old]
                    left += 1
                score = min(1.0, max(0.0, matched_weight / self.denominators[item_id]))
                if best is None or score > best.score:
                    best = DecontaminationMatch(
                        benchmark=item.benchmark,
                        eval_id=item.line,
                        score=score,
                        field=field,
                        window_start=positions[left][0],
                        window_end=pos + self.n,
                    )
        return best


class DecontaminateStage(Stage):
    name = "02k_decontaminate"
    stage_action = "filtered"

    def __init__(
        self,
        config: DecontaminateConfig | None = None,
        project_root: Path | None = None,
    ):
        self.config = config
        self.project_root = project_root or Path.cwd()

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        self.config = self.get_stage_config(ctx, DecontaminateConfig, "decontaminate")
        if self.config is None:
            return records
        cfg = self.config
        paths = [str((self.project_root / path).resolve()) for path in cfg.eval_paths]
        index = BenchmarkIndex(paths, cfg.eval_text_field, cfg.n, cfg.idf)
        kept: list[Record] = []
        dropped: list[Record] = []
        matches: Counter[str] = Counter()
        clean_scores: list[float] = []
        for record in records:
            best = None
            if isinstance(record, ConversationRecord):
                for field in dict.fromkeys(cfg.fields):
                    found = index.match(
                        getattr(record.payload, field) or "", field, cfg.window_factor
                    )
                    if found and (best is None or found.score > best.score):
                        best = found
            clean_scores.append(best.score if best else 0.0)
            if not cfg.calibrate and best and best.score >= cfg.threshold:
                matches[best.benchmark] += 1
                record = record.model_copy(
                    update={
                        "scores": record.scores.model_copy(update={"decon_match": best})
                    }
                )
                details = best.model_dump_json()
                if cfg.action == "drop":
                    dropped.append(
                        record.dropped_by(self.name, "benchmark_overlap", details)
                    )
                    continue
                record = record.with_event(
                    stage=self.name,
                    action="flagged",
                    reason_code="benchmark_overlap",
                    details=details,
                )
            kept.append(record)

        extras = (
            self._calibration(index, clean_scores, records, cfg)
            if cfg.calibrate
            else {}
        )
        StageArtifacts(ctx).write(
            report=StageReport(
                stage=self.name,
                count_in=len(records),
                count_out=len(kept),
                dropped_count=len(dropped),
                drop_reasons={"benchmark_overlap": len(dropped)} if dropped else {},
                benchmark_match_counts=dict(matches),
                flagged_count=sum(matches.values()) if cfg.action == "flag" else 0,
                indexed_items=len(index.items),
                indexed_ngrams=len(index.index),
                short_items_skipped=index.short_items_skipped,
                unscorable_items=sum(value == 0 for value in index.denominators),
                **extras,
            ),
            dropped=dropped,
        )
        return kept

    def _calibration(
        self,
        index: BenchmarkIndex,
        clean_scores: list[float],
        records: list[Record],
        cfg: DecontaminateConfig,
    ) -> dict[str, object]:
        plant_scores: list[float] = []
        synonyms = {
            "calculate": "compute",
            "total": "sum",
            "large": "big",
            "small": "little",
            "number": "quantity",
        }
        prefix = next(
            (
                r.payload.instruction
                for r in records
                if isinstance(r, ConversationRecord)
            ),
            "",
        )
        for item in index.items:
            words = _words(item.text)
            paraphrase = list(words)
            for i, word in enumerate(paraphrase):
                if word in synonyms:
                    paraphrase[i] = synonyms[word]
                    break
            for variant in (
                item.text.swapcase(),
                "  \n ".join(words),
                " ".join(paraphrase),
            ):
                match = index.match(
                    prefix + "\n" + variant, "instruction", cfg.window_factor
                )
                plant_scores.append(match.score if match else 0.0)
        curve = [
            {
                "threshold": threshold / 100,
                "recall": sum(score >= threshold / 100 for score in plant_scores)
                / len(plant_scores),
                "clean_rows_dropped": sum(
                    score >= threshold / 100 for score in clean_scores
                ),
            }
            for threshold in range(1, 101)
        ]
        candidates = [row["threshold"] for row in curve if row["recall"] >= 0.99]
        return {
            "calibration_curve": curve,
            "calibration_plant_count": len(plant_scores),
            "calibration_clean_count": len(clean_scores),
            "recommended_threshold": max(candidates) if candidates else None,
        }
