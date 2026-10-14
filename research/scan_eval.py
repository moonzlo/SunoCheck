"""Evaluate cached coherent comb scans (scan_prepare.py).

Scores (for the scanned base and for the fixed 200 Hz base), all = median line contrast minus a reference:
  sub   - max over subdivisions m of the median contrast at base*(k + j/m)
  dens  - 99.5% quantile of the 3-bin-max contrast over the whole band (dense loop combs)
  both  - max of the two references
Thresholds and AUROC negatives come from pooled human windows (or items) of all evaluated variants.

Usage: uv run python research/scan_eval.py <seconds> <variant[,...]>
"""

import sys
from dataclasses import dataclass

import numpy as np

from corpus import list_items
from lab import auroc
from scan_prepare import scan_path

SPEED: dict[str, float] = {"orig": 1.0, "aac_128": 1.0, "speed_up3": 1.03, "speed_down15": 0.85, "speed_up12": 1.12,
                           "speed_up25": 1.25, "slowed_hall": 0.85, "hall": 1.0, "speed_down4": 0.96}
SCORE_NAMES: tuple[str, ...] = ("fixed_sub", "fixed_dens", "fixed_both", "scan_sub", "scan_dens", "scan_both")
DENSITY_QUANTILE: int = 1  # index into coherent_scan.LINE_QUANTILES (0.995)


@dataclass(frozen=True)
class Row:
    item_id: str
    label: int
    variant: str
    base_hz: float
    scores: dict[str, float]


def load(seconds: int, variants: list[str]) -> list[Row]:
    rows: list[Row] = []
    tag = f"{seconds}s"
    for item in list_items():
        for variant in variants:
            path = scan_path(variant, item.item_id)
            if not path.exists():
                continue
            data = np.load(path)
            if data[f"base_{tag}"].size == 0:
                continue
            density = data[f"quantiles_{tag}"][:, DENSITY_QUANTILE]
            for window in range(data[f"base_{tag}"].size):
                scores: dict[str, float] = {}
                for prefix, key in (("fixed", "fixed_"), ("scan", "")):
                    plain = float(data[f"{key}plain_{tag}"][window])
                    sub = float(data[f"{key}sub_{tag}"][window].max())
                    scores[f"{prefix}_sub"] = plain - sub
                    scores[f"{prefix}_dens"] = plain - float(density[window])
                    scores[f"{prefix}_both"] = plain - max(sub, float(density[window]))
                rows.append(Row(item.item_id, int(item.label == "suno"), variant,
                                float(data[f"base_{tag}"][window]), scores))
    return rows


def scores_of(subset: list[Row], score_name: str, per_item: bool) -> tuple[np.ndarray, np.ndarray]:
    if not per_item:
        return np.array([r.scores[score_name] for r in subset]), np.array([r.label for r in subset])
    grouped: dict[tuple[str, str, int], list[float]] = {}
    for r in subset:
        grouped.setdefault((r.item_id, r.variant, r.label), []).append(r.scores[score_name])
    keys = sorted(grouped)
    return np.array([np.median(grouped[k]) for k in keys]), np.array([k[2] for k in keys])


def report(rows: list[Row], variants: list[str], score_name: str, per_item: bool) -> None:
    human_scores, _ = scores_of([r for r in rows if r.label == 0], score_name, per_item)
    t1, t01 = np.quantile(human_scores, 0.99), np.quantile(human_scores, 0.999)
    line = f"  {score_name:<10} {'item  ' if per_item else 'window'} human n={human_scores.size:<5} p99 {t1:5.2f} p99.9 {t01:5.2f} |"
    for variant in variants:
        suno, _ = scores_of([r for r in rows if r.variant == variant and r.label == 1], score_name, per_item)
        if suno.size == 0:
            continue
        scores = np.concatenate([suno, human_scores])
        labels = np.concatenate([np.ones(suno.size), np.zeros(human_scores.size)])
        line += f" {variant} {auroc(scores, labels):.3f}/{np.mean(suno > t1):.2f}/{np.mean(suno > t01):.2f}"
    print(line)


PAIR_TOLERANCE_HZ: float = 0.05
INCONSISTENT_PENALTY: float = 20.0


def pair_scores(rows: list[Row], score_name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Adjacent window pairs of one item/variant: min of the two scores, penalised if the bases disagree."""
    grouped: dict[tuple[str, str, int], list[Row]] = {}
    for r in rows:
        grouped.setdefault((r.item_id, r.variant, r.label), []).append(r)
    scores: list[float] = []
    labels: list[int] = []
    variants: list[str] = []
    for (_, variant, label), group in grouped.items():
        for first, second in zip(group, group[1:]):
            low = min(first.scores[score_name], second.scores[score_name])
            consistent = abs(first.base_hz - second.base_hz) < PAIR_TOLERANCE_HZ
            scores.append(low if consistent else low - INCONSISTENT_PENALTY)
            labels.append(label)
            variants.append(variant)
    return np.array(scores), np.array(labels), variants


def report_pairs(rows: list[Row], variants: list[str], score_name: str) -> None:
    scores, labels, pair_variants = pair_scores(rows, score_name)
    human = scores[labels == 0]
    t1, t01 = np.quantile(human, 0.99), np.quantile(human, 0.999)
    consistent = np.mean(human > -INCONSISTENT_PENALTY / 2)
    line = f"  {score_name:<10} pair   human n={human.size:<5} p99 {t1:5.2f} p99.9 {t01:5.2f} consistent {consistent:.3f} |"
    for variant in variants:
        mask = (labels == 1) & (np.array(pair_variants) == variant)
        if not mask.any():
            continue
        suno = scores[mask]
        both = np.concatenate([suno, human])
        both_labels = np.concatenate([np.ones(suno.size), np.zeros(human.size)]).astype(np.int8)
        line += f" {variant} {auroc(both, both_labels):.3f}/{np.mean(suno > t1):.2f}/{np.mean(suno > t01):.2f}"
    print(line)


def main(seconds: int, variants: list[str]) -> None:
    rows = load(seconds, variants)
    print(f"{seconds}s windows: {len(rows)}  (per variant: AUROC/TPR@1%/TPR@0.1%)")
    for variant in variants:
        suno = [r for r in rows if r.variant == variant and r.label == 1]
        if suno:
            hit = np.mean([abs(r.base_hz - 200 * SPEED[variant]) < 0.05 for r in suno])
            print(f"  base found {variant:<13} {hit:.3f} of {len(suno)} suno windows")
    for per_item in (False, True):
        for score_name in SCORE_NAMES:
            report(rows, variants, score_name, per_item)
    for score_name in ("scan_dens", "scan_both"):
        report_pairs(rows, variants, score_name)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2].split(","))
