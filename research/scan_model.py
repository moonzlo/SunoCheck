"""Learned head over the speed-robust coherent comb scan (scan_prepare.py v4 cache).

Per window features of the scanned base: median line contrast per band, max subdivision contrast, line
density quantiles, share of strong lines. Grouped 5-fold CV by item group; humans from all variants form
the negatives (thresholds at pooled human FPR). Also reports adjacent-window pairs that must agree on the base.

Usage: uv run python research/scan_model.py <seconds> <variant[,...]>
"""

import sys
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from coherent_scan import LOW_HZ, MAX_HARMONIC
from corpus import list_items
from lab import auroc
from scan_prepare import scan_path

BAND_EDGES_HZ: tuple[float, ...] = (0.0, 1000.0, 2000.0, 4000.0, 8000.0, 12000.0, 15800.0)
STRONG_DB: float = 6.0
FOLDS: int = 5
SEED: int = 0
PAIR_TOLERANCE_HZ: float = 0.05
INCONSISTENT_PENALTY: float = 50.0


@dataclass(frozen=True)
class Table:
    x: NDArray[np.float64]
    label: NDArray[np.int8]
    group: NDArray[np.str_]
    item: NDArray[np.str_]
    variant: NDArray[np.str_]
    base: NDArray[np.float64]


def band_medians(lines: NDArray[np.float64], base: float) -> list[float]:
    freqs = base * np.arange(1, MAX_HARMONIC + 1)
    out: list[float] = []
    for low, high in zip(BAND_EDGES_HZ, BAND_EDGES_HZ[1:]):
        values = lines[(freqs >= low) & (freqs < high) & ~np.isnan(lines)]
        out.append(float(np.median(values)) if values.size else 0.0)
    return out


def window_vector(data: np.lib.npyio.NpzFile, tag: str, w: int) -> list[float]:
    base = float(data[f"base_{tag}"][w])
    lines = data[f"lines_{tag}"][w]
    in_band = lines[(base * np.arange(1, MAX_HARMONIC + 1) >= LOW_HZ) & ~np.isnan(lines)]
    return [float(data[f"plain_{tag}"][w]), float(data[f"sub_{tag}"][w].max()), *data[f"quantiles_{tag}"][w].tolist(),
            *band_medians(lines, base), float(np.mean(in_band > STRONG_DB))]


def load(seconds: int, variants: list[str]) -> Table:
    tag = f"{seconds}s"
    rows: list[list[float]] = []
    meta: list[tuple[int, str, str, str, float]] = []
    for item in list_items():
        for variant in variants:
            path = scan_path(variant, item.item_id)
            if not path.exists():
                continue
            data = np.load(path)
            for w in range(data[f"base_{tag}"].size):
                rows.append(window_vector(data, tag, w))
                meta.append((int(item.label == "suno"), item.group, item.item_id, variant, float(data[f"base_{tag}"][w])))
    return Table(np.array(rows), np.array([m[0] for m in meta], dtype=np.int8), np.array([m[1] for m in meta]),
                 np.array([m[2] for m in meta]), np.array([m[3] for m in meta]), np.array([m[4] for m in meta]))


def cv_scores(table: Table, kind: str) -> NDArray[np.float64]:
    groups = np.unique(table.group)
    rng = np.random.default_rng(SEED)
    fold_of = dict(zip(rng.permutation(groups), np.arange(groups.size) % FOLDS))
    folds = np.array([fold_of[g] for g in table.group])
    scores = np.zeros(table.label.size)
    for fold in range(FOLDS):
        train, test = folds != fold, folds == fold
        if kind == "lr":
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
            model.fit(table.x[train], table.label[train])
            scores[test] = model.decision_function(table.x[test])
        else:
            model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=SEED)
            model.fit(table.x[train], table.label[train])
            probability = np.clip(model.predict_proba(table.x[test])[:, 1], 1e-9, 1 - 1e-9)
            scores[test] = np.log(probability / (1 - probability))
    return scores


def pairs(table: Table, scores: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.int8], NDArray[np.str_]]:
    out_scores: list[float] = []
    out_labels: list[int] = []
    out_variants: list[str] = []
    for i in range(scores.size - 1):
        if table.item[i] != table.item[i + 1] or table.variant[i] != table.variant[i + 1]:
            continue
        low = min(scores[i], scores[i + 1])
        consistent = abs(table.base[i] - table.base[i + 1]) < PAIR_TOLERANCE_HZ
        out_scores.append(low if consistent else low - INCONSISTENT_PENALTY)
        out_labels.append(int(table.label[i]))
        out_variants.append(str(table.variant[i]))
    return np.array(out_scores), np.array(out_labels, dtype=np.int8), np.array(out_variants)


def report(name: str, scores: NDArray[np.float64], labels: NDArray[np.int8], variant_of: NDArray[np.str_], variants: list[str]) -> None:
    human = scores[labels == 0]
    t1, t01 = np.quantile(human, 0.99), np.quantile(human, 0.999)
    line = f"  {name:<12} human n={human.size:<5} |"
    for variant in variants:
        suno = scores[(labels == 1) & (variant_of == variant)]
        if suno.size == 0:
            continue
        both = np.concatenate([suno, human])
        both_labels = np.concatenate([np.ones(suno.size), np.zeros(human.size)]).astype(np.int8)
        line += f" {variant} {auroc(both, both_labels):.4f}/{np.mean(suno > t1):.2f}/{np.mean(suno > t01):.2f}"
    print(line)


def main(seconds: int, variants: list[str]) -> None:
    table = load(seconds, variants)
    print(f"{seconds}s: {table.label.size} windows, suno {int(table.label.sum())}, human {int((table.label == 0).sum())}"
          "  (AUROC/TPR@1%/TPR@0.1% vs pooled humans)")
    baseline = table.x[:, 0] - table.x[:, 3]
    report("plain-q995", baseline, table.label, table.variant, variants)
    for kind in ("lr", "gbm"):
        scores = cv_scores(table, kind)
        report(f"{kind} window", scores, table.label, table.variant, variants)
        pair_scores, pair_labels, pair_variants = pairs(table, scores)
        report(f"{kind} pair", pair_scores, pair_labels, pair_variants, variants)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2].split(","))
