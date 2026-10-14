"""Exclusive coherent comb: 200 Hz lines must stand out against the 50 Hz grid points between them.

Usage: uv run python research/coherent_eval2.py <variant[,variant...]>
"""

import sys
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from coherent_prepare import GRID_HZ, LOW_HZ, NEIGHBOUR_HZ, PATCH_HZ, PLAYBACK_RATE, WINDOWS_SECONDS, coherent_path
from coherent_eval import FOLDS, SEED
from corpus import Label
from lab import all_items, summary
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_HZ: float = 200.0
OFFSET_STEPS: tuple[int, ...] = (-3, -2, -1, 1, 2, 3)


@dataclass(frozen=True)
class Rows:
    c0: NDArray[np.float32]
    off: NDArray[np.float32]
    label: NDArray[np.int8]
    group: NDArray[np.str_]
    source: NDArray[np.str_]


def grid_contrasts(patches: NDArray[np.float16], bin_hz: float) -> NDArray[np.float32]:
    """[windows, grid lines]: line peak (3 central bins) minus neighbourhood median (0.5-5 Hz away), dB."""
    p = patches.astype(np.float32)
    half = p.shape[2] // 2
    lo, hi = max(2, int(NEIGHBOUR_HZ[0] / bin_hz)), int(NEIGHBOUR_HZ[1] / bin_hz)
    line = p[:, :, half - 1 : half + 2].max(axis=2)
    neighbours = np.concatenate([p[:, :, half - hi : half - lo], p[:, :, half + lo + 1 : half + hi + 1]], axis=2)
    return line - np.median(neighbours, axis=2)


def comb_rows(contrasts: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    freqs = LOW_HZ + GRID_HZ * np.arange(contrasts.shape[1])
    on_comb = np.flatnonzero(np.isclose(np.mod(freqs, BASE_HZ), 0.0))
    on_comb = on_comb[(on_comb + min(OFFSET_STEPS) >= 0) & (on_comb + max(OFFSET_STEPS) < contrasts.shape[1])]
    c0 = contrasts[:, on_comb]
    off = np.mean(np.stack([contrasts[:, on_comb + s] for s in OFFSET_STEPS]), axis=0)
    return c0, off


def load_rows(variant: str, seconds: float) -> Rows:
    tag = f"{int(seconds)}s"
    bin_hz = PLAYBACK_RATE / int(seconds * PLAYBACK_RATE)
    c0s, offs, label, group, source = [], [], [], [], []
    for item in all_items():
        path = coherent_path(variant, item.item_id)
        if not path.exists():
            continue
        patches = np.load(path)[f"patches_{tag}"]
        if len(patches) == 0:
            continue
        c0, off = comb_rows(grid_contrasts(patches, bin_hz))
        c0s.append(c0)
        offs.append(off)
        label += [int(item.label == Label.SUNO)] * len(c0)
        group += [item.group] * len(c0)
        source += [item.source] * len(c0)
    return Rows(np.concatenate(c0s), np.concatenate(offs), np.array(label, dtype=np.int8), np.array(group), np.array(source))


def cv_scores(features: NDArray[np.float32], rows: Rows) -> NDArray[np.float64]:
    groups = np.unique(rows.group)
    rng = np.random.default_rng(SEED)
    fold_of_group = dict(zip(rng.permutation(groups), np.arange(len(groups)) % FOLDS))
    folds = np.array([fold_of_group[g] for g in rows.group])
    scores = np.zeros(len(rows.label))
    for fold in range(FOLDS):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
        model.fit(features[folds != fold], rows.label[folds != fold])
        scores[folds == fold] = model.decision_function(features[folds == fold])
    return scores


def main(variants: list[str]) -> None:
    for variant in variants:
        for seconds in WINDOWS_SECONDS:
            rows = load_rows(variant, seconds)
            print(f"== {variant} {int(seconds)}s: suno {int(rows.label.sum())} human {int((rows.label == 0).sum())} windows")
            plain = np.median(rows.c0, axis=1)
            exclusive = np.median(rows.c0 - rows.off, axis=1)
            print(f"  plain median contrast        {summary(plain, rows.label)}")
            print(f"  exclusive median contrast    {summary(exclusive, rows.label)}  (suno med {np.median(exclusive[rows.label == 1]):.1f}, human p99 {np.quantile(exclusive[rows.label == 0], 0.99):.1f} max {exclusive[rows.label == 0].max():.1f})")
            print(f"  logreg [c0, off] (CV)        {summary(cv_scores(np.hstack([rows.c0, rows.off]), rows), rows.label)}")
            print(f"  logreg [c0 - off] (CV)       {summary(cv_scores(rows.c0 - rows.off, rows), rows.label)}")


if __name__ == "__main__":
    main(sys.argv[1].split(","))
