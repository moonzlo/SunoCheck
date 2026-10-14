"""Evaluate coherent comb-line features (from coherent_prepare.py).

Usage: uv run python research/coherent_eval.py <variant[,variant...]>
"""

import sys
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from coherent_prepare import WINDOWS_SECONDS, coherent_path
from corpus import Label
from lab import all_items, summary

FOLDS: int = 5
SEED: int = 0


@dataclass(frozen=True)
class Rows:
    lines: NDArray[np.float32]
    control: NDArray[np.float32]
    label: NDArray[np.int8]
    group: NDArray[np.str_]
    source: NDArray[np.str_]


def load_rows(variant: str, tag: str) -> Rows:
    lines, control, label, group, source = [], [], [], [], []
    for item in all_items():
        path = coherent_path(variant, item.item_id)
        if not path.exists():
            continue
        data = np.load(path)
        for row, ctrl in zip(data[f"lines_{tag}"], data[f"control_{tag}"]):
            lines.append(row)
            control.append(ctrl)
            label.append(int(item.label == Label.SUNO))
            group.append(item.group)
            source.append(item.source)
    return Rows(np.array(lines), np.array(control), np.array(label, dtype=np.int8), np.array(group), np.array(source))


def cv_scores(rows: Rows) -> NDArray[np.float64]:
    groups = np.unique(rows.group)
    rng = np.random.default_rng(SEED)
    fold_of_group = dict(zip(rng.permutation(groups), np.arange(len(groups)) % FOLDS))
    folds = np.array([fold_of_group[g] for g in rows.group])
    scores = np.zeros(len(rows.label))
    for fold in range(FOLDS):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
        model.fit(rows.lines[folds != fold], rows.label[folds != fold])
        scores[folds == fold] = model.decision_function(rows.lines[folds == fold])
    return scores


def main(variants: list[str]) -> None:
    for variant in variants:
        for seconds in WINDOWS_SECONDS:
            tag = f"{int(seconds)}s"
            rows = load_rows(variant, tag)
            print(f"== {variant} {tag}: suno {int(rows.label.sum())} human {int((rows.label == 0).sum())} windows")
            median = np.median(rows.lines, axis=1)
            mean = rows.lines.mean(axis=1)
            control = np.median(rows.control, axis=1)
            print(f"  median line contrast       {summary(median, rows.label)}  (suno med {np.median(median[rows.label == 1]):.1f} dB, human med {np.median(median[rows.label == 0]):.1f} p99 {np.quantile(median[rows.label == 0], 0.99):.1f})")
            print(f"  mean line contrast         {summary(mean, rows.label)}")
            print(f"  median - control median    {summary(median - control, rows.label)}")
            print(f"  control base 203 Hz        {summary(control, rows.label)}")
            print(f"  logreg on 75 lines (CV)    {summary(cv_scores(rows), rows.label)}")
            for source in np.unique(rows.source[rows.label == 0]):
                human = (rows.label == 0) & (rows.source == source)
                print(f"    human {source:<8} n={int(human.sum()):4d} median-contrast p50 {np.median(median[human]):.1f} p99 {np.quantile(median[human], 0.99):.1f} max {median[human].max():.1f}")


if __name__ == "__main__":
    main(sys.argv[1].split(","))
