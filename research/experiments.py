"""Hypothesis experiments on cached spectra.

Usage: uv run python research/experiments.py <experiment> <window_seconds>
Experiments: baseline, bands, side, crystal, hf
"""

import sys
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sunocheck.fakeprint import DEFAULT_MODEL_PATH, load_model
from comb import comb_contrast
from corpus import AUGMENT_NAMES, BLOCK_SECONDS, N_FFT, PLAYBACK_RATE, VARIANT_NAMES
from lab import FloatArray, Windows, all_items, band_bins, fakeprint, load_windows, lofcz_fakeprint, summary
from robust import comb_scan, peak_stats, scan_bases
from warp import lofcz_warp_search, warp_ratios

MAX_WINDOWS: int = 3
FOLDS: int = 5
SEED: int = 0

FeatureFn = Callable[[Windows], NDArray[np.float32]]


@dataclass(frozen=True)
class Dataset:
    features: dict[str, NDArray[np.float32]]
    windows: dict[str, Windows]


def build(window_seconds: float, feature_fns: dict[str, FeatureFn]) -> Dataset:
    return build_for(VARIANT_NAMES, window_seconds, feature_fns)


def build_for(variants: tuple[str, ...], window_seconds: float, feature_fns: dict[str, FeatureFn]) -> Dataset:
    items = all_items()
    blocks = max(1, round(window_seconds / BLOCK_SECONDS))
    features: dict[str, NDArray[np.float32]] = {}
    windows: dict[str, Windows] = {}
    for variant in variants:
        w = load_windows(items, variant, blocks, MAX_WINDOWS)
        for name, fn in feature_fns.items():
            features[f"{variant}|{name}"] = fn(w)
        windows[variant] = Windows(w.mid[:, :0], w.side[:, :0], w.native[:, :0], w.label, w.group, w.item_id, w.source, w.version, w.start_block)
    for variant, w in windows.items():
        print(f"window {window_seconds}s ({blocks} blocks) {variant}: suno {int(w.label.sum())}  human {int((w.label == 0).sum())}")
    return Dataset(features, windows)


def cross_validated_scores(data: Dataset, feature: str, c_value: float) -> dict[str, FloatArray]:
    return cross_validated_scores_from(data, feature, c_value, tuple(data.windows))


def cross_validated_scores_from(data: Dataset, feature: str, c_value: float, train_variants: tuple[str, ...]) -> dict[str, FloatArray]:
    """Train on `train_variants` of training groups, score every variant of held-out groups."""
    groups = np.unique(np.concatenate([w.group for w in data.windows.values()]))
    rng = np.random.default_rng(SEED)
    fold_of_group = dict(zip(rng.permutation(groups), np.arange(len(groups)) % FOLDS))
    scores = {v: np.zeros(len(w.label)) for v, w in data.windows.items()}
    train_windows = {v: data.windows[v] for v in train_variants}
    for fold in range(FOLDS):
        train_x = np.concatenate([data.features[f"{v}|{feature}"][np.array([fold_of_group[g] != fold for g in w.group])] for v, w in train_windows.items()])
        train_y = np.concatenate([w.label[np.array([fold_of_group[g] != fold for g in w.group])] for w in train_windows.values()])
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c_value, max_iter=3000, class_weight="balanced"))
        model.fit(train_x, train_y)
        for v, w in data.windows.items():
            test = np.array([fold_of_group[g] == fold for g in w.group])
            scores[v][test] = model.decision_function(data.features[f"{v}|{feature}"][test])
    return scores


def report(title: str, data: Dataset, scores: dict[str, FloatArray]) -> None:
    print(f"--- {title}")
    for v, w in data.windows.items():
        print(f"  {v:<11} {summary(scores[v], w.label)}")
    pooled_scores = np.concatenate(list(scores.values()))
    pooled_labels = np.concatenate([w.label for w in data.windows.values()])
    print(f"  {'pooled':<11} {summary(pooled_scores, pooled_labels)}")


def exp_baseline(window_seconds: float) -> None:
    model = load_model(DEFAULT_MODEL_PATH)
    data = build(window_seconds, {"lofcz": lambda w: lofcz_fakeprint(w.mid)})
    fixed = {v: data.features[f"{v}|lofcz"] @ model.weights + model.bias for v in data.windows}
    report("lofcz weights as-is (1-8 kHz)", data, fixed)
    report("logreg retrained on lofcz fakeprint", data, cross_validated_scores(data, "lofcz", 1.0))


BANDS_HZ: dict[str, tuple[float, float]] = {
    "1-8k": (1000.0, 8000.0),
    "0.5-4k": (500.0, 4000.0),
    "4-8k": (4000.0, 8000.0),
    "8-12k": (8000.0, 12000.0),
    "12-16k": (12000.0, 15900.0),
    "0.5-16k": (500.0, 15900.0),
}


def exp_bands(window_seconds: float) -> None:
    fns: dict[str, FeatureFn] = {
        name: (lambda w, lo=lo, hi=hi: fakeprint(w.mid, band_bins(lo, hi), 10, -60.0, 5.0)) for name, (lo, hi) in BANDS_HZ.items()
    }
    data = build(window_seconds, fns)
    for name in BANDS_HZ:
        report(f"logreg on fakeprint {name}", data, cross_validated_scores(data, name, 1.0))


def exp_side(window_seconds: float) -> None:
    bins = band_bins(1000.0, 8000.0)
    fns: dict[str, FeatureFn] = {
        "mid": lambda w: fakeprint(w.mid, bins, 10, -60.0, 5.0),
        "side": lambda w: fakeprint(w.side, bins, 10, -60.0, 5.0),
        "both": lambda w: np.hstack([fakeprint(w.mid, bins, 10, -60.0, 5.0), fakeprint(w.side, bins, 10, -60.0, 5.0)]),
    }
    data = build(window_seconds, fns)
    for name in fns:
        report(f"logreg on {name} fakeprint 1-8k", data, cross_validated_scores(data, name, 1.0))


def comb_vector(spectra_db: NDArray[np.float32]) -> NDArray[np.float32]:
    return comb_contrast(spectra_db, 200.0, 1000.0, 15800.0, 1, 0.5).astype(np.float32)


def exp_comb(window_seconds: float) -> None:
    fns: dict[str, FeatureFn] = {
        "comb_vec": lambda w: comb_vector(w.mid),
        "comb_vec_side": lambda w: np.hstack([comb_vector(w.mid), comb_vector(w.side)]),
    }
    data = build(window_seconds, fns)
    report("comb mean contrast, no training", data, {v: data.features[f"{v}|comb_vec"].mean(axis=1) for v in data.windows})
    report("logreg on comb_vec (75 harmonics of 200 Hz)", data, cross_validated_scores(data, "comb_vec", 0.1))
    report("logreg on comb_vec mid+side", data, cross_validated_scores(data, "comb_vec_side", 0.1))


def exp_fusion(window_seconds: float) -> None:
    model = load_model(DEFAULT_MODEL_PATH)
    bins = band_bins(1000.0, 8000.0)
    fns: dict[str, FeatureFn] = {
        "lofcz": lambda w: lofcz_fakeprint(w.mid),
        "fp": lambda w: fakeprint(w.mid, bins, 10, -60.0, 5.0),
        "comb_vec": lambda w: comb_vector(w.mid),
    }
    data = build(window_seconds, fns)
    lofcz = {v: data.features[f"{v}|lofcz"] @ model.weights + model.bias for v in data.windows}
    fp = cross_validated_scores(data, "fp", 1.0)
    comb = cross_validated_scores(data, "comb_vec", 0.1)
    report("A: lofcz as-is", data, lofcz)
    report("B: fakeprint 1-8k retrained", data, fp)
    report("C: comb_vec retrained", data, comb)
    report("B + C (sum of logits)", data, {v: fp[v] + comb[v] for v in data.windows})
    report("A + C (sum of logits)", data, {v: lofcz[v] + comb[v] for v in data.windows})
    for v, w in data.windows.items():
        corr = np.corrcoef(fp[v][w.label == 1], comb[v][w.label == 1])[0, 1]
        print(f"  corr(B, C) on suno windows, {v}: {corr:.2f}")


HF_LINES_HZ: tuple[float, ...] = (19450.0, 19550.0, 20950.0, 21050.0)
NATIVE_BIN_HZ: float = PLAYBACK_RATE / N_FFT


def hf_line_contrast(native_db: NDArray[np.float32]) -> NDArray[np.float32]:
    columns = []
    for hz in HF_LINES_HZ:
        center = int(round(hz / NATIVE_BIN_HZ))
        span = int(round(40.0 / NATIVE_BIN_HZ))
        peak = native_db[:, center - 2 : center + 3].max(axis=1)
        floor = np.median(native_db[:, center - span : center + span + 1], axis=1)
        columns.append(peak - floor)
    return np.stack(columns, axis=1).astype(np.float32)


def exp_hf(window_seconds: float) -> None:
    data = build(window_seconds, {"hf": lambda w: hf_line_contrast(w.native)})
    report("HF lines 19.45/19.55/20.95/21.05 kHz: mean contrast (whole excerpt)", data, {v: data.features[f"{v}|hf"].mean(axis=1) for v in data.windows})


ROBUST_TRAIN: tuple[str, ...] = ("orig", "aac_128", "opus_128")
ROBUST_TRAIN_AUG: tuple[str, ...] = ROBUST_TRAIN + ("speed_up3", "pitch_down2", "master")
ROBUST_VARIANTS: tuple[str, ...] = ROBUST_TRAIN + AUGMENT_NAMES
SCAN_BASES: FloatArray = scan_bases(170.0, 235.0)


def exp_robust(window_seconds: float) -> None:
    model = load_model(DEFAULT_MODEL_PATH)
    bins = band_bins(1000.0, 8000.0)

    def scan(w: Windows) -> NDArray[np.float32]:
        result = comb_scan(w.mid, SCAN_BASES)
        return np.stack([result.best_score, result.best_base_hz, result.score_at_200, result.best_z], axis=1).astype(np.float32)

    fns: dict[str, FeatureFn] = {
        "lofcz": lambda w: lofcz_fakeprint(w.mid),
        "fp": lambda w: fakeprint(w.mid, bins, 10, -60.0, 5.0),
        "scan": scan,
        "peaks": lambda w: peak_stats(w.mid),
    }
    data = build_for(ROBUST_VARIANTS, window_seconds, fns)
    lofcz = {v: data.features[f"{v}|lofcz"] @ model.weights + model.bias for v in data.windows}
    report("A: lofcz as-is", data, lofcz)
    report("B: fakeprint 1-8k LR, trained on codecs only", data, cross_validated_scores_from(data, "fp", 0.01, ROBUST_TRAIN))
    report("B': fakeprint 1-8k LR, trained on codecs + speed_up3/pitch_down2/master", data, cross_validated_scores_from(data, "fp", 0.01, ROBUST_TRAIN_AUG))
    report("C: comb at exactly 200 Hz, no training", data, {v: data.features[f"{v}|scan"][:, 2] for v in data.windows})
    report("D: comb scan 170-235 Hz best score, no training", data, {v: data.features[f"{v}|scan"][:, 0] for v in data.windows})
    report("D': comb scan 170-235 Hz best z-score, no training", data, {v: data.features[f"{v}|scan"][:, 3] for v in data.windows})
    report("E: peak statistics LR, trained on codecs only", data, cross_validated_scores_from(data, "peaks", 1.0, ROBUST_TRAIN))
    print("--- D: best base (Hz) found on suno windows, median [p10, p90]")
    for v, w in data.windows.items():
        base = data.features[f"{v}|scan"][w.label == 1, 1]
        print(f"  {v:<12} {np.median(base):7.2f} [{np.quantile(base, 0.1):7.2f}, {np.quantile(base, 0.9):7.2f}]")


WARP_RATIOS: FloatArray = warp_ratios(0.80, 1.30, 0.0025)


def exp_warp(window_seconds: float) -> None:
    model = load_model(DEFAULT_MODEL_PATH)

    def warp_features(w: Windows) -> NDArray[np.float32]:
        result = lofcz_warp_search(w.mid, model, WARP_RATIOS)
        return np.stack([result.best_score, result.best_ratio, result.unwarped_score], axis=1).astype(np.float32)

    data = build_for(ROBUST_VARIANTS, window_seconds, {"warp": warp_features})
    report("lofcz as-is (ratio 1.0)", data, {v: data.features[f"{v}|warp"][:, 2] for v in data.windows})
    report("lofcz warp search 0.90-1.10, max logit", data, {v: data.features[f"{v}|warp"][:, 0] for v in data.windows})
    print("--- best ratio on suno windows, median [p10, p90]; human best-ratio spread")
    for v, w in data.windows.items():
        ratio = data.features[f"{v}|warp"][:, 1]
        print(f"  {v:<12} suno {np.median(ratio[w.label == 1]):.4f} [{np.quantile(ratio[w.label == 1], 0.1):.4f}, {np.quantile(ratio[w.label == 1], 0.9):.4f}]  human p10-p90 [{np.quantile(ratio[w.label == 0], 0.1):.4f}, {np.quantile(ratio[w.label == 0], 0.9):.4f}]")


EXPERIMENTS: dict[str, Callable[[float], None]] = {
    "warp": exp_warp,
    "robust": exp_robust,
    "hf": exp_hf,
    "baseline": exp_baseline,
    "bands": exp_bands,
    "side": exp_side,
    "comb": exp_comb,
    "fusion": exp_fusion,
}

if __name__ == "__main__":
    EXPERIMENTS[sys.argv[1]](float(sys.argv[2]))
