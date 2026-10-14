"""Warp-aware fakeprint classifier.

Feature: lofcz-style fakeprint of the 1-8 kHz band read in canonical frequency (spectrum sampled at r*f),
dilated by ±DILATE bins so a slightly-off warp still lines up. Training: Suno at its known speed ratio,
humans at random ratios (they must stay human under any warp). Inference: max logit over a ratio grid.

Usage: uv run python research/warp_model.py <window_seconds>
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from sklearn.linear_model import LogisticRegression

from sunocheck.fakeprint import DEFAULT_MODEL_PATH, load_model
from corpus import BLOCK_SECONDS, RESEARCH
from lab import FREQS, FloatArray, Windows, all_items, auroc, band_bins, load_windows
from warp import warp_ratios

BAND: NDArray[np.intp] = band_bins(1000.0, 8000.0)
BAND_HZ: FloatArray = FREQS[BAND]
DILATE: int = 5
FOLDS: int = 5
SEED: int = 0
HUMAN_TRAIN_WARPS: int = 4
HUMAN_WARP_VARIANTS: tuple[str, ...] = ("orig", "aac_128")
RATIO_LOW: float = 0.80
RATIO_HIGH: float = 1.30
SCORES_PATH: Path = RESEARCH / "cache" / "warp_scores.npz"
COARSE_GRID: FloatArray = warp_ratios(RATIO_LOW, RATIO_HIGH, 0.004)
REFINE_STEPS: FloatArray = np.arange(-3, 4) * 0.001
REFINE_TOP: int = 2
MAX_WINDOWS: int = 2

TRAIN_VARIANTS: tuple[str, ...] = ("orig", "aac_128", "opus_128", "speed_up3", "speed_down15", "speed_up25", "pitch_down2", "master", "hall")
TEST_VARIANTS: tuple[str, ...] = (
    "orig", "aac_128", "opus_128", "speed_up3", "speed_down4", "speed_down15", "speed_up12", "speed_up25",
    "pitch_up1", "pitch_down2", "tempo_95", "master", "reverb", "hall", "slowed_hall",
)
KNOWN_RATIO: dict[str, float] = {"speed_up3": 1.03, "speed_down4": 0.96, "speed_down15": 0.85, "speed_up12": 1.12, "speed_up25": 1.25}


def canonical_band(spectra_db: NDArray[np.float32], ratios: FloatArray) -> NDArray[np.float32]:
    """Band spectrum as before a speed change by `ratios` (scalar or per row; linear interpolation at r * f)."""
    positions = BAND_HZ[None, :] * np.reshape(ratios, (-1, 1)) / FREQS[1]
    lower = np.floor(positions).astype(np.intp)
    frac = (positions - lower).astype(np.float32)
    rows = np.arange(spectra_db.shape[0])[:, None]
    return spectra_db[rows, lower] * (1 - frac) + spectra_db[rows, lower + 1] * frac


def feature(spectra_db: NDArray[np.float32], ratio: float | FloatArray) -> NDArray[np.float32]:
    band = canonical_band(spectra_db, np.asarray(ratio, dtype=np.float64)).astype(np.float32)
    hull = np.clip(minimum_filter1d(band, size=10, axis=1, mode="nearest"), -60.0, None)
    residue = np.clip(band - hull, 0.0, 5.0)
    residue = residue / (residue.max(axis=1, keepdims=True) + 1e-6)
    return maximum_filter1d(residue, size=2 * DILATE + 1, axis=1)


@dataclass(frozen=True)
class Scored:
    best: FloatArray
    ratio: FloatArray
    at_one: FloatArray


def score_grid(spectra_db: NDArray[np.float32], weights: FloatArray, bias: float) -> Scored:
    """Coarse grid (COARSE_STEP), then fine steps around the REFINE_TOP best coarse ratios of each row."""
    coarse = np.stack([feature(spectra_db, r) @ weights + bias for r in COARSE_GRID], axis=1)
    at_one = coarse[:, int(np.argmin(np.abs(COARSE_GRID - 1.0)))]
    tops = np.argsort(coarse, axis=1)[:, ::-1][:, :REFINE_TOP]
    best_score, best_ratio = coarse.max(axis=1), COARSE_GRID[coarse.argmax(axis=1)]
    for column in range(REFINE_TOP):
        for step in REFINE_STEPS:
            ratios = COARSE_GRID[tops[:, column]] * np.exp(step)
            scores = np.einsum("ij,j->i", feature(spectra_db, ratios), weights) + bias
            better = scores > best_score
            best_score, best_ratio = np.where(better, scores, best_score), np.where(better, ratios, best_ratio)
    return Scored(best_score, best_ratio, at_one)


def main(window_seconds: float) -> None:
    items = all_items()
    blocks = max(1, round(window_seconds / BLOCK_SECONDS))
    windows: dict[str, Windows] = {}
    for variant in TEST_VARIANTS:
        w = load_windows(items, variant, blocks, MAX_WINDOWS)
        windows[variant] = w
        print(f"{variant:<13} suno {int(w.label.sum()):4d} human {int((w.label == 0).sum()):4d}", flush=True)
    groups = np.unique(np.concatenate([w.group for w in windows.values()]))
    rng = np.random.default_rng(SEED)
    fold_of_group = dict(zip(rng.permutation(groups), np.arange(len(groups)) % FOLDS))
    best = {v: np.zeros(len(w.label)) for v, w in windows.items()}
    ratio = {v: np.zeros(len(w.label)) for v, w in windows.items()}
    at_one = {v: np.zeros(len(w.label)) for v, w in windows.items()}
    for fold in range(FOLDS):
        xs: list[NDArray[np.float32]] = []
        ys: list[NDArray[np.int8]] = []
        for variant in TRAIN_VARIANTS:
            w = windows[variant]
            train = np.array([fold_of_group[g] != fold for g in w.group])
            suno = train & (w.label == 1)
            human = train & (w.label == 0)
            xs.append(feature(w.mid[suno], KNOWN_RATIO.get(variant, 1.0)))
            ys.append(np.ones(int(suno.sum()), dtype=np.int8))
            human_rows = w.mid[human]
            warps = HUMAN_TRAIN_WARPS if variant in HUMAN_WARP_VARIANTS else 0
            for r in rng.uniform(RATIO_LOW, RATIO_HIGH, warps):
                xs.append(feature(human_rows, float(r)))
                ys.append(np.zeros(len(human_rows), dtype=np.int8))
            xs.append(feature(human_rows, 1.0))
            ys.append(np.zeros(len(human_rows), dtype=np.int8))
        x, y = np.concatenate(xs), np.concatenate(ys)
        model = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced")
        model.fit(x, y)
        weights, bias = model.coef_[0].astype(np.float32), float(model.intercept_[0])
        for variant, w in windows.items():
            test = np.array([fold_of_group[g] == fold for g in w.group])
            scored = score_grid(w.mid[test], weights, bias)
            best[variant][test], ratio[variant][test], at_one[variant][test] = scored.best, scored.ratio, scored.at_one
        print(f"fold {fold} trained on {len(y)} rows ({int(y.sum())} suno)", flush=True)
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(SCORES_PATH, **{
        f"{variant}__{key}": value
        for variant, w in windows.items()
        for key, value in (("item", w.item_id), ("label", w.label), ("start", w.start_block),
                           ("best", best[variant]), ("ratio", ratio[variant]), ("at_one", at_one[variant]))
    })
    lofcz = load_model(DEFAULT_MODEL_PATH)
    human_pool_best = np.concatenate([best[v][w.label == 0] for v, w in windows.items()])
    human_pool_one = np.concatenate([at_one[v][w.label == 0] for v, w in windows.items()])
    print(f"--- warp-aware LR vs pooled humans (n={human_pool_best.size}): AUROC/TPR@1%/TPR@0.1% at ratio 1.0 | max over grid")

    def pooled(suno_scores: FloatArray, human_scores: FloatArray) -> str:
        labels = np.concatenate([np.ones(suno_scores.size), np.zeros(human_scores.size)]).astype(np.int8)
        t1, t01 = np.quantile(human_scores, 0.99), np.quantile(human_scores, 0.999)
        return (f"{auroc(np.concatenate([suno_scores, human_scores]), labels):.4f}/"
                f"{np.mean(suno_scores > t1):.3f}/{np.mean(suno_scores > t01):.3f}")

    for variant, w in windows.items():
        suno = w.label == 1
        known = KNOWN_RATIO.get(variant, 1.0)
        ratio_ok = np.mean(np.abs(ratio[variant][suno] / known - 1.0) < 0.005)
        print(f"  {variant:<13} n {int(suno.sum()):4d}  r=1: {pooled(at_one[variant][suno], human_pool_one)}"
              f"  max: {pooled(best[variant][suno], human_pool_best)}  ratio found {ratio_ok:.2f}")
    print(f"lofcz weights loaded for reference: {lofcz.weights.size} bins")


if __name__ == "__main__":
    main(float(sys.argv[1]))
