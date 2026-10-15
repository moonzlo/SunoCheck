"""H21: semitone lock of the shifted-Suno model on the live stream path.

A pitch-shifted Suno track lines its smeared artifact peaks up with the shifted model weights only at the exact
undo ratio r = 2^(-n/12); a ratio 0.2-2 % off misaligns them (H17 showed the lock with the lofcz model). Humans
score randomly along r. Contrast = max score within CORE_STEPS of the exact ratio minus max score at the
neighbor offsets. Replays each recording every 0.5 s from min_seconds, like the listener.

Corpus mode: the same contrast on cached windows, each scored by a shifted model trained without its track
(GroupKFold over the app's training set: in-sample human scores were pushed down exactly at the undo ratios).

Usage:
  uv run python research/shift_lock.py files <wav> [...]
  uv run python research/shift_lock.py corpus
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from corpus import BLOCK_SECONDS, Item, Label, list_items
from lab import load_windows, lofcz_fakeprint
from numpy.typing import NDArray
from shifted_model import SUNO_SHIFTS, dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from warp import warped

from sunocheck.fakeprint import (
    MODEL_SAMPLE_RATE,
    SHIFT_SEMITONES,
    SHIFTED_MODEL_PATH,
    STREAM_HOP,
    empty_stream,
    fakeprint_from_band,
    load_model,
    push_samples,
    resample_to_model_rate,
    warped_bands,
)
from sunocheck.settings import DEFAULT_SETTINGS

RULES: tuple[tuple[float, float], ...] = (
    (8.0, -np.inf),
    (-np.inf, 6.0),
    (-np.inf, 7.0),
    (0.0, 4.0),
    (2.0, 4.0),
    (2.0, 5.0),
    (4.0, 3.0),
    (4.0, 4.0),
)
STEP_SECONDS: float = 0.5
RATIO_STEP: float = 0.0002
CORE_STEPS: NDArray[np.int64] = np.arange(-2, 3)
NEIGHBOR_STEPS: NDArray[np.int64] = np.concatenate([-np.arange(10, 101, 5)[::-1], np.arange(10, 101, 5)])


def ratio_grid() -> NDArray[np.float64]:
    """(semitones, core + neighbor steps) undo ratios."""
    steps: NDArray[np.int64] = np.concatenate([CORE_STEPS, NEIGHBOR_STEPS])
    exact: NDArray[np.float64] = 2.0 ** (-np.array(SHIFT_SEMITONES, dtype=np.float64) / 12)
    return np.outer(exact, np.exp(steps * RATIO_STEP))


def peak_and_contrast(scores: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """scores (rows, semitones, core + neighbor steps) -> per row: best core score, contrast of that semitone."""
    core: int = CORE_STEPS.size
    core_best: NDArray[np.float64] = scores[:, :, :core].max(axis=2)
    contrast: NDArray[np.float64] = core_best - scores[:, :, core:].max(axis=2)
    best: NDArray[np.intp] = np.argmax(core_best, axis=1)
    rows: NDArray[np.intp] = np.arange(scores.shape[0])
    return core_best[rows, best], contrast[rows, best]


def lock_trajectory(mono: NDArray[np.float32]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Per step: (seconds, peak = best core score over semitones, contrast of that semitone)."""
    model = load_model(SHIFTED_MODEL_PATH)
    grid: NDArray[np.float64] = ratio_grid()
    max_frames: int = int(DEFAULT_SETTINGS.max_seconds * MODEL_SAMPLE_RATE) // STREAM_HOP
    step: int = int(STEP_SECONDS * MODEL_SAMPLE_RATE)
    stream = empty_stream()
    seconds: list[float] = []
    steps: list[NDArray[np.float64]] = []
    for start in range(0, mono.size - step + 1, step):
        stream = push_samples(stream, mono[start : start + step], max_frames)
        now: float = (start + step) / MODEL_SAMPLE_RATE
        if not stream.frames or now < DEFAULT_SETTINGS.min_seconds:
            continue
        mean_db: NDArray[np.float64] = np.mean(stream.frames, axis=0, dtype=np.float64)
        steps.append((fakeprint_from_band(warped_bands(mean_db, grid.ravel())) @ model.weights + model.bias).reshape(grid.shape))
        seconds.append(now)
    peaks, contrasts = peak_and_contrast(np.stack(steps))
    return np.array(seconds), peaks, contrasts


def fold_models(
    x: NDArray[np.float32], y: NDArray[np.int8], groups: NDArray[np.str_]
) -> tuple[dict[str, int], NDArray[np.float64], NDArray[np.float64]]:
    """(item -> fold, weights (folds, bins), biases (folds,)) of the app's shifted model setup (C = 1)."""
    fold_of: dict[str, int] = {}
    weights: list[NDArray[np.float64]] = []
    biases: list[float] = []
    for fold, (train, test) in enumerate(GroupKFold(n_splits=5).split(x, y, groups)):
        model = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(x[train], y[train])
        weights.append(model.coef_[0].astype(np.float64))
        biases.append(float(model.intercept_[0]))
        fold_of.update(dict.fromkeys(np.unique(groups[test]).tolist(), fold))
        print(f"  fold {fold} fitted", flush=True)
    return fold_of, np.stack(weights), np.array(biases)


def corpus_scores(
    mids: NDArray[np.float32], folds: NDArray[np.intp], weights: NDArray[np.float64], biases: NDArray[np.float64]
) -> NDArray[np.float64]:
    """(windows, semitones, steps) out-of-fold shifted scores over the ratio grid."""
    grid: NDArray[np.float64] = ratio_grid()
    rows: NDArray[np.intp] = np.arange(len(mids))
    out: NDArray[np.float64] = np.empty((len(mids), *grid.shape))
    for (semitone, step), ratio in np.ndenumerate(grid):
        out[:, semitone, step] = (lofcz_fakeprint(warped(mids, float(ratio))) @ weights.T)[rows, folds] + biases[folds]
    return out


def rule_table(name: str, peaks: NDArray[np.float64], contrasts: NDArray[np.float64]) -> str:
    cells: list[str] = [f"{np.mean((peaks >= p) & (contrasts >= c)):.3f}" for p, c in RULES]
    return f"{name:<26} n={peaks.size:<5} peak max {peaks.max():+6.2f} contrast max {contrasts.max():+6.2f} | " + " ".join(cells)


def corpus() -> None:
    x, y, groups, _, _ = dataset(SUNO_SHIFTS, ((8, 2),))
    fold_of, weights, biases = fold_models(x, y, groups)
    print("rules (peak >= P and contrast >= C): " + " ".join(f"P{p:g}/C{c:g}" for p, c in RULES))
    humans: list[Item] = [i for i in list_items() if i.label == Label.HUMAN]
    suno: list[Item] = [i for i in list_items() if i.label == Label.SUNO]
    for blocks in (4, 8):
        for name, items, variant in (
            ("human orig", humans, "orig"),
            ("human aac_128", humans, "aac_128"),
            ("suno wsola_down1", suno, "wsola_down1"),
            ("suno pitch_down2", suno, "pitch_down2"),
            ("suno orig", suno, "orig"),
        ):
            w = load_windows(items, variant, blocks, 2)
            known: NDArray[np.bool_] = np.array([item in fold_of for item in w.item_id.tolist()])
            folds: NDArray[np.intp] = np.array([fold_of[item] for item in w.item_id[known].tolist()], dtype=np.intp)
            peaks, contrasts = peak_and_contrast(corpus_scores(w.mid[known], folds, weights, biases))
            print(rule_table(f"{blocks * BLOCK_SECONDS:g} s {name}", peaks, contrasts), flush=True)


def detection_second(seconds: NDArray[np.float64], hits: NDArray[np.bool_]) -> float | None:
    """First second at which hits held for confirm_seconds, as decision.update_evidence does."""
    since: float | None = None
    for now, hit in zip(seconds, hits, strict=True):
        if not hit:
            since = None
            continue
        since = now if since is None else since
        if now - since >= DEFAULT_SETTINGS.confirm_seconds:
            return float(now)
    return None


def files(paths: list[Path]) -> None:
    """Per recording and rule: detection second, and the closest joint margin min(peak - P, contrast - C)."""
    print("rules: " + " | ".join(f"P{p:g}/C{c:g}" for p, c in RULES))
    for path in paths:
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        seconds, peaks, contrasts = lock_trajectory(resample_to_model_rate(audio.mean(axis=1), rate))
        cells: list[str] = []
        for p, c in RULES:
            when: float | None = detection_second(seconds, (peaks >= p) & (contrasts >= c))
            margin: float = float(np.minimum(peaks - p, contrasts - c).max())
            cells.append(f"{'-' if when is None else f'{when:.1f}s':>6} {margin:+5.1f}")
        print(f"{path.stem[16:60]:<44} " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "files":
        files([Path(p) for p in sys.argv[2:]])
    elif sys.argv[1] == "corpus":
        corpus()
    else:
        raise SystemExit(f"Unknown mode {sys.argv[1]!r}: use 'files' or 'corpus'")
