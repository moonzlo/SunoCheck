"""H25 pilot: one r-lock search over the slowdown grid with two heads instead of per-case rules.

Every evasion seen so far moves the Suno artifact peaks by an unknown ratio r (resample: lofcz still matches once
undone, H23; pitch shift with resynthesis: the shifted model H18 matches, H21). Both show a sharp score peak at
the true r. One search per evaluation, as the app would run it: every COARSE-th ratio of r 0.80-0.98, refine
+-2 grid steps around each head's best, contrast against 38 neighbors 0.2-2 % away. The current path (H18 >= +8
or H21 lock at the exact semitones) is computed on the same windows for comparison and union.

Corpus: the shifted head is out-of-fold (GroupKFold over its training set, as research/shift_lock.py); the lofcz
weights are external. 2 workers.

Usage:
  uv run python research/r_lock.py files <wav> [...]
  uv run python research/r_lock.py files-with <shifted weights.npz> <wav> [...]
  uv run python research/r_lock.py corpus
"""

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from corpus import BIN_HZ, BLOCK_SECONDS, Item, Label, list_items
from joblib import Parallel, delayed
from lab import FREQS, LOFCZ_LEVEL_OFFSET_DB, band_bins, fakeprint, load_windows
from numpy.typing import NDArray
from shift_lock import fold_models
from shifted_model import SUNO_SHIFTS, dataset

from sunocheck.fakeprint import (
    DEFAULT_MODEL_PATH,
    MODEL_SAMPLE_RATE,
    SHIFT_SEMITONES,
    SHIFTED_MODEL_PATH,
    STREAM_HOP,
    STREAM_MIN_RATIO,
    empty_stream,
    fakeprint_from_band,
    load_model,
    push_samples,
    resample_to_model_rate,
    warped_bands,
)
from sunocheck.settings import DEFAULT_SETTINGS

RATIO_STEP: float = 0.0002
GRID: NDArray[np.float64] = np.exp(np.arange(np.ceil(np.log(STREAM_MIN_RATIO) / RATIO_STEP) + 1, 1) * RATIO_STEP)
CANDIDATES: int = int(np.searchsorted(GRID, 0.98, side="right"))
COARSE_INDEX: NDArray[np.intp] = np.arange(0, CANDIDATES, 4)
REFINE_STEPS: NDArray[np.intp] = np.arange(-2, 3)
NEIGHBOR_STEPS: NDArray[np.intp] = np.concatenate([-np.arange(10, 101, 5)[::-1], np.arange(10, 101, 5)])
SEMITONE_INDEX: NDArray[np.intp] = np.array(
    [int(np.argmin(np.abs(GRID - 2.0 ** (-n / 12)))) for n in SHIFT_SEMITONES], dtype=np.intp
)
LOFCZ, SHIFTED = 0, 1
Rule = tuple[float, float]
LOFCZ_RULES: tuple[Rule, ...] = ((6.0, 5.0), (6.0, 6.0), (8.0, 6.0))
SHIFTED_RULES: tuple[Rule, ...] = ((2.0, 5.0), (4.0, 5.0), (4.0, 6.0), (6.0, 6.0), (8.0, 6.0))
UNIONS: tuple[tuple[Rule, Rule], ...] = (((6.0, 6.0), (4.0, 6.0)), ((6.0, 6.0), (6.0, 6.0)), ((6.0, 6.0), (8.0, 6.0)))
STEP_SECONDS: float = 0.5
WORKERS: int = 2
LOFCZ_BINS: NDArray[np.intp] = band_bins(1000.0, 8000.0)

# grid indices (rows, k) -> scores (rows, k, heads)
ScoreAt = Callable[[NDArray[np.intp]], NDArray[np.float64]]


@dataclass(frozen=True)
class Head:
    weights: NDArray[np.float64]  # (folds, bins)
    biases: NDArray[np.float64]  # (folds,)
    folds: NDArray[np.intp]  # per row


@dataclass(frozen=True)
class Locks:
    """Per row: global search peak and contrast per head (rows, heads); current path at the exact semitones."""

    peaks: NDArray[np.float64]
    contrasts: NDArray[np.float64]
    current_peak: NDArray[np.float64]
    current_contrast: NDArray[np.float64]


def search(score_at: ScoreAt, rows: int) -> Locks:
    coarse: NDArray[np.float64] = score_at(np.tile(COARSE_INDEX, (rows, 1)))
    heads: int = coarse.shape[2]
    peaks: NDArray[np.float64] = np.empty((rows, heads))
    contrasts: NDArray[np.float64] = np.empty((rows, heads))
    every: NDArray[np.intp] = np.arange(rows)
    for head in range(heads):
        center: NDArray[np.intp] = COARSE_INDEX[np.argmax(coarse[:, :, head], axis=1)]
        refine_index: NDArray[np.intp] = np.clip(center[:, None] + REFINE_STEPS, 0, CANDIDATES - 1)
        refine: NDArray[np.float64] = score_at(refine_index)[:, :, head]
        best: NDArray[np.intp] = refine_index[every, np.argmax(refine, axis=1)]
        neighbors: NDArray[np.float64] = score_at(np.clip(best[:, None] + NEIGHBOR_STEPS, 0, GRID.size - 1))[:, :, head]
        peaks[:, head] = refine.max(axis=1)
        contrasts[:, head] = peaks[:, head] - neighbors.max(axis=1)
    semitone_peaks: list[NDArray[np.float64]] = []
    semitone_contrasts: list[NDArray[np.float64]] = []
    for index in SEMITONE_INDEX:
        core: NDArray[np.float64] = score_at(np.tile(index + REFINE_STEPS, (rows, 1)))[:, :, SHIFTED].max(axis=1)
        around: NDArray[np.float64] = score_at(np.tile(index + NEIGHBOR_STEPS, (rows, 1)))[:, :, SHIFTED].max(axis=1)
        semitone_peaks.append(core)
        semitone_contrasts.append(core - around)
    winner: NDArray[np.intp] = np.argmax(np.stack(semitone_peaks, axis=1), axis=1)
    return Locks(
        peaks=peaks,
        contrasts=contrasts,
        current_peak=np.stack(semitone_peaks, axis=1)[every, winner],
        current_contrast=np.stack(semitone_contrasts, axis=1)[every, winner],
    )


def hits(locks: Locks, head: int, rule: Rule) -> NDArray[np.bool_]:
    return (locks.peaks[:, head] >= rule[0]) & (locks.contrasts[:, head] >= rule[1])


def current_hits(locks: Locks) -> NDArray[np.bool_]:
    return (locks.current_peak >= 8.0) | ((locks.current_peak >= 2.0) & (locks.current_contrast >= 5.0))


def union_hits(locks: Locks, union: tuple[Rule, Rule]) -> NDArray[np.bool_]:
    return current_hits(locks) | hits(locks, LOFCZ, union[0]) | hits(locks, SHIFTED, union[1])


def margin(locks: Locks, head: int, rule: Rule) -> float:
    """Closest joint approach min(peak - P, contrast - C) to the rule; > 0 means a hit."""
    return float(np.minimum(locks.peaks[:, head] - rule[0], locks.contrasts[:, head] - rule[1]).max())


def band_prints(spectra: NDArray[np.float32], ratios: NDArray[np.float64]) -> NDArray[np.float32]:
    """lofcz fakeprints of level-offset window spectra, each row read at its own ratio r * f."""
    positions: NDArray[np.float64] = ratios[:, None] * FREQS[LOFCZ_BINS][None, :] / BIN_HZ
    lower: NDArray[np.intp] = np.floor(positions).astype(np.intp)
    frac: NDArray[np.float32] = (positions - lower).astype(np.float32)
    band: NDArray[np.float32] = (
        np.take_along_axis(spectra, lower, axis=1) * (1 - frac) + np.take_along_axis(spectra, lower + 1, axis=1) * frac
    )
    return fakeprint(band, np.arange(LOFCZ_BINS.size), 10, -45.0, 5.0)


def corpus_score_at(spectra: NDArray[np.float32], heads: tuple[Head, ...]) -> ScoreAt:
    every: NDArray[np.intp] = np.arange(len(spectra))

    def score_at(index: NDArray[np.intp]) -> NDArray[np.float64]:
        out: NDArray[np.float64] = np.empty((*index.shape, len(heads)))
        for column in range(index.shape[1]):
            prints: NDArray[np.float32] = band_prints(spectra, GRID[index[:, column]])
            for h, head in enumerate(heads):
                out[:, column, h] = (prints @ head.weights.T)[every, head.folds] + head.biases[head.folds]
        return out

    return score_at


def corpus_case(
    blocks: int, name: str, items: list[Item], variant: str, shifted: tuple[dict[str, int], NDArray[np.float64], NDArray[np.float64]]
) -> str:
    w = load_windows(items, variant, blocks, 2)
    lofcz = load_model(DEFAULT_MODEL_PATH)
    fold_of, weights, biases = shifted
    rows: int = len(w.mid)
    heads: tuple[Head, ...] = (
        Head(lofcz.weights[None, :].astype(np.float64), np.array([lofcz.bias]), np.zeros(rows, dtype=np.intp)),
        Head(weights, biases, np.array([fold_of.get(item, 0) for item in w.item_id.tolist()], dtype=np.intp)),
    )
    locks: Locks = search(corpus_score_at(w.mid + np.float32(LOFCZ_LEVEL_OFFSET_DB), heads), rows)
    lofcz_cells: str = " ".join(f"{np.mean(hits(locks, LOFCZ, r)):.3f}" for r in LOFCZ_RULES)
    shifted_cells: str = " ".join(f"{np.mean(hits(locks, SHIFTED, r)):.3f}" for r in SHIFTED_RULES)
    union_cells: str = " ".join(f"{np.mean(union_hits(locks, u)):.3f}" for u in UNIONS)
    return (
        f"{blocks * BLOCK_SECONDS:g} s {name:<19} n={rows:<5} cur {np.mean(current_hits(locks)):.3f}"
        f" | L max {locks.peaks[:, LOFCZ].max():+5.1f} m {margin(locks, LOFCZ, (6.0, 6.0)):+5.1f}: {lofcz_cells}"
        f" | S max {locks.peaks[:, SHIFTED].max():+5.1f} m {margin(locks, SHIFTED, (4.0, 6.0)):+5.1f}: {shifted_cells}"
        f" | U {union_cells}"
    )


def corpus() -> None:
    x, y, groups, _, _ = dataset(SUNO_SHIFTS, ((8, 2),))
    shifted = fold_models(x, y, groups)
    print("L (lofcz) rules " + " ".join(f"{p:g}/{c:g}" for p, c in LOFCZ_RULES) + "; m = margin to 6/6")
    print("S (shifted, out-of-fold) rules " + " ".join(f"{p:g}/{c:g}" for p, c in SHIFTED_RULES) + "; m = margin to 4/6")
    print("U = current | L | S: " + " ".join(f"L{a[0]:g}/{a[1]:g}+S{b[0]:g}/{b[1]:g}" for a, b in UNIONS), flush=True)
    humans: list[Item] = [i for i in list_items() if i.label == Label.HUMAN]
    suno: list[Item] = [i for i in list_items() if i.label == Label.SUNO]
    cases: list[tuple[str, list[Item], str]] = [
        (f"human {v}", humans, v)
        for v in ("orig", "aac_128", "speed_down4", "speed_down15", "slowed_hall", "tempo_95", "pitch_down2", "hall")
    ] + [(f"suno {v}", suno, v) for v in ("speed_down4", "speed_down15", "wsola_down1", "pitch_down2", "slowed_hall")]
    jobs = [(blocks, name, items, variant, shifted) for blocks in (4, 8) for name, items, variant in cases]
    for line in Parallel(n_jobs=WORKERS)(delayed(corpus_case)(*job) for job in jobs):
        print(line, flush=True)


def live_score_at(mean_db: NDArray[np.float64], weights: NDArray[np.float64], biases: NDArray[np.float64]) -> ScoreAt:
    """weights (heads, bins) of the app models; one row."""

    def score_at(index: NDArray[np.intp]) -> NDArray[np.float64]:
        prints: NDArray[np.float32] = fakeprint_from_band(warped_bands(mean_db, GRID[index[0]]))
        return (prints @ weights.T + biases)[None, :, :]

    return score_at


def detection_second(seconds: NDArray[np.float64], hit: NDArray[np.bool_]) -> str:
    """First second at which hits held for confirm_seconds, as decision.update_evidence does."""
    since: float | None = None
    for now, on in zip(seconds, hit, strict=True):
        if not on:
            since = None
            continue
        since = now if since is None else since
        if now - since >= DEFAULT_SETTINGS.confirm_seconds:
            return f"{now:.1f}s"
    return "-"


def files(shifted_path: Path, paths: list[Path]) -> None:
    lofcz = load_model(DEFAULT_MODEL_PATH)
    shifted = load_model(shifted_path)
    weights: NDArray[np.float64] = np.stack([lofcz.weights, shifted.weights]).astype(np.float64)
    biases: NDArray[np.float64] = np.array([lofcz.bias, shifted.bias])
    max_frames: int = int(DEFAULT_SETTINGS.max_seconds * MODEL_SAMPLE_RATE) // STREAM_HOP
    step: int = int(STEP_SECONDS * MODEL_SAMPLE_RATE)
    print("columns: cur | L 6/6 | S 4/6 | S 6/6 | " + " | ".join(f"U L{a[0]:g}/{a[1]:g}+S{b[0]:g}/{b[1]:g}" for a, b in UNIONS))
    costs: list[float] = []
    for path in paths:
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        mono: NDArray[np.float32] = resample_to_model_rate(audio.mean(axis=1), rate)
        stream = empty_stream()
        seconds: list[float] = []
        steps: list[Locks] = []
        for start in range(0, mono.size - step + 1, step):
            stream = push_samples(stream, mono[start : start + step], max_frames)
            now: float = (start + step) / MODEL_SAMPLE_RATE
            if not stream.frames or now < DEFAULT_SETTINGS.min_seconds:
                continue
            started: float = time.perf_counter()
            steps.append(search(live_score_at(stream.band_sum / len(stream.frames), weights, biases), 1))
            costs.append(time.perf_counter() - started)
            seconds.append(now)
        locks = Locks(*(np.concatenate([getattr(s, f) for s in steps]) for f in ("peaks", "contrasts", "current_peak", "current_contrast")))
        at: NDArray[np.float64] = np.array(seconds)
        cells: list[str] = [
            detection_second(at, current_hits(locks)),
            detection_second(at, hits(locks, LOFCZ, (6.0, 6.0))),
            detection_second(at, hits(locks, SHIFTED, (4.0, 6.0))),
            detection_second(at, hits(locks, SHIFTED, (6.0, 6.0))),
        ] + [detection_second(at, union_hits(locks, u)) for u in UNIONS]
        print(
            f"{path.stem[16:56]:<40} " + " | ".join(f"{c:>6}" for c in cells)
            + f" | L {locks.peaks[:, LOFCZ].max():+5.1f} m {margin(locks, LOFCZ, (6.0, 6.0)):+5.1f}"
            + f" S {locks.peaks[:, SHIFTED].max():+5.1f} m {margin(locks, SHIFTED, (4.0, 6.0)):+5.1f}",
            flush=True,
        )
    print(f"search cost: mean {np.mean(costs) * 1000:.0f} ms, p95 {np.quantile(costs, 0.95) * 1000:.0f} ms per evaluation")


if __name__ == "__main__":
    if sys.argv[1] == "files":
        files(SHIFTED_MODEL_PATH, [Path(p) for p in sys.argv[2:]])
    elif sys.argv[1] == "files-with":
        files(Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]])
    elif sys.argv[1] == "corpus":
        corpus()
    else:
        raise SystemExit(f"Unknown mode {sys.argv[1]!r}: use 'files' or 'corpus'")
