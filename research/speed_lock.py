"""H23: speed lock of the lofcz model over a wide slowdown grid on the live stream path.

A resampled (slowed) Suno track keeps its narrow artifact peaks, just moved to r * f. Reading the spectrum at r * f
restores them only at the exact r: 0.2-2 % off they miss the weights again (as in H21). Humans score randomly
along r, and a max over ~1000 ratios lifts their best score (H8), so a hit needs a high peak and a high contrast
against the neighbors of the winning r. The lofcz weights are external, so the corpus check is out-of-sample.

Usage:
  uv run python research/speed_lock.py files <wav> [...]
  uv run python research/speed_lock.py corpus
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from corpus import BIN_HZ, BLOCK_SECONDS, Item, Label, list_items
from joblib import Parallel, delayed
from lab import FREQS, LOFCZ_LEVEL_OFFSET_DB, band_bins, fakeprint, load_windows
from numpy.typing import NDArray

from sunocheck.fakeprint import (
    DEFAULT_MODEL_PATH,
    MODEL_SAMPLE_RATE,
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
SPEED_RATIOS: NDArray[np.float64] = np.exp(np.arange(np.ceil(np.log(STREAM_MIN_RATIO) / RATIO_STEP) + 1, 1) * RATIO_STEP)
# The unshifted r = 1 peak belongs to the plain fakeprint path; a slowdown under 2 % is not searched.
MAX_PEAK_RATIO: float = 0.98
NEIGHBOR_STEPS: NDArray[np.int64] = np.concatenate([-np.arange(10, 101, 5)[::-1], np.arange(10, 101, 5)])
RULES: tuple[tuple[float, float], ...] = ((8.0, -np.inf), (4.0, 5.0), (5.0, 6.0), (6.0, 5.0), (6.0, 6.0), (6.0, 7.0), (8.0, 6.0))
# Live search: every COARSE-th ratio, then +-REFINE full-grid steps around the best (the peak is +-2 steps wide).
COARSE: int = 4
REFINE: int = 2
MARGIN_RULE: tuple[float, float] = (6.0, 6.0)
STEP_SECONDS: float = 0.5
CHUNK: int = 8
WORKERS: int = 14
LOFCZ_BINS: NDArray[np.intp] = band_bins(1000.0, 8000.0)


def peak_and_contrast(scores: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """scores (rows, SPEED_RATIOS) -> per row: best score at r <= MAX_PEAK_RATIO, its contrast, its ratio."""
    candidates: int = int(np.searchsorted(SPEED_RATIOS, MAX_PEAK_RATIO, side="right"))
    return contrast_at(scores, np.argmax(scores[:, :candidates], axis=1))


def coarse_peak_and_contrast(scores: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """As the live search: best of every COARSE-th ratio, refined within +-REFINE steps, then the contrast."""
    candidates: int = int(np.searchsorted(SPEED_RATIOS, MAX_PEAK_RATIO, side="right"))
    coarse: NDArray[np.intp] = np.arange(0, candidates, COARSE)
    rows: NDArray[np.intp] = np.arange(scores.shape[0])
    center: NDArray[np.intp] = coarse[np.argmax(scores[:, coarse], axis=1)]
    around: NDArray[np.intp] = np.clip(center[:, None] + np.arange(-REFINE, REFINE + 1)[None, :], 0, candidates - 1)
    best: NDArray[np.intp] = around[rows, np.argmax(scores[rows[:, None], around], axis=1)]
    return contrast_at(scores, best)


def contrast_at(
    scores: NDArray[np.float64], best: NDArray[np.intp]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    rows: NDArray[np.intp] = np.arange(scores.shape[0])
    neighbors: NDArray[np.intp] = np.clip(best[:, None] + NEIGHBOR_STEPS[None, :], 0, SPEED_RATIOS.size - 1)
    peaks: NDArray[np.float64] = scores[rows, best]
    return peaks, peaks - scores[rows[:, None], neighbors].max(axis=1), SPEED_RATIOS[best]


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


def speed_trajectory(mono: NDArray[np.float32]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per step of the live stream: (seconds, scores (steps, SPEED_RATIOS))."""
    model = load_model(DEFAULT_MODEL_PATH)
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
        mean_db: NDArray[np.float64] = stream.band_sum / len(stream.frames)
        steps.append(fakeprint_from_band(warped_bands(mean_db, SPEED_RATIOS)) @ model.weights + model.bias)
        seconds.append(now)
    return np.array(seconds), np.stack(steps)


def files(paths: list[Path]) -> None:
    """Per recording and rule: detection second; then max peak, max contrast and the ratio at the end."""
    print("columns: " + " | ".join(f"P{p:g}/C{c:g}" for p, c in RULES))
    for path in paths:
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        seconds, scores = speed_trajectory(resample_to_model_rate(audio.mean(axis=1), rate))
        peaks, contrasts, ratios = coarse_peak_and_contrast(scores)
        cells: list[str] = []
        for p, c in RULES:
            when: float | None = detection_second(seconds, (peaks >= p) & (contrasts >= c))
            cells.append(f"{'-' if when is None else f'{when:.1f}s':>6}")
        print(
            f"{path.stem[16:60]:<44} " + " | ".join(cells)
            + f" | peak max {peaks.max():+5.1f} contrast max {contrasts.max():+5.1f} r_end {ratios[-1]:.4f}",
            flush=True,
        )


def corpus_scores(mids: NDArray[np.float32]) -> NDArray[np.float64]:
    """(windows, SPEED_RATIOS) lofcz logits of the window-mean spectra read at r * f."""
    model = load_model(DEFAULT_MODEL_PATH)
    out: NDArray[np.float64] = np.empty((len(mids), SPEED_RATIOS.size))
    spectra: NDArray[np.float32] = mids + np.float32(LOFCZ_LEVEL_OFFSET_DB)
    for first in range(0, SPEED_RATIOS.size, CHUNK):
        ratios: NDArray[np.float64] = SPEED_RATIOS[first : first + CHUNK]
        positions: NDArray[np.float64] = np.outer(ratios, FREQS[LOFCZ_BINS]) / BIN_HZ
        lower: NDArray[np.intp] = np.floor(positions).astype(np.intp)
        frac: NDArray[np.float32] = (positions - lower).astype(np.float32)
        bands: NDArray[np.float32] = spectra[:, lower] * (1 - frac) + spectra[:, lower + 1] * frac
        flat: NDArray[np.float32] = bands.reshape(-1, LOFCZ_BINS.size)
        prints: NDArray[np.float32] = fakeprint(flat, np.arange(LOFCZ_BINS.size), 10, -45.0, 5.0)
        out[:, first : first + ratios.size] = (prints @ model.weights + model.bias).reshape(len(mids), ratios.size)
    return out


def rule_table(name: str, peaks: NDArray[np.float64], contrasts: NDArray[np.float64]) -> str:
    """Hit rate per rule; `margin`: the closest joint approach min(peak - P, contrast - C) to MARGIN_RULE."""
    cells: list[str] = [f"{np.mean((peaks >= p) & (contrasts >= c)):.3f}" for p, c in RULES]
    margin: float = float(np.minimum(peaks - MARGIN_RULE[0], contrasts - MARGIN_RULE[1]).max())
    return (
        f"{name:<30} n={peaks.size:<5} peak max {peaks.max():+6.2f} contrast max {contrasts.max():+6.2f}"
        f" margin {margin:+6.2f} | " + " ".join(cells)
    )


def corpus() -> None:
    print("rules (peak >= P and contrast >= C): " + " ".join(f"P{p:g}/C{c:g}" for p, c in RULES))
    humans: list[Item] = [i for i in list_items() if i.label == Label.HUMAN]
    suno: list[Item] = [i for i in list_items() if i.label == Label.SUNO]
    cases: tuple[tuple[str, list[Item], str], ...] = tuple(
        (f"human {v}", humans, v)
        for v in ("orig", "aac_128", "speed_down4", "speed_down15", "slowed_hall", "tempo_95", "pitch_down2", "hall")
    ) + tuple((f"suno {v}", suno, v) for v in ("speed_down4", "speed_down15", "slowed_hall", "wsola_down1", "pitch_down2", "orig"))
    jobs = [(blocks, name, items, variant) for blocks in (4, 8) for name, items, variant in cases]
    for line in Parallel(n_jobs=WORKERS)(delayed(corpus_case)(*job) for job in jobs):
        print(line, flush=True)


def corpus_case(blocks: int, name: str, items: list[Item], variant: str) -> str:
    w = load_windows(items, variant, blocks, 2)
    scores: NDArray[np.float64] = corpus_scores(w.mid)
    full_peaks, full_contrasts, _ = peak_and_contrast(scores)
    live_peaks, live_contrasts, _ = coarse_peak_and_contrast(scores)
    label: str = f"{blocks * BLOCK_SECONDS:g} s {name}"
    return rule_table(f"{label} full", full_peaks, full_contrasts) + "\n" + rule_table(f"{label} live", live_peaks, live_contrasts)


if __name__ == "__main__":
    if sys.argv[1] == "files":
        files([Path(p) for p in sys.argv[2:]])
    elif sys.argv[1] == "corpus":
        corpus()
    else:
        raise SystemExit(f"Unknown mode {sys.argv[1]!r}: use 'files' or 'corpus'")
