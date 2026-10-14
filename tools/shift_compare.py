"""Compare shifted-Suno model candidates on the live stream path, every 0.5 s of audio from min_seconds.

A candidate = (weights, spectrum window): window 0 = the running mean over the whole stream (live),
window W = the mean of the last W seconds only.
Per recording and candidate: max shifted score, and the first track second at which the score stayed
>= the candidate threshold for confirm_seconds (as update_evidence does).

Usage: uv run python tmp/shift_compare.py <thresholds,...> <weights.npz,...> <windows,...> <wav> [...]
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from sunocheck.fakeprint import (
    MODEL_SAMPLE_RATE,
    SHIFT_SEMITONES,
    SHIFT_TOLERANCE,
    STREAM_HOP,
    FakeprintModel,
    empty_stream,
    fakeprint_from_band,
    load_model,
    push_samples,
    resample_to_model_rate,
    warped_bands,
)
from sunocheck.settings import DEFAULT_SETTINGS

STEP_SECONDS: float = 0.5
RATIOS: NDArray[np.float64] = np.outer(2.0 ** (-np.array(SHIFT_SEMITONES, dtype=np.float64) / 12), SHIFT_TOLERANCE).ravel()


def window_mean_db(frames: tuple[NDArray[np.float32], ...], window_seconds: float) -> NDArray[np.float64]:
    """Mean dB spectrum of the last `window_seconds` of frames; 0 = the whole running stream."""
    if window_seconds == 0.0:
        return np.mean(frames, axis=0, dtype=np.float64)
    count = max(1, int(window_seconds * MODEL_SAMPLE_RATE) // STREAM_HOP)
    return np.mean(frames[-count:], axis=0, dtype=np.float64)


def score_track(
    mono: NDArray[np.float32], models: list[FakeprintModel], windows: list[float]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Returns (seconds, scores (n_steps, n_candidates))."""
    max_frames = int(DEFAULT_SETTINGS.max_seconds * MODEL_SAMPLE_RATE) // STREAM_HOP
    step = int(STEP_SECONDS * MODEL_SAMPLE_RATE)
    stream = empty_stream()
    seconds: list[float] = []
    rows: list[list[float]] = []
    for start in range(0, mono.size - step + 1, step):
        stream = push_samples(stream, mono[start : start + step], max_frames)
        now = (start + step) / MODEL_SAMPLE_RATE
        if not stream.frames or now < DEFAULT_SETTINGS.min_seconds:
            continue
        row: list[float] = []
        for model, window in zip(models, windows, strict=True):
            fakeprints = fakeprint_from_band(warped_bands(window_mean_db(stream.frames, window), RATIOS))
            row.append(float((fakeprints @ model.weights).max()) + model.bias)
        seconds.append(now)
        rows.append(row)
    return np.array(seconds), np.array(rows)


def detection_second(seconds: NDArray[np.float64], scores: NDArray[np.float64], threshold: float) -> float | None:
    since: float | None = None
    for now, score in zip(seconds, scores, strict=True):
        if score < threshold:
            since = None
            continue
        since = now if since is None else since
        if now - since >= DEFAULT_SETTINGS.confirm_seconds:
            return float(now)
    return None


def main(thresholds: list[float], weight_paths: list[Path], windows: list[float], paths: list[Path]) -> None:
    models = [load_model(path) for path in weight_paths]
    print("candidates: " + ", ".join(f"{i}={p.stem}/w{w:g}" for i, (p, w) in enumerate(zip(weight_paths, windows, strict=True))))
    print(f"thresholds: {thresholds}")
    for path in paths:
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        seconds, scores = score_track(resample_to_model_rate(audio.mean(axis=1), rate), models, windows)
        cells: list[str] = []
        for i, threshold in enumerate(thresholds):
            when = detection_second(seconds, scores[:, i], threshold)
            cells.append(f"{scores[:, i].max():+6.1f} @{'-' if when is None else f'{when:.1f}s':>6}")
        print(f"{path.stem[16:66]:<50} " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main(
        [float(t) for t in sys.argv[1].split(",")],
        [Path(p) for p in sys.argv[2].split(",")],
        [float(w) for w in sys.argv[3].split(",")],
        [Path(p) for p in sys.argv[4:]],
    )
