"""Coherent 200 Hz comb detector (see research/REPORT.md, research/HYPOTHESES.md H15).

Suno output carries phase-stable spectral lines at exact multiples of 200 Hz (48000/240, a 5 ms decoder
frame). One long FFT resolves them as narrow peaks, and they stand out more the longer the window. So the
window grows from the track's first loud audio and is scored at each checkpoint; one checkpoint at or above
its threshold flags the track. Score = median line contrast at k*200 Hz minus the DENSITY_QUANTILE of the
contrast over the whole band, so dense loop-locked combs of human music (a sample repeated every 1/T s) do not
count. Loops still gain with length too, hence thresholds that rise with the window.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from sunocheck.fakeprint import FloatArray


@dataclass(frozen=True)
class Checkpoint:
    seconds: float
    threshold_db: float


# research/comb_growing.py: human max per window length over 1450 tracks (725 x orig/AAC 128) + 0.5 dB,
# never below the previous checkpoint. 0 humans fire; Suno orig/AAC 128 flagged by 5 s 63/33 %,
# by 10 s 83/64 %, by 45 s 96/91 %. H42 / E15 + h42_long_thresholds: 90/180 s keep the same −0.85
# (Jamendo human_max ≪ −0.85; do not loosen below the short-window calibration).
# TTD (2026-10-16): 1.0/1.5/2.0 s inherit the 2.5 s bar (−2.4); FPR 0 on 80 FMA humans (orig∪aac).
# 3.0/4.0 s cut the wait from 2.5→5 for tracks that cross between coarse points (same bars).
CHECKPOINTS: tuple[Checkpoint, ...] = (
    Checkpoint(1.0, -2.4),
    Checkpoint(1.5, -2.4),
    Checkpoint(2.0, -2.4),
    Checkpoint(2.5, -2.4),
    Checkpoint(3.0, -2.3),
    Checkpoint(4.0, -2.2),
    Checkpoint(5.0, -2.2),
    Checkpoint(7.5, -2.1),
    Checkpoint(10.0, -1.9),
    Checkpoint(12.5, -1.75),
    Checkpoint(15.0, -1.25),
    Checkpoint(20.0, -0.85),
    Checkpoint(30.0, -0.85),
    Checkpoint(45.0, -0.85),
    Checkpoint(90.0, -0.85),
    Checkpoint(180.0, -0.85),
)
BASE_HZ: float = 200.0
LOW_HZ: float = 1000.0
HIGH_HZ: float = 15800.0
LOCAL_HZ: float = 5.0
DENSITY_QUANTILE: float = 0.995
POWER_FLOOR: float = 1e-20
# research H41 / E17: clock skew and acoustic paths move lines by ε·f; one contrast FFT then
# max over base = 200·(1+ε) restores the score. Grid ±150 ppm at 0.5 ppm (~0.5/(f_max·T) at 45 s).
PPM_REFINE_LO: float = -150.0
PPM_REFINE_HI: float = 150.0
PPM_REFINE_STEP: float = 0.5
_PPM_GRID: NDArray[np.float64] = np.arange(PPM_REFINE_LO, PPM_REFINE_HI + PPM_REFINE_STEP * 0.5, PPM_REFINE_STEP)


class SampleRateTooLowError(ValueError):
    pass


@dataclass(frozen=True)
class CombState:
    """Capture-rate audio of the track up to the last checkpoint and the scores of checkpoints done so far."""

    chunks: tuple[FloatArray, ...]
    samples: int
    scores: tuple[float, ...]


def empty_comb_state() -> CombState:
    return CombState(chunks=(), samples=0, scores=())


def check_sample_rate(sample_rate: int) -> None:
    if sample_rate < 2 * HIGH_HZ * 1.05:
        raise SampleRateTooLowError(f"Comb detector needs lines up to {HIGH_HZ:.0f} Hz; capture rate {sample_rate} Hz is too low")


def contrast_db(segment: FloatArray, sample_rate: int) -> tuple[NDArray[np.float64], float]:
    """Power over its local (±LOCAL_HZ) linear mean, in dB; and the bin width in Hz."""
    power: NDArray[np.float64] = np.abs(np.fft.rfft(segment.astype(np.float64) * np.hanning(segment.size))) ** 2
    bin_hz: float = sample_rate / segment.size
    local: NDArray[np.float64] = uniform_filter1d(power, int(2 * LOCAL_HZ / bin_hz) + 1)
    return 10.0 * np.log10(np.maximum(power, POWER_FLOOR) / np.maximum(local, POWER_FLOOR)), bin_hz


def comb_score_at_base(contrast: NDArray[np.float64], bin_hz: float, base_hz: float) -> float:
    """Median tooth contrast at k·base_hz minus the band density quantile (same as the r = 1 score)."""
    harmonics: NDArray[np.float64] = np.arange(np.ceil(LOW_HZ / base_hz), HIGH_HZ // base_hz) * base_hz
    centers: NDArray[np.intp] = np.rint(harmonics / bin_hz).astype(np.intp)
    centers = centers[(centers >= 1) & (centers + 1 < contrast.size)]
    if centers.size == 0:
        return float("-inf")
    lines: NDArray[np.float64] = np.max(np.stack([contrast[centers - 1], contrast[centers], contrast[centers + 1]]), axis=0)
    band: NDArray[np.float64] = maximum_filter1d(contrast[int(LOW_HZ / bin_hz) : int(HIGH_HZ / bin_hz)], 3)
    return float(np.median(lines) - np.quantile(band, DENSITY_QUANTILE))


def comb_score(segment: FloatArray, sample_rate: int) -> float:
    """Coherent comb score with ppm refine (H41): max over bases 200·(1+ε), ε on `_PPM_GRID`."""
    contrast, bin_hz = contrast_db(segment, sample_rate)
    best: float = float("-inf")
    for ppm in _PPM_GRID:
        score = comb_score_at_base(contrast, bin_hz, BASE_HZ * (1.0 + float(ppm) * 1e-6))
        if score > best:
            best = score
    return best


def comb_score_fixed(segment: FloatArray, sample_rate: int) -> float:
    """Score at exactly 200 Hz (no ppm refine). Kept for A/B and research scripts."""
    contrast, bin_hz = contrast_db(segment, sample_rate)
    return comb_score_at_base(contrast, bin_hz, BASE_HZ)


def push_comb_audio(state: CombState, chunk: FloatArray, sample_rate: int) -> CombState:
    """Append capture-rate audio; nothing past the last checkpoint is kept."""
    room: int = int(CHECKPOINTS[-1].seconds * sample_rate) - state.samples
    if room <= 0:
        return state
    kept: FloatArray = chunk[:room]
    return CombState(chunks=state.chunks + (kept,), samples=state.samples + kept.size, scores=state.scores)


def due_checkpoint(state: CombState, sample_rate: int) -> Checkpoint | None:
    """The next unscored checkpoint, if enough audio has arrived for it."""
    if len(state.scores) == len(CHECKPOINTS):
        return None
    checkpoint: Checkpoint = CHECKPOINTS[len(state.scores)]
    return checkpoint if state.samples >= int(checkpoint.seconds * sample_rate) else None


def comb_audio(state: CombState, checkpoint: Checkpoint, sample_rate: int) -> FloatArray:
    return np.concatenate(state.chunks)[: int(checkpoint.seconds * sample_rate)]


def add_comb_score(state: CombState, score: float) -> CombState:
    return CombState(chunks=state.chunks, samples=state.samples, scores=state.scores + (score,))


def last_comb_score(state: CombState) -> float | None:
    return state.scores[-1] if state.scores else None


def comb_detected(state: CombState) -> bool:
    return any(score >= checkpoint.threshold_db for score, checkpoint in zip(state.scores, CHECKPOINTS, strict=False))
