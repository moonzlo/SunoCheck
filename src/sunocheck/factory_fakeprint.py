"""Factory (−1 semitone WSOLA) detector: LR on 1–14 kHz fakeprint @ 32 kHz with max-over-r (research H29).

Live capture keeps the native rate for the coherent comb; this path resamples that buffer to 32 kHz so the
8–14 kHz band exists. Joint rule from holdout: factory_max ≥ 0 and (factory_max − off_window_max) ≥ 2.
Scores the best 7.5 s loud window over a short slide (research H29 / TTD): first-loud alone missed
RAVNIR on loopback; slide recovered it without raising human FPR.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soxr
from numpy.typing import NDArray
from scipy.ndimage import minimum_filter1d

from sunocheck.fakeprint import FloatArray

FACTORY_MODEL_PATH: Path = Path(__file__).parent / "models" / "factory_weights.npz"
ANALYSIS_RATE: int = 32000
N_FFT: int = 16384
HOP: int = 1024
BLOCK_SECONDS: float = 2.5
# TTD: 3×2.5 s = 7.5 s window keeps live RAVNIR/Velkira at stock joint; FPR 0 on 40 FMA orig
# (ttd_factory_window). 2 blocks (5 s) drops RAVNIR; 4 blocks only adds latency.
WINDOW_BLOCKS: int = 3
SILENT_DBFS: float = -45.0
HULL_AREA: int = 10
MIN_DB: float = -45.0
MAX_DB: float = 5.0
# Holdout joint (research H29): FPR 0 on 76 humans; live recordings also FPR 0.
FACTORY_LOGIT: float = 0.0
FACTORY_CONTRAST: float = 2.0
# Cap + slide step from h29_window_variants: longer cap alone does not help Dust/Phase/Jernblod;
# soft joint (fact≥−2) raises corpus human FPR — keep stock thresholds, take best window.
SCORE_CAP_SECONDS: float = 60.0
SLIDE_STEP_SECONDS: float = 5.0
R_FACTORY: NDArray[np.float64] = np.linspace(0.9428, 0.9450, 12)
R_OFF: NDArray[np.float64] = np.concatenate(
    [
        np.linspace(0.90, 0.935, 8),
        np.linspace(0.95, 0.98, 8),
        np.array([1.0, 0.8909, 1.0595]),
    ]
)
_WINDOW: NDArray[np.float64] = np.hanning(N_FFT + 1)[:-1]
_FREQS: NDArray[np.float64] = np.arange(N_FFT // 2 + 1, dtype=np.float64) * (ANALYSIS_RATE / N_FFT)


@dataclass(frozen=True)
class FactoryModel:
    weights: FloatArray
    bias: float
    bins: NDArray[np.intp]


@dataclass(frozen=True)
class FactoryScore:
    """`score` is factory_max; `lock_contrast` is factory_max − best off-window logit (H29 joint uses both)."""

    score: float
    lock_contrast: float
    ratio: float


def load_factory_model(path: Path) -> FactoryModel:
    data = np.load(path)
    lo = float(data["band_lo_hz"])
    hi = float(data["band_hi_hz"])
    bins = np.flatnonzero((_FREQS >= lo) & (_FREQS <= hi))
    weights = data["weights"].reshape(-1).astype(np.float32)
    if weights.size != bins.size:
        raise ValueError(f"factory weights length {weights.size} != band bins {bins.size}")
    return FactoryModel(weights=weights, bias=float(np.asarray(data["bias"]).reshape(-1)[0]), bins=bins)


def _fakeprint_band(spectrum_db: NDArray[np.float64], bins: NDArray[np.intp]) -> FloatArray:
    band = spectrum_db[bins]
    hull = np.clip(minimum_filter1d(band, size=HULL_AREA, mode="nearest"), MIN_DB, None)
    residue = np.clip(band - hull, 0.0, MAX_DB)
    return (residue / (residue.max() + 1e-6)).astype(np.float32)


def warp_spectrum(spectrum: NDArray[np.float32], ratio: float) -> NDArray[np.float32]:
    """Read spectrum as if frequencies were scaled by ratio (undo pitch-down => ratio > 1)."""
    src = _FREQS / ratio
    warped = np.interp(src, _FREQS, spectrum.astype(np.float64), left=float(spectrum[0]), right=float(spectrum[-1]))
    return warped.astype(np.float32)


def _frames_per_block() -> int:
    return round(BLOCK_SECONDS * ANALYSIS_RATE / HOP)


def _window_need_samples() -> int:
    return WINDOW_BLOCKS * _frames_per_block() * HOP + N_FFT


def first_loud_start(mono_analysis: FloatArray) -> int | None:
    block = int(0.1 * ANALYSIS_RATE)
    n_blocks = mono_analysis.size // block
    if n_blocks == 0:
        return None
    blocks = mono_analysis[: n_blocks * block].reshape(n_blocks, block)
    loud = np.flatnonzero(10.0 * np.log10(np.mean(blocks**2, axis=1) + 1e-12) >= SILENT_DBFS)
    if loud.size == 0:
        return None
    return int(loud[0]) * block


def spectrum_at(mono_analysis: FloatArray, start_sample: int) -> NDArray[np.float32] | None:
    """Mean spectrum of WINDOW_BLOCKS·BLOCK_SECONDS starting at `start_sample`, or None if too short."""
    need = _window_need_samples()
    audio = mono_analysis[start_sample:]
    if audio.size < need:
        return None
    frames_per_block = _frames_per_block()
    frames = np.lib.stride_tricks.sliding_window_view(audio, N_FFT)[::HOP]
    blocks_n = frames.shape[0] // frames_per_block
    if blocks_n < WINDOW_BLOCKS:
        return None
    specs: list[NDArray[np.float64]] = []
    for index in range(WINDOW_BLOCKS):
        chunk = frames[index * frames_per_block : (index + 1) * frames_per_block]
        power = np.abs(np.fft.rfft(chunk * _WINDOW, axis=1)) ** 2
        specs.append((10.0 * np.log10(np.clip(power, 1e-12, None))).mean(axis=0))
    return np.mean(np.stack(specs), axis=0).astype(np.float32)


def mean_loud_spectrum(mono_analysis: FloatArray) -> NDArray[np.float32] | None:
    """First loud WINDOW_BLOCKS·BLOCK_SECONDS mean spectrum at ANALYSIS_RATE, or None if too short."""
    start = first_loud_start(mono_analysis)
    if start is None:
        return None
    return spectrum_at(mono_analysis, start)


def max_over_r(spectrum: NDArray[np.float32], model: FactoryModel, ratios: NDArray[np.float64]) -> tuple[float, float]:
    best = float("-inf")
    best_r = float(ratios[0])
    for ratio in ratios:
        fp = _fakeprint_band(warp_spectrum(spectrum, 1.0 / float(ratio)).astype(np.float64), model.bins)
        score = float(fp @ model.weights) + model.bias
        if score > best:
            best = score
            best_r = float(ratio)
    return best, best_r


def score_spectrum(spectrum: NDArray[np.float32], model: FactoryModel) -> FactoryScore:
    fact, best_r = max_over_r(spectrum, model, R_FACTORY)
    off, _ = max_over_r(spectrum, model, R_OFF)
    return FactoryScore(score=fact, lock_contrast=fact - off, ratio=best_r)


def factory_score(mono_capture: FloatArray, sample_rate: int, model: FactoryModel) -> FactoryScore:
    """Score capture-rate mono through the 32 kHz factory LR (max-over-r + off-window contrast).

    Takes the best 7.5 s window after first loud, stepping by SLIDE_STEP_SECONDS up to SCORE_CAP_SECONDS.
    """
    if sample_rate != ANALYSIS_RATE:
        mono = soxr.resample(mono_capture.astype(np.float64), sample_rate, ANALYSIS_RATE, quality="HQ").astype(np.float32)
    else:
        mono = mono_capture.astype(np.float32)
    mono = mono[: int(SCORE_CAP_SECONDS * ANALYSIS_RATE)]
    start0 = first_loud_start(mono)
    if start0 is None:
        return FactoryScore(score=float("-inf"), lock_contrast=float("-inf"), ratio=float(R_FACTORY[0]))
    step = int(SLIDE_STEP_SECONDS * ANALYSIS_RATE)
    best = FactoryScore(score=float("-inf"), lock_contrast=float("-inf"), ratio=float(R_FACTORY[0]))
    pos = start0
    while True:
        spectrum = spectrum_at(mono, pos)
        if spectrum is None:
            break
        score = score_spectrum(spectrum, model)
        if score.score > best.score or (
            score.score == best.score and score.lock_contrast > best.lock_contrast
        ):
            best = score
        if factory_joint(best):
            return best
        pos += step
        if pos >= mono.size:
            break
    return best


def factory_joint(score: FactoryScore | None) -> bool:
    return (
        score is not None
        and score.score >= FACTORY_LOGIT
        and score.lock_contrast >= FACTORY_CONTRAST
    )
