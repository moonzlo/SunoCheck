"""Fakeprint features (Afchar et al., "A Fourier Explanation of AI-music Artifacts", ISMIR 2025)
and the logistic-regression head from lofcz/ai-music-detector (MIT)."""

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import minimum_filter1d
from scipy.signal import resample_poly
from scipy.special import expit

FloatArray = NDArray[np.float32]

# Must match the preprocessing the lofcz weights were trained with.
MODEL_SAMPLE_RATE: int = 16000
N_FFT: int = 8192
HOP_LENGTH: int = N_FFT // 2
FREQ_MIN_HZ: float = 1000.0
FREQ_MAX_HZ: float = 8000.0
HULL_AREA: int = 10
MAX_DB: float = 5.0
MIN_DB: float = -45.0

DEFAULT_MODEL_PATH: Path = Path(__file__).parent / "models" / "lofcz_weights.npz"
# Shifted-Suno model (research H18, research/shifted_model.py): same features after an exact semitone undo.
SHIFTED_MODEL_PATH: Path = Path(__file__).parent / "models" / "shifted_weights.npz"

# Streaming: denser frames than the model's hop give the same mean with less variance on short windows.
STREAM_HOP: int = 1024
_FREQS: NDArray[np.float64] = np.linspace(0.0, MODEL_SAMPLE_RATE / 2, N_FFT // 2 + 1)
BAND_BINS: NDArray[np.intp] = np.flatnonzero((_FREQS >= FREQ_MIN_HZ) & (_FREQS <= FREQ_MAX_HZ))

# Pitch-shift evasion (research H16-H18): Suno lowered by n semitones keeps smeared fakeprint peaks, moved to
# 2^(-n/12) * f. Undoing a raise would need content above the 8 kHz Nyquist of the stream.
SHIFT_SEMITONES: tuple[int, ...] = (1, 2, 3)
# Live evasions were off the exact ratio by up to 0.04 % (resampling between 44.1 and 48 kHz).
SHIFT_TOLERANCE: NDArray[np.float64] = np.exp(np.arange(-2, 3) * 0.0002)
# Lowest r = f_observed / f_original the stream can undo, with a margin below 3 semitones down.
STREAM_MIN_RATIO: float = 0.8
STREAM_BINS: NDArray[np.intp] = np.flatnonzero((_FREQS >= FREQ_MIN_HZ * STREAM_MIN_RATIO) & (_FREQS <= FREQ_MAX_HZ))
BAND_OFFSET: int = int(BAND_BINS[0] - STREAM_BINS[0])
_WINDOW: NDArray[np.float64] = np.hanning(N_FFT + 1)[:-1]


@dataclass(frozen=True)
class FakeprintModel:
    weights: FloatArray
    bias: float


def load_model(path: Path) -> FakeprintModel:
    data = np.load(path)
    weights: FloatArray = data["weights"].reshape(-1).astype(np.float32)
    bias: float = float(data["bias"].reshape(-1)[0])
    return FakeprintModel(weights=weights, bias=bias)


def resample_to_model_rate(mono: FloatArray, sample_rate: int) -> FloatArray:
    if sample_rate == MODEL_SAMPLE_RATE:
        return mono
    divisor: int = math.gcd(sample_rate, MODEL_SAMPLE_RATE)
    resampled: NDArray[np.float64] = resample_poly(mono, MODEL_SAMPLE_RATE // divisor, sample_rate // divisor)
    return resampled.astype(np.float32)


def mean_power_db(audio: FloatArray) -> NDArray[np.float64]:
    """Time-averaged power spectrum in dB.

    Mirrors torchaudio.transforms.Spectrogram(n_fft=8192, power=2): periodic Hann window,
    hop n_fft // 2, center=True with reflect padding.
    """
    if audio.size <= N_FFT // 2:
        raise ValueError(f"Need more than {N_FFT // 2} samples at {MODEL_SAMPLE_RATE} Hz, got {audio.size}")
    padded: FloatArray = np.pad(audio, (N_FFT // 2, N_FFT // 2), mode="reflect")
    frames: FloatArray = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP_LENGTH]
    power: NDArray[np.float64] = np.abs(np.fft.rfft(frames * _WINDOW, axis=1)) ** 2
    power_db: NDArray[np.float64] = 10.0 * np.log10(np.clip(power, 1e-10, 1e6))
    return power_db.mean(axis=0)


def band_power_db(frames: FloatArray) -> FloatArray:
    """Per-frame power in dB over STREAM_BINS; `frames` is (n, N_FFT)."""
    power: NDArray[np.float64] = np.abs(np.fft.rfft(frames * _WINDOW, axis=1)[:, STREAM_BINS]) ** 2
    return (10.0 * np.log10(np.clip(power, 1e-10, 1e6))).astype(np.float32)


def fakeprint_from_band(band_db: NDArray[np.float64]) -> FloatArray:
    """Fakeprint of one band spectrum, or of each row of a (n, band) stack."""
    hull: NDArray[np.float64] = np.clip(minimum_filter1d(band_db, size=HULL_AREA, axis=-1, mode="nearest"), MIN_DB, None)
    residue: NDArray[np.float64] = np.clip(band_db - hull, 0.0, MAX_DB)
    return (residue / (residue.max(axis=-1, keepdims=True) + 1e-6)).astype(np.float32)


def fakeprint_from_stream_band(stream_db: NDArray[np.float64]) -> FloatArray:
    return fakeprint_from_band(stream_db[BAND_OFFSET:])


def warped_bands(stream_db: NDArray[np.float64], ratios: NDArray[np.float64]) -> NDArray[np.float64]:
    """(len(ratios), band) model-band spectra as before a pitch change by r = f_observed / f_original."""
    if ratios.min() < STREAM_MIN_RATIO or ratios.max() * FREQ_MAX_HZ > _FREQS[STREAM_BINS[-1]]:
        raise ValueError(f"Ratios {ratios.min():.4f}..{ratios.max():.4f} read outside the stream band")
    positions: NDArray[np.float64] = (np.outer(ratios, _FREQS[BAND_BINS]) - _FREQS[STREAM_BINS[0]]) / _FREQS[1]
    lower: NDArray[np.intp] = np.minimum(np.floor(positions).astype(np.intp), STREAM_BINS.size - 2)
    frac: NDArray[np.float64] = positions - lower
    return stream_db[lower] * (1.0 - frac) + stream_db[lower + 1] * frac


def fakeprint_from_spectrum(spectrum_db: NDArray[np.float64]) -> FloatArray:
    return fakeprint_from_band(spectrum_db[BAND_BINS])


def compute_fakeprint(audio_model_rate: FloatArray) -> FloatArray:
    return fakeprint_from_spectrum(mean_power_db(audio_model_rate))


def score_from_fakeprint(fakeprint: FloatArray, model: FakeprintModel) -> float:
    """Logit of the model; threshold-free and comparable across window lengths."""
    if fakeprint.size != model.weights.size:
        raise ValueError(f"Fakeprint has {fakeprint.size} bins, model expects {model.weights.size}")
    return float(fakeprint @ model.weights) + model.bias


def probability_from_fakeprint(fakeprint: FloatArray, model: FakeprintModel) -> float:
    return float(expit(score_from_fakeprint(fakeprint, model)))


@dataclass(frozen=True)
class SpectrumStream:
    """Running mean of per-frame band spectra over the last `max_frames` frames of 16 kHz audio."""

    pending: FloatArray
    frames: tuple[FloatArray, ...]
    band_sum: NDArray[np.float64]


def empty_stream() -> SpectrumStream:
    return SpectrumStream(
        pending=np.zeros(0, dtype=np.float32), frames=(), band_sum=np.zeros(STREAM_BINS.size, dtype=np.float64)
    )


def push_samples(stream: SpectrumStream, samples: FloatArray, max_frames: int) -> SpectrumStream:
    pending: FloatArray = np.concatenate((stream.pending, samples))
    if pending.size < N_FFT:
        return replace(stream, pending=pending)
    windows: FloatArray = np.lib.stride_tricks.sliding_window_view(pending, N_FFT)[::STREAM_HOP]
    new_frames: FloatArray = band_power_db(windows)
    frames: tuple[FloatArray, ...] = stream.frames + tuple(new_frames)
    band_sum: NDArray[np.float64] = stream.band_sum + new_frames.sum(axis=0, dtype=np.float64)
    excess: int = len(frames) - max_frames
    if excess > 0:
        band_sum = band_sum - np.sum(frames[:excess], axis=0, dtype=np.float64)
        frames = frames[excess:]
    return SpectrumStream(pending=pending[windows.shape[0] * STREAM_HOP :], frames=frames, band_sum=band_sum)


def stream_mean_db(stream: SpectrumStream) -> NDArray[np.float64]:
    if not stream.frames:
        raise ValueError("No complete spectrum frames yet")
    return stream.band_sum / len(stream.frames)


def stream_fakeprint(stream: SpectrumStream) -> FloatArray:
    return fakeprint_from_stream_band(stream_mean_db(stream))


def stream_shifted_score(stream: SpectrumStream, model: FakeprintModel) -> float:
    """Best logit of the shifted-Suno model over SHIFT_SEMITONES pitch-down undos of the running mean spectrum."""
    ratios: NDArray[np.float64] = np.outer(2.0 ** (-np.array(SHIFT_SEMITONES, dtype=np.float64) / 12), SHIFT_TOLERANCE).ravel()
    fakeprints: FloatArray = fakeprint_from_band(warped_bands(stream_mean_db(stream), ratios))
    return float((fakeprints @ model.weights).max()) + model.bias


def ai_probability(mono: FloatArray, sample_rate: int, model: FakeprintModel) -> float:
    return probability_from_fakeprint(compute_fakeprint(resample_to_model_rate(mono, sample_rate)), model)
