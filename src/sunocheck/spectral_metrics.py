"""Model-agnostic descriptors of the fakeprint peak pattern, for comparing approaches beside the logistic model.

- peak_stability: artifact peaks from the generator's upsampling stay at fixed frequencies, while musical
  harmonics move with the notes; correlation of the residue of two consecutive time blocks.
- comb_strength / comb_spacing_hz: regularity of peak spacing (autocorrelation along frequency).
- peak_count / peak_prominence: how many distinct peaks and how far they stand above the floor.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from sunocheck.fakeprint import MODEL_SAMPLE_RATE, N_FFT, STREAM_HOP, FloatArray, SpectrumStream, fakeprint_from_stream_band

BIN_HZ: float = MODEL_SAMPLE_RATE / N_FFT
COMB_MIN_HZ: float = 20.0
COMB_MAX_HZ: float = 1000.0
PEAK_PROMINENCE: float = 0.3
PEAK_TOP: int = 20
STABILITY_MIN_SECONDS: float = 2.0
STABILITY_MAX_SECONDS: float = 10.0
RECENT_SECONDS: float = 0.5
FRAMES_PER_SECOND: float = MODEL_SAMPLE_RATE / STREAM_HOP


@dataclass(frozen=True)
class SpectralMetrics:
    peak_count: int
    peak_prominence: float
    comb_strength: float
    comb_spacing_hz: float
    peak_stability: float | None


def comb_periodicity(fakeprint: FloatArray) -> tuple[float, float]:
    """Highest normalized autocorrelation over plausible peak spacings and the spacing in Hz."""
    centered: NDArray[np.float64] = fakeprint.astype(np.float64) - float(fakeprint.mean())
    spectrum: NDArray[np.complex128] = np.fft.rfft(centered, n=2 * centered.size)
    autocorrelation: NDArray[np.float64] = np.fft.irfft(np.abs(spectrum) ** 2)[: centered.size]
    if autocorrelation[0] <= 0.0:
        return 0.0, 0.0
    normalized: NDArray[np.float64] = autocorrelation / autocorrelation[0]
    first: int = int(np.ceil(COMB_MIN_HZ / BIN_HZ))
    last: int = int(COMB_MAX_HZ / BIN_HZ)
    lag: int = first + int(np.argmax(normalized[first : last + 1]))
    return float(normalized[lag]), lag * BIN_HZ


def peak_profile(fakeprint: FloatArray) -> tuple[int, float]:
    peaks, _ = find_peaks(fakeprint, prominence=PEAK_PROMINENCE)
    top: NDArray[np.float32] = np.sort(fakeprint)[-PEAK_TOP:]
    floor: float = float(np.median(fakeprint)) + 1e-6
    return int(peaks.size), float(top.mean()) / floor


def block_fakeprint(frames: tuple[FloatArray, ...]) -> FloatArray:
    return fakeprint_from_stream_band(np.mean(np.stack(frames), axis=0, dtype=np.float64))


def peak_stability(stream: SpectrumStream) -> float | None:
    block: int = min(len(stream.frames) // 2, int(STABILITY_MAX_SECONDS * FRAMES_PER_SECOND))
    if block < int(STABILITY_MIN_SECONDS * FRAMES_PER_SECOND):
        return None
    older: FloatArray = block_fakeprint(stream.frames[-2 * block : -block])
    newer: FloatArray = block_fakeprint(stream.frames[-block:])
    return float(np.corrcoef(older, newer)[0, 1])


def recent_residue(stream: SpectrumStream) -> FloatArray:
    """Residue of the last ~0.5 s: one column of the live residue spectrogram."""
    count: int = max(1, int(RECENT_SECONDS * FRAMES_PER_SECOND))
    return block_fakeprint(stream.frames[-count:])


def spectral_metrics(fakeprint: FloatArray, stream: SpectrumStream) -> SpectralMetrics:
    comb_strength, comb_spacing_hz = comb_periodicity(fakeprint)
    peak_count, peak_prominence = peak_profile(fakeprint)
    return SpectralMetrics(
        peak_count=peak_count,
        peak_prominence=peak_prominence,
        comb_strength=comb_strength,
        comb_spacing_hz=comb_spacing_hz,
        peak_stability=peak_stability(stream),
    )
