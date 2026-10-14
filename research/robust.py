"""Shift-robust features: comb search over base frequency, and position-free peak statistics."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import find_peaks
from scipy.stats import kurtosis

from lab import FREQS, FloatArray, band_bins

BIN_HZ: float = float(FREQS[1])
SCAN_LOW_HZ: float = 1000.0
SCAN_HIGH_HZ: float = 15800.0
# Reference points between lines; dyadic fractions make any finer comb (e.g. echo ripple at 200/n Hz)
# score as high as the reference and therefore cancel out.
REF_OFFSETS: FloatArray = np.array([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
LINE_SPAN: tuple[int, ...] = (-1, 0, 1, 2)


def line_level(spectra_db: NDArray[np.float32], positions_bins: FloatArray) -> NDArray[np.float32]:
    """Max over a 4-bin span around each fractional bin position; positions of any shape."""
    floor_idx = np.floor(positions_bins).astype(np.intp)
    return np.max(np.stack([spectra_db[:, floor_idx + d] for d in LINE_SPAN]), axis=0)


@dataclass(frozen=True)
class CombScan:
    best_score: FloatArray
    best_base_hz: FloatArray
    score_at_200: FloatArray
    best_z: FloatArray


def scan_bases(low_hz: float, high_hz: float) -> FloatArray:
    """Base step keeps the highest harmonic within half a bin of its true position."""
    step = 0.5 * BIN_HZ / (SCAN_HIGH_HZ / low_hz)
    return np.arange(low_hz, high_hz, step)


def harmonic_contrast(spectra_db: NDArray[np.float32], base_hz: float, low_hz: float, high_hz: float) -> NDArray[np.float32]:
    """[rows, harmonics]: line level at k*base minus mean line level at (k + REF_OFFSETS)*base."""
    k = np.arange(int(np.ceil(low_hz / base_hz)), int(high_hz // base_hz) + 1)
    line = line_level(spectra_db, k * base_hz / BIN_HZ)
    refs = line_level(spectra_db, (k[:, None] + REF_OFFSETS[None, :]) * base_hz / BIN_HZ)
    return line - refs.mean(axis=2)


CEPSTRUM_PAD: int = 1 << 17
CEPSTRUM_BACKGROUND_HZ: tuple[float, float] = (120.0, 400.0)


def comb_scan(spectra_db: NDArray[np.float32], bases_hz: FloatArray) -> CombScan:
    """Cepstral comb search: a line comb with spacing f0 in the dB residue is a peak of |FFT(residue)|
    at 1/f0 cycles per Hz. Prominence is measured against the cepstrum over the background spacing range."""
    residue = residue_db(spectra_db, SCAN_LOW_HZ, SCAN_HIGH_HZ).astype(np.float64)
    residue -= residue.mean(axis=1, keepdims=True)
    residue *= np.hanning(residue.shape[1])[None, :]
    magnitude = np.abs(np.fft.rfft(residue, n=CEPSTRUM_PAD, axis=1))
    spacing_hz = np.full(magnitude.shape[1], np.inf)
    spacing_hz[1:] = CEPSTRUM_PAD * BIN_HZ / np.arange(1, magnitude.shape[1])
    search = (spacing_hz >= bases_hz.min()) & (spacing_hz <= bases_hz.max())
    background = (spacing_hz >= CEPSTRUM_BACKGROUND_HZ[0]) & (spacing_hz <= CEPSTRUM_BACKGROUND_HZ[1])
    log_mag = np.log(magnitude + 1e-9)
    center = np.median(log_mag[:, background], axis=1)
    spread = np.std(log_mag[:, background], axis=1) + 1e-6
    search_idx = np.flatnonzero(search)
    best = search_idx[np.argmax(log_mag[:, search_idx], axis=1)]
    rows = np.arange(log_mag.shape[0])
    at_200 = np.argmin(np.abs(spacing_hz - 200.0))
    near_200 = slice(at_200 - 2, at_200 + 3)
    return CombScan(
        best_score=log_mag[rows, best] - center,
        best_base_hz=spacing_hz[best],
        score_at_200=log_mag[:, near_200].max(axis=1) - center,
        best_z=(log_mag[rows, best] - center) / spread,
    )


def residue_db(spectra_db: NDArray[np.float32], low_hz: float, high_hz: float) -> NDArray[np.float32]:
    band = spectra_db[:, band_bins(low_hz, high_hz)].astype(np.float32)
    hull = uniform_filter1d(minimum_filter1d(band, size=10, axis=1), size=10, axis=1)
    return band - hull


PEAK_STAT_NAMES: tuple[str, ...] = ("frac>3", "frac>6", "frac>9", "n_peaks", "prom_top20", "prom_median", "kurtosis")


def peak_stats(spectra_db: NDArray[np.float32]) -> NDArray[np.float32]:
    residue = residue_db(spectra_db, 1000.0, 15800.0)
    rows = []
    for r in residue:
        peaks, props = find_peaks(r, prominence=1.0, distance=3)
        prom = np.sort(props["prominences"])[::-1] if len(peaks) else np.zeros(1)
        rows.append(
            [
                np.mean(r > 3.0),
                np.mean(r > 6.0),
                np.mean(r > 9.0),
                len(peaks) / r.size,
                prom[:20].mean(),
                np.median(prom),
                kurtosis(r),
            ]
        )
    return np.array(rows, dtype=np.float32)
