"""Model-free comb detector: level at exact multiples of a base frequency vs. the gaps between them."""

import numpy as np
from numpy.typing import NDArray

from lab import FREQS, FloatArray

BIN_HZ: float = float(FREQS[1])


def comb_contrast(spectra_db: NDArray[np.float32], base_hz: float, low_hz: float, high_hz: float, peak_bins: int, gap_fraction: float) -> FloatArray:
    """Per row: mean over harmonics k of [max dB within ±peak_bins of k*base] - [median dB of the gap
    region (k ± gap_fraction*base/2 excluded around the line, i.e. the middle of the gap)].

    Returns an array [rows, harmonics] of contrasts in dB.
    """
    harmonics = np.arange(int(np.ceil(low_hz / base_hz)), int(high_hz // base_hz) + 1)
    contrasts = np.empty((spectra_db.shape[0], len(harmonics)), dtype=np.float64)
    half_gap = gap_fraction * base_hz / 2.0
    for column, k in enumerate(harmonics):
        center = int(round(k * base_hz / BIN_HZ))
        peak = spectra_db[:, center - peak_bins : center + peak_bins + 1].max(axis=1)
        mid_gap = (k + 0.5) * base_hz
        gap_lo, gap_hi = int(round((mid_gap - half_gap) / BIN_HZ)), int(round((mid_gap + half_gap) / BIN_HZ))
        prev_gap = (k - 0.5) * base_hz
        pg_lo, pg_hi = int(round((prev_gap - half_gap) / BIN_HZ)), int(round((prev_gap + half_gap) / BIN_HZ))
        gap = np.concatenate([spectra_db[:, pg_lo : pg_hi + 1], spectra_db[:, gap_lo : gap_hi + 1]], axis=1)
        contrasts[:, column] = peak - np.median(gap, axis=1)
    return contrasts


def comb_score(spectra_db: NDArray[np.float32], base_hz: float, low_hz: float, high_hz: float) -> FloatArray:
    return comb_contrast(spectra_db, base_hz, low_hz, high_hz, 1, 0.5).mean(axis=1)
