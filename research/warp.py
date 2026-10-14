"""Frequency-warp search: undo an unknown speed change r by reading the spectrum at r*f, score, take the max."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sunocheck.fakeprint import FakeprintModel
from lab import FREQS, FloatArray, band_bins, lofcz_fakeprint

LOFCZ_BINS = band_bins(1000.0, 8000.0)


@dataclass(frozen=True)
class WarpResult:
    best_score: FloatArray
    best_ratio: FloatArray
    unwarped_score: FloatArray


def warp_ratios(low: float, high: float, log_step: float) -> FloatArray:
    """Geometric grid that always contains exactly 1.0."""
    below = np.exp(-log_step * np.arange(int(np.floor(np.log(1.0 / low) / log_step)), 0, -1))
    above = np.exp(log_step * np.arange(0, int(np.floor(np.log(high) / log_step)) + 1))
    return np.concatenate([below, above])


def warped(spectra_db: NDArray[np.float32], ratio: float) -> NDArray[np.float32]:
    """Spectrum as it was before a speed change by `ratio` (observed f_obs = ratio * f_orig)."""
    positions = FREQS * ratio / FREQS[1]
    valid = positions <= len(FREQS) - 1
    out = np.empty_like(spectra_db)
    lower = np.floor(positions[valid]).astype(np.intp)
    upper = np.minimum(lower + 1, len(FREQS) - 1)
    frac = (positions[valid] - lower).astype(np.float32)
    out[:, valid] = spectra_db[:, lower] * (1 - frac) + spectra_db[:, upper] * frac
    out[:, ~valid] = spectra_db[:, -1:]
    return out


def lofcz_warp_search(spectra_db: NDArray[np.float32], model: FakeprintModel, ratios: FloatArray) -> WarpResult:
    scores = np.stack([lofcz_fakeprint(warped(spectra_db, r)) @ model.weights + model.bias for r in ratios], axis=1)
    best = scores.argmax(axis=1)
    unwarped = int(np.argmin(np.abs(ratios - 1.0)))
    return WarpResult(scores.max(axis=1), ratios[best], scores[:, unwarped])
