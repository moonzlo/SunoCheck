"""Speed-robust coherent comb detector.

A Suno track sped up by r keeps a phase-stable line comb at 200*r Hz. One long FFT per window; contrast
spectrum = power over its local (±5 Hz) linear mean; coarse scan of the base over low harmonics, refine
with all harmonics, then compare the lines to points between them. A loop-locked human comb (e.g. a
sample repeated every 1/8 s) also has lines at base*(k + j/m); Suno only at base*k.

Usage: uv run python research/coherent_scan.py <seconds> <variant> <item_id> [...]
"""

import sys
from dataclasses import dataclass

import numpy as np
from scipy import fft
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from corpus import PLAYBACK_RATE, VARIANTS
from lab import all_items
from prepare import decode_variant

BASE_LOW_HZ: float = 160.0
BASE_HIGH_HZ: float = 260.0
COARSE_HIGH_HZ: float = 8000.0
LOW_HZ: float = 1000.0
HIGH_HZ: float = 15800.0
LOCAL_HZ: float = 5.0
REFINE_CANDIDATES: int = 100
SCORE_CHUNK: int = 4096
SUBDIVISIONS: tuple[int, ...] = tuple(range(2, 17))
SUNO_BASE_HZ: float = 200.0


@dataclass(frozen=True)
class CombStats:
    base_hz: float
    plain: float
    sub: np.ndarray  # [len(SUBDIVISIONS)] median contrast at base*(k + j/m), 0 < j < m
    lines: np.ndarray  # [MAX_HARMONIC] contrast at k*base for k = 1..MAX_HARMONIC, NaN above HIGH_HZ


MAX_HARMONIC: int = 100


@dataclass(frozen=True)
class ScanResult:
    scanned: CombStats
    fixed: CombStats  # at SUNO_BASE_HZ
    line_quantiles: np.ndarray  # [len(LINE_QUANTILES)] of the 3-bin-max contrast over [LOW_HZ, HIGH_HZ]


LINE_QUANTILES: tuple[float, ...] = (0.99, 0.995, 0.999)


def contrast_spectrum(segment: np.ndarray) -> tuple[np.ndarray, float]:
    power = np.abs(fft.rfft(segment * np.hanning(segment.size))) ** 2
    bin_hz = PLAYBACK_RATE / segment.size
    local = np.maximum(uniform_filter1d(power, int(2 * LOCAL_HZ / bin_hz) + 1), 1e-20)
    return 10 * np.log10(np.maximum(power, 1e-20) / local), bin_hz


def line_values(contrast: np.ndarray, freqs_hz: np.ndarray, bin_hz: float) -> np.ndarray:
    center = np.rint(freqs_hz / bin_hz).astype(np.intp)
    return np.max(np.stack([contrast[center - 1], contrast[center], contrast[center + 1]]), axis=0)


def comb_score(contrast: np.ndarray, bin_hz: float, bases: np.ndarray, high_hz: float) -> np.ndarray:
    """Mean line contrast over harmonics in [LOW_HZ, high_hz] for each base; [bases]."""
    k = np.arange(1, int(high_hz // bases.min()) + 1)
    out = np.empty(bases.size)
    for start in range(0, bases.size, SCORE_CHUNK):
        freqs = bases[start : start + SCORE_CHUNK, None] * k[None, :]
        valid = (freqs >= LOW_HZ) & (freqs <= high_hz)
        values = line_values(contrast, np.where(valid, freqs, LOW_HZ), bin_hz)
        out[start : start + SCORE_CHUNK] = np.where(valid, values, 0.0).sum(axis=1) / valid.sum(axis=1)
    return out


def comb_stats(contrast: np.ndarray, bin_hz: float, base: float) -> CombStats:
    k = np.arange(int(np.ceil(LOW_HZ / base)), int(HIGH_HZ // base))
    plain = float(np.median(line_values(contrast, base * k, bin_hz)))
    sub = np.array([
        np.median(line_values(contrast, (base * (k[:, None] + np.arange(1, m)[None, :] / m)).ravel(), bin_hz))
        for m in SUBDIVISIONS
    ])
    all_k = np.arange(1, MAX_HARMONIC + 1)
    in_band = all_k * base <= HIGH_HZ
    lines = np.full(MAX_HARMONIC, np.nan)
    lines[in_band] = line_values(contrast, base * all_k[in_band], bin_hz)
    return CombStats(base, plain, sub, lines)


def best_base(contrast: np.ndarray, bin_hz: float) -> float:
    coarse_step = 0.5 * bin_hz / (COARSE_HIGH_HZ / BASE_LOW_HZ)
    coarse = np.arange(BASE_LOW_HZ, BASE_HIGH_HZ, coarse_step)
    coarse_scores = comb_score(contrast, bin_hz, coarse, COARSE_HIGH_HZ)
    candidates = coarse[np.argsort(coarse_scores)[::-1][:REFINE_CANDIDATES]]
    fine_step = 0.5 * bin_hz / (HIGH_HZ / BASE_LOW_HZ)
    fine = np.concatenate([np.arange(c - coarse_step, c + coarse_step, fine_step) for c in candidates])
    return float(fine[np.argmax(comb_score(contrast, bin_hz, fine, HIGH_HZ))])


def scan(segment: np.ndarray) -> ScanResult:
    contrast, bin_hz = contrast_spectrum(segment)
    base = best_base(contrast, bin_hz)
    band = maximum_filter1d(contrast[int(LOW_HZ / bin_hz) : int(HIGH_HZ / bin_hz)], 3)
    return ScanResult(comb_stats(contrast, bin_hz, base), comb_stats(contrast, bin_hz, SUNO_BASE_HZ),
                      np.quantile(band, LINE_QUANTILES))


def main(seconds: float, variant_name: str, item_ids: list[str]) -> None:
    variant = next(v for v in VARIANTS if v.name == variant_name)
    items = {i.item_id: i for i in all_items()}
    size = int(seconds * PLAYBACK_RATE)
    for item_id in item_ids:
        mono = decode_variant(items[item_id], variant).mean(axis=1)
        results = [scan(mono[s : s + size]).scanned for s in range(0, mono.size - size + 1, size)][:4]
        print(f"{variant_name:<13} {item_id:<22} {items[item_id].label:<6} " + " | ".join(
            f"{r.base_hz:7.2f}Hz p{r.plain:4.1f} x{r.plain - r.sub.max():4.1f}" for r in results))


if __name__ == "__main__":
    main(float(sys.argv[1]), sys.argv[2], sys.argv[3:])
