"""Coherent comb: ppm refine (H41) restores score after small speed skew."""

import numpy as np
import pytest

from sunocheck.coherent_comb import BASE_HZ, CHECKPOINTS, comb_score, comb_score_fixed

RATE: int = 44100
SECONDS: float = 10.0


def synthetic_comb(sample_rate: int, seconds: float, base_hz: float, seed: int) -> np.ndarray:
    """Tone stack at k·base_hz plus weak noise."""
    n = int(sample_rate * seconds)
    t = np.arange(n, dtype=np.float64) / sample_rate
    audio = np.zeros(n, dtype=np.float64)
    for harmonic in range(5, 70):
        audio += (0.04 / harmonic) * np.sin(2.0 * np.pi * harmonic * base_hz * t)
    rng = np.random.default_rng(seed)
    audio += rng.normal(0.0, 0.0005, size=n)
    peak = np.max(np.abs(audio))
    return (audio / peak * 0.5).astype(np.float32)


def test_ppm_refine_matches_fixed_on_exact_grid() -> None:
    audio = synthetic_comb(RATE, SECONDS, BASE_HZ, seed=1)
    fixed = comb_score_fixed(audio, RATE)
    refined = comb_score(audio, RATE)
    assert refined >= fixed - 0.05


def test_ppm_refine_recovers_50_ppm_skew() -> None:
    skewed_base = BASE_HZ * (1.0 + 50e-6)
    audio = synthetic_comb(RATE, SECONDS, skewed_base, seed=2)
    fixed = comb_score_fixed(audio, RATE)
    refined = comb_score(audio, RATE)
    assert refined > fixed + 2.0


def test_h42_long_checkpoints_present() -> None:
    seconds = [c.seconds for c in CHECKPOINTS]
    assert 90.0 in seconds
    assert 180.0 in seconds
    thr45 = next(c.threshold_db for c in CHECKPOINTS if c.seconds == 45.0)
    thr90 = next(c.threshold_db for c in CHECKPOINTS if c.seconds == 90.0)
    thr180 = next(c.threshold_db for c in CHECKPOINTS if c.seconds == 180.0)
    assert thr90 >= thr45
    assert thr180 >= thr90


def test_ttd_early_and_mid_checkpoints() -> None:
    seconds = [c.seconds for c in CHECKPOINTS]
    assert seconds[:6] == [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    thresholds = [c.threshold_db for c in CHECKPOINTS]
    assert thresholds == sorted(thresholds)
    assert next(c.threshold_db for c in CHECKPOINTS if c.seconds == 1.0) == -2.4
    assert next(c.threshold_db for c in CHECKPOINTS if c.seconds == 5.0) == -2.2

