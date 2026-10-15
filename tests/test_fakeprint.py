import math
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from sunocheck.decision import SPEED_CONTRAST, SPEED_LOGIT
from sunocheck.fakeprint import (
    BAND_BINS,
    DEFAULT_MODEL_PATH,
    MODEL_SAMPLE_RATE,
    N_FFT,
    SHIFTED_MODEL_PATH,
    SPEED_PEAK_COUNT,
    SPEED_RATIOS,
    STREAM_BINS,
    STREAM_HOP,
    empty_stream,
    fakeprint_from_band,
    load_model,
    push_samples,
    score_from_fakeprint,
    speed_score,
    stream_fakeprint,
    stream_mean_db,
    stream_shifted_score,
    warped_bands,
)


@pytest.mark.parametrize("path", [DEFAULT_MODEL_PATH, SHIFTED_MODEL_PATH])
def test_models_match_the_fakeprint_length(path: Path) -> None:
    assert load_model(path).weights.shape == (BAND_BINS.size,)


def test_fakeprint_rows_are_normalized_to_one() -> None:
    rng = np.random.default_rng(0)
    stack = rng.normal(-40.0, 6.0, size=(3, BAND_BINS.size))
    fakeprints = fakeprint_from_band(stack)
    assert fakeprints.shape == (3, BAND_BINS.size)
    assert np.allclose(fakeprints.max(axis=1), 1.0, atol=1e-3)
    assert fakeprints.min() >= 0.0


def test_warp_outside_the_stream_band_is_an_error() -> None:
    with pytest.raises(ValueError):
        warped_bands(np.zeros(STREAM_BINS.size), np.array([0.5]))


def test_shifted_score_runs_on_a_stream() -> None:
    rng = np.random.default_rng(1)
    noise = rng.normal(0.0, 0.1, size=3 * MODEL_SAMPLE_RATE).astype(np.float32)
    stream = push_samples(empty_stream(), noise, 60 * MODEL_SAMPLE_RATE // STREAM_HOP)
    assert stream_mean_db(stream).shape == (STREAM_BINS.size,)
    model = load_model(SHIFTED_MODEL_PATH)
    shifted = stream_shifted_score(stream, model, -math.inf)
    assert np.isfinite(shifted.score)
    assert shifted.lock_contrast is not None and np.isfinite(shifted.lock_contrast)
    assert shifted.original >= score_from_fakeprint(stream_fakeprint(stream), model) - 1e-4


def test_speed_score_finds_the_slowdown_of_model_peaks() -> None:
    model = load_model(DEFAULT_MODEL_PATH)
    freqs: NDArray[np.float64] = np.linspace(0.0, MODEL_SAMPLE_RATE / 2, N_FFT // 2 + 1)
    original: NDArray[np.float64] = np.full(BAND_BINS.size, -40.0)
    original[np.argsort(model.weights)[-40:]] += 5.0
    ratio: float = float(SPEED_RATIOS[SPEED_PEAK_COUNT // 2])
    observed: NDArray[np.float64] = np.interp(freqs[STREAM_BINS] / ratio, freqs[BAND_BINS], original)
    speed = speed_score(observed, model)
    assert abs(speed.ratio / ratio - 1.0) < 0.0005
    assert speed.score >= SPEED_LOGIT and speed.lock_contrast >= SPEED_CONTRAST


def test_lock_contrast_is_skipped_below_the_min_score() -> None:
    rng = np.random.default_rng(1)
    noise = rng.normal(0.0, 0.1, size=3 * MODEL_SAMPLE_RATE).astype(np.float32)
    stream = push_samples(empty_stream(), noise, 60 * MODEL_SAMPLE_RATE // STREAM_HOP)
    assert stream_shifted_score(stream, load_model(SHIFTED_MODEL_PATH), math.inf).lock_contrast is None
