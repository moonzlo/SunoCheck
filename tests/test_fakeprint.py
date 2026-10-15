import math
from pathlib import Path

import numpy as np
import pytest

from sunocheck.fakeprint import (
    BAND_BINS,
    DEFAULT_MODEL_PATH,
    MODEL_SAMPLE_RATE,
    SHIFTED_MODEL_PATH,
    STREAM_BINS,
    STREAM_HOP,
    empty_stream,
    fakeprint_from_band,
    load_model,
    push_samples,
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
    shifted = stream_shifted_score(stream, load_model(SHIFTED_MODEL_PATH), -math.inf)
    assert np.isfinite(shifted.score)
    assert shifted.lock_contrast is not None and np.isfinite(shifted.lock_contrast)


def test_lock_contrast_is_skipped_below_the_min_score() -> None:
    rng = np.random.default_rng(1)
    noise = rng.normal(0.0, 0.1, size=3 * MODEL_SAMPLE_RATE).astype(np.float32)
    stream = push_samples(empty_stream(), noise, 60 * MODEL_SAMPLE_RATE // STREAM_HOP)
    assert stream_shifted_score(stream, load_model(SHIFTED_MODEL_PATH), math.inf).lock_contrast is None
