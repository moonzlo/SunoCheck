"""Experimental rhythm descriptors for testing the hypothesis that Suno timing differs from human music.

Display/logging only: none of these values is calibrated or used for the skip decision yet.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from sunocheck.fakeprint import FloatArray

FloatVector = NDArray[np.float64]

HOP_SECONDS: float = 0.01
N_FFT: int = 1024
HF_SPLIT_HZ: float = 4000.0
MIN_BPM: float = 60.0
MAX_BPM: float = 180.0
WINDOW_SECONDS: float = 8.0
WINDOW_HOP_SECONDS: float = 4.0
LOCAL_LAG_TOLERANCE: float = 0.15
GRID_DIVISIONS: int = 4
GRID_SEARCH_SPAN: float = 0.03
GRID_SEARCH_STEPS: int = 121
MIN_ONSETS_PER_WINDOW: int = 4
MIN_ONSET_GAP_SECONDS: float = 0.05


@dataclass(frozen=True)
class RhythmMetrics:
    bpm: float
    pulse_clarity: float
    tempo_drift_pct: float
    tempo_jump_pct: float
    grid_tightness: float
    hf_attack_ratio: float


@dataclass(frozen=True)
class OnsetEnvelope:
    full: FloatVector
    high: FloatVector
    frame_rate: float


def onset_envelope(audio: FloatArray, sample_rate: int) -> OnsetEnvelope:
    """Positive log-spectral flux, total and above HF_SPLIT_HZ."""
    hop: int = int(round(sample_rate * HOP_SECONDS))
    if audio.size < N_FFT + hop:
        raise ValueError(f"Need at least {N_FFT + hop} samples, got {audio.size}")
    frames: FloatArray = np.lib.stride_tricks.sliding_window_view(audio, N_FFT)[::hop]
    window: FloatArray = np.hanning(N_FFT).astype(np.float32)
    log_magnitude: FloatVector = np.log1p(100.0 * np.abs(np.fft.rfft(frames * window, axis=1))).astype(np.float64)
    flux: FloatVector = np.maximum(np.diff(log_magnitude, axis=0), 0.0)
    freqs: FloatVector = np.fft.rfftfreq(N_FFT, 1.0 / sample_rate)
    return OnsetEnvelope(
        full=flux.sum(axis=1),
        high=flux[:, freqs >= HF_SPLIT_HZ].sum(axis=1),
        frame_rate=sample_rate / hop,
    )


def normalized_autocorrelation(signal: FloatVector) -> FloatVector | None:
    """None when the signal has no variation (no rhythmic content)."""
    centered: FloatVector = signal - signal.mean()
    spectrum = np.fft.rfft(centered, 2 * centered.size)
    autocorr: FloatVector = np.fft.irfft(np.abs(spectrum) ** 2)[: centered.size]
    if autocorr[0] <= 1e-12:
        return None
    return autocorr / autocorr[0]


def peak_lag(autocorr: FloatVector, lo: int, hi: int) -> float:
    """Highest autocorrelation lag in [lo, hi], refined by parabolic interpolation.

    The parabola is only valid at a true local maximum (negative curvature, offset within ±0.5); at a range
    edge on a slope a near-zero curvature would throw the lag thousands of frames away."""
    index: int = lo + int(np.argmax(autocorr[lo : hi + 1]))
    if index <= 0 or index >= autocorr.size - 1:
        return float(index)
    left, center, right = autocorr[index - 1], autocorr[index], autocorr[index + 1]
    curvature: float = float(left - 2.0 * center + right)
    if curvature >= 0.0 or left > center or right > center:
        return float(index)
    return index + 0.5 * float(left - right) / curvature


def onset_times(envelope: FloatVector, frame_rate: float) -> FloatVector:
    peaks, _ = find_peaks(
        envelope,
        height=float(envelope.mean() + envelope.std()),
        distance=max(1, int(MIN_ONSET_GAP_SECONDS * frame_rate)),
    )
    return peaks / frame_rate


def grid_tightness(times: FloatVector, beat_seconds: float) -> float:
    """Best mean resultant length of onset phases on a GRID_DIVISIONS-per-beat grid (1 = perfectly on grid)."""
    base: float = beat_seconds / GRID_DIVISIONS
    periods: FloatVector = base * (1.0 + np.linspace(-GRID_SEARCH_SPAN, GRID_SEARCH_SPAN, GRID_SEARCH_STEPS))
    phases: FloatVector = 2.0 * np.pi * times[:, None] / periods[None, :]
    return float(np.abs(np.exp(1j * phases).mean(axis=0)).max())


def rhythm_metrics(audio: FloatArray, sample_rate: int) -> RhythmMetrics | None:
    """None when the clip has no usable pulse (e.g. ambient, silence) or is too short for two windows."""
    envelope: OnsetEnvelope = onset_envelope(audio, sample_rate)
    frame_rate: float = envelope.frame_rate
    global_autocorr: FloatVector | None = normalized_autocorrelation(envelope.full)
    if global_autocorr is None:
        return None
    lag_lo: int = int(frame_rate * 60.0 / MAX_BPM)
    lag_hi: int = min(int(frame_rate * 60.0 / MIN_BPM), global_autocorr.size - 2)
    if lag_hi <= lag_lo:
        return None
    lag: float = peak_lag(global_autocorr, lag_lo, lag_hi)
    bpm: float = 60.0 * frame_rate / lag

    window: int = int(WINDOW_SECONDS * frame_rate)
    hop: int = int(WINDOW_HOP_SECONDS * frame_rate)
    local_bpms: list[float] = []
    tightness: list[float] = []
    for start in range(0, envelope.full.size - window + 1, hop):
        segment: FloatVector = envelope.full[start : start + window]
        local_autocorr: FloatVector | None = normalized_autocorrelation(segment)
        if local_autocorr is None:
            continue
        local_lag: float = peak_lag(
            local_autocorr,
            int(lag * (1.0 - LOCAL_LAG_TOLERANCE)),
            min(int(np.ceil(lag * (1.0 + LOCAL_LAG_TOLERANCE))), local_autocorr.size - 2),
        )
        local_bpms.append(60.0 * frame_rate / local_lag)
        times: FloatVector = onset_times(segment, frame_rate)
        if times.size >= MIN_ONSETS_PER_WINDOW:
            tightness.append(grid_tightness(times, local_lag / frame_rate))
    if len(local_bpms) < 2 or not tightness:
        return None

    local: FloatVector = np.asarray(local_bpms)
    peaks: NDArray[np.int64] = (onset_times(envelope.full, frame_rate) * frame_rate).astype(np.int64)
    attack_total: float = float(envelope.full[peaks].sum())
    return RhythmMetrics(
        bpm=bpm,
        pulse_clarity=float(global_autocorr[int(round(lag))]),
        tempo_drift_pct=float(local.std() / bpm * 100.0),
        tempo_jump_pct=float(np.abs(np.diff(local)).max() / bpm * 100.0),
        grid_tightness=float(np.mean(tightness)),
        hf_attack_ratio=float(envelope.high[peaks].sum()) / attack_total if attack_total > 0.0 else 0.0,
    )
