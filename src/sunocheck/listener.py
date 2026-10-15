"""Real-time loop: loopback audio -> per-track buffer -> detectors -> decision -> optional skip.

The loop reports everything through `emit` so the CLI and the GUI can share it.
"""

import queue
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from numpy.typing import NDArray

from sunocheck.capture import CaptureFormat, LoopbackCapture, LoopbackDeviceNotFoundError, default_output_id
from sunocheck.coherent_comb import (
    Checkpoint,
    CombState,
    SampleRateTooLowError,
    add_comb_score,
    check_sample_rate,
    comb_audio,
    comb_detected,
    comb_score,
    due_checkpoint,
    empty_comb_state,
    last_comb_score,
    push_comb_audio,
)
from sunocheck.decision import DecisionConfig, Evidence, Phase, initial_evidence, lock_min_score, score_threshold, update_evidence
from sunocheck.fakeprint import (
    MODEL_SAMPLE_RATE,
    SHIFTED_MODEL_PATH,
    STREAM_HOP,
    FakeprintModel,
    FloatArray,
    ShiftedScore,
    SpectrumStream,
    empty_stream,
    load_model,
    push_samples,
    score_from_fakeprint,
    stream_fakeprint,
    stream_shifted_score,
)
from sunocheck.media_keys import press_next_track
from sunocheck.media_session import MediaSessionWatcher, SessionUnavailable, SessionUpdate, TrackInfo, artist_key
from sunocheck.rhythm import RhythmMetrics, rhythm_metrics
from sunocheck.spectral_metrics import SpectralMetrics, recent_residue, spectral_metrics

# Short chunks keep the level meter and charts live; must be a whole number of silence frames.
CHUNK_SECONDS: float = 0.04
READ_TIMEOUT_SECONDS: float = 1.0
SILENCE_FRAME_SECONDS: float = 0.02
RHYTHM_INTERVAL_SECONDS: float = 5.0
RECORDING_MIN_SECONDS: float = 8.0
SESSION_POLL_SECONDS: float = 1.0
SESSION_TIMEOUT_SECONDS: float = 5.0
NO_SIGNAL_DBFS: float = -120.0
DEVICE_POLL_SECONDS: float = 1.0
# Opening PortAudio takes ~50 ms; while no output can be captured, retry at this pace.
REOPEN_SECONDS: float = 5.0


class Mode(StrEnum):
    LOG = "log"
    SKIP = "skip_only"
    # "skip" kept its stored value: before 0.3 the only skip mode also blacklisted the author, so old settings keep that.
    SKIP_AND_BLACKLIST = "skip"


SKIP_MODES: frozenset[Mode] = frozenset({Mode.SKIP, Mode.SKIP_AND_BLACKLIST})


@dataclass(frozen=True)
class ListenConfig:
    mode: Mode
    decision: DecisionConfig
    min_seconds: float
    max_seconds: float
    eval_interval_seconds: float
    silence_db: float
    silence_gap_seconds: float
    cooldown_seconds: float
    rhythm_enabled: bool
    recordings_dir: Path | None
    blacklist: frozenset[str]


@dataclass(frozen=True)
class TrackState:
    """Per-track buffers; sample counts are at MODEL_SAMPLE_RATE except `native_*` and the comb (capture rate).
    `serial` changes with every new track so late background results can be matched to their track."""

    serial: int
    info: TrackInfo | None
    chunks: tuple[FloatArray, ...]
    native_chunks: tuple[FloatArray, ...]
    native_samples: int
    spectrum: SpectrumStream
    comb: CombState
    samples: int
    total_samples: int
    samples_since_eval: int
    samples_since_rhythm: int
    evidence: Evidence
    max_score: float


@dataclass(frozen=True)
class CaptureStarted:
    device_name: str
    sample_rate: int
    channels: int


@dataclass(frozen=True)
class CapturePaused:
    """The default output cannot be captured; capture reopens when the output changes or on the next retry."""

    reason: str


@dataclass(frozen=True)
class ActiveCapture:
    capture: LoopbackCapture
    rate: int
    resampler: soxr.ResampleStream


@dataclass(frozen=True)
class TrackChanged:
    track: TrackInfo | None


@dataclass(frozen=True)
class TrackReset:
    reason: str


@dataclass(frozen=True)
class LevelUpdate:
    dbfs: float


@dataclass(frozen=True)
class Evaluation:
    track: TrackInfo | None
    phase: Phase
    score: float
    shifted_score: float
    lock_contrast: float | None
    threshold: float
    max_score: float
    track_seconds: float
    buffered_seconds: float
    fakeprint: FloatArray
    recent_residue: FloatArray
    spectral: SpectralMetrics
    rhythm: RhythmMetrics | None
    comb_score: float | None
    comb_detected: bool
    blacklisted: bool


@dataclass(frozen=True)
class CombScored:
    """Comb score of the first `checkpoint.seconds` of loud track audio; `detected`: any checkpoint so far fired."""

    track: TrackInfo | None
    checkpoint: Checkpoint
    score: float
    detected: bool


@dataclass(frozen=True)
class CombJob:
    serial: int
    checkpoint: Checkpoint
    future: "Future[float]"

@dataclass(frozen=True)
class SkipPerformed:
    track: TrackInfo | None


@dataclass(frozen=True)
class RecordingSaved:
    path: Path


@dataclass(frozen=True)
class CaptureWarning:
    message: str


ListenerEvent = (
    CaptureStarted
    | CapturePaused
    | TrackChanged
    | TrackReset
    | LevelUpdate
    | Evaluation
    | CombScored
    | SkipPerformed
    | RecordingSaved
    | CaptureWarning
)
Emit = Callable[[ListenerEvent], None]


def new_track(info: TrackInfo | None, serial: int) -> TrackState:
    return TrackState(
        serial=serial,
        info=info,
        chunks=(),
        native_chunks=(),
        native_samples=0,
        spectrum=empty_stream(),
        comb=empty_comb_state(),
        samples=0,
        total_samples=0,
        samples_since_eval=0,
        samples_since_rhythm=0,
        evidence=initial_evidence(),
        max_score=float("-inf"),
    )


def append_audio(track: TrackState, chunk: FloatArray, max_samples: int) -> TrackState:
    chunks: tuple[FloatArray, ...] = track.chunks + (chunk,)
    samples: int = track.samples + chunk.size
    while samples - chunks[0].size >= max_samples:
        samples -= chunks[0].size
        chunks = chunks[1:]
    return replace(
        track,
        chunks=chunks,
        spectrum=push_samples(track.spectrum, chunk, max_samples // STREAM_HOP),
        samples=samples,
        total_samples=track.total_samples + chunk.size,
        samples_since_eval=track.samples_since_eval + chunk.size,
        samples_since_rhythm=track.samples_since_rhythm + chunk.size,
    )


def append_native(track: TrackState, chunk: FloatArray, max_samples: int) -> TrackState:
    """Keep the last `max_samples` of capture-rate audio for the per-track recording."""
    chunks: tuple[FloatArray, ...] = track.native_chunks + (chunk,)
    samples: int = track.native_samples + chunk.size
    while samples - chunks[0].size >= max_samples:
        samples -= chunks[0].size
        chunks = chunks[1:]
    return replace(track, native_chunks=chunks, native_samples=samples)


def is_due_for_evaluation(track: TrackState, min_samples: int, interval_samples: int) -> bool:
    return bool(track.spectrum.frames) and track.samples >= min_samples and track.samples_since_eval >= interval_samples

def rms_dbfs(chunk: FloatArray) -> float:
    rms: float = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
    return float(20.0 * np.log10(max(rms, 1e-10)))


def silence_runs(chunk: FloatArray, sample_rate: int, silence_db: float, carried_run: float) -> tuple[float, float]:
    """Longest silent run inside the chunk (continuing `carried_run`) and the silent run at its end, in seconds."""
    frame: int = int(SILENCE_FRAME_SECONDS * sample_rate)
    frames: FloatArray = chunk[: chunk.size // frame * frame].reshape(-1, frame)
    power: NDArray[np.float64] = np.mean(np.square(frames, dtype=np.float64), axis=1)
    silent: NDArray[np.bool_] = 10.0 * np.log10(np.maximum(power, 1e-20)) < silence_db
    run: float = carried_run
    longest: float = carried_run
    for is_silent in silent:
        run = run + SILENCE_FRAME_SECONDS if is_silent else 0.0
        longest = max(longest, run)
    return longest, run


def describe(track_info: TrackInfo | None) -> str:
    if track_info is None:
        return "-"
    return f"{track_info.artist} - {track_info.title}"


def latest_config(current: ListenConfig, updates: "queue.Queue[ListenConfig]") -> ListenConfig:
    config: ListenConfig = current
    while True:
        try:
            config = updates.get_nowait()
        except queue.Empty:
            return config


def safe_track_name(track_info: TrackInfo | None) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", describe(track_info))[:80]


def recording_name(track: TrackState, now: datetime) -> str:
    return f"{now:%Y%m%d_%H%M%S}_{safe_track_name(track.info)}_s{track.max_score:.1f}.wav"


def finish_track(track: TrackState, config: ListenConfig, sample_rate: int, emit: Emit) -> None:
    """Save the capture-rate buffer: the comb needs lines up to 15.8 kHz, so offline checks need the full rate."""
    if config.recordings_dir is None or track.samples < int(RECORDING_MIN_SECONDS * MODEL_SAMPLE_RATE):
        return
    config.recordings_dir.mkdir(parents=True, exist_ok=True)
    path: Path = config.recordings_dir / recording_name(track, datetime.now())
    sf.write(path, np.concatenate(track.native_chunks), sample_rate, subtype="PCM_16")
    emit(RecordingSaved(path=path))


def is_blacklisted(track: TrackState, config: ListenConfig, skipped: TrackInfo | None) -> bool:
    """`skipped`: the last skipped track. The media session may still report it after the skip, and
    a metadata-only verdict must not skip the next track on stale metadata."""
    return track.info is not None and track.info != skipped and artist_key(track.info.artist) in config.blacklist


def evaluate(
    track: TrackState,
    config: ListenConfig,
    model: FakeprintModel,
    shifted_model: FakeprintModel,
    with_rhythm: bool,
    skipped: TrackInfo | None,
) -> tuple[Evaluation, Evidence]:
    fakeprint: FloatArray = stream_fakeprint(track.spectrum)
    score: float = score_from_fakeprint(fakeprint, model)
    shifted: ShiftedScore = stream_shifted_score(track.spectrum, shifted_model, lock_min_score(config.decision.threshold_offset))
    buffered_seconds: float = track.samples / MODEL_SAMPLE_RATE
    track_seconds: float = track.total_samples / MODEL_SAMPLE_RATE
    threshold: float = score_threshold(buffered_seconds, config.decision.threshold_offset)
    evidence: Evidence = update_evidence(track.evidence, score, shifted, threshold, track_seconds, config.decision)
    blacklisted: bool = is_blacklisted(track, config, skipped)
    if comb_detected(track.comb) or blacklisted:
        evidence = Evidence(phase=Phase.DETECTED, strong_since=evidence.strong_since)
    evaluation = Evaluation(
        track=track.info,
        phase=evidence.phase,
        score=score,
        shifted_score=shifted.score,
        lock_contrast=shifted.lock_contrast,
        threshold=threshold,
        max_score=max(track.max_score, score),
        track_seconds=track_seconds,
        buffered_seconds=buffered_seconds,
        fakeprint=fakeprint,
        recent_residue=recent_residue(track.spectrum),
        spectral=spectral_metrics(fakeprint, track.spectrum),
        rhythm=rhythm_metrics(np.concatenate(track.chunks), MODEL_SAMPLE_RATE) if with_rhythm else None,
        comb_score=last_comb_score(track.comb),
        comb_detected=comb_detected(track.comb),
        blacklisted=blacklisted,
    )
    return evaluation, evidence


def open_capture() -> tuple[ActiveCapture, CaptureStarted] | CapturePaused:
    """Loopback of the current default output, or why it cannot be captured (no loopback, too low a rate,
    a PortAudio error) so the loop can wait for another output instead of stopping."""
    capture = LoopbackCapture(chunk_seconds=CHUNK_SECONDS)
    try:
        capture_format: CaptureFormat = capture.start()
        check_sample_rate(capture_format.sample_rate)
    except (LoopbackDeviceNotFoundError, SampleRateTooLowError, OSError) as error:
        capture.close()
        return CapturePaused(str(error))
    rate: int = capture_format.sample_rate
    resampler = soxr.ResampleStream(rate, MODEL_SAMPLE_RATE, 1, dtype="float32", quality="HQ")
    return ActiveCapture(capture, rate, resampler), CaptureStarted(capture_format.device_name, rate, capture_format.channels)


def run_listener(
    initial_config: ListenConfig,
    config_updates: "queue.Queue[ListenConfig]",
    model: FakeprintModel,
    emit: Emit,
    stop: threading.Event,
) -> None:
    session_watcher = MediaSessionWatcher(poll_seconds=SESSION_POLL_SECONDS, timeout_seconds=SESSION_TIMEOUT_SECONDS)
    session_watcher.start()

    config: ListenConfig = initial_config
    track: TrackState = new_track(None, 0)
    active: ActiveCapture | None = None
    paused: CapturePaused | None = None
    device_id: str | None = None
    next_device_check: float = 0.0
    next_open_attempt: float = 0.0
    silence_run_seconds: float = 0.0
    ignore_until: float = 0.0
    reported_overflows: int = 0
    # Long comb FFTs (up to 0.5 s) run beside the loop so scores and the level meter keep flowing.
    comb_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="comb")
    comb_job: CombJob | None = None
    shifted_model: FakeprintModel = load_model(SHIFTED_MODEL_PATH)
    skipped: TrackInfo | None = None

    try:
        while not stop.is_set():
            config = latest_config(config, config_updates)
            now: float = time.monotonic()

            if comb_job is not None and comb_job.future.done():
                if comb_job.serial == track.serial:
                    score: float = comb_job.future.result()
                    track = replace(track, comb=add_comb_score(track.comb, score))
                    emit(CombScored(track.info, comb_job.checkpoint, score, comb_detected(track.comb)))
                comb_job = None

            while not session_watcher.events.empty():
                match session_watcher.events.get_nowait():
                    case SessionUpdate(track=current) if current != track.info:
                        # Without a capture the track holds no audio: it was finished when the capture closed.
                        if active is not None:
                            finish_track(track, config, active.rate, emit)
                        track = new_track(current, track.serial + 1)
                        emit(TrackChanged(current))
                    case SessionUpdate():
                        pass
                    case SessionUnavailable(message=message):
                        emit(CaptureWarning(f"media session unavailable, track boundaries by silence only: {message}"))

            if now >= next_device_check:
                next_device_check = now + DEVICE_POLL_SECONDS
                current_device: str = default_output_id()
                device_changed: bool = current_device != device_id
                stream_lost: bool = active is not None and not active.capture.is_active()
                retry_due: bool = active is None and now >= next_open_attempt
                if device_changed or stream_lost or retry_due:
                    if active is not None:
                        active.capture.close()
                        finish_track(track, config, active.rate, emit)
                        track = new_track(track.info, track.serial + 1)
                        emit(TrackReset("output device changed" if device_changed else "capture stream stopped"))
                    device_id = current_device
                    next_open_attempt = now + REOPEN_SECONDS
                    reported_overflows = 0
                    silence_run_seconds = 0.0
                    match open_capture():
                        case (ActiveCapture() as opened, CaptureStarted() as started):
                            active, paused = opened, None
                            emit(started)
                        case CapturePaused() as failure:
                            active = None
                            if failure != paused:
                                emit(failure)
                            paused = failure

            if active is None:
                emit(LevelUpdate(NO_SIGNAL_DBFS))
                stop.wait(DEVICE_POLL_SECONDS)
                continue
            rate: int = active.rate

            if active.capture.overflow_count != reported_overflows:
                reported_overflows = active.capture.overflow_count
                emit(CaptureWarning(f"capture overflow #{reported_overflows}: audio was dropped"))

            chunk: FloatArray | None = active.capture.read(timeout_s=READ_TIMEOUT_SECONDS)
            level: float = NO_SIGNAL_DBFS if chunk is None else rms_dbfs(chunk)
            emit(LevelUpdate(level))
            if now < ignore_until:
                continue

            if chunk is None:
                silence_run_seconds += READ_TIMEOUT_SECONDS
                longest_silence: float = silence_run_seconds
            else:
                longest_silence, silence_run_seconds = silence_runs(chunk, rate, config.silence_db, silence_run_seconds)
            if longest_silence >= config.silence_gap_seconds and track.samples > 0:
                finish_track(track, config, rate, emit)
                track = new_track(track.info, track.serial + 1)
                emit(TrackReset("silence"))
            if chunk is None:
                continue
            chunk_model_rate: FloatArray = active.resampler.resample_chunk(chunk)
            if level < config.silence_db or chunk_model_rate.size == 0:
                continue

            track = append_audio(track, chunk_model_rate, int(config.max_seconds * MODEL_SAMPLE_RATE))
            track = append_native(track, chunk, int(config.max_seconds * rate))
            track = replace(track, comb=push_comb_audio(track.comb, chunk, rate))
            checkpoint: Checkpoint | None = due_checkpoint(track.comb, rate)
            if comb_job is None and checkpoint is not None:
                audio: FloatArray = comb_audio(track.comb, checkpoint, rate)
                comb_job = CombJob(track.serial, checkpoint, comb_pool.submit(comb_score, audio, rate))
            interval_samples: int = int(config.eval_interval_seconds * MODEL_SAMPLE_RATE)
            if not is_due_for_evaluation(track, int(config.min_seconds * MODEL_SAMPLE_RATE), interval_samples):
                continue

            was_detected: bool = track.evidence.phase is Phase.DETECTED
            with_rhythm: bool = config.rhythm_enabled and track.samples_since_rhythm >= int(RHYTHM_INTERVAL_SECONDS * MODEL_SAMPLE_RATE)
            evaluation, evidence = evaluate(track, config, model, shifted_model, with_rhythm, skipped)
            track = replace(
                track,
                evidence=evidence,
                samples_since_eval=track.samples_since_eval % interval_samples,
                samples_since_rhythm=0 if with_rhythm else track.samples_since_rhythm,
                max_score=evaluation.max_score,
            )
            emit(evaluation)

            if evaluation.phase is Phase.DETECTED and not was_detected and config.mode in SKIP_MODES:
                press_next_track()
                skipped = track.info
                emit(SkipPerformed(track.info))
                finish_track(track, config, rate, emit)
                track = new_track(track.info, track.serial + 1)
                ignore_until = now + config.cooldown_seconds
    finally:
        comb_pool.shutdown(wait=False, cancel_futures=True)
        session_watcher.stop()
        if active is not None:
            active.capture.close()
            finish_track(track, config, active.rate, emit)
