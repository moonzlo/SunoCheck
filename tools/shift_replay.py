"""Offline replay of the live fakeprint path with the pitch-shift undo (H16) and the slowdown search (H23).

Same stream, scores and update_evidence as the listener (default settings), evaluated every 0.5 s of audio; the
slowdown search runs on every step from 5 s of buffer, as the listener's SPEED_INTERVAL_SECONDS.
Prints per recording: max lofcz / shifted / shifted-model-at-r=1 / slowdown logit and the track second of DETECTED by
the fakeprint paths.

Usage: uv run python tmp/shift_replay.py <wav> [...]
"""

import sys
from pathlib import Path

import soundfile as sf

from sunocheck.decision import (
    THRESHOLD_CURVE,
    DecisionConfig,
    Phase,
    initial_evidence,
    lock_min_score,
    score_threshold,
    update_evidence,
)
from sunocheck.fakeprint import (
    DEFAULT_MODEL_PATH,
    MODEL_SAMPLE_RATE,
    SHIFTED_MODEL_PATH,
    STREAM_HOP,
    SpeedScore,
    empty_stream,
    load_model,
    push_samples,
    resample_to_model_rate,
    score_from_fakeprint,
    speed_score,
    stream_fakeprint,
    stream_mean_db,
    stream_shifted_score,
)
from sunocheck.settings import DEFAULT_SETTINGS

STEP_SECONDS: float = 0.5


def main(paths: list[Path]) -> None:
    model = load_model(DEFAULT_MODEL_PATH)
    shifted_model = load_model(SHIFTED_MODEL_PATH)
    config = DecisionConfig(DEFAULT_SETTINGS.threshold_offset, DEFAULT_SETTINGS.confirm_seconds)
    max_frames = int(DEFAULT_SETTINGS.max_seconds * MODEL_SAMPLE_RATE) // STREAM_HOP
    step = int(STEP_SECONDS * MODEL_SAMPLE_RATE)
    print(f"{'recording':<60} {'max':>6} {'shift':>6} {'r=1':>6} {'speed':>6}  detected")
    for path in paths:
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        mono = resample_to_model_rate(audio.mean(axis=1), rate)
        stream = empty_stream()
        evidence = initial_evidence()
        speed: SpeedScore | None = None
        best, best_shifted, best_original, best_speed = float("-inf"), float("-inf"), float("-inf"), float("-inf")
        detected_at: float | None = None
        for start in range(0, mono.size - step + 1, step):
            stream = push_samples(stream, mono[start : start + step], max_frames)
            seconds = (start + step) / MODEL_SAMPLE_RATE
            if not stream.frames or seconds < DEFAULT_SETTINGS.min_seconds:
                continue
            score = score_from_fakeprint(stream_fakeprint(stream), model)
            shifted = stream_shifted_score(stream, shifted_model, lock_min_score(config.threshold_offset))
            buffered = min(seconds, DEFAULT_SETTINGS.max_seconds)
            if buffered >= THRESHOLD_CURVE[0][0] and evidence.phase is not Phase.DETECTED:
                speed = speed_score(stream_mean_db(stream), model)
                best_speed = max(best_speed, speed.score)
            threshold = score_threshold(buffered, config.threshold_offset)
            evidence = update_evidence(evidence, score, shifted, speed, threshold, seconds, config)
            best, best_shifted, best_original = max(best, score), max(best_shifted, shifted.score), max(best_original, shifted.original)
            if evidence.phase is Phase.DETECTED and detected_at is None:
                detected_at = seconds
        when = "-" if detected_at is None else f"{detected_at:.1f}s"
        print(f"{path.stem[16:76]:<60} {best:+6.1f} {best_shifted:+6.1f} {best_original:+6.1f} {best_speed:+6.1f}  {when}", flush=True)


if __name__ == "__main__":
    main([Path(p) for p in sys.argv[1:]])
