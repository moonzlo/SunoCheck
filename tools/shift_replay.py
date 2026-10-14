"""Offline replay of the live fakeprint path with the pitch-shift undo (H16) over recordings.

Same stream, scores and update_evidence as the listener (default settings), evaluated every 0.5 s of audio.
Prints per recording: max unshifted / shifted logit and the track second of DETECTED by the fakeprint paths.

Usage: uv run python tmp/shift_replay.py <wav> [...]
"""

import sys
from pathlib import Path

import soundfile as sf

from sunocheck.decision import DecisionConfig, Phase, initial_evidence, score_threshold, update_evidence
from sunocheck.fakeprint import (
    DEFAULT_MODEL_PATH,
    MODEL_SAMPLE_RATE,
    SHIFTED_MODEL_PATH,
    STREAM_HOP,
    empty_stream,
    load_model,
    push_samples,
    resample_to_model_rate,
    score_from_fakeprint,
    stream_fakeprint,
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
    print(f"{'recording':<60} {'max':>6} {'shift':>6}  detected")
    for path in paths:
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        mono = resample_to_model_rate(audio.mean(axis=1), rate)
        stream = empty_stream()
        evidence = initial_evidence()
        best, best_shifted, detected_at = float("-inf"), float("-inf"), None
        for start in range(0, mono.size - step + 1, step):
            stream = push_samples(stream, mono[start : start + step], max_frames)
            seconds = (start + step) / MODEL_SAMPLE_RATE
            if not stream.frames or seconds < DEFAULT_SETTINGS.min_seconds:
                continue
            score = score_from_fakeprint(stream_fakeprint(stream), model)
            shifted = stream_shifted_score(stream, shifted_model)
            buffered = min(seconds, DEFAULT_SETTINGS.max_seconds)
            evidence = update_evidence(evidence, score, shifted, score_threshold(buffered, config.threshold_offset), seconds, config)
            best, best_shifted = max(best, score), max(best_shifted, shifted)
            if evidence.phase is Phase.DETECTED and detected_at is None:
                detected_at = seconds
        when = "-" if detected_at is None else f"{detected_at:.1f}s"
        print(f"{path.stem[16:76]:<60} {best:+6.1f} {best_shifted:+6.1f}  {when}", flush=True)


if __name__ == "__main__":
    main([Path(p) for p in sys.argv[1:]])
