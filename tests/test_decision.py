import math

from sunocheck.decision import (
    LOCK_CONTRAST,
    LOCK_LOGIT,
    SHIFTED_LOGIT,
    SPEED_CONTRAST,
    SPEED_LOGIT,
    STRONG_LOGIT,
    DecisionConfig,
    Evidence,
    Phase,
    initial_evidence,
    score_threshold,
    update_evidence,
)
from sunocheck.fakeprint import ShiftedScore, SpeedScore

CONFIG: DecisionConfig = DecisionConfig(threshold_offset=0.0, confirm_seconds=1.0)
THRESHOLD: float = -6.0
QUIET: float = -20.0
QUIET_SHIFT: ShiftedScore = ShiftedScore(score=QUIET, lock_contrast=None, original=QUIET)


def run(scores: list[tuple[float, float, ShiftedScore]]) -> Evidence:
    """Feed (track_seconds, score, shifted) in order, without a slowdown search result."""
    return run_speed([(seconds, score, shifted, None) for seconds, score, shifted in scores])


def run_speed(scores: list[tuple[float, float, ShiftedScore, SpeedScore | None]]) -> Evidence:
    """Feed (track_seconds, score, shifted, speed) in order."""
    evidence: Evidence = initial_evidence()
    for seconds, score, shifted, speed in scores:
        evidence = update_evidence(evidence, score, shifted, speed, THRESHOLD, seconds, CONFIG)
    return evidence


def test_no_decision_on_short_buffers() -> None:
    assert math.isinf(score_threshold(4.9, 0.0))
    assert score_threshold(8.0, 0.0) == -9.0
    assert score_threshold(8.0, 1.5) == -7.5


def test_above_threshold_is_only_suspected() -> None:
    assert run([(10.0, THRESHOLD + 1.0, QUIET_SHIFT)]).phase is Phase.HOLD
    assert run([(10.0, THRESHOLD - 1.0, QUIET_SHIFT)]).phase is Phase.LISTEN


def test_strong_score_needs_confirm_seconds() -> None:
    assert run([(10.0, STRONG_LOGIT, QUIET_SHIFT), (10.5, STRONG_LOGIT, QUIET_SHIFT)]).phase is Phase.HOLD
    assert (
        run([(10.0, STRONG_LOGIT, QUIET_SHIFT), (10.5, STRONG_LOGIT, QUIET_SHIFT), (11.0, STRONG_LOGIT, QUIET_SHIFT)]).phase
        is Phase.DETECTED
    )


def test_a_weak_score_restarts_the_strong_run() -> None:
    evidence = run([(10.0, STRONG_LOGIT, QUIET_SHIFT), (10.5, QUIET, QUIET_SHIFT), (11.0, STRONG_LOGIT, QUIET_SHIFT)])
    assert evidence.phase is Phase.HOLD
    assert evidence.strong_since == 11.0


def test_shifted_score_counts_only_towards_the_strong_run() -> None:
    strong = ShiftedScore(score=SHIFTED_LOGIT, lock_contrast=None, original=QUIET)
    assert run([(10.0, QUIET, strong), (11.0, QUIET, strong)]).phase is Phase.DETECTED
    assert run([(10.0, QUIET, ShiftedScore(score=SHIFTED_LOGIT - 0.1, lock_contrast=None, original=QUIET))]).phase is Phase.LISTEN


def test_shifted_model_at_the_original_pitch_counts_only_towards_the_strong_run() -> None:
    strong = ShiftedScore(score=QUIET, lock_contrast=None, original=SHIFTED_LOGIT)
    assert run([(10.0, QUIET, strong), (11.0, QUIET, strong)]).phase is Phase.DETECTED
    weak = ShiftedScore(score=QUIET, lock_contrast=None, original=SHIFTED_LOGIT - 0.1)
    assert run([(10.0, QUIET, weak), (11.0, QUIET, weak)]).phase is Phase.LISTEN


def test_semitone_lock_needs_both_score_and_contrast() -> None:
    locked = ShiftedScore(score=LOCK_LOGIT, lock_contrast=LOCK_CONTRAST, original=QUIET)
    assert run([(10.0, QUIET, locked), (11.0, QUIET, locked)]).phase is Phase.DETECTED
    low_score = ShiftedScore(score=LOCK_LOGIT - 0.1, lock_contrast=LOCK_CONTRAST + 10.0, original=QUIET)
    assert run([(10.0, QUIET, low_score), (11.0, QUIET, low_score)]).phase is Phase.LISTEN
    low_contrast = ShiftedScore(score=SHIFTED_LOGIT - 0.1, lock_contrast=LOCK_CONTRAST - 0.1, original=QUIET)
    assert run([(10.0, QUIET, low_contrast), (11.0, QUIET, low_contrast)]).phase is Phase.LISTEN


def test_speed_lock_needs_both_score_and_contrast() -> None:
    locked = SpeedScore(score=SPEED_LOGIT, lock_contrast=SPEED_CONTRAST, ratio=0.914)
    assert run_speed([(10.0, QUIET, QUIET_SHIFT, locked), (11.0, QUIET, QUIET_SHIFT, locked)]).phase is Phase.DETECTED
    low_score = SpeedScore(score=SPEED_LOGIT - 0.1, lock_contrast=SPEED_CONTRAST + 10.0, ratio=0.914)
    assert run_speed([(10.0, QUIET, QUIET_SHIFT, low_score), (11.0, QUIET, QUIET_SHIFT, low_score)]).phase is Phase.LISTEN
    low_contrast = SpeedScore(score=SPEED_LOGIT + 10.0, lock_contrast=SPEED_CONTRAST - 0.1, ratio=0.914)
    assert run_speed([(10.0, QUIET, QUIET_SHIFT, low_contrast), (11.0, QUIET, QUIET_SHIFT, low_contrast)]).phase is Phase.LISTEN


def test_detected_is_final() -> None:
    detected = Evidence(phase=Phase.DETECTED, strong_since=10.0)
    assert update_evidence(detected, QUIET, QUIET_SHIFT, None, THRESHOLD, 20.0, CONFIG) == detected
