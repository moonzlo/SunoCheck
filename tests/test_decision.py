import math

from sunocheck.decision import (
    SHIFTED_LOGIT,
    STRONG_LOGIT,
    DecisionConfig,
    Evidence,
    Phase,
    initial_evidence,
    score_threshold,
    update_evidence,
)

CONFIG: DecisionConfig = DecisionConfig(threshold_offset=0.0, confirm_seconds=1.0)
THRESHOLD: float = -6.0
QUIET: float = -20.0


def run(scores: list[tuple[float, float, float]]) -> Evidence:
    """Feed (track_seconds, score, shifted_score) in order."""
    evidence: Evidence = initial_evidence()
    for seconds, score, shifted in scores:
        evidence = update_evidence(evidence, score, shifted, THRESHOLD, seconds, CONFIG)
    return evidence


def test_no_decision_on_short_buffers() -> None:
    assert math.isinf(score_threshold(4.9, 0.0))
    assert score_threshold(8.0, 0.0) == -9.0
    assert score_threshold(8.0, 1.5) == -7.5


def test_above_threshold_is_only_suspected() -> None:
    assert run([(10.0, THRESHOLD + 1.0, QUIET)]).phase is Phase.HOLD
    assert run([(10.0, THRESHOLD - 1.0, QUIET)]).phase is Phase.LISTEN


def test_strong_score_needs_confirm_seconds() -> None:
    assert run([(10.0, STRONG_LOGIT, QUIET), (10.5, STRONG_LOGIT, QUIET)]).phase is Phase.HOLD
    assert run([(10.0, STRONG_LOGIT, QUIET), (10.5, STRONG_LOGIT, QUIET), (11.0, STRONG_LOGIT, QUIET)]).phase is Phase.DETECTED


def test_a_weak_score_restarts_the_strong_run() -> None:
    evidence = run([(10.0, STRONG_LOGIT, QUIET), (10.5, QUIET, QUIET), (11.0, STRONG_LOGIT, QUIET)])
    assert evidence.phase is Phase.HOLD
    assert evidence.strong_since == 11.0


def test_shifted_score_counts_only_towards_the_strong_run() -> None:
    assert run([(10.0, QUIET, SHIFTED_LOGIT), (11.0, QUIET, SHIFTED_LOGIT)]).phase is Phase.DETECTED
    assert run([(10.0, QUIET, SHIFTED_LOGIT - 0.1)]).phase is Phase.LISTEN


def test_detected_is_final() -> None:
    detected = Evidence(phase=Phase.DETECTED, strong_since=10.0)
    assert update_evidence(detected, QUIET, QUIET, THRESHOLD, 20.0, CONFIG) == detected
