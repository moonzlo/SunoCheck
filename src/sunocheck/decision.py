"""Per-track evidence accumulation of the fakeprint logit.

Above the threshold curve a track is only suspected (HOLD): live, dense modern human pop scored up to +3.6
there (Billie Eilish, Море Дождей). It is flagged once scores stay at or above STRONG_LOGIT (live Suno:
+12...+23) for `confirm_seconds` of track time, or the shifted-Suno model score stays at or above
SHIFTED_LOGIT (research H18); otherwise only the coherent comb can flag it (see listener)."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

# Fakeprint logit threshold vs. seconds of buffered audio. Logits of all music grow as the average
# gets longer, so short buffers need a lower bar. No decision before the first point: on 1-4 s
# buffers sparse tonal intros of human tracks score like Suno (live Yandex Music session), while
# real Suno tracks start low and rise.
THRESHOLD_CURVE: tuple[tuple[float, float], ...] = ((5.0, -12.0), (8.0, -9.0), (30.0, -6.0), (60.0, -5.0))
STRONG_LOGIT: float = 10.0
# Shifted-Suno model: human max +6.2 on the corpus (grouped CV), live humans <= +4.6, live evasions +10...+29.
SHIFTED_LOGIT: float = 8.0


class Phase(StrEnum):
    LISTEN = "LISTEN"
    HOLD = "HOLD"
    DETECTED = "DETECTED"


@dataclass(frozen=True)
class DecisionConfig:
    threshold_offset: float
    confirm_seconds: float


@dataclass(frozen=True)
class Evidence:
    """`strong_since`: track time when the current run of strong scores began."""

    phase: Phase
    strong_since: float | None


def initial_evidence() -> Evidence:
    return Evidence(phase=Phase.LISTEN, strong_since=None)


def score_threshold(buffered_seconds: float, threshold_offset: float) -> float:
    if buffered_seconds < THRESHOLD_CURVE[0][0]:
        return float("inf")
    seconds: list[float] = [point[0] for point in THRESHOLD_CURVE]
    logits: list[float] = [point[1] for point in THRESHOLD_CURVE]
    return float(np.interp(buffered_seconds, seconds, logits)) + threshold_offset


def update_evidence(
    evidence: Evidence, score: float, shifted_score: float, threshold: float, track_seconds: float, config: DecisionConfig
) -> Evidence:
    """`shifted_score` (shifted-Suno model) only counts towards the strong run, never towards HOLD."""
    if evidence.phase is Phase.DETECTED:
        return evidence
    strong: bool = score >= max(threshold, STRONG_LOGIT + config.threshold_offset) or shifted_score >= max(
        threshold, SHIFTED_LOGIT + config.threshold_offset
    )
    if not strong:
        return Evidence(phase=Phase.HOLD if score >= threshold else Phase.LISTEN, strong_since=None)
    since: float = track_seconds if evidence.strong_since is None else evidence.strong_since
    phase: Phase = Phase.DETECTED if track_seconds - since >= config.confirm_seconds else Phase.HOLD
    return Evidence(phase=phase, strong_since=since)
