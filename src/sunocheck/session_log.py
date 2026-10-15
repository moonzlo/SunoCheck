"""CSV journals for offline comparison of approaches: every evaluation and every manual label."""

import csv
from datetime import datetime
from pathlib import Path

from sunocheck.listener import Evaluation
from sunocheck.media_session import TrackInfo

EVALUATION_HEADER: tuple[str, ...] = (
    "time",
    "app",
    "artist",
    "title",
    "track_seconds",
    "score",
    "threshold",
    "phase",
    "peak_count",
    "peak_prominence",
    "comb_strength",
    "comb_spacing_hz",
    "peak_stability",
    "bpm",
    "pulse_clarity",
    "tempo_drift_pct",
    "tempo_jump_pct",
    "grid_tightness",
    "hf_attack_ratio",
    "coherent_comb_score",
    "coherent_comb_detected",
    "shifted_score",
    "blacklisted",
    "lock_contrast",
)
LABEL_HEADER: tuple[str, ...] = ("time", "app", "artist", "title", "label", "max_score")


def append_csv_row(path: Path, header: tuple[str, ...], row: tuple[str, ...]) -> None:
    is_new: bool = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(header)
        writer.writerow(row)


def track_columns(track: TrackInfo | None) -> tuple[str, str, str]:
    if track is None:
        return ("", "", "")
    return (track.app_id, track.artist, track.title)


def evaluation_row(evaluation: Evaluation, now: datetime) -> tuple[str, ...]:
    rhythm_values: tuple[str, ...] = ("",) * 6
    if evaluation.rhythm is not None:
        r = evaluation.rhythm
        rhythm_values = tuple(
            f"{value:.4f}"
            for value in (r.bpm, r.pulse_clarity, r.tempo_drift_pct, r.tempo_jump_pct, r.grid_tightness, r.hf_attack_ratio)
        )
    s = evaluation.spectral
    return (
        now.isoformat(timespec="milliseconds"),
        *track_columns(evaluation.track),
        f"{evaluation.track_seconds:.2f}",
        f"{evaluation.score:.3f}",
        f"{evaluation.threshold:.3f}",
        evaluation.phase.value,
        str(s.peak_count),
        f"{s.peak_prominence:.3f}",
        f"{s.comb_strength:.3f}",
        f"{s.comb_spacing_hz:.1f}",
        "" if s.peak_stability is None else f"{s.peak_stability:.3f}",
        *rhythm_values,
        "" if evaluation.comb_score is None else f"{evaluation.comb_score:.3f}",
        str(int(evaluation.comb_detected)),
        f"{evaluation.shifted_score:.3f}",
        str(int(evaluation.blacklisted)),
        "" if evaluation.lock_contrast is None else f"{evaluation.lock_contrast:.3f}",
    )


def label_row(track: TrackInfo | None, label: str, max_score: float, now: datetime) -> tuple[str, ...]:
    return (now.isoformat(timespec="seconds"), *track_columns(track), label, f"{max_score:.3f}")
