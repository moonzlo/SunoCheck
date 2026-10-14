"""Offline comparison of detectors: stream labeled audio through the live engine and rank every metric by AUROC."""

import csv
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from sunocheck.fakeprint import (
    MODEL_SAMPLE_RATE,
    STREAM_HOP,
    FakeprintModel,
    FloatArray,
    empty_stream,
    push_samples,
    resample_to_model_rate,
    score_from_fakeprint,
    stream_fakeprint,
)
from sunocheck.listener import safe_track_name
from sunocheck.media_session import TrackInfo
from sunocheck.spectral_metrics import spectral_metrics

CHECKPOINTS_SECONDS: tuple[float, ...] = (5.0, 10.0, 20.0, 30.0)
CHUNK_SECONDS: float = 0.25
MAX_BUFFER_SECONDS: float = 60.0


class Label(StrEnum):
    AI = "ai"
    HUMAN = "human"


@dataclass(frozen=True)
class LabeledAudio:
    label: Label
    path: Path


@dataclass(frozen=True)
class CheckpointMetrics:
    score: float
    peak_count: float
    peak_prominence: float
    comb_strength: float
    comb_spacing_hz: float
    peak_stability: float | None


METRIC_NAMES: tuple[str, ...] = tuple(field.name for field in fields(CheckpointMetrics))


class StudyInputError(ValueError):
    pass


def labeled_recordings(recordings_dir: Path, labels_path: Path) -> list[LabeledAudio]:
    """Recordings whose track name matches a row in labels.csv; the last label of a track wins."""
    if not labels_path.exists():
        raise StudyInputError(f"No labels file {labels_path}: label tracks in the GUI first")
    labels: dict[str, Label] = {}
    with labels_path.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            labels[safe_track_name(TrackInfo(app_id=row["app"], artist=row["artist"], title=row["title"]))] = Label(row["label"])
    items: list[LabeledAudio] = []
    for path in sorted(recordings_dir.glob("*.wav")):
        matches: list[Label] = [label for name, label in labels.items() if f"_{name}_s" in path.name]
        if matches:
            items.append(LabeledAudio(label=matches[0], path=path))
    return items


def checkpoint_metrics(audio: FloatArray, model: FakeprintModel) -> dict[float, CheckpointMetrics]:
    chunk: int = int(CHUNK_SECONDS * MODEL_SAMPLE_RATE)
    max_frames: int = int(MAX_BUFFER_SECONDS * MODEL_SAMPLE_RATE) // STREAM_HOP
    stream = empty_stream()
    results: dict[float, CheckpointMetrics] = {}
    pending: list[float] = list(CHECKPOINTS_SECONDS)
    for start in range(0, audio.size, chunk):
        stream = push_samples(stream, audio[start : start + chunk], max_frames)
        seconds: float = (start + chunk) / MODEL_SAMPLE_RATE
        while pending and seconds >= pending[0]:
            fakeprint: FloatArray = stream_fakeprint(stream)
            spectral = spectral_metrics(fakeprint, stream)
            results[pending.pop(0)] = CheckpointMetrics(
                score=score_from_fakeprint(fakeprint, model),
                peak_count=float(spectral.peak_count),
                peak_prominence=spectral.peak_prominence,
                comb_strength=spectral.comb_strength,
                comb_spacing_hz=spectral.comb_spacing_hz,
                peak_stability=spectral.peak_stability,
            )
        if not pending:
            break
    return results


def load_model_rate(path: Path) -> FloatArray:
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True, frames=int((max(CHECKPOINTS_SECONDS) + 1) * 48000))
    return resample_to_model_rate(data.mean(axis=1), int(sample_rate))


def auroc(positive: NDArray[np.float64], negative: NDArray[np.float64]) -> float:
    greater: float = float((positive[:, None] > negative[None, :]).mean())
    ties: float = float((positive[:, None] == negative[None, :]).mean())
    return greater + 0.5 * ties


def study_report(items: list[LabeledAudio], model: FakeprintModel) -> str:
    if not any(item.label is Label.AI for item in items) or not any(item.label is Label.HUMAN for item in items):
        raise StudyInputError(
            f"Need both AI and human examples, got {sum(i.label is Label.AI for i in items)} AI"
            f" and {sum(i.label is Label.HUMAN for i in items)} human"
        )
    rows: list[tuple[LabeledAudio, dict[float, CheckpointMetrics]]] = [
        (item, checkpoint_metrics(load_model_rate(item.path), model)) for item in items
    ]
    lines: list[str] = [
        "AUROC (1.0 = AI always higher, 0.0 = AI always lower, 0.5 = useless); n = AI/human with that checkpoint",
        f"{'metric':<18}" + "".join(f"{f'{s:.0f}s':>16}" for s in CHECKPOINTS_SECONDS),
    ]
    for name in METRIC_NAMES:
        cells: list[str] = []
        for seconds in CHECKPOINTS_SECONDS:
            values: dict[Label, list[float]] = {Label.AI: [], Label.HUMAN: []}
            for item, metrics in rows:
                value: float | None = getattr(metrics[seconds], name) if seconds in metrics else None
                if value is not None:
                    values[item.label].append(value)
            if values[Label.AI] and values[Label.HUMAN]:
                area: float = auroc(np.array(values[Label.AI]), np.array(values[Label.HUMAN]))
                cells.append(f"{area:.3f} ({len(values[Label.AI])}/{len(values[Label.HUMAN])})")
            else:
                cells.append("-")
        lines.append(f"{name:<18}" + "".join(f"{cell:>16}" for cell in cells))
    return "\n".join(lines)
