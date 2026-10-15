"""User settings persisted as JSON; the file stores overrides on top of DEFAULT_SETTINGS."""

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from sunocheck.decision import DecisionConfig
from sunocheck.fakeprint import MODEL_SAMPLE_RATE, STREAM_HOP
from sunocheck.listener import ListenConfig, Mode

SETTINGS_PATH: Path = Path(os.environ["APPDATA"]) / "SunoCheck" / "settings.json"


@dataclass(frozen=True)
class AppSettings:
    mode: Mode
    threshold_offset: float
    confirm_seconds: float
    min_seconds: float
    max_seconds: float
    eval_interval_seconds: float
    silence_db: float
    silence_gap_seconds: float
    cooldown_seconds: float
    rhythm_enabled: bool
    record_audio: bool
    session_log_enabled: bool
    notifications_enabled: bool
    listen_on_launch: bool


DEFAULT_SETTINGS = AppSettings(
    mode=Mode.LOG,
    threshold_offset=0.0,
    confirm_seconds=1.0,
    min_seconds=1.0,
    max_seconds=60.0,
    eval_interval_seconds=STREAM_HOP / MODEL_SAMPLE_RATE,
    silence_db=-50.0,
    silence_gap_seconds=0.4,
    cooldown_seconds=3.0,
    rhythm_enabled=True,
    # Off for users: recordings keep up to max_seconds of whatever plays (calls and videos too), the session
    # log grows ~9 MB per hour of listening. Both are research tools.
    record_audio=False,
    session_log_enabled=False,
    notifications_enabled=True,
    listen_on_launch=False,
)


class SettingsFileError(ValueError):
    pass


def load_settings(path: Path) -> AppSettings:
    if not path.exists():
        return DEFAULT_SETTINGS
    stored: dict[str, str | float | int | bool] = json.loads(path.read_text(encoding="utf-8"))
    known: set[str] = {field.name for field in fields(AppSettings)}
    unknown: set[str] = set(stored) - known
    if unknown:
        raise SettingsFileError(f"Unknown keys {sorted(unknown)} in {path}; fix or delete the file")
    merged: dict[str, str | float | int | bool] = {**asdict(DEFAULT_SETTINGS), **stored}
    return AppSettings(
        mode=Mode(merged["mode"]),
        threshold_offset=float(merged["threshold_offset"]),
        confirm_seconds=float(merged["confirm_seconds"]),
        min_seconds=float(merged["min_seconds"]),
        max_seconds=float(merged["max_seconds"]),
        eval_interval_seconds=float(merged["eval_interval_seconds"]),
        silence_db=float(merged["silence_db"]),
        silence_gap_seconds=float(merged["silence_gap_seconds"]),
        cooldown_seconds=float(merged["cooldown_seconds"]),
        rhythm_enabled=bool(merged["rhythm_enabled"]),
        record_audio=bool(merged["record_audio"]),
        session_log_enabled=bool(merged["session_log_enabled"]),
        notifications_enabled=bool(merged["notifications_enabled"]),
        listen_on_launch=bool(merged["listen_on_launch"]),
    )


def save_settings(settings: AppSettings, path: Path) -> None:
    defaults: dict[str, str | float | int | bool] = asdict(DEFAULT_SETTINGS)
    overrides: dict[str, str | float | int | bool] = {k: v for k, v in asdict(settings).items() if v != defaults[k]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(overrides, ensure_ascii=False, indent=2), encoding="utf-8")


def to_listen_config(settings: AppSettings, recordings_dir: Path, blacklist: frozenset[str]) -> ListenConfig:
    return ListenConfig(
        mode=settings.mode,
        decision=DecisionConfig(threshold_offset=settings.threshold_offset, confirm_seconds=settings.confirm_seconds),
        min_seconds=settings.min_seconds,
        max_seconds=settings.max_seconds,
        eval_interval_seconds=settings.eval_interval_seconds,
        silence_db=settings.silence_db,
        silence_gap_seconds=settings.silence_gap_seconds,
        cooldown_seconds=settings.cooldown_seconds,
        rhythm_enabled=settings.rhythm_enabled,
        recordings_dir=recordings_dir if settings.record_audio else None,
        blacklist=blacklist,
    )
