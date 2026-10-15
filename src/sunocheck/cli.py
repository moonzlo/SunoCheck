import argparse
import queue
import threading
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import soundfile as sf

from sunocheck.blacklist import BlacklistAction, BlacklistSource, blacklist_entries, change_blacklist
from sunocheck.coherent_comb import CHECKPOINTS, Checkpoint, check_sample_rate, comb_score
from sunocheck.fakeprint import DEFAULT_MODEL_PATH, FakeprintModel, FloatArray, ai_probability, load_model
from sunocheck.listener import (
    CapturePaused,
    CaptureStarted,
    CaptureWarning,
    CombScored,
    Evaluation,
    LevelUpdate,
    ListenConfig,
    ListenerEvent,
    Mode,
    RecordingSaved,
    SkipPerformed,
    TrackChanged,
    TrackReset,
    describe,
    run_listener,
)
from sunocheck.media_session import TrackInfo, artist_key
from sunocheck.settings import DEFAULT_SETTINGS, AppSettings, to_listen_config
from sunocheck.study import Label, LabeledAudio, labeled_recordings, study_report

AUDIO_EXTENSIONS: frozenset[str] = frozenset({".wav", ".flac", ".ogg", ".mp3"})
MAX_ANALYZE_SECONDS: int = 300
RECORDINGS_DIR: Path = Path.cwd() / "data" / "recordings"
LABELS_PATH: Path = Path.cwd() / "data" / "labels.csv"
BLACKLIST_PATH: Path = Path.cwd() / "data" / "blacklist.csv"


def collect_audio_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"No such file or directory: {path}")
    return files


def analyze_file(path: Path, model: FakeprintModel) -> tuple[float, list[tuple[Checkpoint, float]]]:
    """Fakeprint probability and the comb score of the file's first seconds at each comb checkpoint."""
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    rate: int = int(sample_rate)
    mono: FloatArray = data[: MAX_ANALYZE_SECONDS * rate].mean(axis=1)
    check_sample_rate(rate)
    combs: list[tuple[Checkpoint, float]] = [
        (c, comb_score(mono[: int(c.seconds * rate)], rate)) for c in CHECKPOINTS if c.seconds * rate <= mono.size
    ]
    return ai_probability(mono, rate, model), combs


def run_analyze(paths: list[Path], model: FakeprintModel) -> None:
    for path in collect_audio_files(paths):
        try:
            probability, combs = analyze_file(path, model)
        except sf.LibsndfileError as error:
            print(f"ERROR  {path.name}: {error}")
            continue
        comb_text: str = " ".join(f"{c.seconds:g}s:{score:+.1f}{'!' if score >= c.threshold_db else ''}" for c, score in combs)
        print(f"{probability:6.3f}  comb[{comb_text}] (! = above threshold)  {path.name}")


def format_event(event: ListenerEvent) -> str | None:
    match event:
        case CaptureStarted(device_name=name, sample_rate=rate, channels=channels):
            return f"capture started: '{name}', {rate} Hz, {channels} ch"
        case CapturePaused(reason=reason):
            return f"PAUSED   default output cannot be captured, waiting for another one: {reason}"
        case TrackChanged(track=track):
            return f"TRACK {describe(track)}"
        case TrackReset(reason=reason):
            return f"RESET ({reason})"
        case LevelUpdate():
            return None
        case Evaluation():
            return (
                f"{event.phase.value:<8} score={event.score:6.1f} shifted={event.shifted_score:6.1f}"
                f" thr={event.threshold:6.1f} t={event.track_seconds:5.2f}s"
                f"{' blacklisted' if event.blacklisted else ''} | {describe(event.track)}"
            )
        case CombScored(track=track, checkpoint=checkpoint, score=score, detected=detected):
            return (
                f"COMB     {checkpoint.seconds:4.1f}s score={score:+6.2f} thr={checkpoint.threshold_db:+.2f}"
                f"{' DETECTED' if detected else ''} | {describe(track)}"
            )
        case SkipPerformed(track=track):
            return f"ACTION next track | {describe(track)}"
        case RecordingSaved(path=path):
            return f"RECORDED {path.name}"
        case CaptureWarning(message=message):
            return f"WARNING {message}"


def print_event(event: ListenerEvent) -> None:
    line: str | None = format_event(event)
    if line is not None:
        print(f"[{datetime.now():%H:%M:%S}] {line}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    defaults: AppSettings = DEFAULT_SETTINGS
    parser = argparse.ArgumentParser(
        prog="sunocheck", description="SunoCheck - AI music detection: detects Suno-generated music by its audio fingerprint"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser("analyze", help="score audio files or folders")
    analyze.add_argument("paths", nargs="+", type=Path)

    study = commands.add_parser("study", help="rank detector metrics by AUROC on labeled audio")
    study.add_argument("--ai", nargs="*", type=Path, default=[], help="files or folders of AI tracks")
    study.add_argument("--human", nargs="*", type=Path, default=[], help="files or folders of human tracks")
    study.add_argument("--labeled", action="store_true", help=f"add GUI recordings labeled in {LABELS_PATH}")

    commands.add_parser("gui", help="open the window with live charts and settings")

    listen = commands.add_parser("listen", help="score system audio in real time (console)")
    listen.add_argument("--mode", type=Mode, choices=list(Mode), default=defaults.mode)
    listen.add_argument(
        "--threshold-offset", type=float, default=defaults.threshold_offset, help="shift of the score threshold curve (+ = stricter)"
    )
    listen.add_argument(
        "--confirm-seconds", type=float, default=defaults.confirm_seconds, help="track seconds of strong fakeprint scores needed to flag"
    )
    listen.add_argument("--min-seconds", type=float, default=defaults.min_seconds, help="audio needed before the first score")
    listen.add_argument("--max-seconds", type=float, default=defaults.max_seconds, help="rolling buffer length per track")
    listen.add_argument("--eval-interval", type=float, default=defaults.eval_interval_seconds, help="seconds of new audio between scores")
    listen.add_argument("--silence-db", type=float, default=defaults.silence_db, help="chunks below this RMS dBFS count as silence")
    listen.add_argument("--silence-gap", type=float, default=defaults.silence_gap_seconds, help="silence that marks a track boundary, s")
    listen.add_argument("--cooldown", type=float, default=defaults.cooldown_seconds, help="seconds to ignore audio after a skip")
    listen.add_argument("--rhythm", action="store_true", help="compute experimental rhythm metrics")
    listen.add_argument("--record", action="store_true", help=f"save each track's buffer as WAV into {RECORDINGS_DIR}")
    return parser


def main() -> None:
    args: argparse.Namespace = build_parser().parse_args()
    if args.command == "gui":
        from sunocheck.gui import main as gui_main

        gui_main()
        return
    model: FakeprintModel = load_model(DEFAULT_MODEL_PATH)
    if args.command == "analyze":
        run_analyze(args.paths, model)
        return
    if args.command == "study":
        items: list[LabeledAudio] = [LabeledAudio(Label.AI, p) for p in collect_audio_files(args.ai)]
        items += [LabeledAudio(Label.HUMAN, p) for p in collect_audio_files(args.human)]
        if args.labeled:
            items += labeled_recordings(RECORDINGS_DIR, LABELS_PATH)
        print(study_report(items, model))
        return
    settings = AppSettings(
        mode=args.mode,
        threshold_offset=args.threshold_offset,
        confirm_seconds=args.confirm_seconds,
        min_seconds=args.min_seconds,
        max_seconds=args.max_seconds,
        eval_interval_seconds=args.eval_interval,
        silence_db=args.silence_db,
        silence_gap_seconds=args.silence_gap,
        cooldown_seconds=args.cooldown,
        rhythm_enabled=args.rhythm,
        record_audio=args.record,
        session_log_enabled=DEFAULT_SETTINGS.session_log_enabled,
        notifications_enabled=DEFAULT_SETTINGS.notifications_enabled,
        listen_on_launch=DEFAULT_SETTINGS.listen_on_launch,
    )
    config: ListenConfig = to_listen_config(settings, RECORDINGS_DIR, frozenset(blacklist_entries(BLACKLIST_PATH)))
    updates: queue.Queue[ListenConfig] = queue.Queue()

    def on_event(event: ListenerEvent) -> None:
        print_event(event)
        if not isinstance(event, SkipPerformed):
            return
        track: TrackInfo | None = event.track
        if track is None or not artist_key(track.artist):
            print("BLACKLIST skipped track has no artist: not added", flush=True)
        elif change_blacklist(BLACKLIST_PATH, track.artist, track.title, BlacklistAction.ADD, BlacklistSource.SKIP, datetime.now()):
            print(f"BLACKLIST + {track.artist}", flush=True)
            updates.put(replace(config, blacklist=frozenset(blacklist_entries(BLACKLIST_PATH))))

    run_listener(config, updates, model, on_event, threading.Event())


if __name__ == "__main__":
    main()
