"""Timeline of one track from a session CSV: every ~2.5 s of track time and every phase change.

Usage: uv run python tmp/session_track.py <session.csv> <artist substring>
"""

import csv
import sys
from pathlib import Path


def main(path: Path, artist: str) -> None:
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8-sig")) if artist in r["artist"]]
    last_seconds: float | None = None
    last_phase: str | None = None
    for row in rows:
        seconds = float(row["track_seconds"])
        if last_seconds is None or seconds - last_seconds >= 2.5 or row["phase"] != last_phase:
            print(
                f"{seconds:6.1f} {row['phase']:<8} score {float(row['score']):+6.1f} thr {row['threshold']:>7}"
                f" shifted {float(row['shifted_score']):+6.1f}"
            )
            last_seconds = seconds
        last_phase = row["phase"]


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
