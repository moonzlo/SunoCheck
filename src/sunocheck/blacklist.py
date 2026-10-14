"""Artist blacklist: a track by a blacklisted artist is DETECTED on its first evaluation (skipped in SKIP mode).

The CSV is an append-only journal of add/remove events; the latest event of an artist wins.
"""

import csv
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from sunocheck.media_session import artist_key
from sunocheck.session_log import append_csv_row

BLACKLIST_HEADER: tuple[str, ...] = ("time", "artist", "action", "source", "title")


class BlacklistAction(StrEnum):
    ADD = "add"
    REMOVE = "remove"


class BlacklistSource(StrEnum):
    BUTTON = "button"
    AI_LABEL = "ai_label"
    SKIP = "skip"
    HUMAN_LABEL = "human_label"
    REMOVE_BUTTON = "remove_button"


@dataclass(frozen=True)
class BlacklistEntry:
    artist: str
    title: str
    source: BlacklistSource
    added: str


class MissingArtistError(ValueError):
    pass


def blacklist_entries(path: Path) -> dict[str, BlacklistEntry]:
    """Blacklisted artists by `artist_key`; no journal yet means an empty list."""
    if not path.exists():
        return {}
    entries: dict[str, BlacklistEntry] = {}
    with path.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key: str = artist_key(row["artist"])
            if BlacklistAction(row["action"]) is BlacklistAction.ADD:
                entries[key] = BlacklistEntry(row["artist"], row["title"], BlacklistSource(row["source"]), row["time"])
            else:
                entries.pop(key, None)
    return entries


def blacklist_row(
    entries: dict[str, BlacklistEntry], artist: str, title: str, action: BlacklistAction, source: BlacklistSource, now: datetime
) -> tuple[str, ...] | None:
    """Journal row for the change, or None if the artist is already in the requested state."""
    key: str = artist_key(artist)
    if not key:
        raise MissingArtistError(f"Track '{title}' has no artist: nothing to put in the blacklist")
    if (key in entries) == (action is BlacklistAction.ADD):
        return None
    return (now.isoformat(timespec="seconds"), artist.strip(), action.value, source.value, title)


def change_blacklist(path: Path, artist: str, title: str, action: BlacklistAction, source: BlacklistSource, now: datetime) -> bool:
    """Append the change to the journal; False if nothing changed."""
    row: tuple[str, ...] | None = blacklist_row(blacklist_entries(path), artist, title, action, source, now)
    if row is None:
        return False
    append_csv_row(path, BLACKLIST_HEADER, row)
    return True
