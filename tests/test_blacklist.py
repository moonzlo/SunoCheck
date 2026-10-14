from datetime import datetime
from pathlib import Path

import pytest

from sunocheck.blacklist import BlacklistAction, BlacklistSource, MissingArtistError, blacklist_entries, change_blacklist
from sunocheck.listener import is_blacklisted, new_track
from sunocheck.media_session import TrackInfo
from sunocheck.settings import DEFAULT_SETTINGS, to_listen_config

NOW: datetime = datetime(2026, 10, 14, 23, 0, 0)
RAVEN: TrackInfo = TrackInfo(app_id="app", artist="RAVENSTORMX", title="Stand When I'm Afraid")
BILLIE: TrackInfo = TrackInfo(app_id="app", artist="Billie Eilish", title="No Time To Die")


def test_missing_journal_is_an_empty_list(tmp_path: Path) -> None:
    assert blacklist_entries(tmp_path / "blacklist.csv") == {}


def test_latest_event_of_an_artist_wins(tmp_path: Path) -> None:
    path: Path = tmp_path / "blacklist.csv"
    assert change_blacklist(path, "RavenStormX", "a", BlacklistAction.ADD, BlacklistSource.SKIP, NOW)
    assert change_blacklist(path, "Billie Eilish", "b", BlacklistAction.ADD, BlacklistSource.AI_LABEL, NOW)
    assert change_blacklist(path, " billie eilish ", "c", BlacklistAction.REMOVE, BlacklistSource.HUMAN_LABEL, NOW)
    entries = blacklist_entries(path)
    assert set(entries) == {"ravenstormx"}
    assert entries["ravenstormx"].source is BlacklistSource.SKIP


def test_no_op_changes_write_nothing(tmp_path: Path) -> None:
    path: Path = tmp_path / "blacklist.csv"
    assert change_blacklist(path, "RavenStormX", "a", BlacklistAction.ADD, BlacklistSource.SKIP, NOW)
    assert not change_blacklist(path, "ravenstormx", "b", BlacklistAction.ADD, BlacklistSource.BUTTON, NOW)
    assert not change_blacklist(path, "MAVIRA", "c", BlacklistAction.REMOVE, BlacklistSource.REMOVE_BUTTON, NOW)
    assert len(path.read_text(encoding="utf-8-sig").splitlines()) == 2


def test_empty_artist_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(MissingArtistError):
        change_blacklist(tmp_path / "blacklist.csv", "  ", "a", BlacklistAction.ADD, BlacklistSource.BUTTON, NOW)


@pytest.mark.parametrize(
    ("info", "skipped", "expected"),
    [
        (RAVEN, None, True),
        (RAVEN, RAVEN, False),
        (RAVEN, BILLIE, True),
        (BILLIE, None, False),
        (None, None, False),
    ],
)
def test_blacklisted_track_and_post_skip_guard(info: TrackInfo | None, skipped: TrackInfo | None, expected: bool) -> None:
    config = to_listen_config(DEFAULT_SETTINGS, Path("."), frozenset({"ravenstormx"}))
    assert is_blacklisted(new_track(info, 1), config, skipped) is expected
