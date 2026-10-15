import json
from dataclasses import replace
from pathlib import Path

import pytest

from sunocheck.listener import Mode
from sunocheck.settings import DEFAULT_SETTINGS, SettingsFileError, load_settings, save_settings


def test_missing_file_gives_defaults(tmp_path: Path) -> None:
    assert load_settings(tmp_path / "settings.json") == DEFAULT_SETTINGS


def test_only_overrides_are_stored_and_read_back(tmp_path: Path) -> None:
    path: Path = tmp_path / "settings.json"
    settings = replace(DEFAULT_SETTINGS, mode=Mode.SKIP, threshold_offset=1.5)
    save_settings(settings, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"mode": "skip_only", "threshold_offset": 1.5}
    assert load_settings(path) == settings


def test_old_skip_value_keeps_blacklisting(tmp_path: Path) -> None:
    path: Path = tmp_path / "settings.json"
    path.write_text(json.dumps({"mode": "skip"}), encoding="utf-8")
    assert load_settings(path).mode is Mode.SKIP_AND_BLACKLIST


def test_unknown_key_is_an_error(tmp_path: Path) -> None:
    path: Path = tmp_path / "settings.json"
    path.write_text(json.dumps({"no_such_setting": 1}), encoding="utf-8")
    with pytest.raises(SettingsFileError):
        load_settings(path)
