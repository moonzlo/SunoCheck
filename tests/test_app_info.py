from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from sunocheck.autostart import BACKGROUND_FLAG, autostart_command
from sunocheck.gui import ISSUES_URL, TELEGRAM_URL, about_html, issue_url


def test_autostart_command_quotes_path_with_spaces() -> None:
    command: str = autostart_command(Path(r"C:\Users\Some User\Мои программы\SunoCheck.exe"))
    assert command == rf'"C:\Users\Some User\Мои программы\SunoCheck.exe" {BACKGROUND_FLAG}'


def test_issue_url_prefills_version_and_build() -> None:
    url: str = issue_url("1.2.3", "Windows-11-10.0.28000-SP0", True)
    parts = urlsplit(url)
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == ISSUES_URL
    body: str = parse_qs(parts.query)["body"][0]
    assert body.endswith("SunoCheck 1.2.3 (exe), Windows-11-10.0.28000-SP0")
    assert "data/logs/sunocheck.log" in body


def test_about_shows_version_and_contacts() -> None:
    html: str = about_html("1.2.3")
    assert "Версия 1.2.3" in html
    assert TELEGRAM_URL in html
