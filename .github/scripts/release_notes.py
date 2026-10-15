"""Build GitHub release notes: the CHANGELOG section of the version, checksums and VirusTotal results.

Usage: python release_notes.py <CHANGELOG.md> <version> <owner/repo> <virustotal.json> <out.md>
"""

import json
import re
import sys
from pathlib import Path


class MissingChangelogSectionError(ValueError):
    pass


def changelog_section(changelog: str, version: str) -> str:
    """Body of '## [version]' up to the next '---' separator or version header."""
    pattern: str = rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^---$|^## \[|\Z)"
    match: re.Match[str] | None = re.search(pattern, changelog, re.MULTILINE | re.DOTALL)
    if match is None or not match.group(1).strip():
        raise MissingChangelogSectionError(f"CHANGELOG.md has no '## [{version}]' section; describe the release before tagging")
    return match.group(1).strip()


def scan_row(scan: dict[str, str | int]) -> str:
    flagged: int = int(scan["malicious"]) + int(scan["suspicious"])
    return f"| `{scan['file']}` | `{scan['sha256']}` | [{flagged} / {scan['engines']}]({scan['url']}) |"


def release_notes(section: str, version: str, repo: str, scans: list[dict[str, str | int]]) -> str:
    flagged: bool = any(int(scan["malicious"]) + int(scan["suspicious"]) > 0 for scan in scans)
    false_positive_note: str = (
        "\n> Часть антивирусов помечает файлы. Так бывает с программами, собранными PyInstaller: их распаковщик похож на упаковщики"
        " вирусов. Исходный код и сборка открыты: см. workflow `release.yml` и аттестацию ниже.\n"
        if flagged
        else ""
    )
    rows: str = "\n".join(scan_row(scan) for scan in scans)
    return f"""{section}

## Скачать · Download

Windows 10/11 x64.

- **[SunoCheck-Setup.exe](https://github.com/{repo}/releases/download/v{version}/SunoCheck-Setup.exe)** — установщик, рекомендуется.
  Ставится без прав администратора в `%LOCALAPPDATA%\\Programs\\SunoCheck`, добавляет ярлык в «Пуск», удаляется через «Приложения».
  Запускается за пару секунд. Обновление — установить новую версию поверх.
- **[SunoCheck-Portable.exe](https://github.com/{repo}/releases/download/v{version}/SunoCheck-Portable.exe)** — один файл без установки.
  Каждый запуск занимает 10–20 с: файл распаковывает себя во временную папку.

Данные (записи, разметка, чёрный список) лежат в папке `data` рядом с exe, настройки — в `%APPDATA%\\SunoCheck`.

## Проверка безопасности · Security

| Файл | SHA-256 | VirusTotal (обнаружений / антивирусов) |
|---|---|---|
{rows}
{false_positive_note}
Контрольная сумма скачанного файла должна совпасть с таблицей и с `SHA256SUMS.txt`:

```powershell
(Get-FileHash .\\SunoCheck-Setup.exe -Algorithm SHA256).Hash
```

Файлы собраны GitHub Actions из тега `v{version}` этого репозитория; это подтверждает аттестация сборки:

```powershell
gh attestation verify .\\SunoCheck-Setup.exe --repo {repo}
```

Файлы не подписаны сертификатом, поэтому SmartScreen при первом запуске может показать «Windows защитила ваш компьютер»:
«Подробнее» → «Выполнить в любом случае».
"""


def main(changelog: Path, version: str, repo: str, scans_path: Path, out: Path) -> None:
    scans: list[dict[str, str | int]] = json.loads(scans_path.read_text(encoding="utf-8"))
    section: str = changelog_section(changelog.read_text(encoding="utf-8"), version)
    out.write_text(release_notes(section, version, repo, scans), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("usage: release_notes.py <CHANGELOG.md> <version> <owner/repo> <virustotal.json> <out.md>")
    main(Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4]), Path(sys.argv[5]))
