"""Start with Windows through the per-user Run key; the registry value itself is the setting, so a moved exe shows as off."""

import winreg
from pathlib import Path

RUN_KEY: str = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE_NAME: str = "SunoCheck"
BACKGROUND_FLAG: str = "--background"


def autostart_command(executable: Path) -> str:
    return f'"{executable}" {BACKGROUND_FLAG}'


def read_autostart() -> str | None:
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
        try:
            value, _kind = winreg.QueryValueEx(key, VALUE_NAME)
        except FileNotFoundError:
            return None
    return str(value)


def write_autostart(command: str | None) -> None:
    """None removes the entry."""
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
        if command is not None:
            winreg.SetValueEx(key, VALUE_NAME, 0, winreg.REG_SZ, command)
        elif read_autostart() is not None:
            winreg.DeleteValue(key, VALUE_NAME)
