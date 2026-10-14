"""System-wide media keys via SendInput; Windows routes them to the app that owns the current media session."""

import ctypes
from ctypes import wintypes

VK_MEDIA_NEXT_TRACK: int = 0xB0
INPUT_KEYBOARD: int = 1
KEYEVENTF_EXTENDEDKEY: int = 0x0001
KEYEVENTF_KEYUP: int = 0x0002


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


# MOUSEINPUT is the largest union member; it must be present so sizeof(INPUT) matches the Win32 ABI.
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _InputUnion(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", _InputUnion)]


_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
_user32.SendInput.restype = wintypes.UINT


def _key_event(virtual_key: int, flags: int) -> INPUT:
    return INPUT(type=INPUT_KEYBOARD, union=_InputUnion(ki=KEYBDINPUT(virtual_key, 0, flags, 0, 0)))


def press_next_track() -> None:
    events = (INPUT * 2)(
        _key_event(VK_MEDIA_NEXT_TRACK, KEYEVENTF_EXTENDEDKEY),
        _key_event(VK_MEDIA_NEXT_TRACK, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP),
    )
    sent: int = _user32.SendInput(len(events), events, ctypes.sizeof(INPUT))
    if sent != len(events):
        raise OSError(ctypes.get_last_error(), f"SendInput delivered {sent} of {len(events)} key events")
