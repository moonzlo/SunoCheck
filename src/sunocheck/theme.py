"""Dark theme of the control panel: color tokens, one application stylesheet and the Windows 11 Mica backdrop.

Effects are drawn by the Windows compositor (DWM) or applied once per state change; nothing animates continuously."""

import ctypes
import sys
from ctypes import wintypes
from pathlib import Path
from string import Template

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

ASSETS_DIR: Path = Path(__file__).parent / "assets"
FONT_FAMILIES: list[str] = ["Segoe UI Variable Text", "Segoe UI"]

TEXT: str = "#f4f4f5"
TEXT_SECONDARY: str = "#a1a1aa"
TEXT_MUTED: str = "#71717a"
WINDOW: str = "#1b1b1f"
POPUP: str = "#26262b"
SURFACE: str = "rgba(255, 255, 255, 11)"
SURFACE_BORDER: str = "rgba(255, 255, 255, 20)"
CONTROL: str = "rgba(255, 255, 255, 18)"
CONTROL_HOVER: str = "rgba(255, 255, 255, 30)"
CONTROL_PRESSED: str = "rgba(255, 255, 255, 10)"
CONTROL_BORDER: str = "rgba(255, 255, 255, 26)"
# Taken from the logo gradient: cyan -> blue -> violet.
ACCENT: str = "#4f7dff"
ACCENT_HOVER: str = "#6a91ff"
ACCENT_CYAN: str = "#22d3ee"
ACCENT_VIOLET: str = "#8b5cf6"

# One spacing grid for every layout: window edge, gap between cards, padding inside a card.
OUTER_MARGIN: int = 16
GAP: int = 12
CARD_PADDING: int = 16
GROUP_TITLE_SPACE: int = 40
FORM_ROW_SPACING: int = 12
FORM_COLUMN_SPACING: int = 16

# DWMWA_SYSTEMBACKDROP_TYPE exists since Windows 11 22H2; the immersive dark title bar since Windows 10 20H1.
MICA_MIN_BUILD: int = 22621
DARK_TITLE_MIN_BUILD: int = 19041
DWMWA_USE_IMMERSIVE_DARK_MODE: int = 20
DWMWA_SYSTEMBACKDROP_TYPE: int = 38
DWMSBT_MAINWINDOW: int = 2

STYLE_SHEET_TEMPLATE: Template = Template("""
QWidget { color: $text; }
QMainWindow, QDialog { background: $window; }
QToolTip { background: $popup; color: $text; border: 1px solid $control_border; padding: 4px 8px; }

QMenuBar { background: transparent; padding: 2px 4px; }
QMenuBar::item { background: transparent; padding: 6px 10px; border-radius: 6px; }
QMenuBar::item:selected { background: $control_hover; }
QMenu { background: $popup; border: 1px solid $control_border; }
QMenu::item:selected { background: $control_hover; }
QMenu::separator { height: 1px; background: $control_border; margin: 4px 8px; }

QFrame#card, QGroupBox { background: $surface; border: 1px solid $surface_border; border-radius: 12px; }
QFrame#sidePanel { background: transparent; border: none; }
QGroupBox { padding-top: ${group_title}px; }
QGroupBox::title { subcontrol-origin: padding; subcontrol-position: top left; left: ${card_padding}px; top: 14px;
    color: $text_secondary; }
QLabel#caption { color: $text_muted; font-size: 11px; }
QLabel#section { color: $text_secondary; padding-top: 4px; qproperty-indent: 0; }
QLabel#secondary { color: $text_secondary; font-size: 12px; }
QLabel#metricName { color: $text_muted; font-size: 11px; }
QLabel#metricValue { color: $text_secondary; font-size: 12px; }
QLabel#trackTitle { font-size: 15px; font-weight: 600; }
QLabel#statusChip { color: white; font-size: 13px; font-weight: 600; padding: 0 14px;
    border-radius: 16px; min-width: 88px; }

QPushButton { background: $control; border: 1px solid $control_border; border-radius: 6px; padding: 8px 16px; min-height: 22px; }
QPushButton:hover { background: $control_hover; }
QPushButton:pressed { background: $control_pressed; }
QPushButton:disabled { color: $text_muted; }
QPushButton#primary { background: $accent; border: 1px solid $accent; color: white; font-weight: 600; }
QPushButton#primary:hover { background: $accent_hover; border-color: $accent_hover; }
QPushButton#primary:disabled { background: $control; border-color: $control_border; color: $text_muted; }
QToolButton { background: transparent; border: none; border-radius: 6px; padding: 4px 6px; }
QToolButton:hover { background: $control_hover; }
QToolButton#collapse { background: $control; border: 1px solid $control_border; border-radius: 6px; padding: 0;
    min-width: 28px; max-width: 28px; min-height: 28px; max-height: 28px; }
QToolButton#collapse:hover { background: $control_hover; }
QToolButton#sidebarTongue { background: $surface; border: 1px solid $surface_border; border-left: none;
    border-top-left-radius: 0; border-bottom-left-radius: 0; border-top-right-radius: 9px; border-bottom-right-radius: 9px;
    padding: 0; margin: 0; }
QToolButton#sidebarTongue:hover { background: $control_hover; }
QScrollArea#sideScroll, QScrollArea#advancedScroll { background: transparent; border: none; }
QScrollArea#sideScroll > QWidget > QWidget, QScrollArea#advancedScroll > QWidget > QWidget { background: transparent; }
QScrollArea#sideScroll QScrollBar:vertical { background: transparent; width: 6px; margin: 10px 2px 10px 0; border: none; }
QScrollArea#sideScroll QScrollBar::handle:vertical { background: rgba(255, 255, 255, 40); border-radius: 3px; min-height: 24px; }
QScrollArea#sideScroll QScrollBar::add-line:vertical, QScrollArea#sideScroll QScrollBar::sub-line:vertical { height: 0; width: 0; }
QScrollArea#sideScroll QScrollBar::add-page:vertical, QScrollArea#sideScroll QScrollBar::sub-page:vertical { background: transparent; }

QComboBox, QAbstractSpinBox { background: $control; border: 1px solid $control_border; border-radius: 6px; padding: 6px 12px;
    min-height: 24px; }
QAbstractSpinBox { padding-right: 34px; }
QComboBox:hover, QAbstractSpinBox:hover { background: $control_hover; }
QComboBox:focus, QAbstractSpinBox:focus { border: 1px solid $accent; }
QComboBox::drop-down { border: none; width: 26px; }
QComboBox::down-arrow { image: url($chevron_down); width: 10px; height: 10px; }
QComboBox QAbstractItemView { background: $popup; border: 1px solid $control_border; selection-background-color: $control_hover;
    outline: 0; padding: 4px; }
QComboBox QAbstractItemView::item { padding: 6px 10px; min-height: 22px; border-radius: 4px; }
QComboBox QAbstractItemView::item:selected { background: $control_hover; color: $text; }
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button { subcontrol-origin: border; width: 26px; border: none;
    background: transparent; margin: 2px 4px 2px 0; }
QAbstractSpinBox::up-button { subcontrol-position: top right; }
QAbstractSpinBox::down-button { subcontrol-position: bottom right; }
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover { background: $control_hover; border-radius: 4px; }
QAbstractSpinBox::up-arrow { image: url($chevron_up); width: 9px; height: 9px; }
QAbstractSpinBox::down-arrow { image: url($chevron_down); width: 9px; height: 9px; }

QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid rgba(255, 255, 255, 90); border-radius: 4px; background: $control; }
QCheckBox::indicator:hover { border-color: $text_secondary; }
QCheckBox::indicator:checked { background: $accent; border-color: $accent; image: url($check); }

QTabWidget::pane { background: $surface; border: 1px solid $surface_border; border-radius: 12px; top: -1px; }
QTabWidget::tab-bar { left: 10px; }
QTabBar::tab { background: transparent; color: $text_secondary; padding: 8px 14px; border: none; border-bottom: 2px solid transparent; }
QTabBar::tab:hover { color: $text; }
QTabBar::tab:selected { color: $text; border-bottom: 2px solid $accent; }

QPlainTextEdit, QListWidget { background: rgba(0, 0, 0, 50); border: 1px solid $surface_border; border-radius: 8px; padding: 6px;
    selection-background-color: $accent; }
QListWidget::item { padding: 4px 6px; border-radius: 4px; }
QListWidget::item:selected { background: rgba(79, 125, 255, 90); color: $text; }

QProgressBar { background: $control; border: none; border-radius: 3px; max-height: 6px; }
QProgressBar::chunk { border-radius: 3px;
    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, stop: 0 $accent_cyan, stop: 1 $accent_violet); }

QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
QScrollBar::handle { background: rgba(255, 255, 255, 45); border-radius: 3px; min-height: 24px; min-width: 24px; }
QScrollBar::handle:hover { background: rgba(255, 255, 255, 80); }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

QGraphicsView { background: transparent; border: none; }
""")


class WindowBackdropError(OSError):
    pass


def asset_url(name: str) -> str:
    path: Path = ASSETS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Theme asset not found: {path}")
    return path.as_posix()


def app_style_sheet() -> str:
    return STYLE_SHEET_TEMPLATE.substitute(
        text=TEXT,
        text_secondary=TEXT_SECONDARY,
        text_muted=TEXT_MUTED,
        window=WINDOW,
        popup=POPUP,
        surface=SURFACE,
        surface_border=SURFACE_BORDER,
        control=CONTROL,
        control_hover=CONTROL_HOVER,
        control_pressed=CONTROL_PRESSED,
        control_border=CONTROL_BORDER,
        accent=ACCENT,
        accent_hover=ACCENT_HOVER,
        accent_cyan=ACCENT_CYAN,
        accent_violet=ACCENT_VIOLET,
        group_title=GROUP_TITLE_SPACE,
        card_padding=CARD_PADDING,
        chevron_up=asset_url("chevron-up.svg"),
        chevron_down=asset_url("chevron-down.svg"),
        check=asset_url("check.svg"),
    )


def apply_theme(app: QApplication) -> None:
    """Must run before any widget or plot is created: pyqtgraph reads its defaults at construction."""
    app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
    app.setStyle("Fusion")
    font: QFont = app.font()
    font.setFamilies(FONT_FAMILIES)
    app.setFont(font)
    app.setStyleSheet(app_style_sheet())
    pg.setConfigOptions(antialias=True, background=None, foreground=TEXT_SECONDARY)


def mica_supported() -> bool:
    return sys.getwindowsversion().build >= MICA_MIN_BUILD


def set_dwm_attribute(hwnd: int, attribute: int, value: int) -> None:
    data = ctypes.c_int(value)
    result: int = ctypes.windll.dwmapi.DwmSetWindowAttribute(wintypes.HWND(hwnd), attribute, ctypes.byref(data), ctypes.sizeof(data))
    if result != 0:
        raise WindowBackdropError(f"DwmSetWindowAttribute({attribute}) failed: HRESULT {result & 0xFFFFFFFF:#010x}")


class Margins(ctypes.Structure):
    _fields_ = [("left", ctypes.c_int), ("right", ctypes.c_int), ("top", ctypes.c_int), ("bottom", ctypes.c_int)]


def apply_window_backdrop(hwnd: int, mica: bool) -> None:
    """Dark title bar, and with Mica the backdrop under the whole client area (the window must be translucent)."""
    if sys.getwindowsversion().build >= DARK_TITLE_MIN_BUILD:
        set_dwm_attribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, 1)
    if not mica:
        return
    set_dwm_attribute(hwnd, DWMWA_SYSTEMBACKDROP_TYPE, DWMSBT_MAINWINDOW)
    margins = Margins(-1, -1, -1, -1)
    result: int = ctypes.windll.dwmapi.DwmExtendFrameIntoClientArea(wintypes.HWND(hwnd), ctypes.byref(margins))
    if result != 0:
        raise WindowBackdropError(f"DwmExtendFrameIntoClientArea failed: HRESULT {result & 0xFFFFFFFF:#010x}")
