"""SunoCheck PySide6 front-end: a tray icon for background use and a control panel with both detectors, charts,
settings and manual labels. The detector runs in a worker thread and reports through a queue."""

import ctypes
import getpass
import math
import os
import platform
import queue
import sys
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from importlib import metadata
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QBrush,
    QCloseEvent,
    QColor,
    QDesktopServices,
    QIcon,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
)
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSystemTrayIcon,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sunocheck.autostart import BACKGROUND_FLAG, autostart_command, read_autostart, write_autostart
from sunocheck.blacklist import (
    BlacklistAction,
    BlacklistEntry,
    BlacklistSource,
    MissingArtistError,
    blacklist_entries,
    change_blacklist,
)
from sunocheck.coherent_comb import CHECKPOINTS as COMB_CHECKPOINTS
from sunocheck.coherent_comb import Checkpoint
from sunocheck.decision import SHIFTED_LOGIT, STRONG_LOGIT, Phase
from sunocheck.fakeprint import DEFAULT_MODEL_PATH, FakeprintModel, FloatArray, load_model
from sunocheck.listener import (
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
from sunocheck.rhythm import RhythmMetrics
from sunocheck.session_log import EVALUATION_HEADER, LABEL_HEADER, append_csv_row, evaluation_row, label_row
from sunocheck.settings import SETTINGS_PATH, AppSettings, load_settings, save_settings, to_listen_config
from sunocheck.spectral_metrics import SpectralMetrics
from sunocheck.study import LabeledAudio, StudyInputError, labeled_recordings, study_report

APP_ID: str = "SunoCheck"
APP_NAME: str = "SunoCheck"
APP_TAGLINE: str = "AI music detection"
APP_VERSION: str = metadata.version("sunocheck")
APP_AUTHOR: str = "Mikhail (moonzlo)"
REPO_URL: str = "https://github.com/moonzlo/SunoCheck"
README_URL: str = f"{REPO_URL}#readme"
RELEASES_URL: str = f"{REPO_URL}/releases/latest"
ISSUES_URL: str = f"{REPO_URL}/issues/new"
TELEGRAM_URL: str = "https://t.me/moonzlo"
LOGO_PATH: Path = Path(__file__).parent / "assets" / "sunocheck.png"
FROZEN: bool = getattr(sys, "frozen", False)
# The release .exe is portable: its data lives next to the executable, whatever folder it was started from.
DATA_DIR: Path = (Path(sys.executable).parent if FROZEN else Path.cwd()) / "data"
RECORDINGS_DIR: Path = DATA_DIR / "recordings"
SESSIONS_DIR: Path = DATA_DIR / "sessions"
LABELS_PATH: Path = DATA_DIR / "labels.csv"
BLACKLIST_PATH: Path = DATA_DIR / "blacklist.csv"

EVENT_POLL_MS: int = 15
MAX_LOG_LINES: int = 500
LEVEL_FLOOR_DB: float = -60.0
NOTIFY_MS: int = 6000
NOTIFY_WIDTH: int = 360
NOTIFY_MARGIN: int = 16
# Named pipe per Windows user: a second launch only raises the running window, so two instances never skip twice.
INSTANCE_SERVER: str = f"{APP_ID}-{getpass.getuser()}"
INSTANCE_TIMEOUT_MS: int = 500
ASFW_ANY: int = -1
ABOUT_LOGO_PX: int = 96
HEADER_CONTROL_HEIGHT: int = 36
SCORE_AXIS: tuple[float, float] = (-40.0, 10.0)
COMB_AXIS: tuple[float, float] = (-10.0, 4.0)
SPECTROGRAM_COLUMNS: int = 600
SCORE_SPAN_STEP_SECONDS: float = 30.0
SPECTROGRAM_HEIGHT: int = 128
# Dark blue -> purple -> orange -> pale yellow, indexed by residue 0..255.
SPECTROGRAM_LUT: NDArray[np.uint8] = np.stack(
    [np.interp(np.arange(256), [0, 64, 160, 255], channel) for channel in ([0, 90, 230, 255], [0, 20, 110, 250], [20, 120, 30, 200])],
    axis=1,
).astype(np.uint8)

STATUS_COLORS: dict[str, str] = {
    "STOPPED": "#616161",
    "LISTEN": "#2e7d32",
    "HOLD": "#f9a825",
    "DETECTED": "#c62828",
    "SILENCE": "#1565c0",
}
STATUS_TEXT: dict[str, str] = {
    "STOPPED": "Остановлен",
    "LISTEN": "Слушаю",
    "HOLD": "Подозрение",
    "DETECTED": "Suno",
    "SILENCE": "Тишина",
}
PHASE_COLORS: dict[Phase, str] = {Phase.LISTEN: "#2e7d32", Phase.HOLD: "#f9a825", Phase.DETECTED: "#c62828"}
BLACKLIST_SOURCE_TEXT: dict[BlacklistSource, str] = {
    BlacklistSource.BUTTON: "кнопка «В чёрный список»",
    BlacklistSource.AI_LABEL: "кнопка «Это ИИ»",
    BlacklistSource.SKIP: "автоскип",
    BlacklistSource.HUMAN_LABEL: "кнопка «Это человек»",
    BlacklistSource.REMOVE_BUTTON: "кнопка «Убрать из списка»",
}
COMB_HIT_COLOR: str = "#c62828"
COMB_MISS_COLOR: str = "#9e9e9e"


@dataclass(frozen=True)
class NumericField:
    key: str
    label: str
    minimum: float
    maximum: float
    step: float
    is_int: bool
    advanced: bool


NUMERIC_FIELDS: tuple[NumericField, ...] = (
    NumericField("threshold_offset", "Сдвиг порога (+ строже)", -10, 10, 0.5, False, False),
    NumericField("confirm_seconds", "Подтверждение, с", 0, 10, 0.25, False, False),
    NumericField("min_seconds", "Первая оценка через, с", 0.5, 90, 0.5, False, True),
    NumericField("eval_interval_seconds", "Интервал оценки, с", 0.064, 15, 0.064, False, True),
    NumericField("max_seconds", "Буфер трека, с", 20, 180, 10, False, True),
    NumericField("silence_db", "Порог тишины, dBFS", -90, -20, 5, False, True),
    NumericField("silence_gap_seconds", "Тишина = новый трек, с", 0.1, 5, 0.1, False, True),
    NumericField("cooldown_seconds", "Пауза после скипа, с", 0, 10, 1, False, True),
)
MODE_CHOICES: tuple[tuple[Mode, str], ...] = ((Mode.LOG, "Только отмечать"), (Mode.SKIP, "Скипать (следующий трек)"))

RHYTHM_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("bpm", "Темп, BPM", "{:.1f}"),
    ("pulse_clarity", "Чёткость пульса (0–1)", "{:.3f}"),
    ("tempo_drift_pct", "Дрейф темпа, %", "{:.2f}"),
    ("tempo_jump_pct", "Макс. скачок темпа, %", "{:.2f}"),
    ("grid_tightness", "Попадание в сетку 1/16 (0–1)", "{:.3f}"),
    ("hf_attack_ratio", "Доля ВЧ в атаках (0–1)", "{:.3f}"),
)
SPECTRAL_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("peak_prominence", "Выраженность пиков", "{:.2f}"),
    ("peak_count", "Число пиков", "{:d}"),
    ("peak_stability", "Стабильность пиков", "{:.3f}"),
    ("comb_strength", "Сила гребёнки (fakeprint)", "{:.3f}"),
    ("comb_spacing_hz", "Шаг гребёнки, Гц", "{:.1f}"),
)


class TrayUnavailableError(RuntimeError):
    pass


class InstanceServerError(RuntimeError):
    pass


class LogoMissingError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class WorkerFailed:
    details: str


@dataclass(frozen=True)
class WorkerStopped:
    pass


@dataclass(frozen=True)
class StudyFinished:
    report: str


GuiEvent = ListenerEvent | WorkerFailed | WorkerStopped | StudyFinished


@dataclass(frozen=True)
class ScorePoint:
    seconds: float
    score: float
    shifted_score: float
    threshold: float
    phase: Phase


# ---------- pure helpers ----------


def pool_max(values: FloatArray, buckets: int) -> FloatArray:
    return np.array([chunk.max() for chunk in np.array_split(values, buckets)], dtype=np.float32)


def blank_spectrogram() -> NDArray[np.float32]:
    return np.zeros((SPECTROGRAM_COLUMNS, SPECTROGRAM_HEIGHT), dtype=np.float32)


def scroll_spectrogram(pixels: NDArray[np.float32], residue: FloatArray) -> NDArray[np.float32]:
    """Shift left by one column (axis 0 is time) and put the pooled residue (low frequencies first) at the end."""
    shifted: NDArray[np.float32] = np.roll(pixels, -1, axis=0)
    shifted[-1, :] = np.clip(pool_max(residue, SPECTROGRAM_HEIGHT), 0.0, 1.0)
    return shifted


def fakeprint_text(point: ScorePoint | None) -> str:
    if point is None:
        return "Fakeprint: нет данных"
    threshold: str = "решение с 5 с" if point.threshold == float("inf") else f"порог {point.threshold:+.1f}"
    return f"Fakeprint: балл {point.score:+.1f} ({threshold})"


def shifted_text(point: ScorePoint | None) -> str:
    if point is None:
        return "Suno с пониженным тоном: нет данных"
    return f"Suno с пониженным тоном: балл {point.shifted_score:+.1f} (порог {SHIFTED_LOGIT:+.0f})"


def comb_text(scores: list[tuple[Checkpoint, float]]) -> str:
    if not scores:
        return f"Гребёнка 200 Гц: ждёт первые {COMB_CHECKPOINTS[0].seconds:g} с"
    checkpoint, score = scores[-1]
    return f"Гребёнка 200 Гц: {score:+.2f} дБ по {checkpoint.seconds:g} с звука (порог {checkpoint.threshold_db:+.2f})"


def comb_fired(scores: list[tuple[Checkpoint, float]]) -> tuple[Checkpoint, float] | None:
    return next(((c, s) for c, s in scores if s >= c.threshold_db), None)


def blacklist_item_text(entry: BlacklistEntry) -> str:
    return f"{entry.artist}  —  {BLACKLIST_SOURCE_TEXT[entry.source]}, {entry.added.replace('T', ' ')}, трек «{entry.title}»"


def detection_reason(
    comb_scores: list[tuple[Checkpoint, float]], point: ScorePoint | None, blacklist_entry: BlacklistEntry | None
) -> str:
    if blacklist_entry is not None:
        return f"Исполнитель в чёрном списке (добавлен: {BLACKLIST_SOURCE_TEXT[blacklist_entry.source]})"
    fired: tuple[Checkpoint, float] | None = comb_fired(comb_scores)
    if fired is not None:
        return comb_text([fired])
    if point is not None and point.score < STRONG_LOGIT:
        return f"{shifted_text(point)}, уверенно"
    return f"{fakeprint_text(point)}, уверенно (≥ {STRONG_LOGIT:+.0f})"


def issue_url(version: str, system: str, frozen: bool) -> str:
    build: str = "exe" if frozen else "из исходников"
    body: str = f"Что произошло:\n\n\nЧто ожидалось:\n\n\n---\n{APP_NAME} {version} ({build}), {system}"
    return f"{ISSUES_URL}?{urlencode({'body': body})}"


def about_html(version: str) -> str:
    return (
        f"<h2>{APP_NAME}</h2>"
        f"<p>{APP_TAGLINE}<br>Версия {version}</p>"
        f"<p>Автор: {APP_AUTHOR}<br>"
        f'<a href="{REPO_URL}">GitHub</a> · <a href="{TELEGRAM_URL}">Telegram</a></p>'
        "<p>Лицензия MIT. Веса fakeprint — "
        '<a href="https://github.com/lofcz/ai-music-detector">lofcz/ai-music-detector</a> (MIT).<br>'
        "Не связан с Suno, Inc.</p>"
    )


def activate_running_instance() -> bool:
    """True when this user already runs SunoCheck; connecting is the signal for it to show its window."""
    socket = QLocalSocket()
    socket.connectToServer(INSTANCE_SERVER)
    if not socket.waitForConnected(INSTANCE_TIMEOUT_MS):
        return False
    # Windows blocks focus stealing; the launched process is in the foreground and may hand that right over.
    ctypes.windll.user32.AllowSetForegroundWindow(ASFW_ANY)
    socket.disconnectFromServer()
    return True


def listen_for_instances(on_activate: Callable[[], None]) -> QLocalServer:
    server = QLocalServer()
    if not server.listen(INSTANCE_SERVER):
        raise InstanceServerError(f"Cannot listen on {INSTANCE_SERVER}: {server.errorString()}")

    def accept() -> None:
        server.nextPendingConnection().deleteLater()
        on_activate()

    server.newConnection.connect(accept)
    return server


def logo_pixmap() -> QPixmap:
    pixmap = QPixmap(str(LOGO_PATH))
    if pixmap.isNull():
        raise LogoMissingError(f"Logo not found or unreadable: {LOGO_PATH}")
    return pixmap


def status_icon(color: str) -> QIcon:
    """The logo with a status dot in the bottom-right corner."""
    pixmap: QPixmap = logo_pixmap().scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(color))
    painter.setPen(QPen(QColor("#ffffff"), 3))
    painter.drawEllipse(36, 36, 26, 26)
    painter.end()
    return QIcon(pixmap)


# ---------- notification ----------


class NotificationPopup(QFrame):
    """In-app toast: Windows attributes QSystemTrayIcon balloons to python.exe, and clicking one starts the REPL."""

    def __init__(self, on_click: Callable[[], None]) -> None:
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedWidth(NOTIFY_WIDTH)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._on_click: Callable[[], None] = on_click
        layout = QVBoxLayout(self)
        self._title = QLabel()
        self._title.setWordWrap(True)
        self._title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self._title)
        self._body = QLabel()
        self._body.setWordWrap(True)
        layout.addWidget(self._body)
        hint = QLabel("Клик — открыть пульт")
        hint.setStyleSheet("color: #757575;")
        layout.addWidget(hint)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

    def notify(self, title: str, body: str, color: str) -> None:
        self._title.setText(title)
        self._body.setText(body)
        self.setStyleSheet(
            f"NotificationPopup {{ background: palette(window); border: 1px solid {color}; border-left: 6px solid {color}; }}"
        )
        self.adjustSize()
        area = QApplication.primaryScreen().availableGeometry()
        self.move(area.right() - self.width() - NOTIFY_MARGIN, area.bottom() - self.height() - NOTIFY_MARGIN)
        self.show()
        self.raise_()
        self._timer.start(NOTIFY_MS)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._timer.stop()
        self.hide()
        self._on_click()


# ---------- workers ----------


def run_study_worker(model: FakeprintModel, events: "queue.Queue[GuiEvent]") -> None:
    try:
        items: list[LabeledAudio] = labeled_recordings(RECORDINGS_DIR, LABELS_PATH)
        events.put(StudyFinished(f"сравнение метрик по {len(items)} размеченным записям:\n{study_report(items, model)}"))
    except StudyInputError as error:
        events.put(StudyFinished(f"сравнение метрик невозможно: {error}"))
    except Exception:
        events.put(WorkerFailed(traceback.format_exc()))
        events.put(StudyFinished("сравнение метрик упало, см. ошибку выше"))


def run_worker(
    config: ListenConfig,
    updates: "queue.Queue[ListenConfig]",
    model: FakeprintModel,
    events: "queue.Queue[GuiEvent]",
    stop: threading.Event,
) -> None:
    try:
        run_listener(config, updates, model, events.put, stop)
    except Exception:
        events.put(WorkerFailed(traceback.format_exc()))
    finally:
        events.put(WorkerStopped())


# ---------- window ----------


class DetectorWindow(QMainWindow):
    def __init__(self, model: FakeprintModel, settings: AppSettings) -> None:
        super().__init__()
        self._model: FakeprintModel = model
        self._settings: AppSettings = settings
        self._events: queue.Queue[GuiEvent] = queue.Queue()
        self._config_updates: queue.Queue[ListenConfig] = queue.Queue()
        self._stop: threading.Event = threading.Event()
        self._worker: threading.Thread | None = None
        self._session_path: Path | None = None
        self._quitting: bool = False
        self._tray_hint_shown: bool = False
        self._detected: int = 0
        self._skipped: int = 0
        self._status: str = "STOPPED"
        self._applied_status: str | None = None
        self._track: TrackInfo | None = None
        self._track_max_score: float = float("-inf")
        self._score_points: list[ScorePoint] = []
        self._score_span: float = 0.0
        self._comb_scores: list[tuple[Checkpoint, float]] = []
        self._last_reason: str = ""
        self._spectrogram: NDArray[np.float32] = blank_spectrogram()
        self._blacklist: dict[str, BlacklistEntry] = blacklist_entries(BLACKLIST_PATH)

        self.setWindowTitle(f"{APP_NAME} — {APP_TAGLINE}")
        self.setWindowIcon(QIcon(logo_pixmap()))
        self.resize(1180, 780)
        self._build_layout()
        self._build_menu()
        self._show_blacklist()
        self._build_tray()
        self._popup = NotificationPopup(self._show_window)
        self._reset_track_view(None)
        self._set_status("STOPPED")

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._drain_events)
        self._timer.start(EVENT_POLL_MS)

    # ---------- layout ----------

    def _build_layout(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)
        root.addWidget(self._build_side_panel())
        main = QVBoxLayout()
        main.addWidget(self._build_verdict_card())
        main.addWidget(self._build_tabs(), stretch=1)
        root.addLayout(main, stretch=1)
        self.setCentralWidget(central)

    def _build_verdict_card(self) -> QFrame:
        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        grid = QGridLayout(card)
        self._start_button = QPushButton("Старт")
        self._start_button.setFixedSize(110, HEADER_CONTROL_HEIGHT)
        self._start_button.clicked.connect(self._toggle)
        grid.addWidget(self._start_button, 0, 0, 4, 1, Qt.AlignmentFlag.AlignVCenter)
        self._status_label = QLabel()
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setFixedSize(120, HEADER_CONTROL_HEIGHT)
        grid.addWidget(self._status_label, 0, 1, 4, 1, Qt.AlignmentFlag.AlignVCenter)
        self._track_label = QLabel("Трек: -")
        self._track_label.setStyleSheet("font-size: 15px; font-weight: bold;")
        self._fakeprint_reason = QLabel()
        self._shifted_reason = QLabel()
        self._comb_reason = QLabel()
        for row, label in enumerate((self._track_label, self._fakeprint_reason, self._shifted_reason, self._comb_reason)):
            grid.addWidget(label, row, 2)
        self._level = QProgressBar()
        self._level.setRange(0, int(-LEVEL_FLOOR_DB))
        self._level.setTextVisible(False)
        self._level.setMaximumWidth(180)
        grid.addWidget(self._level, 0, 3)
        self._counters = QLabel()
        grid.addWidget(self._counters, 1, 3)
        self._device_label = QLabel("Устройство: -")
        self._device_label.setStyleSheet("color: #757575;")
        grid.addWidget(self._device_label, 2, 3)
        grid.setColumnStretch(2, 1)
        return card

    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(340)
        layout = QVBoxLayout(panel)

        settings_box = QGroupBox("Распознавание")
        form = QFormLayout(settings_box)
        self._mode_box = QComboBox()
        for mode, title in MODE_CHOICES:
            self._mode_box.addItem(title, mode.value)
        self._mode_box.setCurrentIndex([m for m, _ in MODE_CHOICES].index(self._settings.mode))
        self._mode_box.currentIndexChanged.connect(self._on_setting_changed)
        form.addRow(QLabel("При обнаружении Suno"))
        form.addRow(self._mode_box)
        form.addRow(QLabel("Fakeprint"))

        self._numeric_inputs: dict[str, QSpinBox | QDoubleSpinBox] = {}
        advanced = QWidget()
        advanced_form = QFormLayout(advanced)
        advanced_form.setContentsMargins(0, 0, 0, 0)
        for field in NUMERIC_FIELDS:
            spin: QSpinBox | QDoubleSpinBox = self._numeric_input(field)
            self._numeric_inputs[field.key] = spin
            (advanced_form if field.advanced else form).addRow(field.label, spin)
        self._rhythm_check = QCheckBox("Считать ритм-метрики")
        self._rhythm_check.setChecked(self._settings.rhythm_enabled)
        self._rhythm_check.toggled.connect(self._on_setting_changed)
        advanced_form.addRow(self._rhythm_check)
        self._record_check = QCheckBox("Сохранять записи треков (WAV)")
        self._record_check.setChecked(self._settings.record_audio)
        self._record_check.toggled.connect(self._on_setting_changed)
        advanced_form.addRow(self._record_check)

        toggle = QToolButton()
        toggle.setText("Дополнительно")
        toggle.setCheckable(True)
        toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toggle.setArrowType(Qt.ArrowType.RightArrow)
        advanced.setVisible(False)

        def toggle_advanced(shown: bool) -> None:
            advanced.setVisible(shown)
            toggle.setArrowType(Qt.ArrowType.DownArrow if shown else Qt.ArrowType.RightArrow)

        toggle.toggled.connect(toggle_advanced)
        form.addRow(toggle)
        form.addRow(advanced)
        path_label = QLabel("Настройки сохраняются сразу")
        path_label.setToolTip(str(SETTINGS_PATH))
        path_label.setStyleSheet("color: #9e9e9e;")
        form.addRow(path_label)
        layout.addWidget(settings_box)

        label_box = QGroupBox("Разметка текущего трека")
        label_layout = QVBoxLayout(label_box)
        for title, label in (("Это ИИ", "ai"), ("Это человек", "human")):
            button = QPushButton(title)
            button.clicked.connect(lambda _checked=False, value=label: self._label_track(value))
            label_layout.addWidget(button)
        blacklist_button = QPushButton("Автора — в чёрный список")
        blacklist_button.setToolTip("Все треки автора будут отмечаться (в режиме скипа — скипаться) с первой секунды")
        blacklist_button.clicked.connect(
            lambda: self._blacklist_track(self._track, BlacklistAction.ADD, BlacklistSource.BUTTON)
        )
        label_layout.addWidget(blacklist_button)
        layout.addWidget(label_box)
        layout.addStretch(1)
        return panel

    def _numeric_input(self, field: NumericField) -> QSpinBox | QDoubleSpinBox:
        value: float = float(getattr(self._settings, field.key))
        if field.is_int:
            spin_int = QSpinBox()
            spin_int.setRange(int(field.minimum), int(field.maximum))
            spin_int.setValue(int(value))
            spin_int.valueChanged.connect(self._on_setting_changed)
            return spin_int
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(field.minimum, field.maximum)
        spin.setSingleStep(field.step)
        spin.setValue(value)
        spin.valueChanged.connect(self._on_setting_changed)
        return spin

    def _build_tabs(self) -> QTabWidget:
        tabs = QTabWidget()

        detectors = QWidget()
        detectors_layout = QVBoxLayout(detectors)
        self._score_plot = pg.PlotWidget(title="Fakeprint: балл Suno по треку (красная линия — порог)")
        self._score_plot.setLabel("bottom", "время трека, с")
        self._score_plot.setYRange(*SCORE_AXIS)
        self._score_plot.showGrid(x=True, y=True, alpha=0.2)
        self._score_curve = self._score_plot.plot(pen=pg.mkPen(self.palette().color(QPalette.ColorRole.Text), width=2))
        self._threshold_curve = self._score_plot.plot(pen=pg.mkPen("#c62828", width=1, style=Qt.PenStyle.DashLine))
        for curve in (self._score_curve, self._threshold_curve):
            curve.setDownsampling(auto=True, method="peak")
            curve.setClipToView(True)
        self._score_marker = pg.ScatterPlotItem(size=12)
        self._score_plot.addItem(self._score_marker)
        self._min_seconds_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("#9e9e9e", style=Qt.PenStyle.DotLine))
        self._score_plot.addItem(self._min_seconds_line)
        detectors_layout.addWidget(self._score_plot, stretch=3)

        self._comb_plot = pg.PlotWidget(
            title="Гребёнка 200 Гц по растущему окну (красная точка — выше порога = Suno)"
        )
        self._comb_plot.setLabel("bottom", "секунд звука трека")
        self._comb_plot.setXRange(0.0, COMB_CHECKPOINTS[-1].seconds + 2.5, padding=0)
        self._comb_plot.setLabel("left", "дБ")
        self._comb_plot.setYRange(*COMB_AXIS)
        self._comb_plot.showGrid(x=True, y=True, alpha=0.2)
        self._comb_plot.plot(
            [c.seconds for c in COMB_CHECKPOINTS],
            [c.threshold_db for c in COMB_CHECKPOINTS],
            pen=pg.mkPen("#c62828", style=Qt.PenStyle.DashLine),
        )
        self._comb_curve = self._comb_plot.plot(pen=pg.mkPen("#bdbdbd", width=1))
        self._comb_points = pg.ScatterPlotItem(size=12)
        self._comb_plot.addItem(self._comb_points)
        detectors_layout.addWidget(self._comb_plot, stretch=2)
        tabs.addTab(detectors, "Детекторы")

        spectrum = QWidget()
        spectrum_layout = QVBoxLayout(spectrum)
        self._fakeprint_plot = pg.PlotWidget(title="Отпечаток: спектральные пики 1–8 кГц (у Suno — регулярная «гребёнка»)")
        self._fakeprint_plot.setLabel("bottom", "кГц")
        self._fakeprint_plot.setYRange(0.0, 1.0)
        self._fakeprint_curve = self._fakeprint_plot.plot(pen=pg.mkPen("#6a1b9a", width=1))
        spectrum_layout.addWidget(self._fakeprint_plot)
        self._spectrogram_plot = pg.PlotWidget(title="Остаточная спектрограмма 1–8 кГц (артефакты — ровные горизонтальные линии), время →")
        self._spectrogram_plot.hideAxis("left")
        self._spectrogram_plot.hideAxis("bottom")
        self._spectrogram_image = pg.ImageItem()
        self._spectrogram_image.setLookupTable(SPECTROGRAM_LUT)
        self._spectrogram_plot.addItem(self._spectrogram_image)
        spectrum_layout.addWidget(self._spectrogram_plot)
        tabs.addTab(spectrum, "Спектр")

        experiments = QWidget()
        experiments_layout = QHBoxLayout(experiments)
        self._rhythm_values = self._value_box(experiments_layout, "Ритм (не влияет на скип)", RHYTHM_FIELDS)
        self._spectral_values = self._value_box(experiments_layout, "Пики fakeprint (не влияют на скип)", SPECTRAL_FIELDS)
        study_box = QGroupBox("Сравнение метрик")
        study_layout = QVBoxLayout(study_box)
        study_layout.addWidget(QLabel("AUROC метрик по data/recordings\nи разметке labels.csv"))
        self._study_button = QPushButton("Сравнить метрики")
        self._study_button.clicked.connect(self._start_study)
        study_layout.addWidget(self._study_button)
        study_layout.addStretch(1)
        experiments_layout.addWidget(study_box)
        experiments_layout.addStretch(1)
        tabs.addTab(experiments, "Эксперименты")

        blacklist = QWidget()
        blacklist_layout = QVBoxLayout(blacklist)
        hint = QLabel(
            "Автор попадает сюда, если его трек помечен кнопкой «Это ИИ» или «Автора — в чёрный список», "
            "либо приложение само скипнуло его трек. Треки автора из списка отмечаются с первой секунды, "
            "в режиме скипа — скипаются. «Это человек» на треке автора убирает его из списка."
        )
        hint.setWordWrap(True)
        blacklist_layout.addWidget(hint)
        self._blacklist_view = QListWidget()
        self._blacklist_view.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        blacklist_layout.addWidget(self._blacklist_view, stretch=1)
        remove_button = QPushButton("Убрать из списка")
        remove_button.clicked.connect(self._remove_selected_from_blacklist)
        blacklist_layout.addWidget(remove_button, alignment=Qt.AlignmentFlag.AlignLeft)
        tabs.addTab(blacklist, "Чёрный список")

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(MAX_LOG_LINES)
        self._log.setStyleSheet("font-family: Consolas; font-size: 12px;")
        tabs.addTab(self._log, "Журнал")
        return tabs

    def _value_box(self, layout: QHBoxLayout, title: str, fields: tuple[tuple[str, str, str], ...]) -> dict[str, QLabel]:
        box = QGroupBox(title)
        form = QFormLayout(box)
        values: dict[str, QLabel] = {}
        for key, label, _ in fields:
            value = QLabel("—")
            value.setAlignment(Qt.AlignmentFlag.AlignRight)
            form.addRow(label, value)
            values[key] = value
        layout.addWidget(box, alignment=Qt.AlignmentFlag.AlignTop)
        return values

    def _build_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("Файл")
        file_menu.addAction("Открыть папку данных", self._open_data_dir)
        file_menu.addSeparator()
        quit_action = file_menu.addAction("Выход", self._quit)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))

        settings_menu = bar.addMenu("Настройки")
        settings_menu.setToolTipsVisible(True)
        self._notify_action = settings_menu.addAction("Показывать уведомления")
        self._notify_action.setCheckable(True)
        self._notify_action.setChecked(self._settings.notifications_enabled)
        self._notify_action.toggled.connect(lambda on: self._apply_settings(replace(self._settings, notifications_enabled=on)))
        launch_action = settings_menu.addAction("Слушать сразу после запуска")
        launch_action.setCheckable(True)
        launch_action.setChecked(self._settings.listen_on_launch)
        launch_action.toggled.connect(lambda on: self._apply_settings(replace(self._settings, listen_on_launch=on)))
        self._autostart_action = settings_menu.addAction("Запускать вместе с Windows")
        self._autostart_action.setCheckable(True)
        self._autostart_action.setEnabled(FROZEN)
        if FROZEN:
            self._autostart_action.setChecked(read_autostart() == autostart_command(Path(sys.executable)))
            self._autostart_action.setToolTip("SunoCheck стартует в трее и сразу слушает")
        else:
            self._autostart_action.setToolTip("Доступно только в SunoCheck.exe")
        self._autostart_action.toggled.connect(self._set_autostart)

        help_menu = bar.addMenu("Справка")
        help_menu.addAction("Как это работает", lambda: self._open_url(README_URL))
        help_menu.addAction(f"Проверить обновления (у вас {APP_VERSION})", lambda: self._open_url(RELEASES_URL))
        help_menu.addAction("Сообщить о проблеме", lambda: self._open_url(issue_url(APP_VERSION, platform.platform(), FROZEN)))
        help_menu.addAction("Написать автору в Telegram", lambda: self._open_url(TELEGRAM_URL))
        help_menu.addSeparator()
        help_menu.addAction("О программе", self._show_about)

    def _set_autostart(self, enabled: bool) -> None:
        try:
            write_autostart(autostart_command(Path(sys.executable)) if enabled else None)
        except OSError as error:
            self._autostart_action.blockSignals(True)
            self._autostart_action.setChecked(not enabled)
            self._autostart_action.blockSignals(False)
            QMessageBox.critical(self, "Автозапуск", f"Не удалось изменить автозапуск в реестре Windows:\n{error}")
            return
        self._append_log(f"автозапуск с Windows: {'включён' if enabled else 'выключен'}")

    def _open_url(self, url: str) -> None:
        if not QDesktopServices.openUrl(QUrl(url)):
            QMessageBox.warning(self, "Ссылка не открылась", f"Откройте адрес вручную:\n{url}")

    def _show_about(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(f"О программе {APP_NAME}")
        layout = QVBoxLayout(dialog)
        logo = QLabel()
        logo.setPixmap(
            logo_pixmap().scaled(
                ABOUT_LOGO_PX, ABOUT_LOGO_PX, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        )
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(logo)
        text = QLabel(about_html(APP_VERSION))
        text.setTextFormat(Qt.TextFormat.RichText)
        text.setOpenExternalLinks(True)
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Close).setText("Закрыть")
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def _build_tray(self) -> None:
        self._tray = QSystemTrayIcon(status_icon(STATUS_COLORS["STOPPED"]), self)
        menu = QMenu()
        menu.addAction("Открыть пульт", self._show_window)
        self._tray_toggle = menu.addAction("Старт", self._toggle)
        mode_menu = menu.addMenu("Режим")
        self._mode_actions: dict[Mode, QAction] = {}
        group = QActionGroup(menu)
        for mode, title in MODE_CHOICES:
            action = mode_menu.addAction(title)
            action.setCheckable(True)
            action.setChecked(mode is self._settings.mode)
            action.triggered.connect(lambda _checked=False, value=mode: self._set_mode(value))
            group.addAction(action)
            self._mode_actions[mode] = action
        menu.addSeparator()
        menu.addAction("Текущий трек — ИИ", lambda: self._label_track("ai"))
        menu.addAction("Текущий трек — человек", lambda: self._label_track("human"))
        menu.addAction(
            "Автора — в чёрный список", lambda: self._blacklist_track(self._track, BlacklistAction.ADD, BlacklistSource.BUTTON)
        )
        menu.addSeparator()
        menu.addAction(self._notify_action)
        menu.addAction("Выход", self._quit)
        self._tray_menu = menu
        self._tray.setContextMenu(menu)
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

    # ---------- settings ----------

    def _read_settings_from_widgets(self) -> AppSettings:
        def value(key: str) -> float:
            return float(self._numeric_inputs[key].value())

        return replace(
            self._settings,
            mode=Mode(self._mode_box.currentData()),
            threshold_offset=value("threshold_offset"),
            confirm_seconds=value("confirm_seconds"),
            min_seconds=value("min_seconds"),
            max_seconds=value("max_seconds"),
            eval_interval_seconds=value("eval_interval_seconds"),
            silence_db=value("silence_db"),
            silence_gap_seconds=value("silence_gap_seconds"),
            cooldown_seconds=value("cooldown_seconds"),
            rhythm_enabled=self._rhythm_check.isChecked(),
            record_audio=self._record_check.isChecked(),
        )

    def _on_setting_changed(self) -> None:
        self._apply_settings(self._read_settings_from_widgets())

    def _apply_settings(self, settings: AppSettings) -> None:
        if settings == self._settings:
            return
        self._settings = settings
        save_settings(settings, SETTINGS_PATH)
        self._mode_actions[settings.mode].setChecked(True)
        if self._worker is not None:
            self._config_updates.put(self._listen_config())
        self._draw_score_chart()

    def _listen_config(self) -> ListenConfig:
        return to_listen_config(self._settings, RECORDINGS_DIR, frozenset(self._blacklist))

    def _set_mode(self, mode: Mode) -> None:
        self._mode_box.setCurrentIndex([m for m, _ in MODE_CHOICES].index(mode))

    # ---------- worker control ----------

    def start_listening(self) -> None:
        if self._worker is None:
            self._start()

    def _toggle(self) -> None:
        if self._worker is None:
            self._start()
            return
        self._start_button.setEnabled(False)
        self._start_button.setText("Остановка…")
        self._tray_toggle.setEnabled(False)
        self._stop.set()

    def _start(self) -> None:
        self._stop = threading.Event()
        self._config_updates = queue.Queue()
        self._session_path = SESSIONS_DIR / f"session_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self._worker = threading.Thread(
            target=run_worker,
            args=(self._listen_config(), self._config_updates, self._model, self._events, self._stop),
            daemon=True,
        )
        self._worker.start()
        self._start_button.setText("Стоп")
        self._tray_toggle.setText("Стоп")
        self._set_status("LISTEN")
        self._append_log(f"старт, журнал оценок: {self._session_path}")

    def _quit(self) -> None:
        self._quitting = True
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=3.0)
        self._tray.hide()
        QApplication.quit()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._quitting:
            event.accept()
            return
        event.ignore()
        self.hide()
        if not self._tray_hint_shown:
            self._tray_hint_shown = True
            self._notify(
                f"{APP_NAME} работает в фоне",
                "Клик по иконке в трее — открыть пульт, правый клик — меню и выход.",
                STATUS_COLORS["SILENCE"],
            )

    def _notify(self, title: str, body: str, color: str) -> None:
        if self._settings.notifications_enabled:
            self._popup.notify(title, body, color)

    def _show_window(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick):
            self._show_window()

    # ---------- events ----------

    def _drain_events(self) -> None:
        while True:
            try:
                event: GuiEvent = self._events.get_nowait()
            except queue.Empty:
                return
            self._handle_event(event)

    def _handle_event(self, event: GuiEvent) -> None:
        match event:
            case CaptureStarted(device_name=name, sample_rate=rate, channels=channels):
                self._device_label.setText(f"{name}, {rate} Гц, {channels} кан.")
                self._append_log(f"захват: {name}, {rate} Гц")
            case TrackChanged(track=track):
                self._reset_track_view(track)
                self._append_log(f"трек: {describe(track)}")
            case TrackReset(reason=reason):
                self._reset_track_view(self._track)
                self._set_status("SILENCE")
                self._append_log(f"сброс буфера ({reason})")
            case LevelUpdate(dbfs=dbfs):
                self._level.setValue(int(max(0.0, dbfs - LEVEL_FLOOR_DB)))
            case Evaluation():
                self._on_evaluation(event)
            case CombScored(checkpoint=checkpoint, score=score):
                self._on_comb(checkpoint, score)
            case SkipPerformed(track=track):
                self._skipped += 1
                self._update_counters()
                self._append_log(f">>> СКИП: {describe(track)}")
                self._notify(f"Пропущен Suno: {describe(track)}", self._last_reason, STATUS_COLORS["DETECTED"])
                self._blacklist_track(track, BlacklistAction.ADD, BlacklistSource.SKIP)
            case RecordingSaved(path=path):
                self._append_log(f"запись: {path.name}")
            case CaptureWarning(message=message):
                self._append_log(f"ВНИМАНИЕ: {message}")
            case StudyFinished(report=report):
                self._study_button.setEnabled(True)
                self._study_button.setText("Сравнить метрики")
                self._append_log(report)
            case WorkerFailed(details=details):
                self._append_log(details)
                QMessageBox.critical(self, "Ошибка детектора", details)
            case WorkerStopped():
                self._worker = None
                self._start_button.setEnabled(True)
                self._start_button.setText("Старт")
                self._tray_toggle.setEnabled(True)
                self._tray_toggle.setText("Старт")
                self._set_status("STOPPED")
                self._append_log("остановлено")

    def _on_evaluation(self, evaluation: Evaluation) -> None:
        previous: Phase | None = self._score_points[-1].phase if self._score_points else None
        point = ScorePoint(
            evaluation.track_seconds,
            evaluation.score,
            evaluation.shifted_score,
            evaluation.threshold,
            evaluation.phase,
        )
        self._score_points.append(point)
        self._track_max_score = evaluation.max_score
        self._set_status(evaluation.phase.value)
        self._fakeprint_reason.setText(fakeprint_text(point))
        self._shifted_reason.setText(shifted_text(point))
        self._draw_score_chart()
        self._fakeprint_curve.setData(np.linspace(1.0, 8.0, evaluation.fakeprint.size), evaluation.fakeprint)
        self._spectrogram = scroll_spectrogram(self._spectrogram, evaluation.recent_residue)
        self._spectrogram_image.setImage(self._spectrogram, levels=(0.0, 1.0), autoLevels=False)
        self._show_values(self._spectral_values, SPECTRAL_FIELDS, evaluation.spectral)
        if evaluation.rhythm is not None:
            self._show_values(self._rhythm_values, RHYTHM_FIELDS, evaluation.rhythm)
        if self._session_path is not None:
            append_csv_row(self._session_path, EVALUATION_HEADER, evaluation_row(evaluation, datetime.now()))
        if evaluation.phase is previous:
            return
        self._append_log(
            f"{evaluation.phase.value:<8} {fakeprint_text(point)} t={evaluation.track_seconds:5.2f}s | {describe(evaluation.track)}"
        )
        if evaluation.phase is Phase.DETECTED:
            self._detected += 1
            self._update_counters()
            entry: BlacklistEntry | None = (
                self._blacklist.get(artist_key(evaluation.track.artist))
                if evaluation.blacklisted and evaluation.track is not None
                else None
            )
            reason: str = detection_reason(self._comb_scores, point, entry)
            self._last_reason = reason
            self._append_log(f"обнаружен Suno — {reason}")
            if self._settings.mode is Mode.LOG:
                self._notify(f"Похоже на Suno: {describe(evaluation.track)}", reason, STATUS_COLORS["DETECTED"])

    def _on_comb(self, checkpoint: Checkpoint, score: float) -> None:
        self._comb_scores.append((checkpoint, score))
        self._comb_reason.setText(comb_text(self._comb_scores))
        self._draw_comb_chart()
        self._append_log(comb_text(self._comb_scores))

    def _reset_track_view(self, track: TrackInfo | None) -> None:
        self._track = track
        self._track_max_score = float("-inf")
        self._score_points = []
        self._comb_scores = []
        app: str = f"  [{track.app_id}]" if track is not None else ""
        self._track_label.setText(f"Трек: {describe(track)}{app}")
        self._tray.setToolTip(f"{APP_NAME}: {STATUS_TEXT[self._status]}\n{describe(track)}")
        self._fakeprint_reason.setText(fakeprint_text(None))
        self._shifted_reason.setText(shifted_text(None))
        self._comb_reason.setText(comb_text([]))
        self._draw_score_chart()
        self._draw_comb_chart()
        self._fakeprint_curve.setData([], [])
        self._show_values(self._rhythm_values, RHYTHM_FIELDS, None)
        self._show_values(self._spectral_values, SPECTRAL_FIELDS, None)
        self._spectrogram = blank_spectrogram()
        self._spectrogram_image.setImage(self._spectrogram, levels=(0.0, 1.0), autoLevels=False)

    def _label_track(self, label: str) -> None:
        append_csv_row(LABELS_PATH, LABEL_HEADER, label_row(self._track, label, self._track_max_score, datetime.now()))
        self._append_log(f"разметка: {describe(self._track)} = {label} (max score={self._track_max_score:.1f})")
        if label == "ai":
            self._blacklist_track(self._track, BlacklistAction.ADD, BlacklistSource.AI_LABEL)
        else:
            self._blacklist_track(self._track, BlacklistAction.REMOVE, BlacklistSource.HUMAN_LABEL)

    def _blacklist_track(self, track: TrackInfo | None, action: BlacklistAction, source: BlacklistSource) -> None:
        if track is None:
            self._append_log("чёрный список: трек неизвестен (нет медиасессии), автор не изменён")
            return
        self._change_blacklist(track.artist, track.title, action, source)

    def _change_blacklist(self, artist: str, title: str, action: BlacklistAction, source: BlacklistSource) -> None:
        try:
            changed: bool = change_blacklist(BLACKLIST_PATH, artist, title, action, source, datetime.now())
        except MissingArtistError as error:
            self._append_log(f"чёрный список: {error}")
            return
        if not changed:
            return
        self._blacklist = blacklist_entries(BLACKLIST_PATH)
        self._show_blacklist()
        sign: str = "+" if action is BlacklistAction.ADD else "−"
        self._append_log(f"чёрный список {sign} {artist} ({BLACKLIST_SOURCE_TEXT[source]})")
        if self._worker is not None:
            self._config_updates.put(self._listen_config())

    def _show_blacklist(self) -> None:
        self._blacklist_view.clear()
        for key, entry in sorted(self._blacklist.items()):
            item = QListWidgetItem(blacklist_item_text(entry))
            item.setData(Qt.ItemDataRole.UserRole, key)
            self._blacklist_view.addItem(item)

    def _remove_selected_from_blacklist(self) -> None:
        keys: list[str] = [str(item.data(Qt.ItemDataRole.UserRole)) for item in self._blacklist_view.selectedItems()]
        for key in keys:
            entry: BlacklistEntry = self._blacklist[key]
            self._change_blacklist(entry.artist, entry.title, BlacklistAction.REMOVE, BlacklistSource.REMOVE_BUTTON)

    def _start_study(self) -> None:
        self._study_button.setEnabled(False)
        self._study_button.setText("Считаю…")
        self._append_log("сравнение метрик: запущено по data/recordings + labels.csv")
        threading.Thread(target=run_study_worker, args=(self._model, self._events), daemon=True).start()

    def _open_data_dir(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(DATA_DIR)

    # ---------- drawing ----------

    def _set_status(self, status: str) -> None:
        """Restyling and the tray icon (a Shell_NotifyIcon call) are costly at the evaluation rate: only on change."""
        if status == self._applied_status:
            return
        self._applied_status = status
        self._status = status
        self._status_label.setText(STATUS_TEXT[status])
        self._status_label.setStyleSheet(
            f"background: {STATUS_COLORS[status]}; color: white; font-size: 14px; font-weight: bold; border-radius: 4px;"
        )
        self._tray.setIcon(status_icon(STATUS_COLORS[status]))
        self._tray.setToolTip(f"{APP_NAME}: {STATUS_TEXT[status]}\n{describe(self._track)}")

    def _update_counters(self) -> None:
        self._counters.setText(f"Обнаружено: {self._detected}   Скипнуто: {self._skipped}")

    def _show_values(
        self, labels: dict[str, QLabel], fields: tuple[tuple[str, str, str], ...], metrics: RhythmMetrics | SpectralMetrics | None
    ) -> None:
        for key, _, template in fields:
            value: int | float | None = None if metrics is None else getattr(metrics, key)
            labels[key].setText("—" if value is None else template.format(value))

    def _draw_score_chart(self) -> None:
        points: list[ScorePoint] = self._score_points
        # A new X range relayouts the axis and repaints the whole plot (30-45 ms), so it grows in whole steps.
        last_seconds: float = points[-1].seconds if points else 0.0
        span: float = SCORE_SPAN_STEP_SECONDS * (math.floor((last_seconds + 5.0) / SCORE_SPAN_STEP_SECONDS) + 1)
        if span != self._score_span:
            self._score_span = span
            self._score_plot.setXRange(0.0, span, padding=0)
        self._min_seconds_line.setValue(self._settings.min_seconds)
        self._score_curve.setData([p.seconds for p in points], [p.score for p in points])
        decided: list[ScorePoint] = [p for p in points if p.threshold != float("inf")]
        self._threshold_curve.setData([p.seconds for p in decided], [p.threshold for p in decided])
        if points:
            last: ScorePoint = points[-1]
            self._score_marker.setData([last.seconds], [last.score], brush=pg.mkBrush(PHASE_COLORS[last.phase]), pen=None)
        else:
            self._score_marker.setData([], [])

    def _draw_comb_chart(self) -> None:
        seconds: list[float] = [c.seconds for c, _ in self._comb_scores]
        scores: list[float] = [v for _, v in self._comb_scores]
        self._comb_curve.setData(seconds, scores)
        brushes: list[QBrush] = [
            pg.mkBrush(COMB_HIT_COLOR if v >= c.threshold_db else COMB_MISS_COLOR) for c, v in self._comb_scores
        ]
        self._comb_points.setData(seconds, scores, brush=brushes, pen=None)

    def _append_log(self, message: str) -> None:
        self._log.appendPlainText(f"[{datetime.now():%H:%M:%S}] {message}")


def configure_plots(app: QApplication) -> None:
    """Charts follow the system light/dark palette; must run before any plot widget is created."""
    palette: QPalette = app.palette()
    pg.setConfigOptions(
        antialias=True, background=palette.color(QPalette.ColorRole.Base), foreground=palette.color(QPalette.ColorRole.Text)
    )


def main() -> None:
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    app = QApplication(sys.argv)
    if activate_running_instance():
        return
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setQuitOnLastWindowClosed(False)
    if not QSystemTrayIcon.isSystemTrayAvailable():
        raise TrayUnavailableError("System tray is not available; the background mode needs it")
    configure_plots(app)
    settings: AppSettings = load_settings(SETTINGS_PATH)
    window = DetectorWindow(load_model(DEFAULT_MODEL_PATH), settings)
    server: QLocalServer = listen_for_instances(window._show_window)
    background: bool = BACKGROUND_FLAG in sys.argv[1:]
    if not background:
        window.show()
    if background or settings.listen_on_launch:
        window.start_listening()
    exit_code: int = app.exec()
    server.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
