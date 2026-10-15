"""Offline GUI smoke test: feed synthetic listener events to the window, save screenshots, quit.

The blacklist journal goes to a temporary folder, never to data/.

Usage: uv run python tools/gui_smoke.py <screenshot dir>
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QTabWidget

from sunocheck import gui
from sunocheck.coherent_comb import CHECKPOINTS
from sunocheck.decision import Phase
from sunocheck.fakeprint import DEFAULT_MODEL_PATH, load_model
from sunocheck.gui import DetectorWindow, configure_plots
from sunocheck.listener import CaptureStarted, CombScored, Evaluation, LevelUpdate, SkipPerformed, TrackChanged
from sunocheck.media_session import TrackInfo
from sunocheck.settings import DEFAULT_SETTINGS
from sunocheck.spectral_metrics import SpectralMetrics


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    gui.BLACKLIST_PATH = Path(tempfile.mkdtemp(prefix="sunocheck_smoke_")) / "blacklist.csv"
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    configure_plots(app)
    window = DetectorWindow(load_model(DEFAULT_MODEL_PATH), DEFAULT_SETTINGS)
    window.show()
    track = TrackInfo(app_id="Yandex.Music", artist="SUDO", title="Тонкий Чек")
    rng = np.random.default_rng(0)
    events = [CaptureStarted("Динамики (Realtek)", 48000, 2), TrackChanged(track), LevelUpdate(-14.0)]
    for step in range(1, 90):
        seconds = step * 0.25 * 2
        score = -20 + step * 0.25 + rng.normal(0, 0.8)
        threshold = float("inf") if seconds < 5 else -9.0
        phase = Phase.DETECTED if step > 80 else Phase.HOLD if score > threshold else Phase.LISTEN
        fakeprint = np.clip(rng.random(3585).astype(np.float32) ** 6, 0, 1)
        spectral = SpectralMetrics(peak_count=12, peak_prominence=3.1, comb_strength=0.4, comb_spacing_hz=200.0, peak_stability=0.8)
        events.append(
            Evaluation(
                track=track,
                phase=phase,
                score=score,
                shifted_score=score - 5.0,
                threshold=threshold,
                max_score=score,
                track_seconds=seconds,
                buffered_seconds=seconds,
                fakeprint=fakeprint,
                recent_residue=fakeprint,
                spectral=spectral,
                rhythm=None,
                comb_score=None,
                comb_detected=False,
                blacklisted=False,
            )
        )
        for checkpoint, comb in zip(CHECKPOINTS, (-4.6, -4.1, -2.7, -1.6, -0.8, 0.6), strict=False):
            if seconds == checkpoint.seconds:
                events.append(CombScored(track, checkpoint, comb, comb >= checkpoint.threshold_db))
    events.append(SkipPerformed(track))
    for event in events:
        window._handle_event(event)
    tabs: QTabWidget | None = window.findChild(QTabWidget)
    if tabs is None:
        raise LookupError("DetectorWindow has no QTabWidget")

    def shoot() -> None:
        window.grab().save(str(out / "gui_detectors.png"))
        tabs.setCurrentIndex(1)
        QTimer.singleShot(300, shoot_spectrum)

    def shoot_spectrum() -> None:
        window.grab().save(str(out / "gui_spectrum.png"))
        tabs.setCurrentIndex(4)
        QTimer.singleShot(300, shoot_log)

    def shoot_log() -> None:
        window.grab().save(str(out / "gui_log.png"))
        window._popup.grab().save(str(out / "gui_popup.png"))
        tabs.setCurrentIndex(3)
        QTimer.singleShot(300, shoot_blacklist)

    def shoot_blacklist() -> None:
        window.grab().save(str(out / "gui_blacklist.png"))
        window._quit()

    QTimer.singleShot(800, shoot)
    app.exec()
    print("ok")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
