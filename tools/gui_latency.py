"""GUI responsiveness probe: the real run_listener + real window, with capture replaced by a real-time file replay.

Measures (1) worker cost (evaluate, comb_score), (2) GUI handler cost per event type, (3) delivery delay from
emit to GUI handling, (4) Qt event-loop stalls via a 5 ms probe timer.

Usage: uv run python tmp/gui_latency.py <audio file> <seconds> <tab index: 0 detectors, 1 spectrum> <popup|nopopup>
"""

import queue
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QTabWidget

import sunocheck.listener as listener
from sunocheck.capture import CaptureFormat
from sunocheck.fakeprint import DEFAULT_MODEL_PATH, load_model
from sunocheck.gui import RECORDINGS_DIR, DetectorWindow, GuiEvent, configure_plots
from sunocheck.listener import CHUNK_SECONDS, Evaluation, ListenerEvent, run_listener
from sunocheck.media_session import SessionUpdate, TrackInfo
from sunocheck.settings import DEFAULT_SETTINGS, to_listen_config

PROBE_MS: int = 5
OUT_CSV: Path = Path(__file__).parent / "latency_session.csv"


def stats(name: str, values: list[float]) -> str:
    a = np.asarray(values) * 1000.0
    if a.size == 0:
        return f"  {name:<22} n=0"
    return f"  {name:<22} n={a.size:<5} mean {a.mean():6.1f}  p95 {np.percentile(a, 95):6.1f}  max {a.max():6.1f} ms"


def main(path: Path, seconds: float, tab: int, popup: bool) -> None:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)[: int(seconds * rate)]
    chunk_n = int(rate * CHUNK_SECONDS)
    done = threading.Event()

    class FileCapture:
        def __init__(self, chunk_seconds: float) -> None:
            self.overflow_count = 0
            self._index = 0
            self._start = 0.0

        def start(self) -> CaptureFormat:
            self._start = time.perf_counter()
            return CaptureFormat(device_name=f"file {path.name}", sample_rate=rate, channels=1)

        def read(self, timeout_s: float) -> np.ndarray | None:
            if (self._index + 1) * chunk_n > mono.size:
                done.set()
                time.sleep(timeout_s)
                return None
            time.sleep(max(0.0, self._start + (self._index + 1) * CHUNK_SECONDS - time.perf_counter()))
            chunk = mono[self._index * chunk_n : (self._index + 1) * chunk_n]
            self._index += 1
            return chunk

        def is_active(self) -> bool:
            return True

        def close(self) -> None:
            pass

    class FixedWatcher:
        def __init__(self, poll_seconds: float, timeout_seconds: float) -> None:
            self.events: queue.Queue[SessionUpdate] = queue.Queue()
            self.events.put(SessionUpdate(TrackInfo(app_id="probe", artist="probe", title=path.stem)))

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    worker_times: dict[str, list[float]] = defaultdict(list)

    def timed[**P, R](name: str, function: Callable[P, R]) -> Callable[P, R]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            t = time.perf_counter()
            result = function(*args, **kwargs)
            worker_times[name].append(time.perf_counter() - t)
            return result

        return wrapper

    listener.LoopbackCapture = FileCapture
    listener.MediaSessionWatcher = FixedWatcher
    listener.evaluate = timed("evaluate", listener.evaluate)
    listener.comb_score = timed("comb_score (thread)", listener.comb_score)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    configure_plots(app)
    model = load_model(DEFAULT_MODEL_PATH)
    window = DetectorWindow(model, DEFAULT_SETTINGS)
    window.showMaximized()
    tabs: QTabWidget | None = window.findChild(QTabWidget)
    if tabs is None:
        raise LookupError("DetectorWindow has no QTabWidget")
    tabs.setCurrentIndex(tab)
    window._session_path = OUT_CSV
    if not popup:
        window._popup.notify = lambda title, body, color: None
    config = replace(to_listen_config(DEFAULT_SETTINGS, RECORDINGS_DIR, frozenset()), recordings_dir=None)

    gui_times: dict[str, list[float]] = defaultdict(list)
    gaps: list[float] = []
    sent: dict[int, float] = {}
    delivery: list[float] = []
    eval_arrivals: list[float] = []

    def emit(event: ListenerEvent) -> None:
        sent[id(event)] = time.perf_counter()
        window._events.put(event)

    original = window._handle_event

    def handle(event: GuiEvent) -> None:
        t = time.perf_counter()
        if id(event) in sent:
            delivery.append(t - sent.pop(id(event)))
        if isinstance(event, Evaluation):
            eval_arrivals.append(t)
        original(event)
        gui_times[type(event).__name__].append(time.perf_counter() - t)

    window._handle_event = handle
    stop = threading.Event()
    started = time.perf_counter()
    last = [started]

    def probe() -> None:
        now = time.perf_counter()
        gaps.append(now - last[0] - PROBE_MS / 1000.0)
        last[0] = now
        if gaps[-1] > 0.03:
            print(f"  stall {gaps[-1] * 1000:5.1f} ms at t={now - started:5.2f}s")
        if done.is_set():
            stop.set()
            app.quit()

    probe_timer = QTimer()
    probe_timer.timeout.connect(probe)
    probe_timer.start(PROBE_MS)
    threading.Thread(target=run_listener, args=(config, queue.Queue(), model, emit, stop), daemon=True).start()
    app.exec()
    OUT_CSV.unlink(missing_ok=True)

    print(f"rate {rate}, {seconds:.0f} s")
    print("worker:")
    for name, values in worker_times.items():
        print(stats(name, values))
    print("gui handlers:")
    for name, values in gui_times.items():
        print(stats(name, values))
    print(stats("delivery to GUI", delivery))
    print(stats("eval interval", list(np.diff(eval_arrivals))))
    g = np.asarray(gaps) * 1000.0
    print(
        f"event-loop stalls: p50 {np.percentile(g, 50):.1f}  p95 {np.percentile(g, 95):.1f}  p99 {np.percentile(g, 99):.1f}"
        f"  max {g.max():.1f} ms; >30 ms: {(g > 30).sum()}, >100 ms: {(g > 100).sum()}"
    )
    print("comb (seconds, score, threshold):", [(c.seconds, round(v, 2), c.threshold_db) for c, v in window._comb_scores])
    detected_lines: list[str] = [line for line in window._log.toPlainText().splitlines() if "обнаружен" in line]
    print("status", window._status, "| first DETECTED log line:", detected_lines[:1])


if __name__ == "__main__":
    main(Path(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "popup")
