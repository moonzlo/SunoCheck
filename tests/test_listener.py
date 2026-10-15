"""run_listener end to end on synthetic audio. Capture, the media session and the default output id are fakes:
the loop checks the output once per iteration (DEVICE_POLL_SECONDS = 0), so the n-th id answers the n-th read."""

import queue
import threading
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

import sunocheck.listener as listener
from sunocheck.capture import CaptureFormat
from sunocheck.fakeprint import DEFAULT_MODEL_PATH, FloatArray, load_model
from sunocheck.listener import (
    CHUNK_SECONDS,
    CapturePaused,
    CaptureStarted,
    Evaluation,
    ListenConfig,
    ListenerEvent,
    TrackChanged,
    TrackReset,
    run_listener,
)
from sunocheck.media_session import SessionUpdate, TrackInfo
from sunocheck.settings import DEFAULT_SETTINGS, to_listen_config

RATE: int = 44100
LOW_RATE: int = 16000
RUN_TIMEOUT_SECONDS: float = 60.0
TRACK_A: TrackInfo = TrackInfo("test", "Artist A", "Song A")
TRACK_B: TrackInfo = TrackInfo("test", "Artist B", "Song B")

Item = FloatArray | TrackInfo | None


@dataclass(frozen=True)
class Device:
    rate: int
    items: tuple[Item, ...]


@dataclass(frozen=True)
class Scenario:
    devices: dict[str, Device]
    # Successive answers of default_output_id; the last one repeats. The run stops once `final` runs dry.
    output_ids: tuple[str, ...]
    final: str


def noise(seconds: float, rate: int, seed: int) -> list[Item]:
    chunk: int = int(rate * CHUNK_SECONDS)
    audio: FloatArray = np.random.default_rng(seed).normal(0.0, 0.1, size=int(seconds * rate)).astype(np.float32)
    return [audio[i : i + chunk] for i in range(0, audio.size - chunk + 1, chunk)]


def reads(items: list[Item]) -> int:
    return sum(not isinstance(item, TrackInfo) for item in items)


def run_scenario(scenario: Scenario, monkeypatch: pytest.MonkeyPatch) -> list[ListenerEvent]:
    stop = threading.Event()
    session: queue.Queue[SessionUpdate] = queue.Queue()
    id_calls: list[int] = [0]

    def fake_output_id() -> str:
        answer: str = scenario.output_ids[min(id_calls[0], len(scenario.output_ids) - 1)]
        id_calls[0] += 1
        return answer

    class FakeCapture:
        def __init__(self, chunk_seconds: float) -> None:
            self.overflow_count: int = 0
            self._device_id: str = scenario.output_ids[min(id_calls[0] - 1, len(scenario.output_ids) - 1)]
            self._items: list[Item] = list(scenario.devices[self._device_id].items)

        def start(self) -> CaptureFormat:
            return CaptureFormat(device_name=self._device_id, sample_rate=scenario.devices[self._device_id].rate, channels=1)

        def read(self, timeout_s: float) -> FloatArray | None:
            while self._items and isinstance(marker := self._items[0], TrackInfo):
                session.put(SessionUpdate(marker))
                self._items.pop(0)
            if not self._items:
                if self._device_id == scenario.final:
                    stop.set()
                return None
            item: Item = self._items.pop(0)
            assert not isinstance(item, TrackInfo)
            return item

        def is_active(self) -> bool:
            return True

        def close(self) -> None:
            pass

    class FakeWatcher:
        def __init__(self, poll_seconds: float, timeout_seconds: float) -> None:
            self.events: queue.Queue[SessionUpdate] = session

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    monkeypatch.setattr(listener, "LoopbackCapture", FakeCapture)
    monkeypatch.setattr(listener, "MediaSessionWatcher", FakeWatcher)
    monkeypatch.setattr(listener, "default_output_id", fake_output_id)
    monkeypatch.setattr(listener, "DEVICE_POLL_SECONDS", 0.0)
    monkeypatch.setattr(listener, "REOPEN_SECONDS", 0.0)

    config: ListenConfig = replace(to_listen_config(DEFAULT_SETTINGS, Path("unused"), frozenset()), recordings_dir=None)
    events: list[ListenerEvent] = []
    worker = threading.Thread(target=run_listener, args=(config, queue.Queue(), load_model(DEFAULT_MODEL_PATH), events.append, stop))
    worker.start()
    worker.join(RUN_TIMEOUT_SECONDS)
    stop.set()
    assert not worker.is_alive(), "run_listener did not finish"
    return events


def test_tracks_are_evaluated_and_changed(monkeypatch: pytest.MonkeyPatch) -> None:
    items: list[Item] = [TRACK_A, *noise(3.0, RATE, 0), TRACK_B, *noise(3.0, RATE, 1)]
    events = run_scenario(Scenario({"A": Device(RATE, tuple(items))}, ("A",), "A"), monkeypatch)
    assert [e for e in events if isinstance(e, CaptureStarted)] == [CaptureStarted("A", RATE, 1)]
    assert [e.track for e in events if isinstance(e, TrackChanged)] == [TRACK_A, TRACK_B]
    evaluated: set[TrackInfo | None] = {e.track for e in events if isinstance(e, Evaluation)}
    assert evaluated == {TRACK_A, TRACK_B}


def test_silence_resets_the_track(monkeypatch: pytest.MonkeyPatch) -> None:
    items: list[Item] = [*noise(2.0, RATE, 0), None, *noise(2.0, RATE, 1)]
    events = run_scenario(Scenario({"A": Device(RATE, tuple(items))}, ("A",), "A"), monkeypatch)
    assert TrackReset("silence") in events


def test_capture_follows_the_default_output(monkeypatch: pytest.MonkeyPatch) -> None:
    first: list[Item] = noise(2.0, RATE, 0)
    devices = {"A": Device(RATE, tuple(first)), "B": Device(48000, tuple(noise(2.0, 48000, 1)))}
    events = run_scenario(Scenario(devices, ("A",) * reads(first) + ("B",), "B"), monkeypatch)
    started: list[CaptureStarted] = [e for e in events if isinstance(e, CaptureStarted)]
    assert started == [CaptureStarted("A", RATE, 1), CaptureStarted("B", 48000, 1)]
    switch: int = events.index(started[1])
    assert TrackReset("output device changed") in events[:switch]
    assert any(isinstance(e, Evaluation) for e in events[switch:])


def test_unusable_output_pauses_until_it_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    devices = {"low": Device(LOW_RATE, ()), "B": Device(RATE, tuple(noise(2.0, RATE, 0)))}
    events = run_scenario(Scenario(devices, ("low",) * 5 + ("B",), "B"), monkeypatch)
    paused: list[CapturePaused] = [e for e in events if isinstance(e, CapturePaused)]
    assert len(paused) == 1, "the same reason is reported once, not on every retry"
    assert [e for e in events if isinstance(e, CaptureStarted)] == [CaptureStarted("B", RATE, 1)]
    assert events.index(paused[0]) < events.index(CaptureStarted("B", RATE, 1))
