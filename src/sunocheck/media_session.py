"""Windows media session (SMTC) watcher.

Used as a track-change signal; the artist only matters for the artist blacklist. The SMTC broker can stop
answering system-wide, so all calls run in a daemon thread with explicit timeouts and the
audio loop only consumes events from a queue.
"""

import queue
import threading
import time
from dataclasses import dataclass
from typing import Protocol

from winrt.windows.foundation import AsyncStatus
from winrt.windows.media.control import GlobalSystemMediaTransportControlsSessionManager as SessionManager

STATUS_POLL_SECONDS: float = 0.05


@dataclass(frozen=True)
class TrackInfo:
    app_id: str
    artist: str
    title: str


def artist_key(artist: str) -> str:
    return artist.strip().casefold()


@dataclass(frozen=True)
class SessionUpdate:
    track: TrackInfo | None


@dataclass(frozen=True)
class SessionUnavailable:
    message: str


SessionEvent = SessionUpdate | SessionUnavailable


class SmtcError(RuntimeError):
    """SMTC broker did not answer in time or reported a failed operation."""


class _StopRequested(Exception):
    """Internal: the watcher was stopped while waiting for an SMTC operation."""


class _AsyncOperation[T](Protocol):
    """Structural type of a pywinrt IAsyncOperation."""

    @property
    def status(self) -> AsyncStatus: ...

    def get_results(self) -> T: ...


def _wait_result[T](operation: _AsyncOperation[T], what: str, timeout_s: float, stop: threading.Event) -> T:
    deadline: float = time.monotonic() + timeout_s
    while operation.status == AsyncStatus.STARTED:
        if stop.is_set():
            raise _StopRequested()
        if time.monotonic() >= deadline:
            raise SmtcError(f"{what} did not complete in {timeout_s:.0f} s (SMTC broker is not responding)")
        time.sleep(STATUS_POLL_SECONDS)
    if operation.status != AsyncStatus.COMPLETED:
        raise SmtcError(f"{what} finished with status {operation.status.name}")
    return operation.get_results()


class MediaSessionWatcher:
    def __init__(self, poll_seconds: float, timeout_seconds: float) -> None:
        self._poll_seconds: float = poll_seconds
        self._timeout_seconds: float = timeout_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="smtc-watcher", daemon=True)
        self.events: queue.Queue[SessionEvent] = queue.Queue()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _read_track(self, manager: SessionManager) -> TrackInfo | None:
        session = manager.get_current_session()
        if session is None:
            return None
        properties = _wait_result(
            session.try_get_media_properties_async(), "TryGetMediaPropertiesAsync", self._timeout_seconds, self._stop
        )
        return TrackInfo(app_id=session.source_app_user_model_id, artist=properties.artist, title=properties.title)

    def _run(self) -> None:
        try:
            manager: SessionManager = _wait_result(
                SessionManager.request_async(), "SessionManager.RequestAsync", self._timeout_seconds, self._stop
            )
            last: TrackInfo | None = None
            first: bool = True
            while not self._stop.is_set():
                current: TrackInfo | None = self._read_track(manager)
                if first or current != last:
                    self.events.put(SessionUpdate(current))
                    last, first = current, False
                self._stop.wait(self._poll_seconds)
        except _StopRequested:
            return
        except SmtcError as error:
            self.events.put(SessionUnavailable(str(error)))
