"""WASAPI loopback capture of the default output device."""

import queue
from dataclasses import dataclass

import numpy as np
import pyaudiowpatch as pyaudio
from winrt.windows.media.devices import AudioDeviceRole, MediaDevice

from sunocheck.fakeprint import FloatArray


@dataclass(frozen=True)
class CaptureFormat:
    device_name: str
    sample_rate: int
    channels: int


class LoopbackDeviceNotFoundError(RuntimeError):
    pass


def _default_loopback_device(pa: pyaudio.PyAudio) -> dict[str, str | int | float]:
    wasapi_info: dict[str, str | int] = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    speakers: dict[str, str | int | float] = pa.get_device_info_by_index(int(wasapi_info["defaultOutputDevice"]))
    if speakers["isLoopbackDevice"]:
        return speakers
    for loopback in pa.get_loopback_device_info_generator():
        if str(speakers["name"]) in str(loopback["name"]):
            return loopback
    raise LoopbackDeviceNotFoundError(f"No WASAPI loopback device for default output '{speakers['name']}'")


def default_output_id() -> str:
    """Endpoint id of the default output; changes when the user switches outputs or unplugs the current one.
    A loopback stream stays on the device it was opened on, so the caller reopens capture on a change."""
    return MediaDevice.get_default_audio_render_id(AudioDeviceRole.DEFAULT)


class LoopbackCapture:
    """WASAPI sends no packets while nothing is playing, so read() returns None on timeout."""

    def __init__(self, chunk_seconds: float) -> None:
        self._chunk_seconds: float = chunk_seconds
        self._pa: pyaudio.PyAudio = pyaudio.PyAudio()
        self._queue: queue.Queue[bytes] = queue.Queue()
        self._stream: pyaudio.Stream | None = None
        self._channels: int = 0
        self.overflow_count: int = 0

    def start(self) -> CaptureFormat:
        device: dict[str, str | int | float] = _default_loopback_device(self._pa)
        capture_format = CaptureFormat(
            device_name=str(device["name"]),
            sample_rate=int(device["defaultSampleRate"]),
            channels=int(device["maxInputChannels"]),
        )
        self._channels = capture_format.channels
        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=capture_format.channels,
            rate=capture_format.sample_rate,
            input=True,
            input_device_index=int(device["index"]),
            frames_per_buffer=int(capture_format.sample_rate * self._chunk_seconds),
            stream_callback=self._on_audio,
        )
        return capture_format

    def _on_audio(self, in_data: bytes, frame_count: int, time_info: dict[str, float], status: int) -> tuple[None, int]:
        if status & pyaudio.paInputOverflow:
            self.overflow_count += 1
        self._queue.put(in_data)
        return (None, pyaudio.paContinue)

    def read(self, timeout_s: float) -> FloatArray | None:
        try:
            data: bytes = self._queue.get(timeout=timeout_s)
        except queue.Empty:
            return None
        frames: FloatArray = np.frombuffer(data, dtype=np.float32).reshape(-1, self._channels)
        return frames.mean(axis=1)

    def is_active(self) -> bool:
        """False once PortAudio has stopped the stream, e.g. after the device was invalidated."""
        return self._stream is not None and self._stream.is_active()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
        self._pa.terminate()
