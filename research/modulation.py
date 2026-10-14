"""Envelope modulation spectrum: is there a sharp periodic line (codec frame rate) in Suno band envelopes?

Band energy per 1 ms frame (STFT n_fft 256, hop 48 at 48 kHz), log, high-passed by a 50 ms moving average,
then Welch spectrum of that envelope (10 s segments). Prints the strongest lines (dB above local median)
in 10..480 Hz.
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from numpy.typing import NDArray
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import stft, welch

RATE: int = 48000
N_FFT: int = 256
HOP: int = 48
ENV_RATE: float = RATE / HOP
MAX_SECONDS: float = 90.0
BANDS_HZ: tuple[tuple[float, float], ...] = ((1000.0, 4000.0), (4000.0, 12000.0), (12000.0, 18000.0))
TOP: int = 8

FloatArray = NDArray[np.float64]


def load_mono(path: Path) -> FloatArray:
    data, rate = sf.read(path, dtype="float64", always_2d=True, frames=int(MAX_SECONDS * 48000))
    mono = data.mean(axis=1)
    return mono if rate == RATE else soxr.resample(mono, rate, RATE, quality="VHQ")


def modulation_lines(audio: FloatArray, low_hz: float, high_hz: float) -> list[tuple[float, float]]:
    freqs, _, z = stft(audio, RATE, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=None, padded=False)
    band = (freqs >= low_hz) & (freqs < high_hz)
    envelope = np.log(np.abs(z[band]) ** 2 + 1e-12).mean(axis=0)
    envelope = envelope - uniform_filter1d(envelope, int(0.05 * ENV_RATE))
    mod_freqs, power = welch(envelope, ENV_RATE, nperseg=int(10 * ENV_RATE))
    power_db = 10 * np.log10(power + 1e-20)
    excess = power_db - median_filter(power_db, 41)
    region = (mod_freqs > 10) & (mod_freqs < 480)
    idx = np.flatnonzero(region)
    best = idx[np.argsort(excess[idx])[::-1][:TOP]]
    return [(float(mod_freqs[i]), float(excess[i])) for i in sorted(best)]


def main(paths: list[Path]) -> None:
    for path in paths:
        audio = load_mono(path)
        print(f"== {path.name[:60]}")
        for low, high in BANDS_HZ:
            lines = modulation_lines(audio, low, high)
            print(f"  {low / 1000:4.0f}-{high / 1000:<4.0f}kHz: " + " ".join(f"{f:6.1f}({e:4.1f})" for f, e in lines))


if __name__ == "__main__":
    main([Path(p) for p in sys.argv[1:]])
