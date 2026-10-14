"""Zoom on the 1500 Hz comb: residue (dB above local median) at 1500k and 1500k±50 Hz for k=1..15.

Welch mean of linear power, n_fft 65536 at 48 kHz (0.73 Hz bins), local median over ±40 Hz as the floor.
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from numpy.typing import NDArray
from scipy.signal import welch

RATE: int = 48000
N_FFT: int = 65536
MAX_SECONDS: float = 120.0
SPACING_HZ: float = 1500.0
OFFSETS_HZ: tuple[float, ...] = (-100.0, -50.0, 0.0, 50.0, 100.0)
FLOOR_HZ: float = 40.0

FloatArray = NDArray[np.float64]


def load_mono(path: Path) -> FloatArray:
    data, rate = sf.read(path, dtype="float64", always_2d=True, frames=int(MAX_SECONDS * 48000))
    mono = data.mean(axis=1)
    return mono if rate == RATE else soxr.resample(mono, rate, RATE, quality="VHQ")


def line_db(freqs: FloatArray, power_db: FloatArray, target_hz: float) -> float:
    bin_hz = freqs[1]
    center = int(round(target_hz / bin_hz))
    peak = power_db[center - 2 : center + 3].max()
    span = int(FLOOR_HZ / bin_hz)
    floor = np.median(power_db[center - span : center + span])
    return float(peak - floor)


def main(paths: list[Path]) -> None:
    header = " ".join(f"{int(o):>+5d}" for o in OFFSETS_HZ)
    for path in paths:
        freqs, power = welch(load_mono(path), RATE, nperseg=N_FFT, noverlap=N_FFT // 2)
        power_db = 10.0 * np.log10(np.clip(power, 1e-20, None))
        print(f"== {path.name[:50]}   offsets: {header}")
        for k in range(1, 16):
            values = [line_db(freqs, power_db, k * SPACING_HZ + o) for o in OFFSETS_HZ]
            print(f"  k={k:2d} {k * SPACING_HZ:7.0f} Hz: " + " ".join(f"{v:5.1f}" for v in values))


if __name__ == "__main__":
    main([Path(p) for p in sys.argv[1:]])
