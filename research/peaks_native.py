"""Where exactly do the residual spectral peaks sit at the native rate, and are they shared across Suno tracks?

Mean log-power spectrum (n_fft 16384 at 48 kHz, ~2.9 Hz bins) over the whole track, lower-hull residue,
then peaks listed per file and a frequency histogram across files.
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from numpy.typing import NDArray
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import find_peaks

RATE: int = 48000
N_FFT: int = 16384
HOP: int = 8192
MAX_SECONDS: float = 120.0
HULL_BINS: int = 30
PROMINENCE_DB: float = 1.5

FloatArray = NDArray[np.float64]


def load_mono(path: Path) -> FloatArray:
    data, rate = sf.read(path, dtype="float64", always_2d=True, frames=int(MAX_SECONDS * 48000))
    mono = data.mean(axis=1)
    return mono if rate == RATE else soxr.resample(mono, rate, RATE, quality="VHQ")


def mean_db(audio: FloatArray) -> FloatArray:
    window = np.hanning(N_FFT + 1)[:-1]
    frames = np.lib.stride_tricks.sliding_window_view(audio, N_FFT)[::HOP]
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    return (10.0 * np.log10(np.clip(power, 1e-12, None))).mean(axis=0)


def residue(spectrum_db: FloatArray) -> FloatArray:
    smooth = uniform_filter1d(spectrum_db, 5)
    hull = uniform_filter1d(minimum_filter1d(smooth, HULL_BINS), HULL_BINS)
    return spectrum_db - hull


def main(paths: list[Path]) -> None:
    freqs = np.fft.rfftfreq(N_FFT, 1 / RATE)
    all_peaks: list[FloatArray] = []
    for path in paths:
        res = residue(mean_db(load_mono(path)))
        band = (freqs > 500) & (freqs < 23000)
        peaks, props = find_peaks(np.where(band, res, 0.0), prominence=PROMINENCE_DB, distance=20)
        order = np.argsort(props["prominences"])[::-1][:25]
        top = peaks[order]
        all_peaks.append(freqs[top])
        listing = " ".join(f"{freqs[p]:.0f}({props['prominences'][o]:.1f})" for p, o in zip(top, order))
        print(f"{path.name[:40]:<40} n={len(peaks):3d} | {listing}")
    edges = np.arange(0, 24001, 250)
    counts = np.zeros(len(edges) - 1, dtype=int)
    for peaks_hz in all_peaks:
        counts += np.histogram(peaks_hz, edges)[0] > 0
    common = np.argsort(counts)[::-1][:30]
    print("shared 250 Hz bins (files with a top-25 peak there):")
    print(" ".join(f"{edges[i]:.0f}:{counts[i]}" for i in sorted(common) if counts[i] > 1))


if __name__ == "__main__":
    main([Path(p) for p in sys.argv[1:]])
