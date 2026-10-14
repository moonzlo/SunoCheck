"""Coherent line test: one long FFT per 10 s window (0.1 Hz bins at 44.1 kHz); are the k*200 Hz lines
phase-stable (much narrower and higher than their 0.5-5 Hz neighbourhood)?
Usage: uv run python research/coherent_lines.py <variant> <base_hz> <item_id> [...]
"""

import sys

import numpy as np

from corpus import PLAYBACK_RATE, VARIANTS, list_items
from prepare import decode_variant

WINDOW_SECONDS: float = 10.0
LOW_HZ: float = 1000.0
HIGH_HZ: float = 15800.0
NEIGHBOUR_HZ: tuple[float, float] = (0.5, 5.0)


def line_contrasts(segment: np.ndarray, base_hz: float) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(segment.size))) ** 2
    bin_hz = PLAYBACK_RATE / segment.size
    lo, hi = int(NEIGHBOUR_HZ[0] / bin_hz), int(NEIGHBOUR_HZ[1] / bin_hz)
    contrasts = []
    for k in range(int(np.ceil(LOW_HZ / base_hz)), int(HIGH_HZ // base_hz) + 1):
        center = int(round(k * base_hz / bin_hz))
        line = spectrum[center - 1 : center + 2].max()
        neighbours = np.concatenate([spectrum[center - hi : center - lo], spectrum[center + lo : center + hi]])
        contrasts.append(10 * np.log10(line / np.median(neighbours)))
    return np.array(contrasts)


def main(variant_name: str, base_hz: float, item_ids: list[str]) -> None:
    variant = next(v for v in VARIANTS if v.name == variant_name)
    items = {i.item_id: i for i in list_items()}
    window = int(WINDOW_SECONDS * PLAYBACK_RATE)
    for item_id in item_ids:
        mono = decode_variant(items[item_id], variant).mean(axis=1)
        rows = [line_contrasts(mono[s : s + window], base_hz) for s in range(0, mono.size - window + 1, window)]
        medians = " ".join(f"{np.median(r):5.1f}" for r in rows)
        top = " ".join(f"{np.quantile(r, 0.9):5.1f}" for r in rows)
        print(f"{item_id:<24} {items[item_id].label:<6} median line contrast dB: {medians} | p90: {top}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), sys.argv[3:])
