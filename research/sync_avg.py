"""Synchronous averaging at the decoder frame period: is the 200 Hz comb a phase-locked repeating waveform?

At 44.1 kHz two 5 ms frames are exactly 441 samples. Fold a high-passed window at period P and measure the
energy of the folded mean relative to folds at nearby wrong periods (which only keep incoherent leftovers).
Usage: uv run python research/sync_avg.py <variant> <item_id> [<item_id> ...]
"""

import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

from corpus import PLAYBACK_RATE, VARIANTS, list_items
from prepare import decode_variant

TRUE_PERIOD: int = 441
CONTROL_PERIODS: tuple[int, ...] = (437, 438, 439, 440, 442, 443, 444, 445)
WINDOW_SECONDS: float = 10.0
HIGHPASS_HZ: float = 1000.0


def folded_energy(signal: np.ndarray, period: int) -> float:
    cycles = signal.size // period
    folded = signal[: cycles * period].reshape(cycles, period).mean(axis=0)
    return float(np.mean(folded**2))


def main(variant_name: str, item_ids: list[str]) -> None:
    variant = next(v for v in VARIANTS if v.name == variant_name)
    items = {i.item_id: i for i in list_items()}
    sos = butter(4, HIGHPASS_HZ, btype="highpass", fs=PLAYBACK_RATE, output="sos")
    window = int(WINDOW_SECONDS * PLAYBACK_RATE)
    for item_id in item_ids:
        mono = decode_variant(items[item_id], variant).mean(axis=1)
        filtered = sosfiltfilt(sos, mono)
        ratios = []
        for start in range(0, filtered.size - window + 1, window):
            segment = filtered[start : start + window]
            control = np.mean([folded_energy(segment, p) for p in CONTROL_PERIODS])
            ratios.append(10 * np.log10(folded_energy(segment, TRUE_PERIOD) / control))
        print(f"{item_id:<26} {items[item_id].label:<6} sync gain dB per 10 s: " + " ".join(f"{r:5.1f}" for r in ratios))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
