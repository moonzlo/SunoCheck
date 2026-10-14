"""Mean fakeprint across many tracks per label: which exact frequencies are consistently peaked in Suno?

Usage: uv run python research/crystal_view.py <variant> <low_hz> <high_hz>
"""

import sys

import numpy as np
from scipy.signal import find_peaks

from corpus import Label, cache_path
from lab import FREQS, all_items, band_bins, fakeprint

TOP: int = 40


def mean_fakeprint(variant: str, label: Label, bins: np.ndarray) -> tuple[np.ndarray, int]:
    prints = []
    for item in all_items():
        path = cache_path(variant, item.item_id)
        if item.label != label or not path.exists():
            continue
        mid = np.load(path)["mid"].astype(np.float32)
        prints.append(fakeprint(mid.mean(axis=0, keepdims=True), bins, 10, -60.0, 5.0)[0])
    return np.mean(prints, axis=0), len(prints)


def main(variant: str, low_hz: float, high_hz: float) -> None:
    bins = band_bins(low_hz, high_hz)
    freqs = FREQS[bins]
    for label in (Label.SUNO, Label.HUMAN):
        mean, count = mean_fakeprint(variant, label, bins)
        if count == 0:
            continue
        peaks, props = find_peaks(mean, prominence=0.02)
        order = np.argsort(props["prominences"])[::-1][:TOP]
        top = np.sort(peaks[order])
        print(f"== {label} n={count} mean level {mean.mean():.3f}")
        print("  " + " ".join(f"{freqs[p]:.1f}({mean[p]:.2f})" for p in top))


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))
