"""Scan the comb base frequency: which spacing separates Suno from human best, per codec variant?

Also controls: the same score at off-grid bases (should be ~0 for both labels).
Usage: uv run python research/comb_scan.py <window_seconds> <low_hz> <high_hz>
"""

import sys

import numpy as np

from comb import comb_score
from corpus import BLOCK_SECONDS, VARIANT_NAMES
from lab import all_items, load_windows, summary

BASES_HZ: tuple[float, ...] = (100.0, 150.0, 187.5, 197.0, 200.0, 203.0, 250.0, 300.0, 400.0, 500.0)


def main(window_seconds: float, low_hz: float, high_hz: float) -> None:
    items = all_items()
    blocks = max(1, round(window_seconds / BLOCK_SECONDS))
    for variant in VARIANT_NAMES:
        w = load_windows(items, variant, blocks, 3)
        print(f"== {variant}: suno {int(w.label.sum())} human {int((w.label == 0).sum())} windows")
        for base in BASES_HZ:
            score = comb_score(w.mid, base, low_hz, high_hz)
            suno, human = score[w.label == 1], score[w.label == 0]
            print(f"  base {base:6.1f}: suno med {np.median(suno):5.2f}  human med {np.median(human):5.2f} p99 {np.quantile(human, 0.99):5.2f} | {summary(score, w.label)}")


if __name__ == "__main__":
    main(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]))
