"""Why does the scan miss the 200 Hz base on Suno orig windows? Compare scanned vs fixed stats.

Usage: uv run python research/scan_debug.py <seconds>
"""

import sys

import numpy as np

from corpus import list_items
from scan_prepare import scan_path


def main(seconds: int) -> None:
    tag = f"{seconds}s"
    hits: list[tuple[float, float, float]] = []
    misses: list[tuple[float, float, float, float]] = []
    miss_ids: list[str] = []
    for item in list_items():
        path = scan_path("orig", item.item_id)
        if item.label != "suno" or not path.exists():
            continue
        data = np.load(path)
        for w in range(data[f"base_{tag}"].size):
            fixed = float(data[f"fixed_plain_{tag}"][w])
            dens = fixed - float(data[f"quantiles_{tag}"][w, 1])
            scanned = float(data[f"plain_{tag}"][w])
            if abs(float(data[f"base_{tag}"][w]) - 200.0) < 0.05:
                hits.append((fixed, dens, scanned))
            else:
                misses.append((fixed, dens, scanned, float(data[f"base_{tag}"][w])))
                miss_ids.append(item.item_id)
    hit_array, miss_array = np.array(hits), np.array(misses)
    print(f"hits {len(hits)}: fixed plain median {np.median(hit_array[:, 0]):.2f}, dens median {np.median(hit_array[:, 1]):.2f}")
    print(f"misses {len(misses)}: fixed plain median {np.median(miss_array[:, 0]):.2f}, dens median {np.median(miss_array[:, 1]):.2f}, "
          f"scanned plain median {np.median(miss_array[:, 2]):.2f}")
    print(f"misses with scanned plain > fixed plain (objective prefers wrong base): {np.mean(miss_array[:, 2] > miss_array[:, 0]):.2f}")
    print(f"misses with dens > -2 (strong fixed comb): {np.mean(miss_array[:, 1] > -2):.2f}")
    for row in miss_array[:15]:
        print(f"  fixed {row[0]:5.2f} dens {row[1]:6.2f} scanned {row[2]:5.2f} at {row[3]:.3f} Hz")
    print(" ".join(sorted(set(miss_ids))[:12]))


if __name__ == "__main__":
    main(int(sys.argv[1]))
