"""Per-track comb strength (whole excerpt) by label/version, and per-harmonic contrast profile.

Usage: uv run python research/comb_profile.py <variant>
"""

import sys
from collections import defaultdict

import numpy as np

from comb import comb_contrast
from corpus import Label, cache_path
from lab import all_items

BASE_HZ: float = 200.0
LOW_HZ: float = 1000.0
HIGH_HZ: float = 15800.0


def main(variant: str) -> None:
    by_version: dict[str, list[float]] = defaultdict(list)
    profiles: dict[str, list[np.ndarray]] = defaultdict(list)
    for item in all_items():
        path = cache_path(variant, item.item_id)
        if not path.exists():
            continue
        data = np.load(path)
        mid = data["mid"].astype(np.float32)
        loud = data["block_db"] > -45.0
        if loud.sum() == 0:
            continue
        spectrum = mid[loud].mean(axis=0, keepdims=True)
        contrast = comb_contrast(spectrum, BASE_HZ, LOW_HZ, HIGH_HZ, 1, 0.5)[0]
        key = f"suno {item.version}" if item.label == Label.SUNO else f"human {item.source}"
        by_version[key].append(float(contrast.mean()))
        profiles[item.label].append(contrast)
    for key, values in sorted(by_version.items()):
        v = np.array(values)
        print(f"{key:<16} n={len(v):4d}  p10 {np.quantile(v, 0.1):5.2f}  med {np.median(v):5.2f}  p90 {np.quantile(v, 0.9):5.2f}  max {v.max():5.2f}  share>1.5dB {np.mean(v > 1.5):.2f}")
    harmonics = np.arange(int(np.ceil(LOW_HZ / BASE_HZ)), int(HIGH_HZ // BASE_HZ) + 1) * BASE_HZ
    suno = np.median(np.array(profiles[Label.SUNO]), axis=0)
    human = np.median(np.array(profiles[Label.HUMAN]), axis=0)
    print("per-harmonic median contrast (suno/human), every 2nd harmonic:")
    print(" ".join(f"{f / 1000:.1f}k:{s:.1f}/{h:.1f}" for f, s, h in list(zip(harmonics, suno, human))[::2]))


if __name__ == "__main__":
    main(sys.argv[1])
