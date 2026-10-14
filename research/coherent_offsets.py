"""Diagnose human false positives of the coherent 200 Hz comb: lines at 200k vs 200k+50 vs 200k+100 Hz.

Usage: uv run python research/coherent_offsets.py <top_n>
"""

import sys

import numpy as np
from scipy import fft

from coherent_prepare import coherent_path, harmonic_contrasts
from corpus import PLAYBACK_RATE, VARIANTS, Label
from lab import all_items
from prepare import decode_variant

OFFSETS_HZ: tuple[float, ...] = (0.0, 50.0, 100.0)
SECONDS: float = 10.0


def offset_profile(item_id: str) -> list[float]:
    item = next(i for i in all_items() if i.item_id == item_id)
    mono = decode_variant(item, VARIANTS[0]).mean(axis=1)
    size = int(SECONDS * PLAYBACK_RATE)
    segment = mono[:size]
    power = np.abs(fft.rfft(segment * np.hanning(size))) ** 2
    bin_hz = PLAYBACK_RATE / size
    shifted = [np.median(harmonic_contrasts(np.roll(power, -int(round(o / bin_hz))), bin_hz, 200.0)) for o in OFFSETS_HZ]
    return shifted


def main(top_n: int) -> None:
    scored: list[tuple[float, str, Label, str]] = []
    for item in all_items():
        path = coherent_path("orig", item.item_id)
        if not path.exists():
            continue
        lines = np.load(path)["lines_10s"]
        if len(lines):
            scored.append((float(np.median(lines, axis=1).max()), item.item_id, item.label, item.path.name))
    humans = sorted([s for s in scored if s[2] == Label.HUMAN], reverse=True)[:top_n]
    sunos = sorted([s for s in scored if s[2] == Label.SUNO])[len([s for s in scored if s[2] == Label.SUNO]) // 2 :][:4]
    for score, item_id, label, name in humans + sunos:
        profile = offset_profile(item_id)
        print(f"{label:<6} {item_id:<22} max window median {score:5.1f} | first 10 s at +0/+50/+100 Hz: " + " ".join(f"{p:5.1f}" for p in profile))


if __name__ == "__main__":
    main(int(sys.argv[1]))
