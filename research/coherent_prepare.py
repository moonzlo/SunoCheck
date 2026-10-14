"""Cache coherent comb-line contrasts per item/variant.

Per window (WINDOWS_SECONDS, non-overlapping, loud only): one Hann-windowed FFT of the mid channel at
44.1 kHz; for each harmonic k*BASE_HZ in 1-15.8 kHz: 10*log10(max of 3 bins at the line / median of the
0.5-5 Hz neighbourhood). Same for the control base. Output: research/cache/coherent/<variant>/<item>.npz

Usage: uv run python research/coherent_prepare.py <workers> <all|source[,...]> <variant[,...]> <item_stride>
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import fft

from corpus import PLAYBACK_RATE, RESEARCH, VARIANTS, Item, Variant, list_items
from prepare import MIN_SECONDS, DecodeError, decode_variant

COHERENT_CACHE: Path = RESEARCH / "cache" / "coherent_v2"
WINDOWS_SECONDS: tuple[float, ...] = (5.0, 10.0)
MAX_WINDOWS: int = 4
BASE_HZ: float = 200.0
CONTROL_HZ: float = 203.0
LOW_HZ: float = 1000.0
HIGH_HZ: float = 15800.0
NEIGHBOUR_HZ: tuple[float, float] = (0.5, 5.0)
SILENT_DBFS: float = -45.0


def harmonic_contrasts(power: np.ndarray, bin_hz: float, base_hz: float) -> np.ndarray:
    lo, hi = max(2, int(NEIGHBOUR_HZ[0] / bin_hz)), int(NEIGHBOUR_HZ[1] / bin_hz)
    harmonics = np.arange(int(np.ceil(LOW_HZ / base_hz)), int(HIGH_HZ // base_hz) + 1)
    out = np.empty(len(harmonics))
    for column, k in enumerate(harmonics):
        center = int(round(k * base_hz / bin_hz))
        line = power[center - 1 : center + 2].max()
        neighbours = np.concatenate([power[center - hi : center - lo], power[center + lo : center + hi]])
        out[column] = 10 * np.log10((line + 1e-30) / (np.median(neighbours) + 1e-30))
    return out


GRID_HZ: float = 50.0
PATCH_HZ: float = 6.0


def grid_patches(power: np.ndarray, bin_hz: float) -> np.ndarray:
    """dB patches of ±PATCH_HZ around every multiple of GRID_HZ in [LOW_HZ, HIGH_HZ]: [lines, 2*half+1]."""
    half = int(round(PATCH_HZ / bin_hz))
    centers = np.rint(np.arange(LOW_HZ, HIGH_HZ + 1, GRID_HZ) / bin_hz).astype(np.intp)
    index = centers[:, None] + np.arange(-half, half + 1)[None, :]
    return (10 * np.log10(power[index] + 1e-30)).astype(np.float16)


def window_features(mono: np.ndarray, seconds: float) -> dict[str, np.ndarray]:
    size = int(seconds * PLAYBACK_RATE)
    taper = np.hanning(size)
    lines: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    patches: list[np.ndarray] = []
    starts: list[int] = []
    for start in range(0, mono.size - size + 1, size):
        segment = mono[start : start + size]
        if 10 * np.log10(np.mean(segment**2) + 1e-12) < SILENT_DBFS:
            continue
        power = np.abs(fft.rfft(segment * taper)) ** 2
        bin_hz = PLAYBACK_RATE / size
        lines.append(harmonic_contrasts(power, bin_hz, BASE_HZ))
        controls.append(harmonic_contrasts(power, bin_hz, CONTROL_HZ))
        patches.append(grid_patches(power, bin_hz))
        starts.append(start)
        if len(lines) == MAX_WINDOWS:
            break
    tag = f"{int(seconds)}s"
    return {
        f"lines_{tag}": np.array(lines, dtype=np.float32),
        f"control_{tag}": np.array(controls, dtype=np.float32),
        f"patches_{tag}": np.array(patches, dtype=np.float16),
        f"starts_{tag}": np.array(starts),
    }


def coherent_path(variant: str, item_id: str) -> Path:
    return COHERENT_CACHE / variant / f"{item_id}.npz"


def process(item: Item, variant: Variant) -> str:
    target = coherent_path(variant.name, item.item_id)
    if target.exists():
        return "cached"
    mono = decode_variant(item, variant).mean(axis=1)
    if mono.size < MIN_SECONDS * PLAYBACK_RATE:
        return "short"
    features: dict[str, np.ndarray] = {}
    for seconds in WINDOWS_SECONDS:
        features.update(window_features(mono, seconds))
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".tmp.npz")
    np.savez(partial, **features)
    partial.replace(target)
    return "ok"


def main(workers: int, sources: str, variants: str, stride: int) -> None:
    items = [i for i in list_items() if sources == "all" or i.source in sources.split(",")][::stride]
    selected = [v for v in VARIANTS if v.name in variants.split(",")]
    jobs = [(item, variant) for item in items for variant in selected]
    print(f"items {len(items)}, jobs {len(jobs)}", flush=True)
    counts: dict[str, int] = {}
    failures: list[str] = []
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(process, item, variant): (item, variant) for item, variant in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            item, variant = futures[future]
            try:
                status = future.result()
            except DecodeError as error:
                status = "failed"
                failures.append(f"{item.item_id}/{variant.name}: {error}")
            counts[status] = counts.get(status, 0) + 1
            if done % 200 == 0:
                print(f"{done}/{len(jobs)} {counts}", flush=True)
    print(f"done {counts}")
    for failure in failures[:20]:
        print("FAILED", failure)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]))
