"""Cache speed-robust coherent comb scans per item/variant.

Per loud window (WINDOWS_SECONDS, non-overlapping, up to MAX_WINDOWS): coherent_scan.scan of the mid channel.
Output: research/cache/scan/<variant>/<item>.npz

Usage: uv run python research/scan_prepare.py <workers> <all|source[,...]> <variant[,...]> <item_stride>
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from coherent_prepare import MAX_WINDOWS, SILENT_DBFS
from coherent_scan import ScanResult, scan
from corpus import PLAYBACK_RATE, RESEARCH, VARIANTS, Item, Variant, list_items
from prepare import MIN_SECONDS, DecodeError, decode_variant

SCAN_CACHE: Path = RESEARCH / "cache" / "scan_v4"
WINDOWS_SECONDS: tuple[float, ...] = (5.0, 10.0)


def window_features(mono: np.ndarray, seconds: float) -> dict[str, np.ndarray]:
    size = int(seconds * PLAYBACK_RATE)
    results: list[ScanResult] = []
    starts: list[int] = []
    for start in range(0, mono.size - size + 1, size):
        segment = mono[start : start + size]
        if 10 * np.log10(np.mean(segment**2) + 1e-12) < SILENT_DBFS:
            continue
        results.append(scan(segment))
        starts.append(start)
        if len(results) == MAX_WINDOWS:
            break
    tag = f"{int(seconds)}s"
    return {
        f"start_{tag}": np.array(starts),
        f"base_{tag}": np.array([r.scanned.base_hz for r in results]),
        f"plain_{tag}": np.array([r.scanned.plain for r in results]),
        f"sub_{tag}": np.array([r.scanned.sub for r in results]),
        f"lines_{tag}": np.array([r.scanned.lines for r in results]),
        f"fixed_plain_{tag}": np.array([r.fixed.plain for r in results]),
        f"fixed_sub_{tag}": np.array([r.fixed.sub for r in results]),
        f"fixed_lines_{tag}": np.array([r.fixed.lines for r in results]),
        f"quantiles_{tag}": np.array([r.line_quantiles for r in results]),
    }


def scan_path(variant: str, item_id: str) -> Path:
    return SCAN_CACHE / variant / f"{item_id}.npz"


def process(item: Item, variant: Variant) -> str:
    target = scan_path(variant.name, item.item_id)
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
            if done % 100 == 0:
                print(f"{done}/{len(jobs)} {counts}", flush=True)
    print(f"done {counts}")
    for failure in failures[:20]:
        print("FAILED", failure)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]))
