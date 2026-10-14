"""App comb score (sunocheck.coherent_comb.comb_score) on a window growing from the track start.

Live the app sees a track from its first loud sample; this measures how early the comb can decide. Per item and
variant: leading audio below SILENT_DBFS (0.1 s blocks) is dropped, then the score of the first L seconds for each
L in GROW_SECONDS. Output: research/cache/comb_growing/<variant>/<item>.npz with `seconds` and `scores`.

Usage: uv run python research/comb_growing.py <workers> <variant[,...]> <item_stride>
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from sunocheck.coherent_comb import comb_score
from corpus import PLAYBACK_RATE, RESEARCH, VARIANTS, Item, Variant, list_items
from prepare import DecodeError, decode_variant

GROW_CACHE: Path = RESEARCH / "cache" / "comb_growing"
GROW_SECONDS: tuple[float, ...] = (2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 30.0, 45.0)
SILENT_DBFS: float = -45.0
BLOCK_SECONDS: float = 0.1


def first_loud_sample(mono: np.ndarray) -> int:
    block = int(BLOCK_SECONDS * PLAYBACK_RATE)
    blocks = mono[: mono.size // block * block].reshape(-1, block)
    loud = np.flatnonzero(10 * np.log10(np.mean(blocks**2, axis=1) + 1e-12) >= SILENT_DBFS)
    return int(loud[0]) * block if loud.size else mono.size


def process(item: Item, variant: Variant) -> str:
    target = GROW_CACHE / variant.name / f"{item.item_id}.npz"
    if target.exists():
        return "cached"
    mono = decode_variant(item, variant).mean(axis=1)
    mono = mono[first_loud_sample(mono) :]
    seconds = [s for s in GROW_SECONDS if s * PLAYBACK_RATE <= mono.size]
    scores = [comb_score(mono[: int(s * PLAYBACK_RATE)].astype(np.float32), PLAYBACK_RATE) for s in seconds]
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".tmp.npz")
    np.savez(partial, seconds=np.array(seconds), scores=np.array(scores))
    partial.replace(target)
    return "ok"


def main(workers: int, variants: str, stride: int) -> None:
    items = list_items()[::stride]
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
            if done % 50 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} {counts}", flush=True)
    for failure in failures[:20]:
        print("FAILED", failure)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]))
