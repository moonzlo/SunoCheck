"""Shared analysis helpers: cached spectra -> windows -> features -> metrics."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import minimum_filter1d

from corpus import BIN_HZ, BINS, BLOCK_SECONDS, Item, Label, cache_path, list_items

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]

SILENT_BLOCK_DB: float = -45.0
LOFCZ_LEVEL_OFFSET_DB: float = -3.0
FREQS: FloatArray = np.arange(BINS) * BIN_HZ


@dataclass(frozen=True)
class Windows:
    """Window-mean spectra of one variant; rows aligned across all arrays."""

    mid: NDArray[np.float32]
    side: NDArray[np.float32]
    native: NDArray[np.float32]
    label: NDArray[np.int8]
    group: NDArray[np.str_]
    item_id: NDArray[np.str_]
    source: NDArray[np.str_]
    version: NDArray[np.str_]
    start_block: IntArray


def load_windows(items: list[Item], variant: str, window_blocks: int, max_windows: int) -> Windows:
    mids: list[NDArray[np.float32]] = []
    sides: list[NDArray[np.float32]] = []
    natives: list[NDArray[np.float32]] = []
    meta: list[tuple[int, str, str, str, str, int]] = []
    for item in items:
        path = cache_path(variant, item.item_id)
        if not path.exists():
            continue
        data = np.load(path)
        mid = data["mid"].astype(np.float32)
        side = data["side"].astype(np.float32)
        loud = data["block_db"]
        starts = [s for s in range(0, mid.shape[0] - window_blocks + 1, window_blocks) if loud[s : s + window_blocks].mean() > SILENT_BLOCK_DB]
        for start in starts[:max_windows]:
            mids.append(mid[start : start + window_blocks].mean(axis=0))
            sides.append(side[start : start + window_blocks].mean(axis=0))
            natives.append(data["native"])
            meta.append((int(item.label == Label.SUNO), item.group, item.item_id, item.source, item.version, start))
    columns = list(zip(*meta))
    return Windows(
        mid=np.stack(mids),
        side=np.stack(sides),
        native=np.stack(natives),
        label=np.array(columns[0], dtype=np.int8),
        group=np.array(columns[1]),
        item_id=np.array(columns[2]),
        source=np.array(columns[3]),
        version=np.array(columns[4]),
        start_block=np.array(columns[5], dtype=np.intp),
    )


def band_bins(low_hz: float, high_hz: float) -> IntArray:
    return np.flatnonzero((FREQS >= low_hz) & (FREQS <= high_hz))


def fakeprint(spectra_db: NDArray[np.float32], bins: IntArray, hull_size: int, floor_db: float, cap_db: float) -> NDArray[np.float32]:
    """Deezer/lofcz residue: band dB minus lower hull, clipped to [0, cap], normalized by row max."""
    band = spectra_db[:, bins].astype(np.float64)
    hull = np.clip(minimum_filter1d(band, size=hull_size, axis=1, mode="nearest"), floor_db, None)
    residue = np.clip(band - hull, 0.0, cap_db)
    return (residue / (residue.max(axis=1, keepdims=True) + 1e-6)).astype(np.float32)


def lofcz_fakeprint(spectra_db: NDArray[np.float32]) -> NDArray[np.float32]:
    return fakeprint(spectra_db + LOFCZ_LEVEL_OFFSET_DB, band_bins(1000.0, 8000.0), 10, -45.0, 5.0)


def auroc(scores: FloatArray, labels: NDArray[np.int8]) -> float:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    sorted_scores = scores[order]
    for value in np.unique(sorted_scores[np.r_[False, np.diff(sorted_scores) == 0]]):
        tie = scores == value
        ranks[tie] = ranks[tie].mean()
    positives = labels == 1
    n_pos, n_neg = positives.sum(), (~positives).sum()
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def tpr_at_fpr(scores: FloatArray, labels: NDArray[np.int8], fpr: float) -> float:
    threshold = np.quantile(scores[labels == 0], 1.0 - fpr)
    return float(np.mean(scores[labels == 1] > threshold))


def summary(scores: FloatArray, labels: NDArray[np.int8]) -> str:
    return f"AUROC {auroc(scores, labels):.4f}  TPR@1%FPR {tpr_at_fpr(scores, labels, 0.01):.3f}  TPR@0.1%FPR {tpr_at_fpr(scores, labels, 0.001):.3f}"


def all_items() -> list[Item]:
    return list_items()
