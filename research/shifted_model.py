"""Shifted-Suno model (H18, H19): logistic regression on the lofcz fakeprint after an exact semitone undo.

Pitch shifting smears Suno artifact peaks, and the lofcz weights (trained on unshifted audio) barely see the
smeared pattern (H16/H17). Positives: Suno shift variants warped back by the exact ratio. Negatives: humans
(orig, AAC 128) warped by every semitone ratio of SEMITONES, since live every human is scored at those ratios.
Grouped 5-fold CV by item, reported per window length.

Shift sets: "all" = rubberband + WSOLA (H18); "wsola" = resample + time-stretch only, the tool live evasions
match (H19, research/shift_tool_match.py).

Usage:
  uv run python research/shifted_model.py cv <all|wsola> <windows: blocks:max[,blocks:max...]>
  uv run python research/shifted_model.py fit <all|wsola> <windows> <C> <weights.npz>
  (the app reads src/sunocheck/models/shifted_weights.npz)
"""

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from corpus import BLOCK_SECONDS, Label, list_items
from lab import load_windows, lofcz_fakeprint
from warp import warped

SEMITONES: tuple[int, ...] = (-3, -2, -1, 1, 2, 3)
SHIFT_SETS: dict[str, tuple[tuple[str, int], ...]] = {
    "all": (("pitch_up1", 1), ("pitch_down2", -2), ("wsola_up1", 1), ("wsola_down1", -1)),
    "wsola": (("wsola_up1", 1), ("wsola_down1", -1)),
}
SUNO_SHIFTS: tuple[tuple[str, int], ...] = SHIFT_SETS["all"]
WINDOW_BLOCKS: int = 8
MAX_WINDOWS: int = 2
C_VALUES: tuple[float, ...] = (0.1, 1.0, 10.0)


class Dataset:
    """Rows: fakeprints; labels; CV groups; window index (a human window has len(SEMITONES) rows); window blocks."""

    def __init__(self) -> None:
        self.x: list[NDArray[np.float32]] = []
        self.y: list[NDArray[np.int8]] = []
        self.groups: list[NDArray[np.str_]] = []
        self.windows: list[NDArray[np.intp]] = []
        self.blocks: list[NDArray[np.intp]] = []
        self.next_window = 0

    def add(self, rows: NDArray[np.float32], label: int, groups: NDArray[np.str_], first_window: int, blocks: int) -> None:
        self.x.append(rows)
        self.y.append(np.full(len(rows), label, dtype=np.int8))
        self.groups.append(groups)
        self.windows.append(first_window + np.arange(len(rows)))
        self.blocks.append(np.full(len(rows), blocks, dtype=np.intp))


def dataset(shifts: tuple[tuple[str, int], ...], specs: tuple[tuple[int, int], ...]) -> tuple[NDArray[np.float32], ...]:
    data = Dataset()
    humans = [i for i in list_items() if i.label == Label.HUMAN]
    suno = [i for i in list_items() if i.label == Label.SUNO]
    for blocks, max_windows in specs:
        for variant in ("orig", "aac_128"):
            w = load_windows(humans, variant, blocks, max_windows)
            for n in SEMITONES:
                data.add(lofcz_fakeprint(warped(w.mid, 2 ** (n / 12))), 0, w.item_id, data.next_window, blocks)
            data.next_window += len(w.mid)
        for variant, shift in shifts:
            w = load_windows(suno, variant, blocks, max_windows)
            data.add(lofcz_fakeprint(warped(w.mid, 2 ** (shift / 12))), 1, w.item_id, data.next_window, blocks)
            data.next_window += len(w.mid)
        print(f"  windows of {blocks * BLOCK_SECONDS:g} s loaded", flush=True)
    return tuple(np.concatenate(part) for part in (data.x, data.y, data.groups, data.windows, data.blocks))


def per_window(scores: NDArray[np.float64], windows: NDArray[np.intp]) -> NDArray[np.float64]:
    """Max score per window (a human window is flagged if any of its semitone undos fires)."""
    best = np.full(windows.max() + 1, -np.inf)
    np.maximum.at(best, windows, scores)
    return best[np.unique(windows)]


def parse_specs(text: str) -> tuple[tuple[int, int], ...]:
    return tuple((int(b), int(m)) for b, m in (part.split(":") for part in text.split(",")))


def cv(shifts: tuple[tuple[str, int], ...], specs: tuple[tuple[int, int], ...]) -> None:
    x, y, groups, windows, blocks = dataset(shifts, specs)
    for c_value in C_VALUES:
        scores = np.empty(len(y))
        for train, test in GroupKFold(n_splits=5).split(x, y, groups):
            model = LogisticRegression(C=c_value, max_iter=5000, class_weight="balanced").fit(x[train], y[train])
            scores[test] = model.decision_function(x[test])
        for length in sorted(set(blocks.tolist())):
            sel = blocks == length
            human = per_window(scores[sel & (y == 0)], windows[sel & (y == 0)])
            suno = scores[sel & (y == 1)]
            top = human.max()
            print(f"C={c_value:<5} {length * BLOCK_SECONDS:>4g} s: humans n={human.size} max {top:+.2f}, 99.9% {np.quantile(human, 0.999):+.2f}"
                  f" | suno n={suno.size}: > human max {np.mean(suno > top):.2f}, > max+2 {np.mean(suno > top + 2):.2f}", flush=True)


def fit(shifts: tuple[tuple[str, int], ...], specs: tuple[tuple[int, int], ...], c_value: float, path: Path) -> None:
    x, y, _, _, _ = dataset(shifts, specs)
    model = LogisticRegression(C=c_value, max_iter=5000, class_weight="balanced").fit(x, y)
    np.savez(path, weights=model.coef_.astype(np.float32), bias=model.intercept_.astype(np.float32))
    print(f"saved {path}")


if __name__ == "__main__":
    mode, shift_set, windows_text = sys.argv[1], sys.argv[2], sys.argv[3]
    if mode == "cv":
        cv(SHIFT_SETS[shift_set], parse_specs(windows_text))
    elif mode == "fit":
        fit(SHIFT_SETS[shift_set], parse_specs(windows_text), float(sys.argv[4]), Path(sys.argv[5]))
    else:
        raise SystemExit(f"Unknown mode {mode!r}: use 'cv' or 'fit'")
