"""Which simulated pitch-shift tool do live evasions resemble? (H19)

One LR per Suno shift variant (vs the same humans as shifted_model.py), each scored on live recordings with the
corpus pipeline (prepare.spectra -> mean -> exact undo of one semitone down, best over +-2 steps of 0.02 %).
Also prints the grouped-CV human max per variant model so live scores can be read against it.

Usage: uv run python research/shift_tool_match.py <wav> [...]
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from corpus import PLAYBACK_RATE, Label, list_items
from lab import load_windows, lofcz_fakeprint
from prepare import spectra
from shifted_model import MAX_WINDOWS, SEMITONES, SUNO_SHIFTS, WINDOW_BLOCKS
from warp import warped

UNDO_RATIOS: NDArray[np.float64] = 2 ** (-1 / 12) * np.exp(np.arange(-2, 3) * 0.0002)


def human_set() -> tuple[NDArray[np.float32], NDArray[np.str_]]:
    humans = [i for i in list_items() if i.label == Label.HUMAN]
    rows: list[NDArray[np.float32]] = []
    groups: list[NDArray[np.str_]] = []
    for variant in ("orig", "aac_128"):
        w = load_windows(humans, variant, WINDOW_BLOCKS, MAX_WINDOWS)
        for n in SEMITONES:
            rows.append(lofcz_fakeprint(warped(w.mid, 2 ** (n / 12))))
            groups.append(w.item_id)
    return np.concatenate(rows), np.concatenate(groups)


def live_fakeprints(path: Path) -> NDArray[np.float32]:
    audio, rate = sf.read(path, dtype="float64", always_2d=True)
    mono = audio.mean(axis=1)
    if rate != PLAYBACK_RATE:
        mono = soxr.resample(mono, rate, PLAYBACK_RATE, quality="HQ")
    mean = spectra(np.stack([mono, mono], axis=1))["mid"].astype(np.float32).mean(axis=0, keepdims=True)
    return np.concatenate([lofcz_fakeprint(warped(mean, r)) for r in UNDO_RATIOS])


def main(paths: list[Path]) -> None:
    human_x, human_groups = human_set()
    suno = [i for i in list_items() if i.label == Label.SUNO]
    live = {p.stem[16:56]: live_fakeprints(p) for p in paths}
    header = f"{'variant':<12} {'cv human max':>12} " + " ".join(f"{name[:14]:>14}" for name in live)
    print(header, flush=True)
    for variant, shift in SUNO_SHIFTS:
        w = load_windows(suno, variant, WINDOW_BLOCKS, MAX_WINDOWS)
        suno_x = lofcz_fakeprint(warped(w.mid, 2 ** (shift / 12)))
        x = np.concatenate([human_x, suno_x])
        y = np.concatenate([np.zeros(len(human_x), dtype=np.int8), np.ones(len(suno_x), dtype=np.int8)])
        groups = np.concatenate([human_groups, w.item_id])
        scores = np.empty(len(y))
        for train, test in GroupKFold(n_splits=5).split(x, y, groups):
            scores[test] = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(x[train], y[train]).decision_function(x[test])
        model = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(x, y)
        cells = " ".join(f"{float(model.decision_function(f).max()):>+14.1f}" for f in live.values())
        print(f"{variant:<12} {scores[y == 0].max():>+12.1f} {cells}", flush=True)


if __name__ == "__main__":
    main([Path(p) for p in sys.argv[1:]])
