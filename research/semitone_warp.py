"""Semitone warp search for the lofcz fakeprint (H16): undo a pitch shift of n = -3..+3 semitones.

Per item/variant, first loud window of W blocks: lofcz logit of the spectrum read at 2^(n/12) * f (warp.warped).
Humans (orig, AAC 128) give the false-positive side: the best logit over shifted n != 0. Suno pitch_up1 /
pitch_down2 (rubberband, duration kept) give the hit side: the logit at the shift that undoes it.

Usage: uv run python research/semitone_warp.py
"""

import numpy as np

from sunocheck.fakeprint import DEFAULT_MODEL_PATH, load_model
from corpus import Label, list_items
from lab import load_windows, lofcz_fakeprint
from warp import warped

SEMITONES: tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3)
WINDOW_BLOCKS: tuple[int, ...] = (4, 8, 12)
THRESHOLDS: tuple[float, ...] = (6.0, 8.0, 10.0, 12.0)


def logits(variant: str, label: Label, blocks: int) -> np.ndarray:
    """[windows, len(SEMITONES)] logits."""
    model = load_model(DEFAULT_MODEL_PATH)
    items = [i for i in list_items() if i.label == label]
    windows = load_windows(items, variant, blocks, 1)
    return np.stack(
        [lofcz_fakeprint(warped(windows.mid, 2 ** (n / 12))) @ model.weights + model.bias for n in SEMITONES], axis=1
    )


def main() -> None:
    shifted = [i for i, n in enumerate(SEMITONES) if n != 0]
    for blocks in WINDOW_BLOCKS:
        print(f"== window {blocks * 2.5:g} s")
        humans = np.concatenate([logits(v, Label.HUMAN, blocks) for v in ("orig", "aac_128")])
        human_shifted = humans[:, shifted].max(axis=1)
        print(f"  humans n={len(humans)}: best shifted logit max {human_shifted.max():+.1f}, 99.9% {np.quantile(human_shifted, 0.999):+.1f},"
              f" 99% {np.quantile(human_shifted, 0.99):+.1f}; unshifted max {humans[:, SEMITONES.index(0)].max():+.1f}")
        for variant, undo in (("orig", 0), ("pitch_up1", -1), ("pitch_down2", 2), ("wsola_up1", -1), ("wsola_down1", 1)):
            suno = logits(variant, Label.SUNO, blocks)
            at_undo = suno[:, SEMITONES.index(undo)]
            unshifted = suno[:, SEMITONES.index(0)]
            line = f"  suno {variant:<11} n={len(suno)}: logit at undo n={undo:+d} median {np.median(at_undo):+.1f}, unshifted median {np.median(unshifted):+.1f} |"
            for threshold in THRESHOLDS:
                caught = (unshifted >= threshold) | (suno[:, shifted].max(axis=1) >= threshold)
                line += f" >={threshold:g}: {caught.mean():.2f}"
            print(line)
        for threshold in THRESHOLDS:
            print(f"    humans with best shifted logit >= {threshold:g}: {(human_shifted >= threshold).sum()}")


if __name__ == "__main__":
    main()
