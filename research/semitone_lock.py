"""Semitone lock-in (H17): sharpness of the fakeprint logit peak exactly at a semitone pitch ratio.

Suno artifact lines are one bin wide and digitally exact, so a pitch shift by exactly n semitones is undone
only at r = 2^(n/12) (within ~0.04 %); 0.5-2 % away the logit drops back. Human tuning deviates by cents,
so its logit should not peak sharply at the exact ratio.

contrast(n) = max logit over r = 2^(n/12) * exp(k * STEP), |k| <= PEAK_K
            - max logit over PEAK_K * 5 <= |k| <= SIDE_K
(The median of the side was tried first: human logit curves are jagged in r, so peak - median reached +11.)

Usage:
  uv run python research/semitone_lock.py corpus
  uv run python research/semitone_lock.py files <wav> [...]
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from numpy.typing import NDArray

from sunocheck.fakeprint import DEFAULT_MODEL_PATH, FakeprintModel, load_model
from corpus import BINS, PLAYBACK_RATE, Label, list_items
from lab import FREQS, band_bins, load_windows, lofcz_fakeprint
from prepare import spectra

SEMITONES: tuple[int, ...] = (-3, -2, -1, 1, 2, 3)
STEP: float = 0.0002
PEAK_K: int = 2
SIDE_K: int = 100
OFFSETS: NDArray[np.intp] = np.arange(-SIDE_K, SIDE_K + 1)
SIDE: NDArray[np.bool_] = np.abs(OFFSETS) >= PEAK_K * 5
PEAK: NDArray[np.bool_] = np.abs(OFFSETS) <= PEAK_K
BAND: NDArray[np.intp] = band_bins(1000.0, 8000.0)
WINDOW_BLOCKS: tuple[int, ...] = (8,)
RESULTS: Path = Path(__file__).parent / "cache"
CONTRASTS: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 6.0)


def logit_curve(spectrum_db: NDArray[np.float32], center: float, model: FakeprintModel) -> NDArray[np.float64]:
    """Logits at r = center * exp(OFFSETS * STEP); spectrum read at r * f (warp.warped, band only)."""
    ratios = center * np.exp(OFFSETS * STEP)
    positions = np.clip(FREQS[BAND][None, :] * ratios[:, None] / FREQS[1], 0, BINS - 1)
    lower = np.floor(positions).astype(np.intp)
    upper = np.minimum(lower + 1, BINS - 1)
    frac = (positions - lower).astype(np.float32)
    out = np.zeros((ratios.size, BINS), dtype=np.float32)
    out[:, BAND] = spectrum_db[lower] * (1 - frac) + spectrum_db[upper] * frac
    return lofcz_fakeprint(out) @ model.weights + model.bias


def lock(spectrum_db: NDArray[np.float32], semitones: int, model: FakeprintModel) -> tuple[float, float]:
    """(contrast, peak logit) at the ratio that undoes a shift by `semitones` (observed f = 2^(n/12) f)."""
    curve = logit_curve(spectrum_db, 2 ** (semitones / 12), model)
    peak = float(curve[PEAK].max())
    return peak - float(curve[SIDE].max()), peak


def best_lock(spectrum_db: NDArray[np.float32], model: FakeprintModel) -> tuple[int, float, float]:
    results = [(n, *lock(spectrum_db, n, model)) for n in SEMITONES]
    return max(results, key=lambda item: item[1])


def corpus_contrasts(variant: str, label: Label, blocks: int, semitones: tuple[int, ...], model: FakeprintModel) -> NDArray[np.float64]:
    """[windows, 2]: best (contrast, peak) over `semitones` for each first loud window."""
    windows = load_windows([i for i in list_items() if i.label == label], variant, blocks, 1)
    rows = []
    for spectrum in windows.mid:
        rows.append(max((lock(spectrum, n, model) for n in semitones), key=lambda item: item[0]))
    return np.array(rows)


def corpus(model: FakeprintModel) -> None:
    for blocks in WINDOW_BLOCKS:
        print(f"== window {blocks * 2.5:g} s", flush=True)
        humans = np.concatenate([corpus_contrasts(v, Label.HUMAN, blocks, SEMITONES, model) for v in ("orig", "aac_128")])
        saved: dict[str, NDArray[np.float64]] = {"human": humans}
        c = humans[:, 0]
        print(f"  humans n={len(c)}: contrast max {c.max():+.1f}, 99.9% {np.quantile(c, 0.999):+.1f}, 99% {np.quantile(c, 0.99):+.1f},"
              f" median {np.median(c):+.1f}; " + " ".join(f">={t:g}: {(c >= t).sum()}" for t in CONTRASTS), flush=True)
        for variant, shift in (("pitch_up1", 1), ("pitch_down2", -2), ("wsola_up1", 1), ("wsola_down1", -1)):
            suno = corpus_contrasts(variant, Label.SUNO, blocks, SEMITONES, model)
            saved[variant] = suno
            s = suno[:, 0]
            print(f"  suno {variant:<11} n={len(s)}: best over shifts " + " ".join(f">={t:g}: {(s >= t).mean():.2f}" for t in CONTRASTS), flush=True)
        np.savez(RESULTS / f"semitone_lock_{blocks}.npz", **saved)


def files(paths: list[Path], model: FakeprintModel) -> None:
    print(f"{'recording':<62} {'n':>3} {'contrast':>8} {'peak':>6}")
    for path in paths:
        audio, rate = sf.read(path, dtype="float64", always_2d=True)
        mono = audio.mean(axis=1)
        if rate != PLAYBACK_RATE:
            mono = soxr.resample(mono, rate, PLAYBACK_RATE, quality="HQ")
        mean = spectra(np.stack([mono, mono], axis=1))["mid"].astype(np.float32).mean(axis=0)
        n, contrast, peak = best_lock(mean, model)
        print(f"{path.stem[16:76]:<62} {n:+3d} {contrast:+8.1f} {peak:+6.1f}", flush=True)


if __name__ == "__main__":
    loaded = load_model(DEFAULT_MODEL_PATH)
    if sys.argv[1] == "corpus":
        corpus(loaded)
    elif sys.argv[1] == "files":
        files([Path(p) for p in sys.argv[2:]], loaded)
    else:
        raise SystemExit(f"Unknown mode {sys.argv[1]!r}: use 'corpus' or 'files'")
