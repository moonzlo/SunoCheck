"""Pick the live threshold of the fixed 200 Hz coherent comb score (line median minus the 99.5 % density quantile).

Reads the v3 scan cache (research/cache/scan). Reports, per threshold, the human window FPR, the share of
humans with two consecutive windows above it, and the Suno TPR (single window / two consecutive windows).

Usage: uv run python research/comb_threshold.py <seconds>
"""

import sys

import numpy as np

from corpus import RESEARCH, list_items

CACHE = RESEARCH / "cache" / "scan"
THRESHOLDS: tuple[float, ...] = (-3.5, -3.0, -2.5, -2.0, -1.5, -1.0)
VARIANTS: tuple[str, ...] = ("orig", "aac_128", "slowed_hall")


def main(seconds: int) -> None:
    tag = f"{seconds}s"
    scores: dict[tuple[int, str], list[np.ndarray]] = {}
    for item in list_items():
        for variant in VARIANTS:
            path = CACHE / variant / f"{item.item_id}.npz"
            if not path.exists():
                continue
            data = np.load(path)
            if data[f"fixed_plain_{tag}"].size == 0:
                continue
            value = data[f"fixed_plain_{tag}"] - data[f"quantiles_{tag}"][:, 1]
            scores.setdefault((int(item.label == "suno"), variant), []).append(value)
    humans = [s for (label, _), group in scores.items() if label == 0 for s in group]
    human_windows = np.concatenate(humans)
    print(f"{tag}: human tracks {len(humans)}, windows {human_windows.size}, max {human_windows.max():.2f}")
    for threshold in THRESHOLDS:
        pair_fp = np.mean([np.any((s[:-1] > threshold) & (s[1:] > threshold)) for s in humans])
        line = f"  thr {threshold:5.1f}: human window FPR {np.mean(human_windows > threshold):.4f}, tracks with 2 in a row {pair_fp:.4f} |"
        for variant in ("orig", "aac_128"):
            suno = scores[(1, variant)]
            window_tpr = np.mean(np.concatenate(suno) > threshold)
            track_tpr = np.mean([np.any((s[:-1] > threshold) & (s[1:] > threshold)) for s in suno if s.size > 1])
            line += f" {variant} window {window_tpr:.3f} 2-in-row {track_tpr:.3f}"
        print(line)


if __name__ == "__main__":
    main(int(sys.argv[1]))
