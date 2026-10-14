"""Evaluate the growing-window comb (cache from comb_growing.py).

Per window length: human max / 99.9 % and Suno share above the threshold. Then decision rules over checkpoints:
fire at the first checkpoint L >= start whose score is >= threshold (optionally also the previous checkpoint).
Reports human tracks fired (any variant) and the cumulative share of Suno fired by each checkpoint.

Usage: uv run python research/comb_growing_eval.py
"""

from pathlib import Path

import numpy as np

from comb_growing import GROW_CACHE, GROW_SECONDS
from corpus import Label, list_items

VARIANTS: tuple[str, ...] = ("orig", "aac_128")
THRESHOLDS: tuple[float, ...] = (-3.5, -3.0, -2.75, -2.5, -2.25, -2.0)
STARTS: tuple[float, ...] = (2.5, 5.0, 7.5, 10.0)
SCAN_CACHE: Path = Path(__file__).parent / "cache" / "scan"
PAIR_THRESHOLD: float = -2.5
data_ids: dict[tuple[str, str], list[tuple[str, dict[float, float]]]] = {}


def load() -> dict[tuple[str, str], list[dict[float, float]]]:
    """(label, variant) -> per track {seconds: score}."""
    labels: dict[str, str] = {item.item_id: item.label.value for item in list_items()}
    out: dict[tuple[str, str], list[dict[float, float]]] = {}
    for variant in VARIANTS:
        for path in sorted((GROW_CACHE / variant).glob("*.npz")):
            if path.name.endswith(".tmp.npz"):
                continue
            data = np.load(path)
            track = {float(s): float(v) for s, v in zip(data["seconds"], data["scores"])}
            out.setdefault((labels[path.stem], variant), []).append(track)
    return out


def first_fire(track: dict[float, float], threshold: float, start: float, pair: bool) -> float | None:
    previous: float | None = None
    for seconds in GROW_SECONDS:
        if seconds not in track:
            return None
        score = track[seconds]
        above = score >= threshold
        if seconds >= start and above and (not pair or (previous is not None and previous >= threshold)):
            return seconds
        previous = score
    return None


def load_ids() -> dict[tuple[str, str], list[tuple[str, dict[float, float]]]]:
    labels: dict[str, str] = {item.item_id: item.label.value for item in list_items()}
    out: dict[tuple[str, str], list[tuple[str, dict[float, float]]]] = {}
    for variant in VARIANTS:
        for path in sorted((GROW_CACHE / variant).glob("*.npz")):
            if path.name.endswith(".tmp.npz"):
                continue
            data = np.load(path)
            out.setdefault((labels[path.stem], variant), []).append(
                (path.stem, {float(s): float(v) for s, v in zip(data["seconds"], data["scores"])})
            )
    return out


def pair_rule_times() -> dict[tuple[str, str], float]:
    """(variant, item) -> seconds when two consecutive 10 s windows first both reach PAIR_THRESHOLD."""
    out: dict[tuple[str, str], float] = {}
    for variant in VARIANTS:
        for path in (SCAN_CACHE / variant).glob("*.npz"):
            data = np.load(path)
            if data["fixed_plain_10s"].size < 2:
                continue
            scores = data["fixed_plain_10s"] - data["quantiles_10s"][:, 1]
            hits = np.flatnonzero((scores[:-1] >= PAIR_THRESHOLD) & (scores[1:] >= PAIR_THRESHOLD))
            if hits.size:
                out[(variant, path.stem)] = float((hits[0] + 2) * 10)
    return out


def main() -> None:
    data = load()
    global data_ids
    data_ids = load_ids()
    for key, tracks in sorted(data.items()):
        print(f"{key}: {len(tracks)} tracks")
    print("\nper window length: human max | human 99.9% | Suno orig >= -2.5 | Suno aac_128 >= -2.5")
    for seconds in GROW_SECONDS:
        humans = [t[seconds] for v in VARIANTS for t in data.get((Label.HUMAN.value, v), []) if seconds in t]
        line = f"  {seconds:5.1f} s: n={len(humans):4d}"
        if humans:
            line += f" max {max(humans):6.2f}  99.9% {np.percentile(humans, 99.9):6.2f}"
        for variant in VARIANTS:
            suno = [t[seconds] for t in data.get((Label.SUNO.value, variant), []) if seconds in t]
            if suno:
                line += f" | {variant} {np.mean(np.array(suno) >= -2.5):.3f}"
        print(line)

    print("\nper-length thresholds = human max(L) + margin, checkpoints up to 30 s (any checkpoint fires)")
    usable = [s for s in GROW_SECONDS if s <= 30.0]
    humans_all = [t for v in VARIANTS for t in data.get((Label.HUMAN.value, v), [])]
    human_max = {s: max(t[s] for t in humans_all if s in t) for s in usable}
    for margin in (0.0, 0.25, 0.5, 1.0):
        thresholds = {s: human_max[s] + margin for s in usable}
        near = sum(any(s in t and t[s] >= thresholds[s] - 0.5 for s in usable) for t in humans_all)
        print(f"  margin {margin:4.2f}: thr " + " ".join(f"{s:g}s:{thresholds[s]:+.2f}" for s in usable)
              + f" | humans within 0.5 dB of a threshold: {near}/{len(humans_all)}")
        for variant in VARIANTS:
            suno = data.get((Label.SUNO.value, variant), [])
            fired = [next((s for s in usable if s in t and t[s] >= thresholds[s]), None) for t in suno]
            shares = [np.mean([x is not None and x <= s for x in fired]) for s in usable]
            print(f"    {variant:8} " + "  ".join(f"{s:g}s:{x:.3f}" for s, x in zip(usable, shares)))

    print("\nunion with the pair rule (two consecutive 10 s windows >= -2.5, fires at (i+2)*10 s; v3 scan cache)")
    pair_times = pair_rule_times()
    for margin in (0.25, 0.5):
        thresholds = {s: human_max[s] + margin for s in usable}
        for label in (Label.HUMAN.value, Label.SUNO.value):
            for variant in VARIANTS:
                ids = [i for i in data_ids.get((label, variant), [])]
                fired: list[float | None] = []
                for item_id, track in ids:
                    grow = next((s for s in usable if s in track and track[s] >= thresholds[s]), None)
                    pair = pair_times.get((variant, item_id))
                    fired.append(min([x for x in (grow, pair) if x is not None], default=None))
                if label == Label.HUMAN.value:
                    print(f"  margin {margin}: humans {variant} fired {sum(x is not None for x in fired)}/{len(fired)}")
                else:
                    shares = [np.mean([x is not None and x <= s for x in fired]) for s in usable]
                    print(f"  margin {margin}: suno {variant:8} " + "  ".join(f"{s:g}s:{x:.3f}" for s, x in zip(usable, shares)))

    print("\nrules: human tracks fired (orig+aac) | Suno cumulative share fired by checkpoint")
    header = "  ".join(f"{s:>5.1f}" for s in GROW_SECONDS)
    for pair in (False, True):
        for start in STARTS:
            for threshold in THRESHOLDS:
                humans = [t for v in VARIANTS for t in data.get((Label.HUMAN.value, v), [])]
                fp = sum(first_fire(t, threshold, start, pair) is not None for t in humans)
                print(f"pair={pair!s:5} start={start:4.1f} thr={threshold:5.2f}: humans fired {fp}/{len(humans)}")
                for variant in VARIANTS:
                    suno = data.get((Label.SUNO.value, variant), [])
                    times = [first_fire(t, threshold, start, pair) for t in suno]
                    shares = [np.mean([x is not None and x <= s for x in times]) for s in GROW_SECONDS]
                    print(f"    {variant:8} {'  '.join(f'{x:5.2f}' for x in shares)}")
        print(f"    seconds  {header}")


if __name__ == "__main__":
    main()
