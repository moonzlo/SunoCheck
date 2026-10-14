"""Top-scoring human windows of the coherent comb scan.

Usage: uv run python research/scan_humans.py <seconds> <variant[,...]> <count>
"""

import sys

from scan_eval import load


def main(seconds: int, variants: list[str], count: int) -> None:
    humans = sorted((r for r in load(seconds, variants) if r.label == 0), key=lambda r: -r.scores["scan_dens"])
    for r in humans[:count]:
        print(f"{r.item_id:<40} {r.variant:<12} base {r.base_hz:8.3f}  scan_dens {r.scores['scan_dens']:6.2f}"
              f"  scan_sub {r.scores['scan_sub']:6.2f}  fixed_dens {r.scores['fixed_dens']:6.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2].split(","), int(sys.argv[3]))
