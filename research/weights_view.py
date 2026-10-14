"""Where do fakeprint classifiers put their positive weight? Share of weight mass on the 200 Hz grid."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from sunocheck.fakeprint import DEFAULT_MODEL_PATH, load_model
from corpus import BLOCK_SECONDS
from lab import FREQS, all_items, band_bins, fakeprint, load_windows

BAND = band_bins(1000.0, 8000.0)
BAND_HZ = FREQS[BAND]
GRID_TOLERANCE_HZ: float = 3.0
TOP: int = 40


def describe(name: str, weights: np.ndarray) -> None:
    distance = np.abs(BAND_HZ - 200.0 * np.round(BAND_HZ / 200.0))
    on_grid = distance <= GRID_TOLERANCE_HZ
    positive = np.clip(weights, 0, None)
    expected = on_grid.mean()
    print(f"== {name}: positive weight mass on 200 Hz grid (±{GRID_TOLERANCE_HZ} Hz): {positive[on_grid].sum() / positive.sum():.2f} (chance {expected:.2f})")
    top = np.argsort(weights)[::-1][:TOP]
    print("  top positive: " + " ".join(f"{BAND_HZ[i]:.0f}{'*' if on_grid[i] else ''}" for i in sorted(top)))


def main() -> None:
    describe("lofcz", load_model(DEFAULT_MODEL_PATH).weights)
    w = load_windows(all_items(), "aac_128", round(10.0 / BLOCK_SECONDS), 2)
    x = fakeprint(w.mid, BAND, 10, -60.0, 5.0)
    model = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(x, w.label)
    describe("retrained on aac_128 10 s windows", model.coef_[0])


if __name__ == "__main__":
    main()
