"""Sanity: lofcz logit from cached 32 kHz block spectra vs the reference 16 kHz pipeline on the same 10 s windows."""

import numpy as np
import soxr

from sunocheck.fakeprint import DEFAULT_MODEL_PATH, compute_fakeprint, load_model, score_from_fakeprint
from corpus import BLOCK_SECONDS, PLAYBACK_RATE, VARIANTS, cache_path, list_items
from lab import lofcz_fakeprint
from prepare import decode_variant

WINDOW_BLOCKS: int = 4
model = load_model(DEFAULT_MODEL_PATH)
orig = VARIANTS[0]
for item in [i for i in list_items() if i.source in ("samples", "sudo")]:
    stereo = decode_variant(item, orig)
    mono16 = soxr.resample(stereo.mean(axis=1), PLAYBACK_RATE, 16000, quality="HQ").astype(np.float32)
    mid = np.load(cache_path(orig.name, item.item_id))["mid"].astype(np.float32)
    pairs: list[tuple[float, float]] = []
    for start in range(0, mid.shape[0] - WINDOW_BLOCKS + 1, WINDOW_BLOCKS):
        cached = float(lofcz_fakeprint(mid[start : start + WINDOW_BLOCKS].mean(axis=0, keepdims=True))[0] @ model.weights) + model.bias
        begin = int(start * BLOCK_SECONDS * 16000)
        reference = score_from_fakeprint(compute_fakeprint(mono16[begin : begin + int(WINDOW_BLOCKS * BLOCK_SECONDS * 16000)]), model)
        pairs.append((cached, reference))
    values = np.array(pairs)
    print(f"{item.item_id:<24} {item.label:<6} corr {np.corrcoef(values.T)[0, 1]:.3f}  cached {values[:, 0].round(1)}  ref {values[:, 1].round(1)}")
