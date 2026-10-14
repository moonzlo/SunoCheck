"""Encode every corpus item through each codec variant, decode at the playback rate, cache block spectra.

Usage: uv run python research/prepare.py <workers> <all|source[,source...]> <variant[,variant...]> <item_stride>
"""

import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soxr
from scipy import fft

from corpus import (
    ANALYSIS_RATE,
    BLOCK_SECONDS,
    EXCERPT_SECONDS,
    HOP,
    N_FFT,
    PLAYBACK_RATE,
    VARIANTS,
    FloatArray,
    Item,
    Variant,
    cache_path,
    list_items,
)

MIN_SECONDS: float = 20.0
WINDOW: FloatArray = np.hanning(N_FFT + 1)[:-1]


class DecodeError(RuntimeError):
    pass


def run_ffmpeg(args: list[str]) -> bytes:
    result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args], capture_output=True)
    if result.returncode != 0:
        raise DecodeError(result.stderr.decode("utf-8", "replace")[-500:])
    return result.stdout


def decode_variant(item: Item, variant: Variant) -> FloatArray:
    decode_args = ["-f", "f32le", "-ac", "2", "-ar", str(PLAYBACK_RATE), "-"]
    if not variant.codec_args:
        raw = run_ffmpeg(["-t", str(EXCERPT_SECONDS), "-i", str(item.path), *decode_args])
    else:
        with tempfile.TemporaryDirectory() as tmp:
            encoded = Path(tmp) / f"x.{variant.extension}"
            filters = ["-filter_complex", f"[0:a]{variant.filters}[out]", "-map", "[out]"] if variant.filters else ["-vn"]
            run_ffmpeg(["-t", str(EXCERPT_SECONDS), "-i", str(item.path), *filters, "-ac", "2", *variant.codec_args, str(encoded)])
            raw = run_ffmpeg(["-i", str(encoded), *decode_args])
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).astype(np.float64)


def frames_db_mean(frames: np.ndarray) -> np.ndarray:
    power = np.abs(fft.rfft(frames * WINDOW, axis=1)) ** 2
    return (10.0 * np.log10(np.clip(power, 1e-12, None))).mean(axis=0)


def block_means(signal: np.ndarray, frames_per_block: int) -> np.ndarray:
    """Mean dB spectrum per block of `frames_per_block` frames, computed block by block to bound memory."""
    frames = np.lib.stride_tricks.sliding_window_view(signal, N_FFT)[::HOP]
    blocks = frames.shape[0] // frames_per_block
    return np.stack([frames_db_mean(frames[b * frames_per_block : (b + 1) * frames_per_block]) for b in range(blocks)])


def spectra(stereo: FloatArray) -> dict[str, np.ndarray]:
    mid44 = stereo.mean(axis=1)
    side44 = 0.5 * (stereo[:, 0] - stereo[:, 1])
    mid = soxr.resample(mid44, PLAYBACK_RATE, ANALYSIS_RATE, quality="HQ")
    side = soxr.resample(side44, PLAYBACK_RATE, ANALYSIS_RATE, quality="HQ")
    frames_per_block = round(BLOCK_SECONDS * ANALYSIS_RATE / HOP)
    mid_blocks = block_means(mid, frames_per_block)
    side_blocks = block_means(side, frames_per_block)
    samples_per_block = frames_per_block * HOP
    blocks = mid_blocks.shape[0]
    rms = np.sqrt(np.mean(mid[: blocks * samples_per_block].reshape(blocks, samples_per_block) ** 2, axis=1))
    native_frames = np.lib.stride_tricks.sliding_window_view(mid44, N_FFT)[:: N_FFT // 2]
    native = frames_db_mean(native_frames)
    return {
        "mid": mid_blocks.astype(np.float16),
        "side": side_blocks.astype(np.float16),
        "block_db": (20.0 * np.log10(rms + 1e-12)).astype(np.float32),
        "native": native.astype(np.float32),
    }


def process(item: Item, variant: Variant) -> str:
    target = cache_path(variant.name, item.item_id)
    if target.exists():
        return "cached"
    stereo = decode_variant(item, variant)
    if stereo.shape[0] < MIN_SECONDS * PLAYBACK_RATE:
        return "short"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".tmp.npz")
    np.savez(partial, **spectra(stereo))
    partial.replace(target)
    return "ok"


def main(workers: int, sources: str, variants: str, stride: int) -> None:
    items = [i for i in list_items() if sources == "all" or i.source in sources.split(",")][::stride]
    selected = [v for v in VARIANTS if v.name in variants.split(",")]
    jobs = [(item, variant) for item in items for variant in selected]
    print(f"items {len(items)}, jobs {len(jobs)}", flush=True)
    counts: dict[str, int] = {}
    failures: list[str] = []
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(process, item, variant): (item, variant) for item, variant in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            item, variant = futures[future]
            try:
                status = future.result()
            except DecodeError as error:
                status = "failed"
                failures.append(f"{item.item_id}/{variant.name}: {error}")
            counts[status] = counts.get(status, 0) + 1
            if done % 200 == 0:
                print(f"{done}/{len(jobs)} {counts}", flush=True)
    print(f"done {counts}")
    for failure in failures[:20]:
        print("FAILED", failure)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]))
