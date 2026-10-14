"""Research corpus: labeled source tracks, codec variants, cached block spectra.

Cache layout: research/cache/spec/<variant>/<item_id>.npz with
  mid, side: float16 [blocks, BINS] mean dB per BLOCK_SECONDS block at ANALYSIS_RATE (n_fft N_FFT, hop HOP)
  block_db: float32 [blocks] block loudness (dBFS, mid)
  native: float32 [N_FFT // 2 + 1] mean dB of mid at PLAYBACK_RATE over the excerpt
"""

import csv
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

RESEARCH: Path = Path(__file__).parent
DATA: Path = RESEARCH / "data"
CACHE: Path = RESEARCH / "cache" / "spec"
PROJECT: Path = RESEARCH.parent
SUDO_DIR: Path = PROJECT / "samples" / "sudo"

EXCERPT_SECONDS: float = 60.0
PLAYBACK_RATE: int = 44100
ANALYSIS_RATE: int = 32000
N_FFT: int = 16384
HOP: int = 1024
BLOCK_SECONDS: float = 2.5
BINS: int = N_FFT // 2 + 1
BIN_HZ: float = ANALYSIS_RATE / N_FFT

FloatArray = NDArray[np.float64]


class Label(StrEnum):
    SUNO = "suno"
    HUMAN = "human"


@dataclass(frozen=True)
class Item:
    item_id: str
    label: Label
    source: str
    group: str
    version: str
    path: Path


@dataclass(frozen=True)
class Variant:
    name: str
    codec_args: tuple[str, ...]
    extension: str
    filters: str


AAC_256: tuple[str, ...] = ("-c:a", "aac", "-b:a", "256k")

CODEC_VARIANTS: tuple[Variant, ...] = (
    Variant("orig", (), "", ""),
    Variant("mp3_320", ("-c:a", "libmp3lame", "-b:a", "320k"), "mp3", ""),
    Variant("mp3_128", ("-c:a", "libmp3lame", "-b:a", "128k"), "mp3", ""),
    Variant("aac_256", AAC_256, "m4a", ""),
    Variant("aac_128", ("-c:a", "aac", "-b:a", "128k"), "m4a", ""),
    Variant("opus_128", ("-c:a", "libopus", "-b:a", "128k"), "opus", ""),
    Variant("vorbis_160", ("-c:a", "libvorbis", "-b:a", "160k"), "ogg", ""),
)
AUGMENT_VARIANTS: tuple[Variant, ...] = (
    Variant("speed_up3", AAC_256, "m4a", "aresample=48000,asetrate=49440,aresample=48000"),
    Variant("speed_down4", AAC_256, "m4a", "aresample=48000,asetrate=46080,aresample=48000"),
    Variant("pitch_up1", AAC_256, "m4a", "rubberband=pitch=1.059463"),
    Variant("pitch_down2", AAC_256, "m4a", "rubberband=pitch=0.890899"),
    Variant("tempo_95", AAC_256, "m4a", "atempo=0.95"),
    Variant("wsola_up1", AAC_256, "m4a", "aresample=48000,asetrate=50854,aresample=48000,atempo=0.943874"),
    Variant("wsola_down1", AAC_256, "m4a", "aresample=48000,asetrate=45305,aresample=48000,atempo=1.059463"),
    Variant(
        "master",
        AAC_256,
        "m4a",
        "equalizer=f=100:t=q:w=1:g=3,highshelf=f=8000:g=4,acompressor=threshold=-18dB:ratio=4:attack=5:release=50,alimiter=limit=0.9",
    ),
    Variant("reverb", AAC_256, "m4a", "aecho=0.8:0.6:40|70:0.3|0.2"),
    Variant("speed_down15", AAC_256, "m4a", "aresample=48000,asetrate=40800,aresample=48000"),
    Variant("speed_up12", AAC_256, "m4a", "aresample=48000,asetrate=53760,aresample=48000"),
    Variant("speed_up25", AAC_256, "m4a", "aresample=48000,asetrate=60000,aresample=48000"),
    Variant("hall", AAC_256, "m4a", "aresample=48000[dry];amovie=data/ir_hall.wav[ir];[dry][ir]afir=irnorm=-1:gtype=-1"),
    Variant(
        "slowed_hall",
        AAC_256,
        "m4a",
        "aresample=48000,asetrate=40800,aresample=48000[dry];amovie=data/ir_hall.wav[ir];[dry][ir]afir=irnorm=-1:gtype=-1",
    ),
)
VARIANTS: tuple[Variant, ...] = CODEC_VARIANTS + AUGMENT_VARIANTS
VARIANT_NAMES: tuple[str, ...] = tuple(v.name for v in CODEC_VARIANTS)
AUGMENT_NAMES: tuple[str, ...] = tuple(v.name for v in AUGMENT_VARIANTS)


def safe_id(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "_", text)[:80]


def suno_versions() -> dict[str, str]:
    index = DATA / "suno" / "index.csv"
    return {row["id"]: row["version"] for row in csv.DictReader(index.open(encoding="utf-8"))}


def suno_titles() -> dict[str, str]:
    meta = RESEARCH / "meta" / "kukedlc_metadata.csv"
    return {row["id"]: row["title"] for row in csv.DictReader(meta.open(encoding="utf-8"))}


def list_items() -> list[Item]:
    versions = suno_versions()
    titles = suno_titles()
    items: list[Item] = []
    for path in sorted((DATA / "suno").glob("*.mp3")):
        group = "suno_" + safe_id(titles[path.stem])
        items.append(Item("kuk_" + path.stem[:13], Label.SUNO, "kukedlc", group, versions[path.stem], path))
    for path in sorted((PROJECT / "samples" / "suno").glob("*.mp3")):
        items.append(Item("smp_" + path.stem[:13], Label.SUNO, "samples", "suno_smp_" + path.stem[:8], "v5x", path))
    for index, path in enumerate(sorted(SUDO_DIR.glob("*.mp3"))):
        items.append(Item(f"sudo_{index}", Label.SUNO, "sudo", f"suno_sudo_{index}", "v5x", path))
    for source, pattern in (("fma", "*.mp3"), ("jamendo", "*.opus")):
        for path in sorted((DATA / source).glob(pattern)):
            items.append(Item(f"{source}_{safe_id(path.stem)}", Label.HUMAN, source, f"{source}_{safe_id(path.stem)}", "", path))
    for path in sorted((PROJECT / "samples" / "human").glob("*.mp3")):
        items.append(Item("smp_" + safe_id(path.stem), Label.HUMAN, "samples", "smp_" + safe_id(path.stem), "", path))
    return items


def cache_path(variant: str, item_id: str) -> Path:
    return CACHE / variant / f"{item_id}.npz"
