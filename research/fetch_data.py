"""Download the research corpus over a slow link.

- suno: Kukedlc/suno-ai-music-dataset, one track per prompt, only the first SUNO_PREFIX_BYTES (~60 s).
- fma: one FMA-medium parquet shard (30 s clips), fetched as parallel byte ranges.
- jamendo: the head of one MTG-Jamendo tar (full tracks), complete members only.

Usage: uv run python research/fetch_data.py suno|fma|jamendo
"""

import csv
import http.client
import sys
import tarfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

ROOT: Path = Path(__file__).parent / "data"
HF: str = "https://huggingface.co/datasets"
SUNO_META: Path = Path(__file__).parent / "meta" / "kukedlc_metadata.csv"
EXCLUDED_DIR: Path = Path(__file__).parent.parent / "samples" / "suno"
SUNO_PREFIX_BYTES: int = 1_700_000
FMA_MEDIUM_SHARD: int = 7
FMA_MEDIUM_SHARDS: int = 44
JAMENDO_TAR: int = 42
JAMENDO_HEAD_BYTES: int = 300_000_000
PART_BYTES: int = 8_000_000
WORKERS: int = 8
RETRIES: int = 4


@dataclass(frozen=True)
class SunoItem:
    track_id: str
    version: str
    instrumental: bool
    file_name: str


def fetch_range_once(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=300) as response:
        if response.status != 206:
            raise RuntimeError(f"expected 206 for range request, got {response.status}: {url}")
        return response.read()


def fetch_range(url: str, start: int, end: int) -> bytes:
    """Dropped connections on a slow link are expected; retry the same range, then give up loudly."""
    for attempt in range(RETRIES):
        try:
            return fetch_range_once(url, start, end)
        except (http.client.IncompleteRead, urllib.error.URLError, TimeoutError) as error:
            print(f"range {start}-{end} attempt {attempt + 1} failed: {error!r}", flush=True)
    raise RuntimeError(f"range {start}-{end} failed {RETRIES} times: {url}")


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def download_ranges(url: str, target: Path, total_bytes: int) -> None:
    if target.exists():
        return
    starts = list(range(0, total_bytes, PART_BYTES))
    with ThreadPoolExecutor(WORKERS) as pool:
        parts = list(pool.map(lambda s: fetch_range(url, s, min(s + PART_BYTES, total_bytes) - 1), starts))
    target.write_bytes(b"".join(parts))


def suno_items() -> list[SunoItem]:
    excluded = {p.stem for p in EXCLUDED_DIR.glob("*.mp3")}
    rows = list(csv.DictReader(SUNO_META.open(encoding="utf-8")))
    seen_titles: set[str] = set()
    items: list[SunoItem] = []
    for row in rows:
        old = row["major_model_version"] not in ("v5", "v5.5")
        if row["id"] in excluded or (not old and row["title"] in seen_titles):
            continue
        seen_titles.add(row["title"])
        items.append(SunoItem(row["id"], row["major_model_version"], row["is_instrumental"] == "True", row["file_name"]))
    return items


def fetch_suno_prefix(item: SunoItem, target: Path) -> None:
    if target.exists():
        return
    url = f"{HF}/Kukedlc/suno-ai-music-dataset/resolve/main/{item.file_name}"
    target.write_bytes(fetch_range(url, 0, SUNO_PREFIX_BYTES - 1))


def fetch_suno() -> None:
    target_dir = ROOT / "suno"
    target_dir.mkdir(parents=True, exist_ok=True)
    items = suno_items()
    with (target_dir / "index.csv").open("w", encoding="utf-8", newline="") as index:
        writer = csv.writer(index)
        writer.writerow(["id", "version", "instrumental"])
        writer.writerows([[i.track_id, i.version, i.instrumental] for i in items])
    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(lambda i: fetch_suno_prefix(i, target_dir / f"{i.track_id}.mp3"), items))
    print(f"suno: {len(items)} tracks")


def fetch_fma() -> None:
    target_dir = ROOT / "fma"
    target_dir.mkdir(parents=True, exist_ok=True)
    url = f"{HF}/benjamin-paine/free-music-archive-medium/resolve/main/data/train-{FMA_MEDIUM_SHARD:05d}-of-{FMA_MEDIUM_SHARDS:05d}.parquet"
    parquet_path = ROOT / f"fma_medium_{FMA_MEDIUM_SHARD}.parquet"
    download_ranges(url, parquet_path, remote_size(url))
    table = pq.read_table(parquet_path)
    print(f"fma: columns {table.column_names}, rows {table.num_rows}")
    for row_index, item in enumerate(table.column("audio").to_pylist()):
        name = Path(item["path"]).name if item["path"] else f"fma_{FMA_MEDIUM_SHARD}_{row_index}.mp3"
        (target_dir / name).write_bytes(item["bytes"])
    parquet_path.unlink()


def fetch_jamendo() -> None:
    target_dir = ROOT / "jamendo"
    target_dir.mkdir(parents=True, exist_ok=True)
    tar_path = ROOT / f"jamendo_{JAMENDO_TAR}_head.tar"
    download_ranges(f"{HF}/rkstgr/mtg-jamendo/resolve/main/data/train/{JAMENDO_TAR}.tar", tar_path, JAMENDO_HEAD_BYTES)
    size = tar_path.stat().st_size
    written = 0
    with tar_path.open("rb") as raw, tarfile.open(fileobj=raw, mode="r|") as archive:
        for member in archive:
            if member.offset_data + member.size > size:
                break
            extracted = archive.extractfile(member)
            if member.isfile() and extracted is not None and member.name.endswith(".opus"):
                (target_dir / Path(member.name).name).write_bytes(extracted.read())
                written += 1
    print(f"jamendo: {written} complete tracks")
    tar_path.unlink()


COMMANDS = {"suno": fetch_suno, "fma": fetch_fma, "jamendo": fetch_jamendo}

if __name__ == "__main__":
    COMMANDS[sys.argv[1]]()
