"""Download only what the 5 s pilot needs from Ego-EXTRA (about 6.6 GB instead of 125 GB).

- questions.zip, transcript.zip, protocol_split_wacv.json: downloaded in full (a few MB) and unzipped.
- clips.zip (60 GB): never downloaded whole. The server supports HTTP range requests, so we read
  the zip's index (its "central directory") remotely and fetch only the bytes of the files we want:
  `video_clip_final/*/*/5s/*.mp4` and `video_clip_final/clips_mapping_*_duration_5.json`.

Re-running is safe: files that already exist with the right size are skipped.

Usage:
    python scripts/download_egoextra.py --dry-run                   # just print sizes
    python scripts/download_egoextra.py --mappings-only             # small files only (questions, mappings)
    python scripts/download_egoextra.py --only-used --dry-run       # size of just the clips our splits use
    python scripts/download_egoextra.py --only-used                 # those clips (~3.6 GB): the normal choice
    python scripts/download_egoextra.py                             # every 5 s clip (6.6 GB)
"""

import argparse
import io
import json
import re
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL = "https://iplab.dmi.unict.it/sharing/Ego-EXTRA"
SMALL_FILES = ["questions.zip", "transcript.zip", "protocol_split_wacv.json"]
CLIPS_URL = f"{BASE_URL}/clips.zip"


def http_get(url, start=None, end=None, retries=5):
    """GET a URL (optionally only bytes start..end inclusive), retrying on network errors."""
    headers = {"Range": f"bytes={start}-{end}"} if start is not None else {}
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                return r.read()
        except Exception as e:  # network hiccups are common on long downloads
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  retry {attempt + 1} for {url} ({e}); waiting {wait}s", file=sys.stderr)
            time.sleep(wait)


class HTTPRangeFile(io.RawIOBase):
    """A read-only, seekable file object backed by HTTP range requests.

    zipfile.ZipFile only needs seek/tell/read, so this lets it read a remote zip's index
    without downloading the whole archive.
    """

    def __init__(self, url):
        self.url, self.pos = url, 0
        head = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(head, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        data = http_get(self.url, self.pos, self.pos + n - 1)
        buf[: len(data)] = data
        self.pos += len(data)
        return len(data)


def download_small_files(out_dir):
    """Download the question/transcript zips and the protocol file, then unzip the zips."""
    for name in SMALL_FILES:
        dest = out_dir / name
        if not dest.exists():
            print(f"downloading {name}")
            dest.write_bytes(http_get(f"{BASE_URL}/{name}"))
        if name.endswith(".zip"):
            folder = out_dir / name[: -len(".zip")]
            if not folder.exists():
                with zipfile.ZipFile(dest) as z:
                    z.extractall(folder)


def select_entries(infos, clip_seconds, mappings_only):
    """Pick the zip entries we need: mapping JSONs and (unless mappings_only) the N-second clips."""
    mapping_re = re.compile(rf"video_clip_final/clips_mapping_\d+_duration_{clip_seconds}\.json$")
    clip_re = re.compile(rf"video_clip_final/\d+/[^/]+/{clip_seconds}s/[^/]+\.mp4$")
    keep = [i for i in infos if mapping_re.search(i.filename)]
    if not mappings_only:
        keep += [i for i in infos if clip_re.search(i.filename)]
    return keep


def fetch_entry(info, out_dir):
    """Download one zip entry's bytes directly by offset, decompress if needed, verify CRC, save."""
    dest = out_dir / info.filename
    if dest.exists() and dest.stat().st_size == info.file_size:
        return 0  # already downloaded
    # Every entry starts with a 30-byte "local file header"; its name/extra lengths tell us where the data starts.
    # The server takes ~1 s to answer ANY request, so we fetch header + data in ONE request, with some slack
    # for the header's variable-length part (usually well under 1 KB), instead of two round trips.
    slack = 30 + len(info.filename.encode()) + 1024
    blob = http_get(CLIPS_URL, info.header_offset, info.header_offset + slack + info.compress_size - 1)
    _, _, _, _, _, _, _, _, _, name_len, extra_len = struct.unpack("<4s5H3L2H", blob[:30])
    start = 30 + name_len + extra_len  # where the file's bytes begin inside `blob`
    if start + info.compress_size <= len(blob):
        raw = blob[start:start + info.compress_size]
    else:  # rare: header longer than our slack, so fall back to a second request
        data_start = info.header_offset + start
        raw = http_get(CLIPS_URL, data_start, data_start + info.compress_size - 1)
    if info.compress_type == zipfile.ZIP_STORED:
        data = raw
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(raw, -15)  # -15 = raw deflate stream, as used inside zip files
    else:
        raise ValueError(f"unsupported compression {info.compress_type} for {info.filename}")
    if zlib.crc32(data) != info.CRC:  # integrity check against the checksum stored in the zip index
        raise IOError(f"CRC mismatch for {info.filename}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.rename(dest)  # rename is atomic, so a crash never leaves a half-written file under the real name
    return len(data)


def used_clip_ids(splits_dir):
    """Clip ids referenced by data/splits/*.jsonl (written by `prepare_data --skip-frames`)."""
    ids = set()
    for path in Path(splits_dir).glob("*.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                ids.add(json.loads(line)["clip_id"])
    if not ids:
        raise SystemExit(f"no splits in {splits_dir}: run `python -m src.prepare_data --skip-frames` first")
    return ids


def extract_clips(out_dir, clip_seconds, mappings_only, dry_run, limit, workers, only_ids=None):
    """Read the remote zip index and download the selected entries in parallel."""
    print("reading clips.zip index over HTTP (no full download)...")
    z = zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(CLIPS_URL), buffer_size=1 << 20))
    entries = select_entries(z.infolist(), clip_seconds, mappings_only)
    if only_ids is not None:  # keep mapping JSONs plus only the clips our splits use
        entries = [e for e in entries if e.filename.endswith(".json") or Path(e.filename).stem in only_ids]
    if limit:
        mappings = [e for e in entries if e.filename.endswith(".json")]
        clips = [e for e in entries if e.filename.endswith(".mp4")][:limit]
        entries = mappings + clips
    total = sum(e.file_size for e in entries)
    print(f"{len(entries)} files, {total / 1e9:.2f} GB")
    if dry_run:
        return
    done_bytes, start = 0, time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:  # threads overlap network waits
        futures = [pool.submit(fetch_entry, e, out_dir) for e in entries]
        for k, f in enumerate(as_completed(futures), 1):
            done_bytes += f.result()
            if k % 200 == 0 or k == len(futures):
                rate = done_bytes / max(time.time() - start, 1e-6) / 1e6
                print(f"  {k}/{len(futures)} files, {done_bytes / 1e9:.2f} GB new, {rate:.1f} MB/s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/raw/egoextra")
    ap.add_argument("--clip-seconds", type=int, default=5, choices=[5, 15, 30])
    ap.add_argument("--mappings-only", action="store_true", help="only fetch the clip-to-question mapping JSONs")
    ap.add_argument("--dry-run", action="store_true", help="print what would be downloaded and stop")
    ap.add_argument("--limit", type=int, default=0, help="download at most N clips (for testing)")
    ap.add_argument("--only-used", metavar="SPLITS_DIR", nargs="?", const="data/splits", default=None,
                    help="download only clips referenced by the split files (about 3.6 GB instead of 6.6 GB)")
    ap.add_argument("--workers", type=int, default=24, help="parallel requests; the server is slow to respond, not slow to send")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        download_small_files(out_dir)
    only_ids = used_clip_ids(args.only_used) if args.only_used else None
    extract_clips(out_dir, args.clip_seconds, args.mappings_only, args.dry_run, args.limit, args.workers, only_ids)


if __name__ == "__main__":
    main()
