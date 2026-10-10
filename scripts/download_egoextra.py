"""Download only what the 5 s pilot needs from Ego-EXTRA (about 6.6 GB instead of 125 GB).

- questions.zip, transcript.zip, protocol_split_wacv.json: downloaded in full (a few MB).
- clips.zip (60 GB): read remotely with HTTP range requests; only `*/5s/*.mp4` and
  `clips_mapping_*_duration_5.json` are extracted.

Usage: python scripts/download_egoextra.py --out data/raw/egoextra
"""

BASE_URL = "https://iplab.dmi.unict.it/sharing/Ego-EXTRA"
SMALL_FILES = ["questions.zip", "transcript.zip", "protocol_split_wacv.json"]


def download_small_files(out_dir):
    raise NotImplementedError


def extract_5s_clips(out_dir):
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
