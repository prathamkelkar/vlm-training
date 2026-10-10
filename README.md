# VLM expert-reasoning pilot

Local pilot: LoRA fine-tuning of Qwen3-VL-2B-Instruct on Ego-EXTRA 5 s clips (multiple-choice QA),
on an 8 GB RTX PRO 2000 Blackwell laptop.

| Arm | What it is | Command |
|---|---|---|
| A | Zero-shot | `python -m src.evaluate --split test` |
| B | LoRA on question + answer | `python -m src.train --arm B` |
| C | LoRA with preceding expert-trainee dialogue | `python -m src.train --arm C` |

## Layout

```
configs/pilot.yaml            shared settings (identical across arms)
scripts/download_egoextra.py  fetches only the 5 s clips + questions + split (~6.6 GB)
src/prepare_data.py           raw Ego-EXTRA -> data/splits/*.jsonl + data/frames/
src/common.py                 prompts, encoding, model loading
src/memtest.py                20-step peak-VRAM / step-time test
src/train.py                  LoRA training, best-validation checkpoint
src/evaluate.py               A-E probability scoring
data/raw/egoextra/            downloads (untouched)
data/frames/                  8 JPEGs per clip
data/splits/                  train / val / test JSONL
runs/                         adapters and logs
```

## Setup

```bash
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Needs about 15-17 GB of free disk and `ffmpeg`.

## Order

1. `python scripts/download_egoextra.py --out data/raw/egoextra`
2. `python -m src.prepare_data --config configs/pilot.yaml`
3. `python -m src.memtest --config configs/pilot.yaml`
4. Arm A, then B, then C, all scored on the same test split.

## Licence

Ego-EXTRA forbids commercial use without written consent from the authors. Use it for research only.
