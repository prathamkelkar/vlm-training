# VLM expert-reasoning pilot

Local pilot: LoRA fine-tuning of Qwen3-VL-2B-Instruct on Ego-EXTRA 5 s clips (multiple-choice QA),
on an 8 GB RTX PRO 2000 Blackwell laptop.

- Why each decision was made: [docs/training_plan_brief.md](docs/training_plan_brief.md)
- How the code works, file by file: [docs/code_guide.md](docs/code_guide.md)

| Arm | What it is | Command |
|---|---|---|
| A | Zero-shot | `python -m src.evaluate --split test --both` |
| B | LoRA on the answer letter | `python -m src.train --arm B` |
| C | LoRA on letter + expert's words | `python -m src.train --arm C` |
| C′ (optional) | C with earlier dialogue in the prompt | `python -m src.train --arm C --context` |

## Layout

```
configs/pilot.yaml            shared settings (identical across arms)
scripts/download_egoextra.py  fetches only the clips the splits use (~3.6 GB of a 125 GB dataset)
src/prepare_data.py           raw Ego-EXTRA -> data/splits/*.jsonl + data/frames/
src/common.py                 prompts, encoding, model loading, data loaders
src/memtest.py                peak-VRAM / step-time test
src/train.py                  LoRA training, best-validation checkpoint
src/evaluate.py               A-E probability scoring, blind control, breakdowns
data/raw/egoextra/            downloads (untouched)
data/frames/                  10 JPEGs per clip
data/splits/                  train / val / test JSONL (test is frozen once frames are attached)
runs/                         adapters, logs, eval results
```

## Setup

```bash
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Needs `ffmpeg`, and about 6 GB free disk for clips + frames (plus ~4.3 GB for the model in ~/.cache/huggingface).

## Run order

```bash
python scripts/download_egoextra.py --mappings-only               # questions, transcripts, clip mappings (MBs)
python -m src.prepare_data --skip-frames                          # decide the splits (no video needed)
python scripts/download_egoextra.py --only-used                   # just the clips those splits use (~3.6 GB)
python -m src.prepare_data                                        # extract frames; test split is now frozen
python -m src.memtest                                             # confirm it fits on the real data
python -m src.evaluate --split test --both                        # Arm A and A-blind
python -m src.train --arm B                                       # ~80 min per epoch
python -m src.evaluate --split test --both --adapter runs/arm_b_seed0/best
python -m src.train --arm C
python -m src.evaluate --split test --both --adapter runs/arm_c_seed0/best
```

All commands default to `--config configs/pilot.yaml`.

## Licence

Ego-EXTRA forbids commercial use without written consent from the authors. Use it for research only.
