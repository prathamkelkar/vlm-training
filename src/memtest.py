"""Run a few real LoRA training steps and report peak GPU memory and time per step.

Run this before any real training: it tells you whether the settings in the config fit in 8 GB
and roughly how long an epoch will take. With --dummy it uses random-noise frames, so it works
before the dataset is downloaded (the token count, and therefore memory, is the same).

Usage:
    python -m src.memtest --config configs/pilot.yaml --dummy
    python -m src.memtest --config configs/pilot.yaml            # uses real training records
"""

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.common import load_config, load_model, load_processor, read_jsonl, set_seed
from src.train import add_lora, compute_loss, make_example, prepare_for_training


def make_dummy_record(cfg, tmp_dir):
    """A fake question with random-noise frames at the configured size and count."""
    d = cfg["data"]
    rng = np.random.default_rng(0)
    frames = []
    for i in range(d["num_frames"]):
        path = Path(tmp_dir) / f"{i:03d}.jpg"
        Image.fromarray(rng.integers(0, 255, (d["frame_size"], d["frame_size"], 3), dtype=np.uint8)).save(path)
        frames.append(str(path))
    return {
        "id": "dummy", "frames": frames, "duration": float(d["clip_seconds"]),
        "question": "What should I do next with the wrench?",
        "options": ["Tighten the bolt", "Loosen the bolt", "Put it down", "Pass it to me", "Nothing"],
        "answer_idx": 0, "answer_text": "Tighten the bolt",
        "rationale": " ".join(["word"] * 60),  # the longest Arm C target allowed: worst case for memory
        "context": [],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--dummy", action="store_true", help="use generated frames instead of the dataset")
    ap.add_argument("--arm", default="C", choices=["B", "C"], help="C has longer targets: the worst case")
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    steps = args.steps or cfg["memtest"]["steps"]
    set_seed(cfg["seed"])

    tmp = tempfile.TemporaryDirectory()
    if args.dummy:
        records = [make_dummy_record(cfg, tmp.name)] * steps
    else:
        records = read_jsonl(Path(cfg["data"]["splits_dir"]) / "train.jsonl", steps)
        assert records[0].get("frames"), "train.jsonl has no frames yet: run prepare_data without --skip-frames, or use --dummy"

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    processor = load_processor(cfg)
    model = add_lora(prepare_for_training(load_model(cfg)), cfg)
    print(f"after loading model: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated of {total_gb:.2f} GB")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["train"]["lr"])
    model.train()

    times = []
    torch.cuda.reset_peak_memory_stats()
    for i, rec in enumerate(records):
        inputs, n_target = make_example(rec, processor, cfg, args.arm)
        inputs = inputs.to("cuda")
        torch.cuda.synchronize()  # GPU work is asynchronous; wait for it so the timer measures real work
        t0 = time.time()
        loss = compute_loss(model, inputs, n_target)
        loss.backward()
        optimizer.step()  # a real optimizer step: Adam's state (2 extra copies of the LoRA weights) counts too
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        print(f"step {i + 1:2d}: loss {loss.item():.3f}  seq_len {inputs['input_ids'].shape[1]}  "
              f"{times[-1]:.2f}s  peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    peak = torch.cuda.max_memory_allocated() / 1e9
    reserved = torch.cuda.max_memory_reserved() / 1e9  # what PyTorch actually grabbed from the GPU (>= allocated)
    sec = float(np.median(times[2:] if len(times) > 3 else times))  # skip the first steps: they include warm-up
    n_train = cfg["data"]["train_size"]
    print("\n=== memory test result ===")
    print(f"peak allocated {peak:.2f} GB, peak reserved {reserved:.2f} GB, card {total_gb:.2f} GB")
    print(f"median {sec:.2f} s per training example (GPU only; data loading overlaps in real training)")
    print(f"estimated epoch of {n_train} examples: {sec * n_train / 60:.0f} min "
          f"(+ validation of {cfg['data']['val_size']} questions, roughly a third of a step each)")
    headroom = total_gb - reserved
    if headroom < 0.5:
        print("VERDICT: too tight. Reduce num_frames (10 -> 8), then frame_size (448 -> 352), then lora.r.")
    else:
        print(f"VERDICT: fits, with {headroom:.1f} GB headroom.")


if __name__ == "__main__":
    main()
