"""LoRA fine-tuning of Qwen3-VL-2B on Ego-EXTRA multiple-choice questions.

Arm B: target = answer letter.        Arm C: target = letter + the expert's words.
Both arms see exactly the same input. Loss is computed on the target tokens only.
Validates every half epoch, keeps the best adapter, stops early when validation stops improving.

Usage:
    python -m src.train --config configs/pilot.yaml --arm B
    python -m src.train --config configs/pilot.yaml --arm C
    python -m src.train --config configs/pilot.yaml --arm C --context   # optional C' arm: dialogue in the prompt
    python -m src.train --config configs/pilot.yaml --arm B --limit 64  # quick smoke test on 64 examples
"""

import argparse
import json
import math
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from src.common import (build_messages, encode, load_config, load_model, load_processor, make_loader,
                        read_jsonl, set_seed, target_text)
from src.evaluate import evaluate


# ---------------------------------------------------------------- model setup

def prepare_for_training(model):
    """Memory settings that make a 2B model trainable on 8 GB."""
    model.config.use_cache = False  # the KV cache only helps generation; in training it just wastes memory
    # Gradient checkpointing: don't keep every layer's activations for the backward pass; recompute them
    # instead. Costs ~30% more compute, saves several GB. use_reentrant=False is the modern implementation.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()  # lets gradients flow through checkpointed blocks when the embeddings are frozen
    return model


def add_lora(model, cfg):
    """Freeze the whole model and attach small trainable LoRA matrices to the chosen layers.

    For a frozen weight W (d_out x d_in), LoRA learns B (d_out x r) and A (r x d_in) and uses
    W + (alpha/r) * B @ A. With r=16 that's a tiny fraction of W's parameters. B starts at zero,
    so at step 0 the model behaves exactly like the original.
    """
    from peft import LoraConfig, get_peft_model
    lcfg = cfg["lora"]
    config = LoraConfig(
        r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"],  # a regex string: matches language-model layers only
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)  # freezes everything except the new LoRA parameters
    model.print_trainable_parameters()  # sanity check: should be well under 1% of all parameters
    return model


# ---------------------------------------------------------------- one training example

def make_example(rec, processor, cfg, arm, with_video=True, with_context=False):
    """Encode prompt + answer, and work out how many tokens at the end are the answer (the target).

    Returns (inputs, n_target). The prompt is an exact prefix of prompt+answer, so the last
    n_target tokens of input_ids are precisely the answer tokens.
    """
    target = target_text(rec, arm)
    # Render the prompt alone as a string (cheap: no frames are loaded) to find where the answer starts.
    prompt_text = processor.apply_chat_template(build_messages(rec, cfg, with_video, with_context),
                                                tokenize=False, add_generation_prompt=True)
    full_text, inputs = encode(processor, build_messages(rec, cfg, with_video, with_context, target=target),
                               add_generation_prompt=False)
    assert full_text.startswith(prompt_text), "chat template changed: prompt is not a prefix of prompt+answer"
    n_target = len(processor.tokenizer(full_text[len(prompt_text):], add_special_tokens=False).input_ids)
    return inputs, n_target


def collate_train(batch, processor, cfg, arm, with_context):
    rec = batch[0]
    inputs, n_target = make_example(rec, processor, cfg, arm, with_context=with_context)
    return rec, inputs, n_target


def compute_loss(model, inputs, n_target):
    """Cross-entropy on the answer tokens only.

    A language model's output at position p is its prediction for token p+1. To score the last
    n_target tokens we need the outputs at the n_target positions *before* them. logits_to_keep asks
    the model for only the last n_target+1 positions, instead of ~1,100 positions x 151k vocabulary.
    """
    out = model(**inputs, logits_to_keep=n_target + 1)
    logits = out.logits[0, :-1].float()  # n_target positions; the very last one predicts past the end, so drop it
    targets = inputs["input_ids"][0, -n_target:]  # the answer tokens themselves
    return F.cross_entropy(logits, targets)  # mean over answer tokens of -log P(correct next token)


# ---------------------------------------------------------------- training loop

def train(cfg, arm, with_context=False, seed=None, limit=None, run_name=None):
    seed = cfg["seed"] if seed is None else seed
    set_seed(seed)
    tcfg, d = cfg["train"], cfg["data"]
    run_name = run_name or f"arm_{arm.lower()}{'_ctx' if with_context else ''}_seed{seed}"
    out_dir = Path(tcfg["output_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg))  # save the exact settings next to the results

    train_recs = read_jsonl(Path(d["splits_dir"]) / "train.jsonl", limit)
    val_recs = read_jsonl(Path(d["splits_dir"]) / "val.jsonl", limit)
    print(f"{run_name}: {len(train_recs)} train, {len(val_recs)} val")

    processor = load_processor(cfg)
    model = add_lora(prepare_for_training(load_model(cfg)), cfg)

    # Only the LoRA parameters require gradients, so only they go to the optimizer.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])

    accum = tcfg["grad_accum"]
    steps_per_epoch = math.ceil(len(train_recs) / accum)  # optimizer updates per epoch
    total_steps = steps_per_epoch * tcfg["max_epochs"]
    from transformers import get_cosine_schedule_with_warmup
    # Learning rate: linear ramp-up for warmup steps, then cosine decay towards 0 by the last step.
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(tcfg["warmup_ratio"] * total_steps), total_steps)
    eval_every = max(1, steps_per_epoch // tcfg["evals_per_epoch"])

    log_file = open(out_dir / "log.jsonl", "a")

    def log(entry):
        log_file.write(json.dumps(entry) + "\n")
        log_file.flush()

    def validate(step):
        res = evaluate(model, processor, val_recs, cfg, with_video=True, with_context=with_context,
                       num_workers=tcfg["num_workers"], desc=f"val@{step}")
        model.train()  # evaluate() switched to eval mode (disables dropout); switch back
        log({"step": step, "val_acc": res["accuracy"]})
        print(f"step {step}: val accuracy {res['accuracy']:.3f}")
        return res["accuracy"]

    best_acc, bad_evals, step = -1.0, 0, 0
    if tcfg["eval_at_start"]:
        best_acc = validate(0)  # the untrained model's score; training must beat this to save anything
    model.train()

    collate = partial(collate_train, processor=processor, cfg=cfg, arm=arm, with_context=with_context)
    stop = False
    for epoch in range(tcfg["max_epochs"]):
        # A new shuffle order each epoch (seed + epoch keeps it reproducible).
        loader = make_loader(train_recs, collate, shuffle=True, num_workers=tcfg["num_workers"], seed=seed + epoch)
        running, t0 = 0.0, time.time()
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}")
        for i, (rec, inputs, n_target) in enumerate(pbar):
            inputs = inputs.to("cuda")
            loss = compute_loss(model, inputs, n_target)
            # Divide by accum so the summed gradients over `accum` examples equal the average gradient,
            # i.e. the same update a real batch of `accum` examples would give.
            (loss / accum).backward()  # gradients ADD UP in .grad until we call zero_grad()
            running += loss.item()

            last_in_epoch = i + 1 == len(loader)
            if (i + 1) % accum == 0 or last_in_epoch:
                # Clip: if the gradient vector is longer than max_grad_norm, scale it down. Prevents rare spikes.
                torch.nn.utils.clip_grad_norm_(params, tcfg["max_grad_norm"])
                optimizer.step()  # apply the update to the LoRA weights
                scheduler.step()  # move the learning rate along its schedule
                optimizer.zero_grad(set_to_none=True)  # reset the accumulated gradients
                step += 1
                n = (i % accum) + 1
                entry = {"step": step, "epoch": epoch + 1, "loss": running / n, "lr": scheduler.get_last_lr()[0],
                         "sec_per_update": (time.time() - t0),
                         "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
                log(entry)
                pbar.set_postfix(loss=f"{entry['loss']:.3f}", lr=f"{entry['lr']:.1e}")
                running, t0 = 0.0, time.time()

                if step % eval_every == 0 or last_in_epoch:
                    acc = validate(step)
                    if acc > best_acc:
                        best_acc, bad_evals = acc, 0
                        model.save_pretrained(out_dir / "best")  # saves only the LoRA adapter (tens of MB)
                        print(f"  new best ({acc:.3f}); adapter saved to {out_dir / 'best'}")
                    else:
                        bad_evals += 1
                        if bad_evals >= tcfg["patience_evals"]:
                            print(f"  no improvement for {bad_evals} validations: stopping early")
                            stop = True
                            break
        if stop:
            break

    model.save_pretrained(out_dir / "last")  # final weights too, in case you want to inspect over-training
    summary = {"run": run_name, "arm": arm, "context": with_context, "seed": seed,
               "best_val_acc": best_acc, "optimizer_steps": step,
               "best_adapter": str(out_dir / "best") if (out_dir / "best").exists() else None}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    if summary["best_adapter"] is None:
        print("NOTE: training never beat the untrained model on validation, so no 'best' adapter was saved.")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--arm", required=True, choices=["B", "C"])
    ap.add_argument("--context", action="store_true", help="put the earlier dialogue in the prompt (arm C')")
    ap.add_argument("--seed", type=int, default=None, help="override the config seed (for repeat runs)")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N train/val records (smoke test)")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()
    train(load_config(args.config), args.arm, args.context, args.seed, args.limit, args.run_name)


if __name__ == "__main__":
    main()
