"""Shared helpers used by every script: config, JSONL I/O, prompt building, input encoding, model loading.

Keeping these in one place guarantees that training and evaluation build the model's input in
exactly the same way: a mismatch between the two is one of the most common silent bugs in fine-tuning.
"""

import json
import os
import random
from functools import partial
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # avoid a deadlock warning when DataLoader forks workers

import numpy as np
import torch
import yaml
from qwen_vl_utils import process_vision_info

LETTERS = "ABCDE"  # option labels; the model answers with one of these
PATCH_SIZE = 16  # Qwen3-VL's vision encoder cuts images into 16x16-pixel patches (Qwen2.5-VL used 14)


# ---------------------------------------------------------------- config and files

def load_config(path):
    """Read the YAML config into a plain nested dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """Seed every random number generator we use, so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path, limit=None):
    """Load one JSON object per line; optionally only the first `limit` lines."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def write_jsonl(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- prompts

def build_prompt_text(rec, with_context=False):
    """The text part of the user turn: (optional dialogue) + question + lettered options + instruction."""
    lines = ["This is first-person video from a trainee who is being guided by an expert."]
    if with_context and rec.get("context"):
        # Only dialogue from BEFORE the question; later turns contain the answer (leakage).
        lines += ["Conversation so far:", *rec["context"]]
    lines += [f"Question: {rec['question']}", "Options:"]
    lines += [f"{LETTERS[i]}. {opt}" for i, opt in enumerate(rec["options"])]
    lines.append("Answer with the option's letter from the given choices directly.")
    return "\n".join(lines)


def target_text(rec, arm):
    """What the model is trained to output. Arm B: just the letter. Arm C: letter + the expert's words."""
    letter = LETTERS[rec["answer_idx"]]
    if arm == "B":
        return letter
    if arm == "C":
        return f"{letter}. {rec['rationale']}"  # letter first, so letter-probability scoring still works
    raise ValueError(f"unknown arm {arm!r}")


def build_messages(rec, cfg, with_video=True, with_context=False, target=None):
    """Chat-format messages in the structure Qwen's chat template and qwen_vl_utils expect.

    with_video=False gives the "blind" version (same text, no frames).
    target=None gives a prompt for scoring; a string adds the assistant's answer for training.
    """
    content = []
    if with_video:
        content.append({
            "type": "video",
            "video": [f"file://{Path(p).resolve()}" for p in rec["frames"]],  # a list of frame images = a video
            "sample_fps": len(rec["frames"]) / rec["duration"],  # real timing, so the model's timestamps are right
            "max_pixels": cfg["data"]["frame_size"] ** 2,  # caps per-frame resolution (448*448)
        })
    content.append({"type": "text", "text": build_prompt_text(rec, with_context)})
    messages = [{"role": "user", "content": content}]
    if target is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return messages


# ---------------------------------------------------------------- turning messages into tensors

def encode(processor, messages, add_generation_prompt):
    """Turn chat messages into model inputs (token ids, attention mask, pixel values, grid sizes).

    Returns (text, inputs). `text` is the rendered chat string, useful for debugging and masking.
    """
    # 1) Render the conversation as one string with special tokens, e.g. "<|im_start|>user\n<|vision_start|>...".
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    if messages[-1]["role"] == "assistant":
        # The template ends the answer with "<|im_end|>\n". We train the model to emit <|im_end|> (so it
        # learns to stop) but not the newline after it, which carries no information.
        text = text.rstrip("\n")
    has_video = any(c.get("type") == "video" for m in messages for c in m["content"])
    if not has_video:
        return text, processor(text=[text], return_tensors="pt")
    # 2) Load and resize the frames. Returns each video as (frames tensor, metadata with fps/frame indices).
    _, videos, video_kwargs = process_vision_info(
        messages, image_patch_size=PATCH_SIZE, return_video_kwargs=True, return_video_metadata=True)
    videos, metadata = map(list, zip(*videos))
    # 3) The processor tokenises the text, expands the single video placeholder into one token per visual
    #    patch group (plus "<t seconds>" timestamps), and normalises the pixels.
    inputs = processor(text=[text], videos=videos, video_metadata=metadata,
                       do_resize=False,  # qwen_vl_utils already resized; don't do it twice
                       return_tensors="pt", **video_kwargs)
    return text, inputs


def letter_token_ids(tokenizer):
    """Token id of each answer letter. Scoring compares the model's logits at exactly these ids."""
    ids = []
    for letter in LETTERS:
        tok = tokenizer.encode(letter, add_special_tokens=False)
        assert len(tok) == 1, f"letter {letter} is not a single token: {tok}"
        ids.append(tok[0])
    return ids


def make_loader(records, collate_fn, shuffle=False, num_workers=4, seed=0):
    """A DataLoader that runs `collate_fn` (frame loading + tokenising) in background CPU workers.

    batch_size=1 because each example is one video; the GPU can't fit more at 8 GB.
    While the GPU works on example k, the workers prepare k+1, k+2, ... so the GPU never waits.
    """
    gen = torch.Generator().manual_seed(seed)  # makes the shuffle order repeatable
    return torch.utils.data.DataLoader(
        records, batch_size=1, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers,
        generator=gen, prefetch_factor=4 if num_workers > 0 else None)


def collate_prompt(batch, processor, cfg, with_video, with_context):
    """DataLoader collate for scoring: one record -> (record, prompt inputs ending at the assistant turn)."""
    rec = batch[0]
    _, inputs = encode(processor, build_messages(rec, cfg, with_video, with_context), add_generation_prompt=True)
    return rec, inputs


def prompt_loader(records, processor, cfg, with_video=True, with_context=False, num_workers=4):
    fn = partial(collate_prompt, processor=processor, cfg=cfg, with_video=with_video, with_context=with_context)
    return make_loader(records, fn, shuffle=False, num_workers=num_workers)


# ---------------------------------------------------------------- model

def load_processor(cfg):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(cfg["model_id"])


def load_model(cfg, adapter=None):
    """Load Qwen3-VL in bf16 on the GPU, optionally with a trained LoRA adapter on top.

    bf16 = 2 bytes per parameter, so ~2.1B parameters take ~4.3 GB of the 8 GB card.
    """
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        cfg["model_id"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",  # PyTorch's fused attention kernel: faster and less memory than the naive version
        device_map="cuda",
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)  # wraps the frozen base model with the saved LoRA weights
    return model
