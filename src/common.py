"""Shared helpers: config loading, prompt building, input encoding and model loading."""

LETTERS = "ABCDE"


def load_config(path):
    raise NotImplementedError


def read_jsonl(path, limit=None):
    raise NotImplementedError


def build_messages(rec, cfg, with_answer=False, with_context=False):
    """Chat messages for one question; with_context adds the preceding dialogue (Arm C)."""
    raise NotImplementedError


def encode(processor, messages, add_generation_prompt):
    """Return (chat text, model inputs) via qwen_vl_utils.process_vision_info."""
    raise NotImplementedError


def load_model(cfg, adapter=None):
    """bf16 model on CUDA with SDPA attention; optionally load a LoRA adapter."""
    raise NotImplementedError
