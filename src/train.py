"""LoRA fine-tuning of Qwen3-VL-2B on Ego-EXTRA questions.

Loss is computed on the answer tokens only. Validates after every epoch, keeps the
best adapter and stops early when validation stops improving.

Usage: python -m src.train --config configs/pilot.yaml --arm B
       (--arm C adds the preceding expert-trainee dialogue to each prompt)
"""


def add_lora(model, cfg):
    raise NotImplementedError


def make_example(processor, rec, cfg, with_context):
    """Encode prompt plus answer and mask everything except the answer in labels."""
    raise NotImplementedError


def train(cfg, arm):
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
