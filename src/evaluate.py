"""Multiple-choice accuracy, scored by comparing the model's probabilities for A-E.

No adapter = Arm A (zero-shot). With --adapter = Arm B or C.

Usage: python -m src.evaluate --config configs/pilot.yaml --split test [--adapter runs/arm_b/best] [--arm C]
"""

import torch


@torch.no_grad()
def evaluate(model, processor, records, cfg, with_context=False):
    """Return accuracy and per-question predictions."""
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
