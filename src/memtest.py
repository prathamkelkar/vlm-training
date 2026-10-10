"""Run a few LoRA training steps and report peak VRAM and time per step.

Run this before any real training. With --dummy it uses generated frames, so it
works before the dataset is downloaded.

Usage: python -m src.memtest --config configs/pilot.yaml [--dummy]
"""


def make_dummy_record(cfg):
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
