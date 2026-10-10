"""Multiple-choice accuracy, scored by comparing the model's probabilities for the letters A-E.

No adapter     -> Arm A (zero-shot).      --adapter runs/arm_b_seed0/best -> Arm B (or C).
--blind        -> same model, question and options only, no video (the "blind" control).
--both         -> run with video AND blind in one go.

Usage:
    python -m src.evaluate --config configs/pilot.yaml --split test --both
    python -m src.evaluate --config configs/pilot.yaml --split test --both --adapter runs/arm_b_seed0/best
Results go to runs/eval/<name>_<split>_<video|blind>.json
"""

import argparse
import collections
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm

from src.common import LETTERS, letter_token_ids, load_config, load_model, load_processor, prompt_loader, read_jsonl


@torch.no_grad()  # no gradients needed for scoring: saves memory and time
def evaluate(model, processor, records, cfg, with_video=True, with_context=False, num_workers=4, desc="eval"):
    """Score every record; return accuracy, breakdowns and per-question predictions."""
    model.eval()  # turns off dropout so scoring is deterministic
    letter_ids = torch.tensor(letter_token_ids(processor.tokenizer), device="cuda")
    preds = []
    loader = prompt_loader(records, processor, cfg, with_video, with_context, num_workers)
    for rec, inputs in tqdm(loader, total=len(records), desc=desc, leave=False):
        inputs = inputs.to("cuda")
        # One forward pass over the prompt. logits_to_keep=1: we only need the prediction for the NEXT
        # token (the first token of the answer), not for every position in the prompt.
        logits = model(**inputs, logits_to_keep=1).logits[0, -1].float()  # shape: (vocab_size,)
        log_probs = logits.log_softmax(-1)  # probabilities over the WHOLE vocabulary (~151k tokens)
        letter_mass = log_probs[letter_ids].exp().sum().item()  # how much of that goes to A-E at all
        probs = logits[letter_ids].softmax(-1)  # renormalise over just the 5 letters
        pred = int(probs.argmax())
        preds.append({
            "id": rec["id"], "pred": pred, "answer": rec["answer_idx"], "correct": pred == rec["answer_idx"],
            "p_correct": round(probs[rec["answer_idx"]].item(), 4), "letter_mass": round(letter_mass, 4),
            "scenario": rec["scenario"], "protocol": rec["protocol"], "question_type": rec["question_type"],
        })
    return summarize(preds)


def accuracy_with_ci(correct, n):
    """Accuracy and a 95% confidence half-width (normal approximation): ±1.96 * sqrt(p(1-p)/n)."""
    p = correct / n if n else 0.0
    return p, 1.96 * math.sqrt(p * (1 - p) / n) if n else 0.0


def summarize(preds):
    def group(key, top=None):
        buckets = collections.defaultdict(list)
        for p in preds:
            buckets[str(p[key])].append(p["correct"])
        items = sorted(buckets.items(), key=lambda kv: -len(kv[1]))[:top] if top else sorted(buckets.items())
        return {k: {"n": len(v), "acc": round(sum(v) / len(v), 4)} for k, v in items}

    acc, ci = accuracy_with_ci(sum(p["correct"] for p in preds), len(preds))
    return {
        "accuracy": round(acc, 4),
        "ci95": round(ci, 4),
        "n": len(preds),
        # If one letter dominates the predictions, the model has a position bias (each is 20% of answers).
        "predicted_letters": {LETTERS[k]: v for k, v in sorted(collections.Counter(p["pred"] for p in preds).items())},
        # Near 1.0 = the model really wants to answer with a letter. Low = it wanted to say something else
        # ("The...", "I think..."), so the zero-shot score partly reflects format, not understanding.
        "mean_letter_mass": round(sum(p["letter_mass"] for p in preds) / max(len(preds), 1), 4),
        "by_scenario": group("scenario"),
        "by_protocol": group("protocol"),
        "by_question_type": group("question_type", top=8),
        "predictions": preds,
    }


def print_report(name, res):
    print(f"\n== {name}: accuracy {res['accuracy']:.3f} ± {res['ci95']:.3f} (n={res['n']}, chance 0.200)")
    print(f"   predicted letters {res['predicted_letters']}   mean letter mass {res['mean_letter_mass']:.3f}")
    for key in ("by_scenario", "by_protocol", "by_question_type"):
        print(f"   {key}: " + ", ".join(f"{k} {v['acc']:.2f} (n={v['n']})" for k, v in res[key].items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--adapter", default=None, help="path to a saved LoRA adapter (omit for zero-shot)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--blind", action="store_true", help="no video: question and options only")
    mode.add_argument("--both", action="store_true", help="run with video and blind")
    ap.add_argument("--context", action="store_true", help="include earlier dialogue in the prompt (arm C')")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--name", default=None, help="label for the output file (default: from the adapter path)")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config(args.config)
    records = read_jsonl(Path(cfg["data"]["splits_dir"]) / f"{args.split}.jsonl", args.limit)
    name = args.name or (Path(args.adapter).parent.name if args.adapter else "zero_shot")
    if args.context:
        name += "_ctx"
    processor = load_processor(cfg)
    model = load_model(cfg, args.adapter)

    modes = [True, False] if args.both else [not args.blind]
    out_dir = Path(cfg["train"]["output_dir"]) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    for with_video in modes:
        tag = f"{name}_{args.split}_{'video' if with_video else 'blind'}"
        res = evaluate(model, processor, records, cfg, with_video, args.context, args.num_workers, desc=tag)
        res.update({"name": name, "split": args.split, "with_video": with_video, "adapter": args.adapter})
        (out_dir / f"{tag}.json").write_text(json.dumps(res, indent=1))
        print_report(tag, res)
    print(f"\nresults saved in {out_dir}/")


if __name__ == "__main__":
    main()
