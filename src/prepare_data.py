"""Convert raw Ego-EXTRA files into the standard JSONL used by every other script.

Reads   data/raw/egoextra/questions/final_questions/*.json
        data/raw/egoextra/video_clip_final/clips_mapping_*_duration_5.json
        data/raw/egoextra/protocol_split_wacv.json
Writes  data/splits/{train,val,test}.jsonl, data/splits/split_manifest.json
        data/frames/<clip_id>/000.jpg ... (num_frames JPEGs per clip)

What it fixes (see docs/training_plan_brief.md, Section 4):
- The correct answer is option A 97% of the time in the raw files -> we re-place it so A-E are exactly balanced.
- There is no official train/test split -> we split by VIDEO, balanced per scenario.
- Some videos have 500+ questions -> we cap questions per video.

Usage:
    python -m src.prepare_data --config configs/pilot.yaml               # full run (needs clips downloaded)
    python -m src.prepare_data --config configs/pilot.yaml --skip-frames # splits only, no ffmpeg
"""

import argparse
import collections
import glob
import json
import random
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from src.common import LETTERS, load_config, read_jsonl, write_jsonl

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------- loading raw files

def load_questions(raw_dir):
    """All questions from the 4 scenario files, with scenario and video id attached."""
    questions = []
    for path in sorted(glob.glob(f"{raw_dir}/questions/final_questions/*.json")):
        scenario = Path(path).name[:2]  # file names start with the scenario number, e.g. "03_keep_..."
        for q in json.load(open(path)):
            q["scenario"] = scenario
            q["video_id"] = q["univoc_id"].rsplit("_", 1)[0]  # "03_012_57" -> video "03_012"
            questions.append(q)
    return questions


def load_protocols(raw_dir):
    """video_id -> 1 (expert guides proactively) or 2 (expert only answers). Not a train/test split."""
    data = json.load(open(f"{raw_dir}/protocol_split_wacv.json"))
    return {v["ID"]: v["Protocol"] for v in data["videos"]}


def load_clip_mapping(raw_dir, clip_seconds):
    """question id -> clip id. Several questions close in time share one clip."""
    mapping = {}
    for path in sorted(glob.glob(f"{raw_dir}/video_clip_final/clips_mapping_*_duration_{clip_seconds}.json")):
        for scenario_map in json.load(open(path)).values():  # each file repeats earlier scenarios; that's harmless
            mapping.update(scenario_map)
    return mapping


def clip_path(raw_dir, clip_id, clip_seconds):
    video_id = clip_id.rsplit("_", 1)[0]
    return Path(raw_dir) / "video_clip_final" / clip_id[:2] / video_id / f"{clip_seconds}s" / f"{clip_id}.mp4"


# ---------------------------------------------------------------- cleaning one question

def normalize(text):
    """Lower-case and strip punctuation so 'First drawer.' matches 'first drawer'."""
    return re.sub(r"[^a-z0-9 ]", "", str(text).lower()).strip()


def answer_index(q):
    """Position of the correct answer among the options. correct_answer is a one-item list in the raw data."""
    answer = q["correct_answer"][0] if isinstance(q["correct_answer"], list) else q["correct_answer"]
    for i, opt in enumerate(q["options"]):
        if normalize(opt) == normalize(answer):
            return i
    raise ValueError(f"{q['univoc_id']}: answer {answer!r} not among options")


def expert_rationale(q, answer_text, max_words=60):
    """Arm C's training target text: what the expert actually said in the turn the question came from.

    Falls back to the answer text when that turn has no expert line (~5% of questions) or only
    filler like "Ok," (fewer than 3 words). Capped at max_words so one rambling turn can't blow up memory.
    """
    lines = [t[2:].strip() for t in q["conversation_turn"] if t.startswith("E:")]  # "E: put it there" -> "put it there"
    words = " ".join(lines).split()
    if len(words) < 3:
        words = answer_text.split()
    return " ".join(words[:max_words])


def previous_context(q):
    """Dialogue BEFORE the question (safe to show the model). Used only by the optional context arm."""
    names = {"E": "Expert", "A": "Trainee"}
    out = []
    for turn in q.get("context_previous") or []:
        text = (turn.get("text") or "").strip()
        if text and text != "No previous context available.":
            out.append(f"{names.get(turn.get('speaker'), turn.get('speaker'))}: {text}")
    return out


def question_type(text):
    """First two words, e.g. 'what should'. Used to break down results by kind of question."""
    return " ".join(text.lower().split()[:2])


def to_record(q, clip_id, protocol):
    """Raw question -> our standard record (options are re-ordered later by balance_answer_positions)."""
    idx = answer_index(q)
    answer_text = q["options"][idx]
    return {
        "id": q["univoc_id"],
        "video_id": q["video_id"],
        "scenario": q["scenario"],
        "protocol": protocol,  # 1, 2 or None (13 videos are missing from the protocol file)
        "clip_id": clip_id,
        "question": q["generated_question"].strip(),
        "question_type": question_type(q["generated_question"]),
        "options": list(q["options"]),
        "answer_idx": idx,
        "answer_text": answer_text,
        "rationale": expert_rationale(q, answer_text),
        "context": previous_context(q),
    }


# ---------------------------------------------------------------- splitting and sampling

def split_videos(records, ratios, seed):
    """Assign every VIDEO (not question) to train/val/test, separately within each scenario.

    Greedy: walk videos in random order and give each to the split furthest below its target share of
    questions. Each scenario first gets one video in val and one in test, so none is missing from a split.
    """
    rng = random.Random(f"{seed}-split")  # string seeds are deterministic in Python's random module
    counts = collections.Counter(r["video_id"] for r in records)
    by_scenario = collections.defaultdict(list)
    for vid in sorted(counts):
        by_scenario[vid[:2]].append(vid)
    assignment = {}
    for scenario, vids in sorted(by_scenario.items()):
        rng.shuffle(vids)
        assigned = {s: 0 for s in SPLITS}  # questions assigned to each split so far
        for split, vid in zip(("val", "test"), vids[:2]):
            assignment[vid] = split
            assigned[split] += counts[vid]
        for vid in vids[2:]:
            total = sum(assigned.values()) + counts[vid]
            deficit = {s: ratios[i] * total - assigned[s] for i, s in enumerate(SPLITS)}
            split = max(deficit, key=deficit.get)
            assignment[vid] = split
            assigned[split] += counts[vid]
    return assignment


def cap_per_video(records, max_per_video, rng):
    """Keep at most max_per_video questions from each video, chosen at random."""
    by_video = collections.defaultdict(list)
    for r in records:
        by_video[r["video_id"]].append(r)
    kept = []
    for vid in sorted(by_video):
        group = by_video[vid]
        rng.shuffle(group)
        kept += group[:max_per_video]
    return kept


def balance_answer_positions(records, rng):
    """Re-order each question's options so the correct letter is spread exactly evenly over A-E.

    Plain shuffling would be ~even by chance; assigning positions explicitly makes it exact, so
    "always answer X" scores exactly 20% on every split.
    """
    positions = [i % len(LETTERS) for i in range(len(records))]
    rng.shuffle(positions)
    for rec, pos in zip(records, positions):
        correct = rec["options"][rec["answer_idx"]]
        others = [o for i, o in enumerate(rec["options"]) if i != rec["answer_idx"]]
        rng.shuffle(others)
        others.insert(pos, correct)
        rec["options"], rec["answer_idx"] = others, pos
    return records


def check_no_leakage(splits):
    """Fail loudly if any video or clip appears in more than one split."""
    for key in ("video_id", "clip_id"):
        seen = {s: {r[key] for r in recs} for s, recs in splits.items()}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = seen[a] & seen[b]
            assert not overlap, f"LEAKAGE: {key} in both {a} and {b}: {sorted(overlap)[:5]}"


# ---------------------------------------------------------------- frames

def extract_frames(clip, out_dir, num_frames, size):
    """Save num_frames evenly spaced frames of `clip` as size x size JPEGs. Returns the clip duration (s).

    We pick frame indices ourselves (centre of each of num_frames equal time slices) and let ffmpeg's
    `select` filter grab exactly those frames in one pass.
    """
    out_dir = Path(out_dir)
    info = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets,r_frame_rate", "-of", "json", str(clip)],
        capture_output=True, text=True, check=True).stdout)["streams"][0]
    total = int(info["nb_read_packets"])  # number of frames in the clip (packets = frames for H.264 video)
    num, den = map(int, info["r_frame_rate"].split("/"))
    duration = total / (num / den)  # e.g. 75 frames / 15 fps = 5.0 s
    if len(list(out_dir.glob("*.jpg"))) == num_frames:
        return duration  # already extracted
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = sorted({min(total - 1, int((i + 0.5) * total / num_frames)) for i in range(num_frames)})
    select = "+".join(f"eq(n\\,{i})" for i in indices)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(clip),
         "-vf", f"select='{select}',scale={size}:{size}:flags=bicubic",
         "-fps_mode", "vfr",  # output only the selected frames, don't duplicate to keep a constant rate
         "-q:v", "2",  # JPEG quality (2 = high)
         str(out_dir / "%03d.jpg")],
        check=True)
    frames = sorted(out_dir.glob("*.jpg"))
    while len(frames) < num_frames:  # very short clip: repeat the last frame to keep the count fixed
        extra = out_dir / f"{len(frames) + 1:03d}.jpg"
        extra.write_bytes(frames[-1].read_bytes())
        frames.append(extra)
    return duration


def attach_frames(splits, cfg, workers):
    """Extract frames for every clip used by any split and store their paths + duration on the records."""
    d = cfg["data"]
    clips = sorted({r["clip_id"] for recs in splits.values() for r in recs})
    missing = [c for c in clips if not clip_path(d["raw_dir"], c, d["clip_seconds"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} clips not downloaded (e.g. {missing[:3]}). "
                         f"Run scripts/download_egoextra.py first, or use --skip-frames.")

    def work(clip_id):
        dur = extract_frames(clip_path(d["raw_dir"], clip_id, d["clip_seconds"]),
                             Path(d["frames_dir"]) / clip_id, d["num_frames"], d["frame_size"])
        return clip_id, dur

    durations = {}
    with ThreadPoolExecutor(workers) as pool:  # ffmpeg runs as separate processes, so threads parallelise fine
        for clip_id, dur in tqdm(pool.map(work, clips), total=len(clips), desc="frames"):
            durations[clip_id] = dur
    for recs in splits.values():
        for r in recs:
            r["frames"] = [str(Path(d["frames_dir"]) / r["clip_id"] / f"{i + 1:03d}.jpg") for i in range(d["num_frames"])]
            r["duration"] = durations[r["clip_id"]]


# ---------------------------------------------------------------- main

def summarize(splits):
    for name, recs in splits.items():
        letters = collections.Counter(LETTERS[r["answer_idx"]] for r in recs)
        scen = collections.Counter(r["scenario"] for r in recs)
        print(f"{name:5s} {len(recs):5d} questions, {len({r['video_id'] for r in recs}):3d} videos | "
              f"scenarios {dict(sorted(scen.items()))} | answer letters {dict(sorted(letters.items()))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--skip-frames", action="store_true", help="write splits without extracting frames")
    ap.add_argument("--force", action="store_true", help="allow overwriting an existing (frozen) test split")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config)
    d, seed = cfg["data"], cfg["seed"]
    out = Path(d["splits_dir"])

    # The test set is frozen once created: regenerating it could silently change every reported number.
    if (out / "test.jsonl").exists() and not args.force:
        old = read_jsonl(out / "test.jsonl")
        if any(not r.get("frames") for r in old) and not args.skip_frames:
            print("existing splits have no frames yet; regenerating them with frames (same seed -> same split)")
        else:
            raise SystemExit(f"{out}/test.jsonl exists and is frozen. Use --force only if you mean to change it.")

    questions = load_questions(d["raw_dir"])
    protocols = load_protocols(d["raw_dir"])
    mapping = load_clip_mapping(d["raw_dir"], d["clip_seconds"])
    records = [to_record(q, mapping[q["univoc_id"]], protocols.get(q["video_id"]))
               for q in questions if q["univoc_id"] in mapping]
    print(f"{len(questions)} questions, {len(records)} with a {d['clip_seconds']} s clip")

    assignment = split_videos(records, d["split_ratios"], seed)
    full = {s: [r for r in records if assignment[r["video_id"]] == s] for s in SPLITS}

    rng = random.Random(f"{seed}-sample")
    train = cap_per_video(full["train"], d["max_per_video"], rng)
    if len(train) > d["train_size"]:
        train = rng.sample(train, d["train_size"])
    else:
        print(f"WARNING: only {len(train)} training questions after the per-video cap (wanted {d['train_size']})")
    val = cap_per_video(full["val"], d["max_per_video_eval"], rng)
    val = rng.sample(val, min(d["val_size"], len(val)))
    test = cap_per_video(full["test"], d["max_per_video_eval"], rng)
    splits = {"train": train, "val": val, "test": test}

    for name in SPLITS:  # separate seeded RNG per split, so changing train_size can't alter the test set
        balance_answer_positions(splits[name], random.Random(f"{seed}-options-{name}"))
        splits[name].sort(key=lambda r: r["id"])
    check_no_leakage(splits)
    summarize(splits)

    if args.skip_frames:
        print("--skip-frames: records have no 'frames' yet; rerun without it after downloading clips")
    else:
        attach_frames(splits, cfg, args.workers)

    for name, recs in splits.items():
        write_jsonl(out / f"{name}.jsonl", recs)
    manifest = {"seed": seed, "video_split": assignment,
                "counts": {s: len(v) for s, v in splits.items()},
                "full_counts_before_capping": {s: len(v) for s, v in full.items()}}
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out}/{{train,val,test}}.jsonl")


if __name__ == "__main__":
    main()
