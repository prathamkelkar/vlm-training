"""Convert raw Ego-EXTRA files into the standard JSONL used by every other script.

Reads final_questions/*.json, clips_mapping_*_duration_5.json and protocol_split_wacv.json.
Writes data/splits/{train,val,test}.jsonl and data/frames/<clip_id>/000.jpg ...

Standard record:
    {"id", "video_id", "scenario", "frames": [...], "duration", "question",
     "options": [5 strings], "answer_idx", "context": "preceding dialogue (Arm C)"}

Rules:
- Shuffle options with a fixed seed so the answer position carries no signal.
- Use the official split and check that no video_id appears in more than one split.
"""


def load_questions(raw_dir):
    raise NotImplementedError


def load_clip_mapping(raw_dir, clip_seconds):
    raise NotImplementedError


def to_record(question, clip_path, rng):
    raise NotImplementedError


def extract_frames(clip_path, out_dir, num_frames, size):
    """Sample num_frames evenly with ffmpeg and save them as JPEGs."""
    raise NotImplementedError


def assign_splits(records, split_file):
    raise NotImplementedError


def check_no_leakage(splits):
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
