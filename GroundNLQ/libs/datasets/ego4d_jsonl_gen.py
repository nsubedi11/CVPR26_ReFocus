"""
Generate JSONL dataset files from GroundNLQ JSON annotations.
Replaces the LMDB-based ego4d_gen.py + anchored_cluster_data_gen.ipynb pipeline.
Each line is a self-contained sample consumed by ego4d_jsonl_loader.py.

Usage (from GroundNLQ/):
    python libs/datasets/ego4d_jsonl_gen.py \
        --data_dir ego4d_data \
        --output_dir dataset/jsonl \
        [--splits train val test] [--debug]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from basic_utils import load_json, save_jsonl


def gen_nlq(vid, data_item, test_split=False):
    duration = float(data_item["clip_duration"])
    lang = data_item.get("language_queries", {})
    records = []
    for time, sent, uid in zip(
        lang.get("exact_times", []),
        lang.get("sentences", []),
        lang.get("annotation_uids", []),
    ):
        if not sent:
            continue
        records.append({
            "video_id": vid,
            "query_id": uid,
            "query_type": "nlq",
            "query": sent.strip(),
            "timestamps": [] if test_split else [list(time)],
            "clip_span": [0.0, duration],
            "duration": duration,
            "pred_timestamp": [],
        })
    return records


def gen_narr(vid, data_item):
    duration = float(data_item["clip_duration"])
    narr_passes = data_item.get("narrations", [])
    seen = set()
    records = []
    for narr_pass in narr_passes:
        for uid, sent, time in zip(
            narr_pass.get("annotation_uids", []),
            narr_pass.get("sentences", []),
            narr_pass.get("exact_times", []),
        ):
            if uid in seen or not sent:
                continue
            seen.add(uid)
            records.append({
                "video_id": vid,
                "query_id": uid,
                "query_type": "narr",
                "query": sent.strip(),
                "timestamps": [list(time)],
                "clip_span": [0.0, duration],
                "duration": duration,
                "pred_timestamp": [],
            })
    return records


def gen_nlq_feedback(vid, data_item):
    duration = float(data_item["clip_duration"])
    fb = data_item.get("nlq_feedbacks", {})
    feedbacks = fb.get("feedbacks", [])
    records = []
    for i, (time, sent, uid, pred_time) in enumerate(zip(
        fb.get("exact_times", []),
        fb.get("sentences", []),
        fb.get("annotation_uids", []),
        fb.get("pred_exact_times", []),
    )):
        if not sent:
            continue
        records.append({
            "video_id": vid,
            "query_id": uid,
            "query_type": "nlq_feedback",
            "query": sent.strip(),
            "feedback": feedbacks[i].strip() if i < len(feedbacks) else "",
            "timestamps": [list(time)],
            "clip_span": [0.0, duration],
            "duration": duration,
            "pred_timestamp": list(pred_time),
        })
    return records


def gen_subclip_task(vid, data_item, item_key, query_type, duration, is_feedback=False):
    """Handle goalstep_nlq, hd_epic_nlq and their feedback variants (sub-clip tasks).
    Timestamps in the output are relative to clip_start, matching ego4d_gen.py behavior."""
    item = data_item.get(item_key, {})
    clip_starts = item.get("clip_starts", [])
    clip_ends = item.get("clip_ends", [])
    exact_times = item.get("exact_times", [])
    sentences = item.get("sentences", [])
    uids = item.get("annotation_uids", [])
    pred_times = item.get("pred_exact_times", []) if is_feedback else [[] for _ in uids]
    feedbacks = item.get("feedbacks", []) if is_feedback else []

    records = []
    for i, (time, sent, uid) in enumerate(zip(exact_times, sentences, uids)):
        if not sent:
            continue
        clip_s = max(0.0, float(clip_starts[i]))
        clip_e = min(float(clip_ends[i]), duration)
        clip_dur = clip_e - clip_s
        rel_time = [float(time[0]) - clip_s, float(time[1]) - clip_s]
        pred_t = list(pred_times[i]) if is_feedback and pred_times[i] else []
        record = {
            "video_id": vid,
            "query_id": uid,
            "query_type": query_type,
            "query": sent.strip(),
            "timestamps": [rel_time],
            "clip_span": [clip_s, clip_e],
            "duration": clip_dur,
            "pred_timestamp": pred_t,
        }
        if is_feedback:
            record["feedback"] = feedbacks[i].strip() if i < len(feedbacks) else ""
        records.append(record)

    return records


def process_data(data, task_types, test_split=False):
    records = []
    for vid, data_item in data.items():
        duration = float(data_item.get("clip_duration", 0))
        if "nlq_feedback" in task_types:
            records.extend(gen_nlq_feedback(vid, data_item))
        if "goalstep_nlq_feedback" in task_types:
            records.extend(gen_subclip_task(vid, data_item, "goalstep_nlq_feedbacks", "goalstep_nlq_feedback",
                                            duration, is_feedback=True))
        if "hd_epic_nlq_feedback" in task_types:
            records.extend(gen_subclip_task(vid, data_item, "hd_epic_nlq_feedbacks", "hd_epic_nlq_feedback",
                                            duration, is_feedback=True))
    return records


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    if "train" in args.splits:
        print("Processing training data...")
        train_data = load_json(os.path.join(args.data_dir, "train.json"))
        if args.debug:
            train_data = dict(list(train_data.items())[:20])
        train_specs = [
            (["nlq_feedback", "goalstep_nlq_feedback", "hd_epic_nlq_feedback"], "train_all.jsonl"),
            (["nlq_feedback"],                                                    "ego4d_qnf_train.jsonl"),
            (["goalstep_nlq_feedback"],                                           "goalstep_qnf_train.jsonl"),
            (["hd_epic_nlq_feedback"],                                            "hd_epic_qnf_train.jsonl"),
        ]
        for tasks, dst_file in train_specs:
            records = process_data(train_data, tasks)
            out = os.path.join(args.output_dir, dst_file)
            save_jsonl(records, out)
            print(f"  Saved {len(records)} records → {out}")

    if "val" in args.splits:
        val_specs = [
            ("nlq_feedback_val.json",          ["nlq_feedback"],          "nlq_feedback_val.jsonl"),
            ("goalstep_nlq_feedback_val.json", ["goalstep_nlq_feedback"], "goalstep_nlq_feedback_val.jsonl"),
            ("hd_epic_nlq_feedback_val.json",  ["hd_epic_nlq_feedback"],  "hd_epic_nlq_feedback_val.jsonl"),
        ]
        for src_file, tasks, dst_file in val_specs:
            src = os.path.join(args.data_dir, src_file)
            if not os.path.exists(src):
                print(f"  Skipping {src_file} (not found)")
                continue
            data = load_json(src)
            if args.debug:
                data = dict(list(data.items())[:5])
            records = process_data(data, tasks)
            out = os.path.join(args.output_dir, dst_file)
            save_jsonl(records, out)
            print(f"  Saved {len(records)} {tasks[0]} val records → {out}")

    if "test" in args.splits:
        print("Processing test data...")
        test_specs = [
            ("goalstep_nlq_test.json", ["goalstep_nlq_feedback"], "goalstep_nlq_feedback_test.jsonl", False),
            ("hd_epic_nlq_test.json",  ["hd_epic_nlq_feedback"],  "hd_epic_nlq_feedback_test.jsonl",  False),
        ]
        for src_file, tasks, dst_file, is_nlq_test in test_specs:
            src = os.path.join(args.data_dir, src_file)
            if not os.path.exists(src):
                print(f"  Skipping {src_file} (not found)")
                continue
            data = load_json(src)
            if args.debug:
                data = dict(list(data.items())[:5])
            records = process_data(data, tasks, test_split=is_nlq_test)
            out = os.path.join(args.output_dir, dst_file)
            save_jsonl(records, out)
            print(f"  Saved {len(records)} {tasks[0]} test records → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing train.json, nlq_val.json, etc.")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for JSONL files")
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        choices=["train", "val", "test"])
    parser.add_argument("--debug", action="store_true",
                        help="Process a small subset (20 train clips, 5 val clips)")
    args = parser.parse_args()
    main(args)
