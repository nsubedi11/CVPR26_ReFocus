"""
Build subset LMDBs (video + text features) for each eval split.

Reads the eval JSONL to collect required video_ids and text keys,
then copies only those entries from the full LMDBs.

Output:
  data/features/subset_lmdb/<split_name>/video_features/
  data/features/subset_lmdb/<split_name>/text_features/
"""
import argparse
import json
import os
import sys

import lmdb


SPLITS = [
    ("ego4d_qnf_val",     "data/ego4d_qnf/val.jsonl"),
    ("goalstep_qnf_test", "data/goalstep_qnf/test.jsonl"),
    ("hd_epic_qnf_test",  "data/hd_epic_qnf/test.jsonl"),
]

VIDEO_LMDB = "data/features/offline_lmdb/decord_egovideo_video_features"
TEXT_LMDB  = "data/features/offline_lmdb/new_gte_qwen2_role"


def collect_keys(jsonl_path):
    video_ids = set()
    text_ids = set()
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            video_ids.add(r["video_id"])
            qid = r["query_id"]
            parts = qid.split("__")
            if len(parts) > 1:
                text_ids.add(parts[1])          # NLQ query feats
            if "feedback_earlier" in qid or "feedback_later" in qid:
                text_ids.add(parts[2])
            else:
                text_ids.add(qid)               # feedback feats (keyed by full qid)
    return video_ids, text_ids


def measure_size(env, keys_bytes):
    """Return total byte size of values for the given keys."""
    total = 0
    with env.begin(buffers=True) as txn:
        for k in keys_bytes:
            v = txn.get(k)
            if v is not None:
                total += len(v)
    return total


def copy_subset(src_env, dst_path, keys_bytes, map_size, label):
    os.makedirs(dst_path, exist_ok=True)
    dst_env = lmdb.open(dst_path, map_size=map_size, create=True)
    found = 0
    missing = 0
    with src_env.begin(buffers=True) as src_txn:
        with dst_env.begin(write=True) as dst_txn:
            for k in keys_bytes:
                v = src_txn.get(k)
                if v is None:
                    missing += 1
                    print(f"  [WARN] key not found: {k.decode()!r}", flush=True)
                    continue
                dst_txn.put(k, bytes(v))
                found += 1
    dst_env.close()
    if missing:
        print(f"  {label}: copied {found}, missing {missing}")
    else:
        print(f"  {label}: copied {found} keys")


def main(args):
    out_root = args.output_dir

    src_video_env = lmdb.open(VIDEO_LMDB, readonly=True, create=False,
                               max_readers=4, lock=False)
    src_text_env  = lmdb.open(TEXT_LMDB,  readonly=True, create=False,
                               max_readers=4, lock=False)

    for split_name, jsonl_path in SPLITS:
        if args.splits and split_name not in args.splits:
            continue
        print(f"\n=== {split_name} ===")

        video_ids, text_ids = collect_keys(jsonl_path)
        print(f"  {len(video_ids)} video keys, {len(text_ids)} text keys")

        video_keys = [v.encode() for v in video_ids]
        text_keys  = [t.encode() for t in text_ids]

        # Measure actual data sizes to set map_size (2× headroom)
        print("  Measuring video data size...", flush=True)
        video_bytes = measure_size(src_video_env, video_keys)
        print(f"  Video data: {video_bytes / 1e9:.2f} GB")

        print("  Measuring text data size...", flush=True)
        text_bytes = measure_size(src_text_env, text_keys)
        print(f"  Text data:  {text_bytes / 1e9:.2f} GB")

        split_dir = os.path.join(out_root, split_name)

        print("  Copying video LMDB...", flush=True)
        copy_subset(
            src_video_env,
            os.path.join(split_dir, "video_features"),
            video_keys,
            map_size=max(video_bytes * 2, 1 * 1024**3),
            label="video",
        )

        print("  Copying text LMDB...", flush=True)
        copy_subset(
            src_text_env,
            os.path.join(split_dir, "text_features"),
            text_keys,
            map_size=max(text_bytes * 2, 1 * 1024**3),
            label="text",
        )

        print(f"  Done → {split_dir}")

    src_video_env.close()
    src_text_env.close()
    print("\nAll subsets built.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="data/features/subset_lmdb",
                        help="Root output directory for subset LMDBs")
    parser.add_argument("--splits", nargs="*", default=None,
                        help="Subset of splits to process (default: all)")
    args = parser.parse_args()
    main(args)
