"""
Build release LMDBs from the full training LMDBs.

Two outputs:
  - video_features/  : one entry per unique video_id used in the JSONL files
  - text_features/   : NLQ query keys (unchanged) + feedback keys with
                       '__placeholder' suffix stripped

Resume-safe: already-written keys are skipped on restart.

Usage:
    python tools/build_release_lmdbs.py \
        --data_dir   <path/to/CVPR26_ReFocus/data> \
        --src_video  <path/to/decord_egovideo_video_features> \
        --src_text   <path/to/new_gte_qwen2_role> \
        --out_dir    <output_dir>
"""
import argparse
import json
import os
import time

import lmdb
import tqdm

FEEDBACK_TASKS = frozenset({"nlq_feedback", "goalstep_nlq_feedback", "hd_epic_nlq_feedback"})
COMMIT_EVERY = 5000   # write transaction batch size
MAP_SIZE = 800 * 1000**3  # 800 GB ceiling (LMDB only allocates what it uses on Linux)


def collect_keys(data_dir):
    """Return (video_ids, text_key_map) where text_key_map maps old_key -> new_key."""
    video_ids = set()
    text_key_map = {}  # old_lmdb_key -> new_lmdb_key

    for ds in sorted(os.listdir(data_dir)):
        for split in ("train.jsonl", "val.jsonl"):
            jsonl = os.path.join(data_dir, ds, split)
            if not os.path.exists(jsonl):
                continue
            with open(jsonl) as f:
                lines = [json.loads(l) for l in f]
            for x in lines:
                qid = x["query_id"]
                qt = x["query_type"]
                video_ids.add(x["video_id"])
                if qt in FEEDBACK_TASKS:
                    nlq_uid = qid.split("__")[1]
                    text_key_map[nlq_uid] = nlq_uid
                    if "feedback_earlier" in qid or "feedback_later" in qid:
                        old = qid.split("__")[2] + "__placeholder"
                        new = qid.split("__")[2]
                    else:
                        old = qid + "__placeholder"
                        new = qid
                    text_key_map[old] = new
                else:
                    text_key_map[qid] = qid

    return video_ids, text_key_map


def copy_lmdb_subset(src_dir, dst_dir, key_map, label):
    """Copy entries; skips keys already present in dst (resume support)."""
    os.makedirs(dst_dir, exist_ok=True)

    src_env = lmdb.open(src_dir, map_size=MAP_SIZE, readonly=True, create=False,
                        max_readers=8, readahead=False)
    src_txn = src_env.begin(buffers=True)

    dst_env = lmdb.open(dst_dir, map_size=MAP_SIZE, readonly=False, create=True,
                        max_readers=8)

    # Count already-done entries for resume
    with dst_env.begin() as chk:
        already_done = dst_env.stat()["entries"]
    print(f"  [{label}] {already_done} entries already in dst (resuming)")

    items = list(key_map.items())
    found = skipped = missing = 0
    t0 = time.time()

    batch_buf = {}   # new_key -> bytes, flushed every COMMIT_EVERY entries

    def flush():
        nonlocal found
        with dst_env.begin(write=True) as wtxn:
            for nk, val in batch_buf.items():
                wtxn.put(nk.encode(), val)
        found += len(batch_buf)
        batch_buf.clear()

    with tqdm.tqdm(items, desc=f"[{label}]", unit="key", dynamic_ncols=True) as bar:
        for old_key, new_key in bar:
            # Resume: skip if new_key already written
            with dst_env.begin() as rtxn:
                if rtxn.get(new_key.encode()) is not None:
                    skipped += 1
                    bar.set_postfix(found=found, skipped=skipped, missing=missing)
                    continue

            val = src_txn.get(old_key.encode())
            if val is None:
                missing += 1
                bar.set_postfix(found=found, skipped=skipped, missing=missing)
                continue

            batch_buf[new_key] = bytes(val)
            if len(batch_buf) >= COMMIT_EVERY:
                flush()

            bar.set_postfix(found=found, skipped=skipped, missing=missing)

    if batch_buf:
        flush()

    elapsed = time.time() - t0
    print(f"  [{label}] Done in {elapsed:.0f}s: "
          f"{found} copied, {skipped} skipped (resumed), {missing} missing")
    src_env.close()
    dst_env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir",  required=True, help="CVPR26_ReFocus/data/ directory")
    parser.add_argument("--src_video", required=True, help="Source video feature LMDB dir")
    parser.add_argument("--src_text",  required=True, help="Source text feature LMDB dir")
    parser.add_argument("--out_dir",   required=True, help="Output directory for release LMDBs")
    parser.add_argument("--skip_video", action="store_true", help="Skip video LMDB (already done)")
    parser.add_argument("--skip_text",  action="store_true", help="Skip text LMDB (already done)")
    args = parser.parse_args()

    print("Collecting required keys from JSONL files...")
    video_ids, text_key_map = collect_keys(args.data_dir)
    identity = sum(1 for k, v in text_key_map.items() if k == v)
    print(f"  video_ids:            {len(video_ids)}")
    print(f"  text keys total:      {len(text_key_map)}")
    print(f"    identity (NLQ):     {identity}")
    print(f"    renamed (feedback): {len(text_key_map) - identity}")

    out_video = os.path.join(args.out_dir, "video_features")
    out_text  = os.path.join(args.out_dir, "text_features")

    if not args.skip_video:
        print(f"\nBuilding video LMDB -> {out_video}")
        copy_lmdb_subset(args.src_video, out_video,
                         key_map={v: v for v in video_ids}, label="video")
    else:
        print(f"\nSkipping video LMDB (--skip_video)")

    if not args.skip_text:
        print(f"\nBuilding text LMDB  -> {out_text}")
        copy_lmdb_subset(args.src_text, out_text, text_key_map, label="text")
    else:
        print(f"\nSkipping text LMDB (--skip_text)")

    print("\nDone.")


if __name__ == "__main__":
    main()
