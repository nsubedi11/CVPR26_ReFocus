"""
Pack extracted feature files into LMDB databases for fast random access.

  Video:  .pt files  (one per clip)      → video_features/ LMDB
  Text:   .pkl files (one per shard)     → text_features/  LMDB

LMDB entry format (both DBs):
    key   : bytes (clip_id or annotation_uid)
    value : numpy .npz bytes
              video → {"features": float32 (T, C)}
              text  → {"features": float32 (L, D), "pred_idx"?, "role_idx"?}

Usage:
    # video features
    python pack_to_lmdb.py video \
        --input_dir  path/to/video_feat_pt \
        --output_dir path/to/lmdb/video_features

    # text features (merges all pkl shards)
    python pack_to_lmdb.py text \
        --input_dir  path/to/text_feat_pkl \
        --output_dir path/to/lmdb/text_features
"""
import argparse
import glob
import io
import os
import pickle

import lmdb
import numpy as np
import torch
from tqdm import tqdm

MAP_SIZE = 500 * 1000**3   # 500 GB ceiling (LMDB only allocates what it uses)
COMMIT_EVERY = 2000


def pack_video(input_dir, output_dir):
    pt_files = sorted(glob.glob(os.path.join(input_dir, "*.pt")))
    print(f"Found {len(pt_files)} .pt files")

    os.makedirs(output_dir, exist_ok=True)
    env = lmdb.open(output_dir, map_size=MAP_SIZE)

    buf = {}
    def flush():
        with env.begin(write=True) as txn:
            for k, v in buf.items():
                txn.put(k, v)
        buf.clear()

    for pt_path in tqdm(pt_files):
        clip_id = os.path.splitext(os.path.basename(pt_path))[0]
        feats = torch.load(pt_path, map_location="cpu").numpy().astype(np.float32)
        bio = io.BytesIO()
        np.savez(bio, features=feats)
        buf[clip_id.encode()] = bio.getvalue()
        if len(buf) >= COMMIT_EVERY:
            flush()
    if buf:
        flush()

    env.close()
    with lmdb.open(output_dir, readonly=True, create=False) as e:
        n = e.stat()["entries"]
    print(f"Video LMDB: {n} entries → {output_dir}")


def pack_text(input_dir, output_dir):
    pkl_files = sorted(glob.glob(os.path.join(input_dir, "*.pkl")))
    print(f"Found {len(pkl_files)} .pkl files")

    os.makedirs(output_dir, exist_ok=True)
    env = lmdb.open(output_dir, map_size=MAP_SIZE)

    # resume: skip already-written keys
    with env.begin() as txn:
        already_done = env.stat()["entries"]
    print(f"  {already_done} entries already in dst (resuming)")

    buf = {}

    def flush():
        with env.begin(write=True) as txn:
            for k, v in buf.items():
                txn.put(k, v)
        buf.clear()

    for pkl_path in tqdm(pkl_files):
        # filename is the sanitized annotation UID
        ann_id = os.path.splitext(os.path.basename(pkl_path))[0].replace("_", "/", 0)
        with env.begin() as txn:
            if txn.get(ann_id.encode()) is not None:
                continue
        with open(pkl_path, "rb") as f:
            feat_dict = pickle.load(f)
        bio = io.BytesIO()
        np.savez(bio, **{k: np.array(v) for k, v in feat_dict.items()})
        buf[ann_id.encode()] = bio.getvalue()
        if len(buf) >= COMMIT_EVERY:
            flush()

    if buf:
        flush()

    env.close()
    with lmdb.open(output_dir, readonly=True, create=False) as e:
        n = e.stat()["entries"]
    print(f"Text LMDB: {n} entries → {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    vp = sub.add_parser("video", help="Pack .pt video features into LMDB")
    vp.add_argument("--input_dir",  required=True, help="Directory with .pt files")
    vp.add_argument("--output_dir", required=True, help="Output LMDB directory")

    tp = sub.add_parser("text", help="Pack .pkl text features into LMDB")
    tp.add_argument("--input_dir",  required=True, help="Directory with .pkl shards")
    tp.add_argument("--output_dir", required=True, help="Output LMDB directory")

    args = parser.parse_args()
    if args.mode == "video":
        pack_video(args.input_dir, args.output_dir)
    else:
        pack_text(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
