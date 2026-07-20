"""
Extract per-clip video features using the EgoVideo backbone.

Reads clip IDs from the EM-QnF JSONL files, locates the corresponding .mp4
files, and saves frame-level features as .pt files (one per clip).

Output:
    <output_dir>/<clip_id>.pt   — FloatTensor of shape (T, 512)
    where T = ceil(num_frames / 4) with stride-4 frame sampling.

Usage:
    python extract_video_features.py \
        --jsonl_dir   path/to/data \
        --video_dirs  path/to/clips path/to/full_scale path/to/hd_epic_videos \
        --ckpt        path/to/ckpt_4frames.pth \
        --output_dir  path/to/video_feat_pt \
        [--dataset_divider 4 --assigned_part 0]
"""
import argparse
import glob
import json
import os
import subprocess
import time

import decord
import torch
from torchvision.transforms import v2
from tqdm import tqdm

decord.bridge.set_bridge('torch')

BATCH_SIZE = 32
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 5e10:
    BATCH_SIZE *= 2


def get_video_num_frames(video_path):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=width,height,nb_frames',
           '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.strip().split('\n')
    if len(out) == 3:
        return int(out[0]), int(out[1]), int(out[2]) if out[2].isdigit() else 0
    raise ValueError(f"ffprobe failed for {video_path}: {out}")


def get_resized_hw(h, w, short_side=256):
    if h < w:
        return short_side, int(w * short_side / h)
    return int(h * short_side / w), short_side


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_dirs, clips):
        self.video_dirs = video_dirs
        self.clips = clips
        self.transform = v2.Compose([
            v2.Lambda(lambda x: x.permute(0, 3, 1, 2)),
            v2.CenterCrop(224),
            v2.Lambda(lambda x: x.float() / 255.),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        path = None
        for d in self.video_dirs:
            p = os.path.join(d, f"{clip}.mp4")
            if os.path.exists(p):
                path = p
                break
        if path is None:
            raise FileNotFoundError(f"Clip not found: {clip}")
        _, h, w = get_video_num_frames(path)
        rh, rw = get_resized_hw(h, w)
        video = decord.VideoReader(path, width=rw, height=rh, num_threads=1)
        frames = video.get_batch(range(0, len(video), 4))
        return {"clip": clip, "frames": self.transform(frames)}


def collect_clip_ids(jsonl_dir):
    ids = set()
    for root, _, files in os.walk(jsonl_dir):
        for f in files:
            if f.endswith('.jsonl'):
                with open(os.path.join(root, f)) as fh:
                    for line in fh:
                        ids.add(json.loads(line)["video_id"])
    return sorted(ids)


def main(args):
    from EgoVideo.backbone.model.setup_model import build_model

    assert torch.cuda.is_available(), "GPU required"
    os.makedirs(args.output_dir, exist_ok=True)

    all_clips = collect_clip_ids(args.jsonl_dir)
    done = {os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(os.path.join(args.output_dir, '*.pt'))}
    remaining = sorted(set(all_clips) - done)

    idxs = list(range(len(remaining)))
    assigned = [remaining[i] for i in
                __import__('numpy').array_split(idxs, args.dataset_divider)[args.assigned_part]]

    print(f"Total clips: {len(all_clips)}, remaining: {len(remaining)}, assigned: {len(assigned)}")

    model, tokenizer = build_model(ckpt_path=args.ckpt, num_frames=4)
    model = model.eval().to(torch.float16).cuda()
    text_input = torch.zeros(1, 2, dtype=torch.long).cuda()
    mask = torch.zeros(1, 2, dtype=torch.long).cuda()

    loader = torch.utils.data.DataLoader(
        VideoDataset(args.video_dirs, assigned),
        batch_size=1, shuffle=False, num_workers=3,
    )

    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            t0 = time.time()
            clip = data["clip"][0]
            frames = data["frames"].squeeze(0)  # (F, 3, 224, 224)
            # pad to multiple of 4
            rem = frames.shape[0] % 4
            if rem:
                frames = torch.cat([frames, torch.zeros(4 - rem, 3, 224, 224)], dim=0)
            chunks = torch.split(frames, 4, dim=0)
            feats = []
            for j in range(0, len(chunks), BATCH_SIZE):
                batch = torch.stack(chunks[j:j + BATCH_SIZE]).permute(0, 2, 1, 3, 4).cuda()
                f, _ = model(batch.to(torch.float16), text_input, mask)
                feats.append(f.cpu().float())
            clip_feats = torch.cat(feats, dim=0)
            torch.save(clip_feats, os.path.join(args.output_dir, f"{clip}.pt"))
            print(f"[{i+1}/{len(loader)}] {clip} — {time.time()-t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl_dir",   required=True, help="Root dir containing EM-QnF JSONL files")
    parser.add_argument("--video_dirs",  required=True, nargs="+", help="Directories to search for .mp4 clips")
    parser.add_argument("--ckpt",        required=True, help="EgoVideo checkpoint (ckpt_4frames.pth)")
    parser.add_argument("--output_dir",  required=True, help="Output directory for .pt feature files")
    parser.add_argument("--dataset_divider", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--assigned_part",   type=int, default=0, help="Which partition this worker handles")
    args = parser.parse_args()
    main(args)
