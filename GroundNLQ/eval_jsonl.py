"""
Inference script for GroundNLQ using the new JSONL-based data loader.
Replaces eval_nlq.py for use with ego4d_jsonl_gen.py outputs.

Usage:
    python eval_jsonl.py \
        --config configs/mnaq_finetune.yaml \
        --checkpoint results/ckpt/epoch_049.pth.tar \
        --val_jsonl data/jsonl/nlq_val.jsonl \
        --gt_file ego4d_data/ego4d_nlq_v2_ori_data/nlq_val.json \
        [--output predictions.pkl] [--topk 10]
"""
import argparse
import os
import pickle
import time

import numpy as np
import torch
import torch.utils.data
import tqdm

from libs.core import load_config
from libs.datasets.data_utils import test_collate_fn
from libs.datasets.ego4d_jsonl_loader import Ego4dJsonlDataset
from libs.modeling import make_meta_arch
from libs.utils import fix_random_seed, ReferringRecall


def remap_checkpoint_keys(state_dict):
    """Remap checkpoint keys from old 'search_domain' naming to new FALM naming.

    Checkpoints trained before the rename use 'search_domain_model.*',
    'search_domain_scaler', and 'search_domain_bias'. This function maps
    those to the new names so existing checkpoints load without re-training.
    """
    remapped = {}
    for k, v in state_dict.items():
        if k == "search_domain_scaler":
            k = "falm_scaler"
        elif k == "search_domain_bias":
            k = "falm_bias"
        elif k.startswith("search_domain_model.search_domain_"):
            k = "falm.falm_" + k[len("search_domain_model.search_domain_"):]
        elif k.startswith("search_domain_model."):
            k = "falm." + k[len("search_domain_model."):]
        elif k.startswith("search_domain_"):
            # standalone FALM checkpoint
            k = "falm_" + k[len("search_domain_"):]
        remapped[k] = v
    return remapped


def build_val_loader(cfg_dataset, val_jsonl, task, batch_size=16, num_workers=8):
    ds = Ego4dJsonlDataset(
        is_training=False,
        split=["validation"],
        val_jsonl_file=val_jsonl,
        train_jsonl_file=val_jsonl,      # unused at val time
        video_feat_dir=cfg_dataset["video_feat_dir"],
        text_feat_dir=cfg_dataset["text_feat_dir"],
        feat_stride=cfg_dataset["feat_stride"],
        num_frames=cfg_dataset["num_frames"],
        default_fps=cfg_dataset["default_fps"],
        downsample_rate=cfg_dataset.get("downsample_rate", 1),
        max_seq_len=cfg_dataset["max_seq_len"],
        max_txt_len=cfg_dataset["max_txt_len"],
        tasks=[task],
        input_vid_dim=cfg_dataset["input_vid_dim"],
        input_txt_dim=cfg_dataset["input_txt_dim"],
        num_classes=cfg_dataset.get("num_classes", 1),
        enable_temporal_jittering=False,
        use_pooled_token=cfg_dataset.get("use_pooled_token", False),
        search_domain_dir=None,  # not needed at inference; LMDB scores are training-only
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=test_collate_fn,
        shuffle=False,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    return loader


@torch.no_grad()
def run_inference(loader, model, topk=10, print_freq=50):
    model.eval()
    uid_result_map = {}

    for iter_idx, batch in tqdm.tqdm(enumerate(loader), total=len(loader), desc="inference"):
        _, outputs = model(batch, get_losses=False, get_preds=True, wo_postprocess=True)

        for i, raw_out in enumerate(outputs):
            qid = batch["query_id"][i]
            if qid not in uid_result_map:
                uid_result_map[qid] = raw_out
            else:
                # merge segments across clips for the same query
                uid_result_map[qid]["segments"] = torch.cat(
                    [uid_result_map[qid]["segments"], raw_out["segments"]], dim=0
                )
                uid_result_map[qid]["scores"] = torch.cat(
                    [uid_result_map[qid]["scores"], raw_out["scores"]], dim=0
                )
                uid_result_map[qid]["labels"] = torch.cat(
                    [uid_result_map[qid]["labels"], raw_out["labels"]], dim=0
                )

    print(f"Postprocessing {len(uid_result_map)} queries...")
    qids = list(uid_result_map.keys())
    raw_outputs = [uid_result_map[qid] for qid in qids]
    processed = model.postprocessing(raw_outputs)

    results = []
    for qid, out in zip(qids, processed):
        segs = out["segments"]
        scores = out["scores"]
        labels = out["labels"]

        if len(segs) == 0:
            continue

        # sort and keep topk
        sorted_idx = torch.argsort(scores, descending=True)[:topk]
        segs = segs[sorted_idx].cpu().tolist()
        scores = scores[sorted_idx].cpu().tolist()
        labels = labels[sorted_idx].cpu().tolist()

        pred_times = [[s[0], s[1], sc] for s, sc in zip(segs, scores)]

        results.append({
            "annotation_uid": qid,
            "clip_uid": out["video_id"],
            "predicted_times": pred_times,
            "labels": labels,
        })

    return results


def main(args):
    cfg = load_config(args.config)

    if args.topk > 0:
        cfg["model"]["test_cfg"]["max_seg_num"] = args.topk
    if args.video_feat_dir:
        cfg["dataset"]["video_feat_dir"] = args.video_feat_dir
    if args.text_feat_dir:
        cfg["dataset"]["text_feat_dir"] = args.text_feat_dir

    fix_random_seed(0, include_cuda=True)

    print("Building val loader...")
    val_loader = build_val_loader(
        cfg["dataset"],
        val_jsonl=args.val_jsonl,
        task=args.task,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("Building model...")
    cfg["model"]["use_falm"] = args.use_falm
    cfg["model"]["falm_resume_path"] = None  # loaded from main checkpoint below
    model = make_meta_arch(cfg["model_name"], **cfg["model"])
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    # Support both {state_dict, epoch} dicts (pth.tar) and raw OrderedDict (t7)
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        epoch = ckpt.get("epoch", "?")
    else:
        state_dict = ckpt
        epoch = "?"
    state_dict = remap_checkpoint_keys(state_dict)
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint (epoch {epoch}) from {args.checkpoint}")
    model = model.cuda()

    print(f"Running inference on {len(val_loader.dataset)} {args.task} queries...")
    t0 = time.time()
    results = run_inference(val_loader, model, topk=args.topk if args.topk > 0 else 10)
    print(f"Inference done in {time.time()-t0:.1f}s  |  {len(results)} predictions")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "wb") as f:
            pickle.dump({"version": "1.0", "challenge": f"ego4d_{args.task}_challenge", "results": results}, f)
        print(f"Saved predictions → {args.output}")

    gt_file = args.gt_file or args.val_jsonl
    if gt_file:
        print("\nEvaluating...")
        evaluator = ReferringRecall(
            dataset_name="jsonl",
            gt_file=gt_file,
            task_type=args.task,
        )
        eval_out = evaluator.evaluate(results, verbose=True)
        if args.task == "nlq":
            recall, mIoU, score_str = eval_out
            print(score_str)
        elif args.task in ("nlq_feedback", "goalstep_nlq_feedback", "hd_epic_nlq_feedback"):
            (mIoU, score_str, mean_results, nlq_weighted, nlq_majority, nlq_max,
             pair_weighted, pair_majority, pair_max,
             difficult_results, random_results) = eval_out
            print(score_str)
        else:
            recall, mIoU, score_str = eval_out
            print(score_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config (for model params)")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--val_jsonl", required=True, help="Path to val JSONL (from ego4d_jsonl_gen.py)")
    parser.add_argument("--task", default="nlq",
                        choices=["nlq", "nlq_feedback", "goalstep_nlq", "goalstep_nlq_feedback",
                                 "hd_epic_nlq", "hd_epic_nlq_feedback", "narr"],
                        help="Task type — must match the query_type in val_jsonl")
    parser.add_argument("--gt_file", default="", help="GT annotation JSON for metric computation (optional)")
    parser.add_argument("--output", default="", help="Save prediction pickle to this path (optional)")
    parser.add_argument("--use_falm", action=argparse.BooleanOptionalAction, default=True,
                        help="Use FALM (Feedback ALignment Module). Pass --no-use_falm to disable.")
    parser.add_argument("--topk", default=10, type=int, help="Max predictions per query")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--video_feat_dir", default="", help="Override video_feat_dir from config")
    parser.add_argument("--text_feat_dir", default="", help="Override text_feat_dir from config")
    args = parser.parse_args()
    main(args)
