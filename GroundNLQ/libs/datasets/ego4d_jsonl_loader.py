"""
JSONL-based dataset loader for GroundNLQ.
Replaces ego4d_gen.py (LMDB generation) + ego4d_loader.py (LMDB loading).
Reads a pre-generated JSONL file and applies temporal jittering on the fly.

Register dataset name: "ego4d_jsonl"
Companion data generator: ego4d_jsonl_gen.py
"""
import io
import math
import os
import random

import lmdb
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from basic_utils import load_jsonl
from .data_utils import pad_seq_with_mask
from .datasets import register_dataset

FEEDBACK_TASKS = frozenset({"nlq_feedback", "goalstep_nlq_feedback", "hd_epic_nlq_feedback"})


@register_dataset("ego4d_jsonl")
class Ego4dJsonlDataset(Dataset):
    def __init__(
        self,
        is_training,
        split,
        val_jsonl_file,
        train_jsonl_file,
        video_feat_dir,
        text_feat_dir,
        feat_stride,
        num_frames,
        default_fps,
        downsample_rate,
        max_seq_len,
        max_txt_len,
        tasks,
        input_vid_dim,
        input_txt_dim,
        num_classes=1,
        enable_temporal_jittering=False,
        use_pooled_token=False,
        search_domain_dir=None,
        **kwargs,
    ):
        self.is_training = is_training
        self.feat_stride = feat_stride * downsample_rate
        self.num_frames = num_frames
        self.fps = default_fps
        self.max_seq_len = max_seq_len
        self.max_txt_len = max_txt_len
        self.num_classes = num_classes
        self.enable_temporal_jittering = enable_temporal_jittering
        self.use_pooled_token = use_pooled_token
        self.tasks = set(tasks)

        jsonl_file = train_jsonl_file if is_training else val_jsonl_file
        assert os.path.exists(jsonl_file), f"JSONL not found: {jsonl_file}"
        self.data = [x for x in load_jsonl(jsonl_file) if x["query_type"] in self.tasks]

        assert os.path.exists(video_feat_dir), f"Video feat dir not found: {video_feat_dir}"
        self.video_env = lmdb.open(
            video_feat_dir, map_size=1000**3, readonly=True,
            create=False, max_readers=256, readahead=True,
        )
        self.video_txn = self.video_env.begin(buffers=True)

        assert os.path.exists(text_feat_dir), f"Text feat dir not found: {text_feat_dir}"
        self.text_env = lmdb.open(
            text_feat_dir, map_size=1000**3, readonly=True,
            create=False, max_readers=256, readahead=True,
        )
        self.text_txn = self.text_env.begin(buffers=True)

        self.search_domain_txn = None
        if search_domain_dir is not None:
            assert os.path.exists(search_domain_dir), f"Search domain dir not found: {search_domain_dir}"
            self.sd_env = lmdb.open(
                search_domain_dir, map_size=1000**3, readonly=True,
                create=False, max_readers=256, readahead=True,
            )
            self.search_domain_txn = self.sd_env.begin(buffers=True)

        self.db_attributes = {
            "dataset_name": "Ego4d",
            "nlq_tiou_thresholds": np.array([0.01, 0.3, 0.5]),
            "nlq_topK": np.array([1, 5, 10, 50, 100]),
        }
        split_label = "train" if is_training else "val"
        print(f"Ego4dJsonlDataset [{split_label}]: {len(self.data)} samples from {jsonl_file}")

    def get_attributes(self):
        return self.db_attributes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        vid = item["video_id"]
        query_id = item["query_id"]
        query_type = item["query_type"]
        clip_span = item["clip_span"]
        duration = item["duration"]
        is_feedback = query_type in FEEDBACK_TASKS

        feats = self._get_video_feat_by_vid(vid)  # (T, C)

        feat_stride = self.feat_stride
        seg_s = math.floor(clip_span[0] / feat_stride * self.fps)
        seg_e = math.ceil(clip_span[1] / feat_stride * self.fps)
        feats = feats[seg_s:seg_e]
        assert len(feats) <= self.max_seq_len, (
            f"Features too long: {len(feats)} > {self.max_seq_len} for {vid}"
        )

        vid_len = len(feats)
        feats, feats_mask, _, _ = pad_seq_with_mask([feats], self.max_seq_len)
        feats = feats.squeeze(0)       # (max_seq_len, C)
        feats_mask = feats_mask.squeeze(0)  # (max_seq_len,)

        # Ground-truth segments
        timestamps = item.get("timestamps", [])
        if timestamps:
            ts_arr = np.array(timestamps, dtype=np.float32)  # (N, 2) seconds
            if self.is_training and self.enable_temporal_jittering:
                ts_arr = self._jitter_timestamps(ts_arr, duration)
            segments = torch.from_numpy((ts_arr * self.fps) / feat_stride)  # feature space
            labels = torch.zeros(len(segments), dtype=torch.int64)
            one_hot_labels = F.one_hot(labels, self.num_classes)
        else:
            segments = torch.tensor([])
            one_hot_labels = torch.tensor([])

        # pred_timestamp in feature space (for feedback tasks / saliency)
        pred_ts_sec = item.get("pred_timestamp", [])
        if pred_ts_sec:
            pred_ts_feat = torch.from_numpy(
                (np.array(pred_ts_sec, dtype=np.float32) * self.fps / feat_stride)
            ).long()
        else:
            pred_ts_feat = torch.tensor([], dtype=torch.long)

        # Query and feedback features
        if is_feedback:
            nlq_uid = query_id.split("__")[1]
            query_feats, _ = self._get_query_feat_by_qid(nlq_uid)
            feedback_feats, pred_idx = self._get_query_feat_by_qid(query_id)
        else:
            query_feats, pred_idx = self._get_query_feat_by_qid(query_id)
            feedback_feats = None

        query_starts = torch.tensor([0], dtype=torch.long)
        if len(query_feats) > self.max_txt_len:
            query_feats = query_feats[:self.max_txt_len]

        # Saliency labels (default all-ones; overwritten for feedback when search_domain_dir set)
        saliency_labels = torch.ones(vid_len, dtype=torch.float32)
        contains_labels = not_contains_labels = temporal_labels = None
        contains_scores = not_contains_scores = temporal_scores = None
        contains_cutoff = not_contains_cutoff = None

        if is_feedback and self.search_domain_txn is not None and len(segments) > 0:
            (
                saliency_labels, contains_labels, not_contains_labels, temporal_labels,
                contains_scores, not_contains_scores, temporal_scores,
                contains_cutoff, not_contains_cutoff,
            ) = self._get_saliency_label_by_qid(
                query_id, (seg_s, seg_e), segments.numpy(), pred_ts_feat,
            )

        return {
            "video_id": vid,
            "video_feats": feats,
            "v_mask": feats_mask,
            "vid_len": vid_len,
            "fps": self.fps,
            "duration": duration,
            "feat_stride": self.feat_stride,
            "feat_num_frames": self.num_frames,
            "offset": clip_span[0],
            "expansion_ratio": 1.0,
            "segments": segments,
            "one_hot_labels": one_hot_labels,
            "query_id": query_id,
            "query_feats": query_feats,
            "feedback_feats": feedback_feats,
            "pred_idx": pred_idx,
            "pred_timestamp": pred_ts_feat,
            "is_negative": 0,
            "saliency_labels": saliency_labels,
            "contains_labels": contains_labels,
            "not_contains_labels": not_contains_labels,
            "temporal_labels": temporal_labels,
            "contains_scores": contains_scores,
            "not_contains_scores": not_contains_scores,
            "temporal_scores": temporal_scores,
            "contains_cutoff": contains_cutoff,
            "not_contains_cutoff": not_contains_cutoff,
        }

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def _jitter_timestamps(self, timestamps, duration):
        """Scale + shift ground-truth spans randomly (1-10x, bounded by clip duration)."""
        jittered = []
        for ts in timestamps:
            d = ts[1] - ts[0]
            center = (ts[0] + ts[1]) / 2
            scale = random.randint(1, 10)
            shift = random.uniform(-1, 1) * (scale - 1) * d / 2
            new_center = center - shift
            new_s = max(0.0, new_center - scale * d / 2)
            new_e = min(float(duration), new_center + scale * d / 2)
            jittered.append([new_s, new_e])
        return np.array(jittered, dtype=np.float32)

    # ------------------------------------------------------------------
    # Feature loading
    # ------------------------------------------------------------------

    def _get_video_feat_by_vid(self, vid):
        dump = self.video_txn.get(vid.encode())
        with io.BytesIO(dump) as reader:
            img_dump = np.load(reader, allow_pickle=True)
            v_feat = img_dump["features"].astype(np.float32)
        return torch.from_numpy(v_feat)  # (T, C)

    def _get_query_feat_by_qid(self, qid):
        if "feedback_earlier" in qid or "feedback_later" in qid:
            qid = qid.split("__")[2]
        dump = self.text_txn.get(qid.encode())
        if not dump:
            print(f"[ERROR] LMDB key not found: {qid!r}")
            raise SystemExit(1)
        with io.BytesIO(dump) as reader:
            try:
                q_dump = np.load(reader, allow_pickle=True)
            except Exception as e:
                print(e)
                print(f"[ERROR] Cannot load LMDB key: {qid!r}")
                raise SystemExit(1)
            token_feats = np.asarray(q_dump["features"])
            pred_idx = q_dump.get("pred_idx", None)
        if self.use_pooled_token:
            token_feats = np.expand_dims(token_feats[-1], axis=0)
        return torch.from_numpy(token_feats), pred_idx  # (Lq, Dq)

    # ------------------------------------------------------------------
    # Saliency scoring (required for SearchDomainTransformer training)
    # ------------------------------------------------------------------

    def _gaussian_blur(self, x, k=15, sigma=3):
        half = k // 2
        t = np.arange(-half, half + 1)
        kernel = np.exp(-(t ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()
        return np.convolve(x, kernel, mode="same")

    def _get_saliency_label_by_qid(self, narr_id, clip_extent, gt_segments, pred_timestamp):
        """Load and process saliency scores from the search domain LMDB.
        Ported from ego4d_loader.py; only not_contains_scores is active (others disabled).

        Returns:
            merged_scores, contains_mask, not_contains_mask, temporal_mask,
            contains_scores, not_contains_scores, temporal_scores,
            contains_cutoff, not_contains_cutoff
        """
        clip_len = clip_extent[1] - clip_extent[0]

        # Synthetic temporal feedback items use direction-based masks
        if "feedback_earlier" in narr_id:
            scores = torch.ones(clip_len)
            if len(pred_timestamp) > 0:
                scores[pred_timestamp[0]:] = 0.0
            return scores.clone(), None, None, scores.clone(), None, None, scores, None, None
        if "feedback_later" in narr_id:
            scores = torch.ones(clip_len)
            if len(pred_timestamp) > 0:
                scores[: pred_timestamp[1]] = 0.0
            return scores.clone(), None, None, scores.clone(), None, None, scores, None, None

        dump = self.search_domain_txn.get(narr_id.encode())
        with io.BytesIO(dump) as reader:
            try:
                q_dump = np.load(reader, allow_pickle=True)
            except Exception as e:
                print(f"Cannot load saliency for {narr_id!r}: {e}")
                return (
                    torch.ones(clip_len, dtype=torch.float32),
                    None, None, None, None, None, None, None, None,
                )
            contains_scores = q_dump["contains_scores"]
            not_contains_scores = q_dump["not_contains_scores"]
            temporal_scores = q_dump["temporal_scores"]

        # Disable contains and temporal (same as ego4d_loader.py)
        contains_scores = None
        temporal_scores = None
        not_contains_scores = None if not_contains_scores.ndim == 0 else not_contains_scores

        if all(x is None for x in [contains_scores, not_contains_scores, temporal_scores]):
            print(f"All saliency scores None for {narr_id!r}, defaulting to ones")
            return (
                torch.ones(clip_len, dtype=torch.float32),
                None, None, None, None, None, None, None, None,
            )

        length = clip_len

        if not_contains_scores is not None:
            not_contains_scores = not_contains_scores[clip_extent[0]: clip_extent[1]]
            not_contains_scores = self._gaussian_blur(not_contains_scores, k=15, sigma=5)
            if not_contains_scores.max() > not_contains_scores.min():
                not_contains_scores = (
                    (not_contains_scores - not_contains_scores.min())
                    / (not_contains_scores.max() - not_contains_scores.min())
                )
            not_contains_scores = 1 - not_contains_scores
            length = len(not_contains_scores)

        assert len(gt_segments) == 1, "Only one GT span supported for saliency computation"
        st = math.floor(gt_segments[0][0])
        en = math.ceil(gt_segments[0][1])
        if en - st < 10:
            mid = (st + en) // 2
            st = max(0, mid - 5)
            en = min(length, mid + 5)

        not_contains_cutoff = None
        not_contains_mask = None
        if not_contains_scores is not None:
            std = np.std(not_contains_scores[st:en])
            mean = np.mean(not_contains_scores[st:en])
            not_contains_cutoff = mean - 3 * std
            not_contains_mask = (np.array(not_contains_scores) >= not_contains_cutoff).astype(int)

        merged = np.ones(length, dtype=np.float32)
        if not_contains_mask is not None:
            merged *= not_contains_mask

        merged_tensor = torch.from_numpy(merged)
        not_contains_mask_t = (
            torch.from_numpy(not_contains_mask.astype(np.float32))
            if not_contains_mask is not None else None
        )
        not_contains_scores_t = (
            torch.from_numpy(not_contains_scores.astype(np.float32))
            if not_contains_scores is not None else None
        )

        return (
            merged_tensor,
            None,                    # contains_mask (disabled)
            not_contains_mask_t,
            None,                    # temporal_mask (disabled)
            None,                    # contains_scores (disabled)
            not_contains_scores_t,
            None,                    # temporal_scores (disabled)
            None,                    # contains_cutoff (disabled)
            not_contains_cutoff,
        )
