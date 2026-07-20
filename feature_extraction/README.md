# Feature Extraction

Two LMDB databases are required to run ReFocus inference:

| LMDB | Key | Value shape | Script |
|------|-----|-------------|--------|
| `video_features/` | clip UID | `float32 (T, 512)` | `extract_video_features.py` |
| `text_features/`  | annotation UID | `float32 (L, 3584)` | `extract_text_features.py` |

Each entry is stored as a numpy `.npz` blob. The pipeline has two steps:
**extract** (produces intermediate files) → **pack** (writes LMDB).

---

## 1 — Video features

Uses the [EgoVideo](https://github.com/OpenGVLab/EgoVideo) backbone
(`ckpt_4frames.pth`). Samples every 4th frame, processes in chunks of 4,
produces one 512-dim feature vector per chunk.

```bash
# single GPU
python extract_video_features.py \
    --jsonl_dir   data/ \
    --video_dirs  /path/to/ego4d/clips /path/to/ego4d/full_scale /path/to/hd_epic/videos \
    --ckpt        ckpt_4frames.pth \
    --output_dir  video_feat_pt/

# parallel (4 GPUs / jobs)
for i in 0 1 2 3; do
    python extract_video_features.py \
        --jsonl_dir data/ --video_dirs ... --ckpt ckpt_4frames.pth \
        --output_dir video_feat_pt/ \
        --dataset_divider 4 --assigned_part $i &
done
wait
```

Pack `.pt` files into LMDB:

```bash
python pack_to_lmdb.py video \
    --input_dir  video_feat_pt/ \
    --output_dir features/video_features/
```

---

## 2 — Text features

Uses [gte-Qwen2-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct)
(requires ~16 GB GPU memory with `bfloat16` + Flash Attention 2).

**Two feature types per feedback record:**

- **NLQ query** (keyed by NLQ uid) — plain query text encoded as a single user turn.
- **Feedback** (keyed by feedback uid) — encoded with a **placeholder** conversation template:

  ```
  assistant: "The following might be the moment you are looking for: <START>s-<END>s.
              Does this answer your query?"
  user:      "Feedback: <feedback_text>"
  ```

  The `<START>/<END>` tokens are placeholders; the FALM module injects the actual
  predicted timestamps at inference time, so the text encoder sees a fixed template
  regardless of the prediction.

```bash
# single GPU
python extract_text_features.py \
    --jsonl_dir   data/ \
    --output_dir  text_feat_pkl/ \
    --model_cache /path/to/hf_cache/

# parallel (4 jobs)
for i in 0 1 2 3; do
    python extract_text_features.py \
        --jsonl_dir data/ --output_dir text_feat_pkl/ \
        --dataset_divider 4 --assigned_part $i &
done
wait
```

Pack into LMDB (one `.pkl` per UID, resume-safe):

```bash
python pack_to_lmdb.py text \
    --input_dir  text_feat_pkl/ \
    --output_dir features/text_features/
```

---

## 3 — Point `eval_jsonl.py` at the LMDBs

Pass the paths via CLI or update `GroundNLQ/configs/refocus_emqnf.yaml`:

```bash
python GroundNLQ/eval_jsonl.py \
    --config     GroundNLQ/configs/refocus_emqnf.yaml \
    --checkpoint ckpt/refocus_emqnf.t7 \
    --val_jsonl  data/ego4d_qnf/val.jsonl \
    --task       nlq_feedback \
    --video_feat_dir features/video_features/ \
    --text_feat_dir  features/text_features/
```
