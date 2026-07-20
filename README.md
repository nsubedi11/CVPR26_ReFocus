# ReFocus: Interactive Episodic Memory with User Feedback

**CVPR 2026**

[Nikesh Subedi](https://nsubedi11.github.io/)<sup>1</sup>, [Loris Bazzani](https://lorisbaz.github.io/)<sup>2</sup>, [Ziad Al-Halah](https://users.cs.utah.edu/~ziad/)<sup>1</sup>

<sup>1</sup>University of Utah &nbsp;&nbsp; <sup>2</sup>University of Verona

[![Paper](https://img.shields.io/badge/arXiv-2604.24893-b31b1b)](https://arxiv.org/abs/2604.24893)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://nsubedi11.github.io/refocus/)
[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-green)](https://cvpr.thecvf.com/virtual/2026/poster/37436)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/RainbowMan1/ReFocus)

---

## Overview

We introduce **EM-QnF** (Episodic Memory with Questions and Feedback), a new task that extends natural language query-based episodic memory retrieval to an interactive setting. Rather than relying on a single one-shot prediction, users can provide feedback on the model's initial prediction or add more information, enabling iterative refinement of results.

We also propose **FALM** (Feedback ALignment Module), a plug-and-play module that enables existing temporal grounding models to incorporate user feedback effectively without architectural changes.

## Dataset

The EM-QnF dataset is hosted on HuggingFace: [RainbowMan1/ReFocus](https://huggingface.co/datasets/RainbowMan1/ReFocus).

It contains feedback annotations for three episodic memory benchmarks:

| Dataset | Train | Val | Test |
|---------|-------|-----|------|
| ego4d_qnf | ✓ | ✓ | — |
| goalstep_qnf | ✓ | ✓ | ✓ |
| hd_epic_qnf | ✓ | ✓ | ✓ |

### Prerequisites

The EM-QnF feedback annotations build on top of the original datasets. Please obtain access to:
- [Ego4D](https://ego4d-data.org/) — for Ego4D NLQ
- [Ego4D-GoalStep](https://github.com/facebookresearch/ego4d-goalstep) — for GoalStep NLQ
- [HD-EPIC](https://github.com/hd-epic) — for HD-EPIC NLQ

## Code

### Setup

```bash
cd GroundNLQ
# compile NMS extension
cd libs/utils && python setup.py install --user && cd ../..
```

### Features

Pre-extracted LMDB features (video + text) are required for inference. We provide **per-split subset LMDBs** containing only the features needed for each eval split, so you do not need to download the full feature archives.

| Split | Video features | Text features |
|-------|---------------|---------------|
| ego4d_qnf val | 447 MB (414 clips) | 7.2 GB (22,132 keys) |
| goalstep_qnf test | 91 MB (34 clips) | 4.6 GB (14,144 keys) |
| hd_epic_qnf test | 44 MB (15 clips) | 4.9 GB (15,059 keys) |

Download the subset LMDBs from Google Drive and place them as shown below:

| Split | Link |
|-------|------|
| ego4d_qnf val | [ego4d_qnf_val.zip](https://drive.google.com/file/d/1U6FE8ugr_tGOrZlli5NixZKcxCMHEib0/view?usp=sharing) |
| goalstep_qnf test | [goalstep_qnf_test.zip](https://drive.google.com/file/d/1s-hWy4LQWT7EhlgJ_HB-HivEaKaRz-Eb/view?usp=drive_link) |
| hd_epic_qnf test | [hd_epic_qnf_test.zip](https://drive.google.com/file/d/1FJzMge2WICvRBGkV9FffwGytaqTpa3he/view?usp=drive_link) |

After downloading, unzip each archive and place them as:

```
data/features/subset_lmdb/
├── ego4d_qnf_val/
│   ├── video_features/      # LMDB
│   └── text_features/       # LMDB
├── goalstep_qnf_test/
│   ├── video_features/
│   └── text_features/
└── hd_epic_qnf_test/
    ├── video_features/
    └── text_features/
ckpt/
└── refocus_emqnf.t7
```

### Evaluation

Use `run_eval.sh` from the repo root to reproduce results on all three splits in one go:

```bash
bash run_eval.sh
```

Results (predictions + logs) are written to `results/eval/`.

To evaluate a single split manually:

```bash
cd GroundNLQ

# ego4d_qnf — val
python eval_jsonl.py \
    --config configs/refocus_emqnf.yaml \
    --checkpoint ../ckpt/refocus_emqnf.t7 \
    --val_jsonl ../data/ego4d_qnf/val.jsonl \
    --task nlq_feedback \
    --video_feat_dir ../data/features/subset_lmdb/ego4d_qnf_val/video_features \
    --text_feat_dir  ../data/features/subset_lmdb/ego4d_qnf_val/text_features \
    --output ../results/eval/ego4d_qnf_val.pkl

# goalstep_qnf — test
python eval_jsonl.py \
    --config configs/refocus_emqnf.yaml \
    --checkpoint ../ckpt/refocus_emqnf.t7 \
    --val_jsonl ../data/goalstep_qnf/test.jsonl \
    --task goalstep_nlq_feedback \
    --video_feat_dir ../data/features/subset_lmdb/goalstep_qnf_test/video_features \
    --text_feat_dir  ../data/features/subset_lmdb/goalstep_qnf_test/text_features \
    --output ../results/eval/goalstep_qnf_test.pkl

# hd_epic_qnf — test
python eval_jsonl.py \
    --config configs/refocus_emqnf.yaml \
    --checkpoint ../ckpt/refocus_emqnf.t7 \
    --val_jsonl ../data/hd_epic_qnf/test.jsonl \
    --task hd_epic_nlq_feedback \
    --video_feat_dir ../data/features/subset_lmdb/hd_epic_qnf_test/video_features \
    --text_feat_dir  ../data/features/subset_lmdb/hd_epic_qnf_test/text_features \
    --output ../results/eval/hd_epic_qnf_test.pkl
```

Supported `--task` values: `nlq`, `nlq_feedback`, `goalstep_nlq`, `goalstep_nlq_feedback`, `hd_epic_nlq`, `hd_epic_nlq_feedback`.

### Results

Performance of the released checkpoint (`ckpt/refocus_emqnf.t7`) using the primary **mean Recall** metric:

| Split | R@1, IoU@0.3 | R@1, IoU@0.5 | R@3, IoU@0.3 | R@3, IoU@0.5 | R@5, IoU@0.3 | R@5, IoU@0.5 |
|-------|-------------|-------------|-------------|-------------|-------------|-------------|
| ego4d_qnf val | 33.12 | 23.70 | 51.64 | 38.82 | 59.63 | 46.10 |
| goalstep_qnf test | 26.77 | 20.32 | 46.89 | 37.36 | 56.12 | 46.07 |
| hd_epic_qnf test | 15.06 | 9.09 | 30.94 | 19.45 | 39.56 | 25.58 |

Training code will be released soon. Stay tuned!

## Citation

If you use ReFocus or the EM-QnF dataset in your research, please cite:

```bibtex
@inproceedings{subedi2026refocus,
  title     = {Interactive Episodic Memory with User Feedback},
  author    = {Subedi, Nikesh and Bazzani, Loris and Al-Halah, Ziad},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer
               Vision and Pattern Recognition (CVPR)},
  year      = {2026},
}
```

## License

The EM-QnF annotations are released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Use of the underlying video data is subject to the respective dataset licenses.
