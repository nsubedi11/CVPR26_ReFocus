#!/usr/bin/env bash
# Evaluate ReFocus checkpoint on:
#   - ego4d_qnf    val  (nlq_feedback)
#   - goalstep_qnf test (goalstep_nlq_feedback)
#   - hd_epic_qnf  test (hd_epic_nlq_feedback)
# Results saved to results/eval/<split>/

set -euo pipefail

source /uufs/chpc.utah.edu/common/home/u1472648/miniconda3/etc/profile.d/conda.sh
conda activate py38

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
GROUNDNLQ="$REPO_ROOT/GroundNLQ"
CKPT="$REPO_ROOT/ckpt/refocus_emqnf.t7"
CONFIG="$GROUNDNLQ/configs/refocus_emqnf.yaml"
VIDEO_FEAT="$REPO_ROOT/data/features/offline_lmdb/decord_egovideo_video_features"
TEXT_FEAT="$REPO_ROOT/data/features/offline_lmdb/new_gte_qwen2_role"
RESULTS_DIR="$REPO_ROOT/results/eval"

mkdir -p "$RESULTS_DIR"

cd "$GROUNDNLQ"

run_eval() {
    local name="$1"
    local jsonl="$2"
    local task="$3"

    echo ""
    echo "============================================================"
    echo "  $name  ($task)"
    echo "============================================================"

    local out_pkl="$RESULTS_DIR/${name}.pkl"
    local log_file="$RESULTS_DIR/${name}.log"

    python eval_jsonl.py \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --val_jsonl "$jsonl" \
        --task "$task" \
        --video_feat_dir "$VIDEO_FEAT" \
        --text_feat_dir "$TEXT_FEAT" \
        --output "$out_pkl" \
        --topk 10 \
        --batch_size 16 \
        --num_workers 4 \
        2>&1 | tee "$log_file"

    echo "  Saved predictions → $out_pkl"
    echo "  Full log          → $log_file"
}

run_eval "ego4d_qnf_val"      "$REPO_ROOT/data/ego4d_qnf/val.jsonl"       "nlq_feedback"
run_eval "goalstep_qnf_test"  "$REPO_ROOT/data/goalstep_qnf/test.jsonl"   "goalstep_nlq_feedback"
run_eval "hd_epic_qnf_test"   "$REPO_ROOT/data/hd_epic_qnf/test.jsonl"    "hd_epic_nlq_feedback"

echo ""
echo "All evals done. Results in $RESULTS_DIR"
