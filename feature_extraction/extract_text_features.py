"""
Extract text features for EM-QnF queries and feedback using gte-Qwen2-7B-instruct.

Two feature types are produced per feedback record:

  1. NLQ query features  — keyed by the NLQ annotation UID
       Input: plain query text (no conversation template)

  2. Feedback features   — keyed by the feedback annotation UID
       Input: placeholder conversation template:
           assistant: "The following might be the moment you are looking for: <START>s-<END>s.
                       Does this answer your query?"
           user:      "Feedback: <feedback_text>"
       The placeholder <START>/<END> tokens mark where the model's predicted
       timestamps would appear; actual values are injected at inference time
       by the FALM module, so the text encoder sees a fixed template.

Outputs one .pkl file per annotation UID: {output_dir}/{annotation_uid}.pkl
    {"features": np.float32 (L, D),
     "pred_idx": [start_tok, end_tok],   # feedback only
     "role_idx":  [int, ...]}            # <|im_end|> positions

Usage:
    python extract_text_features.py \
        --jsonl_dir   path/to/data \
        --output_dir  path/to/text_feat_pkl \
        --model_cache path/to/hf_cache \
        [--dataset_divider 4 --assigned_part 0]
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "Alibaba-NLP/gte-Qwen2-7B-instruct"
MAX_LENGTH = 1024
TARGET_TOKEN_ID = 151645   # <|im_end|>


def load_model(cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def encode(tokenizer, model, text):
    tok = tokenizer([text], max_length=MAX_LENGTH, truncation=True, return_tensors="pt")
    tok = {k: v.to(model.device) for k, v in tok.items()}
    with torch.no_grad():
        out = model(**tok)
    feats = out.last_hidden_state.float().squeeze().cpu().numpy().astype(np.float32)
    ids = tok["input_ids"][0]
    role_idx = [i for i, t in enumerate(ids) if t.item() == TARGET_TOKEN_ID]
    return feats, role_idx


def find_placeholder_span(input_ids):
    """Return [first_idx_of_<, last_idx_of_>] marking the <START>/<END> span."""
    first = last = None
    for i in range(1, len(input_ids)):
        if input_ids[i].item() == 366 and first is None:
            first = i
        if input_ids[i].item() == 41329:
            last = i
    if first is None or last is None:
        raise ValueError("Placeholder span tokens not found in input_ids")
    return [first, last]


def encode_feedback_placeholder(tokenizer, model, query, feedback):
    """Encode feedback with placeholder prediction template."""
    placeholder_conv = [
        {"role": "assistant",
         "content": "The following might be the moment you are looking for: <START>s-<END>s.\n"
                    "Does this answer your query?"},
        {"role": "user", "content": f"Feedback: {feedback}"},
    ]
    text = tokenizer.apply_chat_template(
        placeholder_conv, tokenize=False, add_generation_prompt=True
    )
    # strip trailing generation prompt added by apply_chat_template
    tok = tokenizer([text[:-23]], max_length=MAX_LENGTH, truncation=True, return_tensors="pt")
    tok = {k: v.to(model.device) for k, v in tok.items()}
    with torch.no_grad():
        out = model(**tok)
    feats = out.last_hidden_state.float().squeeze().cpu().numpy().astype(np.float32)
    ids = tok["input_ids"][0]
    pred_idx = find_placeholder_span(ids)
    role_idx = [i for i, t in enumerate(ids) if t.item() == TARGET_TOKEN_ID]
    return feats, pred_idx, role_idx


def encode_nlq_query(tokenizer, model, query):
    """Encode a plain NLQ query (no conversation wrapper)."""
    conv = [{"role": "user", "content": query}]
    text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    tok = tokenizer([text[:-23]], max_length=MAX_LENGTH, truncation=True, return_tensors="pt")
    tok = {k: v.to(model.device) for k, v in tok.items()}
    with torch.no_grad():
        out = model(**tok)
    feats = out.last_hidden_state.float().squeeze().cpu().numpy().astype(np.float32)
    ids = tok["input_ids"][0]
    role_idx = [i for i, t in enumerate(ids) if t.item() == TARGET_TOKEN_ID]
    return feats, role_idx


def collect_records(jsonl_dir, divider, part):
    """Return (nlq_uid -> query, feedback_uid -> (query, feedback)) for assigned partition."""
    nlq_queries = {}    # nlq_uid -> query text
    fb_records = {}     # feedback_uid -> (query, feedback)

    all_lines = []
    for root, _, files in os.walk(jsonl_dir):
        for f in sorted(files):
            if f.endswith('.jsonl'):
                with open(os.path.join(root, f)) as fh:
                    all_lines.extend(fh.readlines())

    assigned = np.array_split(range(len(all_lines)), divider)[part]
    for i in assigned:
        r = json.loads(all_lines[i])
        qid = r["query_id"]
        parts = qid.split("__")
        nlq_uid = parts[1] if len(parts) > 1 else qid
        nlq_queries[nlq_uid] = r["query"]
        fb_records[qid] = (r["query"], r.get("feedback", ""))

    return nlq_queries, fb_records


def save_pkl(output_dir, uid, feat_dict):
    # annotation UIDs can contain '/' in rare cases — sanitize for filename
    safe_uid = uid.replace("/", "_")
    with open(os.path.join(output_dir, f"{safe_uid}.pkl"), "wb") as f:
        pickle.dump(feat_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def main(args):
    assert torch.cuda.is_available(), "GPU required"
    os.makedirs(args.output_dir, exist_ok=True)

    # resume: skip already-done UIDs
    done = {os.path.splitext(f)[0] for f in os.listdir(args.output_dir) if f.endswith(".pkl")}

    print("Loading model...")
    tokenizer, model = load_model(args.model_cache)

    print("Collecting records...")
    nlq_queries, fb_records = collect_records(args.jsonl_dir, args.dataset_divider, args.assigned_part)
    print(f"  {len(nlq_queries)} NLQ queries, {len(fb_records)} feedback records")

    print("Encoding NLQ queries...")
    for uid, query in tqdm(nlq_queries.items()):
        if not query or uid in done:
            continue
        feats, role_idx = encode_nlq_query(tokenizer, model, query)
        save_pkl(args.output_dir, uid, {"features": feats, "role_idx": role_idx})

    print("Encoding feedback (placeholder template)...")
    for uid, (query, feedback) in tqdm(fb_records.items()):
        if not feedback or uid in done:
            continue
        feats, pred_idx, role_idx = encode_feedback_placeholder(tokenizer, model, query, feedback)
        save_pkl(args.output_dir, uid, {"features": feats, "pred_idx": pred_idx, "role_idx": role_idx})

    print(f"Done → {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl_dir",   required=True, help="Root dir containing EM-QnF JSONL files")
    parser.add_argument("--output_dir",  required=True, help="Output directory for .pkl feature files")
    parser.add_argument("--model_cache", default=None,  help="HuggingFace cache dir for gte-Qwen2-7B")
    parser.add_argument("--dataset_divider", type=int, default=1)
    parser.add_argument("--assigned_part",   type=int, default=0)
    args = parser.parse_args()
    main(args)
