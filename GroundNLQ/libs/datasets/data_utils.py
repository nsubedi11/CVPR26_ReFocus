import os
import numpy as np
import random
import torch
import lmdb
import pickle
import math
from transformers import BertTokenizer, BertModel

def pad_sequences_1d(sequences, dtype=torch.long, device=torch.device("cpu"), fixed_length=None):
    """ Pad a single-nested list or a sequence of n-d array (torch.tensor or np.ndarray)
    into a (n+1)-d array, only allow the first dim has variable lengths.
    Args:
        sequences: list(n-d tensor or list)
        dtype: np.dtype or torch.dtype
        device:
        fixed_length: pad all seq in sequences to fixed length. All seq should have a length <= fixed_length.
            return will be of shape [len(sequences), fixed_length, ...]
    Returns:
        padded_seqs: ((n+1)-d tensor) padded with zeros
        mask: (2d tensor) of the same shape as the first two dims of padded_seqs,
              1 indicate valid, 0 otherwise
    Examples:
        >>> test_data_list = [[1,2,3], [1,2], [3,motion_window_80,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=torch.long)
        >>> test_data_3d = [torch.randn(2,3,motion_window_80), torch.randn(motion_window_80,3,motion_window_80), torch.randn(1,3,motion_window_80)]
        >>> pad_sequences_1d(test_data_3d, dtype=torch.float)
        >>> test_data_list = [[1,2,3], [1,2], [3,motion_window_80,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=np.float32)
        >>> test_data_3d = [np.random.randn(2,3,motion_window_80), np.random.randn(motion_window_80,3,motion_window_80), np.random.randn(1,3,motion_window_80)]
        >>> pad_sequences_1d(test_data_3d, dtype=np.float32)
    """
    if isinstance(sequences[0], list):
        if "torch" in str(dtype):
            sequences = [torch.tensor(s, dtype=dtype, device=device) for s in sequences]
        else:
            sequences = [np.asarray(s, dtype=dtype) for s in sequences]

    extra_dims = sequences[0].shape[1:]  # the extra dims should be the same for all elements
    lengths = [len(seq) for seq in sequences]
    if fixed_length is not None:
        max_length = fixed_length
    else:
        max_length = max(lengths)
    if isinstance(sequences[0], torch.Tensor):
        assert "torch" in str(dtype), "dtype and input type does not match"
        padded_seqs = torch.zeros((len(sequences), max_length) + extra_dims, dtype=dtype, device=device)
        mask = torch.zeros((len(sequences), max_length), dtype=torch.float32, device=device)
    else:  # np
        assert "numpy" in str(dtype), "dtype and input type does not match"
        padded_seqs = np.zeros((len(sequences), max_length) + extra_dims, dtype=dtype)
        mask = np.zeros((len(sequences), max_length), dtype=np.float32)

    for idx, seq in enumerate(sequences):
        end = lengths[idx]
        padded_seqs[idx, :end] = seq
        mask[idx, :end] = 1
    return padded_seqs, mask  # , lengths


def filter_query_text_feats(query_text_feats, max_len):
    new_query_text_feats = []
    query_starts = []
    current_start = 0
    for query_text_feat in query_text_feats:
        if current_start >= max_len:
            break
        query_starts.append(current_start)
        query_text_feat = query_text_feat[:max_len - current_start]  
        current_start += len(query_text_feat)
        new_query_text_feats.append(query_text_feat)
    return new_query_text_feats, query_starts


def pad_1d_seq(sequences, max_length=None):
    seq_lens = []
    if len(sequences) == 0:
        return torch.tensor([]), torch.tensor([])
    if max_length is None:
        max_length = max([feat.shape[0] for feat in sequences])
    sequence_padded = []
    for seq in sequences:
        add_length = max_length - seq.shape[0]
        seq_lens.append(seq.shape[0])
        if add_length > 0:
            #np.zeros(shape=[add_length], dtype=np.float32)
            add_feature = torch.zeros(add_length, dtype=torch.float32)
            #np.concatenate([seq, add_feature], axis=0)
            seq_ = torch.cat([seq, add_feature], dim=0)
        else:
            seq_ = seq
        sequence_padded.append(seq_)
    sequence_padded = torch.stack(sequence_padded, dim=0)
    seq_lens = torch.tensor(seq_lens, dtype=torch.long)
    return sequence_padded, seq_lens

def pad_seq_with_mask(sequences, max_length=None, should_filter=False):
    sequence_padded, sequence_length, idx = [], [], []
    if max_length is None:
        max_length = max([feat.shape[0] for feat in sequences])
    feature_length = 0
    for i in range(len(sequences)):
        if sequences[i].ndim == 2:
            feature_length = sequences[i].shape[1]
            idx.append(i)
        elif sequences[i].ndim != 2 and not should_filter:
            raise ValueError("The input sequence is not a 2D array.")
                
    if feature_length == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
    
    if should_filter:
        sequences = [sequences[i] for i in idx]
    
    for seq in sequences:
        add_length = max_length - seq.shape[0]
        sequence_length.append(seq.shape[0])
        if add_length > 0:
            add_feature = torch.zeros(add_length, feature_length, dtype=torch.float32)
            seq_ = torch.cat([seq, add_feature], dim=0)
        else:
            seq_ = seq
        sequence_padded.append(seq_)
    mask = torch.zeros(len(sequences), max_length, dtype=torch.bool)
    for i, l in enumerate(sequence_length):
        mask[i, :l] = True
    sequence_padded = torch.stack(sequence_padded, dim=0)
    sequence_length = torch.tensor(sequence_length, dtype=torch.long)
    idx = torch.tensor(idx, dtype=torch.long)
    return sequence_padded, mask, sequence_length, idx

def trivial_batch_collator(batch):
    """
        A batch collator that does nothing
    """
    return batch


def train_collate_fn(batch):
    """
        Collate function for training data
    """
    video_feats = torch.stack([item['video_feats'] for item in batch], dim=0)
    v_mask = torch.stack([item['v_mask'] for item in batch], dim=0)
    query_text_feats = [item['query_feats'] for item in batch]
    feedback_feats = [item['feedback_feats'] for item in batch]
    saliency_labels = [item['saliency_labels'] for item in batch]
    contains_labels = [item['contains_labels'] for item in batch]
    not_contains_labels = [item['not_contains_labels'] for item in batch]
    temporal_labels = [item['temporal_labels'] for item in batch]
    contains_scores = [item['contains_scores'] for item in batch]
    not_contains_scores = [item['not_contains_scores'] for item in batch]
    temporal_scores = [item['temporal_scores'] for item in batch]
    contains_cutoff = [item['contains_cutoff'] for item in batch]
    not_contains_cutoff = [item['not_contains_cutoff'] for item in batch]

    span_labels = [item['span_labels'] for item in batch]
    # masked_feats = [item['masked_feats'] for item in batch]
    # query_starts = [item['query_starts'] for item in batch]
    # sampled_rel_pos = [item['sampled_rel_pos'] for item in batch]
    # masked_rel_pos = [item['masked_rel_pos'] for item in batch]

    video_feats = video_feats.permute(0, 2, 1) # (batch_size, v_dim, v_seq_len)
    video_feats = video_feats.contiguous()
    v_mask = v_mask.bool()
    v_mask = v_mask.unsqueeze(1) # (batch_size, 1, v_seq_len)

    # query_text_feats, q_mask, _, _ = pad_seq_with_mask(query_text_feats)
    # query_text_feats = query_text_feats.permute(0, 2, 1) # (batch_size, q_dim, q_seq_len)
    # query_text_feats = query_text_feats.contiguous()
    # q_mask = q_mask.bool()
    # q_mask = q_mask.unsqueeze(1) # (batch_size, 1, q_seq_len)
    
    # masked_feats, masked_mask, _, batch_idx = pad_seq_with_mask(masked_feats, filter=True)
    # masked_mask = masked_mask.bool()
    # masked_mask = masked_mask.unsqueeze(1) # (batch_size, 1, masked_seq_len)
    # batch_idx = batch_idx.long()

    # query_starts = [query_starts[i] for i in batch_idx]

    # sampled_rel_pos = [sampled_rel_pos[i] for i in batch_idx]
    # sampled_rel_pos, _ = pad_1d_seq(sampled_rel_pos)
    # sampled_rel_pos = sampled_rel_pos.long()

    # masked_rel_pos = [masked_rel_pos[i] for i in batch_idx]
    # masked_rel_pos, _ = pad_1d_seq(masked_rel_pos)
    # masked_rel_pos = masked_rel_pos.long()

    is_negative = torch.tensor([item['is_negative'] for item in batch], dtype=torch.float32)
    pred_idx = [item['pred_idx'] for item in batch]
    pred_timestamp = [item['pred_timestamp'] for item in batch]
    vid_lens = torch.tensor([item['vid_len'] for item in batch], dtype=torch.long)
    
    video_id = [item['video_id'] for item in batch]
    fps = [item['fps'] for item in batch]
   
    duration = [item['duration'] for item in batch]
    true_duration = [item['true_duration'] for item in batch]
    feat_stride = [item['feat_stride'] for item in batch]
    feat_num_frames = [item['feat_num_frames'] for item in batch]
    segments = [item['segments'] for item in batch]
    negative_segments = [item['negative_segments'] for item in batch]
    one_hot_labels = [item['one_hot_labels'] for item in batch]
    query_id = [item['query_id'] for item in batch]
    expansion_ratio = [item['expansion_ratio'] for item in batch]
    offset = [item['offset'] for item in batch]

    after_idxs = [i for i, item in enumerate(batch) if item["after_segment"] is not None]
    after_segments = [batch[i]["after_segment"] for i in after_idxs]
    after_one_hot_labels = [batch[i]["after_labels_one_hot"] for i in after_idxs]
    after_idxs = torch.tensor(after_idxs, dtype=torch.long)
    if len(after_idxs) == 0:
        after_idxs = after_segments = after_one_hot_labels = None

    before_idxs = [i for i, item in enumerate(batch) if item["before_segment"] is not None]
    before_segments = [batch[i]["before_segment"] for i in before_idxs]
    before_one_hot_labels = [batch[i]["before_labels_one_hot"] for i in before_idxs]
    before_idxs = torch.tensor(before_idxs, dtype=torch.long)
    if len(before_idxs) == 0:
        before_idxs = before_segments = before_one_hot_labels = None

    # context_action_gt = torch.stack([item['context_action_gt'] for item in batch])  # (B, C, 2*context_action_order+1)
    # context_action_gt_mask = torch.stack([item['context_action_gt_mask'] for item in batch]) # (B, 1, 2*context_action_order+1)
    # context_duplicate_mask = torch.stack([item['context_duplicate_mask'] for item in batch]) # (B, 1, 2*context_action_order+1)
    

    return {
        'video_id' : video_id,
        'video_feats': video_feats,
        'v_mask': v_mask,
        'query_text_feats': query_text_feats,
        'feedback_feats': feedback_feats,
        # 'q_mask': q_mask,
        'segments' : segments,
        'one_hot_labels' : one_hot_labels,
        'vid_lens' : vid_lens,
        'fps' : fps,
        'duration' : duration,
        'true_duration' : true_duration,
        'feat_stride' : feat_stride,
        'feat_num_frames': feat_num_frames,
        'query_id': query_id,
        'pred_idx': pred_idx,
        'is_negative': is_negative,
        'expansion_ratio': expansion_ratio,
        'after_segments': after_segments,
        'after_idxs': after_idxs,
        'after_one_hot_labels': after_one_hot_labels,
        'before_segments': before_segments,
        'before_idxs': before_idxs,
        'before_one_hot_labels': before_one_hot_labels,
        'negative_segments': negative_segments,
        'pred_timestamp': pred_timestamp,
        'saliency_labels': saliency_labels,
        'contains_labels': contains_labels,
        'not_contains_labels': not_contains_labels,
        'temporal_labels': temporal_labels,
        'contains_scores': contains_scores,
        'not_contains_scores': not_contains_scores,
        'temporal_scores': temporal_scores,
        'span_labels': span_labels,
        'contains_cutoff': contains_cutoff,
        'not_contains_cutoff': not_contains_cutoff,
        # 'context_action_gt': context_action_gt,
        # 'context_action_gt_mask': context_action_gt_mask,
        # 'context_duplicate_mask': context_duplicate_mask,
        'offset': offset
    }

def qcaa_collate_fn(batch):
    """
        Collate function for training data
    """
    video_feats = torch.stack([item['video_feats'] for item in batch], dim=0)
    v_mask = torch.stack([item['v_mask'] for item in batch], dim=0)
    query_text_feats = [item['query_feats'] for item in batch]

    video_feats = video_feats.permute(0, 2, 1) # (batch_size, v_dim, v_seq_len)
    video_feats = video_feats.contiguous()
    v_mask = v_mask.bool()
    v_mask = v_mask.unsqueeze(1) # (batch_size, 1, v_seq_len)

    query_text_feats, q_mask, _, _ = pad_seq_with_mask(query_text_feats)
    query_text_feats = query_text_feats.permute(0, 2, 1) # (batch_size, q_dim, q_seq_len)
    query_text_feats = query_text_feats.contiguous()
    q_mask = q_mask.bool()
    q_mask = q_mask.unsqueeze(1) # (batch_size, 1, q_seq_len)

    video_id = [item['video_id'] for item in batch]
    query_id = [item['query_id'] for item in batch]

    vid_lens = [item['vid_len'] for item in batch]
    vid_lens_tensor = torch.tensor([item['vid_len'] for item in batch], dtype=torch.long)

    context_action_spans = [item['context_action_spans'] for item in batch]
    context_action_gt_indices, context_action_gt_indices_mask = create_indices(context_action_spans, vid_lens)
    context_duplicate_mask = torch.stack([item['context_duplicate_mask'] for item in batch]) # (B, 1, 2*context_action_order+1)
    context_narration_gt = torch.stack([item['context_narration_gt'] for item in batch]) # (B, C, 2*context_narration_order+1)
    context_narration_gt_mask = torch.stack([item['context_narration_gt_mask'] for item in batch]) # (B, 1, 2*context_narration_order+1)

    return {
        'video_feats': video_feats,
        'v_mask': v_mask,
        'query_text_feats': query_text_feats,
        'q_mask': q_mask,
        'video_id': video_id,
        'query_id': query_id,
        'vid_lens': vid_lens_tensor,
        'context_action_gt_indices': context_action_gt_indices,
        'context_action_gt_indices_mask': context_action_gt_indices_mask,
        'context_narration_gt': context_narration_gt,
        'context_narration_gt_mask': context_narration_gt_mask,
        'context_duplicate_mask': context_duplicate_mask
    }

def test_collate_fn(batch):
    """
        Collate function for testing data
    """
    video_feats = torch.stack([item['video_feats'] for item in batch], dim=0)
    v_mask = torch.stack([item['v_mask'] for item in batch], dim=0)
    query_text_feats = [item['query_feats'] for item in batch]
    feedback_feats = [item['feedback_feats'] for item in batch]
    saliency_labels = [item['saliency_labels'] for item in batch]
    contains_labels = [item['contains_labels'] for item in batch]
    not_contains_labels = [item['not_contains_labels'] for item in batch]
    temporal_labels = [item['temporal_labels'] for item in batch]
    contains_scores = [item['contains_scores'] for item in batch]
    not_contains_scores = [item['not_contains_scores'] for item in batch]
    temporal_scores = [item['temporal_scores'] for item in batch]
    contains_cutoff = [item['contains_cutoff'] for item in batch]
    not_contains_cutoff = [item['not_contains_cutoff'] for item in batch]

    video_feats = video_feats.permute(0, 2, 1)  # (B, C, T)
    video_feats = video_feats.contiguous()
    v_mask = v_mask.bool()
    v_mask = v_mask.unsqueeze(1)  # (B, 1, T)

    pred_timestamp = [item['pred_timestamp'] for item in batch]
    pred_idx = [item['pred_idx'] for item in batch]
    is_negative = torch.tensor([item['is_negative'] for item in batch], dtype=torch.float32)
    vid_lens_tensor = torch.tensor([item['vid_len'] for item in batch], dtype=torch.long)

    video_id = [item['video_id'] for item in batch]
    fps = [item['fps'] for item in batch]
    duration = [item['duration'] for item in batch]
    feat_stride = [item['feat_stride'] for item in batch]
    feat_num_frames = [item['feat_num_frames'] for item in batch]
    segments = [item['segments'] for item in batch]
    one_hot_labels = [item['one_hot_labels'] for item in batch]
    query_id = [item['query_id'] for item in batch]
    expansion_ratio = [item['expansion_ratio'] for item in batch]
    offset = [item['offset'] for item in batch]

    return {
        'video_id': video_id,
        'video_feats': video_feats,
        'v_mask': v_mask,
        'query_text_feats': query_text_feats,
        'feedback_feats': feedback_feats,
        'segments': segments,
        'one_hot_labels': one_hot_labels,
        'vid_lens': vid_lens_tensor,
        'fps': fps,
        'duration': duration,
        'feat_stride': feat_stride,
        'feat_num_frames': feat_num_frames,
        'query_id': query_id,
        'pred_idx': pred_idx,
        'pred_timestamp': pred_timestamp,
        'is_negative': is_negative,
        'expansion_ratio': expansion_ratio,
        'offset': offset,
        'saliency_labels': saliency_labels,
        'contains_labels': contains_labels,
        'not_contains_labels': not_contains_labels,
        'temporal_labels': temporal_labels,
        'contains_scores': contains_scores,
        'not_contains_scores': not_contains_scores,
        'temporal_scores': temporal_scores,
        'contains_cutoff': contains_cutoff,
        'not_contains_cutoff': not_contains_cutoff,
    }

def worker_init_reset_seed(worker_id):
    """
        Reset random seed for each worker
    """
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def save_dataset_lmdb(dataset_dict, save_path, task_types=["nlq","narr","mq","goalstep"]):
    """
        Save dataset to lmdb
    """
    return_dict = {}
    for dataset in dataset_dict:
        lmdb_path = os.path.join(save_path, f"{dataset}.lmdb")
        data = dataset_dict[dataset]
        env = lmdb.open(lmdb_path, map_size=1099511627776)
        with env.begin(write=True) as txn:
            for i, item in enumerate(data):
                txn.put(str(i).encode(), pickle.dumps(item))
            for task_type in task_types:
                idxs = [i for i, item in enumerate(data) if task_type == item['task_type']]
                txn.put(f"{task_type}_idxs".encode(), pickle.dumps(idxs))
            txn.put("len".encode(), pickle.dumps(len(data)))
        env.close()
        return_dict[dataset] = lmdb_path

    return return_dict

def create_indices(windows, vid_lens):
    max_len = 0
    for wins in windows:
        for win in wins:
            if win is not None:
                win_l, win_r = win.tolist()
                win_l, win_r = math.floor(win_l), math.ceil(win_r)
                max_len = max(max_len, win_r-win_l+1)
    indices = []
    masks= []
    for i, wins in enumerate(windows):
        idxs = []
        msk = []
        for win in wins:
            if win is None:
                idxs.append(torch.zeros(max_len))
                msk.append(torch.zeros(max_len))
                continue
            win_l, win_r = win.tolist()
            win_l, win_r = math.floor(win_l), math.ceil(win_r)
            if win_l >= vid_lens[i]:
                idxs.append(torch.zeros(max_len))
                msk.append(torch.zeros(max_len))
                continue
            idxs.append(torch.cat([torch.arange(win_l, win_r+1), torch.zeros(max_len-(win_r-win_l+1))]))
            msk.append(torch.cat([torch.ones(win_r-win_l+1), torch.zeros(max_len-(win_r-win_l+1))]))
        indices.append(torch.stack(idxs))
        masks.append(torch.stack(msk))
    indices = torch.stack(indices)
    masks = torch.stack(masks)
    indices = indices.to(torch.long)
    return indices, masks

def get_token_embeddings():
    """
    return token embeddings for "first", "last", "all", "before", "after" tokens as dict
    """

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    words = ["first", "last", "all", "before", "after"]
    inputs = tokenizer(words, return_tensors='pt')
    outputs = model(**inputs)

    embeddings = outputs.last_hidden_state
    
    token_embeddings = {}
    for i, word in enumerate(words):
        token_embeddings[word] = embeddings[i].detach()
        
    return token_embeddings

def get_labels_emb(label_map):
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    labels_emb = {}
    for label, label_text in label_map.items():
        inputs = tokenizer(label_text, return_tensors='pt')
        outputs = model(**inputs)
        labels_emb[label] = outputs.last_hidden_state[0].detach()

    return labels_emb

def tIoU(t1, t2):
    if t1[0] >= t2[1] or t1[1] <= t2[0]:
        return 0.0
    intersection = min(t1[1], t2[1]) - max(t1[0], t2[0])
    union = max(t1[1], t2[1]) - min(t1[0], t2[0])
    return intersection / union

class ClusterFind:
    def __init__(self, clusters):
        self.total_clusters = [cluster for cluster_pass in clusters for cluster in cluster_pass]
        self.ele_to_idx = {}
        for i, cluster in enumerate(self.total_clusters):
            for ele in cluster:
                self.ele_to_idx[ele] = i
        
    def same_cluster(self, ele1, ele2):
        if ele1 not in self.ele_to_idx or ele2 not in self.ele_to_idx:
            return False
        return self.ele_to_idx[ele1] == self.ele_to_idx[ele2]
    
    def cluster_size(self, ele):
        if ele not in self.ele_to_idx:
            raise ValueError(f"{ele} not in clusters")
        return len(self.total_clusters[self.ele_to_idx[ele]])