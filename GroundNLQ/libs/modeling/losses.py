import torch
from torch.nn import functional as F
from torch import nn
import torch.distributed

@torch.jit.script
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Taken from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
         alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = 0.25.
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


@torch.jit.script
def ctr_giou_loss_1d(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Generalized Intersection over Union Loss (Hamid Rezatofighi et. al)
    https://arxiv.org/abs/1902.09630

    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0

    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # giou is reduced to iou in our setting, skip unnecessary steps
    loss = 1.0 - iouk

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

@torch.jit.script
def ctr_diou_loss_1d(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Distance-IoU Loss (Zheng et. al)
    https://arxiv.org/abs/1911.08287

    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0

    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # smallest enclosing box
    lc = torch.max(lp, lg)
    rc = torch.max(rp, rg)
    len_c = lc + rc

    # offset between centers
    rho = 0.5 * (rp - lp - rg + lg)

    # diou
    loss = 1.0 - iouk + torch.square(rho / len_c.clamp(min=eps))

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def pred_gt_contrastive_loss(pred, gt, gt_mask, duplicate_mask, temperature=1.0, across_batch=False, across_pred=False, use_duplicate_mask=False, subsample=1):
    # pred: [B, C, T]
    # gt: [B, C, T]
    # gt_mask: [B, 1, T]
    # duplicate_mask: [B, T, T]

    device = pred.device

    B, C, T = gt.shape
    _, _, pred_T = pred.shape

    pred = pred.permute(0, 2, 1) # [B, T, C]
    gt = gt.permute(0, 2, 1) # [B, T, C]

    gt = F.normalize(gt, dim=-1)
    pred = F.normalize(pred,dim=-1)

    if not use_duplicate_mask:
        duplicate_mask.fill_(1)

    gt_order = (T-1) //2
    pred_order = (pred_T-1) // 2 

    pred_indexes = [0, *[i for i in range(1, pred_order+1, subsample)], *[gt_order+i for i in range(1, pred_order+1, subsample)]]
    p_indexes = [0, *[i for i in range(1, pred_order+1, subsample)], *[pred_order+i for i in range(1, pred_order+1, subsample)]]
    duplicate_pred_gt_mask = duplicate_mask[:, pred_indexes, :]
    duplicate_pred_pred_mask = duplicate_pred_gt_mask[:, :, pred_indexes]
    pred_mask = gt_mask[:,:,pred_indexes]
    pred = pred[:, p_indexes, :]

    _, pT, _ = pred.shape

    if across_batch and across_pred:
        # across_batch_duplicate_mask = []
        # for i in range(B):
        #     across_batch_duplicate_mask.append(torch.cat([torch.ones(T,i*T), duplicate_mask[i], torch.ones(T, (B-i-1)*T)], dim=1))
        # across_batch_duplicate_mask = torch.stack(across_batch_duplicate_mask)

        across_batch_pred_mask = torch.ones(B*pT,B*(pT), device=device).fill_diagonal_(0).view(B,pT,B*pT)
        # across_batch_pred_mask = across_batch_pred_mask * across_batch_duplicate_mask

        flat_gt = torch.flatten(gt, start_dim=0, end_dim=1) # [BT, C]
        flat_pred = torch.flatten(pred, start_dim=0, end_dim=1) # [BpT, C]
        flat_gt_and_pred = torch.cat([flat_gt, flat_pred], dim=0)
        flat_gt_mask = torch.flatten(gt_mask).view(1,1,-1).expand(B, pT, B*T)
        # flat_gt_mask = flat_gt_mask * across_batch_duplicate_mask

        gt_and_pred_mask = torch.cat([flat_gt_mask, across_batch_pred_mask], dim=2)
    elif across_batch and not across_pred:
        flat_gt = torch.flatten(gt, start_dim=0, end_dim=1) # [BT, C]
        flat_gt_mask = torch.flatten(gt_mask)
        flat_gt_mask = flat_gt_mask.view(1,1,-1).expand(B, pT, B*T)
        # flat_gt_mask = flat_gt_mask * across_batch_duplicate_mask
    elif not across_batch and across_pred:
        across_example_pred_mask = torch.ones(pT,pT, device=device).fill_diagonal_(0).view(1,pT,pT)
        across_example_pred_mask = across_example_pred_mask.expand(B,pT,pT) * duplicate_pred_pred_mask # [B, T, T]
        gt_and_pred = torch.cat([gt, pred], dim=1) # [B, 2*T, C]
        per_example_gt_mask = gt_mask.expand(B, pT, T) * duplicate_pred_gt_mask  # [B, T, T]
        gt_and_pred_mask = torch.cat([per_example_gt_mask, across_example_pred_mask], dim=2)
    else:
        per_example_gt_mask = gt_mask.expand(B, pT, T) * duplicate_pred_gt_mask # [B, T, T]


    if across_batch and across_pred:
        cosine_sim_matrix = torch.matmul(pred, flat_gt_and_pred.transpose(1,0))  # [2*BT, 2*BT]
        cosine_sim_matrix = cosine_sim_matrix * gt_and_pred_mask
    elif across_batch and not across_pred:
        cosine_sim_matrix = torch.matmul(pred, flat_gt.transpose(1,0))
        cosine_sim_matrix = cosine_sim_matrix * flat_gt_mask
    elif not across_batch and across_pred:
        cosine_sim_matrix = torch.matmul(pred, gt_and_pred.transpose(1, 2))  # [B, T, T]
        cosine_sim_matrix = cosine_sim_matrix * gt_and_pred_mask
    else:
        cosine_sim_matrix = torch.matmul(pred, gt.transpose(1, 2))  # [B, T, T]
        cosine_sim_matrix = cosine_sim_matrix * per_example_gt_mask

    exp_cosine_sim_matrix = torch.exp(cosine_sim_matrix / temperature)

    if across_batch and across_pred:
        exp_cosine_sim_matrix = exp_cosine_sim_matrix * gt_and_pred_mask
        d_idx = torch.arange(B*pT, device=device).view(B, pT, -1)
        pos = torch.gather(exp_cosine_sim_matrix, dim=-1, index=d_idx).squeeze()
        all_exp_sum = torch.sum(exp_cosine_sim_matrix, dim=-1)
    elif across_batch and not across_pred:
        exp_cosine_sim_matrix = exp_cosine_sim_matrix * flat_gt_mask
        d_idx = torch.arange(B*pT, device=device).view(B, pT, -1)
        pos = torch.gather(exp_cosine_sim_matrix, dim=-1, index=d_idx).squeeze()
        all_exp_sum = torch.sum(exp_cosine_sim_matrix, dim=-1)
    elif not across_batch and across_pred:
        d_idx = torch.tensor(pred_indexes, device=device).view(1,-1,1).expand(B,pT,1)
        exp_cosine_sim_matrix = exp_cosine_sim_matrix * gt_and_pred_mask
        pos = torch.gather(exp_cosine_sim_matrix, dim=-1, index=d_idx).squeeze()
        all_exp_sum = torch.sum(exp_cosine_sim_matrix, dim=-1)
    else:
        d_idx = torch.tensor(pred_indexes, device=device).view(1,-1,1).expand(B,pT,1)
        exp_cosine_sim_matrix = exp_cosine_sim_matrix * per_example_gt_mask
        pos = torch.gather(exp_cosine_sim_matrix, dim=-1, index=d_idx).squeeze()
        all_exp_sum = torch.sum(exp_cosine_sim_matrix, dim=-1)

    nce_losses = pos / (all_exp_sum + 1e-8)

    nce_losses = torch.log(nce_losses + 1e-8)

    nce_losses = nce_losses * pred_mask.squeeze(1)

    total_loss = torch.sum(nce_losses) / torch.sum(pred_mask)

    return -1 * total_loss


@torch.no_grad()
def concat_gather(tensor, concat_dim=0, return_sizes=False, dist_group=None):
    """
    Performs all_gather operation on the provided tensors with different sizes along a specified temporal dimension,
    but only from the ranks in the given dist_group.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if dist_group is None:
        dist_group = torch.distributed.group.WORLD  # Use the default group if no group is specified
    
    # Get the ranks in the custom group
    group_ranks = torch.distributed.get_world_size(group=dist_group)
    
    # Gather the sizes of the tensors' temporal dimension from each rank in the group
    local_size = torch.tensor([tensor.size(concat_dim)], dtype=torch.long, device=tensor.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(group_ranks)]
    torch.distributed.all_gather(all_sizes, local_size, group=dist_group)
    
    # Find the maximum size along the temporal dimension and pad the tensor accordingly
    max_size = max([size.item() for size in all_sizes])
    pad_size = list(tensor.size())
    pad_size[concat_dim] = max_size - tensor.size(concat_dim)
    pad_tensor = torch.zeros(pad_size, dtype=tensor.dtype, device=tensor.device)
    padded_tensor = torch.cat([tensor, pad_tensor], dim=concat_dim)

    # Gather all the padded tensors from the group
    tensors_gather = [torch.zeros_like(padded_tensor) for _ in range(group_ranks)]
    torch.distributed.all_gather(tensors_gather, padded_tensor, group=dist_group)

    # Concatenate the gathered tensors along the temporal dimension and trim to their original sizes
    result = []
    for tensor, size in zip(tensors_gather, all_sizes):
        trimmed_tensor = tensor.narrow(concat_dim, 0, size.item())  # Slice back to the original temporal size
        result.append(trimmed_tensor)
    
    output = torch.cat(result, dim=concat_dim)
    
    if return_sizes:
        return output, torch.tensor([size.item() for size in all_sizes])
    
    return output


def compute_cross_entropy(p, q):
    q = F.log_softmax(q, dim=-1)
    loss = torch.sum(p * q, dim=-1)
    return - loss.mean()


def stablize_logits(logits):
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - logits_max.detach()
    return logits