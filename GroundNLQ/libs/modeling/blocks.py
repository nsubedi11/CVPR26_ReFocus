import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from .weight_init import trunc_normal_

from torchvision.ops import StochasticDepth
from flash_attn.ops.triton.layer_norm import layer_norm_fn, RMSNorm
from flash_attn.modules.mha import MHA as FlashMHA
from flash_attn.modules.mlp import FusedMLP
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_varlen_kvpacked_func

class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""
    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


class MaskedConv1D(nn.Module):
    """
    Masked 1D convolution. Interface remains the same as Conv1d.
    Only support a sub set of 1d convs
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros'
    ):
        super().__init__()
        # element must be aligned
        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        # stride
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode)
        # zero out the bias term if it exists
        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # input length must be divisible by stride
        assert T % self.stride == 0

        # conv
        out_conv = self.conv(x)
        # compute the mask
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.to(x.dtype),
                size=T // self.stride,
                mode='nearest'
            )
        else:
            # masking out the features
            out_mask = mask.to(x.dtype)

        # masking the output, stop grad to mask
        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()
        return out_conv, out_mask


class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """

    def __init__(
            self,
            num_channels,
            eps=1e-5,
            affine=True,
            device=None,
            dtype=None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x ** 2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out


# helper functions for Transformer blocks
def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # return a tensor of size 1 C T
    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)


# attention / transformers
class MaskedMHA(nn.Module):
    """
    Multi Head Attention with mask

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
            self,
            n_embd,  # dimension of the input embedding
            n_head,  # number of heads in multi-head self-attention
            attn_pdrop=0.0,  # dropout rate for the attention map
            proj_pdrop=0.0  # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask, encoder_hidden_states=None, encoder_attention_mask=None):

        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # print("x ", x.shape)
        # print("mask ", mask.shape)
        # print("encoder_hidden_states ", encoder_hidden_states.shape)
        # print("attn_mask ", encoder_attention_mask.shape)

        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            # calculate query, key, values for all heads in batch
            # (B, nh * hs, T)
            q = self.query(x)
            k = self.key(encoder_hidden_states)
            v = self.value(encoder_hidden_states)
            attn_mask = encoder_attention_mask
        else:
            # calculate query, key, values for all heads in batch
            # (B, nh * hs, T)
            k = self.key(x)
            q = self.query(x)
            v = self.value(x)
            attn_mask = mask

        # move head forward to be the batch dim
        # (B, nh * hs, T) -> (B, nh, T, hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        # print("k ", k.shape)
        # print("q ", q.shape)
        # print("v ", v.shape)
        # self-attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q * self.scale) @ k.transpose(-2, -1)
        # print("att 1 ", att.shape)
        # prevent q from attending to invalid tokens
        att = att.masked_fill(torch.logical_not(attn_mask[:, :, None, :]), float('-inf'))
        # print("att 2 ", att.shape)
        # softmax attn
        att = F.softmax(att, dim=-1)
        # print("att 3 ", att.shape)
        att = self.attn_drop(att)
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        out = att @ (v * attn_mask[:, :, :, None].to(v.dtype))
        # print("out 1 ", out.shape)
        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * mask.to(out.dtype)
        return out, mask


class MaskedMHCA(nn.Module):
    """
    Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
            self,
            n_embd,  # dimension of the output features
            n_head,  # number of heads in multi-head self-attention
            n_qx_stride=1,  # dowsampling stride for query and input
            n_kv_stride=1,  # downsampling stride for key and value
            attn_pdrop=0.0,  # dropout rate for the attention map
            proj_pdrop=0.0,  # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        # 1d depthwise conv
        self.query_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        # layernorm
        self.query_norm = LayerNorm(self.n_embd)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        # 1d depthwise conv
        self.key_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        # layernorm
        self.value_norm = LayerNorm(self.n_embd)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # print("self-attention")
        # print("x ", x.shape)
        # print("mask ", mask.shape)

        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q)
        # key, value conv -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v)

        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # print("k ", k.shape)
        # print("q ", q.shape)
        # print("v ", v.shape)

        # move head forward to be the batch dim
        # (B, nh * hs, T'/T'') -> (B, nh, T'/T'', hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # self-attention: (B, nh, T', hs) x (B, nh, hs, T'') -> (B, nh, T', T'')
        att = (q * self.scale) @ k.transpose(-2, -1)
        # print("att 1 ", att.shape)
        # prevent q from attending to invalid tokens
        att = att.masked_fill(torch.logical_not(kv_mask[:, :, None, :]), float('-inf'))
        # print("att 2 ", att.shape)
        # softmax attn
        att = F.softmax(att, dim=-1)
        # print("att 3 ", att.shape)
        att = self.attn_drop(att)
        # (B, nh, T', T'') x (B, nh, T'', hs) -> (B, nh, T', hs)
        out = att @ (v * kv_mask[:, :, :, None].to(v.dtype))
        # print("out 1 ", out.shape)
        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)
        return out, qx_mask


class LocalMaskedMHCA(nn.Module):
    """
    Local Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    The implementation is fairly tricky, code reference from
    https://github.com/huggingface/transformers/blob/master/src/transformers/models/longformer/modeling_longformer.py
    """

    def __init__(
            self,
            n_embd,  # dimension of the output features
            n_head,  # number of heads in multi-head self-attention
            window_size,  # size of the local attention window
            n_qx_stride=1,  # dowsampling stride for query and input
            n_kv_stride=1,  # downsampling stride for key and value
            attn_pdrop=0.0,  # dropout rate for the attention map
            proj_pdrop=0.0,  # dropout rate for projection op
            use_rel_pe=False  # use relative position encoding
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)
        self.window_size = window_size
        self.window_overlap = window_size // 2
        # must use an odd window size
        assert self.window_size > 1 and self.n_head >= 1
        self.use_rel_pe = use_rel_pe

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        # 1d depthwise conv
        self.query_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        # layernorm
        self.query_norm = LayerNorm(self.n_embd)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        # 1d depthwise conv
        self.key_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        # layernorm
        self.value_norm = LayerNorm(self.n_embd)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # relative position encoding
        if self.use_rel_pe:
            self.rel_pe = nn.Parameter(
                torch.zeros(1, 1, self.n_head, self.window_size))
            trunc_normal_(self.rel_pe, std=(2.0 / self.n_embd) ** 0.5)

    @staticmethod
    def _chunk(x, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
        # x: B x nh, T, hs
        # non-overlapping chunks of size = 2w -> B x nh, T//2w, 2w, hs
        x = x.view(
            x.size(0),
            x.size(1) // (window_overlap * 2),
            window_overlap * 2,
            x.size(2),
        )

        # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1
        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2

        # B x nh, #chunks = T//w - 1, 2w, hs
        return x.as_strided(size=chunk_size, stride=chunk_stride)

    @staticmethod
    def _pad_and_transpose_last_two_dims(x, padding):
        """pads rows and then flips rows and columns"""
        # padding value is not important because it will be overwritten
        x = nn.functional.pad(x, padding)
        x = x.view(*x.size()[:-2], x.size(-1), x.size(-2))
        return x

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len):
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        # `== 1` converts to bool or uint8
        beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):]
        ending_mask = ending_mask.expand(ending_input.size())
        # `== 1` converts to bool or uint8
        ending_input.masked_fill_(ending_mask == 1, -float("inf"))

    @staticmethod
    def _pad_and_diagonalize(x):
        """
        shift every row 1 step right, converting columns into diagonals.
        Example::
              chunked_hidden_states: [ 0.4983,  2.6918, -0.0071,  1.0492,
                                       -1.8348,  0.7672,  0.2986,  0.0285,
                                       -0.7584,  0.4206, -0.0405,  0.1599,
                                       2.0514, -1.1600,  0.5372,  0.2629 ]
              window_overlap = num_rows = 4
             (pad & diagonalize) =>
             [ 0.4983,  2.6918, -0.0071,  1.0492, 0.0000,  0.0000,  0.0000
               0.0000,  -1.8348,  0.7672,  0.2986,  0.0285, 0.0000,  0.0000
               0.0000,  0.0000, -0.7584,  0.4206, -0.0405,  0.1599, 0.0000
               0.0000,  0.0000,  0.0000, 2.0514, -1.1600,  0.5372,  0.2629 ]
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = x.size()
        # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1).
        x = nn.functional.pad(
            x, (0, window_overlap + 1)
        )
        # total_num_heads x num_chunks x window_overlap*window_overlap+window_overlap
        x = x.view(total_num_heads, num_chunks, -1)
        # total_num_heads x num_chunks x window_overlap*window_overlap
        x = x[:, :, :-window_overlap]
        x = x.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )
        x = x[:, :, :, :-1]
        return x

    def _sliding_chunks_query_key_matmul(
            self, query, key, num_heads, window_overlap
    ):
        """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This implementation splits the input into overlapping chunks of size 2w with an overlap of size w (window_overlap)
        """
        # query / key: B*nh, T, hs
        bnh, seq_len, head_dim = query.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert query.size() == key.size()

        chunks_count = seq_len // window_overlap - 1

        # B * num_heads, head_dim, #chunks=(T//w - 1), 2w
        chunk_query = self._chunk(query, window_overlap)
        chunk_key = self._chunk(key, window_overlap)

        # matrix multiplication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x 2window_overlap
        diagonal_chunked_attention_scores = torch.einsum(
            "bcxd,bcyd->bcxy", (chunk_query, chunk_key))

        # convert diagonals into columns
        # B * num_heads, #chunks, 2w, 2w+1
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
        # window_overlap previous words). The following column is attention score from each word to itself, then
        # followed by window_overlap columns for the upper triangle.
        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
            (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
        )

        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
                                                                :, :, :window_overlap, : window_overlap + 1
                                                                ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
                                                               :, -1, window_overlap:, : window_overlap + 1
                                                               ]
        # - copying the lower triangle
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
                                                               :, :, -(window_overlap + 1): -1, window_overlap + 1:
                                                               ]

        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
                                                                              :, 0, : window_overlap - 1,
                                                                              1 - window_overlap:
                                                                              ]

        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(
            self, attn_probs, value, num_heads, window_overlap
    ):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
        bnh, seq_len, head_dim = value.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = seq_len // window_overlap - 1
        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap

        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads, seq_len // window_overlap, window_overlap, 2 * window_overlap + 1
        )

        # pad seq_len with w at the beginning of the sequence and another window overlap at the end
        padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)

        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # step 1: depth convolutions
        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q)
        # key, value conv -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v)

        # step 2: query, key, value transforms & reshape
        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        # (B, nh * hs, T) -> (B, nh, T, hs)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        # view as (B * nh, T, hs)
        q = q.view(B * self.n_head, -1, self.n_channels).contiguous()
        k = k.view(B * self.n_head, -1, self.n_channels).contiguous()
        v = v.view(B * self.n_head, -1, self.n_channels).contiguous()

        # step 3: compute local self-attention with rel pe and masking
        q *= self.scale
        # chunked query key attention -> B, T, nh, 2w+1 = window_size
        att = self._sliding_chunks_query_key_matmul(
            q, k, self.n_head, self.window_overlap)

        # rel pe
        if self.use_rel_pe:
            att += self.rel_pe
        # kv_mask -> B, T'', 1
        inverse_kv_mask = torch.logical_not(
            kv_mask[:, :, :, None].view(B, -1, 1))
        # 0 for valid slot, -inf for masked ones
        float_inverse_kv_mask = inverse_kv_mask.type_as(q).masked_fill(
            inverse_kv_mask, -1e4)
        # compute the diagonal mask (for each local window)
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_inverse_kv_mask.new_ones(size=float_inverse_kv_mask.size()),
            float_inverse_kv_mask,
            1,
            self.window_overlap
        )
        att += diagonal_mask

        # ignore input masking for now
        att = nn.functional.softmax(att, dim=-1)
        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        att = att.masked_fill(
            torch.logical_not(kv_mask.squeeze(1)[:, :, None, None]), 0.0)
        att = self.attn_drop(att)

        # step 4: compute attention value product + output projection
        # chunked attn value product -> B, nh, T, hs
        out = self._sliding_chunks_matmul_attn_probs_value(
            att, v, self.n_head, self.window_overlap)
        # transpose to B, nh, hs, T -> B, nh*hs, T
        out = out.transpose(2, 3).contiguous().view(B, C, -1)
        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)
        return out, qx_mask


class TransformerBlock(nn.Module):
    """
    A simple (post layer norm) Transformer block
    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
            self,
            n_embd,  # dimension of the input features
            n_head,  # number of attention heads
            n_ds_strides=(1, 1),  # downsampling strides for q & x, k & v
            n_out=None,  # output dimension, if None, set to input dim
            n_hidden=None,  # dimension of the hidden layer in MLP
            act_layer=nn.GELU,  # nonlinear activation used in MLP, default GELU
            attn_pdrop=0.0,  # dropout rate for the attention map
            proj_pdrop=0.0,  # dropout rate for the projection / MLP
            path_pdrop=0.0,  # drop path rate
            mha_win_size=-1,  # > 0 to use window mha
            use_rel_pe=False,  # if to add rel position encoding to attention
            use_cross_modal=False,  # if to add cross_modal attention
    ):
        super().__init__()
        assert len(n_ds_strides) == 2
        # layer norm for order (B C T)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

        # specify the attention module
        if mha_win_size > 1:
            self.attn = LocalMaskedMHCA(
                n_embd,
                n_head,
                window_size=mha_win_size,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                use_rel_pe=use_rel_pe  # only valid for local attention
            )
        else:
            self.attn = MaskedMHCA(
                n_embd,
                n_head,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop
            )

        self.use_cross_modal = use_cross_modal
        if use_cross_modal:
            self.cross_attn = AttnPriorMHA(
                n_embd,
                n_head,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
            )
            self.ln3 = LayerNorm(n_embd)
            self.cross_pool_skip = nn.Identity()

        # input
        if n_ds_strides[0] > 1:
            kernel_size, stride, padding = \
                n_ds_strides[0] + 1, n_ds_strides[0], (n_ds_strides[0] + 1) // 2
            self.pool_skip = nn.MaxPool1d(
                kernel_size, stride=stride, padding=padding)
        else:
            self.pool_skip = nn.Identity()

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd
        # ok to use conv1d here with stride=1
        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1),
            act_layer(),
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv1d(n_hidden, n_out, 1),
            nn.Dropout(proj_pdrop, inplace=True),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mlp = nn.Identity()

    def forward(self, x, mask, cross_y=None, cross_y_mask=None, pos_embd=None, attn_prior=None, return_attn=False):
        # pre-LN transformer: https://arxiv.org/pdf/2002.04745.pdf

        #  downsample in the multi-head local attention
        out, out_mask = self.attn(self.ln1(x), mask)

        out_mask_float = out_mask.to(out.dtype)
        out = self.pool_skip(x) * out_mask_float + self.drop_path_attn(out)

        # optional cross_modal attention
        att = None
        if self.use_cross_modal and cross_y is not None:
            # print("inside")
            cross_out, cross_out_mask, att = self.cross_attn(self.ln3(out), out_mask_float, self.ln3(cross_y), cross_y_mask, attn_prior)
            out_mask_float = out_mask.to(cross_out_mask.dtype)
            out = self.cross_pool_skip(out) * out_mask_float + self.drop_path_attn(cross_out)

        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.ln2(out)) * out_mask_float)
        # optionally add pos_embd to the output
        if pos_embd is not None:
            out += pos_embd * out_mask_float
        if return_attn:
            return out, out_mask, att
        return out, out_mask


class ConvBlock(nn.Module):
    """
    A simple conv block similar to the basic block used in ResNet
    """

    def __init__(
            self,
            n_embd,  # dimension of the input features
            kernel_size=3,  # conv kernel size
            n_ds_stride=1,  # downsampling stride for the current layer
            expansion_factor=2,  # expansion factor of feat dims
            n_out=None,  # output dimension, if None, set to input dim
            act_layer=nn.ReLU,  # nonlinear activation used after conv, default ReLU
    ):
        super().__init__()
        # must use odd sized kernel
        assert (kernel_size % 2 == 1) and (kernel_size > 1)
        padding = kernel_size // 2
        if n_out is None:
            n_out = n_embd

        # 1x3 (strided) -> 1x3 (basic block in resnet)
        width = n_embd * expansion_factor
        self.conv1 = MaskedConv1D(
            n_embd, width, kernel_size, n_ds_stride, padding=padding)
        self.conv2 = MaskedConv1D(
            width, n_out, kernel_size, 1, padding=padding)

        # attach downsampling conv op
        if n_ds_stride > 1:
            # 1x1 strided conv (same as resnet)
            self.downsample = MaskedConv1D(n_embd, n_out, 1, n_ds_stride)
        else:
            self.downsample = None

        self.act = act_layer()

    def forward(self, x, mask, pos_embd=None):
        identity = x
        out, out_mask = self.conv1(x, mask)
        out = self.act(out)
        out, out_mask = self.conv2(out, out_mask)

        # downsampling
        if self.downsample is not None:
            identity, _ = self.downsample(x, mask)

        # residual connection
        out += identity
        out = self.act(out)

        return out, out_mask


# drop path: from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """

    def __init__(self, init_value=1.0):
        """
        init_value : initial value for the scalar
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        """
        input -> scale * input
        """
        return x * self.scale


# The follow code is modified from
# https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AffineDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) with a per channel scaling factor (and zero init)
    See: https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)),
            requires_grad=True
        )
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(self.scale * x, self.drop_prob, self.training)


class CrossModalDualEncoders(nn.Module):
    def __init__(
        self,
        num_layers,
        n_embd,
        n_head,
        n_ds_strides= (1, 1),
        n_out=None,
        n_hidden=None,
        act_layer=nn.GELU,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        path_pdrop=0.0,
        mha_win_size=-1,
        use_rel_pe=False,
        modality="single",
    ):
        super(CrossModalDualEncoders, self).__init__()
        self.num_heads = n_head
        self.num_layers = num_layers
        assert modality in ["single", "cross_modal_dual"]
        self.modality = modality
    
        self.context_encoder = nn.ModuleList([TransformerBlock(
            n_embd=n_embd,
            n_head=n_head,
            n_ds_strides=n_ds_strides,
            n_out=n_out,
            n_hidden=n_hidden,
            act_layer=act_layer,
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
            mha_win_size=mha_win_size,
            use_rel_pe=use_rel_pe,
            use_cross_modal=True,
        ) for _ in range(num_layers)])
        if self.modality == "cross_modal_dual":
            self.query_encoder = nn.ModuleList([TransformerBlock(
                n_embd=n_embd,
                n_head=n_head,
                n_ds_strides=n_ds_strides,
                n_out=n_out,
                n_hidden=n_hidden,
                act_layer=act_layer,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=-1,
                use_rel_pe=use_rel_pe,
                use_cross_modal=True,
            ) for _ in range(num_layers)])


    def forward(self, context, c_mask, query, q_mask, return_intermediate_queries=False):
        if self.modality == "single":
            for i in range(self.num_layers):
                context, _ = self.context_encoder[i](context, c_mask, query, q_mask)
            return context, query
        elif self.modality == "cross_modal_dual":
            if return_intermediate_queries:
                intermediate_queries = []
                for i in range(self.num_layers):
                    temp_context = context
                    temp_query = query
                    context, _ = self.context_encoder[i](context, c_mask, temp_query, q_mask)
                    query, _ = self.query_encoder[i](query, q_mask, temp_context, c_mask)
                    intermediate_queries.append(query)
                intermediate_queries = torch.stack(intermediate_queries, dim=1) # B, N_l, C, T
                return context, query, intermediate_queries
            else:
                for i in range(self.num_layers):
                    temp_context = context
                    temp_query = query
                    context, _ = self.context_encoder[i](context, c_mask, temp_query, q_mask)
                    query, _ = self.query_encoder[i](query, q_mask, temp_context, c_mask)
                return context, query

class MultiscaleSummarizer(nn.Module):
    def __init__(self,
                 num_layers,
                 num_blocks,
                 n_embd,
                 n_head,
                 n_ds_strides= (1, 1),
                 n_out=None,
                 n_hidden=None,
                 act_layer=nn.GELU,
                 attn_pdrop=0.0,
                 proj_pdrop=0.0,
                 path_pdrop=0.0,
                 mha_win_size=-1,
                 use_rel_pe=False,
                 resolution=10000,
                 summmary_of_summmary=False,
                 max_len=2304,
                 share_model_levels=True,
                 z_value=3):
        super(MultiscaleSummarizer, self).__init__()
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.num_heads = n_head
        self.resolution = resolution
        self.max_len = max_len
        self.summmary_of_summmary = summmary_of_summmary
        self.share_model_levels = share_model_levels
        self.z_value = z_value
        if self.share_model_levels:
            self.summarizer = nn.ModuleList([TransformerBlock(
                n_embd=n_embd,
                n_head=n_head,
                n_ds_strides=n_ds_strides,
                n_out=n_out,
                n_hidden=n_hidden,
                act_layer=act_layer,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=mha_win_size,
                use_rel_pe=use_rel_pe,
                use_cross_modal=True,
            ) for _ in range(num_blocks)])
        else:
            self.summarizer = nn.ModuleList([nn.ModuleList([TransformerBlock(
                n_embd=n_embd,
                n_head=n_head,
                n_ds_strides=n_ds_strides,
                n_out=n_out,
                n_hidden=n_hidden,
                act_layer=act_layer,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=mha_win_size,
                use_rel_pe=use_rel_pe,
                use_cross_modal=True,
            ) for _ in range(num_blocks)]) for _ in range(num_layers)])

        self.summary_token = nn.ParameterList(nn.Parameter(torch.randn(1, n_embd, 1)) for i in range(num_layers))
        self.relative_pos_embed = RelativeSinusoidalPositionalEncoding(resolution, n_embd)

        # TODO: Different summary token for each layer
        # TODO: add positional embedding to x (input) that we cross attend to
        # TODO: Use same pos embeding as x (input) ie based on the vid len
        # TODO: Use learned positional embedding for summary token

    @torch.no_grad()
    def generate_gaussian_attn_weights(self, size, mean_loc, std, min_weight=1e-6):
        """
        Generates a tensor of weights distributed according to a Gaussian curve centered around a specified index.
        
        Parameters:
            size (M) (int): Max Length of the feature
            mean_idx (float): The index at which the Gaussian curve is centered: B, T
            std (float): The standard deviation of the Gaussian distribution. B, 1
            
        Returns:
            torch.Tensor: A tensor of weights following a Gaussian curve centered around mean_idx.
        """
        mean_loc = mean_loc.unsqueeze(2)
        std = std.unsqueeze(2)
        x = torch.arange(size, dtype=torch.float32).to(mean_loc.device)
        x = x.expand((mean_loc.shape[0], mean_loc.shape[1], size))
        weights = torch.exp(-0.5 * ((x - mean_loc) / std) ** 2)
        weights[weights<min_weight] = 0
        weights = weights / (weights.sum(dim=2).unsqueeze(2) + 1e-8)
        weights[weights>min_weight] = 1
        
        return weights.unsqueeze(1)

    def forward(self, x, mask, vid_lens): 
        # x: B, C, T
        # mask: B, 1, T
        # durations: B, 1

        B, C, Tx = x.size()
        dur = vid_lens.view(-1, 1)
        out_feats = []
        out_masks = []
        #priors = []
        attns = []
        
        num_summaries = self.max_len
        std = ((dur-1)/num_summaries)/self.z_value
        summary_tokens = self.summary_token[0].expand((B, C, num_summaries)) # B, C, T
        loc = torch.linspace(0.5/num_summaries, 1-(0.5/num_summaries), num_summaries).expand((B, num_summaries)) # B, T
        loc = loc.to(x.device)
        loc = loc * (dur-1) # B, T
        p = loc.type(torch.long) # B, T
        pe = self.relative_pos_embed(p) # B, C, T
        summary_tokens = summary_tokens + pe # B, C, T
        summary_mask = torch.ones(summary_tokens.size(0), 1, summary_tokens.size(2)).type(torch.bool).to(x.device) # B, 1, T
        
        attn_prior = self.generate_gaussian_attn_weights(Tx, loc, std)
        attn_prior = attn_prior.to(x.device)
        #priors.append(attn_prior)

        curr_out = summary_tokens
        curr_mask = summary_mask
        curr_attn = []
        for j in range(self.num_blocks):
            if self.share_model_levels:
                curr_out, curr_mask, attn = self.summarizer[j](curr_out, curr_mask, x, mask, attn_prior=attn_prior, return_attn=True)
            else:
                curr_out, curr_mask, attn = self.summarizer[0][j](curr_out, curr_mask, x, mask, attn_prior=attn_prior, return_attn=True)
            curr_attn.append(attn)
        attns.append(curr_attn)
        out_feats.append(curr_out)
        out_masks.append(curr_mask)
        for i in range(1, self.num_layers):
            prev_out = out_feats[-1]
            prev_mask = out_masks[-1]
            
            num_summaries = self.max_len // 2**i
            std = ((dur-1)/num_summaries)/self.z_value
            summary_tokens = self.summary_token[i].expand((B, C, num_summaries)) # B, C, T
            loc = torch.linspace(0.5/num_summaries, 1-(0.5/num_summaries), num_summaries).expand((B, num_summaries)) # B, T
            loc = loc.to(x.device)
            loc = loc * (dur-1) # B, T
            p = loc.type(torch.long) # B, T
            pe = self.relative_pos_embed(p) # B, C, T
            summary_tokens = summary_tokens + pe # B, C, T
            summary_mask = torch.ones(summary_tokens.size(0), 1, summary_tokens.size(2)).type(torch.bool).to(x.device) # B, 1, T
            
            attn_prior = self.generate_gaussian_attn_weights(Tx, loc, std)
            attn_prior = attn_prior.to(x.device)

            curr_out = summary_tokens
            curr_mask = summary_mask
            curr_attn = []
            #priors.append(attn_prior)
            if self.summmary_of_summmary:
                for j in range(self.num_blocks):
                    if self.share_model_levels:
                        curr_out, curr_mask, attn = self.summarizer[j](curr_out, curr_mask, prev_out, prev_mask, attn_prior=attn_prior, return_attn=True)
                    else:
                        curr_out, curr_mask, attn = self.summarizer[i][j](curr_out, curr_mask, prev_out, prev_mask, attn_prior=attn_prior, return_attn=True)
                    curr_attn.append(attn)
            else:
                for j in range(self.num_blocks):
                    if self.share_model_levels:
                        curr_out, curr_mask, attn = self.summarizer[j](curr_out, curr_mask, x, mask, attn_prior=attn_prior, return_attn=True)
                    else:
                        curr_out, curr_mask, attn = self.summarizer[i][j](curr_out, curr_mask, x, mask, attn_prior=attn_prior, return_attn=True)
                    curr_attn.append(attn)
                out_feats.append(curr_out)
                out_masks.append(curr_mask)
            attns.append(curr_attn)
        return out_feats, out_masks, attns

class AverageMultiscaleSummarizer(nn.Module):
    def __init__(self,
                 num_layers,
                 num_blocks,
                 n_embd,
                 n_head,
                 n_ds_strides= (1, 1),
                 n_out=None,
                 n_hidden=None,
                 act_layer=nn.GELU,
                 attn_pdrop=0.0,
                 proj_pdrop=0.0,
                 path_pdrop=0.0,
                 mha_win_size=-1,
                 use_rel_pe=False,
                 resolution=10000,
                 summmary_of_summmary=False,
                 max_len=2304,
                 share_model_levels=False):
        super(AverageMultiscaleSummarizer, self).__init__()
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.num_heads = n_head
        self.resolution = resolution
        self.max_len = max_len
        self.summmary_of_summmary = summmary_of_summmary
        self.share_model_levels = share_model_levels
        if self.share_model_levels:
            self.summarizer = nn.ModuleList([TransformerBlock(
                n_embd=n_embd,
                n_head=n_head,
                n_ds_strides=n_ds_strides,
                n_out=n_out,
                n_hidden=n_hidden,
                act_layer=act_layer,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=mha_win_size,
                use_rel_pe=use_rel_pe,
                use_cross_modal=True,
            ) for _ in range(num_blocks)])
        else:
            self.summarizer = nn.ModuleList([nn.ModuleList([TransformerBlock(
                n_embd=n_embd,
                n_head=n_head,
                n_ds_strides=n_ds_strides,
                n_out=n_out,
                n_hidden=n_hidden,
                act_layer=act_layer,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=mha_win_size,
                use_rel_pe=use_rel_pe,
                use_cross_modal=True,
            ) for _ in range(num_blocks)]) for _ in range(num_layers)])

        self.relative_pos_embed = RelativeSinusoidalPositionalEncoding(resolution, n_embd)

        self.duration_resolution = 1000
        self.duration_embedding = nn.Embedding(self.duration_resolution, n_embd)

    def average_feature(self, visual_feature, max_len):
        sp = torch.tensor_split(visual_feature, max_len, dim=0)
        new_visual_feature = torch.stack([torch.mean(s, dim=0) for s in sp])
        new_visual_feature = torch.nan_to_num(new_visual_feature)
        return new_visual_feature

    def forward(self, x, mask, vid_lens):
        # x: B, C, T
        # mask: B, 1, T
        # vid_lens: B, 1
        B, C, T = x.size()
        out_feats = []
        out_masks = []

        duration = vid_lens/self.max_len * (self.duration_resolution-1)
        duration = duration.type(torch.long)
        duration = duration.clamp(max=self.duration_resolution-1)
        duration = self.duration_embedding(duration).unsqueeze(1) # B, 1, C
        duration = duration.permute(0, 2, 1) # B, C, 1

        summary_tokens = []
        for k in range(B):
            summary_tokens.append(self.average_feature(x[k].permute(1, 0)[:vid_lens[k]], self.max_len)) # T, C
        summary_tokens = torch.stack(summary_tokens)
        summary_tokens = summary_tokens.permute(0, 2, 1) # B, C, T
        p = torch.linspace(0, 1, summary_tokens.size(2)).unsqueeze(0).repeat(summary_tokens.size(0), 1).to(summary_tokens.device)
        p = p * (self.resolution-1)
        p = p.type(torch.long)
        pe = self.relative_pos_embed(p)
        summary_tokens = summary_tokens + pe + duration
        summary_mask = torch.ones(summary_tokens.size(0), 1, summary_tokens.size(2)).type(torch.bool).to(x.device)
        curr_out = summary_tokens
        curr_mask = summary_mask
        for j in range(self.num_blocks):
            if self.share_model_levels:
                curr_out, curr_mask = self.summarizer[j](summary_tokens, summary_mask, x, mask)
            else:
                curr_out, curr_mask = self.summarizer[0][j](summary_tokens, summary_mask, x, mask)
        out_feats.append(curr_out)
        out_masks.append(curr_mask)
        for i in range(1, self.num_layers):
            curr_out = out_feats[-1]
            curr_mask = out_masks[-1]
            summary_tokens = []
            for k in range(B):
                summary_tokens.append(self.average_feature(x[k].permute(1, 0)[:vid_lens[k]], self.max_len//(2**i))) # T, C
            summary_tokens = torch.stack(summary_tokens)
            summary_tokens = summary_tokens.permute(0, 2, 1) # B, C, T
            p = torch.linspace(0, 1, summary_tokens.size(2)).unsqueeze(0).repeat(summary_tokens.size(0), 1).to(summary_tokens.device)
            p = p * (self.resolution-1)
            p = p.type(torch.long)
            pe = self.relative_pos_embed(p)
            summary_tokens = summary_tokens + pe + duration
            summary_mask = torch.ones(summary_tokens.size(0), 1, summary_tokens.size(2)).type(torch.bool).to(summary_tokens.device)
            if self.summmary_of_summmary:
                for j in range(self.num_blocks):
                    if self.share_model_levels:
                        curr_out, curr_mask = self.summarizer[j](summary_tokens, summary_mask, curr_out, curr_mask)
                    else:
                        curr_out, curr_mask = self.summarizer[i][j](summary_tokens, summary_mask, curr_out, curr_mask)
                out_feats.append(curr_out)
                out_masks.append(curr_mask)
            else:
                for j in range(self.num_blocks):
                    if self.share_model_levels:
                        curr_out, curr_mask = self.summarizer[j](summary_tokens, summary_mask, x, mask)
                    else:
                        curr_out, curr_mask = self.summarizer[i][j](summary_tokens, summary_mask, x, mask)
                out_feats.append(curr_out)
                out_masks.append(curr_mask)
        return out_feats, out_masks
                    
class RelativeSinusoidalPositionalEncoding(nn.Module):
    def __init__(self, resolution, dim):
        super(RelativeSinusoidalPositionalEncoding, self).__init__()
        self.embedding_table = nn.Embedding(resolution, dim)
        #precompute the positional encoding
        pe = torch.zeros(resolution, dim)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(np.log(resolution) / dim))
        position = torch.arange(0, resolution).unsqueeze(1)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # #add a row of zeros to the positional encoding
        # pe = torch.cat([torch.zeros(1, dim), pe], dim=0)

        #update the embedding table weights
        self.embedding_table.weight = nn.Parameter(pe, requires_grad=False)

    
    def forward(self, x):
        with torch.no_grad():
            embed = self.embedding_table(x) # B, T, C
            embed = embed.permute(0, 2, 1) # B, C, T
            return embed

# TODO: Try Lower resolution to 20. 
class MaskedNarrationDecoder(nn.Module):
    def __init__(self,
                 num_layers,
                 txt_dim,
                 n_embd,
                 n_head,
                 n_ds_strides= (1, 1),
                 n_out=None,
                 n_hidden=None,
                 act_layer=nn.GELU,
                 attn_pdrop=0.0,
                 proj_pdrop=0.0,
                 path_pdrop=0.0,
                 mha_win_size=-1,
                 use_rel_pe=False,
                 resolution=100,
                ):
        super(MaskedNarrationDecoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = n_head
        self.text_dim = txt_dim

        self.decoder_layers = nn.ModuleList([TransformerBlock(
            n_embd=n_embd,
            n_head=n_head,
            n_ds_strides=n_ds_strides,
            n_out=n_out,
            n_hidden=n_hidden,
            act_layer=act_layer,
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
            mha_win_size=mha_win_size,
            use_rel_pe=use_rel_pe,
            use_cross_modal=True,
        ) for _ in range(num_layers)])

        self.masked_embedding = nn.Embedding(1, n_embd)
        self.modality_embedding = nn.Embedding(2, n_embd)

        #self.modality_lin = nn.Linear(n_embd*2, n_embd)
        self.modality_lin = nn.Conv1d(n_embd*2, n_embd, 1, stride=1, padding=0, groups=n_embd)
        self.relative_pos_embed = RelativeSinusoidalPositionalEncoding(
            resolution,
            n_embd,
        )
        #self.out_lin = nn.Linear(n_embd, txt_dim)
        self.out_lin = nn.Conv1d(n_embd, txt_dim, 1, stride=1, padding=0, groups=n_embd)

    def forward(self, mask_shape, x_mask, y, y_mask, sampled_rel_pos, masked_rel_pos, modality):
        # mask_shape: B, T
        # y shape: B, C, T
        if modality == "single":
            modality_value = 0
        elif modality == "cross_modal":
            modality_value = 1
        else:
            raise ValueError("modality must be either single or cross_modal")
        mask_tokens = torch.zeros(mask_shape).type(torch.long).to(y.device)
        modality_tokens = torch.ones(mask_shape).type(torch.long).to(y.device) * modality_value

        x = self.masked_embedding(mask_tokens) # B, T, C
        x = x.permute(0, 2, 1) # B, C, T
        modality_tokens = self.modality_embedding(modality_tokens) # B, T, C
        modality_tokens = modality_tokens.permute(0, 2, 1) # B, C, T
        x = torch.cat([x, modality_tokens], dim=1) # B, 2C, T
        x = nn.functional.relu(self.modality_lin(x)) # B, C, T

        masked_pos = self.relative_pos_embed(masked_rel_pos)
        x = x + masked_pos
        sampled_pos = self.relative_pos_embed(sampled_rel_pos)
        y = y + sampled_pos
        
        for i in range(self.num_layers):
            x, _ = self.decoder_layers[i](x, x_mask, y, y_mask)
        output = self.out_lin(x) # B, C, T
        output = torch.tanh(output)
        output = output.permute(0, 2, 1) # B, T, C
        return output

class AttnPriorMHA(nn.Module):
    """
    Multi Head Attention with attn prior that can be added to the attention map

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
            self,
            n_embd,  # dimension of the input embedding
            n_head,  # number of heads in multi-head self-attention
            attn_pdrop=0.0,  # dropout rate for the attention map
            proj_pdrop=0.0  # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # attn prior scaler
        self.attn_scaler = nn.Parameter(torch.ones(1, self.n_head, 1, 1))

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask, encoder_hidden_states=None, encoder_attention_mask=None, attn_prior=None):

        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # print("x ", x.shape)
        # print("mask ", mask.shape)
        # print("encoder_hidden_states ", encoder_hidden_states.shape)
        # print("attn_mask ", encoder_attention_mask.shape)

        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            # calculate query, key, values for all heads in batch
            # (B, nh * hs, T)
            q = self.query(x)
            k = self.key(encoder_hidden_states)
            v = self.value(encoder_hidden_states)
            attn_mask = encoder_attention_mask
        else:
            # calculate query, key, values for all heads in batch
            # (B, nh * hs, T)
            k = self.key(x)
            q = self.query(x)
            v = self.value(x)
            attn_mask = mask

        # move head forward to be the batch dim
        # (B, nh * hs, T) -> (B, nh, T, hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        # print("k ", k.shape)
        # print("q ", q.shape)
        # print("v ", v.shape)
        # self-attention: (B, nh, Tx, hs) x (B, nh, hs, Ty) -> (B, nh, Ty, Tx)
        att = (q * self.scale) @ k.transpose(-2, -1)
        if attn_prior is not None:
            B, nh, Ty, Tx = att.shape
            if nh==1:
                attn_prior = attn_prior.expand(B, nh, Ty, Tx)
            att = att + self.attn_scaler * attn_prior
            # att = self.attn_scaler * attn_prior
        #print("att 1 ", att.shape)
        # prevent q from attending to invalid tokens
        att = att.masked_fill(torch.logical_not(attn_mask[:, :, None, :]), float('-inf'))
        #print("att 2 ", att.shape)
        # softmax attn
        att = F.softmax(att, dim=-1)
        #print("att 3 ", att.shape)
        # print("att 3 ", att.shape)
        att = self.attn_drop(att)
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        out = att @ (v * attn_mask[:, :, :, None].to(v.dtype))
        # print("out 1 ", out.shape)
        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * mask.to(out.dtype)
        return out, mask, att

class FeedForward(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, act_layer=nn.GELU, dropout=0.0):
        super(FeedForward, self).__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.act = act_layer()
        self.dropout = dropout
        self.masked_convs = nn.ModuleList([])
        self.dropouts = nn.ModuleList([])
        assert num_layers > 0
        if num_layers == 1:
            assert hidden_dim == out_dim
        self.masked_convs.append(MaskedConv1D(in_dim, hidden_dim, 1))
        self.dropouts.append(nn.Dropout(dropout))
        for _ in range(num_layers-2):
            self.masked_convs.append(MaskedConv1D(hidden_dim, hidden_dim, 1))
            self.dropouts.append(nn.Dropout(dropout))
        self.masked_convs.append(MaskedConv1D(hidden_dim, out_dim, 1))

    def forward(self, x, mask):
        for i in range(self.num_layers):
            x, mask = self.masked_convs[i](x, mask)
            x = self.act(x)
            if i < self.num_layers-1:
                x = self.dropouts[i](x)
        return x, mask


class FPN_FeedForward(nn.Module):
    def __init__(self, fpn_levels, num_layers, in_dim, hidden_dim, out_dim, act_layer=nn.GELU, dropout=0.0):
        super(FPN_FeedForward, self).__init__()
        self.fpn_levels = fpn_levels
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout
        assert fpn_levels > 0
        self.fpn_ff = nn.ModuleList([FeedForward(num_layers, in_dim, hidden_dim, out_dim, act_layer, dropout) for _ in range(fpn_levels)])

    def forward(self, inputs, fpn_masks):
        assert len(inputs) == self.fpn_levels
        assert len(fpn_masks) == self.fpn_levels

        outputs = []
        masks = []
        for i in range(self.fpn_levels):
            x = inputs[i]
            mask = fpn_masks[i]
            x, mask = self.fpn_ff[i](x, mask)
            outputs.append(x)
            masks.append(mask)
        return outputs, masks

class ConsecutiveMasking(nn.Module):
    def __init__(self, dim, consecutive_mask_range, droprate=0.0):
        super(ConsecutiveMasking, self).__init__()
        self.consecutive_mask_range = consecutive_mask_range
        self.droprate = droprate
        self.mask_token = nn.Parameter(torch.randn(1, dim, 1))

    def forward(self, x, video_lens):
        if self.training:
            B, C, T = x.size()
            dropped_x = torch.clone(x)
            consecutive_size = torch.randint(self.consecutive_mask_range[0], self.consecutive_mask_range[1], (B,)).to(x.device)
            num_mask = (video_lens * self.droprate) // consecutive_size
            num_mask = num_mask.to(torch.long)
            max_mask = torch.max(num_mask).to(torch.long).item()
            mask_start = torch.rand((B, max_mask), device=x.device) * (video_lens-consecutive_size).view(-1, 1)
            mask_start = mask_start.to(torch.long)
            for i in range(B):
                for j in range(num_mask[i]):
                    dropped_x[i, :, mask_start[i, j]:mask_start[i, j]+consecutive_size[i]] = self.mask_token
            return dropped_x
        else:
            return x
        
class QCAA(nn.Module):
    def __init__(self,
                lin_embed_layers,
                vid_n_layers,
                txt_n_layers,
                cross_n_layers,
                n_head,
                n_txt_in,
                n_embd,
                n_embd_ks,
                n_vid_in,
                max_len,
                mha_win_size,
                attn_pdrop=0.0,
                proj_pdrop=0.0,
                path_pdrop=0.0,
                with_ln=True,
                context_action_order=5,
                context_action_use_pos=True,
                use_abs_pe=True,
                use_rel_pe=False,
                ):
        super().__init__()
        self.lin_embed_layers = lin_embed_layers
        self.vid_n_layers = vid_n_layers
        self.txt_n_layers = txt_n_layers
        self.cross_n_layers = cross_n_layers
        self.n_head = n_head
        self.n_txt_in = n_txt_in
        self.n_embd = n_embd
        self.n_vid_in = n_vid_in
        self.max_len = max_len
        self.mha_win_size = mha_win_size
        self.attn_pdrop = attn_pdrop
        self.proj_pdrop = proj_pdrop
        self.path_pdrop = path_pdrop
        self.with_ln = with_ln
        self.context_action_order = context_action_order
        self.context_action_use_pos = context_action_use_pos
        self.relu = nn.ReLU(inplace=True)

        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe


        pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd ** 0.5)
        self.register_buffer("pos_embd", pos_embd, persistent=False)

        # txt_embedding network using linear projection
        self.txt_embd = nn.ModuleList()
        self.txt_embd_norm = nn.ModuleList()
        for idx in range(lin_embed_layers):
            if idx == 0:
                in_channels = n_txt_in
            else:
                in_channels = n_embd
            self.txt_embd.append(MaskedConv1D(
                in_channels, n_embd, 1,
                stride=1, padding=0, bias=(not with_ln)
            )
            )
            if with_ln:
                self.txt_embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.txt_embd_norm.append(nn.Identity())
        
        self.txt_stem = nn.ModuleList([TransformerBlock(
            n_embd, n_head,
            n_ds_strides=(1, 1),
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
            mha_win_size=-1,
            use_rel_pe=use_rel_pe,
            use_cross_modal=False,
        ) for _ in range(txt_n_layers)])

        
        # vid_embedding network using convs
        self.vid_embd = nn.ModuleList()
        self.vid_embd_norm = nn.ModuleList()
        for idx in range(lin_embed_layers):
            if idx == 0:
                in_channels = n_vid_in
            else:
                in_channels = n_embd
            self.vid_embd.append(MaskedConv1D(
                in_channels, n_embd, n_embd_ks,
                stride=1, padding=n_embd_ks // 2, bias=(not with_ln)
            )
            )
            if with_ln:
                self.vid_embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.vid_embd_norm.append(nn.Identity())

        self.vid_stem = nn.ModuleList([TransformerBlock(
            n_embd, n_head,
            n_ds_strides=(1, 1),
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
            mha_win_size=self.mha_win_size,
            use_rel_pe=self.use_rel_pe,
            use_cross_modal=True,
        ) for _ in range(vid_n_layers)])


        self.vid_text_stem = CrossModalDualEncoders(
            num_layers=cross_n_layers,
            n_embd=n_embd,
            n_head=n_head,
            n_ds_strides=(1, 1),
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
            mha_win_size=self.mha_win_size,
            use_rel_pe=self.use_rel_pe,
            modality="single",
        )

        if self.context_action_order != -1:
            # self.detached_context_action = detached_context_action
            self.context_action_use_pos = context_action_use_pos
            if self.context_action_use_pos:
                self.context_action_token = torch.nn.Parameter(torch.empty(1, n_embd, 1))
                torch.nn.init.normal_(self.context_action_token, std=0.02)
                self.succeeding_action_token = torch.nn.Parameter(torch.empty(1, n_embd, 1))
                torch.nn.init.normal_(self.succeeding_action_token, std=0.02)
                self.preceding_action_token = torch.nn.Parameter(torch.empty(1, n_embd, 1))
                torch.nn.init.normal_(self.preceding_action_token, std=0.02)
                context_action_pos_embed = get_sinusoid_encoding(self.context_action_order+1, n_embd) / (n_embd ** 0.5) # 1, C, N+1
                self.register_buffer("context_action_pos", context_action_pos_embed, persistent=False) # (1,C,N+1)
            else:
                self.context_action_tokens = torch.nn.Parameter(torch.empty(1, n_embd, 2*self.context_action_order+1))
                torch.nn.init.normal_(self.context_action_tokens, std=0.02)
            self.context_action_vis_lin = nn.Parameter(torch.empty(n_embd, n_vid_in, 2*self.context_action_order+1))
            torch.nn.init.normal_(self.context_action_vis_lin, std=0.02)
            self.context_action_vis_bias = nn.Parameter(torch.empty(n_vid_in, 2*self.context_action_order+1))
            torch.nn.init.normal_(self.context_action_vis_bias, std=0.02)

    def forward(self, src_vid, src_vid_mask, src_txt, src_txt_mask):
        B, C, Tt = src_txt.size()
        pe = self.pos_embd

        B, C, T = src_vid.size()

        # vid_embedding network
        for idx in range(len(self.vid_embd)):
            src_vid, src_vid_mask = self.vid_embd[idx](src_vid, src_vid_mask)
            src_vid = self.relu(self.vid_embd_norm[idx](src_vid))

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        # stem network
        for idx in range(len(self.vid_stem)):
            src_vid, src_vid_mask = self.vid_stem[idx](src_vid, src_vid_mask)

        assert src_txt is not None
        
        for idx in range(len(self.txt_embd)):
            src_txt, src_txt_mask = self.txt_embd[idx](src_txt, src_txt_mask)
            src_txt = self.relu(self.txt_embd_norm[idx](src_txt))

        if self.use_abs_pe and self.training:
            pe = self.pos_embd
            # add pe to x
            src_txt = src_txt + pe[:, :, :Tt] * src_txt_mask.to(src_txt.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            pe = self.pos_embd
            # add pe to x
            src_txt = src_txt + pe[:, :, :Tt] * src_txt_mask.to(src_txt.dtype)
        
        lang_idx = 0

        if self.context_action_order != -1:
            lang_idx = 2*self.context_action_order+1
            prev_size = src_txt.shape[2]
            # CURR TOKEN ORDER : LANGUAGE+1, LANGUAGE+2, ...
            if self.context_action_use_pos:
                succeeding_tokens = self.succeeding_action_token.expand(-1, -1, self.context_action_order) + self.context_action_pos[:, :, 1:] 
                preceding_tokens = self.preceding_action_token.expand(-1, -1, self.context_action_order) + self.context_action_pos[:, :, 1:] 
                current_token = self.context_action_token
                ca_tokens = torch.cat([current_token, succeeding_tokens, preceding_tokens], dim=2)
                assert ca_tokens.shape == (1, src_txt.shape[1], 2*self.context_action_order+1)
            else:
                ca_tokens = self.context_action_tokens
            src_txt = torch.cat([ca_tokens.expand(src_txt.shape[0], -1, -1), src_txt], dim=2)
            src_txt_mask = torch.cat([torch.ones(src_txt_mask.shape[0], 1, ca_tokens.shape[2], device=src_txt_mask.device), src_txt_mask] , dim=2)
            # CURR TOKEN ORDER : DURING, AFTER+1, AFTER+2, ..., AFTER+Cn, BEFORE+1, BEFORE+2, ..., BEFORE+Cn, LANGUAGE+1, LANGUAGE+2, ...
            assert src_txt.shape[2] == prev_size + 2*self.context_action_order+1
            assert src_txt_mask.shape[2] == prev_size + 2*self.context_action_order+1

        for i in range(len(self.txt_stem)):
            src_txt, src_txt_mask = self.txt_stem[i](src_txt, src_txt_mask)
        
        context_action_lang_only = None
        if self.context_action_order != -1:
            context_action_tokens = src_txt[:, :, :lang_idx]
            # CURR TOKEN ORDER : DURING, AFTER+1, AFTER+2, ..., AFTER+Cn, BEFORE+1, BEFORE+2, ..., BEFORE+Cn
            assert context_action_tokens.shape[2] == 2*self.context_action_order+1
            # context_action_pred =  torch.einsum("bct,cot->bot", context_action_tokens, self.context_action_lin) # apply linear transformation to each token separately | B, C, T
            # context_action_pred = F.gelu(context_action_pred + self.context_action_bias.unsqueeze(0)) # (B, C, T) + (1, C, T)
            context_action_lang_only = torch.einsum("bct,cot->bot", context_action_tokens, self.context_action_vis_lin.expand(-1, -1, context_action_tokens.shape[2])) # B, C, Vin
            context_action_lang_only = context_action_lang_only + self.context_action_vis_bias.unsqueeze(0).expand(-1, -1, context_action_lang_only.shape[2]) # (B, C, Vin) + (1, C, Vin)
            # if self.detached_context_action:
            #     context_action_tokens = context_action_tokens.detach()
            #     src_txt = torch.cat([context_action_tokens, src_txt[:, :, lang_idx:]], dim=2)
        
        # if self.single_token_language:
            #     src_query = src_txt[:, :, :lang_idx+1]
            #     src_query_mask = src_txt_mask[:, :, :lang_idx+1]
        # else:
        src_query = src_txt
        src_query_mask = src_txt_mask

        # if not self.add_language:
        #     src_query = src_query[:, :, :lang_idx]
        #     src_query_mask = src_query_mask[:, :, :lang_idx]

        #     assert src_query.shape[2] == 2*self.context_action_order+1
        
        # sampled_idxs = torch.arange(2*self.context_action_order+1, device=src_query.device)
        # with torch.no_grad():
        #     if self.training and self.random_anticipation:
        #         idxs_sampling = torch.rand(2*self.context_action_order+1, device=src_query.device)
        #         sampled_idxs = torch.nonzero(idxs_sampling < 0.5, as_tuple=True)[0]
        #         remove_lang = torch.rand(1).item() < 0.5
        #         if sampled_idxs.numel() == 0 and remove_lang:
        #             sampled_idxs = torch.tensor([0], device=src_query.device)
        #         if remove_lang: # remove language as well
        #             src_query = src_query[:, :, sampled_idxs]
        #             src_query_mask = src_query_mask[:, :, sampled_idxs]
        #         else:
        #             sampled_tokens = src_query[:, :, sampled_idxs]
        #             src_query = torch.cat([sampled_tokens, src_query[:, :, lang_idx:]], dim=2)
        #             src_query_mask = torch.cat([src_query_mask[:, :, sampled_idxs], src_query_mask[:, :, lang_idx:]], dim=2)

        src_vid, src_query = self.vid_text_stem(src_vid, src_vid_mask, src_query, src_query_mask)

        context_action_with_vis = None
        if self.context_action_order != -1:
            context_action_tokens = src_query[:, :, :lang_idx]
            # CURR TOKEN ORDER : DURING, AFTER+1, AFTER+2, ..., AFTER+Cn, BEFORE+1, BEFORE+2, ..., BEFORE+Cn
            assert context_action_tokens.shape[2] == 2*self.context_action_order+1
            # context_action_pred =  torch.einsum("bct,cot->bot", context_action_tokens, self.context_action_lin) # apply linear transformation to each token separately | B, C, T
            # context_action_pred = F.gelu(context_action_pred + self.context_action_bias.unsqueeze(0)) # (B, C, T) + (1, C, T)
            context_action_with_vis = torch.einsum("bct,cot->bot", context_action_tokens, self.context_action_vis_lin) # B, C, Vin
            context_action_with_vis = context_action_with_vis + self.context_action_vis_bias.unsqueeze(0) # (B, C, Vin) + (1, C, Vin)

        return src_vid, src_vid_mask, src_txt, src_txt_mask, context_action_lang_only, context_action_with_vis

class QCAA_Lang(nn.Module):
    def __init__(self,
                lin_embed_layers,
                txt_n_layers,
                n_head,
                n_txt_in,
                n_embd,
                n_embd_ks,
                n_vid_in,
                max_len,
                mha_win_size,
                attn_pdrop=0.0,
                proj_pdrop=0.0,
                path_pdrop=0.0,
                with_ln=True,
                context_action_order=5,
                context_action_use_pos=True,
                use_abs_pe=True,
                use_rel_pe=False,
                uniform_vid_sample=0
                ):
        super().__init__()
        self.lin_embed_layers = lin_embed_layers
        self.txt_n_layers = txt_n_layers
        self.n_head = n_head
        self.n_txt_in = n_txt_in
        self.n_embd = n_embd
        self.n_vid_in = n_vid_in
        self.max_len = max_len
        self.mha_win_size = mha_win_size
        self.attn_pdrop = attn_pdrop
        self.proj_pdrop = proj_pdrop
        self.path_pdrop = path_pdrop
        self.with_ln = with_ln
        self.context_action_order = context_action_order
        self.context_action_use_pos = context_action_use_pos
        self.relu = nn.ReLU(inplace=True)
        self.uniform_vid_sample = uniform_vid_sample

        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe

        pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd ** 0.5)
        self.register_buffer("pos_embd", pos_embd, persistent=False)

        # vid proj

        self.vid_proj = nn.Linear(n_vid_in, n_embd)

        # txt_embedding network using linear projection
        self.txt_embd = nn.ModuleList()
        self.txt_embd_norm = nn.ModuleList()
        for idx in range(lin_embed_layers):
            if idx == 0:
                in_channels = n_txt_in
            else:
                in_channels = n_embd
            self.txt_embd.append(MaskedConv1D(
                in_channels, n_embd, 1,
                stride=1, padding=0, bias=(not with_ln)
            )
            )
            if with_ln:
                self.txt_embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.txt_embd_norm.append(nn.Identity())
        
        self.txt_stem = nn.ModuleList([TransformerBlock(
            n_embd, n_head,
            n_ds_strides=(1, 1),
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
            mha_win_size=-1,
            use_rel_pe=use_rel_pe,
            use_cross_modal=False,
        ) for _ in range(txt_n_layers)])

        if self.context_action_order != -1:
            # self.detached_context_action = detached_context_action
            self.context_action_use_pos = context_action_use_pos
            if self.context_action_use_pos:
                self.context_action_token = torch.nn.Parameter(torch.empty(1, n_embd, 1))
                torch.nn.init.normal_(self.context_action_token, std=0.02)
                self.succeeding_action_token = torch.nn.Parameter(torch.empty(1, n_embd, 1))
                torch.nn.init.normal_(self.succeeding_action_token, std=0.02)
                self.preceding_action_token = torch.nn.Parameter(torch.empty(1, n_embd, 1))
                torch.nn.init.normal_(self.preceding_action_token, std=0.02)
                context_action_pos_embed = get_sinusoid_encoding(self.context_action_order+1, n_embd) / (n_embd ** 0.5) # 1, C, N+1
                self.register_buffer("context_action_pos", context_action_pos_embed, persistent=False) # (1,C,N+1)
            else:
                self.context_action_tokens = torch.nn.Parameter(torch.empty(1, n_embd, 2*self.context_action_order+1))
                torch.nn.init.normal_(self.context_action_tokens, std=0.02)
            self.context_action_vis_lin = nn.Parameter(torch.empty(n_embd, n_vid_in, 2*self.context_action_order+1))
            torch.nn.init.normal_(self.context_action_vis_lin, std=0.02)
            self.context_narration_lin = nn.Parameter(torch.empty(n_embd, n_txt_in, 2*self.context_action_order+1))
            torch.nn.init.normal_(self.context_narration_lin, std=0.02)
            self.context_action_vis_bias = nn.Parameter(torch.empty(n_vid_in, 2*self.context_action_order+1))
            torch.nn.init.normal_(self.context_action_vis_bias, std=0.02)
            self.context_narration_bias = nn.Parameter(torch.empty(n_txt_in, 2*self.context_action_order+1))
            torch.nn.init.normal_(self.context_narration_bias, std=0.02)

    def forward(self, src_vid, src_vid_mask, src_txt, src_txt_mask, vid_lens):
        
        B, C, Tt = src_txt.size()
        _, vC, _ = src_vid.size()


        if not self.uniform_vid_sample:
            avg_vid = torch.sum(src_vid, dim=-1) / (torch.sum(src_vid_mask, dim=-1) + 1e-6)
            proj_vid = self.relu(self.vid_proj(avg_vid)) # B, C
            proj_vid = proj_vid.unsqueeze(2) # B, C, 1
        else:
            indices = torch.linspace(0,1,self.uniform_vid_sample, device=src_vid.device).unsqueeze(0).expand(B, self.uniform_vid_sample)
            indices = (indices*(vid_lens-1).view(B,1)).long()
            v_indices = indices.unsqueeze(1).expand(-1, vC, -1)
            p_indices = indices.unsqueeze(1).expand(-1, self.n_embd, -1)
            select_vid = torch.gather(src_vid, 2, v_indices)
            select_vid = select_vid.permute(0, 2, 1) # B, T, C
            proj_vid = self.relu(self.vid_proj(select_vid)) # B, T, C
            proj_vid = proj_vid.permute(0, 2, 1) # B, C, T
            pe = self.pos_embd.expand(B, -1, -1)
            position = torch.gather(pe, 2, p_indices)
            proj_vid = proj_vid+position  # B, C, T
        
        _, _, vT = proj_vid.size()

        

        pe = self.pos_embd
        assert src_txt is not None
        
        for idx in range(len(self.txt_embd)):
            src_txt, src_txt_mask = self.txt_embd[idx](src_txt, src_txt_mask)
            src_txt = self.relu(self.txt_embd_norm[idx](src_txt))

        if self.use_abs_pe and self.training:
            pe = self.pos_embd
            # add pe to x
            src_txt = src_txt + pe[:, :, :Tt] * src_txt_mask.to(src_txt.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            pe = self.pos_embd
            # add pe to x
            src_txt = src_txt + pe[:, :, :Tt] * src_txt_mask.to(src_txt.dtype)
        
        # Add projected video to the beginning
        src_txt = torch.cat([proj_vid, src_txt], dim=2)
        src_txt_mask = torch.cat([torch.ones(src_txt_mask.shape[0], 1, proj_vid.size(2), device=src_txt_mask.device), src_txt_mask], dim=2)

        if self.context_action_order != -1:
            prev_size = src_txt.shape[2]
            # CURR TOKEN ORDER : AVG_VID, LANGUAGE+1, LANGUAGE+2, ...
            if self.context_action_use_pos:
                succeeding_tokens = self.succeeding_action_token.expand(-1, -1, self.context_action_order) + self.context_action_pos[:, :, 1:] 
                preceding_tokens = self.preceding_action_token.expand(-1, -1, self.context_action_order) + self.context_action_pos[:, :, 1:] 
                current_token = self.context_action_token
                ca_tokens = torch.cat([current_token, succeeding_tokens, preceding_tokens], dim=2)
                assert ca_tokens.shape == (1, src_txt.shape[1], 2*self.context_action_order+1)
            else:
                ca_tokens = self.context_action_tokens
            src_txt = torch.cat([ca_tokens.expand(src_txt.shape[0], -1, -1), src_txt], dim=2)
            src_txt_mask = torch.cat([torch.ones(src_txt_mask.shape[0], 1, ca_tokens.shape[2], device=src_txt_mask.device), src_txt_mask] , dim=2)
            # CURR TOKEN ORDER : DURING, AFTER+1, AFTER+2, ..., AFTER+Cn, BEFORE+1, BEFORE+2, ..., BEFORE+Cn, AVG_VID, LANGUAGE+1, LANGUAGE+2, ...
            assert src_txt.shape[2] == prev_size + 2*self.context_action_order+1
            assert src_txt_mask.shape[2] == prev_size + 2*self.context_action_order+1

        for i in range(len(self.txt_stem)):
            src_txt, src_txt_mask = self.txt_stem[i](src_txt, src_txt_mask)
        
        context_action_lang_only = None
        context_narration_lang_only = None
        if self.context_action_order != -1:
            context_action_tokens = src_txt[:, :, :2*self.context_action_order+1]
            # CURR TOKEN ORDER : DURING, AFTER+1, AFTER+2, ..., AFTER+Cn, BEFORE+1, BEFORE+2, ..., BEFORE+Cn
            assert context_action_tokens.shape[2] == 2*self.context_action_order+1
            context_action_lang_only = torch.einsum("bct,cot->bot", context_action_tokens, self.context_action_vis_lin) # B, C, Vin
            context_action_lang_only = context_action_lang_only + self.context_action_vis_bias.unsqueeze(0) # (B, C, Vin) + (1, C, Vin)

            context_narration_lang_only = torch.einsum("bct,cot->bot", context_action_tokens, self.context_narration_lin) # B, C, Vin
            context_narration_lang_only = context_narration_lang_only + self.context_narration_bias.unsqueeze(0) # (B, C, Vin) + (1, C, Vin)
        
        # remove vid feats
        if self.context_action_order != -1:
            src_txt = torch.cat([src_txt[:, :, :2*self.context_action_order+1], src_txt[:, :, 2*self.context_action_order+1+vT:]], dim=2)
            src_txt_mask = torch.cat([src_txt_mask[:, :, :2*self.context_action_order+1], src_txt_mask[:, :, 2*self.context_action_order+1+vT:]], dim=2)
            assert src_txt.shape[2] == 2*self.context_action_order+1 + Tt
        elif self.context_action_order == -1:
            src_txt = src_txt[:, :, vT:]
            src_txt_mask = src_txt_mask[:, :, vT:]
            assert src_txt.shape[2] == Tt

        return src_txt, src_txt_mask, context_action_lang_only, context_narration_lang_only

class MLP(nn.Module):
    def __init__(self, n_in, n_out, n_layers, n_hidden, pdrop):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.layers.append(nn.Linear(n_in, n_hidden))
            else:
                self.layers.append(nn.Linear(n_hidden, n_hidden))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(pdrop))
        if n_layers == 0:
            self.layers.append(nn.Linear(n_in, n_out))
        else:
            self.layers.append(nn.Linear(n_hidden, n_out))

    def forward(self, x):
        prev = x
        for layer in self.layers[:-1]:
            prev = layer(prev)
        x = self.layers[-1](prev)
        return x, prev

class FlashTransformerBlock(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        cross_attn=False,
        n_out=None,
        n_hidden=None,
        n_ds_strides=(1, 1),
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0,  # dropout rate for the projection / MLP
        path_pdrop=0.0,  # drop path rate
        mha_win_size=-1,
    ):
        # Assume FusedMLP and fused_bias_fc and fused_dropout_add_ln is True
        super().__init__()
        n_hidden = n_hidden or 4 * n_embd
        n_out = n_out or n_embd
        self.downsample = n_ds_strides[0] > 1
        self.cross_attn = cross_attn

        self.attn = FlashMHA(
            embed_dim=n_embd,
            num_heads=n_head,
            dropout=attn_pdrop,
            window_size=(mha_win_size,mha_win_size),
            causal=False,
            fused_bias_fc=True,
            use_flash_attn=True,
            cross_attn=False,
        )
        # convert to bFloat16
        self.attn = self.attn.to(torch.bfloat16)
        self.dropout1 = nn.Dropout(proj_pdrop)
        self.drop_path1 = StochasticDepth(path_pdrop, mode="row")
        self.norm1 = nn.LayerNorm(n_embd)
        self.mlp = FusedMLP(
            in_features=n_embd,
            hidden_features=n_hidden,
            out_features=n_out,
        )
        self.use_cross_attn = cross_attn
        if cross_attn:
            self.cross_dropout = nn.Dropout(proj_pdrop)
            self.cross_drop_path = StochasticDepth(path_pdrop, mode="row")
            self.cross_norm = nn.LayerNorm(n_embd)
            self.cross_attn = FlashMHA(
                embed_dim=n_embd,
                num_heads=n_head,
                dropout=attn_pdrop,
                causal=False,
                fused_bias_fc=True,
                use_flash_attn=True,
                cross_attn=True,
            )
            # convert to bFloat16
            self.cross_attn = self.cross_attn.to(torch.bfloat16)

        self.dropout2 = nn.Dropout(proj_pdrop)
        self.drop_path2 = StochasticDepth(path_pdrop, mode="row")
        self.norm2 = nn.LayerNorm(n_embd)
        kernel_size = n_ds_strides[0] + 1 if n_ds_strides[0] > 1 else 3
        stride, padding = n_ds_strides[1], kernel_size // 2
        if self.downsample:
            self.downsample_pool = nn.MaxPool1d(
                kernel_size, stride=stride, padding=padding)
            self.downsample_conv = MaskedConv1D(
            n_embd, n_embd, kernel_size,
            stride=stride, padding=padding, groups=n_embd, bias=False
            )
        # else:
        #     self.conv = MaskedConv1D(
        #     n_embd, n_embd, kernel_size,
        #     stride=stride, padding=padding, groups=n_embd, bias=False
        # )
        #     self.conv_norm = nn.LayerNorm(n_embd)


    def forward(
        self,
        context,
        context_pe,
        context_mask,
        self_varlen_params,
        residual=None,
        query=None,
        query_pe=None,
        query_mask=None,
        cross_varlen_params=None,
    ):
        r"""Pass the input through the encoder layer.
        Args:
            context: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
            query: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        """
        assert "indices" in self_varlen_params
        assert "cu_seqlens" in self_varlen_params
        assert "max_seqlen" in self_varlen_params

        if context_pe is not None:
            context = context+ context_pe

        if self.downsample:
            B, _, T = context_mask.shape
            normal_shaped_x = pad_input(context, self_varlen_params["indices"], B, T)
            normal_shaped_x = normal_shaped_x.permute(0, 2, 1) # B C T
            downsampled_x, downsampled_mask = self.downsample_conv(normal_shaped_x, context_mask)
            downsampled_context, downsampled_indices, downsampled_cu_seqlens, downsampled_max_seqlen, _ = unpad_input(downsampled_x.permute(0, 2, 1), downsampled_mask.squeeze(1))
            
            downsampled_varlen_params = {
                "indices": downsampled_indices,
                "cu_seqlens": downsampled_cu_seqlens,
                "max_seqlen": downsampled_max_seqlen,
            }
            context = downsampled_context
            if residual is not None:
                residual = pad_input(residual, self_varlen_params["indices"], B, T) # B T C
                residual = residual.permute(0, 2, 1) # B C T
                residual = self.downsample_pool(residual) * downsampled_mask.float() # B C T
                residual, _, _, _, _ = unpad_input(residual.permute(0, 2, 1), downsampled_mask.squeeze(1))
            
            self_varlen_params = downsampled_varlen_params
        # else:
        #     B, _, T = context_mask.shape
        #     normal_shaped_x = pad_input(context, self_varlen_params["indices"], B, T)
        #     normal_shaped_x = normal_shaped_x.permute(0, 2, 1) # B C T
        #     context, _ = self.conv(normal_shaped_x, context_mask)
        #     context = context.permute(0, 2, 1) # B T C
        #     context = self.conv_norm(context)
        #     context, indices, cu_seqlens, max_seqlen, _ = unpad_input(context, context_mask.squeeze(1))


        if self.drop_path1.p == 0 or not self.training:
            rowscale1 = None
        else:
            rowscale1 = self.drop_path1(
                torch.ones(
                    context.shape[:-1],
                    device=context.device,
                    dtype=context.dtype,
                )
            )
        context, residual = layer_norm_fn(
            context,
            self.norm1.weight,
            self.norm1.bias,
            residual=residual,
            eps=self.norm1.eps,
            dropout_p=self.dropout1.p if self.training else 0.0,
            rowscale=rowscale1,
            prenorm=True,
            is_rms_norm=False
        )

        context = self.attn(x=context.to(torch.bfloat16), cu_seqlens=self_varlen_params["cu_seqlens"], max_seqlen=self_varlen_params["max_seqlen"])
        context = context.to(torch.float32)

        if self.use_cross_attn:
            if self.cross_drop_path.p == 0 or not self.training:
                X_rowscale = None
            else:
                X_rowscale = self.cross_drop_path(
                    torch.ones(
                        context.shape[:-1],
                        device=context.device,
                        dtype=context.dtype,
                    )
                )
            context, residual = layer_norm_fn(
                context,
                self.cross_norm.weight,
                self.cross_norm.bias,
                residual=residual,
                eps=self.cross_norm.eps,
                dropout_p=self.cross_dropout.p if self.training else 0.0,
                rowscale=X_rowscale,
                prenorm=True,
                is_rms_norm=False
            )
            assert "indices" in cross_varlen_params
            assert "cu_seqlens" in cross_varlen_params
            assert "max_seqlen" in cross_varlen_params
            assert query is not None, "query must be provided for cross attention"

            context = self.cross_attn(x=context.to(torch.bfloat16),
                                        x_kv=query.to(torch.bfloat16),
                                        cu_seqlens=self_varlen_params["cu_seqlens"], 
                                        max_seqlen=self_varlen_params["max_seqlen"], 
                                        cu_seqlens_k=cross_varlen_params["cu_seqlens"], 
                                        max_seqlen_k=cross_varlen_params["max_seqlen"])
            context = context.to(torch.float32)

        if self.drop_path2.p == 0 or not self.training:
            rowscale2 = None
        else:
            rowscale2 = self.drop_path2(
                torch.ones(
                    context.shape[:-1],
                    device=context.device,
                    dtype=context.dtype,
                )
            )
        context, residual = layer_norm_fn(
            context,
            self.norm2.weight,
            self.norm2.bias,
            residual=residual,
            eps=self.norm2.eps,
            dropout_p=self.dropout2.p if self.training else 0.0,
            rowscale=rowscale2,
            prenorm=True,
            is_rms_norm=False
        )
        context = self.mlp(context)
        if self.downsample:
            return context, downsampled_mask, residual, downsampled_varlen_params
        else:
            return context, context_mask, residual


class AlternateTransformerBlock(nn.Module):
    """
    A simple (post layer norm) Transformer block
    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
            self,
            n_embd,  # dimension of the input features
            n_head,  # number of attention heads
            n_ds_strides=(1, 1),  # downsampling strides for q & x, k & v
            n_out=None,  # output dimension, if None, set to input dim
            n_hidden=None,  # dimension of the hidden layer in MLP
            act_layer=nn.GELU,  # nonlinear activation used in MLP, default GELU
            attn_pdrop=0.0,  # dropout rate for the attention map
            proj_pdrop=0.0,  # dropout rate for the projection / MLP
            path_pdrop=0.0,  # drop path rate
            mha_win_size=-1,  # > 0 to use window mha
            use_rel_pe=False,  # if to add rel position encoding to attention
            cross_attn=False,  # if to add cross_modal attention
    ):
        super().__init__()
        assert len(n_ds_strides) == 2
        # layer norm for order (B C T)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

        self.attn = AlternateMaskedMHCA(
            n_embd,
            n_head,
            n_qx_stride=n_ds_strides[0],
            n_kv_stride=n_ds_strides[1],
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            window_size=mha_win_size,
        )

        self.use_cross_attn = cross_attn
        if self.use_cross_attn:
            self.cross_attn = AlternateMHA(
                n_embd,
                n_head,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
            )
            self.ln3 = LayerNorm(n_embd)
            self.cross_pool_skip = nn.Identity()

        # input
        if n_ds_strides[0] > 1:
            kernel_size, stride, padding = \
                n_ds_strides[0] + 1, n_ds_strides[0], (n_ds_strides[0] + 1) // 2
            self.pool_skip = nn.MaxPool1d(
                kernel_size, stride=stride, padding=padding)
        else:
            self.pool_skip = nn.Identity()

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd
        # ok to use conv1d here with stride=1
        self.mlp = FusedMLP(
            in_features=n_embd,
            hidden_features=n_hidden,
            out_features=n_out,
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mlp = nn.Identity()

    def forward(self, x, mask, cross_y=None, cross_y_mask=None, pos_embd=None, attn_prior=None, return_attn=False):
        # pre-LN transformer: https://arxiv.org/pdf/2002.04745.pdf

        #  downsample in the multi-head local attention
        out, out_mask = self.attn(self.ln1(x), mask)

        out_mask_float = out_mask.to(out.dtype)
        out = self.pool_skip(x) * out_mask_float + self.drop_path_attn(out)

        # optional cross_modal attention
        if self.use_cross_attn and cross_y is not None:
            # print("inside")
            cross_out, cross_out_mask = self.cross_attn(self.ln3(out), out_mask_float, cross_y, cross_y_mask)
            out_mask_float = out_mask.to(cross_out_mask.dtype)
            out = self.cross_pool_skip(out) * out_mask_float + self.drop_path_attn(cross_out)

        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.ln2(out).permute(0,2,1)).permute(0,2,1) * out_mask_float)
        # optionally add pos_embd to the output
        if pos_embd is not None:
            out += pos_embd * out_mask_float
        return out, out_mask

class AlternateMaskedMHCA(nn.Module):
    def __init__(
            self,
            n_embd,  # dimension of the output features
            n_head,  # number of heads in multi-head self-attention
            window_size=-1,
            n_qx_stride=1,  # dowsampling stride for query and input
            n_kv_stride=1,  # downsampling stride for key and value
            attn_pdrop=0.0,  # dropout rate for the attention map
            proj_pdrop=0.0,  # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.window_size = window_size

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        if self.n_qx_stride > 1:
            self.downsample=True
        else:
            self.downsample=False

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2

        if self.downsample:
            # 1d depthwise conv
            self.query_conv = MaskedConv1D(
                self.n_embd, self.n_embd, kernel_size,
                stride=stride, padding=padding, groups=self.n_embd, bias=False
            )
            # layernorm
            self.query_norm = LayerNorm(self.n_embd)

            # key, value conv (depthwise)
            kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
            stride, padding = self.n_kv_stride, kernel_size // 2
            # 1d depthwise conv
            self.key_conv = MaskedConv1D(
                self.n_embd, self.n_embd, kernel_size,
                stride=stride, padding=padding, groups=self.n_embd, bias=False
            )
            self.key_norm = LayerNorm(self.n_embd)
            self.value_conv = MaskedConv1D(
                self.n_embd, self.n_embd, kernel_size,
                stride=stride, padding=padding, groups=self.n_embd, bias=False
            )
            # layernorm
            self.value_norm = LayerNorm(self.n_embd)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # print("self-attention")
        # print("x ", x.shape)
        # print("mask ", mask.shape)
        
        if self.downsample:
            # query conv -> (B, nh * hs, T')
            q, qx_mask = self.query_conv(x, mask)
            q = self.query_norm(q)
            # key, value conv -> (B, nh * hs, T'')
            k, kv_mask = self.key_conv(x, mask)
            k = self.key_norm(k)
            v, _ = self.value_conv(x, mask)
            v = self.value_norm(v)

            # projections
            q = self.query(q)
            k = self.key(k)
            v = self.value(v)
        else:
            q = self.query(x)
            k = self.key(x)
            v = self.value(x)

            qx_mask = mask
            kv_mask = mask

        # print("k ", k.shape)
        # print("q ", q.shape)
        # print("v ", v.shape)

        # move head forward to be the batch dim
        # (B, nh * hs, T'/T'') -> (B, nh, T'/T'', hs)

        q_packed, q_indices, q_cu_seqlens, q_max_seqlen, _ = unpad_input(q.permute(0, 2, 1), qx_mask.squeeze(1))
        k_packed, _, _, _, _ = unpad_input(k.permute(0, 2, 1), kv_mask.squeeze(1))
        v_packed, _, _, _, _ = unpad_input(v.permute(0, 2, 1), kv_mask.squeeze(1))

        q_packed = q_packed.view(-1, self.n_head, self.n_channels)
        k_packed = k_packed.view(-1, self.n_head, self.n_channels)
        v_packed = v_packed.view(-1, self.n_head, self.n_channels)

        qkv_packed = torch.stack([q_packed, k_packed, v_packed], dim=1) # (B, 3, nh, hs)

        qkv_packed = qkv_packed.to(torch.bfloat16)

        out = flash_attn_varlen_qkvpacked_func(
            qkv_packed,
            q_cu_seqlens,
            q_max_seqlen,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            causal=False,
            window_size=(self.window_size, self.window_size),
        )

        out = out.to(torch.float32)

        out = out.view(-1, self.n_head * self.n_channels)

        out = pad_input(out, q_indices, B, qx_mask.size(2))
        out = out.permute(0, 2, 1) # B, C, T


        # k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        # q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        # v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)
        return out, qx_mask

class AlternateMHA(nn.Module):
    def __init__(
            self,
            n_embd,  # dimension of the output features
            n_head,  # number of heads in multi-head self-attention
            attn_pdrop=0.0,  # dropout rate for the attention map
            proj_pdrop=0.0,  # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask, y, y_mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # print("self-attention")
        # print("x ", x.shape)
        # print("mask ", mask.shape)
    
        q = self.query(x)
        k = self.key(y)
        v = self.value(y)

        qx_mask = mask
        kv_mask = y_mask

        # print("k ", k.shape)
        # print("q ", q.shape)
        # print("v ", v.shape)

        # move head forward to be the batch dim
        # (B, nh * hs, T'/T'') -> (B, nh, T'/T'', hs)

        q_packed, q_indices, q_cu_seqlens, q_max_seqlen, _ = unpad_input(q.permute(0, 2, 1), qx_mask.squeeze(1))
        k_packed, k_indices, kv_cu_seqlens, kv_max_seqlen, _ = unpad_input(k.permute(0, 2, 1), kv_mask.squeeze(1))
        v_packed, _, _, _, _ = unpad_input(v.permute(0, 2, 1), kv_mask.squeeze(1))

        q_packed = q_packed.view(-1, self.n_head, self.n_channels)
        k_packed = k_packed.view(-1, self.n_head, self.n_channels)
        v_packed = v_packed.view(-1, self.n_head, self.n_channels)

        kv_packed = torch.stack([k_packed, v_packed], dim=1) # (B, 2, nh, hs)
        
        q_packed = q_packed.to(torch.bfloat16)
        kv_packed = kv_packed.to(torch.bfloat16)

        out = flash_attn_varlen_kvpacked_func(
            q_packed,
            kv_packed,
            q_cu_seqlens,
            kv_cu_seqlens,
            q_max_seqlen,
            kv_max_seqlen,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            causal=False,
        )

        out = out.to(torch.float32)

        out = out.view(-1, self.n_head * self.n_channels)

        out = pad_input(out, q_indices, B, qx_mask.size(2))
        out = out.permute(0, 2, 1) # B, C, T


        # k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        # q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        # v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)
        return out, qx_mask