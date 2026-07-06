import math
import os

import torch
from torch import nn
from torch.nn import functional as F

from .models import register_meta_arch, make_backbone, make_neck, make_generator
from .blocks import MaskedConv1D, Scale, LayerNorm, MaskedNarrationDecoder, TransformerBlock, FPN_FeedForward, ConsecutiveMasking, QCAA, QCAA_Lang, get_sinusoid_encoding, MLP, FlashTransformerBlock
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss, pred_gt_contrastive_loss
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.modules.mha import MHA as FlashMHA


from ..utils import batched_nms

def nan_check(x, name):
    if torch.isnan(x).any():
        print(f"NaN detected in tensor: {name}, shape: {x.shape}")
        assert False

class PtConvClsHead(nn.Module):
    """
    1D Conv heads for classification
    """

    def __init__(
            self,
            input_dim,
            feat_dim,
            num_classes,
            prior_prob=0.01,
            num_layers=3,
            kernel_size=3,
            act_layer=nn.ReLU,
            with_ln=False,
            empty_cls=[],
    ):
        super().__init__()
        self.act = act_layer()

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers - 1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        # transformer block to model long range dependencies
        # self.transformer = TransformerBlock(
        #     n_embd=feat_dim,
        #     n_head=4,
        #     path_pdrop=0.1,
        #     mha_win_size=-1,
        # )

        # classifier
        self.cls_head = MaskedConv1D(
            feat_dim, num_classes, kernel_size,
            stride=1, padding=kernel_size // 2
        )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights associated with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            #cur_out, _ = self.transformer(cur_out, cur_mask)
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits,)

        # fpn_masks remains the same
        return out_logits


class PtConvRegHead(nn.Module):
    """
    Shared 1D Conv heads for regression
    Simlar logic as PtTransformerClsHead with separated implementation for clarity
    """

    def __init__(
            self,
            input_dim,
            feat_dim,
            fpn_levels,
            num_layers=3,
            kernel_size=3,
            act_layer=nn.ReLU,
            with_ln=False
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers - 1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        # segment regression
        self.offset_head = MaskedConv1D(
            feat_dim, 2, kernel_size,
            stride=1, padding=kernel_size // 2
        )

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat # B, C, T
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            out_offsets += (F.relu(self.scale[l](cur_offsets)),)

        # fpn_masks remains the same
        return out_offsets


@register_meta_arch("LocPointTransformer")
class PtTransformer(nn.Module):
    """
        Transformer based model for single stage action localization
    """

    def __init__(
            self,
            backbone_type,  # a string defines which backbone we use
            fpn_type,  # a string defines which fpn we use
            backbone_arch,  # a tuple defines # layers in embed / stem / branch
            scale_factor,  # scale factor between branch layers
            input_vid_dim,  # input video feat dim
            input_txt_dim,  # input text feat dim
            max_seq_len,  # max sequence length (used for training)
            summary_len,
            max_buffer_len_factor,  # max buffer size (defined a factor of max_seq_len)
            n_head,  # number of heads for self-attention in transformer
            n_mha_win_size,  # window size for self attention; -1 to use full seq
            embd_kernel_size,  # kernel size of the embedding network
            embd_dim,  # output feat channel of the embedding network
            embd_with_ln,  # attach layernorm to embedding network
            fpn_dim,  # feature dim on FPN
            fpn_with_ln,  # if to apply layer norm at the end of fpn
            fpn_start_level,  # start level of fpn
            head_dim,  # feature dim for head
            regression_range,  # regression range on each level of FPN
            head_num_layers,  # number of layers in the head (including the classifier)
            head_kernel_size,  # kernel size for reg/cls heads
            head_with_ln,  # attach layernorm to reg/cls heads
            use_abs_pe,  # if to use abs position encoding
            use_rel_pe,  # if to use rel position encoding
            num_classes,  # number of action classes
            train_cfg,  # other cfg for training
            test_cfg,  # other cfg for testing
            consecutive_masking,
            narr_decoder,
            narr_decoder_cfg,
            num_summarizer_blocks,
            summary_resolution,
            summary_of_summary,
            single_token_language,
            anchor_localization,
            anchor_localization_cfg,
            consecutive_masking_cfg,
            provide_visual_info,
            provide_visual_info_cfg,
            contrastive_learning,
            contrastive_learning_cfg,
            predict_visual_info,
            predict_visual_info_cfg,
            localization_refinement,
            localization_refinement_cfg,
            fb_drop_nlq_ratio,
            train_feedback_only,
            video_mask_ratio,
            falm_scale,
            localization_loss_weight,
            falm_cfg,
            falm_resume_path=None,
            use_falm=True,
    ):
        super().__init__()
        self.input_txt_dim = input_txt_dim
        self.summary_len = summary_len
        # re-distribute params to backbone / neck / head
        self.reg_range = regression_range
        if summary_len != -1:
            self.fpn_levels = backbone_arch[-1]
        else:
            self.fpn_levels = backbone_arch[-1] + 1
        self.fpn_strides = [scale_factor ** i for i in range(self.fpn_levels)]
        assert len(self.fpn_strides) == len(self.reg_range), (self.fpn_strides, self.reg_range)
        self.scale_factor = scale_factor
        # #classes = num_classes + 1 (background) with last category as background
        # e.g., num_classes = 10 -> 0, 1, ..., 9 as actions, 10 as background
        self.num_classes = num_classes

        # check the feature pyramid and local attention window size
        self.max_seq_len = max_seq_len
        self.mha_win_size = n_mha_win_size

        # Vid mask ratio
        self.video_mask_ratio = video_mask_ratio
        # Drop nlq ratio when using feedback
        self.fb_drop_nlq_ratio = fb_drop_nlq_ratio

        # max_div_factor = 1
        # for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
        #     stride = s * (w // 2) * 2 if w > 1 else s
        #     assert max_seq_len % stride == 0, "max_seq_len %d must be divisible by fpn stride and window size %d" % (
        #         max_seq_len, stride)
        #     if max_div_factor < stride:
        #         max_div_factor = stride
        # self.max_div_factor = max_div_factor

        # training time config
        self.train_center_sample = train_cfg['center_sample']
        assert self.train_center_sample in ['radius', 'none']
        self.train_center_sample_radius = train_cfg['center_sample_radius']
        self.train_loss_weight = train_cfg['loss_weight']
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.train_droppath = train_cfg['droppath']
        self.train_attn_dropout = train_cfg['attn_dropout']
        self.train_label_smoothing = train_cfg['label_smoothing']
        self.negative_sample_ratio = train_cfg['negative_sample']
        self.negative_loss_weight = train_cfg['negative_loss_weight']

        # test time config
        self.test_pre_nms_thresh = test_cfg['pre_nms_thresh']
        self.test_pre_nms_topk = test_cfg['pre_nms_topk']
        self.test_iou_threshold = test_cfg['iou_threshold']
        self.test_min_score = test_cfg['min_score']
        self.test_max_seg_num = test_cfg['max_seg_num']
        self.test_nms_method = test_cfg['nms_method']
        assert self.test_nms_method in ['soft', 'hard', 'none']
        self.test_duration_thresh = test_cfg['duration_thresh']
        self.test_multiclass_nms = test_cfg['multiclass_nms']
        self.test_nms_sigma = test_cfg['nms_sigma']
        self.test_voting_thresh = test_cfg['voting_thresh']

        self.consecutive_masking = consecutive_masking
        if self.consecutive_masking:
            mask_range = consecutive_masking_cfg['consecutive_mask_range']
            droprate = consecutive_masking_cfg['droprate']
            self.consecutive_mask = ConsecutiveMasking(input_vid_dim, mask_range, droprate)

        
        self.gt_provided = provide_visual_info
        self.predict_context_action = predict_visual_info
        # either provide visual info or predict visual info
        if self.gt_provided or self.predict_context_action:
            assert self.gt_provided != self.predict_context_action
            if self.gt_provided:
                assert provide_visual_info_cfg['context_action_order'] >= 0

            if self.predict_context_action:
                assert predict_visual_info_cfg['context_action_order'] >= -1
                self.predict_context_action_order = predict_visual_info_cfg['context_action_order']
                self.vis_info_loss_type = predict_visual_info_cfg['loss_type']
                self.vis_info_loss = "vis" in self.vis_info_loss_type
                self.vis_info_loss_weight = predict_visual_info_cfg['vis_info_loss_weight']
                self.contrastive_temperature = predict_visual_info_cfg['temperature']
                self.loss_across_batch = predict_visual_info_cfg['across_batch']
                self.loss_across_pred = predict_visual_info_cfg['across_pred']
                self.use_duplicate_mask = predict_visual_info_cfg['use_duplicate_mask']
                self.order_loss = predict_visual_info_cfg['order_loss']
                order_num_layers = predict_visual_info_cfg['order_num_layers']
                self.order_loss_weight = predict_visual_info_cfg['order_loss_weight']
                self.order_distance = predict_visual_info_cfg['order_distance']
                self.l2_loss = predict_visual_info_cfg['l2_loss']
                self.l2_loss_weight = predict_visual_info_cfg['l2_loss_weight']
                self.lang_info_loss = "lang" in self.vis_info_loss_type
                self.lang_info_loss_weight = predict_visual_info_cfg['lang_info_loss_weight']
                self.uniform_vid_sample = predict_visual_info_cfg['uniform_vid_sample']
                self.predict_context_action_resume = True if 'resume_path' in predict_visual_info_cfg else False


        self.contrastive_learning = contrastive_learning
        if self.contrastive_learning:
            self.contrastive_learning_cfg = contrastive_learning_cfg
            self.contrastive_loss_weight = contrastive_learning_cfg['contrastive_loss_weight']
            self.background_ratio = contrastive_learning_cfg['background_ratio']
            self.temperature = contrastive_learning_cfg['temperature']

        # self.vid_mask_token = nn.Parameter(torch.empty(input_vid_dim))
        # torch.nn.init.normal_(self.vid_mask_token, mean=0.0, std=0.02)

        # self.txt_separator_token = nn.Parameter(torch.empty(1, input_txt_dim))
        # torch.nn.init.normal_(self.txt_separator_token, mean=0.0, std=0.02)
        # self.query_txt_token = nn.Parameter(torch.empty(1, input_txt_dim))
        # torch.nn.init.normal_(self.query_txt_token, mean=0.0, std=0.02)
        # self.feedback_txt_token = nn.Parameter(torch.empty(1, input_txt_dim))
        # torch.nn.init.normal_(self.feedback_txt_token, mean=0.0, std=0.02)
        # self.pred_timestamp_token = nn.Parameter(torch.empty(1, input_txt_dim))
        # torch.nn.init.normal_(self.pred_timestamp_token, mean=0.0, std=0.02)
        self.falm_scaler = nn.Parameter(torch.empty(1))
        self.falm_bias = nn.Parameter(torch.empty(1))
        torch.nn.init.constant_(self.falm_scaler, 1.0)
        torch.nn.init.constant_(self.falm_bias, 0.0)
        pos_embd = get_sinusoid_encoding(self.max_seq_len, input_txt_dim) / (input_txt_dim ** 0.5)
        self.vis_lin = nn.Linear(input_vid_dim, input_txt_dim)
        self.center_width_lin = nn.Linear(2, input_txt_dim)
        self.register_buffer("pos_embd", pos_embd, persistent=False)

        # backbone network: conv + transformer
        assert backbone_type in ['convTransformer', 'flashTransformer']
        self.flash_attn = True if backbone_type == 'flashTransformer' else False
        self.backbone = make_backbone(
            backbone_type,
            **{
                'n_vid_in': input_vid_dim,
                'n_txt_in': input_txt_dim,
                'n_embd': embd_dim,
                'n_head': n_head,
                'n_embd_ks': embd_kernel_size,
                'max_len': max_seq_len,
                'arch': backbone_arch,
                'mha_win_size': self.mha_win_size,
                'scale_factor': scale_factor,
                'with_ln': embd_with_ln,
                'attn_pdrop': self.train_attn_dropout,
                'proj_pdrop': self.train_dropout,
                'path_pdrop': self.train_droppath,
                'use_abs_pe': use_abs_pe,
                'use_rel_pe': use_rel_pe,
                'gt_provided': self.gt_provided,
                'gt_provided_cfg': provide_visual_info_cfg if self.gt_provided else None,
                'predict_context_action_cfg': predict_visual_info_cfg,
                'falm_scale': falm_scale,
            }
        )

        # fpn network: identity
        self.fpn_dim = fpn_dim
        assert fpn_type == 'identity'
        self.neck = make_neck(
            fpn_type,
            **{
                'in_channels': [embd_dim] * (self.fpn_levels),
                'out_channel': fpn_dim,
                'scale_factor': scale_factor,
                'start_level': fpn_start_level,
                'with_ln': fpn_with_ln
            }
        )

        # narration decoder
        self.narr_decoder = None
        if narr_decoder:
            assert narr_decoder_cfg is not None
            self.narr_loss_weight = narr_decoder_cfg['narr_loss_weight']
            """
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
            """
            self.narr_decoder = MaskedNarrationDecoder(
                num_layers=narr_decoder_cfg['num_layers'],
                txt_dim=input_txt_dim,
                n_embd=embd_dim,
                n_head=narr_decoder_cfg['n_head'],
                n_ds_strides=(1, 1),
                n_out=None,
                n_hidden=None,
                act_layer=nn.GELU,
                attn_pdrop=narr_decoder_cfg['dropout'],
                proj_pdrop=narr_decoder_cfg['dropout'],
                path_pdrop=narr_decoder_cfg['droppath'],
                mha_win_size=narr_decoder_cfg['n_mha_win_size'],
                use_rel_pe=narr_decoder_cfg['use_rel_pe'],
                resolution=narr_decoder_cfg['resolution'],
            )

        if self.summary_len != -1:
            loc_len = summary_len * max_buffer_len_factor
        else:
            loc_len = max_seq_len * max_buffer_len_factor
        # location generator: points
        self.point_generator = make_generator(
            'point',
            **{
                'max_seq_len': loc_len,
                'fpn_strides': self.fpn_strides,
                'regression_range': self.reg_range
            }
        )
        self.anchor_localization = anchor_localization
        if anchor_localization:
            self.anchor_localization_cfg = anchor_localization_cfg
            self.anchor_localization_weight = anchor_localization_cfg['anchor_localization_weight']
            anchor_num_layers = anchor_localization_cfg['num_layers']
            anchor_dropout = anchor_localization_cfg['dropout']
            self.after_fpn_net = FPN_FeedForward(fpn_levels=self.fpn_levels, num_layers=anchor_num_layers, in_dim=fpn_dim, hidden_dim=fpn_dim, out_dim=fpn_dim, dropout=anchor_dropout)
            self.before_fpn_net = FPN_FeedForward(fpn_levels=self.fpn_levels, num_layers=anchor_num_layers, in_dim=fpn_dim, hidden_dim=fpn_dim, out_dim=fpn_dim, dropout=anchor_dropout)

        assert self.num_classes > 0
        self.head_dim = head_dim
        self.head_kernel_size = head_kernel_size
        self.head_with_ln = head_with_ln
        self.head_num_layers = head_num_layers
        self.empty_cls = train_cfg['head_empty_cls']
        self.localization_refinement = localization_refinement
        if localization_refinement:
            self.localization_refinement_cfg = localization_refinement_cfg
            self.direction_sample_radius = localization_refinement_cfg['direction_sample_radius']
            self.localization_refinement_weight = localization_refinement_cfg['localization_refinement_weight']
            refine_num_layers = localization_refinement_cfg['num_layers']
            refine_dropout = localization_refinement_cfg['dropout']
            refine_dim = localization_refinement_cfg['dim']
            remain_dim = refine_dim - 5
            points_pos = get_sinusoid_encoding(self.max_seq_len, remain_dim) / (remain_dim ** 0.5) 
            self.register_buffer("points_pos", points_pos, persistent=False) 
            self.level_token = nn.Parameter(torch.empty(self.fpn_levels, remain_dim))
            torch.nn.init.normal_(self.level_token, mean=0.0, std=0.02)
            self.pre_refine_class_head = PtConvClsHead(
                fpn_dim, head_dim, 3,
                kernel_size=head_kernel_size,
                prior_prob=self.train_cls_prior_prob,
                with_ln=head_with_ln,
                num_layers=1,
                empty_cls=train_cfg['head_empty_cls']
            )
            self.pre_refine_offset_head = PtConvRegHead(
                fpn_dim, head_dim, self.fpn_levels,
                kernel_size=head_kernel_size,
                act_layer=nn.ReLU,
                with_ln=head_with_ln,
                num_layers=1
            )
            self.refine_transformers = nn.ModuleList([
                TransformerBlock(
                    n_embd=refine_dim,
                    n_head=4,
                    proj_pdrop=refine_dropout,
                    path_pdrop=0.1,
                    mha_win_size=-1,
                ) for _ in range(refine_num_layers)
            ])
            self.cls_head = MaskedConv1D(refine_dim, self.num_classes, 1)
            bias_value = -(math.log((1 - self.train_cls_prior_prob) / self.train_cls_prior_prob))
            torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)
            self.reg_head = MaskedConv1D(refine_dim, 2, 1)
        else:
            # classification and regression heads
            self.cls_head = PtConvClsHead(
                fpn_dim, head_dim, self.num_classes,
                kernel_size=head_kernel_size,
                prior_prob=self.train_cls_prior_prob,
                with_ln=head_with_ln,
                num_layers=head_num_layers,
                empty_cls=train_cfg['head_empty_cls']
            )
            self.reg_head = PtConvRegHead(
                fpn_dim, head_dim, len(self.fpn_strides),
                kernel_size=head_kernel_size,
                act_layer=nn.ReLU,
                with_ln=head_with_ln,
                num_layers=head_num_layers
            )

        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.loss_normalizer =  {s: train_cfg['init_loss_norm'] for s in ["main", "after", "before", "refine"]}
        self.loss_normalizer_momentum = 0.9

        self.localization_loss_weight = localization_loss_weight
        self.use_falm = use_falm
        self.falm_scale = falm_scale

        if use_falm and falm_cfg is not None:
            self.falm = FALM(**falm_cfg)
            if falm_resume_path is not None:
                missing, unexpected = self.falm.load_state_dict(
                    torch.load(falm_resume_path, map_location="cpu"), strict=False
                )
                print(f"Loaded FALM from {falm_resume_path}")
                if missing:
                    print(f"  Missing keys: {missing}")
                if unexpected:
                    print(f"  Unexpected keys: {unexpected}")
        else:
            self.falm = None


    def set_localization_loss_weight(self, weight):
        self.localization_loss_weight = weight
    
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad_(True)

    @property
    def device(self):
        try:
            return int(os.environ["LOCAL_RANK"])
        except:
            return torch.device("cuda:0")

    def pad_seq_with_mask(self, sequences, max_length=None, should_filter=False):
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
                add_feature = torch.zeros(add_length, feature_length, dtype=torch.float32, device=self.device)
                seq_ = torch.cat([seq, add_feature], dim=0)
            else:
                seq_ = seq
            sequence_padded.append(seq_)
        mask = torch.zeros(len(sequences), max_length, dtype=torch.bool, device=self.device)
        for i, l in enumerate(sequence_length):
            mask[i, :l] = True
        sequence_padded = torch.stack(sequence_padded, dim=0)
        sequence_length = torch.tensor(sequence_length, dtype=torch.long)
        idx = torch.tensor(idx, dtype=torch.long)
        return sequence_padded, mask, sequence_length, idx

    def _gaussian_blur(self, x: torch.Tensor, k: int = 15, sigma: float = 3.0):
        """
        Apply 1D Gaussian blur to a tensor of shape [B, T, 1].

        Args:
            x: torch.Tensor of shape [B, T, 1]
            k: kernel size (odd integer)
            sigma: standard deviation for Gaussian kernel
        """
        assert k % 2 == 1, "Kernel size must be odd"

        device = x.device
        half = k // 2
        t = torch.arange(-half, half + 1, device=device, dtype=x.dtype)
        kernel = torch.exp(-(t**2) / (2 * sigma**2))
        kernel /= kernel.sum()

        # conv1d expects shape [B, C, T]
        x = x.permute(0, 2, 1)  # [B, 1, T]

        # Make kernel shape [C_out, C_in, k]
        kernel = kernel.view(1, 1, -1)

        # Apply 1D convolution with padding='same'
        x_blurred = F.conv1d(x, kernel, padding=half, groups=1)

        # Restore to original shape [B, T, 1]
        return x_blurred.permute(0, 2, 1)


    def forward(self, video_list, get_losses=True, get_preds=False, get_hit=False, dist_group=None, wo_postprocess=False):
        # video_list:  <class 'list'> 1
        # video_list[0] <class 'dict'>
        
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        """
        'video_id' : video_id,
        'video_feats': video_feats,
        'v_mask': v_mask,
        'query_text_feats': query_text_feats,
        'q_mask': q_mask,
        'masked_feats': masked_feats,
        'masked_mask': masked_mask,
        'query_starts': query_starts,
        'masked_batch_idx': batch_idx,
        'sampled_rel_pos': sampled_rel_pos,
        'masked_rel_pos': masked_rel_pos,
        'segments' : segments,
        'one_hot_labels' : one_hot_labels,
        'fps' : fps,
        'vid_lens' : vid_lens,
        'duration' : duration,
        'feat_stride' : feat_stride,
        'feat_num_frames': feat_num_frames,
        'query_id': query_id,
        "is_negative": is_negative,
        'after_segments': after_segments,
        'after_idxs': after_idxs,
        'before_segments': before_segments,
        'before_idxs': before_idxs
        """

        src_vid, src_vid_mask = video_list["video_feats"], video_list["v_mask"]
        vid_lens = video_list["vid_lens"]
        src_txt = video_list["query_text_feats"]
        src_feedback = video_list["feedback_feats"]
        pred_idx = video_list["pred_idx"]
        pred_timestamp = video_list["pred_timestamp"]

        saliency_labels = video_list["saliency_labels"]

        for idx in range(len(saliency_labels)):
            saliency_labels[idx] = saliency_labels[idx].to(self.device).view(-1, 1)
        saliency_labels, _, _, _ = self.pad_seq_with_mask(saliency_labels, max_length=self.max_seq_len)

        
        src_vid = src_vid.to(self.device)
        src_vid_mask = src_vid_mask.to(self.device)
        vid_lens = vid_lens.to(self.device)

        src_txt_clone = [src_txt[i].clone() for i in range(len(src_txt))]

        feedback_idxs = torch.tensor([i for i in range(len(src_txt)) if pred_idx[i] is not None], dtype=torch.long, device=self.device)

        # segments = video_list["segments"]
        for i in range(len(src_txt_clone)):
            src_txt_clone[i] = src_txt_clone[i].to(self.device)
            if pred_idx[i] is not None:
                timestamp = pred_timestamp[i]
                # timestamp = segments[i][0]
                # print(timestamp)
                start = torch.clamp(timestamp[0], min=0, max=vid_lens[i]-1).floor().long()
                end = torch.clamp(timestamp[1], min=1, max=vid_lens[i]).ceil().long()
                if start == end:
                    end = start + 1
                pred_pe =  self.pos_embd[0,:,start:end].permute(1,0)
                vids = src_vid[i, :, start:end].permute(1,0)
                vids = self.vis_lin(vids)
                vids = vids+pred_pe
                vid_mean = torch.mean(vids, dim=0, keepdim=True)
                s_e_pe = self.pos_embd[0,:,[start, end]].permute(1,0)
                s_e_vid = src_vid[i, :, [start, end]].permute(1,0)
                s_e_vid = self.vis_lin(s_e_vid)
                s_e_vid = s_e_vid+s_e_pe
                pred_tokens = torch.cat([vid_mean, s_e_vid], dim=0)
                if src_feedback[i] is None:
                    src_txt_clone[i] =  torch.cat([src_txt_clone[i][:pred_idx[i][0]+1], src_txt_clone[i][pred_idx[i][1]+1:], pred_tokens], dim=0)
                else:
                    src_feedback[i] = src_feedback[i].to(self.device)
                    # print("pred_idx[i]: ", pred_idx[i])
                    # print("src_feedback[i].shape: ", src_feedback[i].shape)
                    # print("pred_tokens.shape: ", pred_tokens.shape)
                    # print("src_txt[i].shape: ", src_txt[i].shape)
                    src_txt_clone[i] =  torch.cat([src_txt_clone[i], src_feedback[i][:pred_idx[i][0]+1], src_feedback[i][pred_idx[i][1]+1:], pred_tokens], dim=0)
                    # print("final src_txt[i].shape: ", src_txt[i].shape)
                # add positional embedding for the text but not pred tokens
                
                pos_txt_embed = self.pos_embd[0,:,:src_txt_clone[i].shape[0]-pred_tokens.shape[0]].permute(1,0)
                src_txt_clone[i][:pos_txt_embed.shape[0]] += pos_txt_embed
             
        if self.training and self.video_mask_ratio > 0:
            masks = []
            for i in range(len(src_vid)):
                # sample based on vid_lens
                curr_vid_len = vid_lens[i]
                mask = torch.rand(src_vid.shape[2]) < self.video_mask_ratio
                mask[curr_vid_len:] = False
                masks.append(mask)
            
            masks = torch.stack(masks, dim=0).unsqueeze(1)
            masks = masks.to(self.device)
            
            add_tokens = self.vid_mask_token.unsqueeze(1).unsqueeze(0).expand(src_vid.shape[0], -1,  src_vid.shape[2])
            src_vid = torch.where(masks, add_tokens, src_vid)

        src_txt, src_txt_mask, _, _ = self.pad_seq_with_mask(src_txt_clone)
        src_txt = src_txt.permute(0, 2, 1)
        src_txt_mask = src_txt_mask.bool()
        src_txt_mask = src_txt_mask.unsqueeze(1)

        src_txt = src_txt.to(self.device)
        src_txt_mask = src_txt_mask.to(self.device)
        
        is_negative = video_list["is_negative"].to(self.device)

        is_positive = (is_negative < 0.5)
        after_idxs = video_list.get("after_idxs", None)
        after_segments = video_list.get("after_segments", None)
        after_one_hot_labels = video_list.get("after_one_hot_labels", None)
        before_idxs = video_list.get("before_idxs", None)
        before_segments = video_list.get("before_segments", None)
        before_one_hot_labels = video_list.get("before_one_hot_labels", None)

        # if self.flash_attn:
        #     # Convert the input to bfloat16
        #     src_vid = src_vid.to(torch.bfloat16)
        #     src_vid_mask = src_vid_mask.to(torch.bfloat16)
        #     src_txt = src_txt.to(torch.bfloat16)
        #     src_txt_mask = src_txt_mask.to(torch.bfloat16)
        #     # is_negative = is_negative.to(torch.bfloat16)
        #     # is_

        # weight_scale = is_negative.masked_fill(is_negative == 1.0, self.negative_loss_weight).masked_fill(is_negative == 0.0, 1.0)

        # masked_narr_feats, masked_narr_mask = video_list.get("masked_feats", None), video_list.get("masked_mask", None)
        # narr_batch_idx = video_list.get("masked_batch_idx", None)
        # query_starts = video_list.get("query_starts", None)
        # sampled_rel_pos = video_list.get("sampled_rel_pos", None)
        # masked_rel_pos = video_list.get("masked_rel_pos", None)

        # if masked_narr_feats is not None:
        #     masked_narr_feats, masked_narr_mask = masked_narr_feats.to(self.device), masked_narr_mask.to(self.device)
        #     narr_batch_idx = narr_batch_idx.to(self.device)
        #     query_starts = query_starts
        #     sampled_rel_pos = sampled_rel_pos
        #     masked_rel_pos = masked_rel_pos

        # forward the network (backbone -> neck -> heads)

        if self.consecutive_masking:
            src_vid = self.consecutive_mask(src_vid, vid_lens)
        # if self.gt_provided:
        #     feats, masks, q_feats, q_mask, context_action_lang_only, iterated_action_tokens_vis_pred, sampled_idxs = self.backbone(src_vid, src_vid_mask, src_txt, src_txt_mask, vid_lens, context_action_vis_gt=context_action_vis_gt)
        # else:
        
        
        if self.use_falm and self.falm is not None:
            _, falm_results = self.falm(video_list, get_preds=True, get_losses=False)
            pred_saliency = falm_results["pred_saliency"]
            falm_feats = falm_results["falm_feats"]
            if feedback_idxs.numel() > 0:
                pred_saliency = self.falm_scaler * pred_saliency + self.falm_bias
                pred_saliency = torch.clamp(pred_saliency, min=0.0, max=1.0)
        else:
            pred_saliency = None
            falm_feats = None

        


        feats, masks, q_feats, q_mask, context_action_lang_only, context_narration_lang_only= self.backbone(src_vid, src_vid_mask, src_txt, src_txt_mask, vid_lens, feedback_idxs, pred_saliency, falm_feats)
        # print("len(feats): ",len(feats))
        # feats:  <class 'tuple'> 6
        # 0 item_feats:  <class 'torch.Tensor'> torch.Size([1, 384, 2304])
        # 1 item_feats:  <class 'torch.Tensor'> torch.Size([1, 384, 1152])
        # 2 item_feats:  <class 'torch.Tensor'> torch.Size([1, 384, 576])
        # 3 item_feats:  <class 'torch.Tensor'> torch.Size([1, 384, 288])
        # 4 item_feats:  <class 'torch.Tensor'> torch.Size([1, 384, 144])
        # 5 item_feats:  <class 'torch.Tensor'> torch.Size([1, 384, 72])
        
        fpn_feats, fpn_masks = self.neck(feats, masks)

        # compute the point coordinate along the FPN
        # this is used for computing the GT or decode the final results
        # points: List[T x 4] with length = # fpn levels
        # (shared across all samples in the mini-batch)
        points = self.point_generator(fpn_feats)

        assert self.num_classes > 0

        # if self.localization_refinement:
        #     concat_points = torch.cat(points, dim=0)
        #     point_idxs = concat_points[:, 0].long()
        #     pre_refine_out_cls_logits = self.pre_refine_class_head(fpn_feats, fpn_masks)
        #     pre_refine_out_offsets = self.pre_refine_offset_head(fpn_feats, fpn_masks)

        #     flat_logits = torch.cat(pre_refine_out_cls_logits, dim=2)
        #     flat_offsets = torch.cat(pre_refine_out_offsets, dim=2)
        #     assert flat_logits.size(2) == len(point_idxs), f"{flat_logits.size(2)} != {len(point_idxs)}"
        #     pos_embed = self.points_pos[:,:,point_idxs] # 1, C, T
        #     level_embed = torch.cat([self.level_token[None, i].repeat(2304//2**i, 1) for i in range(self.fpn_levels)],dim=0) # T, C
        #     level_embed = level_embed.unsqueeze(0).permute(0, 2, 1) # 1, C, T
        #     assert pos_embed.size() == level_embed.size(), f"{pos_embed.size()} != {level_embed.size()}"
        #     pos_level_embed = pos_embed + level_embed
        #     refine_feats = torch.cat([flat_logits, flat_offsets, pos_level_embed.expand(flat_logits.size(0), -1, -1)], dim=1) # B, C, T
        #     refine_masks = torch.cat(fpn_masks, dim=2) # B, 1, T
        #     for idx in range(len(self.refine_transformers)):
        #         refine_feats, _ = self.refine_transformers[idx](refine_feats, refine_masks)
        #     flat_out_cls_logits, _ = self.cls_head(refine_feats, refine_masks)
        #     flat_out_offsets, _ = self.reg_head(refine_feats, refine_masks)
        #     flat_out_offsets = F.relu(flat_out_offsets)

        #     out_cls_logits = [] # Reconstruct FPN level outputs
        #     out_offsets = []
        #     curr_pos = 0
        #     for i in range(self.fpn_levels):
        #         out_cls_logits.append(flat_out_cls_logits[:, :, curr_pos:curr_pos+(self.max_seq_len//self.fpn_strides[i])])
        #         out_offsets.append(flat_out_offsets[:, :, curr_pos:curr_pos+(self.max_seq_len//self.fpn_strides[i])])
        #         curr_pos += (self.max_seq_len//self.fpn_strides[i])
            
        #     pre_refine_out_cls_logits = [x.permute(0, 2, 1) for x in pre_refine_out_cls_logits]
        #     pre_refine_out_offsets = [x.permute(0, 2, 1) for x in pre_refine_out_offsets]
        # else:
        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        # out_offset: List[B, 2, T_i]
        out_offsets = self.reg_head(fpn_feats, fpn_masks)
        
        # permute the outputs
        # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]

        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        main_fpn_masks = [x.squeeze(1) for x in fpn_masks]

        # return loss during training
        losses = None
        results = None
        if get_losses:
            assert "segments" in video_list, "GT action labels does not exist"
            assert "one_hot_labels" in video_list, "GT action labels does not exist"
            # masked_narr_is_valid = masked_narr_feats is not None and masked_narr_feats.size(0) != 0 and self.narr_decoder
            # pred_narr_feats = None
            # if masked_narr_is_valid:
            #     narr_feats = q_feats[narr_batch_idx]
            #     merged_query, sampled_query_mask = self.merge_query_tokens(narr_feats, query_starts)
            #     pred_narr_feats = self.narr_decoder(
            #         masked_narr_feats.shape[:2], masked_narr_mask, merged_query, sampled_query_mask, sampled_rel_pos, masked_rel_pos, modality
            #     ) # B, T, C

            # generate segment/lable List[N x 2] / List[N] with length = B
            segments = video_list["segments"]
            segments_list = [video_list["segments"]]
            one_hot_labels = [video_list["one_hot_labels"]]
            if after_segments is not None:
                segments_list.append(after_segments)
                one_hot_labels.append(after_one_hot_labels)
            if before_segments is not None:
                segments_list.append(before_segments)
                one_hot_labels.append(before_one_hot_labels)
            cls_labels = []
            offsets = []
            for segments, one_hot_labels in zip(segments_list, one_hot_labels):
                assert segments[0] is not None, "GT action labels does not exist"
                gt_segments = [x.to(self.device) for x in segments]
                
                assert one_hot_labels[0] is not None, "GT action labels does not exist"
                #gt_labels = [x['one_hot_labels'].to(self.device) for x in video_list]
                gt_labels = [x.to(self.device) for x in one_hot_labels]

                # compute the gt labels for cls & reg
                # list of prediction targets
                gt_cls_labels, gt_offsets = self.label_points(
                    points, gt_segments, gt_labels, self.num_classes)
                cls_labels.append(gt_cls_labels)
                offsets.append(gt_offsets)
            
            
            main_gt_cls_labels, main_gt_offsets = cls_labels[0], offsets[0]
            
            # compute the loss and return
            losses = self.losses(
                main_fpn_masks,
                out_cls_logits, out_offsets,
                main_gt_cls_labels, main_gt_offsets,
                loss_type="main"
                 # pred_narr_feats, masked_narr_mask, masked_narr_feats, weight_scale
            )

            losses["final_loss"] = losses["final_loss"] * self.localization_loss_weight
                            
                # span_loss = self.saliency_loss(span.permute(0,2,1), span_mask, span_labels)
                # losses["span_loss"] = span_loss
                # losses["final_loss"] += span_loss * self.span_loss_weight
                # losses["final_loss"] = saliency_loss * self.saliency_loss_weight

            # if self.localization_refinement:
            #     gt_segments = [x.to(self.device) for x in video_list["segments"]]

            #     pre_refine_cls_labels, pre_refine_offsets, pre_refine_reg_mask = self.label_directional_points(points, gt_segments)

            #     refine_loss = self.losses(
            #         main_fpn_masks,
            #         pre_refine_out_cls_logits, pre_refine_out_offsets,
            #         pre_refine_cls_labels, pre_refine_offsets,
            #         loss_type="refine", reg_mask=pre_refine_reg_mask
            #     )

            #     for k, v in refine_loss.items():
            #         if k != "final_loss":
            #             losses[f"refine_{k}"] = v
            #     losses["final_loss"] += refine_loss["final_loss"] * self.localization_refinement_weight

            # if self.anchor_localization:
            #     nxt_index = 1
            #     if after_segments is not None:
            #         after_gt_cls_labels, after_gt_offsets = cls_labels[nxt_index], offsets[nxt_index]
            #         after_fpn_feats = [fpn_feats[i][after_idxs] for i in range(len(fpn_feats))]
            #         after_fpn_masks = [fpn_masks[i][after_idxs] for i in range(len(fpn_masks))]
            #         after_fpn_feats, after_fpn_masks = self.after_fpn_net(after_fpn_feats, after_fpn_masks)
            #         after_out_cls_logits = self.cls_head(after_fpn_feats, after_fpn_masks)
            #         after_out_offsets = self.reg_head(after_fpn_feats, after_fpn_masks)
            #         after_out_cls_logits = [x.permute(0, 2, 1) for x in after_out_cls_logits]
            #         after_out_offsets = [x.permute(0, 2, 1) for x in after_out_offsets]
            #         after_fpn_masks = [x.squeeze(1) for x in after_fpn_masks]
            #         losses_after = self.losses(
            #             after_fpn_masks,
            #             after_out_cls_logits, after_out_offsets,
            #             after_gt_cls_labels, after_gt_offsets, 
            #             loss_type="after"
            #         )
            #         for k, v in losses_after.items():
            #             if k != "final_loss":
            #                 losses[f"after_{k}"] = v
            #         losses["final_loss"] += losses_after["final_loss"] * self.anchor_localization_weight
            #         nxt_index += 1
                
            #     if before_segments is not None:
            #         before_gt_cls_labels, before_gt_offsets = cls_labels[nxt_index], offsets[nxt_index]
            #         before_fpn_feats = [fpn_feats[i][before_idxs] for i in range(len(fpn_feats))]
            #         before_fpn_masks = [fpn_masks[i][before_idxs] for i in range(len(fpn_masks))]
            #         before_fpn_feats, before_fpn_masks = self.before_fpn_net(before_fpn_feats, before_fpn_masks)
            #         before_out_cls_logits = self.cls_head(before_fpn_feats, before_fpn_masks)
            #         before_out_offsets = self.reg_head(before_fpn_feats, before_fpn_masks)
            #         before_out_cls_logits = [x.permute(0, 2, 1) for x in before_out_cls_logits]
            #         before_out_offsets = [x.permute(0, 2, 1) for x in before_out_offsets]
            #         before_fpn_masks = [x.squeeze(1) for x in before_fpn_masks]
            #         losses_before = self.losses(
            #             before_fpn_masks,
            #             before_out_cls_logits, before_out_offsets,
            #             before_gt_cls_labels, before_gt_offsets, 
            #             loss_type="before"
            #         )
            #         for k, v in losses_before.items():
            #             if k != "final_loss":
            #                 losses[f"before_{k}"] = v
            #         losses["final_loss"] += losses_before["final_loss"] * self.anchor_localization_weight
            #         nxt_index += 1

            # if self.contrastive_learning:
            #     assert "negative_segments" in video_list, "negative segments does not exist"
            #     contrast_feats, contrast_labels = self.extract_contrastive_features(feats[0], video_list["segments"], video_list["negative_segments"], after_segments, after_idxs, before_segments, before_idxs)
            #     if contrast_feats is not None:
            #         losses["contrastive_loss"] = contrastive_loss(contrast_feats, contrast_labels, self.temperature, dist_group)
            #         losses["final_loss"] += losses["contrastive_loss"] * self.contrastive_loss_weight


            # if self.predict_context_action and not self.predict_context_action_order == -1 and not self.predict_context_action_resume:
            #     vis_info_loss = self.get_vis_info_loss(context_action_lang_only, context_action_vis_gt, context_action_vis_gt_mask)
            #     if vis_info_loss is not None:
            #         losses["vis_info_loss"] = vis_info_loss
            #         losses["final_loss"] += vis_info_loss * self.vis_info_loss_weight 
                

            # if pred_masked_narr is not None:
            #     # masked_narr_mask: B, 1, T
            #     # pred_masked_narr: B, T, C
            #     # gt_masked_narr: B, T, C
            #     masked_narr_mask = masked_narr_mask.squeeze(1)
            #     B, T, C = pred_masked_narr.shape
            #     pred_masked_narr = pred_masked_narr.permute(0, 2, 1).contiguous().view(B*T, C)
            #     gt_masked_narr = gt_masked_narr.view(B*T, C)
            #     masked_narr_mask = masked_narr_mask.view(B*T, 1)
            #     narr_loss = nn.functional.mse_loss(pred_masked_narr, gt_masked_narr, reduction="none")
            #     narr_loss = torch.sum(narr_loss * masked_narr_mask) / (torch.sum(masked_narr_mask) + 1e-12)
            #     narr_loss = narr_loss / self.input_txt_dim
            #     losses["narr_loss"] = narr_loss
            #     losses["final_loss"] += self.narr_loss_weight * narr_loss
                

        if get_preds:
            # decode the actions (sigmoid / stride, etc)
            if get_hit:
                results = context_action_lang_only, context_narration_lang_only
            else:
                results = self.inference(
                    video_list, points, main_fpn_masks,
                    out_cls_logits, out_offsets, self.num_classes, pred_saliency, wo_postprocess
                )

        return losses, results

    def saliency_loss(self, saliency_pred, saliency_mask, saliency_gt):
        loss = F.binary_cross_entropy(saliency_pred, saliency_gt, reduction='none')

        # print("saliency_pred.shape: ", saliency_pred.shape)
        # print("saliency_mask.shape: ", saliency_mask.shape)
        # print("saliency_gt.shape: ", saliency_gt.shape)
        # print("loss.shape: ", loss.shape)

        # Apply the mask
        loss = loss * saliency_mask
        loss = loss.squeeze(2)

        # Normalize by number of valid (unmasked) elements to avoid bias
        # valid_count = saliency_mask.sum()
        # if valid_count == 0:
        #     return torch.tensor(0.0, device=saliency_pred.device)
        
        loss = loss.sum(dim=1) / (saliency_mask.sum(dim=1) + 1e-12)
        return loss.mean()
    
    def get_vis_info_loss(self, context_action_lang_only, context_action_gt, context_action_gt_mask):
        # context_action_lang_only: B, C, 2 * context_action_order + 1
        # iterated_action_tokens_vis_pred: B, Iterations - 1, C, 2 * context_action_order + 1
        # context_action_gt: B, Iterations, C, 2 * context_action_order + 1
        # context_action_gt_mask: B, 1, 2 * context_action_order + 1
        _, I, _, _ = context_action_gt.size()
        if torch.sum(context_action_gt_mask) == 0:
            return None
        if self.vis_info_loss_type == "mse":
            pred = context_action_lang_only
            gt = context_action_gt[:,0]
            gt_mask = context_action_gt_mask
            curr_loss = nn.functional.mse_loss(pred, gt, reduction="none")  # (B, C, 2 * context_action_order + 1), (B, C, 2 * context_action_order + 1) -> (B, 2 * context_action_order + 1)
            curr_loss = torch.sum(curr_loss * gt_mask) / (torch.sum(gt_mask))
            vis_info_loss = curr_loss
        elif self.vis_info_loss_type == "cosine":    
            context_action_gt = F.normalize(context_action_gt, p=2, dim=2)
            context_action_lang_only = F.normalize(context_action_lang_only, p=2, dim=1)
            vis_info_loss = 0
            context_action_gt_mask = context_action_gt_mask.squeeze(1)
            pred = context_action_lang_only
            gt = context_action_gt[:,0]
            gt_mask = context_action_gt_mask
            curr_loss = 1 - nn.functional.cosine_similarity(pred, gt, dim=1)  # (B, C, 2 * context_action_order + 1), (B, C, 2 * context_action_order + 1) -> (B, 2 * context_action_order + 1)
            curr_loss = torch.sum(curr_loss * gt_mask) / (torch.sum(gt_mask))
            vis_info_loss = curr_loss
        elif self.vis_info_loss_type == "contrastive":
            assert I == 1
            context_action_gt = context_action_gt.squeeze(1)
            vis_info_loss = pred_gt_contrastive_loss(context_action_lang_only, context_action_gt, context_action_gt_mask, self.contrastive_temperature)
        return vis_info_loss
        
    @torch.no_grad()
    def query_feature_extraction(self, video_list, padding_id=0):
        """
            Generate batched features and masks from a list of dict items
        """
        ids = [torch.LongTensor(x['query_words']["input_ids"]) for x in video_list]
        attn_masks = [torch.LongTensor(x['query_words']["attention_mask"]) for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in ids])
        max_len = feats_lens.max(0).values.item()

        # batch input shape B, T
        batch_shape = [len(ids), max_len]
        batched_inputs = ids[0].new_full(batch_shape, padding_id)
        for feat, pad_feat in zip(ids, batched_inputs):
            pad_feat[..., :feat.shape[-1]].copy_(feat)

        # generate the mask
        batched_masks = attn_masks[0].new_full(batch_shape, 0)
        for feat, pad_feat in zip(attn_masks, batched_masks):
            pad_feat[..., :feat.shape[-1]].copy_(feat)
        # push to device
        batched_inputs = batched_inputs.to(self.device)
        batched_masks = batched_masks.to(self.device)

        
        outputs = self.text_model(input_ids=batched_inputs, attention_mask=batched_masks)
        last_state = outputs.last_hidden_state
        last_state = last_state.permute(0, 2, 1).contiguous()

        batched_masks = batched_masks.unsqueeze(1)
        
        #print("last_state: ", last_state.shape)
        #print("batched_masks: ", batched_masks.shape)

        return last_state, batched_masks
    
    # def merge_query_tokens(self, query, query_starts):
    #     merged_queries = []
    #     max_length=0
    #     seq_lens = []
    #     for b, query_start in enumerate(query_starts):
    #         merged_query = query[b, :, query_start]
    #         merged_queries.append(merged_query)
    #         max_length = max(max_length, merged_query.shape[1])
    #         seq_lens.append(merged_query.shape[1])
    #     for i in range(len(merged_queries)):
    #         add_length = max_length - merged_queries[i].shape[1]
    #         if add_length > 0:
    #             add_feature = torch.zeros(merged_queries[i].shape[0], add_length, dtype=torch.float32, device=query.device)
    #             merged_queries[i] = torch.cat([merged_queries[i], add_feature], dim=1)

        mask = torch.zeros((len(seq_lens), max_length), dtype=torch.bool, device=query.device)
        for i, l in enumerate(seq_lens):
            mask[i, :l] = True
        mask = mask.unsqueeze(1)
        return torch.stack(merged_queries, dim=0), mask

    @torch.no_grad()
    def label_directional_points(self, points, gt_segments):
        # concat points on all fpn levels List[T x 4] -> F T x 4
        # This is shared for all samples in the mini-batch
        concat_points = torch.cat(points, dim=0)

        gt_cls, gt_offset, reg_mask = [], [], []
        # loop over each video sample
        for gt_segment in gt_segments:
            cls_targets, reg_targets, r_mask = self.label_directional_points_single_video(
                concat_points, gt_segment
            )
            # "cls_targets: " #points, num_classes
            # "reg_targets: " #points, 2
            # append to list (len = # images, each of size FT x C)
            gt_cls.append(cls_targets)
            gt_offset.append(reg_targets)
            reg_mask.append(r_mask)

        reg_mask = torch.stack(reg_mask, dim=0)

        return gt_cls, gt_offset, reg_mask

    @torch.no_grad()
    def label_directional_points_single_video(self, concat_points, gt_segment):
        # concat_points : F T x 4 (t, regression range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, 3), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets, reg_targets, torch.zeros((num_pts), dtype=torch.bool, device=self.device)

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        centers = (gt_segment[:, 1] + gt_segment[:, 0])/2
        centers = centers[None, :].repeat(num_pts, 1)

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x 2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        reg_targets = torch.stack((left, right), dim=-1)

        if self.train_center_sample == 'radius':
            # center of all segments F T x N
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
            # center sampling based on stride radius
            # compute the new boundaries:
            # concat_points[:, 3] stores the stride
            t_mins = \
                center_pts - concat_points[:, 3, None] * self.train_center_sample_radius
            t_maxs = \
                center_pts + concat_points[:, 3, None] * self.train_center_sample_radius


            gt_segs_width = (gt_segment[:,1] - gt_segment[:,0])/2

            dir_min = center_pts - gt_segs_width * self.direction_sample_radius
            dir_max = center_pts + gt_segs_width * self.direction_sample_radius

            total_gt_segs_bound = torch.stack((dir_min, dir_max), dim=-1)
            right_dir_segs_bound = torch.stack((dir_min, center_pts), dim=-1)
            left_dir_segs_bound = torch.stack((center_pts, dir_max), dim=-1)

            points_expanded = concat_points[:, 0].view(-1, 1, 1).expand(-1, 1, 2)

            total_direction_bound = points_expanded - total_gt_segs_bound
            total_direction_bound[:, :, 1] = -total_direction_bound[:, :, 1]

            right_dir_bound = points_expanded - right_dir_segs_bound
            right_dir_bound[:, :, 1] = -right_dir_bound[:, :, 1]

            left_dir_bound = points_expanded - left_dir_segs_bound
            left_dir_bound[:, :, 1] = -left_dir_bound[:, :, 1]


            # prevent t_mins / maxs from over-running the action boundary
            # left: torch.maximum(t_mins, gt_segs[:, :, 0])
            # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
            # F T x N (distance to the new boundary)
            cb_dist_left = concat_points[:, 0, None] \
                            - torch.maximum(t_mins, gt_segs[:, :, 0])
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) \
                            - concat_points[:, 0, None]

            # F T x N x 2
            center_seg = torch.stack(
                (cb_dist_left, cb_dist_right), -1)

            # F T x N
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0

            inside_right_dir_seg_mask = right_dir_bound.min(-1)[0]>0
            inside_left_dir_seg_mask = left_dir_bound.min(-1)[0]>0

            total_direction_seg_mask = total_direction_bound.min(-1)[0]>0

            inside_right_dir_seg_mask = torch.logical_and(inside_right_dir_seg_mask, torch.logical_not(inside_gt_seg_mask))
            inside_left_dir_seg_mask = torch.logical_and(inside_left_dir_seg_mask, torch.logical_not(inside_gt_seg_mask))

            stacked_dir_target = torch.stack((inside_gt_seg_mask, inside_right_dir_seg_mask, inside_left_dir_seg_mask), dim=-1)
            stacked_dir_target = stacked_dir_target.float()
            # print(stacked_dir_target.shape)

        else:
            # inside an gt action
            raise NotImplementedError()
            # inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]

        # F T x N
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )


        centers.masked_fill_(total_direction_seg_mask == 0, float('inf'))


        # if there are still more than one ground-truths for one point
        # pick the ground-truth with the shortest distance to the action center for the point
        min_center, min_center_inds = centers.min(dim=1)
        # print(min_center_inds.shape)

        # target_with_center_dist = torch.cat([torch.arange(num_pts).view(-1,1,1).expand(-1,2,1),stacked_dir_target, min_center_inds.view(-1,1,1).expand(-1,2,1)], dim=2)
        # centers_with_range = torch.cat([torch.arange(num_pts).view(-1,1).expand(-1,1), centers, min_center_inds.view(-1,1)], dim=1)
        # print(centers_with_range[:20])
        # print(target_with_center_dist[:20])


        cls_targets = stacked_dir_target[range(num_pts), min_center_inds]
        # # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)

        # OK to use min_len_inds
        reg_targets = reg_targets[range(num_pts), min_center_inds]
        # normalization based on stride
        reg_targets /= concat_points[:, 3, None]

        positive_mask = cls_targets[:,0] > 0

        inside_regress_range = inside_regress_range.sum(dim=1)
        # positive_mask = torch.logical_and(positive_mask, inside_regress_range)


        reg_targets_with_range = torch.cat([concat_points[:,0].view(-1,1), reg_targets], dim=1)
        # print(reg_targets_with_range[positive_mask])
        cls_targets_with_range = torch.cat([concat_points[:,0].view(-1,1), cls_targets], dim=1)
        any_pos_mask = cls_targets.sum(dim=1) > 0
        # print(cls_targets_with_range[any_pos_mask])

        return cls_targets, reg_targets, positive_mask


    @torch.no_grad()
    def label_points(self, points, gt_segments, gt_labels, num_classes):
        # concat points on all fpn levels List[T x 4] -> F T x 4
        # This is shared for all samples in the mini-batch
        num_levels = len(points)
        concat_points = torch.cat(points, dim=0)

        gt_cls, gt_offset = [], []
        # loop over each video sample
        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            assert len(gt_segment) == len(gt_label), (gt_segment, gt_label)
            cls_targets, reg_targets = self.label_points_single_video(
                concat_points, gt_segment, gt_label, num_classes
            )
            # "cls_targets: " #points, num_classes
            # "reg_targets: " #points, 2
            # append to list (len = # images, each of size FT x C)
            gt_cls.append(cls_targets)
            gt_offset.append(reg_targets)

        return gt_cls, gt_offset

    @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_segment, gt_label, num_classes):
        # concat_points : F T x 4 (t, regression range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, num_classes), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets, reg_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x 2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        reg_targets = torch.stack((left, right), dim=-1)

        if self.train_center_sample == 'radius':
            # center of all segments F T x N
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
            # center sampling based on stride radius
            # compute the new boundaries:
            # concat_points[:, 3] stores the stride
            t_mins = \
                center_pts - concat_points[:, 3, None] * self.train_center_sample_radius
            t_maxs = \
                center_pts + concat_points[:, 3, None] * self.train_center_sample_radius

            # prevent t_mins / maxs from over-running the action boundary
            # left: torch.maximum(t_mins, gt_segs[:, :, 0])
            # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
            # F T x N (distance to the new boundary)
            cb_dist_left = concat_points[:, 0, None] \
                           - torch.maximum(t_mins, gt_segs[:, :, 0])
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) \
                            - concat_points[:, 0, None]
            # F T x N x 2
            center_seg = torch.stack(
                (cb_dist_left, cb_dist_right), -1)

            # F T x N
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0
        else:
            # inside an gt action
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]

        # F T x N
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        # limit the regression range for each location and inside the center radius
        lens.masked_fill_(inside_gt_seg_mask == 0, float('inf'))
        lens.masked_fill_(inside_regress_range == 0, float('inf'))

        # if there are still more than one ground-truths for one point
        # pick the ground-truth with the shortest duration for the point (easiest to regress)
        # corner case: multiple actions with very similar durations (e.g., THUMOS14)
        # make sure that each point can only map with at most one ground-truth
        # F T x N -> F T
        min_len, min_len_inds = lens.min(dim=1)
        min_len_mask = torch.logical_and(
            (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
        ).to(reg_targets.dtype)

        # cls_targets: F T x C; reg_targets F T x 2
        # gt_label_one_hot = F.one_hot(gt_label, num_classes).to(reg_targets.dtype)
        gt_label_one_hot = gt_label.to(reg_targets.dtype)
        cls_targets = min_len_mask @ gt_label_one_hot
        # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)

        # OK to use min_len_inds
        reg_targets = reg_targets[range(num_pts), min_len_inds]
        # normalization based on stride
        reg_targets /= concat_points[:, 3, None]

        return cls_targets, reg_targets

    def losses(
            self, fpn_masks,
            out_cls_logits, out_offsets,
            gt_cls_labels, gt_offsets, loss_type="main", weight_scale=None, reg_mask=None
    ):
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        valid_mask = torch.cat(fpn_masks, dim=1)

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        gt_cls = torch.stack(gt_cls_labels)
        if reg_mask is None:
            pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)
        else:
            pos_mask = reg_mask

        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        self.loss_normalizer[loss_type] = self.loss_normalizer_momentum * self.loss_normalizer[loss_type] + (
                1 - self.loss_normalizer_momentum) * max(num_pos, 1)

        # gt_cls is already one hot encoded now, simply masking out
        gt_target = gt_cls[valid_mask]

        num_classes = gt_target.shape[-1]

        # optional label smoothing
        gt_target *= 1 - self.train_label_smoothing
        gt_target += self.train_label_smoothing / (num_classes + 1)

        # focal loss
        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask],
            gt_target,
            reduction='none'
        )
        if weight_scale is not None:
            loss_weight = torch.ones(valid_mask.shape, device=valid_mask.device) * weight_scale.unsqueeze(1)
            loss_weight = loss_weight[valid_mask]
            loss_weight = loss_weight.unsqueeze(1)
            loss_weight = loss_weight.expand(-1, 1)
            cls_loss *= loss_weight
        cls_loss = cls_loss.sum()
        cls_loss /= self.loss_normalizer[loss_type]

        # 2. regression using IoU/GIoU loss (defined on positive samples)
        # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]
        gt_offsets = torch.stack(gt_offsets)[pos_mask]
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # giou loss defined on positive samples
            reg_loss = ctr_diou_loss_1d(
                pred_offsets,
                gt_offsets,
                reduction='sum'
            )
            reg_loss /= self.loss_normalizer[loss_type]

        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        # return a dict of losses
        final_loss = cls_loss + reg_loss * loss_weight
        return {'cls_loss': cls_loss,
                'reg_loss': reg_loss,
                'final_loss': final_loss}

    @torch.no_grad()
    def inference(
            self,
            video_list,
            points, fpn_masks,
            out_cls_logits, out_offsets, num_classes, saliency, wo_postprocess=False
    ):
        # video_list B (list) [dict]
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [B, T_i, C]
        results = []

        # 1: gather video meta information
        vid_idxs = video_list['video_id']
        vid_fps = video_list['fps']
        vid_lens = video_list['duration']
        token_lens = video_list['vid_lens']
        vid_ft_stride = video_list['feat_stride']
        vid_ft_nframes = video_list['feat_num_frames']
        vid_expansion_ratio = video_list['expansion_ratio']
        vid_offsets = video_list['offset']
        vid_saliency = video_list.get('saliency_labels', None)
        vid_pred_timestamp = video_list["pred_timestamp"]
        vid_pred_idx = video_list["pred_idx"]
        annotation_uids = video_list["query_id"]


        saliency_idx = 0
        # 2: inference on each single video and gather the results
        # upto this point, all results use timestamps defined on feature grids
        for idx, (vidx, fps, vlen, stride, nframes, exp_ratio, overall_offset, token_len, ann_uid) in enumerate(
                zip(vid_idxs, vid_fps, vid_lens, vid_ft_stride, vid_ft_nframes, vid_expansion_ratio, vid_offsets, token_lens, annotation_uids)
        ):
            # gather per-video outputs
            cls_logits_per_vid = [x[idx] for x in out_cls_logits]
            offsets_per_vid = [x[idx] for x in out_offsets]
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]
            # inference on a single video (should always be the case)
            results_per_vid = self.inference_single_video(
                points, fpn_masks_per_vid,
                cls_logits_per_vid, offsets_per_vid, num_classes
            )
            # pass through video meta info
            results_per_vid['video_id'] = vidx
            results_per_vid['fps'] = fps
            results_per_vid['duration'] = vlen
            results_per_vid['feat_stride'] = stride
            results_per_vid['feat_num_frames'] = nframes
            results_per_vid['expansion_ratio'] = exp_ratio
            results_per_vid['offset'] = overall_offset
            results_per_vid['annotation_uid'] = ann_uid
            # if vid_saliency is not None:
            if vid_pred_idx[idx] is not None and len(saliency):
                results_per_vid['saliency_labels'] = vid_saliency[idx]
                results_per_vid['pred_saliency'] = saliency[saliency_idx]
                results_per_vid['token_len'] = token_len
                    
                saliency_idx += 1
                results_per_vid['pred_timestamp'] = vid_pred_timestamp[idx]
            results.append(results_per_vid)

        # step 3: postprocessing
        if not wo_postprocess:
            results = self.postprocessing(results)

        return results

    @torch.no_grad()
    def inference_single_video(
            self,
            points,
            fpn_masks,
            out_cls_logits,
            out_offsets,
            num_classes
    ):
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [T_i, C]
        segs_all = []
        scores_all = []
        cls_idxs_all = []

        # loop over fpn levels
        for cls_i, offsets_i, pts_i, mask_i in zip(
                out_cls_logits, out_offsets, points, fpn_masks
        ):
            # sigmoid normalization for output logits
            pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1)).flatten()

            # Apply filtering to make NMS faster following detectron2
            # 1. Keep seg with confidence score > a threshold
            keep_idxs1 = (pred_prob > self.test_pre_nms_thresh)
            pred_prob = pred_prob[keep_idxs1]
            topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_pre_nms_topk, topk_idxs.size(0))
            pred_prob, idxs = pred_prob.sort(descending=True)
            pred_prob = pred_prob[:num_topk].clone()
            topk_idxs = topk_idxs[idxs[:num_topk]].clone()

            # fix a warning in pytorch 1.9
            pt_idxs = torch.div(
                topk_idxs, num_classes, rounding_mode='floor'
            )
            cls_idxs = torch.fmod(topk_idxs, num_classes)

            # 3. gather predicted offsets
            offsets = offsets_i[pt_idxs]
            pts = pts_i[pt_idxs]

            # 4. compute predicted segments (denorm by stride for output offsets)
            seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
            seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]
            pred_segs = torch.stack((seg_left, seg_right), -1)

            # 5. Keep seg with duration > a threshold (relative to feature grids)
            seg_areas = seg_right - seg_left
            keep_idxs2 = seg_areas > self.test_duration_thresh

            # *_all : N (filtered # of segments) x 2 / 1
            segs_all.append(pred_segs[keep_idxs2])
            scores_all.append(pred_prob[keep_idxs2])
            cls_idxs_all.append(cls_idxs[keep_idxs2])

        # cat along the FPN levels (F N_i, C)
        segs_all, scores_all, cls_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
        ]
        results = {'segments': segs_all,
                   'scores': scores_all,
                   'labels': cls_idxs_all}

        return results

    @torch.no_grad()
    def postprocessing(self, results):
        # input : list of dictionary items
        # (1) push to CPU; (2) NMS; (3) convert to actual time stamps
        processed_results = []
        for results_per_vid in results:
            # unpack the meta info
            vidx = results_per_vid['video_id']
            fps = results_per_vid['fps']
            vlen = results_per_vid['duration']
            stride = results_per_vid['feat_stride']
            nframes = results_per_vid['feat_num_frames']
            offset = results_per_vid['offset']
            exp_ratio = results_per_vid['expansion_ratio']
            ann_uid = results_per_vid["annotation_uid"]
            if "saliency_labels" in results_per_vid:
                token_len = results_per_vid['token_len']
                saliency_labels = results_per_vid['saliency_labels']
                pred_saliency = results_per_vid['pred_saliency']
                pred_timestamp = results_per_vid['pred_timestamp']
            # 1: unpack the results and move to CPU
            segs = results_per_vid['segments'].detach().cpu()
            scores = results_per_vid['scores'].detach().cpu()
            labels = results_per_vid['labels'].detach().cpu()

            if self.flash_attn:
                # convert to float32
                segs = segs.float()
                scores = scores.float()
                labels = labels.float()
            if self.test_nms_method != 'none':
                # 2: batched nms (only implemented on CPU)
                segs, scores, labels = batched_nms(
                    segs, scores, labels,
                    self.test_iou_threshold,
                    self.test_min_score,
                    self.test_max_seg_num,
                    use_soft_nms=(self.test_nms_method == 'soft'),
                    multiclass=self.test_multiclass_nms,
                    sigma=self.test_nms_sigma,
                    voting_thresh=self.test_voting_thresh
                )
            # 3: convert from feature grids to seconds
            if segs.shape[0] > 0:
                segs = segs * (stride / fps) / (exp_ratio)
                # truncate all boundaries within [0, duration]
                segs[segs <= 0.0] *= 0.0
                segs[segs >= vlen] = segs[segs >= vlen] * 0.0 + vlen
            # 4: repack the results
            out = {
                 'video_id': vidx,
                 'segments': segs,
                 'scores': scores,
                 'labels': labels,
                }
            if 'saliency_labels' in results_per_vid:
                out['saliency_labels'] = saliency_labels
                out['pred_saliency'] = pred_saliency
                out['pred_timestamp'] = pred_timestamp
                out['vlen'] = token_len
            processed_results.append(out)
                 

        return processed_results

@register_meta_arch("FALM")
class FALM(nn.Module):
    """
        Transformer based model for single stage action localization
    """

    def __init__(
            self,
            arch,  # a tuple defines # layers in embed / stem / branch
            input_vid_dim,  # input video feat dim
            input_txt_dim,  # input text feat dim
            max_seq_len,  # max sequence length (used for training)
            n_head,  # number of heads for self-attention in transformer
            n_mha_win_size,  # window size for self attention; -1 to use full seq
            embd_dim,  # output feat channel of the embedding network
            embd_with_ln,  # attach layernorm to embedding network
            train_cfg,  # other cfg for training
            saliency_loss_weight,
            scores_loss_weight,
            multi_level_saliency_span,
            **kwargs
    ):
        super().__init__()

        self.arch = arch
        self.mha_win_size = n_mha_win_size
        self.max_seq_len = max_seq_len
        self.relu = nn.ReLU(inplace=True)
        # self.random_anticipation = random_anticipation

        pos_embd = get_sinusoid_encoding(self.max_seq_len + 2, input_txt_dim) / (input_txt_dim ** 0.5)
        n_emdb_pos = get_sinusoid_encoding(self.max_seq_len + 2, embd_dim) / (embd_dim ** 0.5)
        self.register_buffer("pos_embd", pos_embd, persistent=False)
        self.register_buffer("n_emdb_pos", n_emdb_pos, persistent=False)

        self.vis_lin = nn.Linear(input_vid_dim, input_txt_dim)

        # vid_embedding network using convs
        self.vid_embd = nn.ModuleList()
        self.vid_embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = input_vid_dim
            else:
                in_channels = embd_dim
            # self.vid_embd.append(MaskedConv1D(
            #     in_channels, embd_dim, embd_dim_ks,
            #     stride=1, padding=embd_dim_ks // 2, bias=(not with_ln)
            # )
            self.vid_embd.append(MaskedConv1D(
                in_channels, embd_dim, 1,
                stride=1, padding=1 // 2, bias=(not embd_with_ln)
            )
            )
            if embd_with_ln:
                self.vid_embd_norm.append(
                    LayerNorm(embd_dim)
                )
            else:
                self.vid_embd_norm.append(nn.Identity())

        self.txt_embd = nn.ModuleList()
        self.txt_embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = input_txt_dim
            else:
                in_channels = embd_dim
            self.txt_embd.append(MaskedConv1D(
                in_channels, embd_dim, 1,
                stride=1, padding=1 // 2, bias=(not embd_with_ln)
            )
            )
            if embd_with_ln:
                self.txt_embd_norm.append(
                    LayerNorm(embd_dim)
                )
            else:
                self.txt_embd_norm.append(nn.Identity())

        self.txt_stem = nn.ModuleList([FlashTransformerBlock(
            embd_dim, n_head,
            n_ds_strides=(1, 1),
            attn_pdrop=train_cfg['dropout'],
            proj_pdrop=train_cfg['attn_dropout'],
            path_pdrop=train_cfg['droppath'],
            mha_win_size=-1,
        ) for _ in range(arch[1])])

        self.vid_stem = nn.ModuleList([FlashTransformerBlock(
            embd_dim, n_head,
            n_ds_strides=(1, 1),
            attn_pdrop=train_cfg['dropout'],
            proj_pdrop=train_cfg['attn_dropout'],
            path_pdrop=train_cfg['droppath'],
            mha_win_size=self.mha_win_size,
        ) for _ in range(arch[2])])

        self.falm_mlp = None
        self.multi_level_saliency_span = multi_level_saliency_span
        self.falm_stem = None
        self.falm_stem = nn.ModuleList([FlashTransformerBlock(
        embd_dim, n_head,
        n_ds_strides=(1, 1),
        attn_pdrop=train_cfg['dropout'],
        proj_pdrop=train_cfg['attn_dropout'],
        path_pdrop=train_cfg['droppath'],
        mha_win_size=self.mha_win_size,
        cross_attn=True,
        ) for _ in range(arch[3])])

        saliency_mlp_num_layers = arch[4]
        assert saliency_mlp_num_layers > 0, "saliency_mlp_num_layers must be greater than 0"
        self.falm_mlp_num_layers = saliency_mlp_num_layers
        self.falm_mlp = MLP(n_in=embd_dim,
                                n_out=1,
                                n_layers=self.falm_mlp_num_layers,
                                n_hidden=embd_dim//2,
                                pdrop=train_cfg['dropout'])
        self.falm_contains_mlp = MLP(n_in=embd_dim,
                                n_out=1,
                                n_layers=self.falm_mlp_num_layers,
                                n_hidden=embd_dim//2,
                                pdrop=train_cfg['dropout'])
        self.falm_not_contains_mlp = MLP(n_in=embd_dim,
                                n_out=1,
                                n_layers=self.falm_mlp_num_layers,
                                n_hidden=embd_dim//2,
                                pdrop=train_cfg['dropout'])
        self.falm_temporal_mlp = MLP(n_in=embd_dim,
                                n_out=1,
                                n_layers=self.falm_mlp_num_layers,
                                n_hidden=embd_dim//2,
                                pdrop=train_cfg['dropout'])
        
        self.falm_contains_score_mlp = MLP(n_in=embd_dim,
                                n_out=1,
                                n_layers=self.falm_mlp_num_layers,
                                n_hidden=embd_dim//2,
                                pdrop=train_cfg['dropout'])
        self.falm_not_contains_score_mlp = MLP(n_in=embd_dim,
                                n_out=1,
                                n_layers=self.falm_mlp_num_layers,
                                n_hidden=embd_dim//2,
                                pdrop=train_cfg['dropout'])

        self.cutoff_token = nn.Parameter(torch.empty(embd_dim, 1))
        nn.init.xavier_uniform_(self.cutoff_token)
        self.falm_cutoff_mlp = MLP(n_in=embd_dim,
                                n_out=2,
                                n_layers=self.falm_mlp_num_layers,
                                n_hidden=embd_dim//2,
                                pdrop=train_cfg['dropout'])
        


        self.multi_level_saliency_span = multi_level_saliency_span
        self.saliency_loss_weight = saliency_loss_weight
        self.scores_loss_weight = scores_loss_weight

    def set_localization_loss_weight(self, weight):
        self.localization_loss_weight = weight
    
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad_(True)

    @property
    def device(self):
        try:
            return int(os.environ["LOCAL_RANK"])
        except:
            return torch.device("cuda:0")

    def pad_seq_with_mask(self, sequences, max_length=None, should_filter=False):
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
                add_feature = torch.zeros(add_length, feature_length, dtype=torch.float32, device=self.device)
                seq_ = torch.cat([seq, add_feature], dim=0)
            else:
                seq_ = seq
            sequence_padded.append(seq_)
        mask = torch.zeros(len(sequences), max_length, dtype=torch.bool, device=self.device)
        for i, l in enumerate(sequence_length):
            mask[i, :l] = True
        sequence_padded = torch.stack(sequence_padded, dim=0)
        sequence_length = torch.tensor(sequence_length, dtype=torch.long)
        idx = torch.tensor(idx, dtype=torch.long)
        return sequence_padded, mask, sequence_length, idx

    def add_norm_scaled_noise(self, x, noise_scale=0.1):
        """
        Add norm-scaled noise to each token in a (B, C, T) tensor.

        Args:
            x (torch.Tensor): shape (B, C, T)
            noise_scale (float): fraction of norm to scale noise by

        Returns:
            torch.Tensor: noisy tensor
        """
        # Compute norms over feature dimension (C)
        norms = x.norm(dim=1, keepdim=True)  # shape (B, 1, T)

        # Generate Gaussian noise per token
        noise = torch.randn_like(x)

        # Normalize noise vectors per token
        noise_norms = noise.norm(dim=1, keepdim=True) + 1e-8
        noise = noise / noise_norms

        # Scale noise magnitude relative to token norm
        noise = noise * norms * noise_scale

        return x + noise

    def forward(self, video_list, get_losses=True, get_preds=False):
        # video_list:  <class 'list'> 1
        # video_list[0] <class 'dict'>
        
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        src_vid, src_vid_mask = video_list["video_feats"], video_list["v_mask"]
        vid_lens = video_list["vid_lens"]
        src_txt = video_list["query_text_feats"]
        src_feedback = video_list["feedback_feats"]
        pred_idx = video_list["pred_idx"]
        pred_timestamp = video_list["pred_timestamp"]

        saliency_labels = video_list["saliency_labels"]

        for idx in range(len(saliency_labels)):
            saliency_labels[idx] = saliency_labels[idx].to(self.device).view(-1, 1)
        saliency_labels, _, _, _ = self.pad_seq_with_mask(saliency_labels, max_length=self.max_seq_len)

        
        src_vid = src_vid.to(self.device)
        src_vid_mask = src_vid_mask.to(self.device)
        vid_lens = vid_lens.to(self.device)
        
        src_txt_clone = [src_txt[i].clone() for i in range(len(src_txt))]

        feedback_idxs = torch.tensor([i for i in range(len(src_txt)) if pred_idx[i] is not None], dtype=torch.long, device=self.device)

        if len(feedback_idxs) == 0:
            losses = results = None
            if get_losses:
                losses = {}
            if get_preds:
                results = {
                    "query_id":[],
                    "pred_timestamp": [],
                    "saliency_labels": [],
                    "pred_saliency": [],
                    "falm_feats": [],
                    "vlen": [],
                }
            return losses, results

        # segments = video_list["segments"]
        for i in range(len(src_txt_clone)):
            src_txt_clone[i] = src_txt_clone[i].to(self.device)
            if pred_idx[i] is not None:
                timestamp = pred_timestamp[i]
                # timestamp = segments[i][0]
                # print(timestamp)
                start = torch.clamp(timestamp[0], min=0, max=vid_lens[i]-1).floor().long()
                end = torch.clamp(timestamp[1], min=1, max=vid_lens[i]).ceil().long()
                if start == end:
                    end = start + 1
                pred_pe =  self.pos_embd[0,:,start:end].permute(1,0)
                vids = src_vid[i, :, start:end].permute(1,0)
                vids = self.vis_lin(vids)
                vids = vids+pred_pe
                vid_mean = torch.mean(vids, dim=0, keepdim=True)
                s_e_pe = self.pos_embd[0,:,[start, end]].permute(1,0)
                s_e_vid = src_vid[i, :, [start, end]].permute(1,0)
                s_e_vid = self.vis_lin(s_e_vid)
                s_e_vid = s_e_vid+s_e_pe
                pred_tokens = torch.cat([vid_mean, s_e_vid], dim=0)
                if src_feedback[i] is None:
                    src_txt_clone[i] =  torch.cat([src_txt_clone[i][:pred_idx[i][0]+1], src_txt_clone[i][pred_idx[i][1]+1:], pred_tokens], dim=0)
                else:
                    src_feedback[i] = src_feedback[i].to(self.device)
                    # print("pred_idx[i]: ", pred_idx[i])
                    # print("src_feedback[i].shape: ", src_feedback[i].shape)
                    # print("pred_tokens.shape: ", pred_tokens.shape)
                    # print("src_txt[i].shape: ", src_txt[i].shape)
                    src_txt_clone[i] =  torch.cat([src_txt_clone[i], src_feedback[i][:pred_idx[i][0]+1], src_feedback[i][pred_idx[i][1]+1:], pred_tokens], dim=0)
                    # print("final src_txt[i].shape: ", src_txt[i].shape)
                # add positional embedding for the text but not pred tokens
                
                pos_txt_embed = self.pos_embd[0,:,:src_txt_clone[i].shape[0]-pred_tokens.shape[0]].permute(1,0)
                src_txt_clone[i][:pos_txt_embed.shape[0]] += pos_txt_embed

        src_txt_clone = [src_txt_clone[i].to(self.device) for i in range(len(src_txt_clone))]
        
        src_txt, src_txt_mask, _, _ = self.pad_seq_with_mask(src_txt_clone)
        src_txt = src_txt.permute(0, 2, 1)
        src_txt_mask = src_txt_mask.bool()
        src_txt_mask = src_txt_mask.unsqueeze(1)

        src_txt = src_txt.to(self.device)
        src_txt_mask = src_txt_mask.to(self.device)

        src_vid = src_vid[feedback_idxs]
        original_vid_mask = src_vid_mask
        src_vid_mask = src_vid_mask[feedback_idxs]
        src_txt = src_txt[feedback_idxs]
        src_txt_mask = src_txt_mask[feedback_idxs]
        saliency_labels = saliency_labels[feedback_idxs]

        B, C, T = src_vid.size()
        B, C, txtT = src_txt.size()

        # if self.training:
        #     src_vid = self.add_norm_scaled_noise(src_vid, 0.1) * src_vid_mask.to(src_vid.dtype)

        for idx in range(len(self.txt_embd)):
            src_txt, src_txt_mask = self.txt_embd[idx](src_txt, src_txt_mask)
            src_txt = self.relu(self.txt_embd_norm[idx](src_txt))


        for idx in range(len(self.vid_embd)):
            src_vid, src_vid_mask = self.vid_embd[idx](src_vid, src_vid_mask)
            src_vid = self.relu(self.vid_embd_norm[idx](src_vid))
        
        # add positional embedding
        src_vid = src_vid + self.n_emdb_pos[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        src_vid =torch.cat([self.cutoff_token.unsqueeze(0).expand(B, -1, -1), src_vid], dim=2)
        src_vid_mask = torch.cat([torch.ones(B, 1, 1, dtype=src_vid_mask.dtype, device=self.device), src_vid_mask], dim=2)

        src_query, src_query_mask = src_txt, src_txt_mask
        src_query_packed, query_indices, query_cu_seqlens, query_max_seqlen_in_batch, _ = unpad_input(src_query.permute(0,2,1), src_query_mask.squeeze(1))
        query_varlen_params = {"cu_seqlens": query_cu_seqlens, "indices": query_indices, "max_seqlen": query_max_seqlen_in_batch}
        q_residual = None
        for idx in range(len(self.txt_stem)):
            src_query_packed, src_query_mask, q_residual = self.txt_stem[idx](src_query_packed, None, src_query_mask, query_varlen_params, q_residual)

        src_vid_packed, vid_indices, vid_cu_seqlens, vid_max_seqlen_in_batch, _ = unpad_input(src_vid.permute(0,2,1), src_vid_mask.squeeze(1))
        vid_varlen_params = {"cu_seqlens": vid_cu_seqlens, "indices": vid_indices, "max_seqlen": vid_max_seqlen_in_batch}
        residual = None
        for idx in range(len(self.vid_stem)):
            src_vid_packed, src_vid_mask, residual = self.vid_stem[idx](src_vid_packed, None, src_vid_mask, vid_varlen_params, residual)

        saliency_pred = []
        contains_pred = []
        contains_score_pred = []
        not_contains_pred = []
        not_contains_score_pred = []
        temporal_pred = []
        contains_cutoff_pred = []
        not_contains_cutoff_pred = []
        sal_residual = residual
        sal_packed = src_vid_packed
        final_feats = None
        for idx in range(len(self.falm_stem)):
            sal_packed, _, sal_residual = self.falm_stem[idx](sal_packed, None, src_vid_mask, vid_varlen_params, sal_residual, src_query_packed, None, src_query_mask, query_varlen_params)
                # saliency prediction
            sal_padded = pad_input(sal_packed, vid_indices, src_vid_mask.size(0), src_vid_mask.size(2)) # B, T, C
            cutoff_token = sal_padded[:,0:1,:]
            sal_padded = sal_padded[:,1:,:]  # remove cutoff token

            contains, contains_feats = self.falm_contains_mlp(sal_padded)
            contains = torch.sigmoid(contains)

            not_contains, not_contains_feats = self.falm_not_contains_mlp(sal_padded)
            not_contains = torch.sigmoid(not_contains)

            temporal, temporal_feats = self.falm_temporal_mlp(sal_padded)
            temporal = torch.sigmoid(temporal)

            contains_score, _ = self.falm_contains_score_mlp(sal_padded)
            contains_score = torch.sigmoid(contains_score)

            not_contains_score, _ = self.falm_not_contains_score_mlp(sal_padded)
            not_contains_score = torch.sigmoid(not_contains_score)

            cutoffs, _ = self.falm_cutoff_mlp(cutoff_token)
            cutoffs = torch.sigmoid(cutoffs)
            contains_cutoff = cutoffs[:,:,0:1]
            not_contains_cutoff = cutoffs[:,:,1:2]
            # sal_feats = torch.cat([contains_feats, not_contains_feats, temporal_feats], dim=-1)

            saliency, _ = self.falm_mlp(sal_padded)
            saliency, _ = self.falm_mlp(sal_padded)
            saliency = torch.sigmoid(saliency)


            #threshold contains score at contains cutoff
            # th_contains_score = (contains_score >= contains_cutoff.expand(-1, contains_score.size(1), -1)).float()
            # th_not_contains_score = (not_contains_score >= not_contains_cutoff.expand(-1, not_contains_score.size(1), -1)).float()
            # th_temporal = (temporal >= 0.5).float()
            # saliency = th_contains_score * th_not_contains_score * th_temporal

            # # 0-1 normalization
            # min_val = saliency.min(dim=1, keepdim=True)[0]
            # max_val = saliency.max(dim=1, keepdim=True)[0]

            # saliency = (saliency - min_val) / (max_val - min_val + 1e-12)
            # # clamp to [0, 1]
            # saliency = torch.clamp(saliency, 0.0, 1.0)
            
            saliency_pred.append(saliency)
            contains_pred.append(contains)
            not_contains_pred.append(not_contains)
            temporal_pred.append(temporal)
            contains_score_pred.append(contains_score)
            not_contains_score_pred.append(not_contains_score)
            contains_cutoff_pred.append(contains_cutoff)
            not_contains_cutoff_pred.append(not_contains_cutoff)
        
        
        final_feats = sal_padded

        assert "saliency_labels" in video_list, "GT action labels does not exist"
        # saliency_labels = video_list.get("saliency_labels", None)
        # for idx in range(len(saliency_labels)):
        #     saliency_labels[idx] = saliency_labels[idx].to(self.device).view(-1, 1)
        # saliency_labels, _, _, _ = self.pad_seq_with_mask(saliency_labels, max_length=self.max_seq_len)
        # saliency_labels = saliency_labels.to(self.device)
        saliency_mask = src_vid_mask.permute(0,2,1)
        saliency_mask = saliency_mask[:, 1:, :]
        original_vid_mask = original_vid_mask.permute(0,2,1)
        contains_labels_ori = video_list.get("contains_labels", None)
        contains_idx = [i for i in range(len(contains_labels_ori)) if contains_labels_ori[i] is not None]
        if len(contains_idx) > 0:
            for i in range(len(contains_labels_ori)):
                if contains_labels_ori[i] is not None:
                    contains_labels_ori[i] = contains_labels_ori[i].to(self.device).view(-1, 1)
            contains_labels = [contains_labels_ori[i] for i in contains_idx]
            contains_labels, _, _, _ = self.pad_seq_with_mask(contains_labels, max_length=self.max_seq_len)
            contains_mask = original_vid_mask[contains_idx]
            contains_labels = contains_labels.to(self.device)
        
        not_contains_labels_ori = video_list.get("not_contains_labels", None)
        not_contains_idx = [i for i in range(len(not_contains_labels_ori)) if not_contains_labels_ori[i] is not None]
        if len(not_contains_idx) > 0:
            for i in range(len(not_contains_labels_ori)):
                if not_contains_labels_ori[i] is not None:
                    not_contains_labels_ori[i] = not_contains_labels_ori[i].to(self.device).view(-1, 1)
            not_contains_labels = [not_contains_labels_ori[i] for i in not_contains_idx]
            not_contains_labels, _, _, _ = self.pad_seq_with_mask(not_contains_labels, max_length=self.max_seq_len)
            not_contains_mask = original_vid_mask[not_contains_idx]
            not_contains_labels = not_contains_labels.to(self.device)
        
        temporal_labels_ori = video_list.get("temporal_labels", None)
        temporal_idx = [i for i in range(len(temporal_labels_ori)) if temporal_labels_ori[i] is not None]
        if len(temporal_idx) > 0:
            for i in range(len(temporal_labels_ori)):
                if temporal_labels_ori[i] is not None:
                    temporal_labels_ori[i] = temporal_labels_ori[i].to(self.device).view(-1, 1)
            temporal_labels = [temporal_labels_ori[i] for i in temporal_idx]
            temporal_labels, _, _, _ = self.pad_seq_with_mask(temporal_labels, max_length=self.max_seq_len)
            temporal_mask = original_vid_mask[temporal_idx]
            temporal_labels = temporal_labels.to(self.device)
        
        contains_scores_lbl_ori = video_list.get("contains_scores", None)
        contains_scores_idx = [i for i in range(len(contains_scores_lbl_ori)) if contains_scores_lbl_ori[i] is not None]
        if len(contains_scores_idx) > 0:
            for i in range(len(contains_scores_lbl_ori)):
                if contains_scores_lbl_ori[i] is not None:
                    contains_scores_lbl_ori[i] = contains_scores_lbl_ori[i].to(self.device).view(-1, 1)
            contains_scores_lbl = [contains_scores_lbl_ori[i] for i in contains_scores_idx]
            contains_scores_lbl, _, _, _ = self.pad_seq_with_mask(contains_scores_lbl, max_length=self.max_seq_len)
            contains_scores_mask = original_vid_mask[contains_scores_idx]
            contains_scores_lbl = contains_scores_lbl.to(self.device)
        
        
        not_contains_scores_lbl_ori = video_list.get("not_contains_scores", None)
        not_contains_scores_idx = [i for i in range(len(not_contains_scores_lbl_ori)) if not_contains_scores_lbl_ori[i] is not None]
        if len(not_contains_scores_idx) > 0:
            for i in range(len(not_contains_scores_lbl_ori)):
                if not_contains_scores_lbl_ori[i] is not None:
                    not_contains_scores_lbl_ori[i] = not_contains_scores_lbl_ori[i].to(self.device).view(-1, 1)
            not_contains_scores_lbl = [not_contains_scores_lbl_ori[i] for i in not_contains_scores_idx]
            not_contains_scores_lbl, _, _, _ = self.pad_seq_with_mask(not_contains_scores_lbl, max_length=self.max_seq_len)
            not_contains_scores_mask = original_vid_mask[not_contains_scores_idx]
            not_contains_scores_lbl = not_contains_scores_lbl.to(self.device)

        contains_cutoff_labels_ori = video_list.get("contains_cutoff", None)
        contains_cutoff_idx = [i for i in range(len(contains_cutoff_labels_ori)) if contains_cutoff_labels_ori[i] is not None]
        if len(contains_cutoff_idx) > 0:
            contains_cutoff_labels = [contains_cutoff_labels_ori[i] for i in contains_cutoff_idx]
            contains_cutoff_labels = torch.tensor(contains_cutoff_labels, dtype=torch.float32, device=self.device).view(-1, 1, 1)
            contains_cutoff_mask = torch.ones_like(contains_cutoff_labels, dtype=torch.bool, device=self.device)

        not_contains_cutoff_labels_ori = video_list.get("not_contains_cutoff", None)
        not_contains_cutoff_idx = [i for i in range(len(not_contains_cutoff_labels_ori)) if not_contains_cutoff_labels_ori[i] is not None]
        if len(not_contains_cutoff_idx) > 0:
            not_contains_cutoff_labels = [not_contains_cutoff_labels_ori[i] for i in not_contains_cutoff_idx]
            not_contains_cutoff_labels = torch.tensor(not_contains_cutoff_labels, dtype=torch.float32, device=self.device).view(-1, 1, 1)
            not_contains_cutoff_mask = torch.ones_like(not_contains_cutoff_labels, dtype=torch.bool, device=self.device)

        # return loss during training
        losses = None
        results = None
        if get_losses:
            
            # temporal_scores_lbl = video_list.get("temporal_scores", None)
            # temporal_scores_idx = [i for i in range(len(temporal_scores_lbl)) if temporal_scores_lbl[i] is not None]
            # if len(temporal_scores_idx) > 0:
            #     for i in range(len(temporal_scores_lbl)):
            #         if temporal_scores_lbl[i] is not None:
            #             temporal_scores_lbl[i] = temporal_scores_lbl[i].to(self.device).view(-1, 1)
            #     temporal_scores_lbl = [temporal_scores_lbl[i] for i in temporal_scores_idx]
            #     temporal_scores_lbl, _, _, _ = self.pad_seq_with_mask(temporal_scores_lbl, max_length=self.max_seq_len)
            #     temporal_scores_mask = saliency_mask[temporal_scores_idx]
            #     temporal_scores_lbl = temporal_scores_lbl.to(self.device)
            losses = {}
            
            

            if self.multi_level_saliency_span:
                for i in range(len(saliency_pred)):
                    assert False, "span loss not implemented for multi level saliency and span"
                    curr_saliency = saliency_pred[i]
                    saliency_loss = self.saliency_loss(curr_saliency, saliency_mask, saliency_labels)
                    if "saliency_loss" not in losses:
                        losses["saliency_loss"] = saliency_loss
                    else:
                        losses["saliency_loss"] += saliency_loss
                losses["saliency_loss"] /= len(saliency_pred)
            else:
                saliency_loss = self.saliency_loss(saliency_pred[-1], saliency_mask, saliency_labels)
                losses["saliency_loss"] = saliency_loss
                # saliency_scores_loss = self.saliency_loss_mse(saliency_pred[-1], saliency_mask, saliency_labels)
                # losses["saliency_scores_loss"] = saliency_scores_loss
                # if len(contains_idx) > 0:
                #     contains_loss = self.saliency_loss(contains_pred[-1][contains_idx], contains_mask, contains_labels)
                #     losses["contains_loss"] = contains_loss
                # if len(not_contains_idx) > 0:
                #     not_contains_loss = self.saliency_loss(not_contains_pred[-1][not_contains_idx], not_contains_mask, not_contains_labels)
                #     losses["not_contains_loss"] = not_contains_loss
                if len(temporal_idx) > 0:
                    temporal_loss = self.saliency_loss(temporal_pred[-1][temporal_idx], temporal_mask, temporal_labels)
                    losses["temporal_loss"] = temporal_loss
                # if len(contains_scores_idx) > 0:
                #     contains_scores_loss = self.saliency_loss_l2(contains_score_pred[-1][contains_scores_idx], contains_scores_mask, contains_scores_lbl)
                #     losses["contains_scores_loss"] = contains_scores_loss
                # if len(not_contains_scores_idx) > 0:
                #     not_contains_scores_loss = self.saliency_loss_l2(not_contains_score_pred[-1][not_contains_scores_idx], not_contains_scores_mask, not_contains_scores_lbl)
                #     losses["not_contains_scores_loss"] = not_contains_scores_loss
                # if len(temporal_scores_idx) > 0:
                #     temporal_scores_loss = self.saliency_loss_mse(temporal_pred[-1][temporal_scores_idx], temporal_scores_mask, temporal_scores_lbl)
                #     losses["temporal_scores_loss"] = temporal_scores_loss
                # if len(contains_cutoff_idx) > 0:
                #     contains_cutoff_loss = self.saliency_loss_l2(contains_cutoff_pred[-1][contains_cutoff_idx], contains_cutoff_mask, contains_cutoff_labels)
                #     losses["contains_cutoff_loss"] = contains_cutoff_loss
                # if len(not_contains_cutoff_idx) > 0:
                #     not_contains_cutoff_loss = self.saliency_loss_l2(not_contains_cutoff_pred[-1][not_contains_cutoff_idx], not_contains_cutoff_mask, not_contains_cutoff_labels)
                #     losses["not_contains_cutoff_loss"] = not_contains_cutoff_loss

            losses["final_loss"] = losses["saliency_loss"].clone()
            # if "saliency_scores_loss" in losses:
            #     losses["final_loss"] += losses["saliency_scores_loss"].clone()
            # if "contains_loss" in losses:
            #     losses["final_loss"] += losses["contains_loss"].clone() 
            # if "not_contains_loss" in losses:
            #     losses["final_loss"] += losses["not_contains_loss"].clone() 
            if "temporal_loss" in losses:
                if "final_loss" not in losses:
                    losses["final_loss"] = losses["temporal_loss"].clone()
                losses["final_loss"] += losses["temporal_loss"].clone() 
            # if "contains_scores_loss" in losses:
            #     if "final_loss" not in losses:
            #         losses["final_loss"] = losses["contains_scores_loss"].clone()
            #     losses["final_loss"] += losses["contains_scores_loss"].clone() * self.scores_loss_weight
            # if "not_contains_scores_loss" in losses:
            #     if "final_loss" not in losses:
            #         losses["final_loss"] = losses["not_contains_scores_loss"].clone()
            #     losses["final_loss"] += losses["not_contains_scores_loss"].clone()  * self.scores_loss_weight
            # if "temporal_scores_loss" in losses:
            #     losses["final_loss"] += losses["temporal_scores_loss"].clone()
                            
        if get_preds:
            # decode the actions (sigmoid / stride, etc)
                results = {}
                results["fps"] = 30 # hard code for now
                results["feat_stride"] = 16 # hard code for now
                results["query_id"] = video_list["query_id"]
                results["video_id"] = video_list["video_id"]
                results["vlen"] = video_list["vid_lens"]
                list_pred_timestamp = []
                for i in range(len(video_list["pred_timestamp"])):
                    list_pred_timestamp.append(video_list["pred_timestamp"][i])
                results["pred_timestamp"] = list_pred_timestamp
                results["pred_saliency"] = saliency_pred[-1]
                results["pred_contains"] = contains_pred[-1]
                results["pred_not_contains"] = not_contains_pred[-1]
                results["pred_temporal"] = temporal_pred[-1]
                results["pred_contains_score"] = contains_score_pred[-1]
                results["pred_not_contains_score"] = not_contains_score_pred[-1]
                results["pred_contains_cutoff"] = contains_cutoff_pred[-1]
                results["pred_not_contains_cutoff"] = not_contains_cutoff_pred[-1]
                results["falm_feats"] = final_feats
                
                results["saliency_labels"] = saliency_labels

        return losses, results

    def saliency_loss_l2(self, saliency_pred, saliency_mask, saliency_gt):
        loss = F.mse_loss(saliency_pred, saliency_gt, reduction='none')

        # print("loss.shape: ", loss.shape)
        # Apply the mask
        loss = loss * saliency_mask
        loss = loss.squeeze(2)

        # Normalize by number of valid (unmasked) elements to avoid bias
        # valid_count = saliency_mask.sum()
        # if valid_count == 0:
        #     return torch.tensor(0.0, device=saliency_pred.device)

        loss = loss.sum(dim=1) / (saliency_mask.squeeze(2).sum(dim=1) + 1e-12)

        return loss.mean()

    def saliency_loss_l1(self, saliency_pred, saliency_mask, saliency_gt):
        loss = F.l1_loss(saliency_pred, saliency_gt, reduction='none')

        # print("loss.shape: ", loss.shape)
        # Apply the mask
        loss = loss * saliency_mask
        loss = loss.squeeze(2)

        # Normalize by number of valid (unmasked) elements to avoid bias
        # valid_count = saliency_mask.sum()
        # if valid_count == 0:
        #     return torch.tensor(0.0, device=saliency_pred.device)

        loss = loss.sum(dim=1) / (saliency_mask.squeeze(2).sum(dim=1) + 1e-12)

        return loss.mean()
    
    def saliency_loss(self, saliency_pred, saliency_mask, saliency_gt):
        loss = F.binary_cross_entropy(saliency_pred, saliency_gt, reduction='none')

        # print("saliency_pred.shape: ", saliency_pred.shape)
        # print("saliency_mask.shape: ", saliency_mask.shape)
        # print("saliency_gt.shape: ", saliency_gt.shape)
        # print("loss.shape: ", loss.shape)
        # Apply the mask
        loss = loss * saliency_mask
        loss = loss.squeeze(2)

        # Normalize by number of valid (unmasked) elements to avoid bias
        # valid_count = saliency_mask.sum()
        # if valid_count == 0:
        #     return torch.tensor(0.0, device=saliency_pred.device)

        loss = loss.sum(dim=1) / (saliency_mask.squeeze(2).sum(dim=1) + 1e-12)

        return loss.mean()


# @register_meta_arch("QCAALang")
# class QCAA_Lang_Transfomer(nn.Module):
#     def __init__(
#             self,
#             backbone_type,  # a string defines which backbone we use
#             fpn_type,  # a string defines which fpn we use
#             backbone_arch,  # a tuple defines # layers in embed / stem / branch
#             scale_factor,  # scale factor between branch layers
#             input_vid_dim,  # input video feat dim
#             input_txt_dim,  # input text feat dim
#             max_seq_len,  # max sequence length (used for training)
#             summary_len,
#             max_buffer_len_factor,  # max buffer size (defined a factor of max_seq_len)
#             n_head,  # number of heads for self-attention in transformer
#             n_mha_win_size,  # window size for self attention; -1 to use full seq
#             embd_kernel_size,  # kernel size of the embedding network
#             embd_dim,  # output feat channel of the embedding network
#             embd_with_ln,  # attach layernorm to embedding network
#             fpn_dim,  # feature dim on FPN
#             fpn_with_ln,  # if to apply layer norm at the end of fpn
#             fpn_start_level,  # start level of fpn
#             head_dim,  # feature dim for head
#             regression_range,  # regression range on each level of FPN
#             head_num_layers,  # number of layers in the head (including the classifier)
#             head_kernel_size,  # kernel size for reg/cls heads
#             head_with_ln,  # attach layernorm to reg/cls heads
#             use_abs_pe,  # if to use abs position encoding
#             use_rel_pe,  # if to use rel position encoding
#             num_classes,  # number of action classes
#             train_cfg,  # other cfg for training
#             test_cfg,  # other cfg for testing
#             consecutive_masking,
#             narr_decoder,
#             narr_decoder_cfg,
#             num_summarizer_blocks,
#             summary_resolution,
#             summary_of_summary,
#             single_token_language,
#             anchor_localization,
#             anchor_localization_cfg,
#             consecutive_masking_cfg,
#             provide_visual_info,
#             provide_visual_info_cfg,
#             contrastive_learning,
#             contrastive_learning_cfg,
#             predict_visual_info,
#             predict_visual_info_cfg,
#             localization_refinement,
#             localization_refinement_cfg
#     ):
#         super().__init__()

#         self.layers = predict_visual_info_cfg['layers']
#         self.embed_dim = predict_visual_info_cfg['embd_dim']
#         self.n_head = predict_visual_info_cfg['n_head']
        
#         self.train_dropout = train_cfg['dropout']
#         self.train_droppath = train_cfg['droppath']
#         self.context_action_order = predict_visual_info_cfg['context_action_order']
#         self.context_action_use_pos = predict_visual_info_cfg['use_pos']
#         self.max_seq_len = max_seq_len

#         self.vis_info_loss_type = predict_visual_info_cfg['loss_type']
#         self.vis_info_loss = "vis" in self.vis_info_loss_type
#         self.vis_info_loss_weight = predict_visual_info_cfg['vis_info_loss_weight']
#         self.loss_temperature = predict_visual_info_cfg['temperature']
#         self.loss_across_batch = predict_visual_info_cfg['across_batch']
#         self.loss_across_pred = predict_visual_info_cfg['across_pred']
#         self.use_duplicate_mask = predict_visual_info_cfg['use_duplicate_mask']
#         self.order_loss = predict_visual_info_cfg['order_loss']
#         order_num_layers = predict_visual_info_cfg['order_num_layers']
#         self.order_loss_weight = predict_visual_info_cfg['order_loss_weight']
#         self.order_distance = predict_visual_info_cfg['order_distance']
#         self.l2_loss = predict_visual_info_cfg['l2_loss']
#         self.l2_loss_weight = predict_visual_info_cfg['l2_loss_weight']
#         self.lang_info_loss = "lang" in self.vis_info_loss_type
#         self.lang_info_loss_weight = predict_visual_info_cfg['lang_info_loss_weight']
#         self.uniform_vid_sample = predict_visual_info_cfg['uniform_vid_sample']
#         self.subsample = predict_visual_info_cfg['subsample']
#         self.difference_loss = predict_visual_info_cfg['difference_loss']
#         self.difference_loss_weight = predict_visual_info_cfg['difference_loss_weight']

#         total_tokens = 2 * self.context_action_order + 1
#         order_gt = torch.triu(torch.ones(total_tokens,total_tokens)).view(-1).long()
#         self.register_buffer("order_gt", order_gt)
#         order_mask = torch.ones(total_tokens, total_tokens).fill_diagonal_(0)
#         row_indices = torch.arange(total_tokens).view(-1, 1)
#         col_indices = torch.arange(total_tokens).view(1, -1)
#         # Compute the distance from the diagonal
#         distance_m = torch.abs(row_indices - col_indices)
#         # Create the mask where distance is within the radius
#         distance_mask = (distance_m <= self.order_distance).float()
#         order_mask = order_mask * distance_mask
#         self.register_buffer("order_mask", order_mask)
#         self.order_mlp = MLP(
#             n_in=2*input_vid_dim,
#             n_out=2,
#             n_layers=order_num_layers,
#             n_hidden=embd_dim,
#             pdrop=self.train_dropout
#         )
#         reorder_idx = torch.tensor([*[2*self.context_action_order+1-i for i in range(1, self.context_action_order+1)], 0,  *[i+1 for i in range(0, self.context_action_order)]])
#         self.register_buffer("reorder_idx", reorder_idx)
        
#         self.qcaa = QCAA_Lang(
#             lin_embed_layers=1,
#             txt_n_layers=self.layers,
#             n_head=self.n_head,
#             n_txt_in=input_txt_dim,
#             n_embd=self.embed_dim,
#             n_embd_ks=embd_kernel_size,
#             n_vid_in=input_vid_dim,
#             max_len=max_seq_len,
#             mha_win_size=n_mha_win_size,
#             attn_pdrop=self.train_dropout,
#             proj_pdrop=self.train_dropout,
#             path_pdrop=self.train_droppath,
#             with_ln=embd_with_ln,
#             context_action_order=self.context_action_order,
#             context_action_use_pos=self.context_action_use_pos,
#             use_abs_pe=use_abs_pe,
#             use_rel_pe=use_rel_pe,
#             uniform_vid_sample=self.uniform_vid_sample,
#         )

#     @property
#     def device(self):
#         try:
#             return int(os.environ["LOCAL_RANK"])
#         except:
#             return torch.device("cuda:0")


#     def forward(self, video_list, get_losses=True, get_preds=False, get_hit=False, dist_group=None, wo_postprocess=False):
#         """
#             Forward function for the model
#         """
#         src_txt, src_txt_mask = video_list["query_text_feats"].to(self.device), video_list["q_mask"].to(self.device)
#         src_vid, src_vid_mask = video_list["video_feats"].to(self.device), video_list["v_mask"].to(self.device)

#         vid_lens = video_list["vid_lens"].to(self.device)


#         with torch.no_grad():
#             context_narration_gt = video_list["context_narration_gt"].to(self.device)
#             context_narration_gt_mask = video_list["context_narration_gt_mask"].to(self.device)

#             context_action_vis_gt_indices = video_list["context_action_gt_indices"].to(self.device)
#             context_action_gt_indices_mask = video_list["context_action_gt_indices_mask"].to(self.device) # B, S, I
#             context_action_vis_gt_mask = (~torch.all(context_action_gt_indices_mask==0, dim=-1)).to(torch.int64).unsqueeze(1)
            
#             B,C,T = src_vid.size()
#             B,S,I = context_action_vis_gt_indices.size()
#             # assert torch.eq(context_action_vis_gt_mask, context_narration_gt_mask).all()
#             context_duplicate_mask = video_list["context_duplicate_mask"].to(self.device)
#             indices = context_action_vis_gt_indices.unsqueeze(1).expand(-1, C, -1, -1)  # B, C, S, I
#             selected_feats = torch.gather(src_vid.view(B,C,1,T).expand(-1,-1, S,-1), 3, indices)  # B, C, S, I
#             selected_feats = selected_feats.permute(0, 2, 3, 1) # B, S, I, C
            
#             # context_action_vis_gt = (selected_feats * context_action_gt_indices_mask.unsqueeze(-1)).sum(dim=2) 
#             # context_action_vis_gt = context_action_vis_gt / (context_action_gt_indices_mask.sum(-1).unsqueeze(-1)+1e-6) #B, S, C
#             context_action_vis_gt = []
#             for i in range(B):
#                 sum_feat = torch.sum(selected_feats[i] * context_action_gt_indices_mask[i].unsqueeze(-1), dim=1)
#                 sum_feat = sum_feat / (context_action_gt_indices_mask[i].sum(-1).unsqueeze(-1)+1e-6)
#                 context_action_vis_gt.append(sum_feat)
#             context_action_vis_gt = torch.stack(context_action_vis_gt, dim=0) # B, S, C
#             context_action_vis_gt = context_action_vis_gt
#             context_action_vis_gt = context_action_vis_gt.permute(0, 2, 1)  # B, C, S

#         src_txt, src_txt_mask, context_action_lang_only, context_narration_lang_only = self.qcaa(src_vid, src_vid_mask, src_txt, src_txt_mask, vid_lens)
#         losses = None
#         if get_losses:
#             losses = {}
#             loss_types = []
#             contexts = []
#             gts = []
#             gt_masks = []
#             if self.vis_info_loss:
#                 loss_types.append("vis")
#                 contexts.append(context_action_lang_only)
#                 gts.append(context_action_vis_gt)
#                 gt_masks.append(context_action_vis_gt_mask)
#                 vis_losses = self.get_vis_info_loss(context_action_lang_only, context_action_vis_gt, context_action_vis_gt_mask, context_duplicate_mask)
#                 if vis_losses is not None:
#                     losses.update(vis_losses)
#                     if "final_loss" in losses:
#                         losses["final_loss"] += vis_losses.get("vis_info_loss", 0) * self.vis_info_loss_weight + self.difference_loss_weight * vis_losses.get("vis_difference_loss", 0)
#                     elif len(losses) > 0:
#                         final_loss = vis_losses.get("vis_info_loss", 0) * self.vis_info_loss_weight + self.difference_loss_weight * vis_losses.get("vis_difference_loss", 0)
#                         losses["final_loss"] = final_loss.clone()
#                 else:
#                     assert not self.training, "No vis Loss when training"
#             if self.lang_info_loss:
#                 loss_types.append("lang")
#                 contexts.append(context_narration_lang_only)
#                 gts.append(context_narration_gt)
#                 gt_masks.append(context_narration_gt_mask)
#                 lang_info_losses = self.get_lang_info_loss(context_narration_lang_only, context_narration_gt, context_narration_gt_mask, context_duplicate_mask)
#                 if lang_info_losses is not None:
#                     losses.update(lang_info_losses)
#                     if "final_loss" in losses:
#                         losses["final_loss"] += lang_info_losses.get("lang_info_loss", 0) * self.lang_info_loss_weight + self.difference_loss_weight * lang_info_losses.get("lang_difference_loss", 0)
#                     elif len(losses) > 0:
#                         final_loss = lang_info_losses.get("lang_info_loss", 0) * self.lang_info_loss_weight + self.difference_loss_weight * lang_info_losses.get("lang_difference_loss", 0)
#                         losses["final_loss"] = final_loss.clone()
#                 else:
#                     assert not self.training, "No lang Loss when training"

#             if self.l2_loss:
#                 for loss_type, context, gt, gt_mask in zip(loss_types, contexts, gts, gt_masks):
#                     l2_loss = self.get_l2_loss(context, gt, gt_mask)
#                     if l2_loss is not None:
#                         losses[f"l2_{loss_type}_loss"] = l2_loss
#                         if "final_loss" in losses:
#                             losses["final_loss"] += self.l2_loss_weight * l2_loss
#                         else:
#                             losses["final_loss"] = self.l2_loss_weight * l2_loss

#             if self.order_loss:
#                 for loss_type, context, gt, gt_mask in zip(loss_types, contexts, gts, gt_masks):
#                     order_loss = self.get_order_loss(context, gt, gt_mask)
#                     if order_loss is not None:
#                         losses[f"order_{loss_type}_loss"] = order_loss
#                         if "final_loss" in losses:
#                             losses["final_loss"] += self.order_loss_weight * order_loss
#                         else:
#                             losses["final_loss"] = self.order_loss_weight * order_loss
           
                
#         preds = None
#         if get_hit:
#             if self.difference_loss:
#                 context_action_pred = context_action_lang_only[:, :, 1:] + context_action_lang_only[:, :, 0].unsqueeze(2)
#                 context_action_pred = torch.cat([context_action_lang_only[:, :, 0].unsqueeze(2), context_action_pred], dim=2)
#                 context_narration_pred = context_narration_lang_only[:, :, 1:] + context_narration_lang_only[:, :, 0].unsqueeze(2)
#                 context_narration_pred = torch.cat([context_narration_lang_only[:, :, 0].unsqueeze(2), context_narration_pred], dim=2)
#             else:
#                 context_action_pred = context_action_lang_only
#                 context_narration_pred = context_narration_lang_only
#             preds = context_action_pred, context_narration_pred

#         return losses, preds

#     def get_lang_info_loss(self, context_narration_lang_only, context_narration_gt, context_narration_gt_mask, context_duplicate_mask):
#         if torch.sum(context_narration_gt_mask) == 0:
#             return None
#         _, _, pred_T = context_narration_lang_only.size()
#         _, _, T = context_narration_gt.size()
#         losses = {}
#         if self.difference_loss:
#             gt_order = (T-1) //2
#             pred_order = (pred_T-1) // 2 
#             pred_indexes = [0, *[i for i in range(1, pred_order+1, self.subsample)], *[gt_order+i for i in range(1, pred_order+1, self.subsample)]]
#             extracted_gt = context_narration_gt[:, :, pred_indexes]
#             extracted_gt_mask = context_narration_gt_mask[:, :, pred_indexes]
#             context_differences = extracted_gt[:, :, 1:] - extracted_gt[:, :, 0].unsqueeze(2)
#             context_differences_gt = torch.cat([extracted_gt[:, :, 0].unsqueeze(2), context_differences], dim=2)
#             diff_loss = self.get_l2_loss(context_narration_lang_only, context_differences_gt, extracted_gt_mask)
#             assert diff_loss is not None
#             losses["lang_difference_loss"] = diff_loss
#             context_pred = context_narration_lang_only[:, :, 1:] + context_narration_lang_only[:, :, 0].unsqueeze(2)
#             context_pred_all = torch.cat([context_narration_lang_only[:, :, 0].unsqueeze(2), context_pred], dim=2)
#         else:
#             context_pred_all = context_narration_lang_only
                
#         lang_info_loss = pred_gt_contrastive_loss(context_pred_all, context_narration_gt, context_narration_gt_mask, context_duplicate_mask, self.loss_temperature, self.loss_across_batch, self.loss_across_pred, self.use_duplicate_mask, self.subsample)
#         losses["lang_info_loss"] = lang_info_loss
#         return losses

#     def get_order_loss(self, order_pred, gt, gt_mask):
#         B,C,T = order_pred.size()
#         order_pred = order_pred.permute(0, 2, 1)
#         order_pred = order_pred[:, self.reorder_idx]
#         gt = gt.permute(0, 2, 1)
#         gt = gt[:, self.reorder_idx]
#         gt_mask = gt_mask[:, :, self.reorder_idx]
        
#         pred_x1 = order_pred.unsqueeze(2).expand(-1, -1, T, -1)
#         pred_x2 = order_pred.unsqueeze(1).expand(-1, T, -1, -1)

#         gt_x1 = gt.unsqueeze(2).expand(-1, -1, T, -1)
#         gt_x2 = gt.unsqueeze(1).expand(-1, T, -1, -1)
        
#         pairs1 = torch.cat((pred_x1, gt_x2), dim=-1).view(B, -1, 2 * C)
#         pairs2 = torch.cat((gt_x1, pred_x2), dim=-1).view(B, -1, 2 * C)
#         pairs3 = torch.cat((pred_x1, pred_x2), dim=1).view(B, -1, 2 * C)
#         pairs4 = torch.cat((gt_x1, gt_x2), dim=-1).view(B, -1, 2 * C)
#         pairs = torch.cat((pairs1, pairs2, pairs3, pairs4), dim=0)

#         gt_x2_mask = (self.order_mask * gt_mask)
#         gt_x1_mask = (self.order_mask * gt_mask).permute(0,2,1)

#         pairs1_mask = gt_x2_mask.flatten()
#         pairs2_mask = gt_x1_mask.flatten()
#         pairs3_mask = (self.order_mask.unsqueeze(0).expand(B, -1, -1)).flatten()
#         pairs4_mask = (gt_x1_mask * gt_x2_mask).flatten()
#         mask = torch.cat((pairs1_mask, pairs2_mask, pairs3_mask, pairs4_mask), dim=0).flatten()
        
#         out = self.order_mlp(pairs).view(-1, 2)
#         targets = self.order_gt.flatten().repeat(4).repeat(B)
#         order_loss = F.cross_entropy(out, targets, reduction="none")
#         order_loss = torch.sum(order_loss * mask) / (torch.sum(mask) + 1e-6)
#         return order_loss
    
#     def get_l2_loss(self, pred, gt, gt_mask):
#         if torch.sum(gt_mask) == 0:
#             return None
#         curr_loss = nn.functional.mse_loss(pred, gt, reduction="none")  # (B, C, 2 * context_action_order + 1), (B, C, 2 * context_action_order + 1) -> (B, 2 * context_action_order + 1)
#         curr_loss = torch.sum(curr_loss * gt_mask) / (torch.sum(gt_mask))
#         return curr_loss
        
#     def get_vis_info_loss(self, context_action_lang_only, context_action_gt, context_action_gt_mask, context_duplicate_mask):
#         if torch.sum(context_action_gt_mask) == 0:
#             return None
#         _, _, pred_T = context_action_lang_only.size()
#         _, _, T = context_action_gt.size()
#         losses = {}
#         if self.difference_loss:
#             gt_order = (T-1) //2
#             pred_order = (pred_T-1) // 2 
#             pred_indexes = [0, *[i for i in range(1, pred_order+1, self.subsample)], *[gt_order+i for i in range(1, pred_order+1, self.subsample)]]
#             extracted_gt = context_action_gt[:, :, pred_indexes]
#             extracted_gt_mask = context_action_gt_mask[:, :, pred_indexes]
#             context_differences = extracted_gt[:, :, 1:] - extracted_gt[:, :, 0].unsqueeze(2)
#             context_differences_gt = torch.cat([extracted_gt[:, :, 0].unsqueeze(2), context_differences], dim=2)
#             diff_loss = self.get_l2_loss(context_action_lang_only, context_differences_gt, extracted_gt_mask)
#             assert diff_loss is not None
#             losses["vis_difference_loss"] = diff_loss
#             context_pred = context_action_lang_only[:, :, 1:] + context_action_lang_only[:, :, 0].unsqueeze(2)
#             context_pred_all = torch.cat([context_action_lang_only[:, :, 0].unsqueeze(2), context_pred], dim=2)
#         else:
#             context_pred_all = context_action_lang_only
#         vis_info_loss = pred_gt_contrastive_loss(context_pred_all, context_action_gt, context_action_gt_mask, context_duplicate_mask, self.loss_temperature, self.loss_across_batch, self.loss_across_pred, self.use_duplicate_mask, self.subsample)
#         losses["vis_info_loss"] = vis_info_loss
#         return losses
