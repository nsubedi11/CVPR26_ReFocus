import torch
from torch import nn
from torch.nn import functional as F

from .models import register_backbone
from .blocks import (get_sinusoid_encoding, TransformerBlock, MLP, FlashTransformerBlock, MaskedConv1D, LayerNorm, CrossModalDualEncoders, QCAA_Lang)
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.ops.triton.layer_norm import layer_norm_fn


@register_backbone("convTransformer")
class ConvTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """

    def __init__(
            self,
            n_vid_in,  # input video feature dimension
            n_txt_in,  # input text feature dimension
            n_embd,  # embedding dimension (after convolution)
            n_head,  # number of head for self-attention in transformers
            n_embd_ks,  # conv kernel size of the embedding network
            max_len,  # max sequence length
            arch=(1,2,2,2,6),  # (#linear, #vid_transformer, #txt_transformer, #crosstransformer, #pyramidtransformer)
            mha_win_size=-1,  # size of local window for mha
            scale_factor=2,  # dowsampling rate for the branch,
            with_ln=False,  # if to attach layernorm after conv
            attn_pdrop=0.0,  # dropout rate for the attention map
            proj_pdrop=0.0,  # dropout rate for the projection / MLP
            path_pdrop=0.0,  # droput rate for drop path
            use_abs_pe=False,  # use absolute position embedding
            use_rel_pe=False,  # use relative position embedding
            gt_provided=False,  # if ground truth is provided
            gt_provided_cfg=None,  # configuration for gt_provided
            predict_context_action_cfg=None,  # configuration for predict_context_action
    ):
        super().__init__()

        add_language = predict_context_action_cfg["add_language"]
        freeze_qcaa = predict_context_action_cfg["freeze_qcaa"]
        lang_prob = predict_context_action_cfg["lang_prob"]
        context_action_order = predict_context_action_cfg["context_action_order"]
        uniform_vid_sample = predict_context_action_cfg["uniform_vid_sample"]
        qcaa_layers = predict_context_action_cfg["layers"]
        qcaa_embd = predict_context_action_cfg["embd_dim"]
        qcaa_n_head = predict_context_action_cfg["n_head"]
        self.query_dropout = predict_context_action_cfg["query_dropout"]

        # random_anticipation = predict_context_action_cfg["random_anticipation"]
        context_action_use_pos = predict_context_action_cfg["use_pos"]
        self.predict_context_action = context_action_order >=0
        assert len(arch) == 5
        assert not (gt_provided and self.predict_context_action), "gt_provided and predict_context_action cannot be both True"

        if gt_provided:
            add_language = gt_provided_cfg["add_language"]
            context_action_order = gt_provided_cfg["context_action_order"]
            
        if not gt_provided and not self.predict_context_action:
            assert add_language, "Language must be added if context action is not predicted or gt is not provided"
        assert not (gt_provided and self.predict_context_action), "gt_provided and predict_context_action cannot be both True"
        self.arch = arch
        self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe
        self.add_language = add_language
        self.lang_prob = lang_prob
        self.context_action_order = context_action_order
        # self.random_anticipation = random_anticipation

        pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd ** 0.5)
        self.register_buffer("pos_embd", pos_embd, persistent=False)

        # vid_embedding network using convs
        self.vid_embd = nn.ModuleList()
        self.vid_embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
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
        self.qcaa = QCAA_Lang(
                lin_embed_layers=1,
                txt_n_layers=qcaa_layers,
                n_head=qcaa_n_head,
                n_txt_in=n_txt_in,
                n_embd=qcaa_embd,
                n_embd_ks=n_embd_ks,
                n_vid_in=n_vid_in,
                max_len=max_len,
                mha_win_size=-1,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                with_ln=with_ln,
                context_action_order=context_action_order,
                context_action_use_pos=context_action_use_pos,
                use_abs_pe=use_abs_pe,
                use_rel_pe=use_rel_pe,
                uniform_vid_sample=uniform_vid_sample,
            )
        
        
        self.qcaa_lin = MaskedConv1D(
            qcaa_embd, n_embd, 1,
            stride=1, padding=0,
        )
        self.qcaa_norm = LayerNorm(n_embd)

        self.vid_stem = nn.ModuleList([TransformerBlock(
            n_embd, n_head,
            n_ds_strides=(1, 1),
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
            mha_win_size=self.mha_win_size,
            use_rel_pe=self.use_rel_pe,
        ) for _ in range(arch[2])])


        self.vid_text_stem = CrossModalDualEncoders(
            num_layers=arch[3],
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

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()

        for idx in range(arch[4]):
            self.branch.append(TransformerBlock(
                n_embd, n_head,
                n_ds_strides=(self.scale_factor, self.scale_factor),
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=self.mha_win_size,
                use_rel_pe=self.use_rel_pe,
                use_cross_modal=False,
            )
        )

        # init weights
        self.apply(self.__init_weights__)

        if predict_context_action_cfg.get("resume_path", False):
            qcaa_state_dict = torch.load(predict_context_action_cfg["resume_path"])
            qcaa_state_dict = {k.replace("qcaa.", ""): v for k, v in qcaa_state_dict.items() if k.startswith("qcaa.")}
            missing_keys, unexpected_keys = self.qcaa.load_state_dict(qcaa_state_dict)
            assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"
            assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"
            if freeze_qcaa:
                self.qcaa.requires_grad_(False)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self,
     src_vid, 
     src_vid_mask, 
     src_txt, 
     src_txt_mask,
     vid_lens,
     context_action_vis_gt=None,
     ):
        assert context_action_vis_gt is None, "Context action visual ground truth is not supported"
        # if self.gt_provided:
        #     assert context_action_vis_gt is not None, "Context action visual ground truth must be provided if gt_provided is True"
        
        B, C, T = src_vid.size()

        src_query, src_query_mask, context_action_lang_only, context_narration_lang_only = self.qcaa(
            src_vid, src_vid_mask, src_txt, src_txt_mask, vid_lens
        )

        src_query, src_query_mask = self.qcaa_lin(src_query, src_query_mask)
        src_query = self.relu(self.qcaa_norm(src_query))

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

        if self.training and torch.rand(1).item() > self.lang_prob:
            src_query = src_query[:,:, :2*self.context_action_order+1]
            src_query_mask = src_query_mask[:, :, :2*self.context_action_order+1]

        _, _, queryT = src_query.size()
        
        if self.training and self.query_dropout > 0:
            for _ in range(1000):
                drop_mask = (torch.rand(B, 1, queryT, device=src_query.device) > self.query_dropout).float()
                temp_mask = (src_query_mask * drop_mask)
                if (temp_mask.view(B, queryT).sum(-1)>0).all():
                    break
            
            if not (temp_mask.view(B, queryT).sum(-1)>0).all():
                assert False, "MASK FAILED"


            # Apply the dropout mask
            src_query = src_query * drop_mask  # Set dropped tokens to 0
            src_query_mask = src_query_mask * drop_mask  # Update the mask accordingly

        src_vid, src_query = self.vid_text_stem(src_vid, src_vid_mask, src_query, src_query_mask)

        
        out_feats = tuple()
        out_masks = tuple()
        # 1x resolution
        out_feats += (src_vid,)
        out_masks += (src_vid_mask,)

        # main branch with downsampling
        for idx in range(len(self.branch)):
            src_vid, src_vid_mask  = self.branch[idx](src_vid, src_vid_mask, src_query, src_query_mask)
            out_feats += (src_vid,)
            out_masks += (src_vid_mask,)

        return out_feats, out_masks, src_query, src_query_mask, context_action_lang_only, context_narration_lang_only


@register_backbone("flashTransformer")
class FlashTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """

    def __init__(
            self,
            n_vid_in,  # input video feature dimension
            n_txt_in,  # input text feature dimension
            n_embd,  # embedding dimension (after convolution)
            n_head,  # number of head for self-attention in transformers
            n_embd_ks,  # conv kernel size of the embedding network
            max_len,  # max sequence length
            arch=(1,2,2,2,6),  # (#linear, #vid_transformer, #txt_transformer, #crosstransformer, #pyramidtransformer)
            mha_win_size=-1,  # size of local window for mha
            scale_factor=2,  # dowsampling rate for the branch,
            with_ln=False,  # if to attach layernorm after conv
            attn_pdrop=0.0,  # dropout rate for the attention map
            proj_pdrop=0.0,  # dropout rate for the projection / MLP
            path_pdrop=0.0,  # droput rate for drop path
            use_abs_pe=False,  # use absolute position embedding
            use_rel_pe=False,  # use relative position embedding
            gt_provided=False,  # if ground truth is provided
            gt_provided_cfg=None,  # configuration for gt_provided
            predict_context_action_cfg=None,  # configuration for predict_context_action
            falm_scale=-1,
    ):
        super().__init__()
        print("USING FLASH TRANSFORMER")
        
        add_language = predict_context_action_cfg["add_language"]
        freeze_qcaa = predict_context_action_cfg["freeze_qcaa"]
        lang_prob = predict_context_action_cfg["lang_prob"]
        context_action_order = predict_context_action_cfg["context_action_order"]
        uniform_vid_sample = predict_context_action_cfg["uniform_vid_sample"]
        qcaa_layers = predict_context_action_cfg["layers"]
        qcaa_embd = predict_context_action_cfg["embd_dim"]
        qcaa_n_head = predict_context_action_cfg["n_head"]
        self.query_dropout = predict_context_action_cfg["query_dropout"]

        # random_anticipation = predict_context_action_cfg["random_anticipation"]
        context_action_use_pos = predict_context_action_cfg["use_pos"]
        self.predict_context_action = context_action_order >=0
        assert len(arch) == 5
        assert not (gt_provided and self.predict_context_action), "gt_provided and predict_context_action cannot be both True"

        if gt_provided:
            add_language = gt_provided_cfg["add_language"]
            context_action_order = gt_provided_cfg["context_action_order"]
            
        if not gt_provided and not self.predict_context_action:
            assert add_language, "Language must be added if context action is not predicted or gt is not provided"
        assert not (gt_provided and self.predict_context_action), "gt_provided and predict_context_action cannot be both True"
        self.arch = arch
        self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe
        self.add_language = add_language
        self.lang_prob = lang_prob
        self.context_action_order = context_action_order
        self.n_embd = n_embd
        # self.random_anticipation = random_anticipation

        pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd ** 0.5)
        self.register_buffer("pos_embd", pos_embd, persistent=False)

        # vid_embedding network using convs
        self.vid_embd = nn.ModuleList()
        self.vid_embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = n_vid_in
            else:
                in_channels = n_embd
            # self.vid_embd.append(MaskedConv1D(
            #     in_channels, n_embd, n_embd_ks,
            #     stride=1, padding=n_embd_ks // 2, bias=(not with_ln)
            # )
            self.vid_embd.append(MaskedConv1D(
                in_channels, n_embd, 1,
                stride=1, padding=1 // 2, bias=(not with_ln)
            )
            )
            if with_ln:
                self.vid_embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.vid_embd_norm.append(nn.Identity())
        
        # self.qcaa = QCAA_Lang(
        #         lin_embed_layers=1,
        #         txt_n_layers=qcaa_layers,
        #         n_head=qcaa_n_head,
        #         n_txt_in=n_txt_in,
        #         n_embd=qcaa_embd,
        #         n_embd_ks=n_embd_ks,
        #         n_vid_in=n_vid_in,
        #         max_len=max_len,
        #         mha_win_size=-1,
        #         attn_pdrop=attn_pdrop,
        #         proj_pdrop=proj_pdrop,
        #         path_pdrop=path_pdrop,
        #         with_ln=with_ln,
        #         context_action_order=context_action_order,
        #         context_action_use_pos=context_action_use_pos,
        #         use_abs_pe=use_abs_pe,
        #         use_rel_pe=use_rel_pe,
        #         uniform_vid_sample=uniform_vid_sample,
        #     )
        
        # self.qcaa_lin = MaskedConv1D(
        #     qcaa_embd, n_embd, 1,
        #     stride=1, padding=0,
        # )
        # self.qcaa_norm = LayerNorm(n_embd)

        self.txt_embd = nn.ModuleList()
        self.txt_embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = n_txt_in
            else:
                in_channels = n_embd
            self.txt_embd.append(MaskedConv1D(
                in_channels, n_embd, 1,
                stride=1, padding=1 // 2, bias=(not with_ln)
            )
            )
            if with_ln:
                self.txt_embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.txt_embd_norm.append(nn.Identity())

        self.txt_stem = nn.ModuleList([FlashTransformerBlock(
            n_embd, n_head,
            n_ds_strides=(1, 1),
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
            mha_win_size=-1,
        ) for _ in range(arch[1])])

        self.vid_stem = nn.ModuleList([FlashTransformerBlock(
            n_embd, n_head,
            n_ds_strides=(1, 1),
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
            mha_win_size=self.mha_win_size,
        ) for _ in range(arch[2])])

        self.vid_text_stem = nn.ModuleList([FlashTransformerBlock(
            n_embd, n_head,
            n_ds_strides=(1, 1),
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
            mha_win_size=self.mha_win_size,
            cross_attn=True,
        ) for _ in range(arch[3])])
        
        self.bottom_norm = nn.LayerNorm(n_embd)

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        self.branch_norms = nn.ModuleList()

        for idx in range(arch[4]):
            self.branch.append(FlashTransformerBlock(
                n_embd, n_head,
                n_ds_strides=(self.scale_factor, self.scale_factor),
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=self.mha_win_size,
                )
            )
            self.branch_norms.append(nn.LayerNorm(n_embd))
        # init weights
        self.apply(self.__init_weights__)

        self.falm_scale = falm_scale
        # self.search_domain_scale_token = nn.Parameter(torch.randn(1, 1, n_embd))
        # self.search_domain_scale_mlp = nn.ModuleList([MLP(
        #     n_in=n_embd,
        #     n_out= 2,
        #     n_layers=2,
        #     n_hidden=n_embd,
        #     pdrop=proj_pdrop,
        # ) for _ in range(arch[3])])
        # for mlp in self.search_domain_scale_mlp:
        #     with torch.no_grad():
        #     # mlp.layers[-1] is linear layer
        #         nn.init.normal_(mlp.layers[-1].weight, mean=0.0, std=1e-3)
        #         mlp.layers[-1].bias.copy_(torch.tensor([1.0, 0.0]))

        # self.search_domain_mlp = MLP(
        #     n_in=2*n_embd,
        #     n_out= n_embd,
        #     n_layers=1,
        #     n_hidden=n_embd,
        #     pdrop=proj_pdrop,
        # )


        # if predict_context_action_cfg.get("resume_path", False):
        #     qcaa_state_dict = torch.load(predict_context_action_cfg["resume_path"])
        #     qcaa_state_dict = {k.replace("qcaa.", ""): v for k, v in qcaa_state_dict.items() if k.startswith("qcaa.")}
        #     missing_keys, unexpectedkeys = self.qcaa.load_state_dict(qcaa_state_dict)
        #     assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"
        #     assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"
        #     if freeze_qcaa:
        #         self.qcaa.requires_grad_(False)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self,
     src_vid, 
     src_vid_mask, 
     src_txt, 
     src_txt_mask,
     vid_lens,
     feedback_idxs,
     pred_saliency,
     falm_feats,
     context_action_vis_gt=None,
     ): 
        assert context_action_vis_gt is None, "Context action visual ground truth is not supported"
        # if self.gt_provided:
        #     assert context_action_vis_gt is not None, "Context action visual ground truth must be provided if gt_provided is True"
        
        B, vC, T = src_vid.size()
        B, tC, txtT = src_txt.size()

        for idx in range(len(self.txt_embd)):
            src_txt, src_txt_mask = self.txt_embd[idx](src_txt, src_txt_mask)
            src_txt = self.relu(self.txt_embd_norm[idx](src_txt))

        # text_pe = self.pos_embd[:, :, :txtT].expand(B, -1, -1)
        # if self.use_abs_pe and self.training:
        #     assert txtT <= self.max_len, "Reached max length."
        #     pe = self.pos_embd
        #     # add pe to x
        #     src_txt = src_txt + pe[:, :, :txtT] * src_txt_mask.to(src_txt.dtype)

        # # inference: re-interpolate position embeddings for over-length sequences
        # if self.use_abs_pe and (not self.training):
        #     if txtT >= self.max_len:
        #         pe = F.interpolate(
        #             self.pos_embd, txtT, mode='linear', align_corners=False)
        #     else:
        #         pe = self.pos_embd
        #     # add pe to x
        #     src_txt = src_txt + pe[:, :, :txtT] * src_txt_mask.to(src_txt.dtype)

        src_query, src_query_mask = src_txt, src_txt_mask
        src_query_packed, query_indices, query_cu_seqlens, query_max_seqlen_in_batch, _ = unpad_input(src_query.permute(0,2,1), src_query_mask.squeeze(1))
        query_varlen_params = {"cu_seqlens": query_cu_seqlens, "indices": query_indices, "max_seqlen": query_max_seqlen_in_batch}

        q_residual = None
        for idx in range(len(self.txt_stem)):
            src_query_packed, src_query_mask, q_residual = self.txt_stem[idx](src_query_packed, None, src_query_mask, query_varlen_params, q_residual)

        src_query_packed = layer_norm_fn(
                    src_query_packed,
                    self.bottom_norm.weight,
                    self.bottom_norm.bias,
                    residual=q_residual,
                    eps=self.bottom_norm.eps,
                    dropout_p=0.0,
                    prenorm=False,
                )

        context_action_lang_only, context_narration_lang_only = None, None
        # src_query, src_query_mask, context_action_lang_only, context_narration_lang_only = self.qcaa(
        #     src_vid, src_vid_mask, src_txt, src_txt_mask, vid_lens
        # )
        
        # TODO: dont add PE to context action tokens
        # _,_, queryT = src_query.size()
        # query_pe = self.pos_embd[:, :, :queryT].expand(B, -1, -1)
        # src_query, src_query_mask = self.qcaa_lin(src_query, src_query_mask)
        # src_query = self.relu(self.qcaa_norm(src_query))

        # vid_embedding network
        for idx in range(len(self.vid_embd)):
            src_vid, src_vid_mask = self.vid_embd[idx](src_vid, src_vid_mask)
            src_vid = self.relu(self.vid_embd_norm[idx](src_vid))

        # training: using fixed length position embeddings
        vid_pe = self.pos_embd[:, :, :T].expand(B, -1, -1)
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        # inference: re-interpolate position embeddingzs for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        src_vid_packed, vid_indices, vid_cu_seqlens, vid_max_seqlen_in_batch, _ = unpad_input(src_vid.permute(0,2,1), src_vid_mask.squeeze(1))
        vid_varlen_params = {"cu_seqlens": vid_cu_seqlens, "indices": vid_indices, "max_seqlen": vid_max_seqlen_in_batch}
        # vid_pe_packed, _, _, _, _ = unpad_input(vid_pe.permute(0,2,1), src_vid_mask.squeeze(1))
        residual = None
        # stem network
        for idx in range(len(self.vid_stem)):
            src_vid_packed, src_vid_mask, residual = self.vid_stem[idx](src_vid_packed, None, src_vid_mask, vid_varlen_params, residual)
        
        if feedback_idxs.numel() > 0:
            src_v = pad_input(src_vid_packed, vid_indices, src_vid_mask.size(0), src_vid_mask.size(2))
            res = pad_input(residual, vid_indices, src_vid_mask.size(0), src_vid_mask.size(2))
            src_v[feedback_idxs] = src_v[feedback_idxs] * pred_saliency
            res[feedback_idxs] = res[feedback_idxs] * pred_saliency
            src_vid_packed, vid_indices, vid_cu_seqlens, vid_max_seqlen_in_batch, _ = unpad_input(src_v, src_vid_mask.squeeze(1))
            vid_varlen_params = {"cu_seqlens": vid_cu_seqlens, "indices": vid_indices, "max_seqlen": vid_max_seqlen_in_batch}
            residual, _, _, _, _ = unpad_input(res, src_vid_mask.squeeze(1))

        # if feedback_idxs.numel() > 0:
        #     src_v = pad_input(src_vid_packed, vid_indices, src_vid_mask.size(0), src_vid_mask.size(2))
        #     res = pad_input(residual, vid_indices, src_vid_mask.size(0), src_vid_mask.size(2))
        #     search_domain_vid_feats = src_v[feedback_idxs].clone() 
        #     search_domain_vid_feats = torch.cat([search_domain_vid_feats, falm_feats], dim=2)
        #     search_domain_vid_feats, _ = self.search_domain_mlp(search_domain_vid_feats)
        #     src_v[feedback_idxs] = search_domain_vid_feats
        #     res_sd_feats = res[feedback_idxs].clone() 
        #     res_sd_feats = torch.cat([res_sd_feats, falm_feats], dim=2)
        #     res_sd_feats, _ = self.search_domain_mlp(res_sd_feats)
        #     res[feedback_idxs] = res_sd_feats
        #     src_vid_packed, vid_indices, vid_cu_seqlens, vid_max_seqlen_in_batch, _ = unpad_input(src_v, src_vid_mask.squeeze(1))
        #     vid_varlen_params = {"cu_seqlens": vid_cu_seqlens, "indices": vid_indices, "max_seqlen": vid_max_seqlen_in_batch}
        #     residual, _, _, _, _ = unpad_input(res, src_vid_mask.squeeze(1))

        # if feedback_idxs.numel() > 0:
        #     src_v = pad_input(src_vid_packed, vid_indices, src_vid_mask.size(0), src_vid_mask.size(2))
        #     res = pad_input(residual, vid_indices, src_vid_mask.size(0), src_vid_mask.size(2))
        #     src_v[feedback_idxs] = src_v[feedback_idxs] * pred_saliency
        #     res[feedback_idxs] = res[feedback_idxs] * pred_saliency

        #     search_domain_scale_tokens = torch.zeros(B, 1, self.n_embd, device=src_vid.device)
        #     search_domain_scale_tokens[feedback_idxs] = self.search_domain_scale_token.expand(len(feedback_idxs), -1, -1)
        #     search_domain_scale_masks = torch.zeros(B, 1, 1, device=src_vid.device, dtype=torch.bool)
        #     search_domain_scale_masks[feedback_idxs] = True

        #     src_v = torch.cat([search_domain_scale_tokens, src_v], dim=1)
        #     src_vid_mask = torch.cat([search_domain_scale_masks, src_vid_mask], dim=2)
        #     res = torch.cat([search_domain_scale_tokens, res], dim=1)

        #     src_vid_packed, vid_indices, vid_cu_seqlens, vid_max_seqlen_in_batch, _ = unpad_input(src_v, src_vid_mask.squeeze(1))
        #     vid_varlen_params = {"cu_seqlens": vid_cu_seqlens, "indices": vid_indices, "max_seqlen": vid_max_seqlen_in_batch}
        #     residual, _, _, _, _ = unpad_input(res, src_vid_mask.squeeze(1))


        if self.training and torch.rand(1).item() > self.lang_prob:
            src_query = src_query[:,:, :2*self.context_action_order+1]
            src_query_mask = src_query_mask[:, :, :2*self.context_action_order+1]
        
        # if self.training and self.query_dropout > 0:
        #     for _ in range(1000):
        #         drop_mask = (torch.rand(B, 1, queryT, device=src_query.device) > self.query_dropout).float()
        #         temp_mask = (src_query_mask * drop_mask)
        #         if (temp_mask.view(B, queryT).sum(-1)>0).all():
        #             break
            
        #     if not (temp_mask.view(B, queryT).sum(-1)>0).all():
        #         assert False, "MASK FAILED"

        #     # Apply the dropout mask
        #     src_query = src_query * drop_mask  # Set dropped tokens to 0
        #     src_query_mask = src_query_mask * drop_mask  # Update the mask accordingly

        # query_pe_packed, _, _, _, _ = unpad_input(query_pe.permute(0,2,1), src_query_mask.squeeze(1))
        # saliency_pred = []
        # span_pred = []
        # if self.search_domain_stem is not None:
        #     sal_residual = residual
        #     sal_packed = src_vid_packed
        #     for idx in range(len(self.search_domain_stem)):
        #         sal_packed, _, sal_residual = self.search_domain_stem[idx](sal_packed, None, src_vid_mask, vid_varlen_params, sal_residual, src_query_packed, None, src_query_mask, query_varlen_params)
        #         if self.search_domain_temporal_mlp is not None and feedback_idxs.numel() > 0:
        #             # saliency prediction
        #             saliency_packed = torch.sigmoid(self.search_domain_temporal_mlp(sal_packed))
        #             saliency = pad_input(saliency_packed, vid_indices, src_vid_mask.size(0), src_vid_mask.size(2))
        #             # saliency = saliency.permute(0,2,1)
        #             saliency = saliency[feedback_idxs]

        #             span_packed = torch.sigmoid(self.search_domain_span_mlp(sal_packed))
        #             span = pad_input(span_packed, vid_indices, src_vid_mask.size(0), src_vid_mask.size(2))
        #             # span = span.permute(0,2,1)
        #             span = span[feedback_idxs]
                    
        #             saliency_pred.append(saliency)
        #             span_pred.append(span)


        for idx in range(len(self.vid_text_stem)):
            src_vid_packed, src_vid_mask, residual = self.vid_text_stem[idx](src_vid_packed, None, src_vid_mask, vid_varlen_params, residual, src_query_packed, None, src_query_mask, query_varlen_params)
        # if feedback_idxs.numel() > 0:
        #     src_v = pad_input(src_vid_packed, vid_indices, src_vid_mask.size(0), src_vid_mask.size(2))
        #     res = pad_input(residual, vid_indices, src_vid_mask.size(0), src_vid_mask.size(2))
            
        #     search_domain_scale_tokens = src_v[feedback_idxs][:, :1, :]
        #     search_domain_scales = self.search_domain_scale_mlp[idx](search_domain_scale_tokens)[0].squeeze(1)  # NF x 2
        #     curr_search_domain = pred_saliency * search_domain_scales[:, 0:1].unsqueeze(1) + search_domain_scales[:, 1:2].unsqueeze(1)
        #     curr_search_domain = torch.clamp(curr_search_domain, 0.0, 1.0)
        #     src_v[feedback_idxs][:, 1:, :] = src_v[feedback_idxs][:, 1:, :] * curr_search_domain
        #     res[feedback_idxs][:, 1:, :] = res[feedback_idxs][:, 1:, :] * curr_search_domain

        #     src_vid_packed, _, _, _, _ = unpad_input(src_v, src_vid_mask.squeeze(1))
        #     residual, _, _, _, _ = unpad_input(res, src_vid_mask.squeeze(1))

        # if feedback_idxs.numel() > 0:
        #     src_v = pad_input(src_vid_packed, vid_indices, src_vid_mask.size(0), src_vid_mask.size(2))
        #     res = pad_input(residual, vid_indices, src_vid_mask.size(0), src_vid_mask.size(2))

        #     # remove the search domain scale token
        #     src_v = src_v[:, 1:, :]
        #     res = res[:, 1:, :]
        #     src_vid_mask = src_vid_mask[:, :, 1:]
        #     src_vid_packed, vid_indices, vid_cu_seqlens, vid_max_seqlen_in_batch, _ = unpad_input(src_v, src_vid_mask.squeeze(1))
        #     vid_varlen_params = {"cu_seqlens": vid_cu_seqlens, "indices": vid_indices, "max_seqlen": vid_max_seqlen_in_batch}
        #     residual, _, _, _, _ = unpad_input(res, src_vid_mask.squeeze(1))

        # if self.search_domain_stem is not None and feedback_idxs.numel() > 0:
        #     src_v = pad_input(src_vid_packed, vid_indices, src_vid_mask.size(0), src_vid_mask.size(2))
        #     res = pad_input(residual, vid_indices, src_vid_mask.size(0), src_vid_mask.size(2))
            
        #     if torch.rand(1).item() < self.search_domain_scheduler.get_current() or (not self.training):
        #         src_v[feedback_idxs] = src_v[feedback_idxs] * saliency_pred[-1]
        #         res[feedback_idxs] = res[feedback_idxs] * saliency_pred[-1]
        #     else:
        #         noisy_labels = (saliency_labels + torch.rand(B, T, 1, device=saliency_labels.device) * (self.search_domain_noise *2) - self.search_domain_noise) * src_vid_mask.permute(0, 2, 1)
        #         # clip 0 to 1
        #         noisy_labels = torch.clamp(noisy_labels, 0.0, 1.0)
        #         # print(noisy_labels.shape, src_v.shape, res.shape, feedback_idxs.shape, src_vid_mask.shape)
        #         # print(noisy_labels.min(), noisy_labels.max(), noisy_labels.mean())
        #         src_v[feedback_idxs] = src_v[feedback_idxs] * noisy_labels[feedback_idxs]
        #         res[feedback_idxs] = res[feedback_idxs] * noisy_labels[feedback_idxs]

        #     src_vid_packed, _, _, _, _ = unpad_input(src_v, src_vid_mask.squeeze(1))
        #     residual, _, _, _, _ = unpad_input(res, src_vid_mask.squeeze(1))
        
        bottom_feats_packed = layer_norm_fn(
                    src_vid_packed,
                    self.bottom_norm.weight,
                    self.bottom_norm.bias,
                    residual=residual,
                    eps=self.bottom_norm.eps,
                    dropout_p=0.0,
                    prenorm=False,
                )

        bottom_feats = pad_input(bottom_feats_packed, vid_indices, src_vid_mask.size(0), src_vid_mask.size(2))
        bottom_feats = bottom_feats.permute(0,2,1)
        
        out_feats = tuple()
        out_masks = tuple()
        # 1x resolution
        out_feats += (bottom_feats,)
        out_masks += (src_vid_mask,)

        temp_varlen_params = {"cu_seqlens": vid_cu_seqlens, "indices": vid_indices, "max_seqlen": vid_max_seqlen_in_batch}

        # main branch with downsampling
        for idx in range(len(self.branch)):
            src_vid_packed, src_vid_mask, residual, downsampled_varlen_params = self.branch[idx](src_vid_packed, None, src_vid_mask, temp_varlen_params, residual, src_query_packed, None, src_query_mask, query_varlen_params)
            # select every other pe
            # vid_pe = vid_pe[:, :, ::self.scale_factor]
            # vid_pe_packed, _, _, _, _ = unpad_input(vid_pe.permute(0,2,1), src_vid_mask.squeeze(1))
            branch_feats_packed = layer_norm_fn(
                    src_vid_packed,
                    self.branch_norms[idx].weight,
                    self.branch_norms[idx].bias,
                    residual=residual,
                    eps=self.branch_norms[idx].eps,
                    dropout_p=0.0,
                    prenorm=False,
                )
            branch_feats = pad_input(branch_feats_packed, downsampled_varlen_params["indices"], src_vid_mask.size(0), src_vid_mask.size(2))
            branch_feats = branch_feats.permute(0,2,1)
            temp_varlen_params = downsampled_varlen_params
            out_feats += (branch_feats,)
            out_masks += (src_vid_mask,)

        return out_feats, out_masks, src_query, src_query_mask, context_action_lang_only, context_narration_lang_only

