import math
import random

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from scipy.optimize import linear_sum_assignment

from mask2former_video.modeling.transformer_decoder.video_mask2former_transformer_decoder import SelfAttentionLayer,\
    CrossAttentionLayer, FFNLayer, MLP, _get_activation_fn

class VideoInstanceSequence(object):
    def __init__(self, start_time: int, matched_gt_id: int = -1, maximum_chache=10):
        self.sT = start_time
        self.eT = -1
        self.maximum_chache = maximum_chache
        self.dead = False
        self.gt_id = matched_gt_id
        self.invalid_frames = 0
        self.embeds = []
        self.pos_embeds = []
        self.pred_logits = []
        self.pred_masks = []
        self.appearance = []

        # CTVIS
        self.pos_embeds = []
        self.long_scores = []
        self.similarity_guided_pos_embed = None
        self.similarity_guided_pos_embed_list = []
        self.momentum = 0.75

    def update_pos(self, pos_embed):
        self.pos_embeds.append(pos_embed)

        if len(self.similarity_guided_pos_embed_list) == 0:
            self.similarity_guided_pos_embed = pos_embed
            self.similarity_guided_pos_embed_list.append(pos_embed)
        else:
            assert len(self.pos_embeds) > 1
            # Similarity-Guided Feature Fusion
            # https://arxiv.org/abs/2203.14208v1
            all_pos_embed = []
            for embedding in self.pos_embeds[:-1]:
                all_pos_embed.append(embedding)
            all_pos_embed = torch.stack(all_pos_embed, dim=0)

            similarity = torch.sum(torch.einsum("bc,c->b",
                                                F.normalize(all_pos_embed, dim=-1),
                                                F.normalize(pos_embed.squeeze(), dim=-1)
                                                )) / all_pos_embed.shape[0]

            # TODO, using different similarity function
            beta = max(0, similarity)
            self.similarity_guided_pos_embed = (1 - beta) * self.similarity_guided_pos_embed + beta * pos_embed
            self.similarity_guided_pos_embed_list.append(self.similarity_guided_pos_embed)

        if len(self.pos_embeds) > self.maximum_chache:
            self.pos_embeds.pop(0)

class VideoInstanceCutter(nn.Module):

    def __init__(
            self,
            hidden_dim: int = 256,
            feedforward_dim: int = 2048,
            num_head: int = 8,
            decoder_layer_num: int = 6,
            mask_dim: int = 256,
            num_classes: int = 25,
            num_new_ins: int = 100,
            training_select_threshold: float = 0.1,
            # inference
            inference_select_threshold: float = 0.1,
            kick_out_frame_num: int = 8,
            # ablation

    ):
        super().__init__()

        self.num_heads = num_head
        self.hidden_dim = hidden_dim
        self.num_layers = decoder_layer_num
        self.num_classes = num_classes
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            sa_func = CrossAttentionLayer if _ < 3 else SelfAttentionLayer
            self.transformer_self_attention_layers.append(
                sa_func(
                    d_model=hidden_dim,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=feedforward_dim,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.pos_embed = MLP(mask_dim, hidden_dim, hidden_dim, 3)

        # mask features projection
        self.mask_feature_proj = nn.Conv2d(
            mask_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.new_ins_embeds = nn.Embedding(1, hidden_dim)
        # self.disappear_embed = nn.Embedding(1, hidden_dim)

        # record previous frame information
        self.last_seq_ids = None
        self.track_queries = None
        self.track_embeds = None
        self.cur_disappear_embeds = None
        self.prev_frame_indices = None
        self.tgt_ids_for_track_queries = None
        self.disappear_fq_mask = None
        self.disappear_tgt_id = None
        self.disappear_trcQ_id = None
        self.disappeared_tgt_ids = []
        self.video_ins_hub = dict()
        self.gt_ins_hub = dict()

        self.num_new_ins = num_new_ins
        self.training_select_thr = training_select_threshold
        self.inference_select_thr = inference_select_threshold
        self.kick_out_frame_num = kick_out_frame_num

    def _clear_memory(self):
        del self.video_ins_hub
        self.video_ins_hub = dict()
        self.gt_ins_hub = dict()
        self.last_seq_ids = None
        self.track_queries = None
        self.track_embeds = None
        self.cur_disappear_embeds = None
        self.prev_frame_indices = None
        self.tgt_ids_for_track_queries = None
        self.disappear_fq_mask = None
        self.disappear_tgt_id = None
        self.disappeared_tgt_ids = []
        self.disappear_trcQ_id = None
        return

    def readout(self, read_type: str = "last"):
        assert read_type in ["last", "last_valid", "last_pos", "last_valid_pos"]

        if read_type == "last":
            out_embeds = []
            for seq_id in self.last_seq_ids:
                out_embeds.append(self.video_ins_hub[seq_id].embeds[-1])
            if len(out_embeds):
                return torch.stack(out_embeds, dim=0).unsqueeze(1)  # q, 1, c
            else:
                return torch.empty(size=(0, 1, self.hidden_dim), dtype=torch.float32).to("cuda")
        elif read_type == "last_pos":
            out_pos_embeds = []
            for seq_id in self.last_seq_ids:
                out_pos_embeds.append(self.video_ins_hub[seq_id].similarity_guided_pos_embed)
            if len(out_pos_embeds):
                return torch.stack(out_pos_embeds, dim=0).unsqueeze(1)  # q, 1, c
            else:
                return torch.empty(size=(0, 1, self.hidden_dim), dtype=torch.float32).to("cuda")
        else:
            raise NotImplementedError

    def forward(self, frame_embeds, frame_reid_embeds, mask_features, targets, frames_info, matcher,
                resume=False, using_thr=False, stage=1):
        ori_mask_features = mask_features
        mask_features_shape = mask_features.shape
        mask_features = self.mask_feature_proj(mask_features.flatten(0, 1)).reshape(*mask_features_shape)  # (b, t, c, h, w)

        frame_embeds = frame_embeds.permute(2, 3, 0, 1)  # t, q, b, c
        T, fQ, B, _ = frame_embeds.shape
        assert B == 1
        all_outputs = []

        for i in range(T):
            if i == 0 and resume is False:
                self._clear_memory()
            det_outputs, ms_outputs = [], []
            new_ins_embeds = self.new_ins_embeds.weight.unsqueeze(1).repeat(self.num_new_ins, B, 1)  # nq, b, c
            single_frame_embeds = frame_embeds[i]  # q, b, c
            targets_i = targets[i].copy()

            frame_queries_pos = self.get_mask_pos_embed(frames_info["pred_masks"][i][0][None],
                                                        ori_mask_features[:, i, ...])

            matched_frame_query_id_for_each_trc_query = self.match_with_embeds(single_frame_embeds, frames_info, i)
            self.track_embeds = frame_queries_pos[matched_frame_query_id_for_each_trc_query] if self.track_queries is not None else None

            output = new_ins_embeds
            for j in range(3):
                output = self.transformer_cross_attention_layers[j](
                    output, single_frame_embeds,
                    query_pos=frame_queries_pos, pos=frame_queries_pos,
                )
                output = self.transformer_self_attention_layers[j](
                    output,
                    torch.cat([self.track_queries, output], dim=0) if self.track_queries is not None else output,
                    query_pos=frame_queries_pos,
                    pos=torch.cat([self.track_embeds, frame_queries_pos], dim=0) if self.track_embeds is not None else frame_queries_pos,
                )
                output = self.transformer_ffn_layers[j](output)
                det_outputs.append(output)

            trc_det_queries = torch.cat([self.track_queries, det_outputs[-1]], dim=0) if self.track_queries is not None else det_outputs[-1]
            trc_det_queries_pos = torch.cat([self.track_embeds, frame_queries_pos], dim=0) if self.track_embeds is not None else frame_queries_pos

            for j in range(3, self.num_layers):
                trc_det_queries = self.transformer_cross_attention_layers[j](
                    trc_det_queries, single_frame_embeds,
                    query_pos=trc_det_queries_pos, pos=frame_queries_pos
                )
                trc_det_queries = self.transformer_self_attention_layers[j](trc_det_queries)
                trc_det_queries = self.transformer_ffn_layers[j](trc_det_queries)
                ms_outputs.append(trc_det_queries)

            det_outputs = torch.stack(det_outputs, dim=0)  # (L1, nQ, B, C)
            det_outputs_class, det_outputs_mask = self.prediction(det_outputs, mask_features[:, i, ...])
            ms_outputs = torch.stack(ms_outputs, dim=0)  # (L2, tQ+nQ, B, C)
            ms_outputs_class, ms_outputs_mask = self.prediction(ms_outputs, mask_features[:, i, ...])
            out_dict = {
                "pred_logits": ms_outputs_class[-1],  # b, q, k+1
                "pred_masks": ms_outputs_mask[-1],  # b, q, h, w
            }

            # matching with gt
            indices, new_ins_indices = matcher(out_dict, targets_i, self.prev_frame_indices)
            out_dict.update({
                "indices": indices
            })

            # aux loss
            out_dict.update({
                "aux_outputs": self._set_aux_loss(ms_outputs_class, ms_outputs_mask, self.disappear_tgt_id),
                "disappear_tgt_id": -10000 if self.disappear_tgt_id is None else self.disappear_tgt_id,
            })

            # new ins detection loss
            det_out_dict = {
                "pred_logits": det_outputs_class[-1],
                "pred_masks": det_outputs_mask[-1],
                "indices": new_ins_indices,
                "aux_outputs": self._set_aux_loss(det_outputs_class, det_outputs_mask),
                "disappear_tgt_id": -10000,
            }

            out_dict.update({
                "det_outputs": det_out_dict
            })

            all_outputs.append(out_dict)

            if stage == 1:
                # as the quality of the output queries is not as good as to be track queries in the early training phase.
                tgt_ids_for_each_query = torch.full(size=(ms_outputs.shape[1],), dtype=torch.int64,
                                                    fill_value=-1).to("cuda")
                tgt_ids_for_each_query[indices[0][0]] = indices[0][1]
                activated_queries_bool = torch.ones(size=(ms_outputs.shape[1], )).to("cuda") < 0
            elif stage == 2:
                tgt_ids_for_each_query = torch.full(size=(ms_outputs.shape[1],), dtype=torch.int64,
                                                    fill_value=-1).to("cuda")
                tgt_ids_for_each_query[indices[0][0]] = indices[0][1]
                select_queries_bool = torch.rand(size=(len(indices[0][0]),), dtype=torch.float32).to("cuda") > 0.5
                kick_out_src_indices = indices[0][0][select_queries_bool]
                activated_queries_bool = torch.ones(size=(ms_outputs.shape[1], )).to("cuda") < 0
                activated_queries_bool[indices[0][0]] = True
                activated_queries_bool[kick_out_src_indices] = False
            elif stage == 3:
                tgt_ids_for_each_query = torch.full(size=(ms_outputs.shape[1],), dtype=torch.int64,
                                                    fill_value=-1).to("cuda")
                tgt_ids_for_each_query[indices[0][0]] = indices[0][1]
                activated_queries_bool = torch.ones(size=(ms_outputs.shape[1],)).to("cuda") < 0
                activated_queries_bool[indices[0][0]] = True
                pred_scores = torch.max(ms_outputs_class[-1, 0].softmax(-1)[:, :-1], dim=-1)[0]
                activated_queries_bool = activated_queries_bool & (pred_scores > self.training_select_thr)
            else:
                raise NotImplementedError

            self.track_queries = ms_outputs[-1][activated_queries_bool]  # q', b, c
            select_query_tgt_ids = tgt_ids_for_each_query[activated_queries_bool]  # q',
            prev_src_indices = torch.nonzero(select_query_tgt_ids + 1).squeeze(-1)
            prev_tgt_indices = torch.index_select(select_query_tgt_ids, dim=0, index=prev_src_indices)
            self.prev_frame_indices = (prev_src_indices, prev_tgt_indices)
            self.tgt_ids_for_track_queries = tgt_ids_for_each_query[activated_queries_bool]

        return all_outputs

    def inference(self, frame_embeds, frame_reid_embeds, mask_features, frames_info, start_frame_id, resume=False, to_store="cpu"):
        ori_mask_features = mask_features
        mask_features_shape = mask_features.shape
        mask_features = self.mask_feature_proj(mask_features.flatten(0, 1)).reshape(
            *mask_features_shape)  # (b, t, c, h, w)

        frame_embeds = frame_embeds.permute(2, 3, 0, 1)  # t, q, b, c
        T, fQ, B, _ = frame_embeds.shape
        assert B == 1

        for i in range(T):
            if i == 0 and resume is False:
                self._clear_memory()
            det_outputs, ms_outputs = [], []
            new_ins_embeds = self.new_ins_embeds.weight.unsqueeze(1).repeat(self.num_new_ins, B, 1)  # nq, b, c
            single_frame_embeds = frame_embeds[i]  # q, b, c

            frame_queries_pos = self.get_mask_pos_embed(frames_info["pred_masks"][i][0][None],
                                                        ori_mask_features[:, i, ...])

            matched_frame_query_id_for_each_trc_query = self.match_with_embeds(single_frame_embeds, frames_info, i)
            self.track_embeds = frame_queries_pos[
                matched_frame_query_id_for_each_trc_query] if self.track_queries is not None else None

            output = new_ins_embeds
            for j in range(3):
                output = self.transformer_cross_attention_layers[j](
                    output, single_frame_embeds,
                    query_pos=frame_queries_pos, pos=frame_queries_pos,
                )
                output = self.transformer_self_attention_layers[j](
                    output,
                    torch.cat([self.track_queries, output], dim=0) if self.track_queries is not None else output,
                    query_pos=frame_queries_pos,
                    pos=torch.cat([self.track_embeds, frame_queries_pos],
                                  dim=0) if self.track_embeds is not None else frame_queries_pos,
                )
                output = self.transformer_ffn_layers[j](output)
                det_outputs.append(output)

            trc_det_queries = torch.cat([self.track_queries, det_outputs[-1]],
                                        dim=0) if self.track_queries is not None else det_outputs[-1]
            trc_det_queries_pos = torch.cat([self.track_embeds, frame_queries_pos],
                                            dim=0) if self.track_embeds is not None else frame_queries_pos

            for j in range(3, self.num_layers):
                trc_det_queries = self.transformer_cross_attention_layers[j](
                    trc_det_queries, single_frame_embeds,
                    query_pos=trc_det_queries_pos, pos=frame_queries_pos
                )
                trc_det_queries = self.transformer_self_attention_layers[j](trc_det_queries)
                trc_det_queries = self.transformer_ffn_layers[j](trc_det_queries)
                ms_outputs.append(trc_det_queries)

            # det_outputs = torch.stack(det_outputs, dim=0)  # (L1, nQ, B, C)
            # det_outputs_class, det_outputs_mask = self.prediction(det_outputs, mask_features[:, i, ...])
            ms_outputs = torch.stack(ms_outputs, dim=0)  # (L2, tQ+nQ, B, C)
            ms_outputs_class, ms_outputs_mask = self.prediction(ms_outputs, mask_features[:, i, ...])

            cur_seq_ids = []
            pred_scores = torch.max(ms_outputs_class[-1, 0].softmax(-1)[:, :-1], dim=1)[0]
            valid_queries_bool = pred_scores > self.inference_select_thr
            for k, valid in enumerate(valid_queries_bool):
                if self.last_seq_ids is not None and k < len(self.last_seq_ids):
                    seq_id = self.last_seq_ids[k]
                else:
                    seq_id = random.randint(0, 100000)
                    while seq_id in self.video_ins_hub:
                        seq_id = random.randint(0, 100000)
                    assert not seq_id in self.video_ins_hub
                if valid:
                    if not seq_id in self.video_ins_hub:
                        self.video_ins_hub[seq_id] = VideoInstanceSequence(start_frame_id + i, seq_id)
                    self.video_ins_hub[seq_id].embeds.append(ms_outputs[-1, k, 0, :])
                    self.video_ins_hub[seq_id].pred_logits.append(ms_outputs_class[-1, 0, k, :])
                    self.video_ins_hub[seq_id].pred_masks.append(
                        ms_outputs_mask[-1, 0, k, ...].to(to_store).to(torch.float32))
                    self.video_ins_hub[seq_id].invalid_frames = 0
                    self.video_ins_hub[seq_id].appearance.append(True)

                    cur_seq_ids.append(seq_id)
                elif self.last_seq_ids is not None and seq_id in self.last_seq_ids:
                    self.video_ins_hub[seq_id].invalid_frames += 1
                    if self.video_ins_hub[seq_id].invalid_frames >= self.kick_out_frame_num:
                        self.video_ins_hub[seq_id].dead = True
                        continue
                    self.video_ins_hub[seq_id].embeds.append(ms_outputs[-1, k, 0, :])
                    self.video_ins_hub[seq_id].pred_logits.append(ms_outputs_class[-1, 0, k, :])
                    self.video_ins_hub[seq_id].pred_masks.append(
                        ms_outputs_mask[-1, 0, k, ...].to(to_store).to(torch.float32))
                    # self.video_ins_hub[seq_id].pred_masks.append(
                    #     torch.zeros_like(outputs_mask[-1, 0, k, ...]).to(to_store).to(torch.float32))
                    self.video_ins_hub[seq_id].appearance.append(False)

                    cur_seq_ids.append(seq_id)
            self.last_seq_ids = cur_seq_ids
            self.track_queries = self.readout("last")

    def match_with_embeds(self, cur_frame_embeds, frames_info, frame_id):
        if self.track_queries is None:
            return None
        ref_embeds, cur_embeds = self.track_queries.detach()[:, 0, :], cur_frame_embeds.detach()[:, 0, :]
        ref_embeds = ref_embeds / (ref_embeds.norm(dim=1)[:, None] + 1e-6)
        cur_embeds = cur_embeds / (cur_embeds.norm(dim=1)[:, None] + 1e-6)
        cos_sim = torch.mm(ref_embeds, cur_embeds.transpose(0, 1))
        C = 1 - cos_sim
        largest_cost_indices = torch.max(C, dim=1)[1]  # q',

        if self.training:
            tgt_id_for_each_trc_query = self.tgt_ids_for_track_queries
            frame_tgt2src = {t.item(): s for s, t in zip(frames_info["indices"][frame_id][0][0],
                                                         frames_info["indices"][frame_id][0][1])}
            matched_src_id_for_each_trc_query = torch.full_like(tgt_id_for_each_trc_query, fill_value=-1)
            for idx, tgt_id in enumerate(tgt_id_for_each_trc_query):
                if tgt_id.item() in frame_tgt2src:
                    matched_src_id_for_each_trc_query[idx] = frame_tgt2src[tgt_id.item()]
                else:
                    # could be tgt_id == -1, a false positive
                    # or this track query disappears in the current frame
                    matched_src_id_for_each_trc_query[idx] = largest_cost_indices[idx]
        else:
            indices = linear_sum_assignment(C.cpu())
            tid2sid = {tid: sid for tid, sid in zip(indices[0], indices[1])}
            matched_src_id_for_each_trc_query = torch.full(size=(self.track_queries.shape[0], ), fill_value=-1, dtype=torch.int64).to("cuda")
            for idx in range(matched_src_id_for_each_trc_query.shape[0]):
                if idx in tid2sid:
                    print("match_with_embeds function test mode")
                    matched_src_id_for_each_trc_query[idx] = tid2sid[idx]
                else:
                    matched_src_id_for_each_trc_query[idx] = largest_cost_indices[idx]

        return matched_src_id_for_each_trc_query

    def prediction(self, outputs, mask_features):
        # outputs (l, q, b, c)
        # mask_features (b, c, h, w)
        decoder_output = self.decoder_norm(outputs.transpose(1, 2))
        outputs_class = self.class_embed(decoder_output)  # l, b, q, k+1
        mask_embed = self.mask_embed(decoder_output)      # l, b, q, c
        outputs_mask = torch.einsum("lbqc,bchw->lbqhw", mask_embed, mask_features)

        return outputs_class, outputs_mask

    def get_mask_pos_embed(self, mask, mask_features):
        """
        mask: b, q, h, w
        mask_features: b, c, h, w
        """
        pos_embeds_list = []
        num_chunk = mask.shape[1] // 50 + 1
        for i in range(num_chunk):
            start = i * 50
            end = start + 50 if start + 50 < mask.shape[1] else mask.shape[1]

            seg_mask = (mask[:, start:end, :, :].sigmoid() > 0.5).to("cuda")
            mask_feats = seg_mask[:, :, None, :, :] * mask_features[:, None, ...]  # b, q, c, h, w
            pos_embeds = torch.sum(mask_feats.flatten(3, 4), dim=-1) / (
                    torch.sum(seg_mask.flatten(2, 3), dim=-1, keepdim=True) + 1e-8)
            pos_embeds = self.pos_embed(pos_embeds)
            pos_embeds_list.append(pos_embeds.transpose(0, 1))

        return torch.cat(pos_embeds_list, dim=0)

    # def filter_new_ins(self, pred_logits, pred_masks):
    #     """
    #     pred_logits: tq+nq, k+1
    #     pred_masks: tq+nq, h, w
    #     """
    #     new_ins_scores = F.softmax(pred_logits[-self.num_new_ins:, :], dim=-1)[:, :-1]  # tq+nq, k
    #     max_scores, max_indices = torch.max(new_ins_scores, dim=1)
    #     _, sorted_indices = torch.sort(max_scores, dim=0, descending=True)
    #
    #
    #     if self.track_queries is None or self.track_queries.shape[0] == 0:


    @torch.jit.unused
    def _set_aux_loss(self, outputs_cls, outputs_mask, disappear_tgt_id=None):
        return [{"pred_logits": a,
                 "pred_masks": b,
                 "disappear_tgt_id": -10000 if disappear_tgt_id is None else disappear_tgt_id,
                 } for a, b
                in zip(outputs_cls[:-1], outputs_mask[:-1])]
