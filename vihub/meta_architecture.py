import logging
import random
from typing import Tuple, List
import einops
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

from mask2former_video.utils.memory import retry_if_cuda_oom
from dvis_Plus.meta_architecture import MinVIS
from .matcher import FrameMatcher, NewInsHungarianMatcher
from .criterion import ViHubCriterion
from .track_module import VideoInstanceCutter
from .utils.mask_nms import mask_nms


@META_ARCH_REGISTRY.register()
class ViHub_online(MinVIS):
    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            sem_seg_head: nn.Module,
            criterion: nn.Module,
            num_queries: int,
            object_mask_threshold: float,
            overlap_threshold: float,
            metadata,
            size_divisibility: int,
            sem_seg_postprocess_before_inference: bool,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            # video head
            tracker: nn.Module,
            num_frames: int,
            window_inference: bool,
            frame_matcher: nn.Module,
            new_ins_matcher: nn.Module,
            inference_select_thr: float,
            vihub_criterion: nn.Module,
            using_thr: bool,
            # inference
            task: str,
            max_num: int,
            max_iter_num: int,
            window_size: int,
            noise_frame_num: int = 2,
            temporal_score_type: str = 'mean',
            mask_nms_thr: float = 0.5,
            # training
            using_frame_num: List = None,
            increasing_step: List = None,
    ):
        super().__init__(
            backbone=backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
            num_queries=num_queries,
            object_mask_threshold=object_mask_threshold,
            overlap_threshold=overlap_threshold,
            metadata=metadata,
            size_divisibility=size_divisibility,
            sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            # video
            num_frames=num_frames,
            window_inference=window_inference,
        )
        # frozen the segmenter
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.sem_seg_head.parameters():
            p.requires_grad_(False)

        self.tracker = tracker

        self.frame_matcher = frame_matcher
        self.new_ins_matcher = new_ins_matcher
        self.inference_select_thr = inference_select_thr
        self.vihub_criterion = vihub_criterion
        self.using_thr = using_thr

        self.max_num = max_num
        self.iter = 0
        self.max_iter_num = max_iter_num
        self.window_size = window_size
        self.task = task
        assert self.task in ['vis', 'vss', 'vps'], "Only support vis, vss and vps !"
        inference_dict = {
            'vis': self.inference_video_vis,
            'vss': self.inference_video_vss,
            'vps': self.inference_video_vps,
        }
        self.inference_video_task = inference_dict[self.task]
        self.noise_frame_num = noise_frame_num
        self.temporal_score_type = temporal_score_type
        self.mask_nms_thr = mask_nms_thr

        self.using_frame_num = using_frame_num
        self.increasing_step = increasing_step

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        frame_matcher = FrameMatcher(
            cost_class=class_weight,
            cost_dice=dice_weight,
            cost_mask=mask_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        new_ins_matcher = NewInsHungarianMatcher(
            cost_class=class_weight,
            cost_dice=dice_weight,
            cost_mask=mask_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            num_new_ins=cfg.MODEL.VIDEO_HEAD.NUM_NEW_INS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight,
                       "loss_ce_det": class_weight, "loss_mask_det": mask_weight, "loss_dice_det": dice_weight,}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers * 10 - 1):  # more is harmless
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        vihub_criterion = ViHubCriterion(
            sem_seg_head.num_classes,
            new_ins_matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            num_new_ins=cfg.MODEL.VIDEO_HEAD.NUM_NEW_INS,
        )

        tracker = VideoInstanceCutter(
            hidden_dim=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            feedforward_dim=cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD,
            num_head=cfg.MODEL.MASK_FORMER.NHEADS,
            decoder_layer_num=cfg.MODEL.TRACKER.DECODER_LAYERS,
            mask_dim=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            num_new_ins=cfg.MODEL.VIDEO_HEAD.NUM_NEW_INS,
            training_select_threshold=cfg.MODEL.VIDEO_HEAD.TRAINING_SELECT_THRESHOLD,
            inference_select_threshold=cfg.MODEL.VIDEO_HEAD.INFERENCE_SELECT_THRESHOLD,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": None,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # video
            "tracker": tracker,
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "window_inference": cfg.MODEL.MASK_FORMER.TEST.WINDOW_INFERENCE,
            "frame_matcher": frame_matcher,
            "new_ins_matcher": new_ins_matcher,
            "inference_select_thr": cfg.MODEL.VIDEO_HEAD.INFERENCE_SELECT_THRESHOLD,
            "vihub_criterion": vihub_criterion,
            "using_thr": cfg.MODEL.VIDEO_HEAD.USING_THR,
            # inference
            "task": cfg.MODEL.MASK_FORMER.TEST.TASK,
            "noise_frame_num": cfg.MODEL.VIDEO_HEAD.NOISE_FRAME_NUM,
            "temporal_score_type": cfg.MODEL.VIDEO_HEAD.TEMPORAL_SCORE_TYPE,
            "max_num": cfg.MODEL.MASK_FORMER.TEST.MAX_NUM,
            "max_iter_num": cfg.SOLVER.MAX_ITER,
            "window_size": cfg.MODEL.MASK_FORMER.TEST.WINDOW_SIZE,
            "mask_nms_thr": cfg.MODEL.VIDEO_HEAD.MASK_NMS_THR,
            # training
            "using_frame_num": cfg.INPUT.USING_FRAME_NUM,
            "increasing_step": cfg.INPUT.STEPS,
        }

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        # for running demo on very long videos
        if 'keep' in batched_inputs[0].keys():
            self.keep = batched_inputs[0]['keep']
        else:
            self.keep = False

        if self.using_frame_num is None:
            images = []
            for video in batched_inputs:
                for frame in video["image"]:
                    images.append(frame.to(self.device))
            select_fi_set = [i for i in range(len(images))]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)
        else:
            if self.iter < self.increasing_step[0]:
                using_frame_num = self.using_frame_num[0]
                self.using_thr = False
            else:
                using_frame_num = self.using_frame_num[1]
                self.using_thr = True
            images = []

            video_length = len(batched_inputs[0]["image"])
            if using_frame_num <= 0 or using_frame_num > video_length:
                using_frame_num = video_length
            if using_frame_num == video_length:
                select_fi_set = np.arange(0, video_length)
            else:
                start_fi = random.randint(0, using_frame_num - 1)
                end_fi = start_fi + using_frame_num - 1
                if end_fi >= video_length:
                    start_fi = video_length - using_frame_num
                    end_fi = video_length - 1
                select_fi_set = np.arange(start_fi, end_fi + 1)
            assert len(select_fi_set) == using_frame_num

            for video in batched_inputs:
                for fi, frame in enumerate(video["image"]):
                    if fi in select_fi_set:
                        images.append(frame.to(self.device))
            if self.iter in [1000, 4000, 5000, 8000, 9999, 11000, 14000, 15000, 16000, 17000, 19000,
                             20000, 24000, 25000, 30000, 35000, 34000, 37000, 40000, 50000, 59000]:
                print(f"iter: {self.iter}, length of images: {len(images)}, using_thr: {self.using_thr}")
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)
            self.num_frames = using_frame_num

        self.backbone.eval()
        self.sem_seg_head.eval()
        with torch.no_grad():
            features = self.backbone(images.tensor)
            image_outputs = self.sem_seg_head(features)
            frame_embeds= image_outputs['pred_embds'].clone().detach()  # (b, c, t, q)
            frame_reid_embeds = image_outputs['pred_reid_embed'].clone().detach()  # (b, c, t, q)
            mask_features = image_outputs['mask_features'].clone().detach().unsqueeze(0)
            pred_logits, pred_masks = image_outputs["pred_logits"].flatten(0, 1), image_outputs["pred_masks"].transpose(
                1, 2).flatten(0, 1)
            del image_outputs['mask_features']
            torch.cuda.empty_cache()
            image_outputs = {"pred_logits": pred_logits, "pred_masks": pred_masks}
        B, _, T, Q = frame_embeds.shape
        video_targets = self.prepare_targets(batched_inputs, images, select_fi_set)
        video_targets = self.split_video_targets(video_targets, clip_len=1)

        frame_targets = []
        for b in range(B):
            frame_targets.extend([item[b] for item in video_targets])
        frame_indices, aux_frame_indices, valid_masks = self.frame_matcher(image_outputs, frame_targets, 0.01)
        new_frame_indices, new_aux_frame_indices, new_valid_masks, new_pred_logits, new_pred_masks = [], [], [], [], []
        for i in range(T):
            new_frame_indices.append([frame_indices[b * T + i] for b in range(B)])
            new_aux_frame_indices.append([aux_frame_indices[b * T + i] for b in range(B)])
            new_valid_masks.append([valid_masks[b * T + i] for b in range(B)])
            # new_valid_masks.append([torch.ones_like(valid_masks[b*T+i]).to(torch.bool) for b in range(B)])
            new_pred_logits.append([pred_logits[b * T + i] for b in range(B)])
            new_pred_masks.append([pred_masks[b * T + i] for b in range(B)])
        frame_indices_info = {"indices": new_frame_indices, "aux_indices": new_aux_frame_indices,
                              "valid": new_valid_masks,
                              "pred_logits": new_pred_logits, "pred_masks": new_pred_masks, }

        stage = 1
        if self.iter >= 5000:
            stage = 2
        elif self.iter >= 10000:
            stage = 3
        self.iter += 1

        outputs = self.tracker(frame_embeds, frame_reid_embeds, mask_features, video_targets, frame_indices_info, self.new_ins_matcher, stage=stage)

        losses = self.vihub_criterion(outputs, video_targets)

        for k in list(losses.keys()):
            if k in self.vihub_criterion.weight_dict:
                losses[k] *= self.vihub_criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)
                print(f"loss {k} has not founded in the vihub_criterion.weight_dict, so removed!")
                exit(0)
        return losses

    def inference(self, batched_inputs):
        # for running demo on very long videos
        if 'keep' in batched_inputs[0].keys():
            self.keep = batched_inputs[0]['keep']
        else:
            self.keep = False

        if 'long_video_start_fidx' in batched_inputs[0].keys():
            long_video_start_fidx = batched_inputs[0]['long_video_start_fidx']
        else:
            long_video_start_fidx = -1

        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        outputs = self.run_window_inference(images.tensor, long_video_start_fidx=long_video_start_fidx)

        if len(outputs["pred_logits"]) == 0:
            video_output = {
                "image_size": (images.image_sizes[0], images.image_sizes[1]),
                "pred_scores": [],
                "pred_labels": [],
                "pred_masks": [],
                "pred_ids": [],
                "task": self.task,
            }
            return video_output
        mask_cls_results = outputs["pred_logits"]  # b, n, k+1
        mask_pred_results = outputs["pred_masks"]  # b, n, t, h, w
        pred_ids = outputs["pred_ids"]  # b, n
        # pred_ids = [torch.arange(0, outputs['pred_masks'].size(1))]

        mask_cls_result = mask_cls_results[0]
        mask_pred_result = mask_pred_results[0]

        pred_id = pred_ids[0]
        first_resize_size = (images.tensor.shape[-2], images.tensor.shape[-1])

        input_per_image = batched_inputs[0]
        image_size = images.image_sizes[0] # image size without padding after data augmentation

        height = input_per_image.get('height', image_size[0]) # raw image size before data augmentation
        width = input_per_image.get('width', image_size[1])

        return retry_if_cuda_oom(self.inference_video_task)(
            mask_cls_result, mask_pred_result, image_size, height, width, first_resize_size, pred_id
        )

    def prepare_targets(self, targets, images, select_fi_set):  # TODO, datamapper match with the function
        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)
            fi2idx = {fi: idx for idx, fi in enumerate(select_fi_set)}

            gt_classes_per_video = targets_per_video["instances"][select_fi_set[0]].gt_classes.to(self.device)
            gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                if f_i not in select_fi_set:
                    continue
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                _update_cls = gt_classes_per_video == -1
                gt_classes_per_video[_update_cls] = targets_per_frame.gt_classes[_update_cls]
                gt_ids_per_video.append(targets_per_frame.gt_ids)
                if isinstance(targets_per_frame.gt_masks, BitMasks):
                    gt_masks_per_video[:, fi2idx[f_i], :h, :w] = targets_per_frame.gt_masks.tensor
                else:
                    gt_masks_per_video[:, fi2idx[f_i], :h, :w] = targets_per_frame.gt_masks

            gt_ids_per_video = torch.stack(gt_ids_per_video, dim=1)  # ntgt, T
            gt_ids_per_video[gt_masks_per_video.sum(dim=(2, 3)) == 0] = -1
            valid_bool_frame = (gt_ids_per_video != -1)
            valid_bool_clip = valid_bool_frame.any(dim=-1)

            gt_classes_per_video = gt_classes_per_video[valid_bool_clip].long()  # N,
            gt_ids_per_video = gt_ids_per_video[valid_bool_clip].long()  # N, num_frames
            gt_masks_per_video = gt_masks_per_video[valid_bool_clip].float()  # N, num_frames, H, W
            valid_bool_frame = valid_bool_frame[valid_bool_clip]

            if len(gt_ids_per_video) > 0:
                min_id = max(gt_ids_per_video[valid_bool_frame].min(), 0)
                gt_ids_per_video[valid_bool_frame] -= min_id

            gt_instances.append(
                {
                    "labels": gt_classes_per_video, "ids": gt_ids_per_video, "masks": gt_masks_per_video,
                    "video_len": targets_per_video["video_len"], "frame_idx": targets_per_video["frame_idx"],
                }
            )
        return gt_instances

    def split_video_targets(self, clip_targets, clip_len=1):
        clip_target_splits = dict()
        for targets_per_video in clip_targets:
            labels = targets_per_video["labels"] # Ni (number of instances)

            ids = targets_per_video["ids"] # Ni, T
            masks = targets_per_video["masks"] # Ni, T, H, W
            frame_idx = targets_per_video["frame_idx"] # T

            masks_splits = masks.split(clip_len, dim=1)
            ids_splits = ids.split(clip_len, dim=1)

            prev_valid = torch.zeros_like(labels).bool()
            last_valid = torch.zeros_like(labels).bool()
            for clip_idx, (_masks, _ids) in enumerate(zip(masks_splits, ids_splits)):
                valid_inst = _masks.sum(dim=(1, 2, 3)) > 0.
                new_inst = (prev_valid == False) & (valid_inst == True)
                disappear_inst_ref2last = (last_valid == True) & (valid_inst == False)

                if not clip_idx in clip_target_splits:
                    clip_target_splits[clip_idx] = []

                clip_target_splits[clip_idx].append(
                    {
                        "labels": labels, "ids": _ids.squeeze(1), "masks": _masks.squeeze(1),
                        "video_len": targets_per_video["video_len"],
                        "frame_idx": frame_idx[clip_idx * clip_len:(clip_idx + 1) * clip_len],
                        "valid_inst": valid_inst,
                        "new_inst": new_inst,
                        "disappear_inst": disappear_inst_ref2last,
                    }
                )

                prev_valid = prev_valid | valid_inst
                last_valid = valid_inst

        return list(clip_target_splits.values())

    def run_window_inference(self, images_tensor, window_size=30, long_video_start_fidx=-1):
        video_start_idx = long_video_start_fidx if long_video_start_fidx >= 0 else 0

        num_frames = len(images_tensor)
        to_store = "cpu"
        iters = len(images_tensor) // window_size
        if len(images_tensor) % window_size != 0:
            iters += 1
        for i in range(iters):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size
            # segmenter inference
            features = self.backbone(images_tensor[start_idx:end_idx])
            out = self.sem_seg_head(features)
            # remove unnecessary variables to save GPU memory
            del features['res2'], features['res3'], features['res4'], features['res5']
            for j in range(len(out['aux_outputs'])):
                del out['aux_outputs'][j]['pred_masks'], out['aux_outputs'][j]['pred_logits']
            # referring tracker inference
            frame_embds = out['pred_embds']  # (b, c, t, q)
            frame_reid_embeds = out["pred_reid_embed"]
            mask_features = out['mask_features'].unsqueeze(0) # as B == 1
            pred_logits, pred_masks = out["pred_logits"].flatten(0, 1), out["pred_masks"].transpose(1, 2).flatten(0, 1)

            B, _, T, _ = frame_embds.shape
            H, W = mask_features.shape[-2:]

            pred_scores = torch.max(pred_logits.softmax(dim=-1)[:, :, :-1], dim=-1)[0]
            # valid_masks = pred_scores > self.inference_select_thr
            valid_masks = pred_scores > 0.01
            new_pred_logits, new_pred_masks, new_valid_masks,  = [], [], []
            for t in range(T):
                # assert B == 1
                pred_scores_t = pred_scores[t] # q,
                valid_masks_t = valid_masks[t] # q,
                pred_masks_t = pred_masks[t]   # q, h, w
                if valid_masks_t.sum() == 0:
                    max_idx = torch.argmax(pred_scores_t)
                    valid_masks_t[max_idx] = True

                # valid_pred_scores_t = pred_scores_t[valid_masks_t] # q",
                # valid_pred_masks_t  = pred_masks_t[valid_masks_t]  # q", h, w
                #
                # _, sorted_indices = torch.sort(valid_pred_scores_t, dim=0, descending=True)
                # sorted_valid_scores_t = valid_pred_scores_t[sorted_indices]
                # sorted_valid_masks_t  = valid_pred_masks_t[sorted_indices]
                #
                # valid_nms_indices = mask_nms(sorted_valid_masks_t[:, None, ...], sorted_valid_scores_t, nms_thr=self.mask_nms_thr)
                # sort_back_valid_nms_mask = torch.zeros_like(valid_pred_scores_t, dtype=torch.bool, device=self.device) # q",
                # sort_back_valid_nms_mask[sorted_indices] = torch.as_tensor(valid_nms_indices).to(torch.bool).to(self.device)
                # new_valid_masks_t = torch.zeros_like(valid_masks_t, dtype=torch.bool, device=self.device) # q,
                # new_valid_masks_t[valid_masks_t] = sort_back_valid_nms_mask

                new_pred_logits.append([pred_logits[b * T + t] for b in range(B)])
                new_pred_masks.append([pred_masks[b * T + t] for b in range(B)])
                # new_valid_masks.append([new_valid_masks_t])
                new_valid_masks.append([valid_masks_t])
            frame_info = {"pred_logits": new_pred_logits, "pred_masks": new_pred_masks, "valid": new_valid_masks}

            if i != 0 or self.keep:
                self.tracker.inference(frame_embds, frame_reid_embeds, mask_features, frame_info, video_start_idx+start_idx, resume=True, to_store=to_store)
            else:
                self.tracker.inference(frame_embds, frame_reid_embeds, mask_features, frame_info, video_start_idx+start_idx, to_store=to_store)

        logits_list = []
        masks_list = []
        seq_id_list = []
        dead_seq_id_list = []
        for seq_id, ins_seq in self.tracker.video_ins_hub.items():
            if len(ins_seq.pred_masks) < self.noise_frame_num:
                # if ins_seq.sT + len(ins_seq.pred_masks) == num_frames, which means this object appeared at the end of
                # this clip and cloud be exists in the next clip.
                if ins_seq.sT + len(ins_seq.pred_masks) < video_start_idx + num_frames:
                    continue
            full_masks = torch.zeros(num_frames, H, W).to(torch.float32).to(to_store)
            seq_logits = []
            seq_start_t = ins_seq.sT
            for j in range(len(ins_seq.pred_masks)):
                if seq_start_t + j < video_start_idx:
                    continue
                re_j = seq_start_t + j - video_start_idx
                full_masks[re_j, :, :] = ins_seq.pred_masks[j]
                seq_logits.append(ins_seq.pred_logits[j])
            if len(seq_logits) == 0:
                continue
            seq_logits = torch.stack(seq_logits, dim=0).mean(0)  # n, c -> c
            logits_list.append(seq_logits)
            masks_list.append(full_masks)
            assert ins_seq.gt_id == seq_id
            seq_id_list.append(seq_id)
            if ins_seq.dead:
                dead_seq_id_list.append(seq_id)

        for seq_id in dead_seq_id_list:
            self.tracker.video_ins_hub.pop(seq_id)

        if len(logits_list) > 0:
            pred_cls = torch.stack(logits_list, dim=0)[None, ...]  # b, n, c
            pred_masks = torch.stack(masks_list, dim=0)[None, ...]  # b, n, t, h, w
            pred_ids = torch.as_tensor(seq_id_list).to(torch.int64)[None, :]  # b, n
        else:
            pred_cls = []
            pred_masks = []
            pred_ids = []

        outputs = {
            "pred_logits": pred_cls,
            "pred_masks": pred_masks,
            "pred_ids": pred_ids
        }

        return outputs

    def inference_video_vis(
        self, pred_cls, pred_masks, img_size, output_height, output_width,
        first_resize_size, pred_id, aux_pred_cls=None,
    ):
        if len(pred_cls) > 0:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            if aux_pred_cls is not None:
                aux_pred_cls = F.softmax(aux_pred_cls, dim=-1)[:, :-1]
                scores = torch.maximum(scores, aux_pred_cls.to(scores))
            labels = torch.arange(
                self.sem_seg_head.num_classes, device=self.device
            ).unsqueeze(0).repeat(scores.shape[0], 1).flatten(0, 1) #TODO, pay attention to the use of scores.shape[0] != self.num_queries
            # keep top-K predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.max_num, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.sem_seg_head.num_classes
            pred_masks = pred_masks[topk_indices]
            pred_ids = pred_id[topk_indices]

            # interpolation to original image size
            pred_masks = F.interpolate(
                pred_masks, size=first_resize_size, mode="bilinear", align_corners=False
            )
            pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
            pred_masks = F.interpolate(
                pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
            )
            masks = pred_masks > 0.
            del pred_masks

            out_scores = scores_per_image.tolist()
            out_labels = labels_per_image.tolist()
            out_ids = pred_ids.tolist()
            out_masks = [m for m in masks.cpu()]
        else:
            out_scores = []
            out_labels = []
            out_masks = []
            out_ids = []

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
            "pred_ids": out_ids,
            "task": "vis",
        }

        return video_output

    def inference_video_vps(
        self, pred_cls, pred_masks, img_size, output_height, output_width,
        first_resize_size, pred_id, aux_pred_cls=None,
    ):
        pred_cls = F.softmax(pred_cls, dim=-1)
        if aux_pred_cls is not None:
            aux_pred_cls = F.softmax(aux_pred_cls, dim=-1)[:, :-1]
            pred_cls[:, :-1] = torch.maximum(pred_cls[:, :-1], aux_pred_cls.to(pred_cls))
        mask_pred = pred_masks
        scores, labels = pred_cls.max(-1)

        # filter out the background prediction
        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_ids = pred_id[keep]
        cur_masks = mask_pred[keep]

        # interpolation to original image size
        cur_masks = F.interpolate(
            cur_masks, size=first_resize_size, mode="bilinear", align_corners=False
        )
        cur_masks = cur_masks[:, :, :img_size[0], :img_size[1]].sigmoid()
        cur_masks = F.interpolate(
            cur_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
        )
        cur_prob_masks = cur_scores.view(-1, 1, 1, 1).to(cur_masks.device) * cur_masks

        # initial panoptic_seg and segments infos
        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((cur_masks.size(1), h, w), dtype=torch.int32, device=cur_masks.device)
        segments_infos = []
        out_ids = []
        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask
            return {
                "image_size": (output_height, output_width),
                "pred_masks": panoptic_seg.cpu(),
                "segments_infos": segments_infos,
                "pred_ids": out_ids,
                "task": "vps",
            }
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)  # (t, h, w)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class < len(self.metadata.thing_dataset_id_to_contiguous_id)
                # filter out the unstable segmentation results
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)
                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue
                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1
                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_infos.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )
                    out_ids.append(cur_ids[k])

            return {
                "image_size": (output_height, output_width),
                "pred_masks": panoptic_seg.cpu(),
                "segments_infos": segments_infos,
                "pred_ids": out_ids,
                "task": "vps",
            }

    def inference_video_vss(
        self, pred_cls, pred_masks, img_size, output_height, output_width,
        first_resize_size, pred_id, aux_pred_cls=None,
    ):
        mask_cls = F.softmax(pred_cls, dim=-1)[..., :-1]
        if aux_pred_cls is not None:
            aux_pred_cls = F.softmax(aux_pred_cls, dim=-1)[:, :-1]
            mask_cls[..., :-1] = torch.maximum(mask_cls[..., :-1], aux_pred_cls.to(mask_cls))
        mask_pred = pred_masks
        # interpolation to original image size
        cur_masks = F.interpolate(
            mask_pred, size=first_resize_size, mode="bilinear", align_corners=False
        )
        cur_masks = cur_masks[:, :, :img_size[0], :img_size[1]].sigmoid()
        cur_masks = F.interpolate(
            cur_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
        )

        semseg = torch.einsum("qc,qthw->cthw", mask_cls, cur_masks)
        sem_score, sem_mask = semseg.max(0)
        sem_mask = sem_mask
        return {
                "image_size": (output_height, output_width),
                "pred_masks": sem_mask.cpu(),
                "task": "vss",
            }