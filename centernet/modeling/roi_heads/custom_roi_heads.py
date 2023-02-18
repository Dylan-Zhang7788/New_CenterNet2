# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import json
import math
import torch
from torch import nn
from torch.autograd.function import Function
from typing import Dict, List, Optional, Tuple, Union

from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.cascade_rcnn import CascadeROIHeads
from detectron2.modeling.roi_heads.box_head import build_box_head
from .custom_fast_rcnn import CustomFastRCNNOutputLayers


@ROI_HEADS_REGISTRY.register()
class CustomROIHeads(StandardROIHeads):
    @classmethod
    def _init_box_head(self, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        del ret['box_predictor']
        ret['box_predictor'] = CustomFastRCNNOutputLayers(
            cfg, ret['box_head'].output_shape)
        self.debug = cfg.DEBUG
        if self.debug:
            self.debug_show_name = cfg.DEBUG_SHOW_NAME
            self.save_debug = cfg.SAVE_DEBUG
            self.vis_thresh = cfg.VIS_THRESH
            self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(
                torch.device(cfg.MODEL.DEVICE)).view(3, 1, 1)
            self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(
                torch.device(cfg.MODEL.DEVICE)).view(3, 1, 1)
        return ret

    def forward(self, images, features, proposals, targets=None):
        """
        enable debug
        """
        if not self.debug:
            del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            if self.debug:
                from ..debug import debug_second_stage
                denormalizer = lambda x: x * self.pixel_std + self.pixel_mean
                debug_second_stage(
                    [denormalizer(images[0].clone())],
                    pred_instances, proposals=proposals,
                    debug_show_name=self.debug_show_name)
            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
# 继承关系：CustomCascadeROIHeads → CascadeROIHeads → StandardROIHeads → ROIHeads
class CustomCascadeROIHeads(CascadeROIHeads):
    @classmethod
    def _init_box_head(self, cfg, input_shape):
        # 示例里定义为True
        self.mult_proposal_score = cfg.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE
        ret = super()._init_box_head(cfg, input_shape)
        del ret['box_predictors']
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        box_predictors = []
        for box_head, bbox_reg_weights in zip(ret['box_heads'], cascade_bbox_reg_weights):
            box_predictors.append(
                CustomFastRCNNOutputLayers(
                    cfg, box_head.output_shape,
                    box2box_transform=Box2BoxTransform(weights=bbox_reg_weights)
                ))
        ret['box_predictors'] = box_predictors
        self.debug = cfg.DEBUG
        if self.debug:
            self.debug_show_name = cfg.DEBUG_SHOW_NAME
            self.save_debug = cfg.SAVE_DEBUG
            self.vis_thresh = cfg.VIS_THRESH
            self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(
                torch.device(cfg.MODEL.DEVICE)).view(3, 1, 1)
            self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(
                torch.device(cfg.MODEL.DEVICE)).view(3, 1, 1)
        return ret


    def _forward_box(self, features, proposals, targets=None):
        """
        Add mult proposal scores at testing
        """
        # proposals是8维的list 表示8张图
        # 每个元素包含proposal_boxes, objectness_logits, gt_classes, gt_boxes
        if (not self.training) and self.mult_proposal_score:
            if len(proposals) > 0 and proposals[0].has('scores'):
                proposal_scores = [
                    p.get('scores') for p in proposals]
            else:
                proposal_scores = [
                    p.get('objectness_logits') for p in proposals]
        # box_in_features就是 p3 p4 p5 p6 p7
        features = [features[f] for f in self.box_in_features]
        head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]
        for k in range(self.num_cascade_stages):
            if k > 0:
                # 根据预测的选框 再次产生proposal，同样是是8维的list 表示8张图
                # 但是目前为止每个元素只包括 proposal_boxes
                proposals = self._create_proposals_from_boxes(prev_pred_boxes, image_sizes)
                if self.training:
                    # 经过了_match_and_label_boxes的proposal
                    # 每个元素包含 proposal_boxes, gt_classes, gt_boxes 但不包含objectness_logits
                    proposals = self._match_and_label_boxes(proposals, k, targets)
            # 经过runstage的predictions是一个list 两个量分别是类别预测结果和选框预测结果
            # 类别预测结果：[2048,类别数+1] 选框预测结果[2048,4]
            predictions = self._run_stage(features, proposals, k)
            # prev_pred_boxes8维的list 表示8张图 
            # 每张图中存放的是 [256,4] 256表示每张图产生256个框
            prev_pred_boxes = self.box_predictor[k].predict_boxes(predictions, proposals)
            head_outputs.append((self.box_predictor[k], predictions, proposals))

        if self.training:
            losses = {}
            storage = get_event_storage()
            for stage, (predictor, predictions, proposals) in enumerate(head_outputs):
                with storage.name_scope("stage{}".format(stage)):
                    stage_losses = predictor.losses(predictions, proposals)
                losses.update({k + "_stage{}".format(stage): v for k, v in stage_losses.items()})
            return losses
        else:
            # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
            # 每个类别都有一个对应的score
            scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in head_outputs]
            # 同样还是 每个类别都有一个score
            scores = [
                sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                for scores_per_image in zip(*scores_per_stage)
            ]
            
            if self.mult_proposal_score:
                scores = [(s * ps[:, None]) ** 0.5 \
                    for s, ps in zip(scores, proposal_scores)]

            predictor, predictions, proposals = head_outputs[-1]
            boxes = predictor.predict_boxes(predictions, proposals)
            pred_instances, _ = fast_rcnn_inference(
                boxes,
                scores,
                image_sizes,
                predictor.test_score_thresh,
                predictor.test_nms_thresh,
                predictor.test_topk_per_image,
            )
            
            return pred_instances

    def forward(self, images, features, proposals, targets=None):
        '''
        enable debug
        '''
        if not self.debug:
            del images
        if self.training:
            # label_and_sample_proposals是定义在standard_roihead里的函数
            # 把proposal和target关联在一起
            # 采样后的结果：proposal由2000下降到了256(这个值是人为设定的)
            # 并且每个proposal都有了一个与他对应的gt
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            losses = self._forward_box(features, proposals, targets)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            # import pdb; pdb.set_trace()
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            if self.debug:
                from ..debug import debug_second_stage
                denormalizer = lambda x: x * self.pixel_std + self.pixel_mean
                debug_second_stage(
                    [denormalizer(x.clone()) for x in images],
                    pred_instances, proposals=proposals,
                    save_debug=self.save_debug,
                    debug_show_name=self.debug_show_name,
                    vis_thresh=self.vis_thresh)
            return pred_instances, {}