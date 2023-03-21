import math
import json
import numpy as np
import torch
from torch import nn
from functools import partial
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling import build_backbone, build_proposal_generator
from detectron2.modeling import detector_postprocess
from detectron2.utils.events import get_event_storage
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList
from .centernet_detector import CenterNetDetector
from ..dense_heads.GFL import MY_GFLModule
from mmdet.core.bbox.coder.delta_xywh_bbox_coder import bbox2delta,delta2bbox
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def MY_bbox2delta(proposals, gt, means=(0., 0., 0., 0.), stds=(1., 1., 1., 1.)):

    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0]
    gh = gt[..., 3] - gt[..., 1]

    dx = (gx - px)
    dy = (gy - py)
    dw = (gw / pw)
    dh = (gh / ph)
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas

def MY_delta2bbox(rois,
               deltas,
               means=(0., 0., 0., 0.),
               stds=(1., 1., 1., 1.),
               max_shape=None,
               wh_ratio_clip=16 / 1000,
               clip_border=True,
               add_ctr_clamp=False,
               ctr_clamp=32):
    
    num_bboxes, num_classes = deltas.size(0), deltas.size(1) // 4
    if num_bboxes == 0:
        return deltas

    deltas = deltas.reshape(-1, 4)

    means = deltas.new_tensor(means).view(1, -1)
    stds = deltas.new_tensor(stds).view(1, -1)
    denorm_deltas = deltas * stds + means

    dxy = denorm_deltas[:, :2]
    dwh = denorm_deltas[:, 2:]

    # Compute width/height of each roi
    rois_ = rois.repeat(1, num_classes).reshape(-1, 4)
    pxy = ((rois_[:, :2] + rois_[:, 2:]) * 0.5)
    pwh = (rois_[:, 2:] - rois_[:, :2])

    max_ratio = np.abs(np.log(wh_ratio_clip))
    if add_ctr_clamp:
        dxy_wh = torch.clamp(dxy_wh, max=ctr_clamp, min=-ctr_clamp)
        dwh = torch.clamp(dwh, max=max_ratio)
    else:
        dwh = dwh.clamp(min=-max_ratio, max=max_ratio)

    gxy = pxy + dxy
    gwh = pwh * dwh
    x1y1 = gxy - (gwh * 0.5)
    x2y2 = gxy + (gwh * 0.5)
    bboxes = torch.cat([x1y1, x2y2], dim=-1)
    if clip_border and max_shape is not None:
        bboxes[..., 0::2].clamp_(min=0, max=max_shape[1])
        bboxes[..., 1::2].clamp_(min=0, max=max_shape[0])
    bboxes = bboxes.reshape(num_bboxes, -1)
    return bboxes

@META_ARCH_REGISTRY.register()
class CenterNetDetector_value(MY_GFLModule):
    def __init__(self, cfg):
        super().__init__(cfg,in_channels=160)
        self.mean, self.std = cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD
        self.vis_period=cfg.VIS_PERIOD
        self.input_format='BGR'
        self.draw_period=2000
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))
        
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()) # TODO: change to a more precise name
        self.strides=[(8, 8), (16, 16), (32, 32)]
        self.x=nn.Conv2d(in_channels=3,out_channels=1,kernel_size=3,padding=0)
        torch.nn.init.normal_(self.x.weight, std=0.01)
        torch.nn.init.constant_(self.x.bias, 0)
    
    def prepare_gt_box_and_class(self,image_list,targets):
        gt_bboxes = []
        gt_labels = []
        img_metas = []
        for i in range(len(image_list)):
            gt_bboxes_per_img = targets[i].gt_boxes.tensor
            gt_labels_per_img = targets[i].gt_classes
            gt_bboxes.append(gt_bboxes_per_img)
            gt_labels.append(gt_labels_per_img)
        return gt_bboxes,gt_labels
    
    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list
    
    def loss(self,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes=[]
        featmap_sizes.append(tuple([int(x / 8) for x in img_metas[0]['pad_shape']]))
        featmap_sizes.append(tuple([int(x / 16) for x in img_metas[0]['pad_shape']]))
        featmap_sizes.append(tuple([int(x / 32) for x in img_metas[0]['pad_shape']]))
    

        device = 'cuda'
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        # label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        label_channels = 1

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        for i in range(len(anchor_list)):
            self.loss_single(
                anchor_list[i],
                labels_list[i],
                label_weights_list[i],
                bbox_targets_list[i],
                self.strides[i])

        return 0

    def loss_single(self, anchors, labels, label_weights,
                    bbox_targets, stride):
        """Compute loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            stride (tuple): Stride in this scale level.
            num_total_samples (int): Number of positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)

        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_anchors = anchors[pos_inds]

            reg = MY_bbox2delta(pos_anchors , pos_bbox_targets)
            rectify=MY_delta2bbox(pos_anchors,reg)
            reg_xy=reg[:,:2].int()
            reg_wh=reg[:,2:].log2().int()
            list_dx=list_dy=list_dw=list_dh=[0]
            list_dx+=reg_xy[:,0].tolist()
            list_dy+=reg_xy[:,1].tolist()
            list_dw+=reg_wh[:,0].tolist()
            list_dh+=reg_wh[:,1].tolist()
            storage = get_event_storage()
            if storage.iter % self.draw_period == 0:
                fig,axes=plt.subplots(2,2)
                sns.histplot(list_dx, kde=True,ax=axes[0,0])
                # 显示图形
                sns.histplot(list_dy, kde=True,ax=axes[0,1])
                # 显示图形
                sns.histplot(list_dw, kde=True,ax=axes[1,0])
                # 显示图形
                sns.histplot(list_dh, kde=True,ax=axes[1,1])
                # 显示图形
                plt.savefig('out.jpg')

        return 0
    
    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        # img=images
        # img=np.asarray(img.tensor.cpu().permute(0,2,3,1)[0])
        # import cv2
        # cv2.imwrite("output/1.jpg",img)
        img_metas=[]
        meta={}
        for i , img_size in enumerate(images.image_sizes):
            meta['img_shape']=img_size
            meta['pad_shape']=tuple(images.tensor.shape[-2:])
            img_metas.append(meta)

        gt_bboxes,gt_labels=self.prepare_gt_box_and_class(images,gt_instances)
        self.loss(gt_bboxes,gt_labels,img_metas)


        proposal_losses = {'loss': self.x(images.tensor).view(-1).sum()*0}

        return proposal_losses


    @property
    def device(self):
        return self.pixel_mean.device


    @torch.no_grad()
    def inference(self, batched_inputs, do_postprocess=True):
        images = self.preprocess_image(batched_inputs)
        inp = images.tensor
        features = self.backbone(inp)
        proposals, _ = self.proposal_generator(images, features, None)

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes):
            if do_postprocess:
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            else:
                r = results_per_image
                processed_results.append(r)
        return processed_results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images
