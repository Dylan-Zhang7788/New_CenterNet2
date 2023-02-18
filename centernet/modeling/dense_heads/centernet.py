
import math
import json
import copy
from typing import List, Dict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.layers import ShapeSpec, cat
from detectron2.structures import Instances, Boxes
from detectron2.modeling import detector_postprocess
from detectron2.utils.comm import get_world_size
from detectron2.config import configurable

from ..layers.heatmap_focal_loss import heatmap_focal_loss_jit
from ..layers.heatmap_focal_loss import binary_heatmap_focal_loss_jit
from ..layers.iou_loss import IOULoss
from ..layers.ml_nms import ml_nms
from ..debug import debug_train, debug_test
from .utils import reduce_sum, _transpose
from .centernet_head import CenterNetHead

__all__ = ["CenterNet"]

INF = 100000000

@PROPOSAL_GENERATOR_REGISTRY.register()
class CenterNet(nn.Module):
    @configurable
    def __init__(self, 
        # input_shape: Dict[str, ShapeSpec],
        in_channels=256,
        *,
        num_classes=80,
        in_features=("p3", "p4", "p5", "p6", "p7"),
        strides=(8, 16, 32, 64, 128),
        score_thresh=0.05,
        hm_min_overlap=0.8,
        loc_loss_type='giou',
        min_radius=4,
        hm_focal_alpha=0.25,
        hm_focal_beta=4,
        loss_gamma=2.0,
        reg_weight=2.0,
        not_norm_reg=True,
        with_agn_hm=False,
        only_proposal=False,
        as_proposal=False,
        not_nms=False,
        pos_weight=1.,
        neg_weight=1.,
        sigmoid_clamp=1e-4,
        ignore_high_fp=-1.,
        center_nms=False,
        sizes_of_interest=[[0,80],[64,160],[128,320],[256,640],[512,10000000]],
        more_pos=False,
        more_pos_thresh=0.2,
        more_pos_topk=9,
        pre_nms_topk_train=1000,
        pre_nms_topk_test=1000,
        post_nms_topk_train=100,
        post_nms_topk_test=100,
        nms_thresh_train=0.6,
        nms_thresh_test=0.6,
        no_reduce=False,
        not_clamp_box=False,
        debug=False,
        vis_thresh=0.5,
        pixel_mean=[103.530,116.280,123.675],
        pixel_std=[1.0,1.0,1.0],
        device='cuda',
        centernet_head=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_features = in_features
        self.strides = strides
        self.score_thresh = score_thresh
        self.min_radius = min_radius
        self.hm_focal_alpha = hm_focal_alpha
        self.hm_focal_beta = hm_focal_beta
        self.loss_gamma = loss_gamma
        self.reg_weight = reg_weight
        self.not_norm_reg = not_norm_reg
        self.with_agn_hm = with_agn_hm
        self.only_proposal = only_proposal
        self.as_proposal = as_proposal
        self.not_nms = not_nms
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.sigmoid_clamp = sigmoid_clamp
        self.ignore_high_fp = ignore_high_fp
        self.center_nms = center_nms
        self.sizes_of_interest = sizes_of_interest
        self.more_pos = more_pos
        self.more_pos_thresh = more_pos_thresh
        self.more_pos_topk = more_pos_topk
        self.pre_nms_topk_train = pre_nms_topk_train
        self.pre_nms_topk_test = pre_nms_topk_test
        self.post_nms_topk_train = post_nms_topk_train
        self.post_nms_topk_test = post_nms_topk_test
        self.nms_thresh_train = nms_thresh_train
        self.nms_thresh_test = nms_thresh_test
        self.no_reduce = no_reduce
        self.not_clamp_box = not_clamp_box
        
        self.debug = debug
        self.vis_thresh = vis_thresh
        if self.center_nms:
            self.not_nms = True
        self.iou_loss = IOULoss(loc_loss_type)
        assert (not self.only_proposal) or self.with_agn_hm
        # delta for rendering heatmap
        self.delta = (1 - hm_min_overlap) / (1 + hm_min_overlap)
        if centernet_head is None:
            self.centernet_head = CenterNetHead(
                in_channels=in_channels,
                num_levels=len(in_features),
                with_agn_hm=with_agn_hm,
                only_proposal=only_proposal)
        else:
            self.centernet_head = centernet_head
        if self.debug:
            pixel_mean = torch.Tensor(pixel_mean).to(
                torch.device(device)).view(3, 1, 1)
            pixel_std = torch.Tensor(pixel_std).to(
                torch.device(device)).view(3, 1, 1)
            self.denormalizer = lambda x: x * pixel_std + pixel_mean

    @classmethod
    def from_config(cls, cfg, input_shape):
        # 跟RCNN里的from_config基本上是一样的
        ret = {
            # 'input_shape': input_shape,
            'in_channels': input_shape[
                cfg.MODEL.CENTERNET.IN_FEATURES[0]].channels,
            'num_classes': cfg.MODEL.CENTERNET.NUM_CLASSES,
            'in_features': cfg.MODEL.CENTERNET.IN_FEATURES,
            'strides': cfg.MODEL.CENTERNET.FPN_STRIDES,
            'score_thresh': cfg.MODEL.CENTERNET.INFERENCE_TH,
            'loc_loss_type': cfg.MODEL.CENTERNET.LOC_LOSS_TYPE,
            'hm_min_overlap': cfg.MODEL.CENTERNET.HM_MIN_OVERLAP,
            'min_radius': cfg.MODEL.CENTERNET.MIN_RADIUS,
            'hm_focal_alpha': cfg.MODEL.CENTERNET.HM_FOCAL_ALPHA,
            'hm_focal_beta': cfg.MODEL.CENTERNET.HM_FOCAL_BETA,
            'loss_gamma': cfg.MODEL.CENTERNET.LOSS_GAMMA,
            'reg_weight': cfg.MODEL.CENTERNET.REG_WEIGHT,
            'not_norm_reg': cfg.MODEL.CENTERNET.NOT_NORM_REG,
            'with_agn_hm': cfg.MODEL.CENTERNET.WITH_AGN_HM,
            'only_proposal': cfg.MODEL.CENTERNET.ONLY_PROPOSAL,
            'as_proposal': cfg.MODEL.CENTERNET.AS_PROPOSAL,
            'not_nms': cfg.MODEL.CENTERNET.NOT_NMS,
            'pos_weight': cfg.MODEL.CENTERNET.POS_WEIGHT,
            'neg_weight': cfg.MODEL.CENTERNET.NEG_WEIGHT,
            'sigmoid_clamp': cfg.MODEL.CENTERNET.SIGMOID_CLAMP,
            'ignore_high_fp': cfg.MODEL.CENTERNET.IGNORE_HIGH_FP,
            'center_nms': cfg.MODEL.CENTERNET.CENTER_NMS,
            'sizes_of_interest': cfg.MODEL.CENTERNET.SOI,
            'more_pos': cfg.MODEL.CENTERNET.MORE_POS,
            'more_pos_thresh': cfg.MODEL.CENTERNET.MORE_POS_THRESH,
            'more_pos_topk': cfg.MODEL.CENTERNET.MORE_POS_TOPK,
            'pre_nms_topk_train': cfg.MODEL.CENTERNET.PRE_NMS_TOPK_TRAIN,
            'pre_nms_topk_test': cfg.MODEL.CENTERNET.PRE_NMS_TOPK_TEST,
            'post_nms_topk_train': cfg.MODEL.CENTERNET.POST_NMS_TOPK_TRAIN,
            'post_nms_topk_test': cfg.MODEL.CENTERNET.POST_NMS_TOPK_TEST,
            'nms_thresh_train': cfg.MODEL.CENTERNET.NMS_TH_TRAIN,
            'nms_thresh_test': cfg.MODEL.CENTERNET.NMS_TH_TEST,
            'no_reduce': cfg.MODEL.CENTERNET.NO_REDUCE,
            'not_clamp_box': cfg.INPUT.NOT_CLAMP_BOX,
            'debug': cfg.DEBUG,
            'vis_thresh': cfg.VIS_THRESH,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'device': cfg.MODEL.DEVICE,
            'centernet_head': CenterNetHead(
                cfg, [input_shape[f] for f in cfg.MODEL.CENTERNET.IN_FEATURES]),
        }
        return ret


    # 这里的images，就是图像，B*c*h*w
    # features_dict是由backbone输出的特征图，是字典{名字：特征图}，gt就是gt
    def forward(self, images, features_dict, gt_instances):
        # 获取到的features 处理一下，self.in_features有默认值 p3-p7
        features = [features_dict[f] for f in self.in_features]
        # 输入centernet_head里，不是ROIhead，是centernet_head
        clss_per_level, reg_pred_per_level, agn_hm_pred_per_level = \
            self.centernet_head(features)
        # compute_grids 计算网格 输出的是 每个level上的一系列中心点
        # 并且输出的是中心点在原图中的绝对坐标 笔记里有记的
        # 相当于 把特征图的像素 映射回了 原图 
        # grids: list，level数个 [每个特征图的像素个数，2] 2表示x，y
        # 注意 x，y 和 h，w不一样 前者是点，是一个数，后者是长度，是一堆的x,y
        # 为什么grids的输出不涉及到尺寸 原因是坐标点的个数,就是特征图的h,w相乘,特征图里有多少点,这里就是多少点
        grids = self.compute_grids(features)

        # reg_pred_per_level: list，level数个 [batch数,channel,h，w]
        # 这样就得到了每个level的特征图尺寸
        # 得到的 shapes_per_level 是[level数,2]
        shapes_per_level = grids[0].new_tensor(
                    [(x.shape[2], x.shape[3]) for x in reg_pred_per_level])
        
        if not self.training:
            return self.inference(
                images, clss_per_level, reg_pred_per_level, 
                agn_hm_pred_per_level, grids)
        else:
            pos_inds, labels, reg_targets, flattened_hms = \
                self._get_ground_truth(
                    grids, shapes_per_level, gt_instances)
            # 输出结果：logits_pred: M x F, reg_pred: M x 4, agn_hm_pred: M
            logits_pred, reg_pred, agn_hm_pred = self._flatten_outputs(
                clss_per_level, reg_pred_per_level, agn_hm_pred_per_level)

            if self.more_pos:
                # add more pixels as positive if \
                #   1. they are within the center3x3 region of an object
                #   2. their regression losses are small (<self.more_pos_thresh)
                pos_inds, labels = self._add_more_pos(
                    reg_pred, gt_instances, shapes_per_level)
            
            losses = self.losses(
                pos_inds, labels, reg_targets, flattened_hms,
                logits_pred, reg_pred, agn_hm_pred)
            
            proposals = None
            if self.only_proposal:
                agn_hm_pred_per_level = [x.sigmoid() for x in agn_hm_pred_per_level]
                proposals = self.predict_instances(
                    grids, agn_hm_pred_per_level, reg_pred_per_level, 
                    images.image_sizes, [None for _ in agn_hm_pred_per_level])
            elif self.as_proposal: # category specific bbox as agnostic proposals
                clss_per_level = [x.sigmoid() for x in clss_per_level]
                proposals = self.predict_instances(
                    grids, clss_per_level, reg_pred_per_level, 
                    images.image_sizes, agn_hm_pred_per_level)
            if self.only_proposal or self.as_proposal:
                # 这个地方就是换了个名字，然后把pred_classes移除了
                for p in range(len(proposals)):
                    proposals[p].proposal_boxes = proposals[p].get('pred_boxes')
                    proposals[p].objectness_logits = proposals[p].get('scores')
                    proposals[p].remove('pred_boxes')
                    proposals[p].remove('scores')
                    proposals[p].remove('pred_classes')

            if self.debug:
                debug_train(
                    [self.denormalizer(x) for x in images], 
                    gt_instances, flattened_hms, reg_targets, 
                    labels, pos_inds, shapes_per_level, grids, self.strides)
            return proposals, losses


    def losses(
        self, pos_inds, labels, reg_targets, flattened_hms,
        logits_pred, reg_pred, agn_hm_pred):
        '''
        Inputs:
            pos_inds: N 
            这里的N不是说每个图有N个框,也不是说这个Batch里总共N个框,就是简单的有N个pos_inds
            因为相同的框可能被不同的level同时care,那这个框就会出现两次,label也是同理
            labels: N
            reg_targets: M x 4
            flattened_hms: M x C
            logits_pred: M x C
            reg_pred: M x 4
            agn_hm_pred: M x 1 or None
            N: number of positive locations in all images
            M: number of pixels from all FPN levels
            C: number of classes
        '''
        assert (torch.isfinite(reg_pred).all().item())
        num_pos_local = pos_inds.numel()
        num_gpus = get_world_size()
        # 分布式训练的一个东西
        if self.no_reduce:
            total_num_pos = num_pos_local * num_gpus
        else:
            total_num_pos = reduce_sum(
                pos_inds.new_tensor([num_pos_local])).item()
        num_pos_avg = max(total_num_pos / num_gpus, 1.0)

        losses = {}
        if not self.only_proposal:
            pos_loss, neg_loss = heatmap_focal_loss_jit(
                logits_pred.float(), flattened_hms.float(), pos_inds, labels,
                alpha=self.hm_focal_alpha, 
                beta=self.hm_focal_beta, 
                gamma=self.loss_gamma, 
                reduction='sum',
                sigmoid_clamp=self.sigmoid_clamp,
                ignore_high_fp=self.ignore_high_fp,
            )
            pos_loss = self.pos_weight * pos_loss / num_pos_avg
            neg_loss = self.neg_weight * neg_loss / num_pos_avg
            losses['loss_centernet_pos'] = pos_loss
            losses['loss_centernet_neg'] = neg_loss
        # torch.nonzero 输出的非零元素的索引，这里squeeze是减少他的维度
        # reg_inds的维数，是pos_inds维数的9倍
        reg_inds = torch.nonzero(reg_targets.max(dim=1)[0] >= 0).squeeze(1)
        # 再用索引把预测值给取出来
        reg_pred = reg_pred[reg_inds]
        # 把真值也给取出来
        reg_targets_pos = reg_targets[reg_inds]
        # 这一步是提取最大值，但是只有proposal的情况下，总共就一维，所以是不变的
        reg_weight_map = flattened_hms.max(dim=1)[0]
        # reg_weight_map的维数，同样也是pos_inds维数的9倍
        reg_weight_map = reg_weight_map[reg_inds]
        # 示例里的no_norm_reg是True 所以这里的reg_weight_map全是1
        reg_weight_map = reg_weight_map * 0 + 1 \
            if self.not_norm_reg else reg_weight_map
        if self.no_reduce:
            reg_norm = max(reg_weight_map.sum(), 1)
        else:
            reg_norm = max(reduce_sum(reg_weight_map.sum()).item() / num_gpus, 1)
        # reg_weight 默认值是2.0
        # 计算iou loss 这里只计算了有真值的地方的loss，其他地方不管
        reg_loss = self.reg_weight * self.iou_loss(
            reg_pred, reg_targets_pos, reg_weight_map,
            reduction='sum') / reg_norm
        losses['loss_centernet_loc'] = reg_loss
        # 在这里这个设置的是True
        if self.with_agn_hm:
            # 这里做的就相当于把flattened_hms的第一维给压缩了
            cat_agn_heatmap = flattened_hms.max(dim=1)[0] # M
            # 这里输出的就是centernet的正样本loss和负样本loss
            agn_pos_loss, agn_neg_loss = binary_heatmap_focal_loss_jit(
                agn_hm_pred.float(), cat_agn_heatmap.float(), pos_inds,
                alpha=self.hm_focal_alpha, 
                beta=self.hm_focal_beta, 
                gamma=self.loss_gamma,
                sigmoid_clamp=self.sigmoid_clamp,
                ignore_high_fp=self.ignore_high_fp,
            )
            agn_pos_loss = self.pos_weight * agn_pos_loss / num_pos_avg
            agn_neg_loss = self.neg_weight * agn_neg_loss / num_pos_avg
            # 两个loss，就是centernet1中的loss 真值为1，和真值不为1两种情况
            losses['loss_centernet_agn_pos'] = agn_pos_loss
            losses['loss_centernet_agn_neg'] = agn_neg_loss
    
        if self.debug:
            print('losses', losses)
            print('total_num_pos', total_num_pos)
        return losses


    def compute_grids(self, features):
        grids = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            # 默认strides=(8, 16, 32, 64, 128)
            shifts_x = torch.arange(
                0, w * self.strides[level], 
                step=self.strides[level],
                dtype=torch.float32, device=feature.device)
            shifts_y = torch.arange(
                0, h * self.strides[level], 
                step=self.strides[level],
                dtype=torch.float32, device=feature.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            grids_per_level = torch.stack((shift_x, shift_y), dim=1) + \
                self.strides[level] // 2
            grids.append(grids_per_level)
        return grids


    def _get_ground_truth(self, grids, shapes_per_level, gt_instances):
        '''
        Input:
            grids: list of tensors [(hl x wl, 2)]_l
            shapes_per_level: list of tuples L x 2:
            gt_instances: gt instances
        Retuen:
            pos_inds: N
            labels: N
            reg_targets: M x 4
            flattened_hms: M x C or M x 1
            N: number of objects in all images
            M: number of pixels from all FPN levels
        '''

        # get positive pixel index
        if not self.more_pos:# 这个东西的默认值是false
            # 得到的东西是 真值center坐标的索引，还要真值的列表
            pos_inds, labels = self._get_label_inds(
                gt_instances, shapes_per_level)
                # shapes_per_level 是[level数,2]
        else:
            pos_inds, labels = None, None
        heatmap_channels = self.num_classes
        L = len(grids)
        # 每一个level的点 也就是loc的数目 list 共5个值
        num_loc_list = [len(loc) for loc in grids]

        # shapes_per_level.new_ones(num_loc_list[l]) 尺寸和num_loc_list[l]一样
        # 类型和shapes_per_level一样，然后全是1 
        # self.strides[l] 就是(8,16,32,64,128)
        # strides 是把这些都连起来了 strides的shape是M，M = num_loc_list 所有元素的值相加 
        # strides = [8,8,8,8,8...16,16,16.......128,128,128]
        # 一共有num_loc_list[0]个8，num_loc_list[1]个16....num_loc_list[4]个128
        strides = torch.cat([
            shapes_per_level.new_ones(num_loc_list[l]) * self.strides[l] \
            for l in range(L)]).float() # M,M=num_loc_list[0]+num_loc_list[1]+...+num_loc_list[4]
        
        # strides的shape是M，reg_size_ranges的shape就是[M,2]
        # 把self.sizes_of_interest[l] 扩展成shape为（num_loc_list[l], 2）的数组
        # reg_size_ranges表示每个level所care的reg的size的范围
        reg_size_ranges = torch.cat([
            shapes_per_level.new_tensor(self.sizes_of_interest[l]).float().view(
            1, 2).expand(num_loc_list[l], 2) for l in range(L)]) # [M,2]
        # grids 也 cat 一下 gird的数目和num_loc_list的数目是相等的，都是M（可以去看笔记）
        # M 表示总共有多少个位置点
        grids = torch.cat(grids, dim=0) # [M,2]
        M = grids.shape[0]

        reg_targets = []
        flattened_hms = []
        for i in range(len(gt_instances)): # images
            boxes = gt_instances[i].gt_boxes.tensor # N x 4
            area = gt_instances[i].gt_boxes.area() # N
            gt_classes = gt_instances[i].gt_classes # N in [0, self.num_classes]

            N = boxes.shape[0]
            if N == 0:
                reg_targets.append(grids.new_zeros((M, 4)) - INF)
                flattened_hms.append(
                    grids.new_zeros((
                        M, 1 if self.only_proposal else heatmap_channels)))
                continue

            # 这里好像用到了广播机制
            # l是left t是top r是right b是bottom 也就是box边框距离grid点的位置
            l = grids[:, 0].view(M, 1) - boxes[:, 0].view(1, N) # M x N
            t = grids[:, 1].view(M, 1) - boxes[:, 1].view(1, N) # M x N
            r = boxes[:, 2].view(1, N) - grids[:, 0].view(M, 1) # M x N
            b = boxes[:, 3].view(1, N) - grids[:, 1].view(M, 1) # M x N
            reg_target = torch.stack([l, t, r, b], dim=2) # M x N x 4

            # 这里的center是box的中心 
            centers = ((boxes[:, [0, 1]] + boxes[:, [2, 3]]) / 2) # N x 2
            # 下面几步做的是，让centers和grid一致 center算出来是带小数点的，但像素不能是小数
            # (centers_expanded/strides_expanded).int() 这一步是center变到了缩小后的特征图的点上，int是四舍五入的操作
            # 再乘一个strides_expanded 就是再变回来的过程，和grid里的操作一样了，后面的 +strides_expanded/2 也是为了和grid一致
            # 这样的center是近似了之后的
            centers_expanded = centers.view(1, N, 2).expand(M, N, 2) # M x N x 2
            strides_expanded = strides.view(M, 1, 1).expand(M, N, 2)
            centers_discret = ((centers_expanded / strides_expanded).int() * \
                strides_expanded).float() + strides_expanded / 2 # M x N x 2
            
            # is_peak 看girds中的某一个点 是不是和center重合了
            is_peak = (((grids.view(M, 1, 2).expand(M, N, 2) - \
                centers_discret) ** 2).sum(dim=2) == 0) # M x N
            # is_in_boxes 看girds中的某个点，是不是在box里面
            # 根据他的距离定义，只要最小的那个值不是负数，就一定在，否则一定不在
            is_in_boxes = reg_target.min(dim=2)[0] > 0 # M x N
            # is_center3x3 看grid点是不是在center点的3*3周围
            is_center3x3 = self.get_center3x3(
                grids, centers, strides) & is_in_boxes # M x N
            # 看reg_target属于哪个level 这里作者也写了 要把它和assign_fpn_level合并
            is_cared_in_the_level = self.assign_reg_fpn(
                reg_target, reg_size_ranges) # M x N
            # reg_mask是 M*N 的True和Flase 
            # M表示这个点是第M个点（这个里头就包含了在哪个level），N表示第N个box
            # reg_mask中，在center点周围3*3区域，且在box内部，且被所在level care的点才是True
            reg_mask = is_center3x3 & is_cared_in_the_level # M x N

            # 得到的结果是所有grid和点centers距离的平方，这里的centers是没有近似过的
            # 注意 这里N个框分到了N个维度，相当于分了N层，每一层有M个网格点，但只有1个center点
            dist2 = ((grids.view(M, 1, 2).expand(M, N, 2) - \
                centers_expanded) ** 2).sum(dim=2) # M x N
            # peak的点 距离肯定是0
            dist2[is_peak] = 0
            # self.delta=(1-hm_min_overlap)/(1+hm_min_overlap) area=gt_boxes.area()
            radius2 = self.delta ** 2 * 2 * area # N
            # self.min_radius=4
            # torch.clamp 裁剪函数，radius2中小于min_radius平方的元素一律赋值min_radius平方
            radius2 = torch.clamp(
                radius2, min=self.min_radius ** 2)
            # 距离/radius2
            weighted_dist2 = dist2 / radius2.view(1, N).expand(M, N) # M x N            
            # 最后获得reg_target shape：M*4
            # 如果一个grid点不对应任何center点，那么他的那四个维度全是负无穷
            # 否则，它对应那个center点，存放的就是哪个center点的
            reg_target = self._get_reg_targets(
                reg_target, weighted_dist2.clone(), reg_mask, area) # M x 4

            if self.only_proposal:
                # 平铺了的heatmap，跟论文里的定义是一致的，值很小的地方会被置0
                flattened_hm = self._create_agn_heatmaps_from_dist(
                    weighted_dist2.clone()) # M x 1
            else:
                flattened_hm = self._create_heatmaps_from_dist(
                    weighted_dist2.clone(), gt_classes, 
                    channels=heatmap_channels) # M x C
            # 一个batch里有B个图，每个图append一次，所以现在的reg_target是一个8维的list
            # 每一维都是 M*4的数组，M代表一共M个点，4是box的那四个维度
            # flattened_hms也是一样的 hm 是heatmap的缩写
            reg_targets.append(reg_target)
            flattened_hms.append(flattened_hm)
        
        # transpose im first training_targets to level first ones
        # num_loc_list grid中点的数目，不论原图大小是多少，这个的数目是一定的
        # reg_targets，flattened_hms都变成了一个5维的list
        # list的每个维度，代表了一个level，每个level中包含所有batch的所有框的所有grid点
        # 比如说他的第一个维度，记图1的level1的点为1-1，图2的level1的点2-1
        # 那么他存的就是： 1-1，2-1,3-1,4-1.....8-1
        # 这些点按照顺序排在一起，但是是并在了一个维度里
        reg_targets = _transpose(reg_targets, num_loc_list)
        flattened_hms = _transpose(flattened_hms, num_loc_list)
        # 除以一下步长
        for l in range(len(reg_targets)):
            reg_targets[l] = reg_targets[l] / float(self.strides[l])
        # 再把他们的所有维度连起来
        # 记图1，level1的点为1-1,那么现在的 reg_targets就是：
        # 1-1，2-1,3-1.....8-1,1-2,2-2...8-2，.....1-5,2-5....8-5
        # flattened_hms 也是一样
        reg_targets = cat([x for x in reg_targets], dim=0) # MB x 4
        flattened_hms = cat([x for x in flattened_hms], dim=0) # MB x C
        return pos_inds, labels, reg_targets, flattened_hms


    def _get_label_inds(self, gt_instances, shapes_per_level):
        '''
        Inputs:
            gt_instances: [n_i], sum n_i = N
            shapes_per_level: L x 2 [(h_l, w_l)]_L
            shapes_per_level 是[level数,2]
        Returns:
            pos_inds: N'
            labels: N'
        '''
        pos_inds = []
        labels = []
        L = len(self.strides) # stride=(8,16,32,64,128) 所以L=5
        B = len(gt_instances) # 这个就是batch_size
        # 每个特征图的h和w
        shapes_per_level = shapes_per_level.long() # 将数字转化为一个长整型
        
        # 长度为5的数组，记录每个level共有几个位置（计算面积）
        loc_per_level = (shapes_per_level[:, 0] * shapes_per_level[:, 1]).long() # L
        level_bases = []
        s = 0
        for l in range(L):
            level_bases.append(s)
            s = s + B * loc_per_level[l] # 就是字面上那么算的，就是没有append最后一个s
        level_bases = shapes_per_level.new_tensor(level_bases).long() # L
        strides_default = shapes_per_level.new_tensor(self.strides).float() # L
        for im_i in range(B):
            targets_per_im = gt_instances[im_i]
            bboxes = targets_per_im.gt_boxes.tensor # n x 4
            n = bboxes.shape[0] #n是box的数目
            centers = ((bboxes[:, [0, 1]] + bboxes[:, [2, 3]]) / 2) # n x 2
            
            # 把centers扩展到每一个level上面
            centers = centers.view(n, 1, 2).expand(n, L, 2).contiguous()
            if self.not_clamp_box: # 这个默认是false
                h, w = gt_instances[im_i]._image_size
                centers[:, :, 0].clamp_(min=0).clamp_(max=w-1)
                centers[:, :, 1].clamp_(min=0).clamp_(max=h-1)
            # 这里把strides也去扩展一下，为了后面相除
            strides = strides_default.view(1, L, 1).expand(n, L, 2)
            # 就是把center映射到每一张特征图上
            # 想一下这个，像素值可以理解成是第几个点 例如(16,32)是原图上横着数第16个，竖着数第32个点
            # 那么现在把原图卷积成缩小了8倍的特征图，那这个点不就是横着数第2个，数着数第3个点了嘛
            centers_inds = (centers / strides).long() # n x L x 2
            Ws = shapes_per_level[:, 1].view(1, L).expand(n, L) # 只取宽，扩展到n L
            pos_ind = level_bases.view(1, L).expand(n, L) + \
                       im_i * loc_per_level.view(1, L).expand(n, L) + \
                       centers_inds[:, :, 1] * Ws + \
                       centers_inds[:, :, 0] # n x L
            # 根据box的大小，判定他属于哪个尺度的特征图
            is_cared_in_the_level = self.assign_fpn_level(bboxes)
            pos_ind = pos_ind[is_cared_in_the_level].view(-1)
            label = targets_per_im.gt_classes.view(
                n, 1).expand(n, L)[is_cared_in_the_level].view(-1)

            # pos_inds是一个n维的向量，n表示n个box，每个位置上是那个box中心点的索引
            # 前面已经将box归到了不同的特征图里
            pos_inds.append(pos_ind) # n'
            labels.append(label) # n'
        pos_inds = torch.cat(pos_inds, dim=0).long()
        labels = torch.cat(labels, dim=0)
        return pos_inds, labels # N, N （N不一定是多少，跟N个框那个不是一个东西）

    def assign_fpn_level(self, boxes):
        '''
        Inputs:
            boxes: n x 4
            size_ranges: L x 2
        Return:
            is_cared_in_the_level: n x L
        '''
        size_ranges = boxes.new_tensor(
            self.sizes_of_interest).view(len(self.sizes_of_interest), 2) # L x 2 Level个范围 2 表示一个最大值一个最小值
        crit = ((boxes[:, 2:] - boxes[:, :2]) **2).sum(dim=1) ** 0.5 / 2 # n 算box的大小 ((x^2+y^2)^0.5)/2
        n, L = crit.shape[0], size_ranges.shape[0]
        crit = crit.view(n, 1).expand(n, L)  
        size_ranges_expand = size_ranges.view(1, L, 2).expand(n, L, 2)  # 把这两个都扩展成n*L
        is_cared_in_the_level = (crit >= size_ranges_expand[:, :, 0]) & \
            (crit <= size_ranges_expand[:, :, 1])
        '''
        返回值的意思是每个box归于哪个level 例如 n=3 L=5 那么返回的就是
        True  False False False False
        False True  False False False
        False True  True  False False

        这种矩阵的形式 哪个地方是True,就表明box属于哪个level
        注意 因为范围是有重叠的,所以一个box可能会出现在两个level上
        '''
        return is_cared_in_the_level


    def assign_reg_fpn(self, reg_targets_per_im, size_ranges):
        '''
        TODO (Xingyi): merge it with assign_fpn_level
        Inputs:
            reg_targets_per_im: M x N x 4
            size_ranges: M x 2
        '''
        # (l+r)^2+(t+b)^2 开根号除以二 不知道算了个啥 对角线的一半？
        crit = ((reg_targets_per_im[:, :, :2] + \
            reg_targets_per_im[:, :, 2:])**2).sum(dim=2) ** 0.5 / 2 # M x N
        is_cared_in_the_level = (crit >= size_ranges[:, [0]]) & \
            (crit <= size_ranges[:, [1]])
        return is_cared_in_the_level


    def _get_reg_targets(self, reg_targets, dist, mask, area):
        '''
          reg_targets (M x N x 4): long tensor
          dist (M x N)
          is_*: M x N
        '''
        dist[mask == 0] = INF * 1.0
        # 取5个框里面，weighted_dist2最小的那个，把他的值和索引都取出来
        min_dist, min_inds = dist.min(dim=1) # M
        # 根据上面的weighted_dist2最小原则，把 M*N*4的框 变成了M*4 
        reg_targets_per_im = reg_targets[
            range(len(reg_targets)), min_inds] # M x N x 4 --> M x 4
        # 如果weighted_dist2的最小值是无穷，那么就把按个点的值，换成负无穷
        reg_targets_per_im[min_dist == INF] = - INF
        return reg_targets_per_im


    def _create_heatmaps_from_dist(self, dist, labels, channels):
        '''
        dist: M x N
        labels: N
        return:
          heatmaps: M x C
        '''
        heatmaps = dist.new_zeros((dist.shape[0], channels))
        for c in range(channels):
            inds = (labels == c) # N
            if inds.int().sum() == 0:
                continue
            heatmaps[:, c] = torch.exp(-dist[:, inds].min(dim=1)[0])
            zeros = heatmaps[:, c] < 1e-4
            heatmaps[zeros, c] = 0
        return heatmaps


    def _create_agn_heatmaps_from_dist(self, dist):
        '''
        TODO (Xingyi): merge it with _create_heatmaps_from_dist
        dist: M x N
        return:
          heatmaps: M x 1
        '''
        heatmaps = dist.new_zeros((dist.shape[0], 1))
        # torch.exp表示计算e的多少次方
        # 找到距离最小的那个点，计算e的dist次方
        # 注意，这里的heatmap是直接用dist算出来的，每一个点都是有值的
        # 只是值太小的，被置成了0
        heatmaps[:, 0] = torch.exp(-dist.min(dim=1)[0])
        zeros = heatmaps < 1e-4
        heatmaps[zeros] = 0
        return heatmaps


    def _flatten_outputs(self, clss, reg_pred, agn_hm_pred):
        # Reshape: (N, F, Hl, Wl) -> (N, Hl, Wl, F) -> (sum_l N*Hl*Wl, F)
        clss = cat([x.permute(0, 2, 3, 1).reshape(-1, x.shape[1]) \
            for x in clss], dim=0) if clss[0] is not None else None
        # permute的结果是[8,H,W,4] 之后进行reshape，就是：[8*H*W,4] 
        # 之后再依次首尾相接
        # 记图1的level1为1-1 图2的level1为2-1 
        # reshape后每个level的结果都是 1-1,2-1,3-1...8-1
        # 之后再将5个level进行cat 最终结果就是：1-1,2-1,3-1..8-1,1-2,2-2...8-2....1-5,2-5...8-5
        reg_pred = cat(
            [x.permute(0, 2, 3, 1).reshape(-1, 4) for x in reg_pred], dim=0)            
        agn_hm_pred = cat([x.permute(0, 2, 3, 1).reshape(-1) \
            for x in agn_hm_pred], dim=0) if self.with_agn_hm else None
        return clss, reg_pred, agn_hm_pred


    def get_center3x3(self, locations, centers, strides):
        '''
        Inputs:
            locations: M x 2
            centers: N x 2
            strides: M
        '''
        M, N = locations.shape[0], centers.shape[0]
        locations_expanded = locations.view(M, 1, 2).expand(M, N, 2) # M x N x 2
        centers_expanded = centers.view(1, N, 2).expand(M, N, 2) # M x N x 2
        strides_expanded = strides.view(M, 1, 1).expand(M, N, 2) # M x N
        centers_discret = ((centers_expanded / strides_expanded).int() * \
            strides_expanded).float() + strides_expanded / 2 # M x N x 2
        # 缩放前面已经说了，这样的话，每个center点都有一个grid与他重合
        # 然后 下面是回看center点与grid点的x，y坐标之差，是不是小于步长满足这个条件的
        # 就是center点，以及他周围一圈的，一共3*3=9个点
        '''
        * * *
        * c *
        * * *
        c表示center,* 加上 c 一共9个点,所以叫get_center3*3
        '''
        dist_x = (locations_expanded[:, :, 0] - centers_discret[:, :, 0]).abs()
        dist_y = (locations_expanded[:, :, 1] - centers_discret[:, :, 1]).abs()
        return (dist_x <= strides_expanded[:, :, 0]) & \
            (dist_y <= strides_expanded[:, :, 0])


    @torch.no_grad()
    def inference(self, images, clss_per_level, reg_pred_per_level, 
        agn_hm_pred_per_level, grids):
        logits_pred = [x.sigmoid() if x is not None else None \
            for x in clss_per_level]
        agn_hm_pred_per_level = [x.sigmoid() if x is not None else None \
            for x in agn_hm_pred_per_level]

        if self.only_proposal:
            proposals = self.predict_instances(
                grids, agn_hm_pred_per_level, reg_pred_per_level, 
                images.image_sizes, [None for _ in agn_hm_pred_per_level])
        else:
            proposals = self.predict_instances(
                grids, logits_pred, reg_pred_per_level, 
                images.image_sizes, agn_hm_pred_per_level)
        if self.as_proposal or self.only_proposal:
            for p in range(len(proposals)):
                proposals[p].proposal_boxes = proposals[p].get('pred_boxes')
                proposals[p].objectness_logits = proposals[p].get('scores')
                proposals[p].remove('pred_boxes')

        if self.debug:
            debug_test(
                [self.denormalizer(x) for x in images], 
                logits_pred, reg_pred_per_level, 
                agn_hm_pred_per_level, preds=proposals,
                vis_thresh=self.vis_thresh, 
                debug_show_name=False)
        return proposals, {}


    @torch.no_grad()
    def predict_instances(
        self, grids, logits_pred, reg_pred, image_sizes, agn_hm_pred, 
        is_proposal=False):
        # 这里的logits_pred对应的是agn_hm_pred_per_level reg_pred对应reg_pred_per_level
        # logits_pred list，level数个[8,1,h,w]，预测的heatmap
        # reg_pred list，level数个[8,4,h,w]
        # image_sizes,list 8个(h,w) 存的就是每一张图的长宽
        # 然后agn_hm_pred是5个none
        sampled_boxes = []
        for l in range(len(grids)):
            sampled_boxes.append(self.predict_single_level(
                grids[l], logits_pred[l], reg_pred[l] * self.strides[l],
                image_sizes, agn_hm_pred[l], l, is_proposal=is_proposal))
        # 原本的boxlists是level数个 图片数个 instance
        # 现在变成 图片数个 level数个 instance
        boxlists = list(zip(*sampled_boxes))
        # 再把不同level数上的
        boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
        # 这里把box nms到200个
        boxlists = self.nms_and_topK(
            boxlists, nms=not self.not_nms)
        return boxlists

    
    @torch.no_grad()
    def predict_single_level(
        self, grids, heatmap, reg_pred, image_sizes, agn_hm, level, 
        is_proposal=False):
        # 这里的heatmap对应的agn_hm_pred_per_level reg_pred是预测值 agn_hm是热图的预测值
        # heatmap，level数个[8,1,h,w]，预测的heatmap
        # reg_pred list，level数个[8,4,h,w]
        # image_sizes,list 8个(h,w) 存的就是每一张图的长宽
        # 然后agn_hm_pred是5个none
        N, C, H, W = heatmap.shape
        # put in the same format as grids
        if self.center_nms:
            heatmap_nms = nn.functional.max_pool2d(
                heatmap, (3, 3), stride=1, padding=1)
            heatmap = heatmap * (heatmap_nms == heatmap).float()
        heatmap = heatmap.permute(0, 2, 3, 1) # N x H x W x C
        heatmap = heatmap.reshape(N, -1, C) # N x HW x C
        box_regression = reg_pred.view(N, 4, H, W).permute(0, 2, 3, 1) # N x H x W x 4 
        box_regression = box_regression.reshape(N, -1, 4) #  N x HW x 4

        candidate_inds = heatmap > self.score_thresh # 0.05
        pre_nms_top_n = candidate_inds.contiguous().view(N, -1).sum(1) # N
        pre_nms_topk = self.pre_nms_topk_train if self.training else self.pre_nms_topk_test
        pre_nms_top_n = pre_nms_top_n.clamp(max=pre_nms_topk) # N

        if agn_hm is not None: # 默认值是none
            agn_hm = agn_hm.view(N, 1, H, W).permute(0, 2, 3, 1)
            agn_hm = agn_hm.reshape(N, -1)
            heatmap = heatmap * agn_hm[:, :, None]

        results = []
        for i in range(N):
            # 每一张图中box所属的类别 这里都是一类
            per_box_cls = heatmap[i] # HW x C
            # 每一个张图的candidate_inds
            per_candidate_inds = candidate_inds[i] # n
            # 将heat_map值>0.05的box筛选出来
            per_box_cls = per_box_cls[per_candidate_inds] # n
            
            # .nonzero() 是Python中的函数，返回两个数组
            # 第一个是非零元素的行索引，第二个是非零元素的列索引
            per_candidate_nonzeros = per_candidate_inds.nonzero() # n
            # 用第一个数组筛选，得到的是位置
            per_box_loc = per_candidate_nonzeros[:, 0] # n
            # 用第二个数组筛选，得到的是类别
            per_class = per_candidate_nonzeros[:, 1] # n

            per_box_regression = box_regression[i] # HW x 4
            # 用位置，提取出box的4个维度
            per_box_regression = per_box_regression[per_box_loc] # n x 4
            # 用位置信息，提取坐标（x,y）
            per_grids = grids[per_box_loc] # n x 2

            per_pre_nms_top_n = pre_nms_top_n[i] # 1

            # 看一下 是不是提取的多了
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                # .topK是pytorch的函数，作用是按行提取出topk，并输出元素是否为topk的true，False
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_grids = per_grids[top_k_indices]
            
            # 中心点，加上偏移量
            detections = torch.stack([
                per_grids[:, 0] - per_box_regression[:, 0],
                per_grids[:, 1] - per_box_regression[:, 1],
                per_grids[:, 0] + per_box_regression[:, 2],
                per_grids[:, 1] + per_box_regression[:, 3],
            ], dim=1) # n x 4

            # avoid invalid boxes in RoI heads
            detections[:, 2] = torch.max(detections[:, 2], detections[:, 0] + 0.01)
            detections[:, 3] = torch.max(detections[:, 3], detections[:, 1] + 0.01)
            boxlist = Instances(image_sizes[i])
            boxlist.scores = torch.sqrt(per_box_cls) \
                if self.with_agn_hm else per_box_cls # n
            # import pdb; pdb.set_trace()
            boxlist.pred_boxes = Boxes(detections)
            boxlist.pred_classes = per_class
            results.append(boxlist)
        return results

    
    @torch.no_grad()
    def nms_and_topK(self, boxlists, nms=True):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            nms_thresh = self.nms_thresh_train if self.training else \
                self.nms_thresh_test
            result = ml_nms(boxlists[i], nms_thresh) if nms else boxlists[i]
            if self.debug:
                print('#proposals before nms', len(boxlists[i]))
                print('#proposals after nms', len(result))
            num_dets = len(result)
            post_nms_topk = self.post_nms_topk_train if self.training else \
                self.post_nms_topk_test
            if num_dets > post_nms_topk:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(
                    cls_scores.float().cpu(),
                    num_dets - post_nms_topk + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            if self.debug:
                print('#proposals after filter', len(result))
            results.append(result)
        return results

    
    @torch.no_grad()
    def _add_more_pos(self, reg_pred, gt_instances, shapes_per_level):
        labels, level_masks, c33_inds, c33_masks, c33_regs = \
            self._get_c33_inds(gt_instances, shapes_per_level)
        N, L, K = labels.shape[0], len(self.strides), 9
        c33_inds[c33_masks == 0] = 0
        reg_pred_c33 = reg_pred[c33_inds].detach() # N x L x K
        invalid_reg = c33_masks == 0
        c33_regs_expand = c33_regs.view(N * L * K, 4).clamp(min=0)
        if N > 0:
            with torch.no_grad():
                c33_reg_loss = self.iou_loss(
                    reg_pred_c33.view(N * L * K, 4), 
                    c33_regs_expand, None,
                    reduction='none').view(N, L, K).detach() # N x L x K
        else:
            c33_reg_loss = reg_pred_c33.new_zeros((N, L, K)).detach()
        c33_reg_loss[invalid_reg] = INF # N x L x K
        c33_reg_loss.view(N * L, K)[level_masks.view(N * L), 4] = 0 # real center
        c33_reg_loss = c33_reg_loss.view(N, L * K)
        if N == 0:
            loss_thresh = c33_reg_loss.new_ones((N)).float()
        else:
            loss_thresh = torch.kthvalue(
                c33_reg_loss, self.more_pos_topk, dim=1)[0] # N
        loss_thresh[loss_thresh > self.more_pos_thresh] = self.more_pos_thresh # N
        new_pos = c33_reg_loss.view(N, L, K) < \
            loss_thresh.view(N, 1, 1).expand(N, L, K)
        pos_inds = c33_inds[new_pos].view(-1) # P
        labels = labels.view(N, 1, 1).expand(N, L, K)[new_pos].view(-1)
        return pos_inds, labels
        
    
    @torch.no_grad()
    def _get_c33_inds(self, gt_instances, shapes_per_level):
        '''
        TODO (Xingyi): The current implementation is ugly. Refactor.
        Get the center (and the 3x3 region near center) locations of each objects
        Inputs:
            gt_instances: [n_i], sum n_i = N
            shapes_per_level: L x 2 [(h_l, w_l)]_L
        '''
        labels = []
        level_masks = []
        c33_inds = []
        c33_masks = []
        c33_regs = []
        L = len(self.strides)
        B = len(gt_instances)
        shapes_per_level = shapes_per_level.long()
        loc_per_level = (shapes_per_level[:, 0] * shapes_per_level[:, 1]).long() # L
        level_bases = []
        s = 0
        for l in range(L):
            level_bases.append(s)
            s = s + B * loc_per_level[l]
        level_bases = shapes_per_level.new_tensor(level_bases).long() # L
        strides_default = shapes_per_level.new_tensor(self.strides).float() # L
        K = 9
        dx = shapes_per_level.new_tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1]).long()
        dy = shapes_per_level.new_tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1]).long()
        for im_i in range(B):
            targets_per_im = gt_instances[im_i]
            bboxes = targets_per_im.gt_boxes.tensor # n x 4
            n = bboxes.shape[0]
            if n == 0:
                continue
            centers = ((bboxes[:, [0, 1]] + bboxes[:, [2, 3]]) / 2) # n x 2
            centers = centers.view(n, 1, 2).expand(n, L, 2)

            strides = strides_default.view(1, L, 1).expand(n, L, 2) # 
            centers_inds = (centers / strides).long() # n x L x 2

            # 这里有修改 原本的被注释了
            # center_grids = centers_inds * strides + strides // 2  # n x L x 2
            center_grids = centers_inds * strides + torch.div(strides,2,rounding_mode='trunc') # n x L x 2
            l = center_grids[:, :, 0] - bboxes[:, 0].view(n, 1).expand(n, L)
            t = center_grids[:, :, 1] - bboxes[:, 1].view(n, 1).expand(n, L)
            r = bboxes[:, 2].view(n, 1).expand(n, L) - center_grids[:, :, 0]
            b = bboxes[:, 3].view(n, 1).expand(n, L) - center_grids[:, :, 1] # n x L
            reg = torch.stack([l, t, r, b], dim=2) # n x L x 4
            reg = reg / strides_default.view(1, L, 1).expand(n, L, 4).float()
            
            Ws = shapes_per_level[:, 1].view(1, L).expand(n, L)
            Hs = shapes_per_level[:, 0].view(1, L).expand(n, L)
            expand_Ws = Ws.view(n, L, 1).expand(n, L, K)
            expand_Hs = Hs.view(n, L, 1).expand(n, L, K)
            label = targets_per_im.gt_classes.view(n).clone()
            mask = reg.min(dim=2)[0] >= 0 # n x L
            mask = mask & self.assign_fpn_level(bboxes)
            labels.append(label) # n
            level_masks.append(mask) # n x L

            Dy = dy.view(1, 1, K).expand(n, L, K)
            Dx = dx.view(1, 1, K).expand(n, L, K)
            c33_ind = level_bases.view(1, L, 1).expand(n, L, K) + \
                       im_i * loc_per_level.view(1, L, 1).expand(n, L, K) + \
                       (centers_inds[:, :, 1:2].expand(n, L, K) + Dy) * expand_Ws + \
                       (centers_inds[:, :, 0:1].expand(n, L, K) + Dx) # n x L x K
            
            c33_mask = \
                ((centers_inds[:, :, 1:2].expand(n, L, K) + dy) < expand_Hs) & \
                ((centers_inds[:, :, 1:2].expand(n, L, K) + dy) >= 0) & \
                ((centers_inds[:, :, 0:1].expand(n, L, K) + dx) < expand_Ws) & \
                ((centers_inds[:, :, 0:1].expand(n, L, K) + dx) >= 0)
            # TODO (Xingyi): think about better way to implement this
            # Currently it hard codes the 3x3 region
            c33_reg = reg.view(n, L, 1, 4).expand(n, L, K, 4).clone()
            c33_reg[:, :, [0, 3, 6], 0] -= 1
            c33_reg[:, :, [0, 3, 6], 2] += 1
            c33_reg[:, :, [2, 5, 8], 0] += 1
            c33_reg[:, :, [2, 5, 8], 2] -= 1
            c33_reg[:, :, [0, 1, 2], 1] -= 1
            c33_reg[:, :, [0, 1, 2], 3] += 1
            c33_reg[:, :, [6, 7, 8], 1] += 1
            c33_reg[:, :, [6, 7, 8], 3] -= 1
            c33_mask = c33_mask & (c33_reg.min(dim=3)[0] >= 0) # n x L x K
            c33_inds.append(c33_ind)
            c33_masks.append(c33_mask)
            c33_regs.append(c33_reg)
        
        if len(level_masks) > 0:
            labels = torch.cat(labels, dim=0)
            level_masks = torch.cat(level_masks, dim=0)
            c33_inds = torch.cat(c33_inds, dim=0).long()
            c33_regs = torch.cat(c33_regs, dim=0)
            c33_masks = torch.cat(c33_masks, dim=0)
        else:
            labels = shapes_per_level.new_zeros((0)).long()
            level_masks = shapes_per_level.new_zeros((0, L)).bool()
            c33_inds = shapes_per_level.new_zeros((0, L, K)).long()
            c33_regs = shapes_per_level.new_zeros((0, L, K, 4)).float()
            c33_masks = shapes_per_level.new_zeros((0, L, K)).bool()
        return labels, level_masks, c33_inds, c33_masks, c33_regs # N x L, N x L x K
