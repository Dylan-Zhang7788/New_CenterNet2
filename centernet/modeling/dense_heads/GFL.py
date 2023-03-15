import math
import torch
import torch.nn.functional as F
from torch import nn
from atss_core.layers import Scale
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from functools import partial
from atss_core.modeling.rpn.anchor_generator import AnchorGenerator
from atss_core.structures.bounding_box import BoxList
from atss_core.structures.boxlist_ops import boxlist_iou
from atss_core.structures.boxlist_ops import cat_boxlist
from .MY_mm_func import images_to_levels,  anchor_inside_flags, unmap

INF = 100000000


class BoxCoder(object):

    def __init__(self, cfg):
        self.cfg = cfg

    def encode(self, gt_boxes, anchors):
        if self.cfg.MODEL.ATSS.REGRESSION_TYPE == 'POINT':
            TO_REMOVE = 1  # TODO remove
            anchors_w = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
            anchors_h = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
            anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
            anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2

            w = self.cfg.MODEL.ATSS.ANCHOR_SIZES[0] / self.cfg.MODEL.ATSS.ANCHOR_STRIDES[0]
            l = w * (anchors_cx - gt_boxes[:, 0]) / anchors_w
            t = w * (anchors_cy - gt_boxes[:, 1]) / anchors_h
            r = w * (gt_boxes[:, 2] - anchors_cx) / anchors_w
            b = w * (gt_boxes[:, 3] - anchors_cy) / anchors_h
            targets = torch.stack([l, t, r, b], dim=1)
        elif self.cfg.MODEL.ATSS.REGRESSION_TYPE == 'BOX':
            TO_REMOVE = 1  # TODO remove
            ex_widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
            ex_heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
            ex_ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
            ex_ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

            gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + TO_REMOVE
            gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + TO_REMOVE
            gt_ctr_x = (gt_boxes[:, 2] + gt_boxes[:, 0]) / 2
            gt_ctr_y = (gt_boxes[:, 3] + gt_boxes[:, 1]) / 2

            wx, wy, ww, wh = (10., 10., 5., 5.)
            targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
            targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
            targets_dw = ww * torch.log(gt_widths / ex_widths)
            targets_dh = wh * torch.log(gt_heights / ex_heights)
            targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)

        return targets

    def decode(self, preds, anchors):
        if self.cfg.MODEL.ATSS.REGRESSION_TYPE == 'POINT':
            TO_REMOVE = 1  # TODO remove
            anchors_w = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
            anchors_h = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
            anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
            anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2

            w = self.cfg.MODEL.ATSS.ANCHOR_SIZES[0] / self.cfg.MODEL.ATSS.ANCHOR_STRIDES[0]
            x1 = anchors_cx - preds[:, 0] / w * anchors_w
            y1 = anchors_cy - preds[:, 1] / w * anchors_h
            x2 = anchors_cx + preds[:, 2] / w * anchors_w
            y2 = anchors_cy + preds[:, 3] / w * anchors_h
            pred_boxes = torch.stack([x1, y1, x2, y2], dim=1)
        elif self.cfg.MODEL.ATSS.REGRESSION_TYPE == 'BOX':
            anchors = anchors.to(preds.dtype)

            TO_REMOVE = 1  # TODO remove
            widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
            heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
            ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
            ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

            wx, wy, ww, wh = (10., 10., 5., 5.)
            dx = preds[:, 0::4] / wx
            dy = preds[:, 1::4] / wy
            dw = preds[:, 2::4] / ww
            dh = preds[:, 3::4] / wh

            # Prevent sending too large values into torch.exp()
            dw = torch.clamp(dw, max=math.log(1000. / 16))
            dh = torch.clamp(dh, max=math.log(1000. / 16))

            pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
            pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
            pred_w = torch.exp(dw) * widths[:, None]
            pred_h = torch.exp(dh) * heights[:, None]

            pred_boxes = torch.zeros_like(preds)
            pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * (pred_w - 1)
            pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * (pred_h - 1)
            pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * (pred_w - 1)
            pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * (pred_h - 1)
        return pred_boxes

class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x
    
class MY_GFL_AnchorGenerator(AnchorGenerator):
    def __init__(self,
        sizes=(128, 256, 512),
        aspect_ratios=(0.5, 1.0, 2.0),
        anchor_strides=(8, 16, 32),
        straddle_thresh=0,):
        super(MY_GFL_AnchorGenerator,self).__init__(sizes, aspect_ratios, anchor_strides, straddle_thresh)
    
    def forward(self, image_list, feature_maps):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
        GFL_anchors = []
        ATSS_anchors = []
        valid_flag = []
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            ATSS_anchors_in_image=[]
            valid_flag_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                valid_flag_per_feature_map=torch.ones_like(anchors_per_feature_map).bool()[:,0]
                valid_flag_image.append(valid_flag_per_feature_map)
                boxlist = BoxList(
                    anchors_per_feature_map, (image_height, image_width), mode="xyxy"
                )
                self.add_visibility_to(boxlist)
                ATSS_anchors_in_image.append(boxlist)

            GFL_anchors.append(anchors_over_all_feature_maps)
            ATSS_anchors.append(ATSS_anchors_in_image)
            valid_flag.append( valid_flag_image)
        return GFL_anchors,ATSS_anchors,valid_flag

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
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def make_anchor_generator_gfl(config):
    anchor_sizes = config.MODEL.ATSS.ANCHOR_SIZES  # (64,128,256,512,1024)
    aspect_ratios = config.MODEL.ATSS.ASPECT_RATIOS # (1.0)
    anchor_strides = config.MODEL.ATSS.ANCHOR_STRIDES # (8,16,32,64,128)
    straddle_thresh = config.MODEL.ATSS.STRADDLE_THRESH # 0
    octave = config.MODEL.ATSS.OCTAVE # 2.0
    scales_per_octave = config.MODEL.ATSS.SCALES_PER_OCTAVE # 1

    assert len(anchor_strides) == len(anchor_sizes), "Only support FPN now"
    new_anchor_sizes = []
    for size in anchor_sizes: # anchor_sizes = (64,128,256,512,1024)
        per_layer_anchor_sizes = []
        for scale_per_octave in range(scales_per_octave): # scales_per_octave = 1
            octave_scale = octave ** (scale_per_octave / float(scales_per_octave))
            per_layer_anchor_sizes.append(octave_scale * size)
        new_anchor_sizes.append(tuple(per_layer_anchor_sizes))
    # 最后得到的结果：[(64.0,),(128.0,),(256.0,),(512.0,),(1024.0,)]

    anchor_generator = MY_GFL_AnchorGenerator(
        tuple(new_anchor_sizes), aspect_ratios, anchor_strides, straddle_thresh) 
    # new_anchor_sizes=[(64.0,),(128.0,),(256.0,),(512.0,),(1024.0,)]
    # aspect_ratios=(0.5,1.0,2.0) anchor_strides=(8,16,32,64,128) straddle_thresh= 0
    return anchor_generator

@PROPOSAL_GENERATOR_REGISTRY.register()
class MY_GFLModule(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(MY_GFLModule, self).__init__()
        self.cfg = cfg
        self.in_features=("p3", "p4", "p5", "p6", "p7")
        self.reg_max = 16
        # num_classes = cfg.MODEL.ATSS.NUM_CLASSES - 1
        num_classes = 1 # 后面改的 写了固定值 只有正样本这一个类别
        num_anchors = len(cfg.MODEL.ATSS.ASPECT_RATIOS) * cfg.MODEL.ATSS.SCALES_PER_OCTAVE
        in_channels=160 # 这个地方写了固定值后面出错的话要改
        self.box_coder = BoxCoder(cfg)
        self.anchor_generator = make_anchor_generator_gfl(cfg)
        '''
        下面是我补充的GFL
        '''
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(4):
            chn = 160
            self.cls_convs.append(nn.Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            self.cls_convs.append(nn.GroupNorm(32, 160, eps=1e-05, affine=True))
            self.cls_convs.append(nn.ReLU(inplace=True))
            self.reg_convs.append(nn.Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            self.reg_convs.append(nn.GroupNorm(32, 160, eps=1e-05, affine=True))
            self.reg_convs.append(nn.ReLU(inplace=True))
        self.gfl_cls = nn.Conv2d(160, num_classes, 3, padding=1)
        self.gfl_reg = nn.Conv2d(160, 4 * (self.reg_max + 1), 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(5)])

        for modules in [self.cls_convs, self.reg_convs,
                        self.gfl_cls, self.gfl_reg ,
                        self.scales]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

    def ATSS_prepare_targets(self, targets, anchors):
            cls_labels = []
            reg_targets = []
            for im_i in range(len(targets)):
                targets_per_im = targets[im_i]   # 每张图里的真值样本框的数量 不一定是多少 假设形状是[N,4]
                # assert targets_per_im.mode == "xyxy"
                bboxes_per_im = targets_per_im.gt_boxes.tensor
                labels_per_im = torch.ones_like(targets_per_im.gt_classes)
                # anchors 5个尺度上的每一个点上都有一个anchor()
                anchors_per_im =cat_boxlist(anchors[im_i])
                num_gt = bboxes_per_im.shape[0]

                if self.cfg.MODEL.ATSS.POSITIVE_TYPE == 'ATSS':
                    num_anchors_per_loc = len(self.cfg.MODEL.ATSS.ASPECT_RATIOS) * self.cfg.MODEL.ATSS.SCALES_PER_OCTAVE

                    num_anchors_per_level = [len(anchors_per_level.bbox) for anchors_per_level in anchors[im_i]]
                    # ious 的形状是 [N,M] N是anchor的总数目，M是真值的总数目
                    ious = boxlist_iou(anchors_per_im, targets_per_im)

                    gt_cx = (bboxes_per_im[:, 2] + bboxes_per_im[:, 0]) / 2.0
                    gt_cy = (bboxes_per_im[:, 3] + bboxes_per_im[:, 1]) / 2.0
                    gt_points = torch.stack((gt_cx, gt_cy), dim=1)

                    anchors_cx_per_im = (anchors_per_im.bbox[:, 2] + anchors_per_im.bbox[:, 0]) / 2.0
                    anchors_cy_per_im = (anchors_per_im.bbox[:, 3] + anchors_per_im.bbox[:, 1]) / 2.0
                    anchor_points = torch.stack((anchors_cx_per_im, anchors_cy_per_im), dim=1)
                    # distances的形状也是 [N,M] N是anchor的总数目，M是真值的总数目
                    distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()

                    # Selecting candidates based on the center distance between anchor box and object
                    candidate_idxs = []
                    # 为anchor分配索引
                    star_idx = 0
                    for level, anchors_per_level in enumerate(anchors[im_i]):
                        end_idx = star_idx + num_anchors_per_level[level]
                        # 将每一个level上的距离取出
                        distances_per_level = distances[star_idx:end_idx, :]
                        topk = min(self.cfg.MODEL.ATSS.TOPK * num_anchors_per_loc, num_anchors_per_level[level])
                        # 取出单个level上于每个真值距离值最小的9*3个anchor的索引，注意 每个真值，都取9*3个anchor
                        _, topk_idxs_per_level = distances_per_level.topk(topk, dim=0, largest=False)
                        # 加上该level的初始索引 就得到了这9个anchor的绝对索引
                        candidate_idxs.append(topk_idxs_per_level + star_idx)
                        star_idx = end_idx
                    # 将各个level上的anchor的索引连接起来
                    # candidate_idxs的形状是 [5*27,M] 5*27是5层每一层选3*9个框，M是真值的总数目
                    # 注意，不同真值对应的索引可能是同一个！！！！！！！
                    # 并且不同的真值 对应的索引的范围 也是相同的！！！！！！！！！
                    candidate_idxs = torch.cat(candidate_idxs, dim=0)

                    # Using the sum of mean and standard deviation as the IoU threshold to select final positive samples
                    # 根据anchor的索引，提取出这几个anchor与对应真值的IOU
                    # candidate_ious的shape是[N2,M] N2是根据距离筛选剩下的anchor的数目，M是真值的数目
                    candidate_ious = ious[candidate_idxs, torch.arange(num_gt)]
                    iou_mean_per_gt = candidate_ious.mean(0)
                    iou_std_per_gt = candidate_ious.std(0)
                    # mean 和 std 的shape都是[M]一维数组，M是真值数目
                    iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
                    # iou_thresh_per_gt[None, :] 的shape是[1,M]
                    # anchor中 与某个gt的iou大于阈值的，才是该真值对应的正样本
                    is_pos = candidate_ious >= iou_thresh_per_gt[None, :]

                    # Limiting the final positive samples’ center to object
                    # anchor_num 是anchor的总数目 比如每个点有三个anchor 那anchor_num就是所有level总网格点数目的三倍
                    anchor_num = anchors_cx_per_im.shape[0]
                    for ng in range(num_gt): 
                        # 这里做的目的是后续把candidate_idxs展平，跟之前的candidate_idxs对照就可以知道原理
                        candidate_idxs[:, ng] += ng * anchor_num
                    e_anchors_cx = anchors_cx_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
                    e_anchors_cy = anchors_cy_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
                    # 这里把candidate_idxs展平了
                    candidate_idxs = candidate_idxs.view(-1)
                    l = e_anchors_cx[candidate_idxs].view(-1, num_gt) - bboxes_per_im[:, 0]
                    t = e_anchors_cy[candidate_idxs].view(-1, num_gt) - bboxes_per_im[:, 1]
                    r = bboxes_per_im[:, 2] - e_anchors_cx[candidate_idxs].view(-1, num_gt)
                    b = bboxes_per_im[:, 3] - e_anchors_cy[candidate_idxs].view(-1, num_gt)
                    is_in_gts = torch.stack([l, t, r, b], dim=1).min(dim=1)[0] > 0.01
                    # 这里的is_pos的形状依然是[5*27,M]，因为这里的is_pos还没有被展平，后面也被展平了
                    is_pos = is_pos & is_in_gts

                    # if an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
                    # 生成一个和iou形状相同但是值全部是负无穷的张量，转置一下，之后把它也展平
                    ious_inf = torch.full_like(ious, -INF).t().contiguous().view(-1)
                    # 根据 is_pos 提取出 candidate_idxs 中被定义为正样本的anchor的索引 这个地方view不view无所谓，因为前面candidate_idxs就已经被展平了
                    index = candidate_idxs.view(-1)[is_pos.view(-1)]
                    # 将正样本索引处的iou赋值，iou里是多少，这里就是多少
                    ious_inf[index] = ious.t().contiguous().view(-1)[index]
                    # 再改变一下形状 现在 ious_inf 的形状是[N,M] N是anchor的总数目，M是真值的总数目
                    ious_inf = ious_inf.view(num_gt, -1).t()
                    # 如果一个anchor被分配给了多个gt，那么取与他iou最大的那个gt
                    anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)
                    cls_labels_per_im = labels_per_im[anchors_to_gt_indexs]
                    # 其他的anchor一律认为是背景，类别为0
                    cls_labels_per_im[anchors_to_gt_values == -INF] = 0
                    # bboxes_per_im形状是[M,4]存放的是M个真值的边框信息
                    # 注意 和所有gt的iou都是0的anchor，他的 matched_gts 信息记录的是第0个真值框的信息
                    # 这个其实影响不到什么 因为后续背景类是不会参与边框loss的计算的，所以他匹配的真值边框是多少都无所谓
                    matched_gts = bboxes_per_im[anchors_to_gt_indexs]

                reg_targets_per_im = self.box_coder.encode(matched_gts, anchors_per_im.bbox)
                cls_labels.append(cls_labels_per_im)
                reg_targets.append(reg_targets_per_im)

            return cls_labels, reg_targets

    def prepare_gt_box_and_class(self,image_list,targets):
        gt_bboxes = []
        gt_labels = []
        img_metas = []
        for i in range(len(image_list)):
            gt_bboxes_per_img = targets[i].gt_boxes.tensor
            gt_labels_per_img = targets[i].gt_classes
            gt_bboxes.append(gt_bboxes_per_img)
            gt_labels.append(gt_labels_per_img)
        return gt_bboxes,gt_labels,

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Get targets for GFL head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(anchor_list) 

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg)

    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           label_channels=1,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors, 4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                anchors (Tensor): All anchors in the image with shape (N, 4).
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4).
                pos_inds (Tensor): Indices of positive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        # assign gt and sample anchors
        inside_flags=torch.ones_like(flat_anchors[:,0]).bool()
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
            split_inside_flags = torch.split(inside_flags, num_level_anchors)
            num_level_anchors_inside = [
                int(flags.sum()) for flags in split_inside_flags
            ]
            return num_level_anchors_inside

    def forward_single(self, x, scale):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls and quality joint scores for a single
                    scale level the channel number is num_classes.
                bbox_pred (Tensor): Box distribution logits for a single scale
                    level, the channel number is 4*(n+1), n is max value of
                    integral set.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.gfl_cls(cls_feat)
        bbox_pred = scale(self.gfl_reg(reg_feat)).float()
        return cls_score, bbox_pred

    def forward(self, images, features_dict, targets=None):
        features = [features_dict[f] for f in self.in_features]
        gfl_anchor_list, atss_anchor_list,valid_flag_list = self.anchor_generator(images, features)
        box_cls, box_regression = multi_apply(self.forward_single, features, self.scales)
        gt_bboxes,gt_labels=self.prepare_gt_box_and_class(images,targets)
        cls_reg_targets = self.get_targets(
            gfl_anchor_list,
            valid_flag_list,
            gt_bboxes,
            gt_bboxes_ignore_list=None,
            gt_labels_list=gt_labels,
            label_channels=1)



        labels, reg_targets = self.ATSS_prepare_targets(targets, atss_anchor_list)

        
