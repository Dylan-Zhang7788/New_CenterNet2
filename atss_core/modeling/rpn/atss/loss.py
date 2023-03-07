import torch
from torch import nn
import os
from ..utils import concat_box_prediction_layers
from atss_core.layers import SigmoidFocalLoss
from atss_core.modeling.matcher import Matcher
from atss_core.structures.boxlist_ops import boxlist_iou
from atss_core.structures.boxlist_ops import cat_boxlist


INF = 100000000


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor


class ATSSLossComputation(object):

    def __init__(self, cfg, box_coder):
        self.cfg = cfg
        self.cls_loss_func = SigmoidFocalLoss(cfg.MODEL.ATSS.LOSS_GAMMA, cfg.MODEL.ATSS.LOSS_ALPHA)
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.matcher = Matcher(cfg.MODEL.ATSS.FG_IOU_THRESHOLD, cfg.MODEL.ATSS.BG_IOU_THRESHOLD, True)
        self.box_coder = box_coder

    def GIoULoss(self, pred, target, anchor, weight=None):
        pred_boxes = self.box_coder.decode(pred.view(-1, 4), anchor.view(-1, 4))
        pred_x1 = pred_boxes[:, 0]
        pred_y1 = pred_boxes[:, 1]
        pred_x2 = pred_boxes[:, 2]
        pred_y2 = pred_boxes[:, 3]
        pred_x2 = torch.max(pred_x1, pred_x2)
        pred_y2 = torch.max(pred_y1, pred_y2)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

        gt_boxes = self.box_coder.decode(target.view(-1, 4), anchor.view(-1, 4))
        target_x1 = gt_boxes[:, 0]
        target_y1 = gt_boxes[:, 1]
        target_x2 = gt_boxes[:, 2]
        target_y2 = gt_boxes[:, 3]
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

        x1_intersect = torch.max(pred_x1, target_x1)
        y1_intersect = torch.max(pred_y1, target_y1)
        x2_intersect = torch.min(pred_x2, target_x2)
        y2_intersect = torch.min(pred_y2, target_y2)
        area_intersect = torch.zeros(pred_x1.size()).to(pred)
        mask = (y2_intersect > y1_intersect) * (x2_intersect > x1_intersect)
        area_intersect[mask] = (x2_intersect[mask] - x1_intersect[mask]) * (y2_intersect[mask] - y1_intersect[mask])

        x1_enclosing = torch.min(pred_x1, target_x1)
        y1_enclosing = torch.min(pred_y1, target_y1)
        x2_enclosing = torch.max(pred_x2, target_x2)
        y2_enclosing = torch.max(pred_y2, target_y2)
        area_enclosing = (x2_enclosing - x1_enclosing) * (y2_enclosing - y1_enclosing) + 1e-7

        area_union = pred_area + target_area - area_intersect + 1e-7
        ious = area_intersect / area_union
        gious = ious - (area_enclosing - area_union) / area_enclosing

        losses = 1 - gious

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()

    def prepare_targets(self, targets, anchors):
        cls_labels = []
        reg_targets = []
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]   # 每张图里的真值样本框的数量 不一定是多少 假设形状是[N,4]
            # assert targets_per_im.mode == "xyxy"
            bboxes_per_im = targets_per_im.gt_boxes.tensor
            labels_per_im = torch.ones_like(targets_per_im.gt_classes)
            # anchors 5个尺度上的每一个点上都有一个anchor()
            anchors_per_im = cat_boxlist(anchors[im_i])
            num_gt = bboxes_per_im.shape[0]
            if num_gt == 0: 
                cls_labels_per_im=torch.zeros((anchors_per_im.bbox.shape[0])).cuda()
                reg_targets_per_im=torch.zeros_like(anchors_per_im.bbox).cuda()
                cls_labels.append(cls_labels_per_im)
                reg_targets.append(reg_targets_per_im)
                continue

            if self.cfg.MODEL.ATSS.POSITIVE_TYPE == 'SSC':
                object_sizes_of_interest = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, INF]]
                area_per_im = targets_per_im.area()
                expanded_object_sizes_of_interest = []
                points = []
                for l, anchors_per_level in enumerate(anchors[im_i]):
                    anchors_per_level = anchors_per_level.bbox
                    anchors_cx_per_level = (anchors_per_level[:, 2] + anchors_per_level[:, 0]) / 2.0
                    anchors_cy_per_level = (anchors_per_level[:, 3] + anchors_per_level[:, 1]) / 2.0
                    points_per_level = torch.stack((anchors_cx_per_level, anchors_cy_per_level), dim=1)
                    points.append(points_per_level)
                    object_sizes_of_interest_per_level = \
                        points_per_level.new_tensor(object_sizes_of_interest[l])
                    expanded_object_sizes_of_interest.append(
                        object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
                    )
                expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
                points = torch.cat(points, dim=0)

                xs, ys = points[:, 0], points[:, 1]
                l = xs[:, None] - bboxes_per_im[:, 0][None]
                t = ys[:, None] - bboxes_per_im[:, 1][None]
                r = bboxes_per_im[:, 2][None] - xs[:, None]
                b = bboxes_per_im[:, 3][None] - ys[:, None]
                reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0.01

                max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
                is_cared_in_the_level = \
                    (max_reg_targets_per_im >= expanded_object_sizes_of_interest[:, [0]]) & \
                    (max_reg_targets_per_im <= expanded_object_sizes_of_interest[:, [1]])

                locations_to_gt_area = area_per_im[None].repeat(len(points), 1)
                locations_to_gt_area[is_in_boxes == 0] = INF
                locations_to_gt_area[is_cared_in_the_level == 0] = INF
                locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

                cls_labels_per_im = labels_per_im[locations_to_gt_inds]
                cls_labels_per_im[locations_to_min_area == INF] = 0
                matched_gts = bboxes_per_im[locations_to_gt_inds]
            elif self.cfg.MODEL.ATSS.POSITIVE_TYPE == 'ATSS':
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

            elif self.cfg.MODEL.ATSS.POSITIVE_TYPE == 'TOPK':
                gt_cx = (bboxes_per_im[:, 2] + bboxes_per_im[:, 0]) / 2.0
                gt_cy = (bboxes_per_im[:, 3] + bboxes_per_im[:, 1]) / 2.0
                gt_points = torch.stack((gt_cx, gt_cy), dim=1)

                anchors_cx_per_im = (anchors_per_im.bbox[:, 2] + anchors_per_im.bbox[:, 0]) / 2.0
                anchors_cy_per_im = (anchors_per_im.bbox[:, 3] + anchors_per_im.bbox[:, 1]) / 2.0
                anchor_points = torch.stack((anchors_cx_per_im, anchors_cy_per_im), dim=1)

                distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()
                distances = distances / distances.max() / 1000
                ious = boxlist_iou(anchors_per_im, targets_per_im)

                is_pos = ious * False
                for ng in range(num_gt):
                    _, topk_idxs = (ious[:, ng] - distances[:, ng]).topk(self.cfg.MODEL.ATSS.TOPK, dim=0)
                    l = anchors_cx_per_im[topk_idxs] - bboxes_per_im[ng, 0]
                    t = anchors_cy_per_im[topk_idxs] - bboxes_per_im[ng, 1]
                    r = bboxes_per_im[ng, 2] - anchors_cx_per_im[topk_idxs]
                    b = bboxes_per_im[ng, 3] - anchors_cy_per_im[topk_idxs]
                    is_in_gt = torch.stack([l, t, r, b], dim=1).min(dim=1)[0] > 0.01
                    is_pos[topk_idxs[is_in_gt == 1], ng] = True

                ious[is_pos == 0] = -INF
                anchors_to_gt_values, anchors_to_gt_indexs = ious.max(dim=1)

                cls_labels_per_im = labels_per_im[anchors_to_gt_indexs]
                cls_labels_per_im[anchors_to_gt_values == -INF] = 0
                matched_gts = bboxes_per_im[anchors_to_gt_indexs]
            elif self.cfg.MODEL.ATSS.POSITIVE_TYPE == 'IoU':
                match_quality_matrix = boxlist_iou(targets_per_im, anchors_per_im)
                matched_idxs = self.matcher(match_quality_matrix)
                targets_per_im = targets_per_im.copy_with_fields(['labels'])
                matched_targets = targets_per_im[matched_idxs.clamp(min=0)]

                cls_labels_per_im = matched_targets.get_field("labels")
                cls_labels_per_im = cls_labels_per_im.to(dtype=torch.float32)

                # Background (negative examples)
                bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
                cls_labels_per_im[bg_indices] = 0

                # discard indices that are between thresholds
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                cls_labels_per_im[inds_to_discard] = -1

                matched_gts = matched_targets.bbox

                # Limiting positive samples’ center to object
                # in order to filter out poor positives and use the centerness branch
                pos_idxs = torch.nonzero(cls_labels_per_im > 0).squeeze(1)
                pos_anchors_cx = (anchors_per_im.bbox[pos_idxs, 2] + anchors_per_im.bbox[pos_idxs, 0]) / 2.0
                pos_anchors_cy = (anchors_per_im.bbox[pos_idxs, 3] + anchors_per_im.bbox[pos_idxs, 1]) / 2.0
                l = pos_anchors_cx - matched_gts[pos_idxs, 0]
                t = pos_anchors_cy - matched_gts[pos_idxs, 1]
                r = matched_gts[pos_idxs, 2] - pos_anchors_cx
                b = matched_gts[pos_idxs, 3] - pos_anchors_cy
                is_in_gts = torch.stack([l, t, r, b], dim=1).min(dim=1)[0] > 0.01
                cls_labels_per_im[pos_idxs[is_in_gts == 0]] = -1
            else:
                raise NotImplementedError

            reg_targets_per_im = self.box_coder.encode(matched_gts, anchors_per_im.bbox)
            cls_labels.append(cls_labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return cls_labels, reg_targets

    def compute_centerness_targets(self, reg_targets, anchors):
        gts = self.box_coder.decode(reg_targets, anchors)
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l = anchors_cx - gts[:, 0]
        t = anchors_cy - gts[:, 1]
        r = gts[:, 2] - anchors_cx
        b = gts[:, 3] - anchors_cy
        left_right = torch.stack([l, r], dim=1)
        top_bottom = torch.stack([t, b], dim=1)
        centerness = torch.sqrt((left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        assert not torch.isnan(centerness).any()
        return centerness

    def __call__(self, box_cls, box_regression, centerness, targets, anchors):
        labels, reg_targets = self.prepare_targets(targets, anchors)

        N = len(labels)
        box_cls_flatten, box_regression_flatten = concat_box_prediction_layers(box_cls, box_regression)
        centerness_flatten = [ct.permute(0, 2, 3, 1).reshape(N, -1, 1) for ct in centerness]
        centerness_flatten = torch.cat(centerness_flatten, dim=1).reshape(-1)

        labels_flatten = torch.cat(labels, dim=0)
        reg_targets_flatten = torch.cat(reg_targets, dim=0)
        anchors_flatten = torch.cat([cat_boxlist(anchors_per_image).bbox for anchors_per_image in anchors], dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        num_gpus = get_num_gpus()
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        cls_loss = self.cls_loss_func(box_cls_flatten, labels_flatten.int()) / num_pos_avg_per_gpu

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        anchors_flatten = anchors_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]
        centerness_targets = self.compute_centerness_targets(reg_targets_flatten, anchors_flatten)
        sum_centerness_targets_avg_per_gpu = reduce_sum(centerness_targets.sum()).item() / float(num_gpus)

        if pos_inds.numel() > 0:
            reg_loss = self.GIoULoss(box_regression_flatten, reg_targets_flatten, anchors_flatten,
                                     weight=centerness_targets) / sum_centerness_targets_avg_per_gpu
            centerness_loss = self.centerness_loss_func(centerness_flatten, centerness_targets) / num_pos_avg_per_gpu
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss * self.cfg.MODEL.ATSS.REG_LOSS_WEIGHT, centerness_loss


def make_atss_loss_evaluator(cfg, box_coder):
    loss_evaluator = ATSSLossComputation(cfg, box_coder)
    return loss_evaluator
