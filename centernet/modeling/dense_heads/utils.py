import cv2
import torch
from torch import nn
from detectron2.utils.comm import get_world_size
from detectron2.structures import pairwise_iou, Boxes
# from .data import CenterNetCrop
import torch.nn.functional as F
import numpy as np
from detectron2.structures import Boxes, ImageList, Instances

__all__ = ['reduce_sum', '_transpose']

INF = 1000000000

def _transpose(training_targets, num_loc_list):
    '''
    num_loc_list 是 每一个level的grid点数 格式是[L1,L2,L3,L4,L5]
    This function is used to transpose image first training targets to 
        level first ones
    :return: level first training targets
    '''
    # 经历了下面这一步的training_targets，还是8维度的list
    # 每一个维度里又分了5个list，是把点按照level分成了5个等级
    for im_i in range(len(training_targets)):
        training_targets[im_i] = torch.split(
            training_targets[im_i], num_loc_list, dim=0)

    targets_level_first = []
    # 把每一个level上的target连接起来 
    for targets_per_level in zip(*training_targets):
        targets_level_first.append(
            torch.cat(targets_per_level, dim=0))
    return targets_level_first


def reduce_sum(tensor):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    tensor = tensor.clone()
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor