import math
from typing import List
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, get_norm
from detectron2.config import configurable
from ..layers.deform_conv import DFConv2d

__all__ = ["CenterNetHead"]

class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class CenterNetHead(nn.Module):
    @configurable
    def __init__(self, 
        # input_shape: List[ShapeSpec],
        in_channels,
        num_levels,
        *,
        num_classes=80,
        with_agn_hm=False,
        only_proposal=False,
        norm='GN',
        num_cls_convs=4,
        num_box_convs=4,
        num_share_convs=0,
        use_deformable=False,
        prior_prob=0.01):
        super().__init__()
        self.num_classes = num_classes
        self.with_agn_hm = with_agn_hm
        self.only_proposal = only_proposal 
        self.out_kernel = 3

        # centernet2里
        # self.only_proposal默认值是TRUE
        # use_deformable默认是false
        head_configs = {
            "cls": (num_cls_convs if not self.only_proposal else 0, \
                use_deformable),
            "bbox": (num_box_convs, use_deformable), # num_box_convs 默认是4
            "share": (num_share_convs, use_deformable)} # num_share_convs 默认是0

        # in_channels = [s.channels for s in input_shape]
        # assert len(set(in_channels)) == 1, \
        #     "Each level must have the same channel!"
        # in_channels = in_channels[0]
        channels = {
            'cls': in_channels,
            'bbox': in_channels,
            'share': in_channels,
        }
        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            channel = channels[head]
            for i in range(num_convs):
                if use_deformable and i == num_convs - 1:
                    conv_func = DFConv2d
                else:
                    conv_func = nn.Conv2d
                tower.append(conv_func(
                    # 卷积核3 步长1 填充1 则大小不变
                        in_channels if i == 0 else channel,
                        channel, 
                        kernel_size=3, stride=1, 
                        padding=1, bias=True
                ))
                if norm == 'GN' and channel % 32 != 0:
                    tower.append(nn.GroupNorm(25, channel))
                elif norm != '':
                    tower.append(get_norm(norm, channel))
                tower.append(nn.ReLU())
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=self.out_kernel,
            stride=1, padding=self.out_kernel // 2
        )

        self.scales = nn.ModuleList(
            [Scale(init_value=1.0) for _ in range(num_levels)])

        for modules in [
            self.cls_tower, self.bbox_tower,
            self.share_tower,
            self.bbox_pred,
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
        
        torch.nn.init.constant_(self.bbox_pred.bias, 8.)
        prior_prob = prior_prob
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        if self.with_agn_hm:
            self.agn_hm = nn.Conv2d(
                in_channels, 1, kernel_size=self.out_kernel,
                stride=1, padding=self.out_kernel // 2 # //表示整除 例如7//2=3
            )
            torch.nn.init.constant_(self.agn_hm.bias, bias_value)
            torch.nn.init.normal_(self.agn_hm.weight, std=0.01)

        if not self.only_proposal:
            cls_kernel_size = self.out_kernel
            self.cls_logits = nn.Conv2d(
                in_channels, self.num_classes,
                kernel_size=cls_kernel_size, 
                stride=1,
                padding=cls_kernel_size // 2,
            )

            torch.nn.init.constant_(self.cls_logits.bias, bias_value)
            torch.nn.init.normal_(self.cls_logits.weight, std=0.01)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {
            # 'input_shape': input_shape,
            'in_channels': [s.channels for s in input_shape][0],
            'num_levels': len(input_shape),
            'num_classes': cfg.MODEL.CENTERNET.NUM_CLASSES,
            'with_agn_hm': cfg.MODEL.CENTERNET.WITH_AGN_HM,
            'only_proposal': cfg.MODEL.CENTERNET.ONLY_PROPOSAL,
            'norm': cfg.MODEL.CENTERNET.NORM,
            'num_cls_convs': cfg.MODEL.CENTERNET.NUM_CLS_CONVS,
            'num_box_convs': cfg.MODEL.CENTERNET.NUM_BOX_CONVS,
            'num_share_convs': cfg.MODEL.CENTERNET.NUM_SHARE_CONVS,
            'use_deformable': cfg.MODEL.CENTERNET.USE_DEFORMABLE,
            'prior_prob': cfg.MODEL.CENTERNET.PRIOR_PROB,
        }
        return ret

    def forward(self, x):
        # x就是输入进来的特征图
        # x:list level个[batch,256(通道数),h,w]
        clss = []
        bbox_reg = []
        agn_hms = []
        # share_tower cls_tower bbox_tower都是在init里生成的
        # 都是conv(3*3 不改变大小与通道)+GN+ReLU
        for l, feature in enumerate(x):
            feature = self.share_tower(feature)
            cls_tower = self.cls_tower(feature)
            bbox_tower = self.bbox_tower(feature)
            # clss是none
            if not self.only_proposal:
                clss.append(self.cls_logits(cls_tower))
            else:
                clss.append(None)

            # 默认为TRUE
            # self.agn_hms是卷积层 conv(3*3,步长1 填充1，通道变为1)
            # 输出的agn_hms是特征图
            if self.with_agn_hm: 
                agn_hms.append(self.agn_hm(bbox_tower))
            else:
                agn_hms.append(None)
            # bbox_pred是卷积层 conv(3*3,步长1 填充1，通道变为4)
            reg = self.bbox_pred(bbox_tower)
            reg = self.scales[l](reg)
            bbox_reg.append(F.relu(reg))
        # 经过了上面的操作，输出的 clss全是none因为这里的head只用于生成proposal，不涉及类别
        # bbox_reg是一个有5个维度的list，每个维度代表一个level，每个level中是一个[8,4,h,w]的数组
        # 8代表有8张图，即batchsize，4代表了box的四个维度，h，w是特征图中像素的坐标
        # agn_hms跟bbox_reg的结构是一样的，5个维度的list，每个维度代表一个level，每个level中是一个[8,1,h,w]的数组
        # 后面的两个同样是点的坐标，heatmap每个点只有一个值
        return clss, bbox_reg, agn_hms