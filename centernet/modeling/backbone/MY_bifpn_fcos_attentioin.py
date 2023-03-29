# This file is modified from https://github.com/aim-uofa/AdelaiDet/blob/master/adet/modeling/backbone/bifpn.py
# The original file is under 2-clause BSD License for academic use, and *non-commercial use*.
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import Conv2d, ShapeSpec, get_norm

from detectron2.modeling.backbone import Backbone, build_resnet_backbone
from detectron2.modeling import BACKBONE_REGISTRY
from .dlafpn import dla34

__all__ = []


def swish(x):
    return x * x.sigmoid()


def split_name(name):
    for i, c in enumerate(name):
        if not c.isalpha():
            return name[:i], int(name[i:])
    raise ValueError()

## 定义通道域类
class ChannelAttention(nn.Module):
    def __init__(self, in_planes=160, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        # self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out =self.shared_MLP(self.avg_pool(x))# self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out =self.shared_MLP(self.max_pool(x))# self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

## 定义空间域类
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

## CBAM将通道域与空间域串联起来形成混合域
class CBAM(nn.Module):
    def __init__(self, planes=160):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, features):
        out_feature=[]
        for idx , x in enumerate(features):
            x = self.ca(x) * x  #  执行通道注意力机制，并为通道赋予权重
            x = self.sa(x) * x  #  执行空间注意力机制，并为通道赋予权重
            out_feature.append(x)
        return out_feature

class FeatureMapResampler(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm=""):
        super(FeatureMapResampler, self).__init__()
        if in_channels != out_channels:
            self.reduction = Conv2d(
                in_channels, out_channels, kernel_size=1,
                bias=(norm == ""),
                norm=get_norm(norm, out_channels),
                activation=None
            )
        else:
            self.reduction = None

        assert stride <= 2
        self.stride = stride

    def forward(self, x):
        if self.reduction is not None:
            x = self.reduction(x)

        if self.stride == 2:
            x = F.max_pool2d(
                x, kernel_size=self.stride + 1,
                stride=self.stride, padding=1
            )
        elif self.stride == 1:
            pass
        else:
            raise NotImplementedError()
        return x


class BackboneWithTopLevels(Backbone):
    def __init__(self, backbone, out_channels, num_top_levels, norm=""):
        super(BackboneWithTopLevels, self).__init__()
        self.backbone = backbone
        backbone_output_shape = backbone.output_shape()

        self._out_feature_channels = {name: shape.channels for name, shape in backbone_output_shape.items()}
        self._out_feature_strides = {name: shape.stride for name, shape in backbone_output_shape.items()}
        self._out_features = list(self._out_feature_strides.keys())

        last_feature_name = max(self._out_feature_strides.keys(), key=lambda x: split_name(x)[1])
        self.last_feature_name = last_feature_name
        self.num_top_levels = num_top_levels

        last_channels = self._out_feature_channels[last_feature_name]
        last_stride = self._out_feature_strides[last_feature_name]

        prefix, suffix = split_name(last_feature_name)
        prev_channels = last_channels
        for i in range(num_top_levels):
            name = prefix + str(suffix + i + 1)
            self.add_module(name, FeatureMapResampler(
                prev_channels, out_channels, 2, norm
            ))
            prev_channels = out_channels

            self._out_feature_channels[name] = out_channels
            self._out_feature_strides[name] = last_stride * 2 ** (i + 1)
            self._out_features.append(name)

    def forward(self, x):
        outputs = self.backbone(x)
        last_features = outputs[self.last_feature_name]
        prefix, suffix = split_name(self.last_feature_name)

        x = last_features
        for i in range(self.num_top_levels):
            name = prefix + str(suffix + i + 1)
            x = self.__getattr__(name)(x)
            outputs[name] = x

        return outputs


class SingleBiFPN(Backbone):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
        self, in_channels_list, out_channels, norm=""
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
        """
        super(SingleBiFPN, self).__init__()

        self.out_channels = out_channels
        # build 5-levels bifpn
        if len(in_channels_list) == 5:
            self.nodes = [
                {'feat_level': 3, 'inputs_offsets': [3, 4]},
                {'feat_level': 2, 'inputs_offsets': [2, 5]},
                {'feat_level': 1, 'inputs_offsets': [1, 6]},
                {'feat_level': 0, 'inputs_offsets': [0, 7]},
                {'feat_level': 1, 'inputs_offsets': [1, 7, 8]},
                {'feat_level': 2, 'inputs_offsets': [2, 6, 9]},
                {'feat_level': 3, 'inputs_offsets': [3, 5, 10]},
                {'feat_level': 4, 'inputs_offsets': [4, 11]},
            ]
        elif len(in_channels_list) == 3:
            self.nodes = [
                {'feat_level': 1, 'inputs_offsets': [1, 2]},
                {'feat_level': 0, 'inputs_offsets': [0, 3]},
                {'feat_level': 1, 'inputs_offsets': [1, 3, 4]},
                {'feat_level': 2, 'inputs_offsets': [2, 5]},
            ]
        else:
            raise NotImplementedError

        node_info = [_ for _ in in_channels_list]

        num_output_connections = [0 for _ in in_channels_list]
        for fnode in self.nodes:
            feat_level = fnode["feat_level"]
            # inputs_offsets 表示某个节点输入的连线数目
            inputs_offsets = fnode["inputs_offsets"]
            inputs_offsets_str = "_".join(map(str, inputs_offsets))
            for input_offset in inputs_offsets:
                num_output_connections[input_offset] += 1

                in_channels = node_info[input_offset]
                if in_channels != out_channels:
                    lateral_conv = Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        norm=get_norm(norm, out_channels)
                    )
                    self.add_module(
                        "lateral_{}_f{}".format(input_offset, feat_level), lateral_conv
                    )
            node_info.append(out_channels)
            # num_output_connections 表示某个节点向外的连线数目 节点的编号方法看笔记
            num_output_connections.append(0)

            # generate attention weights
            name = "weights_f{}_{}".format(feat_level, inputs_offsets_str)
            self.__setattr__(name, nn.Parameter(
                    torch.ones(len(inputs_offsets), dtype=torch.float32),
                    requires_grad=True
                ))

            # generate convolutions after combination
            name = "outputs_f{}_{}".format(feat_level, inputs_offsets_str)
            self.add_module(name, Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                norm=get_norm(norm, out_channels),
                bias=(norm == "")
            ))

    def forward(self, feats):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "p5") to
                feature map tensor for each feature level in high to low resolution order.
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["n2", "n3", ..., "n6"].
        """
        feats = [_ for _ in feats]
        num_levels = len(feats)
        num_output_connections = [0 for _ in feats]
        for fnode in self.nodes:
            feat_level = fnode["feat_level"]
            inputs_offsets = fnode["inputs_offsets"]
            inputs_offsets_str = "_".join(map(str, inputs_offsets))
            input_nodes = []
            _, _, target_h, target_w = feats[feat_level].size()
            for input_offset in inputs_offsets:
                num_output_connections[input_offset] += 1
                input_node = feats[input_offset]

                # reduction
                if input_node.size(1) != self.out_channels:
                    name = "lateral_{}_f{}".format(input_offset, feat_level)
                    input_node = self.__getattr__(name)(input_node)

                # maybe downsample 如果input_node的size和该层特征的size不一样 就调整成一样的
                _, _, h, w = input_node.size()
                if h > target_h and w > target_w:
                    height_stride_size = int((h - 1) // target_h + 1)
                    width_stride_size = int((w - 1) // target_w + 1)
                    assert height_stride_size == width_stride_size == 2
                    input_node = F.max_pool2d(
                        input_node, kernel_size=(height_stride_size + 1, width_stride_size + 1),
                        stride=(height_stride_size, width_stride_size), padding=1
                    )
                elif h <= target_h and w <= target_w:
                    if h < target_h or w < target_w:
                        input_node = F.interpolate(
                            input_node,
                            size=(target_h, target_w),
                            mode="nearest"
                        )
                else:
                    raise NotImplementedError()
                input_nodes.append(input_node)

            # attention 给输入进来的node添加权重 然后进行加和
            name = "weights_f{}_{}".format(feat_level, inputs_offsets_str)
            weights = F.relu(self.__getattr__(name))
            norm_weights = weights / (weights.sum() + 0.0001)

            new_node = torch.stack(input_nodes, dim=-1)
            new_node = (norm_weights * new_node).sum(dim=-1)
            new_node = swish(new_node)

            name = "outputs_f{}_{}".format(feat_level, inputs_offsets_str)
            # 卷积一下 加入feats中
            feats.append(self.__getattr__(name)(new_node))

            num_output_connections.append(0)

        output_feats = []
        for idx in range(num_levels):
            for i, fnode in enumerate(reversed(self.nodes)):
                if fnode['feat_level'] == idx:
                    output_feats.append(feats[-1 - i])
                    break
            else:
                raise ValueError()
        return output_feats

class SingleBiFPN_attention(Backbone):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
        self, in_channels_list, out_channels, norm=""
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
        """
        super(SingleBiFPN_attention, self).__init__()

        self.out_channels = out_channels
        # build 5-levels bifpn
        if len(in_channels_list) == 5:
            self.nodes = [
                {'feat_level': 3, 'inputs_offsets': [3, 4]},
                {'feat_level': 2, 'inputs_offsets': [2, 5]},
                {'feat_level': 1, 'inputs_offsets': [1, 6]},
                {'feat_level': 0, 'inputs_offsets': [0, 7]},
                {'feat_level': 1, 'inputs_offsets': [1, 7, 8]},
                {'feat_level': 2, 'inputs_offsets': [2, 6, 9]},
                {'feat_level': 3, 'inputs_offsets': [3, 5, 10]},
                {'feat_level': 4, 'inputs_offsets': [4, 11]},
            ]
        elif len(in_channels_list) == 3:
            self.nodes = [
                {'feat_level': 1, 'inputs_offsets': [1, 2]},
                {'feat_level': 0, 'inputs_offsets': [0, 3]},
                {'feat_level': 1, 'inputs_offsets': [1, 3, 4]},
                {'feat_level': 2, 'inputs_offsets': [2, 5]},
            ]
        else:
            raise NotImplementedError

        node_info = [_ for _ in in_channels_list]

        num_output_connections = [0 for _ in in_channels_list]
        for fnode in self.nodes:
            feat_level = fnode["feat_level"]
            # inputs_offsets 表示某个节点输入的连线数目
            inputs_offsets = fnode["inputs_offsets"]
            inputs_offsets_str = "_".join(map(str, inputs_offsets))
            for input_offset in inputs_offsets:
                num_output_connections[input_offset] += 1

                in_channels = node_info[input_offset]
                if in_channels != out_channels:
                    lateral_conv = Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        norm=get_norm(norm, out_channels)
                    )
                    self.add_module(
                        "lateral_{}_f{}".format(input_offset, feat_level), lateral_conv
                    )
            node_info.append(out_channels)
            # num_output_connections 表示某个节点向外的连线数目 节点的编号方法看笔记
            num_output_connections.append(0)

            # generate attention weights
            name = "weights_f{}_{}".format(feat_level, inputs_offsets_str)
            self.__setattr__(name, nn.Parameter(
                    torch.ones(len(inputs_offsets), dtype=torch.float32),
                    requires_grad=True
                ))

            # generate convolutions after combination
            name = "outputs_f{}_{}".format(feat_level, inputs_offsets_str)
            self.add_module(name, Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                norm=get_norm(norm, out_channels),
                bias=(norm == "")
            ))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        in_planes=160
        ratio=16
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, feats):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "p5") to
                feature map tensor for each feature level in high to low resolution order.
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["n2", "n3", ..., "n6"].
        """
        feats = [_ for _ in feats]
        num_levels = len(feats)
        num_output_connections = [0 for _ in feats]
        for fnode in self.nodes:
            feat_level = fnode["feat_level"]
            inputs_offsets = fnode["inputs_offsets"]
            inputs_offsets_str = "_".join(map(str, inputs_offsets))
            input_nodes = []
            _, _, target_h, target_w = feats[feat_level].size()
            for input_offset in inputs_offsets:
                num_output_connections[input_offset] += 1
                input_node = feats[input_offset]

                # reduction
                if input_node.size(1) != self.out_channels:
                    name = "lateral_{}_f{}".format(input_offset, feat_level)
                    input_node = self.__getattr__(name)(input_node)

                # maybe downsample 如果input_node的size和该层特征的size不一样 就调整成一样的
                _, _, h, w = input_node.size()
                if h > target_h and w > target_w:
                    height_stride_size = int((h - 1) // target_h + 1)
                    width_stride_size = int((w - 1) // target_w + 1)
                    assert height_stride_size == width_stride_size == 2
                    input_node = F.max_pool2d(
                        input_node, kernel_size=(height_stride_size + 1, width_stride_size + 1),
                        stride=(height_stride_size, width_stride_size), padding=1
                    )
                elif h <= target_h and w <= target_w:
                    if h < target_h or w < target_w:
                        input_node = F.interpolate(
                            input_node,
                            size=(target_h, target_w),
                            mode="nearest"
                        )
                else:
                    raise NotImplementedError()
                input_nodes.append(input_node)

            # attention 给输入进来的node添加权重 然后进行加和
            name = "weights_f{}_{}".format(feat_level, inputs_offsets_str)
            weights = F.relu(self.__getattr__(name))
            norm_weights = weights / (weights.sum() + 0.0001)

            new_node = torch.stack(input_nodes, dim=-1)
            new_node = (norm_weights * new_node).sum(dim=-1)
            new_node = swish(new_node)

            name = "outputs_f{}_{}".format(feat_level, inputs_offsets_str)
            # 卷积一下 加入feats中
            feats.append(self.__getattr__(name)(new_node))

            num_output_connections.append(0)

        output_feats = []
        for idx in range(num_levels):
            for i, fnode in enumerate(reversed(self.nodes)):
                if fnode['feat_level'] == idx:
                    avg_out =self.shared_MLP(self.avg_pool(feats[-1 - i]))
                    max_out =self.shared_MLP(self.max_pool(feats[-1 - i]))
                    out = avg_out + max_out
                    out = out / (torch.max(out) + 1e-8)
                    output_feats.append(out* feats[-1 - i])
                    break
            else:
                raise ValueError()
        return output_feats

class MY_CBAM_BiFPN(Backbone):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
        self, bottom_up, in_features, out_channels, num_top_levels, num_repeats, norm=""
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            num_top_levels (int): the number of the top levels (p6 or p7).
            num_repeats (int): the number of repeats of MY_CBAM_BiFPN.
            norm (str): the normalization to use.
        """
        super(MY_CBAM_BiFPN, self).__init__()
        assert isinstance(bottom_up, Backbone)
        # add extra feature levels (i.e., 6 and 7)
        self.bottom_up = BackboneWithTopLevels(
            bottom_up, out_channels,
            num_top_levels, norm
        )
        bottom_up_output_shapes = self.bottom_up.output_shape()

        in_features = sorted(in_features, key=lambda x: split_name(x)[1])
        self._size_divisibility = 128 #bottom_up_output_shapes[in_features[-1]].stride
        self.out_channels = out_channels
        self.min_level = split_name(in_features[0])[1]

        # add the names for top blocks
        prefix, last_suffix = split_name(in_features[-1])
        for i in range(num_top_levels):
            in_features.append(prefix + str(last_suffix + i + 1))
        self.in_features = in_features

        # generate output features
        self._out_features = ["p{}".format(split_name(name)[1]) for name in in_features]
        self._out_feature_strides = {
            out_name: bottom_up_output_shapes[in_name].stride
            for out_name, in_name in zip(self._out_features, in_features)
        }
        self._out_feature_channels = {k: out_channels for k in self._out_features}

        # build bifpn
        self.repeated_bifpn = nn.ModuleList()
        for i in range(num_repeats): # BiFPN要repeat多少次
            # 最开头的BiFPN进入的特征图通道数是dla输出的特征图通道数
            # 也就是[128,256,512]
            if i == 0:
                in_channels_list = [
                    bottom_up_output_shapes[name].channels for name in in_features
                ]
            # 第2个 第3个BiFPN block 特征图的通道数就确定了
            else:
                in_channels_list = [
                    self._out_feature_channels[name] for name in self._out_features
                ]
                # in_channels_list=[128,256,512]
            self.repeated_bifpn.append(SingleBiFPN(
                in_channels_list, out_channels, norm))
            self.repeated_bifpn.append(CBAM())

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "p5") to
                feature map tensor for each feature level in high to low resolution order.
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["n2", "n3", ..., "n6"].
        """
        bottom_up_features = self.bottom_up(x)
        feats = [bottom_up_features[f] for f in self.in_features]

        for bifpn in self.repeated_bifpn:
             feats = bifpn(feats)

        return dict(zip(self._out_features, feats))

class MY_PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        super(MY_PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        # feats: [1,256,129,257]
        n, c, _, _ = feats.size()
        # 分别将特征图池化为1*1 3*3 6*6 8*8，然后铺平（看笔记）
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        # 首尾相连一下
        center = torch.cat(priors, -1)
        return center

class MY_PAN_BiFPN_P35(Backbone):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
        self, bottom_up, in_features, out_channels, num_top_levels, num_repeats, norm="",cbam=False):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            num_top_levels (int): the number of the top levels (p6 or p7).
            num_repeats (int): the number of repeats of MY_CBAM_BiFPN.
            norm (str): the normalization to use.
        """
        super(MY_PAN_BiFPN_P35, self).__init__()
        assert isinstance(bottom_up, Backbone)
        # add extra feature levels (i.e., 6 and 7)
        self.bottom_up = BackboneWithTopLevels(
            bottom_up, out_channels,
            num_top_levels, norm
        )
        bottom_up_output_shapes = self.bottom_up.output_shape()

        in_features = sorted(in_features, key=lambda x: split_name(x)[1])
        self._size_divisibility = 128 #bottom_up_output_shapes[in_features[-1]].stride
        self.out_channels = out_channels
        self.min_level = split_name(in_features[0])[1]

        # add the names for top blocks
        prefix, last_suffix = split_name(in_features[-1])
        for i in range(num_top_levels):
            in_features.append(prefix + str(last_suffix + i + 1))
        self.in_features = in_features

        # generate output features
        self._out_features = ["p{}".format(split_name(name)[1]) for name in in_features]
        self._out_feature_strides = {
            out_name: bottom_up_output_shapes[in_name].stride
            for out_name, in_name in zip(self._out_features, in_features)
        }
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self.psp=MY_PSPModule()
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=160,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU()
        )
        # 本项目中 nn.Conv2d(in_channels=2048,out_channels=256)
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=160,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU()
        )
        # 本项目中 nn.Conv2d(in_channels=1024,out_channels=256)
        self.f_value = nn.Conv2d(in_channels=160, out_channels=160,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=160, out_channels=160,
                           kernel_size=1, stride=1, padding=0)
        # build bifpn
        self.upsample=nn.ConvTranspose2d(in_channels=160,out_channels=160,kernel_size=3,padding=1,stride=2)
        self.repeated_bifpn = nn.ModuleList()
        for i in range(num_repeats): # BiFPN要repeat多少次
            # 最开头的BiFPN进入的特征图通道数是dla输出的特征图通道数
            # 也就是[128,256,512]
            if i == 0:
                in_channels_list = [
                    bottom_up_output_shapes[name].channels for name in in_features
                ]
            # 第2个 第3个BiFPN block 特征图的通道数就确定了
            else:
                in_channels_list = [
                    self._out_feature_channels[name] for name in self._out_features
                ]
                # in_channels_list=[128,256,512]
            if cbam == True and i % 2 == 0:
                self.repeated_bifpn.append(SingleBiFPN_attention(
                    in_channels_list, out_channels, norm))
            else:
                self.repeated_bifpn.append(SingleBiFPN(
                    in_channels_list, out_channels, norm))


    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "p5") to
                feature map tensor for each feature level in high to low resolution order.
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["n2", "n3", ..., "n6"].
        """
        bottom_up_features = self.bottom_up(x)
        feats = [bottom_up_features[f] for f in self.in_features]
        feats_value=[]
        feats_key=[]
        for i, bifpn in enumerate(self.repeated_bifpn):
            feats = bifpn(feats)
            if i <= 3 : pass 
            else:
                for j,feat in enumerate(feats):
                    if j==0:
                        feat_upsample=self.upsample(feat)
                        feat_value=self.psp(self.f_value(feat_upsample)).permute(0, 2, 1)
                        feat_key=self.psp(self.f_key(feat_upsample))
                        feats_value.append(feat_value)
                        feats_key.append(feat_key)
                    feat_value=self.f_value(feat)
                    feat_value=self.psp(feat_value).permute(0, 2, 1)
                    feat_key=self.psp(self.f_key(feat))
                    feats_value.append(feat_value)
                    feats_key.append(feat_key)
                for k, feat in enumerate(feats):
                    b = 3 if k != 2 else 2
                    ith_value = torch.cat([feats_value[q] for q in range(k,k+b)],dim=1)
                    ith_key = torch.cat([feats_key[q] for q in range(k,k+b)],dim=-1)
                    ith_query = feat.view(feat.shape[0],feat.shape[1],-1).permute(0,2,1)                        
                    sim_map = torch.matmul(ith_query, ith_key) # 张量（矩阵）相乘
                    sim_map = (0.00625) * sim_map
                    sim_map = F.softmax(sim_map, dim=-1)

                    context = torch.matmul(sim_map, ith_value)
                    context = context.permute(0, 2, 1).contiguous()
                    context = context.view(feat.shape[0], feat.shape[1], feat.shape[2],feat.shape[3])
                    context = self.W(context)
                    feats[k] = context
                        
        return dict(zip(self._out_features, feats))


class MY_PAN_BiFPN_P37(Backbone):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
        self, bottom_up, in_features, out_channels, num_top_levels, num_repeats, norm="",cbam=False
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            num_top_levels (int): the number of the top levels (p6 or p7).
            num_repeats (int): the number of repeats of MY_CBAM_BiFPN.
            norm (str): the normalization to use.
        """
        super(MY_PAN_BiFPN_P37, self).__init__()
        assert isinstance(bottom_up, Backbone)
        # add extra feature levels (i.e., 6 and 7)
        self.bottom_up = BackboneWithTopLevels(
            bottom_up, out_channels,
            num_top_levels, norm
        )
        bottom_up_output_shapes = self.bottom_up.output_shape()

        in_features = sorted(in_features, key=lambda x: split_name(x)[1])
        self._size_divisibility = 128 #bottom_up_output_shapes[in_features[-1]].stride
        self.out_channels = out_channels
        self.min_level = split_name(in_features[0])[1]

        # add the names for top blocks
        prefix, last_suffix = split_name(in_features[-1])
        for i in range(num_top_levels):
            in_features.append(prefix + str(last_suffix + i + 1))
        self.in_features = in_features

        # generate output features
        self._out_features = ["p{}".format(split_name(name)[1]) for name in in_features]
        self._out_feature_strides = {
            out_name: bottom_up_output_shapes[in_name].stride
            for out_name, in_name in zip(self._out_features, in_features)
        }
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self.psp=MY_PSPModule()
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=160,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU()
        )
        # 本项目中 nn.Conv2d(in_channels=2048,out_channels=256)
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=160,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU()
        )
        # 本项目中 nn.Conv2d(in_channels=1024,out_channels=256)
        self.f_value = nn.Conv2d(in_channels=160, out_channels=160,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=160, out_channels=160,
                           kernel_size=1, stride=1, padding=0)
        # build bifpn
        self.upsample=nn.ConvTranspose2d(in_channels=160,out_channels=160,kernel_size=3,padding=1,stride=2)
        self.repeated_bifpn = nn.ModuleList()
        for i in range(num_repeats): # BiFPN要repeat多少次
            # 最开头的BiFPN进入的特征图通道数是dla输出的特征图通道数
            # 也就是[128,256,512]
            if i == 0:
                in_channels_list = [
                    bottom_up_output_shapes[name].channels for name in in_features
                ]
            # 第2个 第3个BiFPN block 特征图的通道数就确定了
            else:
                in_channels_list = [
                    self._out_feature_channels[name] for name in self._out_features
                ]
                # in_channels_list=[128,256,512]
            if cbam == True and i % 2 == 0:
                self.repeated_bifpn.append(SingleBiFPN_attention(
                    in_channels_list, out_channels, norm))
            else:
                self.repeated_bifpn.append(SingleBiFPN(
                    in_channels_list, out_channels, norm))

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "p5") to
                feature map tensor for each feature level in high to low resolution order.
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["n2", "n3", ..., "n6"].
        """
        bottom_up_features = self.bottom_up(x)
        feats = [bottom_up_features[f] for f in self.in_features]
        feats_value=[]
        feats_key=[]
        for i, bifpn in enumerate(self.repeated_bifpn):
            if i <= 2 : pass 
            else:
                feats = bifpn(feats)
                for j,feat in enumerate(feats):
                    if j==0:
                        feat_upsample=self.upsample(feat)
                        feat_value=self.psp(self.f_value(feat_upsample)).permute(0, 2, 1)
                        feat_key=self.psp(self.f_key(feat_upsample))
                        feats_value.append(feat_value)
                        feats_key.append(feat_key)
                    feat_value=self.f_value(feat)
                    feat_value=self.psp(feat_value).permute(0, 2, 1)
                    feat_key=self.psp(self.f_key(feat))
                    feats_value.append(feat_value)
                    feats_key.append(feat_key)
                for k, feat in enumerate(feats):
                    if k != 4:
                        ith_value = torch.cat([feats_value[q] for q in range(k,k+3)],dim=1)
                        ith_key = torch.cat([feats_key[q] for q in range(k,k+3)],dim=-1)
                        ith_query = feat.view(feat.shape[0],feat.shape[1],-1).permute(0,2,1)                        
                        sim_map = torch.matmul(ith_query, ith_key) # 张量（矩阵）相乘
                        sim_map = (0.00625) * sim_map
                        sim_map = F.softmax(sim_map, dim=-1)

                        context = torch.matmul(sim_map, ith_value)
                        context = context.permute(0, 2, 1).contiguous()
                        context = context.view(feat.shape[0], feat.shape[1], feat.shape[2],feat.shape[3])
                        context = self.W(context)
                        feats[k] = context
                        
        return dict(zip(self._out_features, feats))


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )

@BACKBONE_REGISTRY.register()
def MY_build_fcos_resnet_bifpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.BIFPN.OUT_CHANNELS
    num_repeats = cfg.MODEL.BIFPN.NUM_BIFPN
    top_levels = 2

    backbone = MY_CBAM_BiFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        num_top_levels=top_levels,
        num_repeats=num_repeats,
        norm=cfg.MODEL.BIFPN.NORM
    )
    return backbone

@BACKBONE_REGISTRY.register()
def MY_build_p37_fcos_resnet_bifpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.BIFPN.OUT_CHANNELS
    num_repeats = cfg.MODEL.BIFPN.NUM_BIFPN
    assert cfg.MODEL.BIFPN.NUM_LEVELS == 5
    top_levels = 2

    backbone = MY_CBAM_BiFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        num_top_levels=top_levels,
        num_repeats=num_repeats,
        norm=cfg.MODEL.BIFPN.NORM
    )
    return backbone

@BACKBONE_REGISTRY.register()
def MY_build_p35_fcos_dla_bifpn_attention_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = dla34(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.BIFPN.OUT_CHANNELS
    num_repeats = cfg.MODEL.BIFPN.NUM_BIFPN
    top_levels = 0

    backbone = MY_CBAM_BiFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        num_top_levels=top_levels,
        num_repeats=num_repeats,
        norm=cfg.MODEL.BIFPN.NORM
    )
    return backbone


# 自己写的 sensors论文一系列的build_backbone
@BACKBONE_REGISTRY.register()
def MY_build_p35_fcos_dla_bifpn_pan_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = dla34(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.BIFPN.OUT_CHANNELS
    num_repeats = cfg.MODEL.BIFPN.NUM_BIFPN
    assert cfg.MODEL.BIFPN.NUM_LEVELS == 3
    top_levels = 0

    backbone = MY_PAN_BiFPN_P35(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        num_top_levels=top_levels,
        num_repeats=num_repeats,
        norm=cfg.MODEL.BIFPN.NORM,
        cbam=False
    )
    return backbone

@BACKBONE_REGISTRY.register()
def MY_build_p37_fcos_dla_bifpn_pan_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = dla34(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.BIFPN.OUT_CHANNELS
    num_repeats = cfg.MODEL.BIFPN.NUM_BIFPN
    assert cfg.MODEL.BIFPN.NUM_LEVELS == 3
    top_levels = 0

    backbone = MY_PAN_BiFPN_P37(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        num_top_levels=top_levels,
        num_repeats=num_repeats,
        norm=cfg.MODEL.BIFPN.NORM,
        cbam=False
    )
    return backbone

@BACKBONE_REGISTRY.register()
def MY_build_p35_fcos_dla_bifpn_pan_cbam_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = dla34(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.BIFPN.OUT_CHANNELS
    num_repeats = cfg.MODEL.BIFPN.NUM_BIFPN
    assert cfg.MODEL.BIFPN.NUM_LEVELS == 3
    top_levels = 0

    backbone = MY_PAN_BiFPN_P35(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        num_top_levels=top_levels,
        num_repeats=num_repeats,
        norm=cfg.MODEL.BIFPN.NORM,
        cbam=True
    )
    return backbone

@BACKBONE_REGISTRY.register()
def MY_build_p37_fcos_dla_bifpn_pan_cbam_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = dla34(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.BIFPN.OUT_CHANNELS
    num_repeats = cfg.MODEL.BIFPN.NUM_BIFPN
    assert cfg.MODEL.BIFPN.NUM_LEVELS == 3
    top_levels = 0

    backbone = MY_PAN_BiFPN_P37(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        num_top_levels=top_levels,
        num_repeats=num_repeats,
        norm=cfg.MODEL.BIFPN.NORM,
        cbam=True
    )
    return backbone

# 到这里为止 是sensors论文的build_backbone

@BACKBONE_REGISTRY.register()
def MY_build_dla_backbone(cfg, input_shape: ShapeSpec):
     bottom_up = dla34(cfg)
     backbone = bottom_up

     return backbone