import torch
from atss_core.modeling.rpn.atss.atss import ATSSHead
from atss_core.modeling.rpn.atss.atss import BoxCoder
from atss_core.modeling.rpn.atss.inference import make_atss_postprocessor
from atss_core.modeling.rpn.atss.loss import make_atss_loss_evaluator
from atss_core.modeling.rpn.anchor_generator import make_anchor_generator_atss

from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY

__all__ = ["MY_ATSSModule"]

@PROPOSAL_GENERATOR_REGISTRY.register()
class MY_ATSSModule(torch.nn.Module):

    def __init__(self, cfg, in_channels):
        super(MY_ATSSModule, self).__init__()
        self.in_features=("p3", "p4", "p5", "p6", "p7")
        self.cfg = cfg
        self.head = ATSSHead(cfg, in_channels)
        box_coder = BoxCoder(cfg)
        self.loss_evaluator = make_atss_loss_evaluator(cfg, box_coder)
        self.box_selector_test = make_atss_postprocessor(cfg, box_coder)
        # 先在这里生成了 anchor_generator 为了之后调用
        self.anchor_generator = make_anchor_generator_atss(cfg)

    def forward(self, images, features_dict, targets=None):
        # 类别边框还有中心点的结果 feature就是backbone输出的结果
        features = [features_dict[f] for f in self.in_features]
        box_cls, box_regression, centerness = self.head(features)
        # 这里直接调用了 AnchorGenerator 这个类，生成了一系列的anchor 具体过程不重要
        # anchors 是一个boxlist，分为5个level 每个上的每一个格子都有3个候选框
        # 5个level 每个level的box的shape是[N,4] N 不一定是多少
        anchors = self.anchor_generator(images, features)
 
        if self.training:
            return self._forward_train(box_cls, box_regression, centerness, targets, anchors)
        else:
            return self._forward_test(box_cls, box_regression, centerness, anchors)

    def _forward_train(self, box_cls, box_regression, centerness, targets, anchors):
        # self.loss_evaluator 就是 ATSSLossComputation 里面有真值与锚框匹配的过程
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
            box_cls, box_regression, centerness, targets, anchors
        )
        losses = {
            "loss_cls": loss_box_cls*0.8,
            "loss_reg": loss_box_reg*0.8,
            "loss_centerness": loss_centerness*0.8
        }
        proposals,_=self._forward_test(box_cls, box_regression, centerness,anchors)
        
        for p in range(len(proposals)):
            proposals[p].objectness_logits = proposals[p].get('scores')
            proposals[p].remove('scores')

        return proposals, losses

    def _forward_test(self, box_cls, box_regression, centerness, anchors):
        boxes = self.box_selector_test(box_cls, box_regression, centerness, anchors)
        return boxes, {}


# def build_atss(cfg, in_channels):
#     return ATSSModule(cfg, in_channels)
