import re

import torch
from torch import nn
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d

from .base import BaseModel


def _freeze_batchnorm(module: nn.Module) -> None:
    """
    Replace the BatchNorm2d layers by FrozenBatchNorm2d in place (recursively).
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(child.num_features, eps=child.eps)
            with torch.no_grad():
                frozen.weight.copy_(child.weight)  # ty:ignore[call-non-callable]
                frozen.bias.copy_(child.bias)  # ty:ignore[call-non-callable]
                frozen.running_mean.copy_(child.running_mean)  # ty:ignore[call-non-callable,invalid-argument-type]
                frozen.running_var.copy_(child.running_var)  # ty:ignore[call-non-callable,invalid-argument-type]
            setattr(module, name, frozen)
        else:
            _freeze_batchnorm(child)


class FasterRCNNModel(BaseModel):
    def __init__(
        self,
        num_classes: int = 14,
        img_size: int = 512,
        trainable_backbone_layers: int = 5,
        score_thresh: float = 0.001,
        max_detections: int = 300,
        small_anchors: bool = False,
        freeze_bn: bool = False,
        backbone_lr_mult: float = 1.0,
        layer_lr_decay: float = 1.0,
    ) -> None:
        super().__init__()
        self.backbone_lr_mult = backbone_lr_mult
        self.layer_lr_decay = layer_lr_decay
        self.detector = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=img_size,
            max_size=img_size,
            box_score_thresh=score_thresh,
            box_detections_per_img=max_detections,
        )
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

        if freeze_bn:
            _freeze_batchnorm(self.detector)

        if small_anchors:
            self.detector.rpn.anchor_generator = AnchorGenerator(
                sizes=((16,), (32,), (64,), (128,), (256,)),
                aspect_ratios=((0.5, 1.0, 2.0),) * 5,
            )

    def forward(
        self,
        img: list[torch.Tensor],
        boxes: list[torch.Tensor] | None = None,
        labels: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        if self.training:
            assert boxes is not None and labels is not None, 'Boxes and labels are required.'
            # The detector reserves 0 for the background
            targets = [
                {'boxes': b, 'labels': lbl + 1} for b, lbl in zip(boxes, labels, strict=True)
            ]
            return self.detector(img, targets)

        detections = self.detector(img)
        for det in detections:
            det['labels'] = det['labels'] - 1
        return {'detections': detections}

    def get_param_groups(self, base_lr: float) -> list[dict]:
        """
        Parameter groups with the learning rates of the model parts: the FPN, RPN and the heads
        use 'base_lr', the ResNet stages use 'base_lr * backbone_lr_mult * layer_lr_decay ** (4 - s)'
        (see the constructor). The groups of the same learning rate are merged.
        """
        by_lr: dict[float, list[nn.Parameter]] = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('detector.backbone.body.'):
                match = re.search(r'\.layer(\d)\.', name)
                stage = int(match.group(1)) if match else 0  # conv1 and bn1 form the stem
                mult = self.backbone_lr_mult * self.layer_lr_decay ** (4 - stage)
            else:
                mult = 1.0
            by_lr.setdefault(base_lr * mult, []).append(param)
        return [{'params': params, 'lr': lr} for lr, params in sorted(by_lr.items())]
