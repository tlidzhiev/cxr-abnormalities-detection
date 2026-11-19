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
                frozen.weight.copy_(child.weight)  # ty:ignore[not-callable]
                frozen.bias.copy_(child.bias)  # ty:ignore[not-callable]
                frozen.running_mean.copy_(child.running_mean)  # ty:ignore[not-callable,invalid-argument-type]
                frozen.running_var.copy_(child.running_var)  # ty:ignore[not-callable,invalid-argument-type]
            setattr(module, name, frozen)
        else:
            _freeze_batchnorm(child)


def _to_float32(output):
    """
    Cast the low precision floating tensors of a (nested) module output to float32.
    """
    if isinstance(output, torch.Tensor):
        if output.dtype in (torch.bfloat16, torch.float16):
            return output.float()
        return output
    if isinstance(output, (list, tuple)):
        return type(output)(_to_float32(o) for o in output)
    return output


def _float32_output_hook(module: nn.Module, args, output):
    return _to_float32(output)


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

        # Box geometry in float32 under autocast (bf16 AMP): torchvision makes the anchors in
        # the dtype of the feature maps and decodes the boxes in the dtype of the predicted
        # deltas, so with bf16 (8-bit mantissa) the coordinates of the anchors, proposals,
        # regression targets and detections would be rounded to 2 px at 256-512. Casting
        # these outputs keeps the convolutions in bf16 and the box coding, losses and NMS in
        # float32. A no-op in float32; hooks do not change the state_dict keys.
        for module in (
            self.detector.rpn.anchor_generator,
            self.detector.rpn.head,
            self.detector.roi_heads.box_predictor,
        ):
            module.register_forward_hook(_float32_output_hook)

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

    def compile(self, *args, **kwargs) -> None:
        """
        Compile the dense parts of the detector in place, each as one graph: the backbone
        with the FPN, the RPN head, the box head and the box predictor (almost all the FLOPs).

        The rest of the detector (the resize transform, the anchors, the proposal filtering
        and NMS, the matching and sampling of the targets, the RoI pooling, the losses and
        the postprocessing) stays eager: it has data-dependent shapes ('torch.where(mask)',
        NMS, a Python loop over 'torch.unique'), which break the graph in torchvision, so
        compiling the whole model gives ~30 graphs and fails with 'fullgraph=True'.
        The arguments are the ones of 'torch.compile' (e.g. 'fullgraph', 'mode', 'dynamic').
        """
        for module in (
            self.detector.backbone,
            self.detector.rpn.head,
            self.detector.roi_heads.box_head,
            self.detector.roi_heads.box_predictor,
        ):
            module.compile(*args, **kwargs)

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
