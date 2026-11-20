import math
import re
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNConvFCHead
from torchvision.models.detection.rpn import RPNHead
from transformers import Dinov2Config, Dinov2Model

from .base import BaseModel
from .faster_rcnn import _float32_output_hook

# Pyramid levels given to the detector (the strides of the ResNet-50 FPN of the baseline), plus
# the 'pool' level (stride 64) made by max pooling of the last one
_PYRAMID_STRIDES = (4, 8, 16, 32)

# Statistics of the RAD-DINO image processor (grayscale images replicated to 3 channels)
_RAD_DINO_MEAN = (0.5307, 0.5307, 0.5307)
_RAD_DINO_STD = (0.2583, 0.2583, 0.2583)


class LayerNorm2d(nn.LayerNorm):
    """
    LayerNorm over the channels of each pixel of a (B, C, H, W) feature map (as in ViTDet).
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


def _pyramid_level(in_channels: int, out_channels: int, scale: float) -> nn.Sequential:
    """
    A level of the ViTDet simple feature pyramid: resample the ViT feature map by 'scale' (a power
    of 2: transposed convolutions up, max pooling down), then a 1x1 and a 3x3 convolution.
    """
    log_scale = math.log2(scale)
    if log_scale != round(log_scale):
        raise ValueError(f'The scale of a pyramid level must be a power of 2, got {scale}.')
    layers: list[nn.Module] = []
    channels = in_channels
    for i in range(max(int(log_scale), 0)):
        if i > 0:
            layers += [LayerNorm2d(channels), nn.GELU()]
        layers.append(nn.ConvTranspose2d(channels, channels // 2, kernel_size=2, stride=2))
        channels //= 2
    for _ in range(max(-int(log_scale), 0)):
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    layers += [
        nn.Conv2d(channels, out_channels, kernel_size=1, bias=False),
        LayerNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        LayerNorm2d(out_channels),
    ]
    for layer in layers:
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    return nn.Sequential(*layers)


class RadDinoBackbone(nn.Module):
    """
    RAD-DINO (ViT-B/14) with the ViTDet simple feature pyramid, as a torchvision detection backbone.

    Patch 14 gives strides of 14 / 7 / 3.5, but torchvision assumes the integer strides of the
    feature maps (AnchorGenerator: 'image_size // grid_size') and rounds the RoIAlign scales to
    powers of 2 (MultiScaleRoIAlign). So the image (H, W) is resized to (H, W) * 14 / feature_stride
    before the ViT: its token grid is (H, W) / feature_stride, that is exactly the stride
    'feature_stride' of the input image (e.g. 512 -> 448 -> 32 x 32 tokens for stride 16,
    512 -> 896 -> 64 x 64 tokens for stride 8). The pyramid is built from the last layer at the
    strides 4, 8, 16, 32 (keys '0'-'3') and 64 ('pool'), like the ResNet-50 FPN.
    """

    def __init__(
        self,
        model_name: str = 'microsoft/rad-dino',
        pretrained: bool = True,
        feature_stride: int = 16,
        out_channels: int = 256,
        drop_path_rate: float = 0.1,
        grad_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        config = Dinov2Config.from_pretrained(model_name)
        config.drop_path_rate = drop_path_rate
        if pretrained:
            self.vit = Dinov2Model.from_pretrained(model_name, config=config)
        else:  # architecture only, e.g. for inference from a checkpoint of the detector
            self.vit = Dinov2Model(config)
        if grad_checkpointing:
            self.vit.activation_checkpointing_enable()

        self.patch_size: int = config.patch_size  # ty:ignore[invalid-assignment]
        self.num_layers: int = config.num_hidden_layers
        self.feature_stride = feature_stride
        self.out_channels = out_channels  # read by torchvision FasterRCNN

        hidden_size = self.vit.config.hidden_size
        self.pyramid = nn.ModuleList(
            _pyramid_level(hidden_size, out_channels, feature_stride / stride)
            for stride in _PYRAMID_STRIDES
        )

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        h, w = x.shape[-2:]
        if h % self.feature_stride or w % self.feature_stride:
            raise ValueError(
                f'The image size {h}x{w} must be divisible by feature_stride={self.feature_stride}.'
            )
        grid = (h // self.feature_stride, w // self.feature_stride)
        x = F.interpolate(
            x,
            size=(grid[0] * self.patch_size, grid[1] * self.patch_size),
            mode='bicubic',
            align_corners=False,
            antialias=True,
        )
        tokens = self.vit(pixel_values=x).last_hidden_state[:, 1:]  # drop the CLS token
        feats = tokens.transpose(1, 2).reshape(tokens.shape[0], -1, *grid)

        out = OrderedDict((str(i), level(feats)) for i, level in enumerate(self.pyramid))
        out['pool'] = F.max_pool2d(out[str(len(self.pyramid) - 1)], kernel_size=1, stride=2)
        return out


class RadDinoFasterRCNNModel(BaseModel):
    def __init__(
        self,
        num_classes: int = 14,
        img_size: int = 512,
        model_name: str = 'microsoft/rad-dino',
        pretrained: bool = True,
        feature_stride: int = 16,
        drop_path_rate: float = 0.1,
        grad_checkpointing: bool = False,
        score_thresh: float = 0.001,
        max_detections: int = 300,
        small_anchors: bool = False,
        backbone_lr_mult: float = 1.0,
        layer_lr_decay: float = 0.7,
    ) -> None:
        super().__init__()
        self.backbone_lr_mult = backbone_lr_mult
        self.layer_lr_decay = layer_lr_decay

        backbone = RadDinoBackbone(
            model_name=model_name,
            pretrained=pretrained,
            feature_stride=feature_stride,
            drop_path_rate=drop_path_rate,
            grad_checkpointing=grad_checkpointing,
        )
        self.num_vit_layers = backbone.num_layers

        sizes = (16, 32, 64, 128, 256) if small_anchors else (32, 64, 128, 256, 512)
        anchor_generator = AnchorGenerator(
            sizes=tuple((s,) for s in sizes),
            aspect_ratios=((0.5, 1.0, 2.0),) * len(sizes),
        )
        channels = backbone.out_channels
        # Heads of the Faster R-CNN v2 (2 convs in the RPN, 4 convs + 1 fc in the box head), with
        # LayerNorm instead of BatchNorm (ViTDet)
        rpn_head = RPNHead(channels, anchor_generator.num_anchors_per_location()[0], conv_depth=2)
        box_head = FastRCNNConvFCHead(
            (channels, 7, 7), [channels] * 4, [1024], norm_layer=LayerNorm2d
        )
        self.detector = FasterRCNN(
            backbone,
            num_classes=num_classes + 1,  # + background
            min_size=img_size,
            max_size=img_size,
            image_mean=list(_RAD_DINO_MEAN),
            image_std=list(_RAD_DINO_STD),
            rpn_anchor_generator=anchor_generator,
            rpn_head=rpn_head,
            box_head=box_head,
            box_score_thresh=score_thresh,
            box_detections_per_img=max_detections,
        )

        # Box geometry in float32 under bf16 autocast, see FasterRCNNModel
        for module in (
            self.detector.rpn.anchor_generator,
            self.detector.rpn.head,
            self.detector.roi_heads.box_predictor,
        ):
            module.register_forward_hook(_float32_output_hook)  # ty:ignore[missing-union-attribute]

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
        Compile the dense parts of the detector in place (the backbone with the pyramid, the RPN
        head, the box head and the box predictor), see FasterRCNNModel.compile.
        """
        for module in (
            self.detector.backbone,
            self.detector.rpn.head,
            self.detector.roi_heads.box_head,
            self.detector.roi_heads.box_predictor,
        ):
            module.compile(*args, **kwargs)  # ty:ignore[missing-union-attribute]

    def _vit_layer_id(self, name: str) -> int:
        """
        Depth of a ViT parameter for the layer-wise LR decay: 0 for the embeddings, i + 1 for the
        block i, num_layers + 1 for the final LayerNorm.
        """
        match = re.search(r'\.encoder\.layer\.(\d+)\.', name)
        if match:
            return int(match.group(1)) + 1
        if '.embeddings.' in name:
            return 0
        return self.num_vit_layers + 1

    def get_param_groups(self, base_lr: float) -> list[dict]:
        """
        Parameter groups with the learning rates and weight decay of the model parts.

        The pyramid, RPN and box heads use 'base_lr'. The ViT parameters use
        'base_lr * backbone_lr_mult * layer_lr_decay ** (num_layers + 1 - layer_id)' (layer-wise
        LR decay, ViTDet / BEiT). The biases, norms, LayerScale and the position / CLS tokens
        get no weight decay; the other groups use the 'weight_decay' of the optimizer.
        """
        groups: dict[tuple[float, bool], list[nn.Parameter]] = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('detector.backbone.vit.'):
                depth = self.num_vit_layers + 1 - self._vit_layer_id(name)
                mult = self.backbone_lr_mult * self.layer_lr_decay**depth
            else:
                mult = 1.0
            no_decay = param.ndim <= 1 or name.endswith(
                ('position_embeddings', 'cls_token', 'mask_token')
            )
            groups.setdefault((base_lr * mult, no_decay), []).append(param)

        param_groups = []
        for (lr, no_decay), params in sorted(groups.items()):
            group: dict = {'params': params, 'lr': lr}
            if no_decay:
                group['weight_decay'] = 0.0
            param_groups.append(group)
        return param_groups
