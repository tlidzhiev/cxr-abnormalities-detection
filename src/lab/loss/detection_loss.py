import torch

from .base import BaseLoss


class DetectionLoss(BaseLoss):
    def __init__(self) -> None:
        super().__init__()
        self.loss_names = [
            'loss',
            'loss_classifier',
            'loss_box_reg',
            'loss_objectness',
            'loss_rpn_box_reg',
        ]

    def forward(
        self,
        loss_classifier: torch.Tensor,
        loss_box_reg: torch.Tensor,
        loss_objectness: torch.Tensor,
        loss_rpn_box_reg: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        loss = loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg
        return {
            'loss': loss,
            'loss_classifier': loss_classifier,
            'loss_box_reg': loss_box_reg,
            'loss_objectness': loss_objectness,
            'loss_rpn_box_reg': loss_rpn_box_reg,
        }
