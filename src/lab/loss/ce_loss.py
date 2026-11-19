import torch
from torch import nn

from .base import BaseLoss


class CrossEntropyLoss(BaseLoss):
    """
    Wrapper over PyTorch CrossEntropyLoss.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.loss_names = ['loss']

    def forward(
        self,
        logits: torch.Tensor,
        label: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Compute the cross-entropy loss, returned under the 'loss' key.
        """
        return {'loss': self.loss(logits, label)}
