import torch
from torch import nn

from .base import BaseModel
from .utils import get_activation, initialize_weights


class BaselineModel(BaseModel):
    def __init__(
        self,
        num_features: int,
        num_classes,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # people say it can approximate any function...
            nn.Linear(in_features=num_features, out_features=hidden_dim),
            get_activation('relu'),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            get_activation('relu'),
            nn.Linear(in_features=hidden_dim, out_features=num_classes),
        )

        initialize_weights(self, activation='relu')

    def forward(self, img: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        """
        Return a dict with the 'logits' of the input images.
        """
        return {'logits': self.net(img.flatten(1))}
