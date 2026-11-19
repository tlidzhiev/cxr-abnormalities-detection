from typing import Literal

import torch
from torchmetrics.classification import MulticlassAccuracy

from .base import BaseMetric


class AccuracyMetric(BaseMetric):
    def __init__(
        self,
        num_classes: int,
        device: str,
        name: str | None = None,
        top_k: int = 1,
        average: Literal['micro', 'macro', 'weighted'] = 'micro',
    ) -> None:
        """
        Multiclass accuracy.

        Parameters
        ----------
        num_classes : int
            Number of classes.
        device : str
            Device for the metric ('cuda', 'cpu' or 'auto').
        name : str or None, optional
            Metric name.
        top_k : int, optional
            Number of top predictions to consider correct.
        average : {'micro', 'macro', 'weighted'}, optional
            Averaging strategy across classes.
        """
        super().__init__(name=name)
        self.num_classes = num_classes
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.accuracy = MulticlassAccuracy(
            num_classes=num_classes,
            top_k=top_k,
            average=average,
        ).to(device)

    def update(  # ty:ignore[incompatible-method-override]
        self,
        logits: torch.Tensor,
        label: torch.Tensor,
        **kwargs,
    ) -> None:
        """
        Accumulate the logits and labels of a batch.
        """
        with torch.no_grad():
            self.accuracy.update(logits, label)

    def compute(self) -> torch.Tensor:
        """
        Compute accuracy over all samples since the last reset (a 0-dim tensor).
        """
        return self.accuracy.compute()

    def reset(self) -> None:
        """
        Reset the accumulated state.
        """
        self.accuracy.reset()

    def __call__(
        self,
        logits: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute accuracy (of the given batch, if any) and reset the state.

        Used for per-batch train metrics.
        """
        if logits is not None and label is not None:
            self.update(logits=logits, label=label)
        value = self.compute()
        self.reset()
        return value

    def __repr__(self) -> str:
        return f'{type(self).__name__}(num_classes={self.num_classes})'
