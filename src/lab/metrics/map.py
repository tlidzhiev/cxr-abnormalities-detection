from typing import Any

import numpy as np
import torch

from .base import BaseMetric
from .metrics import mean_average_precision


class MeanAveragePrecision(BaseMetric):
    def __init__(
        self,
        num_classes: int = 14,
        iou_threshold: float = 0.4,
        name: str | None = None,
    ) -> None:
        """
        Mean average precision over the classes at a fixed IoU threshold (VOC all-point AP,
        as in the VinBigData competition, which uses IoU 0.4).

        The predictions and the ground truth are accumulated over all the batches and AP is
        computed in 'compute', so AP is not an average of batch values.

        Parameters
        ----------
        num_classes : int, optional
            Number of the classes (labels are 0..num_classes-1, no background).
        iou_threshold : float, optional
            IoU threshold of a true positive.
        name : str or None, optional
            Metric name.
        """
        super().__init__(name=name)
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self._preds: dict[str, dict[str, np.ndarray]] = {}
        self._gts: dict[str, dict[str, np.ndarray]] = {}
        # AP of each class from the last 'compute'
        self.class_ap: dict[int, float] = {}

    def update(  # ty:ignore[invalid-method-override]
        self,
        img: list[torch.Tensor],
        boxes: list[torch.Tensor],
        labels: list[torch.Tensor],
        detections: list[dict[str, torch.Tensor]],
        image_id: list[str],
        **kwargs,
    ) -> None:
        """
        Accumulate the predictions and the ground truth of a batch.

        The boxes are normalized by the image size, so that the predicted and the
        ground-truth boxes are compared in the same units.
        """
        for im, gt_boxes, gt_labels, det, key in zip(
            img, boxes, labels, detections, image_id, strict=True
        ):
            h, w = im.shape[-2:]
            scale = torch.tensor([w, h, w, h], dtype=torch.float32)
            self._gts[key] = {
                'boxes': (gt_boxes.detach().cpu().float() / scale).numpy(),
                'labels': gt_labels.detach().cpu().numpy(),
            }
            self._preds[key] = {
                'boxes': (det['boxes'].detach().cpu().float() / scale).numpy(),
                'scores': det['scores'].detach().cpu().float().numpy(),
                'labels': det['labels'].detach().cpu().numpy(),
            }

    def compute(self) -> float:
        """
        Compute mAP over all the samples since the last reset.
        """
        value, self.class_ap = mean_average_precision(
            self._preds,
            self._gts,
            iou_thr=self.iou_threshold,
            num_classes=self.num_classes,
        )
        return value

    def reset(self) -> None:
        """
        Reset the accumulated state.
        """
        self._preds = {}
        self._gts = {}

    def __call__(self, **kwargs: Any) -> float:
        """
        Compute mAP of the given batch only, and reset the state.
        """
        self.update(**kwargs)
        value = self.compute()
        self.reset()
        return value

    def __repr__(self) -> str:
        return (
            f'{type(self).__name__}(num_classes={self.num_classes}, '
            f'iou_threshold={self.iou_threshold})'
        )
