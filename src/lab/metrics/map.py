import re
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
        class_names: list[str] | None = None,
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
        class_names : list[str] or None, optional
            Class names for the per-class results 'AP{iou}_{class}' (class ids if None).
        name : str or None, optional
            Metric name.
        """
        super().__init__(name=name)
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        names = class_names if class_names is not None else [str(c) for c in range(num_classes)]
        assert len(names) == num_classes, 'One name per class is needed.'
        # 'Nodule/Mass' -> 'AP40_nodule_mass'
        prefix = f'AP{round(iou_threshold * 100)}'
        self._class_keys = [
            f'{prefix}_' + re.sub(r'[^0-9a-zA-Z]+', '_', n).strip('_').lower() for n in names
        ]
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

    @property
    def result_names(self) -> list[str]:
        return [self.name, *self._class_keys]

    def compute_all(self) -> dict[str, float]:
        """
        mAP and AP of each class. A class without ground truth in the data has no AP and is
        not reported.
        """
        results = {self.name: self.compute()}
        for c, ap in self.class_ap.items():
            results[self._class_keys[c]] = ap
        return results

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
