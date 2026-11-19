from collections.abc import Callable
from typing import Any

import numpy as np


class DetectionTransform:
    """
    Adapter of an Albumentations pipeline for a detection instance.

    The instance has 'img' (H, W, C uint8 array), 'boxes' (N, 4 array, xyxy in pixels) and
    'labels' (N array). The pipeline must be built with 'bbox_params' (format 'pascal_voc',
    label_fields=['labels']). Other fields of the instance are kept as they are.
    """

    def __init__(self, compose: Callable[..., dict[str, Any]]) -> None:
        """
        Parameters
        ----------
        compose : Callable
            'albumentations.Compose' with 'bbox_params'.
        """
        self.compose = compose

    def __call__(self, instance: dict[str, Any]) -> dict[str, Any]:
        out = self.compose(
            image=instance['img'],
            bboxes=instance['boxes'],
            labels=instance['labels'],
        )
        return {
            **instance,
            'img': out['image'],
            'boxes': np.asarray(out['bboxes'], dtype=np.float32).reshape(-1, 4),
            'labels': np.asarray(out['labels'], dtype=np.int64),
        }
