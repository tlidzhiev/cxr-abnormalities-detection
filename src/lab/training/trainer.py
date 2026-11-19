from typing import Any

import torch
from torchvision.utils import make_grid

from lab.logger.image import draw_detections
from lab.metrics.tracker import MetricTracker

from .base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer for object detection models.

    A batch is a dict of lists (one element per image): 'img', 'boxes', 'labels', 'image_id'.
    In train mode the model returns its own losses, which the criterion aggregates; in eval
    mode it returns the 'detections' only, so there is no loss on the validation, and the
    metrics are accumulated over the whole partition.

    Batch transforms are not supported (the images have different sizes in general).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # No losses in eval mode, so only the metrics are tracked on the validation
        self.eval_metric_tracker = MetricTracker(
            *[n for m in self.metrics['inference'] for n in m.result_names]
        )
        self.classes: list[str] | None = getattr(self.train_loader.dataset, 'classes', None)

    def process_batch(self, batch: dict[str, Any], part: str) -> dict[str, Any]:
        """
        Run a batch through the model, compute loss and metrics, and backpropagate the loss.
        """
        is_train = part == 'train'
        metric_tracker = self.train_metric_tracker if is_train else self.eval_metric_tracker

        batch = self.prepare_batch(batch, is_train=is_train)

        with self.autocast():
            outputs = self.model(**batch)
            batch.update(outputs)
            if is_train:
                batch.update(self.criterion(**batch))

        if is_train:
            self.backward(batch['loss'])  # Sum of all losses is always called loss
            for loss_name in self.criterion.loss_names:
                metric_tracker.update(loss_name, batch[loss_name].detach())

        for met in self.metrics['train' if is_train else 'inference']:
            if is_train:
                metric_tracker.update(met.name, met(**batch))
            else:
                # Accumulated over the whole partition, computed in '_evaluate_part'
                met.update(**batch)

        return batch

    @torch.no_grad()
    def _log_batch(self, batch: dict[str, Any], part: str) -> None:
        """
        Log the first images of an evaluation batch, one row per image: the ground-truth boxes
        on the left, the predicted boxes (score above 'log_score_thresh') on the right.
        """
        if 'detections' not in batch:
            return

        num_images = min(self.cfg_trainer.log_images, len(batch['img']))
        tile_size = self.cfg_trainer.log_tile_size
        score_thresh = self.cfg_trainer.log_score_thresh

        rows = []
        for i in range(num_images):
            det = batch['detections'][i]
            keep = det['scores'] >= score_thresh
            truth = draw_detections(
                batch['img'][i],
                batch['boxes'][i],
                batch['labels'][i],
                None,
                self.classes,
                tile_size,
            )
            pred = draw_detections(
                batch['img'][i],
                det['boxes'][keep],
                det['labels'][keep],
                det['scores'][keep],
                self.classes,
                tile_size,
            )
            rows.append(torch.cat([truth, pred], dim=2))  # side by side

        grid = make_grid(rows, nrow=1, padding=2, pad_value=255)
        self.writer.add_image('truth_vs_predictions', grid.permute(1, 2, 0).numpy())
