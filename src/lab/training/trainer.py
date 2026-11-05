from typing import Any

import torch
import torch.nn.functional as F
from torchvision.utils import draw_bounding_boxes, make_grid

from lab.metrics.tracker import MetricTracker

from .base import BaseTrainer
from .utils import to_device

# One color per class
COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0',
    '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#800000',
]  # fmt: skip


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

        batch = to_device(batch, self.cfg_trainer.device_tensors, self.device)

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

    def _draw(
        self,
        img: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor | None = None,
        tile_size: int = 512,
    ) -> torch.Tensor:
        """
        Draw the boxes on an image (3, H, W) in [0, 1]; returns a uint8 tile (3, S, S).
        """
        tile = (img.detach().float().cpu() * 255).round().to(torch.uint8)
        if len(boxes) > 0:  # normal images have no boxes
            texts = [
                (self.classes[c] if self.classes else str(c))
                + ('' if scores is None else f' {s:.2f}')
                for c, s in zip(
                    labels.tolist(),
                    scores.tolist() if scores is not None else [0.0] * len(labels),
                    strict=True,
                )
            ]
            tile = draw_bounding_boxes(
                tile,
                boxes.detach().float().cpu(),
                texts,
                colors=[COLORS[c % len(COLORS)] for c in labels.tolist()],
                width=3,
            )
        return F.interpolate(tile[None].float(), size=(tile_size, tile_size))[0].byte()

    @torch.no_grad()
    def _log_batch(self, batch: dict[str, Any], part: str) -> None:
        """
        Log the first images of an evaluation batch, one row per image: the ground-truth boxes
        on the left, the predicted boxes (score above 'log_score_thresh') on the right.
        """
        if 'detections' not in batch:
            return

        num_images = min(self.cfg_trainer.get('log_images', 4), len(batch['img']))
        tile_size = self.cfg_trainer.get('log_tile_size', 512)
        score_thresh = self.cfg_trainer.get('log_score_thresh', 0.3)

        rows = []
        for i in range(num_images):
            det = batch['detections'][i]
            keep = det['scores'] >= score_thresh
            truth = self._draw(
                batch['img'][i], batch['boxes'][i], batch['labels'][i], None, tile_size
            )
            pred = self._draw(
                batch['img'][i],
                det['boxes'][keep],
                det['labels'][keep],
                det['scores'][keep],
                tile_size,
            )
            rows.append(torch.cat([truth, pred], dim=2))  # side by side

        grid = make_grid(rows, nrow=1, padding=2, pad_value=255)
        self.writer.add_image('truth_vs_predictions', grid.permute(1, 2, 0).numpy())
