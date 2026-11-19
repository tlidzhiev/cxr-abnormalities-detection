from typing import Any

import torch
from torchvision.utils import make_grid

from lab.logger.image import caption_image

from .base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer for image classification models.
    """

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

            all_losses = self.criterion(**batch)
            batch.update(all_losses)

        if is_train:
            self.backward(batch['loss'])  # Sum of all losses is always called loss

        # Update metrics for each loss (in case of multiple losses)
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
        Log a grid of the first images of the batch, each captioned with its true and
        predicted class (green if the prediction is correct, red otherwise).
        """
        num_images = min(self.cfg_trainer.log_images, len(batch['img']))
        labels = batch['label'][:num_images].tolist()
        preds = batch['logits'][:num_images].argmax(dim=-1).tolist()

        tiles = [
            caption_image(img, f'true {label} | pred {pred}', 'green' if label == pred else 'red')
            for img, label, pred in zip(batch['img'][:num_images], labels, preds, strict=True)
        ]
        grid = make_grid(tiles, nrow=8, padding=2, pad_value=255)
        self.writer.add_image('predictions', grid.permute(1, 2, 0).numpy())
