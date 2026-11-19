import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from lab.metrics.base import BaseMetric
from lab.metrics.tracker import MetricTracker
from lab.model.base import BaseModel
from lab.utils.io import get_root, write_json

from .runner import BaseRunner

logger = logging.getLogger(__name__)


class Inferencer(BaseRunner):
    """
    Evaluates the detection model on the dataset and saves predictions, without optimizers
    and writers.

    For each partition the predictions are saved to '{save_path}/{partition}/predictions.json':
    {image_id: {'boxes': [[x1, y1, x2, y2], ...], 'scores': [...], 'labels': [...]}} with
    the boxes normalized to [0, 1] by the image size (so they do not depend on the input
    size) and the labels in 0..num_classes-1. The 'test' partition has no ground truth, so
    only its predictions are saved (no metrics).
    """

    def __init__(
        self,
        config: DictConfig,
        device: str,
        model: BaseModel,
        dataloaders: dict[str, DataLoader],
        metrics: dict[str, list[BaseMetric]],
        save_path: Path | str,
        batch_transforms: dict[str, dict[str, nn.Sequential]],
        skip_model_load: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        config : DictConfig
            Experiment config with the 'inferencer' section.
        device : str
            Device for tensors and model.
        model : BaseModel
            PyTorch model.
        dataloaders : dict[str, DataLoader]
            Dataloader of each partition.
        metrics : dict[str, list[BaseMetric]]
            Metrics for inference ('inference').
        save_path : Path or str
            Directory to save predictions to.
        batch_transforms : dict[str, dict[str, nn.Sequential]]
            Not supported for detection (images of a batch are separate tensors), kept for
            the interface of the entrypoint.
        skip_model_load : bool, optional
            Skip loading model weights from the checkpoint.
        """
        assert skip_model_load or config.inferencer.get('from_pretrained') is not None, (
            'Provide checkpoint or set skip_model_load=True'
        )

        self.config = config
        self.cfg_inferencer = self.config.inferencer
        super().__init__(
            device=device,
            model=model,
            metrics=metrics,
            batch_transforms=batch_transforms,
            device_tensors=list(self.cfg_inferencer.device_tensors),
            use_bfloat16=self.cfg_inferencer.get('use_bfloat16', False),
        )

        # 'inferencer.partitions' (optional) selects the partitions to run, e.g. [test]
        partitions = self.cfg_inferencer.get('partitions')
        self.eval_dataloaders = {
            k: v
            for k, v in dataloaders.items()
            if k != 'train' and (partitions is None or k in partitions)
        }
        assert self.eval_dataloaders, f'No partitions to run (inferencer.partitions={partitions}).'

        # Path definition
        self.save_path = get_root() / save_path

        # Define metrics
        self.metrics = metrics
        self.eval_metric_tracker = MetricTracker(
            *[n for m in self.metrics['inference'] for n in m.result_names],
        )

        if not skip_model_load:
            # inferencer.use_ema loads the EMA (shadow) weights of the checkpoint
            self.load_weights(
                self.cfg_inferencer.from_pretrained,
                use_ema=self.cfg_inferencer.get('use_ema', False),
            )

        self.compile_model(self.cfg_inferencer.get('compile'))

    def run_inference(self) -> dict[str, Any]:
        """
        Run inference on each partition.

        Returns
        -------
        dict[str, Any]
            Metrics of each partition.
        """
        part_logs = {}
        for part, dataloader in self.eval_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    @torch.no_grad()
    def _inference_part(self, part: str, dataloader: DataLoader) -> dict[str, Any]:
        """
        Run inference on a partition and save predictions.

        Returns
        -------
        dict[str, Any]
            Metrics of the partition.
        """
        self.model.eval()
        self.eval_metric_tracker.reset()

        self._predictions: dict[str, dict[str, list]] = {}

        # Create save dir
        (self.save_path / part).mkdir(exist_ok=True, parents=True)

        for batch in tqdm(dataloader, desc=part, total=len(dataloader)):
            self.process_batch(batch=batch, part=part)

        write_json(self._predictions, self.save_path / part / 'predictions.json')

        if part == 'test':
            return {}

        # Metrics are accumulated over all batches and computed once per partition
        for met in self.metrics['inference']:
            for name, value in met.compute_all().items():
                self.eval_metric_tracker.update(name, value)
            met.reset()

        return self.eval_metric_tracker.result()

    def process_batch(self, batch: dict[str, Any], part: str) -> dict[str, Any]:
        """
        Run a batch through the model, update metrics and save predictions to disk.
        """
        batch = self.prepare_batch(batch, is_train=False)

        with self.autocast():
            outputs = self.model(**batch)
        batch.update(outputs)

        if part != 'test':  # the test partition has no ground truth
            for met in self.metrics['inference']:
                met.update(**batch)

        for img, det, image_id in zip(
            batch['img'], batch['detections'], batch['image_id'], strict=True
        ):
            h, w = img.shape[-2:]
            # float64 on the CPU: MPS does not support it
            scale = torch.tensor([w, h, w, h], dtype=torch.float64)
            boxes = (det['boxes'].cpu().double() / scale).clamp(0, 1)
            self._predictions[image_id] = {
                'boxes': boxes.round(decimals=5).tolist(),
                'scores': det['scores'].cpu().double().round(decimals=4).tolist(),
                'labels': det['labels'].cpu().tolist(),
            }

        return batch
