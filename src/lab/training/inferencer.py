import json
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
from lab.utils.io import get_root

from .checkpoint import Checkpointer
from .utils import autocast, compile_model, get_amp_dtype, to_device

logger = logging.getLogger(__name__)


class Inferencer:
    """
    Evaluates the detection model on the dataset and saves predictions, without optimizers
    and writers.

    For each partition the predictions are saved to '{save_path}/{partition}/predictions.json':
    {image_id: {'boxes': [[x1, y1, x2, y2], ...], 'scores': [...], 'labels': [...]}} with
    the boxes normalized to [0, 1] by the image size (so they do not depend on the input
    size) and the labels in 0..num_classes-1.
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

        self.device = device
        self.amp_dtype = get_amp_dtype(self.cfg_inferencer.get('amp'))

        self.model = model
        self.batch_transforms = batch_transforms

        self.eval_dataloaders = {k: v for k, v in dataloaders.items() if k != 'train'}

        # Path definition
        self.save_path = get_root() / save_path

        # Define metrics
        self.metrics = metrics
        self.eval_metric_tracker = MetricTracker(
            *[n for m in self.metrics['inference'] for n in m.result_names],
        )

        if not skip_model_load:
            # Initialize model. If the checkpoint was trained with EMA and
            # inferencer.use_ema is set, load the EMA (shadow) weights instead.
            pretrained_path = config.inferencer.get('from_pretrained')
            use_ema = config.inferencer.get('use_ema', False)
            source = 'EMA weights' if use_ema else 'model weights'
            logger.info(f'Loading {source} from: {pretrained_path}...')
            self.model = Checkpointer.from_pretrained(
                model=self.model,
                path=pretrained_path,
                device=self.device,
                use_ema=use_ema,
            )

        if (compile_cfg := self.cfg_inferencer.get('compile')) is not None:
            compile_model(self.model, compile_cfg)
            logger.info(f'Compiling model with options: {compile_cfg}')

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

        with (self.save_path / part / 'predictions.json').open('w') as handle:
            json.dump(self._predictions, handle)

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
        batch = to_device(batch, self.cfg_inferencer.device_tensors, self.device)

        with autocast(self.device, self.amp_dtype):
            outputs = self.model(**batch)
        batch.update(outputs)

        for met in self.metrics['inference']:
            met.update(**batch)

        for img, det, image_id in zip(
            batch['img'], batch['detections'], batch['image_id'], strict=True
        ):
            h, w = img.shape[-2:]
            scale = torch.tensor([w, h, w, h], device=det['boxes'].device)
            boxes = (det['boxes'].double() / scale).clamp(0, 1)
            self._predictions[image_id] = {
                'boxes': boxes.round(decimals=5).cpu().tolist(),
                'scores': det['scores'].double().round(decimals=4).cpu().tolist(),
                'labels': det['labels'].cpu().tolist(),
            }

        return batch
