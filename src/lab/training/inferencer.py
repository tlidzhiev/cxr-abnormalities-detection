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

from .runner import BaseRunner

logger = logging.getLogger(__name__)


class Inferencer(BaseRunner):
    """
    Evaluates the model on the dataset and saves predictions, without optimizers and writers.
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
            Transforms applied to the whole batch, by tensor name.
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

        self.eval_dataloaders = {k: v for k, v in dataloaders.items() if k != 'train'}
        self._sample_offset = 0  # index of the next saved sample in the current partition

        # Path definition
        self.save_path = get_root() / save_path

        # Define metrics
        self.eval_metric_tracker = MetricTracker(
            *[m.name for m in self.metrics['inference']],
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

        self._sample_offset = 0

        # Create save dir
        (self.save_path / part).mkdir(exist_ok=True, parents=True)

        for batch in tqdm(dataloader, desc=part):
            self.process_batch(batch=batch, part=part)

        for met in self.metrics['inference']:
            self.eval_metric_tracker.update(met.name, met.compute())
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

        for met in self.metrics['inference']:
            met.update(**batch)

        # Some saving logic. This is an example
        # Use if you need to save predictions on disk

        batch_size = batch['logits'].shape[0]
        current_id = self._sample_offset
        self._sample_offset += batch_size

        for i in range(batch_size):
            # Clone because of https://github.com/pytorch/pytorch/issues/1995
            logits = batch['logits'][i].clone()
            label = batch['label'][i].clone()
            pred_label = logits.argmax(dim=-1)

            output_id = current_id + i
            output = {
                'pred_label': pred_label,
                'label': label,
            }

            # You can use safetensors or other lib here
            torch.save(output, self.save_path / part / f'output_{output_id}.pth')

        return batch
