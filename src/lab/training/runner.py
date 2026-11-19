import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from lab.metrics.base import BaseMetric
from lab.model.base import BaseModel

logger = logging.getLogger(__name__)


class BaseRunner:
    """
    Base class of the trainer and the inferencer: a model run on batches on a device.

    Provides the batch preparation, mixed precision, compilation and metric computation
    they share.
    """

    def __init__(
        self,
        device: str,
        model: BaseModel,
        metrics: dict[str, list[BaseMetric]],
        batch_transforms: dict[str, dict[str, nn.Sequential]],
        device_tensors: list[str],
        use_bfloat16: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        device : str
            Device for tensors and model.
        model : BaseModel
            PyTorch model.
        metrics : dict[str, list[BaseMetric]]
            Metrics for training ('train') and inference ('inference').
        batch_transforms : dict[str, dict[str, nn.Sequential]]
            Transforms applied to the whole batch, by tensor name.
        device_tensors : list[str]
            Names of the batch entries to move to the device.
        use_bfloat16 : bool
            Enable bfloat16 mixed precision.
        """
        self.device = device
        self.model = model
        self.metrics = metrics
        self.batch_transforms = batch_transforms
        self.device_tensors = device_tensors

        self.amp_dtype = torch.bfloat16 if use_bfloat16 else None
        if self.amp_dtype is not None:
            logger.info(f'AMP enabled with dtype={self.amp_dtype}')

    def prepare_batch(self, batch: dict[str, Any], is_train: bool) -> dict[str, Any]:
        """
        Move the batch tensors to the device and apply the batch transforms on it ('train'
        or 'inference' ones depending on 'is_train'). Transforming on the device is faster
        than instance transforms, especially on GPU.
        """
        for key in self.device_tensors:
            if key not in batch:
                continue
            value = batch[key]
            if isinstance(value, (list, tuple)):
                batch[key] = [t.to(self.device, non_blocking=True) for t in value]
            else:
                batch[key] = value.to(self.device, non_blocking=True)

        transforms = self.batch_transforms.get('train' if is_train else 'inference')
        if transforms is not None:
            for name, transform in transforms.items():
                if name in batch:
                    batch[name] = transform(batch[name])
        return batch

    def autocast(self) -> torch.autocast:
        """
        Autocast context for the forward pass and loss (a no-op without AMP).
        """
        return torch.autocast(
            device_type=torch.device(self.device).type,
            dtype=self.amp_dtype,
            enabled=self.amp_dtype is not None,
        )

    def load_checkpoint(self, path: Path | str) -> dict[str, Any]:
        """
        Load a checkpoint from disk onto the device.

        The config is stored as a plain container, so the safe 'weights_only' mode is enough.
        """
        return torch.load(path, map_location=self.device, weights_only=True)

    def load_weights(self, path: Path | str, use_ema: bool = False) -> None:
        """
        Init the model with weights from a .pth file (a checkpoint or a bare state dict).

        With 'use_ema', the EMA weights of the checkpoint are loaded instead of the raw ones.

        Raises
        ------
        RuntimeError
            If 'use_ema' is True but the checkpoint has no EMA state.
        """
        logger.info(f'Loading {"EMA" if use_ema else "model"} weights from: {path}...')
        checkpoint = self.load_checkpoint(path)
        if not use_ema:
            self.model.load_state_dict(checkpoint.get('state_dict', checkpoint))
            return

        if 'ema' not in checkpoint:
            raise RuntimeError(
                f'use_ema=True but no EMA state found in checkpoint: {path}. '
                'The model was likely trained without EMA.'
            )
        prefix = 'module.'  # AveragedModel stores the averaged model as 'module'
        self.model.load_state_dict(
            {
                key.removeprefix(prefix): value
                for key, value in checkpoint['ema'].items()
                if key.startswith(prefix)
            }
        )

    def compile_model(self, compile_cfg: DictConfig | dict[str, Any] | None) -> None:
        """
        Compile the model in place with 'torch.compile'; no-op if 'compile_cfg' is None.

        'nn.Module.compile' keeps the module object and its state_dict keys unchanged
        (unlike wrapping with 'torch.compile'), so checkpoints and EMA work as before.
        The config entries are passed to it as keyword arguments (e.g. 'mode', 'fullgraph').
        """
        if compile_cfg is None:
            return
        logger.info(f'Compiling model with options: {compile_cfg}')
        self.model.compile(**dict(compile_cfg))
