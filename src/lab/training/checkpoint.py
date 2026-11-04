import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self, cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.swa_utils import AveragedModel

from lab.logger.base import BaseWriter

logger = logging.getLogger(__name__)


@dataclass
class Monitor:
    mode: Literal['off', 'min', 'max'] = 'off'
    metric: str = ''
    early_stop: int | None = None
    best: float = field(init=False)
    not_improved_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.mode not in ('off', 'min', 'max'):
            raise ValueError(f"Monitor mode must be 'off', 'min' or 'max', got '{self.mode}'.")
        if self.early_stop is not None and self.early_stop <= 0:
            self.early_stop = None
        self.best = -math.inf if self.mode == 'max' else math.inf

    @classmethod
    def from_string(cls, spec: str, early_stop: int | None = None) -> Self:
        if spec == 'off':
            return cls()
        mode, metric = spec.split()
        return cls(mode=cast(Literal['min', 'max'], mode), metric=metric, early_stop=early_stop)

    @property
    def enabled(self) -> bool:
        return self.mode != 'off'

    @property
    def should_stop(self) -> bool:
        return self.early_stop is not None and self.not_improved_count >= self.early_stop

    def update(self, logs: Mapping[str, Any]) -> bool:
        if not self.enabled:
            return False
        if self.metric not in logs:
            logger.warning(
                f"Metric '{self.metric}' is not found. Performance monitoring is disabled."
            )
            self.mode = 'off'
            return False

        metric = float(logs[self.metric])
        improved = metric < self.best if self.mode == 'min' else metric > self.best
        if improved:
            self.best = metric
            self.not_improved_count = 0
        else:
            self.not_improved_count += 1
        return improved


class Checkpointer:
    def __init__(
        self,
        checkpoint_dir: Path | str,
        config: DictConfig,
        device: str,
        monitor: Monitor | None = None,
        writer: BaseWriter | None = None,
    ) -> None:
        self.directory = Path(checkpoint_dir)
        self.config = config
        self.device = device
        self.monitor = monitor or Monitor()
        self.writer = writer

    def path(self, step: int, kind: Literal['step', 'best', 'interrupted'] = 'step') -> Path:
        match kind:
            case 'best':
                return self.directory / 'model-best.pth'
            case 'interrupted':
                return self.directory / 'checkpoint-interrupted.pth'
            case _:
                return self.directory / f'checkpoint-step-{step}.pth'

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        ema: AveragedModel | None = None,
        scaler: torch.amp.GradScaler | None = None,
        kind: Literal['step', 'best', 'interrupted'] = 'step',
    ) -> Path:
        """
        Save a checkpoint with model, optimizer, lr_scheduler, EMA, GradScaler and monitor states.

        Returns
        -------
        Path
            Path to the saved file.
        """
        path = self.path(step, kind)
        state = {
            'arch': type(model).__name__,
            'step': step,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'monitor_best': self.monitor.best,
            'not_improved_count': self.monitor.not_improved_count,
            'config': OmegaConf.to_container(self.config, resolve=True),
        }
        if ema is not None:
            state['ema'] = ema.state_dict()
        if scaler is not None and scaler.is_enabled():
            state['scaler'] = scaler.state_dict()

        # Write to a temporary file first, so an interruption never leaves a truncated checkpoint
        tmp_path = path.with_name(path.name + '.tmp')
        torch.save(state, tmp_path)
        tmp_path.replace(path)
        if self.writer is not None:
            self.writer.add_checkpoint(str(path), str(self.directory.parent))
        return path

    def restore(
        self,
        path: Path | str,
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        ema: AveragedModel | None = None,
        scaler: torch.amp.GradScaler | None = None,
    ) -> int:
        """
        Resume from a checkpoint, loading all states in place.

        Returns
        -------
        int
            The global optimizer step to resume from.

        Raises
        ------
        RuntimeError
            If the model, optimizer or lr_scheduler config differs from the checkpoint one.
        """
        checkpoint = self._load(path, self.device)

        current = OmegaConf.to_container(self.config, resolve=True)
        assert isinstance(current, dict)
        if checkpoint['config']['model'] != current['model']:
            raise RuntimeError(
                'Architecture configuration given in the config file is different '
                'from that of the checkpoint.'
            )
        if (
            checkpoint['config']['optimizer'] != current['optimizer']
            or checkpoint['config']['lr_scheduler'] != current['lr_scheduler']
        ):
            raise RuntimeError(
                'Optimizer or lr_scheduler given in the config file is different '
                'from that of the checkpoint. Cannot resume optimizer and scheduler states.'
            )

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if ema is not None and 'ema' in checkpoint:
            ema.load_state_dict(checkpoint['ema'])
        # Missing when resuming a float16 run from a non-float16 checkpoint: keep the fresh scale
        if scaler is not None and scaler.is_enabled() and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])

        mnt_best = checkpoint.get('monitor_best')
        if self.monitor.enabled and mnt_best is not None and math.isfinite(mnt_best):
            self.monitor.best = mnt_best
            self.monitor.not_improved_count = checkpoint.get('not_improved_count', 0)
        return checkpoint.get('step', checkpoint.get('global_step', 0))

    @classmethod
    def from_pretrained(
        cls,
        model: nn.Module,
        path: Path | str,
        device: str = 'cpu',
        use_ema: bool = False,
    ) -> nn.Module:
        """
        Init the model with weights from a .pth file.

        Parameters
        ----------
        model : nn.Module
            PyTorch model.
        path : Path or str
            Checkpoint or bare model state dict.
        device : str, optional
            Device to map the checkpoint tensors to.
        use_ema : bool, optional
            Load the EMA weights instead of the raw ones (requires an 'ema' state).

        Returns
        -------
        nn.Module
            The initialized model.

        Raises
        ------
        RuntimeError
            If 'use_ema' is True but the checkpoint has no EMA state.
        """
        checkpoint = cls._load(path, device)

        if use_ema:
            if 'ema' not in checkpoint:
                raise RuntimeError(
                    f'use_ema=True but no EMA state found in checkpoint: {path}. '
                    'The model was likely trained without EMA.'
                )
            prefix = 'module.'
            weights = {
                key.removeprefix(prefix): value
                for key, value in checkpoint['ema'].items()
                if key.startswith(prefix)
            }
        else:
            weights = checkpoint.get('state_dict', checkpoint)
        model.load_state_dict(weights)
        return model

    @staticmethod
    def _load(path: Path | str, device: str) -> dict[str, Any]:
        """
        Load a checkpoint from disk.

        The config is stored as a plain container, so the safe 'weights_only' mode is enough.
        """
        return torch.load(path, map_location=device, weights_only=True)
