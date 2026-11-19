import logging
import time
from collections.abc import Generator
from contextlib import contextmanager

import torch

from lab.metrics.tracker import MetricTracker

logger = logging.getLogger(__name__)

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EarlyStopping:
    monitor: str = 'off'  # 'off' or '<min|max> <metric_name>'
    patience: int | None = None  # evaluations without improvement before stopping
    best: float = field(init=False)
    not_improved_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        parts = self.monitor.split()
        if self.monitor == 'off':
            self.mode, self.metric = 'off', ''
        elif len(parts) == 2 and parts[0] in ('min', 'max'):
            self.mode, self.metric = parts
        else:
            raise ValueError(
                f"monitor must be 'off' or '<min|max> <metric_name>': {self.monitor!r}"
            )

        if self.patience is not None:
            if self.patience < 1:
                raise ValueError(f'patience must be positive, got {self.patience}.')
            if self.mode == 'off':
                raise ValueError("Early stopping requires monitor to be set (not 'off').")
        self.best = -math.inf if self.mode == 'max' else math.inf

    @property
    def should_stop(self) -> bool:
        return self.patience is not None and self.not_improved_count >= self.patience

    def update(self, logs: dict[str, Any]) -> bool:
        if self.mode == 'off':
            return False
        value = float(logs[self.metric])
        improved = value < self.best if self.mode == 'min' else value > self.best
        if improved:
            self.best = value
            self.not_improved_count = 0
        else:
            self.not_improved_count += 1
        return improved

    def state_dict(self) -> dict[str, Any]:
        best = self.best if math.isfinite(self.best) else None
        return {'best': best, 'not_improved_count': self.not_improved_count}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self.mode != 'off' and state.get('best') is not None:
            self.best = float(state['best'])
            self.not_improved_count = int(state.get('not_improved_count', 0))


class StepTimer:
    """
    Measures the time of waiting for the data and the compute time of a train step.

    A step may fetch several batches (gradient accumulation), so the data time is summed
    over them and the compute time is the rest of the step.

    When disabled, CUDA is not synchronized and nothing is recorded.
    """

    def __init__(self, enabled: bool, device: str) -> None:
        """
        Parameters
        ----------
        enabled : bool
            Measure the times (CUDA is synchronized at the end of a step).
        device : str
            Device of the model.
        """
        self.enabled = enabled
        self.device = device
        self.sync = enabled and torch.device(device).type == 'cuda'
        self._step_start = 0.0
        self.data_time = 0.0
        self.step_time = 0.0

    @property
    def metric_names(self) -> list[str]:
        """
        Names of the tracked metrics (empty when disabled).
        """
        return ['data_time', 'step_time'] if self.enabled else []

    def start_step(self) -> None:
        """
        Start the step.
        """
        self._step_start = time.perf_counter()
        self.data_time = 0.0

    @contextmanager
    def data(self) -> Generator[None]:
        """
        Measure waiting for a batch, adding it to the data time of the step.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            self.data_time += time.perf_counter() - start

    def end_step(self) -> None:
        """
        Finish the step, waiting for the CUDA kernels to complete.
        """
        if self.sync:
            torch.cuda.synchronize(self.device)
        self.step_time = time.perf_counter() - self._step_start - self.data_time

    def record(self, tracker: MetricTracker) -> None:
        """
        Add the times of the step to the running averages of the tracker.
        """
        if self.enabled:
            tracker.update('data_time', self.data_time)
            tracker.update('step_time', self.step_time)
