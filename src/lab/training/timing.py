import time
from collections.abc import Generator
from contextlib import contextmanager

import torch

from lab.metrics.tracker import MetricTracker


class StepTimer:
    """
    Measures the time of waiting for the data and the compute time of a train step.

    A step may fetch several batches (gradient accumulation), so the data time is summed
    over them and the compute time is the rest of the step.

    When disabled, all methods are no-ops and CUDA is not synchronized.
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
        if self.enabled:
            self._step_start = time.perf_counter()
            self.data_time = 0.0

    @contextmanager
    def data(self) -> Generator[None]:
        """
        Measure waiting for a batch, adding it to the data time of the step.
        """
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            self.data_time += time.perf_counter() - start

    def end_step(self) -> None:
        """
        Finish the step, waiting for the CUDA kernels to complete.
        """
        if self.enabled:
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
