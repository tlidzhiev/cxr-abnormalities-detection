from pathlib import Path
from typing import Any, Literal

import numpy as np


class BaseWriter:
    """
    Base class for experiment tracking writers.
    """

    def __init__(
        self,
        project_config: dict[str, Any],
        project_name: str,
        run_id: str | None = None,
        run_name: str | None = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        project_config : dict[str, Any]
            Configuration of the project.
        project_name : str
            Name of the project.
        run_id : str or None, optional
            Unique identifier of the run.
        run_name : str or None, optional
            Human-readable name of the run.
        """
        self.project_config = project_config
        self.project_name = project_name
        self.run_id = run_id
        self.run_name = run_name
        self.mode = ''
        self.step = 0

    def set_step(self, step: int, mode: Literal['train', 'val', 'test'] | str = 'train') -> None:
        """
        Set the current step and mode (partition name) of the tracker.
        """
        self.mode = mode
        self.step = step

    def add_checkpoint(self, checkpoint_path: str, save_dir: str) -> None:
        """
        Log a checkpoint file.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement add_checkpoint method')

    def add_scalar(self, name: str, value: float) -> None:
        """
        Log a scalar.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement add_scalar method')

    def add_scalars(self, values: dict[str, float]) -> None:
        """
        Log several scalars.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement add_scalars method')

    def add_image(self, name: str, image: np.ndarray | Path | str) -> None:
        """
        Log an image (array or path to a file).
        """
        raise NotImplementedError(f'{type(self).__name__} must implement add_image method')

    def add_histogram(self, name: str, values: np.ndarray, bins: int | None = None):
        """
        Log a histogram of the values.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement add_histogram method')

    def close(self) -> None:
        """
        Finalize and close the tracker.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement close method')

    def _object_name(self, name: str) -> str:
        """
        Prefix the object name with the current mode to separate partitions.
        """
        return f'{self.mode}_{name}'
