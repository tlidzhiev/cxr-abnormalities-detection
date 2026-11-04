from pathlib import Path
from typing import Any, Literal

import comet_ml
import numpy as np

from .base import BaseWriter


class CometMLWriter(BaseWriter):
    """
    Experiment tracking via CometML, see https://www.comet.com/docs/v2/.
    """

    def __init__(
        self,
        project_config: dict[str, Any],
        project_name: str,
        run_name: str,
        workspace: str | None = None,
        run_id: str | None = None,
        mode: Literal['online', 'offline'] = 'online',
        log_code: bool = False,
        log_graph: bool = False,
        auto_metric_logging: bool = False,
        auto_param_logging: bool = False,
        **kwargs,
    ) -> None:
        """
        The API key is expected to be provided by the user in the terminal.

        Parameters
        ----------
        project_config : dict[str, Any]
            Configuration of the project.
        project_name : str
            Name of the project inside CometML.
        run_name : str
            Name of the run.
        workspace : str or None, optional
            Workspace name, used when working in a team.
        run_id : str or None, optional
            Unique identifier of the run (experiment key).
        mode : {'online', 'offline'}, optional
            Log to the remote server or locally.
        log_code, log_graph, auto_metric_logging, auto_param_logging : bool, optional
            Corresponding CometML options.
        """
        super().__init__(project_config, project_name, run_id, run_name, **kwargs)

        # An offline experiment needs no API key
        if mode == 'online':
            comet_ml.login()

        resume = project_config.get('trainer', {}).get('resume_from') is not None

        if resume:
            if mode == 'offline':
                exp_class = comet_ml.ExistingOfflineExperiment
            else:
                exp_class = comet_ml.ExistingExperiment

            self.exp = exp_class(experiment_key=self.run_id)
        else:
            if mode == 'offline':
                exp_class = comet_ml.OfflineExperiment
            else:
                exp_class = comet_ml.Experiment

            self.exp = exp_class(
                project_name=project_name,
                workspace=workspace,
                experiment_key=self.run_id,
                log_code=log_code,
                log_graph=log_graph,
                auto_metric_logging=auto_metric_logging,
                auto_param_logging=auto_param_logging,
            )
            self.exp.set_name(run_name)
            self.exp.log_parameters(parameters=_nested_lists_to_str(project_config))

    def add_checkpoint(self, checkpoint_path: str, save_dir: str) -> None:
        # For comet, save dir is not required
        # It is kept for consistency with base class
        self.exp.log_model(name='checkpoints', file_or_folder=checkpoint_path, overwrite=True)

    def add_scalar(self, name: str, value: float) -> None:
        self.exp.log_metrics(
            {self._object_name(name): value},
            step=self.step,
        )

    def add_scalars(self, values: dict[str, float]) -> None:
        self.exp.log_metrics(
            {self._object_name(k): v for k, v in values.items()},
            step=self.step,
        )

    def add_image(self, name: str, image: np.ndarray | Path | str) -> None:
        self.exp.log_image(image_data=image, name=self._object_name(name), step=self.step)

    def add_histogram(
        self,
        name: str,
        values: np.ndarray,
        bins: int | None = None,
    ) -> None:
        # For comet, bins argument is not required
        # It is kept for consistency with base class
        self.exp.log_histogram_3d(values=values, name=self._object_name(name), step=self.step)

    def close(self) -> None:
        self.exp.end()


def _nested_lists_to_str(value: Any) -> Any:
    # Comet rejects nested lists (lists of lists or dicts) as parameter values.
    if isinstance(value, dict):
        return {key: _nested_lists_to_str(item) for key, item in value.items()}
    if isinstance(value, list) and any(isinstance(item, list | dict) for item in value):
        return str(value)
    return value
