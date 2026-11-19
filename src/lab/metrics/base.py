import torch


class BaseMetric:
    """
    Base class for all metrics.
    """

    def __init__(self, name: str | None = None, *args, **kwargs):
        """
        Set the metric name (the class name by default).
        """

        self.name = name if name is not None else type(self).__name__

    def update(self, **kwargs):
        """
        Update the metric state with batch data.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement update method.')

    def compute(self) -> float | torch.Tensor:
        """
        Compute the metric over all batches since the last reset.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement compute method.')

    @property
    def result_names(self) -> list[str]:
        """
        Names of all the values returned by 'compute_all' (the metric name by default).
        """
        return [self.name]

    def compute_all(self) -> dict[str, float | torch.Tensor]:
        """
        Compute all the values of the metric (a metric can report several, e.g. per class).
        """
        return {self.name: self.compute()}

    def reset(self) -> None:
        """
        Reset the accumulated metric state.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement reset method.')

    def __call__(self, **kwargs) -> float | torch.Tensor:
        """
        Compute the metric for a batch.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement __call__ method.')

    def __str__(self) -> str:
        return f'{type(self).__name__}()'

    def __repr__(self) -> str:
        return self.__str__()
