import torch


class MetricTracker:
    """
    Aggregates metrics from many batches as running averages.
    """

    def __init__(self, *keys: str) -> None:
        """
        Create a tracker for the metric names (including loss names).
        """
        # Totals may be 0-dim tensors, so a device is synced only when a result is read
        self._totals: dict[str, float | torch.Tensor] = dict.fromkeys(keys, 0.0)
        self._counts: dict[str, int] = dict.fromkeys(keys, 0)

    def reset(self) -> None:
        """
        Reset all metrics.
        """
        for key in self._totals:
            self._totals[key] = 0.0
            self._counts[key] = 0

    def update(self, key: str, value: float | torch.Tensor) -> None:
        """
        Add a batch value to the running average of the metric.

        A tensor is accumulated on its device without a host sync; it must be detached.
        """
        self._totals[key] = self._totals[key] + value
        self._counts[key] += 1

    def __getitem__(self, key: str) -> float:
        """
        Return the average of the metric (0.0 if it was never updated).
        """
        count = self._counts[key]
        return float(self._totals[key] / count) if count > 0 else 0.0

    def result(self) -> dict[str, float]:
        """
        Return the average of each metric.
        """
        return {key: self[key] for key in self._totals}

    def keys(self) -> list[str]:
        """
        Return all metric names.
        """
        return list(self._totals)
