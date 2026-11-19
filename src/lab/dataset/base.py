import random
from collections.abc import Callable
from typing import Any

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Base class for the datasets.

    Given an index (list of dicts), processes different datasets for the same
    task in the identical manner. A subclass has to build the index and to implement
    'load_instance'.
    """

    def __init__(
        self,
        index: list[dict[str, Any]],
        limit: int | None = None,
        shuffle_index: bool = False,
        instance_transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        index : list[dict[str, Any]]
            Metadata of each element; the 'path' field is required.
        limit : int or None, optional
            If not None, keep only the first 'limit' elements.
        shuffle_index : bool, optional
            Shuffle the index with a fixed seed (42).
        instance_transforms : Callable or None, optional
            Transform applied to a whole instance (dict -> dict), so that it can change
            several fields together, e.g. an image and its boxes.
        """
        self._assert_index_is_valid(index)

        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        self._index: list[dict[str, Any]] = index

        self.instance_transforms = instance_transforms

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Load an element, preprocess it, and combine it into a dict.

        The key names must stay consistent across the dataset, collate_fn, loss and model.
        """
        instance_data = self.load_instance(self._index[index])
        instance_data = self.preprocess_data(instance_data)
        return instance_data

    def __len__(self) -> int:
        return len(self._index)

    def load_instance(self, entry: dict[str, Any]) -> dict[str, Any]:
        """
        Load an element described by an index entry into a dict of data.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement load_instance method.')

    def preprocess_data(self, instance_data: dict[str, Any]) -> dict[str, Any]:
        """
        Apply the instance transform to the whole instance.
        """
        if self.instance_transforms is not None:
            instance_data = self.instance_transforms(instance_data)
        return instance_data

    @staticmethod
    def _filter_records_from_dataset(
        index: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Filter elements of the index.

        Not used yet; call before shuffling and limiting.
        """
        # Filter logic
        return index

    @staticmethod
    def _assert_index_is_valid(index: list[dict[str, Any]]) -> None:
        """
        Check that every index entry has the 'path' field.
        """
        for entry in index:
            assert 'path' in entry, (
                "Each dataset item should include field 'path' - path to sample file."
            )

    @staticmethod
    def _sort_index(index: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Sort the index.

        Not used yet; call after filtering and before shuffling and limiting.
        """
        # Sorting logic
        return index

    @staticmethod
    def _shuffle_and_limit_index(
        index: list[dict[str, Any]],
        limit: int | None,
        shuffle_index: bool,
    ) -> list[dict[str, Any]]:
        """
        Shuffle the index with a fixed seed and limit its length.
        """
        if shuffle_index:
            # Use a local Random instance so this doesn't reset the global
            # RNG state (and clobber the configured trainer/inferencer seed).
            random.Random(42).shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index
