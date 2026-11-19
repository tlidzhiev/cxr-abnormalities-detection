import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import safetensors.torch
import torch
from datasets import load_dataset
from torchvision.transforms import v2
from tqdm.auto import tqdm

from lab.utils.io import get_root, read_json, write_json

from .base import BaseDataset

logger = logging.getLogger(__name__)


class MNISTDataset(BaseDataset):
    """
    MNIST dataset, see https://yann.lecun.com/exdb/mnist/.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        split: Literal['train', 'test'] = 'train',
        limit: int | None = None,
        shuffle_index: bool = False,
        instance_transforms: dict[str, Callable] | None = None,
        force_reindex: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        root : Path or str or None, optional
            Dataset directory relative to the project root. If None, 'data/mnist/{split}'.
        split : {'train', 'test'}, optional
            Dataset split.
        limit : int or None, optional
            Limit the number of samples.
        shuffle_index : bool, optional
            Whether to shuffle the index.
        instance_transforms : dict[str, Callable] or None, optional
            Instance transforms to apply.
        force_reindex : bool, optional
            Recreate the index file even if it exists.
        """
        if root is None:
            root = get_root() / 'data' / 'mnist' / split
        else:
            root = get_root() / root

        index_path = root / 'index.json'
        if index_path.exists() and not force_reindex:
            index: list[dict[str, Any]] = read_json(str(index_path))  # ty:ignore[invalid-assignment]
        else:
            index: list[dict[str, Any]] = self._create_index(split, root)

        super().__init__(
            index=index,
            limit=limit,
            shuffle_index=shuffle_index,
            instance_transforms=instance_transforms,
        )

    def _create_index(
        self,
        split: Literal['train', 'test'],
        data_path: Path,
    ) -> list[dict[str, Any]]:
        """
        Download MNIST, save each image as a safetensors file and write index.json.

        Returns
        -------
        list[dict[str, Any]]
            Index with the 'path' and 'label' of each element.
        """

        index: list[dict[str, Any]] = []
        data_path.mkdir(exist_ok=True, parents=True)

        transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        mnist_dataset = load_dataset('ylecun/mnist', split=split)

        logger.info(f'Processing MNIST data for {split} part...')
        for i, item in tqdm(enumerate(mnist_dataset), total=len(mnist_dataset)):
            img, label = transform(item['image']), item['label']

            save_dict = {'img': img}
            save_path = data_path / f'{i:06}.safetensors'
            safetensors.torch.save_file(save_dict, save_path)

            index.append({'path': str(save_path), 'label': int(label)})

        write_json(index, str(data_path / 'index.json'))

        logger.info(f'Successfully processed {len(index)} {split} MNIST images.')
        return index
