from collections.abc import Generator
from itertools import repeat
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, IterableDataset


def inf_loop(dataloader: DataLoader) -> Generator:
    """
    Cycle over the dataloader endlessly (for iteration-based training).
    """
    for loader in repeat(dataloader):
        yield from loader


def list_collate_fn(batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """
    Collate instances into a dict of lists (no stacking).

    Needed when elements of a batch have different shapes, e.g. a different number
    of boxes per image.
    """
    return {key: [instance[key] for instance in batch] for key in batch[0]}


def _to_device(
    batch_transforms: dict[str, dict[str, nn.Sequential]],
    device: str,
) -> None:
    """
    Move batch transforms to the device.

    Batch transforms must be nn.Module: use nn.Sequential in the config
    (not torchvision.Compose).
    """
    for transforms in batch_transforms.values():
        if transforms is not None:
            for tensor_name, transform in transforms.items():
                transforms[tensor_name] = transform.to(device)


def get_dataloaders(
    config: DictConfig, device: str
) -> tuple[
    dict[str, DataLoader],
    dict[str, dict[str, nn.Sequential]],
]:
    """
    Create dataloaders for each dataset partition, and the batch transforms.

    Parameters
    ----------
    config : DictConfig
        Hydra experiment config.
    device : str
        Device for the batch transforms.

    Returns
    -------
    dataloaders : dict[str, DataLoader]
        Dataloader of each partition.
    batch_transforms : dict[str, dict[str, nn.Sequential]]
        Transforms applied to the whole batch, by tensor name.
    """

    batch_transforms = instantiate(config.transforms.batch_transforms)
    _to_device(batch_transforms, device)
    datasets = instantiate(config.dataset)

    dataloaders = {}
    for partition in config.dataset:
        if partition not in ['train', 'val', 'test'] or config.dataloader.get(partition) is None:
            continue

        dataset = datasets[partition]

        if not isinstance(dataset, IterableDataset):
            batch_size = config.dataloader[partition].batch_size
            assert batch_size <= len(dataset), (
                f'The batch size ({batch_size}) cannot '
                f'be larger than the dataset length ({len(dataset)})'
            )

        loader_cfg = config.dataloader[partition]
        extra = {}
        if loader_cfg.get('sampler') is not None:
            # Hydra cannot pass the dataset to a nested config: the sampler is a partial
            # called with the dataset here (and the sampler replaces shuffling)
            extra = {'sampler': instantiate(loader_cfg.sampler)(dataset), 'shuffle': False}
        partition_dataloader = instantiate(loader_cfg, dataset=dataset, **extra)
        dataloaders[partition] = partition_dataloader

    return dataloaders, batch_transforms
