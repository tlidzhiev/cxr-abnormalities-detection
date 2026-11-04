from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn


def to_device(batch: dict[str, Any], keys: list[str], device: str) -> dict[str, Any]:
    """
    Move the tensors with the given keys to the device. A key can hold a tensor or a list
    of tensors (e.g. images of different sizes or boxes of different numbers).
    """
    for key in keys:
        if key not in batch:
            continue
        value = batch[key]
        if isinstance(value, (list, tuple)):
            batch[key] = [t.to(device, non_blocking=True) for t in value]
        else:
            batch[key] = value.to(device, non_blocking=True)
    return batch


def transform_batch(
    batch: dict[str, Any],
    batch_transforms: dict[str, dict[str, nn.Sequential]],
    is_train: bool,
) -> dict[str, Any]:
    """
    Apply batch transforms to the tensors by name, 'train' or 'inference' ones
    depending on 'is_train'. Faster than instance transforms, especially on GPU.
    """
    transforms = batch_transforms.get('train' if is_train else 'inference')
    if transforms is None:
        return batch

    for name, transform in transforms.items():
        if name in batch:
            batch[name] = transform(batch[name])
    return batch


AMP_DTYPES = {'float16': torch.float16, 'bfloat16': torch.bfloat16}


def get_amp_dtype(name: str | None) -> torch.dtype | None:
    """
    Map the 'amp' config value to the autocast dtype; None disables AMP.
    """
    if name is None:
        return None
    if name not in AMP_DTYPES:
        raise ValueError(f'AMP dtype must be one of {list(AMP_DTYPES)} or null, got {name!r}.')
    return AMP_DTYPES[name]


def autocast(device: str, dtype: torch.dtype | None) -> torch.autocast:
    """
    Autocast context for the device type; a no-op if 'dtype' is None.
    """
    return torch.autocast(
        device_type=torch.device(device).type,
        dtype=dtype,
        enabled=dtype is not None,
    )


def compile_model(model: nn.Module, compile_cfg: DictConfig | dict[str, Any] | None) -> None:
    """
    Compile the model in place with 'torch.compile'; no-op if 'compile_cfg' is None.

    'nn.Module.compile' keeps the module object and its state_dict keys unchanged
    (unlike wrapping with 'torch.compile'), so checkpoints and EMA work as before.
    The config entries are passed to it as keyword arguments (e.g. 'mode', 'fullgraph').
    """
    if compile_cfg is None:
        return
    compile_cfg = dict(compile_cfg)
    model.compile(**compile_cfg)
