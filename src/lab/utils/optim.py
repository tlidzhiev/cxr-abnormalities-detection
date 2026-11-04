from collections.abc import Callable
from math import cos, pi

from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


def get_lr_scheduler(
    config: DictConfig,
    optimizer: Optimizer,
    total_steps: int,
) -> LRScheduler:
    """
    Get a learning rate scheduler from config: 'constant', or 'cosine' with linear warmup.

    Parameters
    ----------
    config : DictConfig
        Experiment config.
    optimizer : Optimizer
        PyTorch optimizer.
    total_steps : int
        Total number of optimizer steps.

    Returns
    -------
    LRScheduler
        Learning rate scheduler.
    """
    scheduler_name = config.lr_scheduler.name

    if scheduler_name == 'constant':
        return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    if scheduler_name != 'cosine':
        raise ValueError(f'Unknown LR scheduler: {scheduler_name}.')

    num_training_steps = total_steps
    if num_training_steps <= 0:
        raise ValueError('The number of training steps must be positive.')

    warmup_ratio = float(config.lr_scheduler.get('warmup_ratio', 0.0))
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError('warmup_ratio must be in the range [0, 1).')

    min_lr = float(config.lr_scheduler.get('min_lr', 0.0))
    if min_lr < 0.0:
        raise ValueError('min_lr must be non-negative.')

    base_lrs = [float(param_group['lr']) for param_group in optimizer.param_groups]
    if any(min_lr > base_lr for base_lr in base_lrs):
        raise ValueError('min_lr must not exceed the initial LR of any parameter group.')

    num_warmup_steps = min(
        round(num_training_steps * warmup_ratio),
        num_training_steps - 1,
    )

    def make_lr_lambda(base_lr: float) -> Callable[[int], float]:
        min_lr_ratio = min_lr / base_lr if base_lr > 0.0 else 0.0

        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step / max(1, num_warmup_steps))

            progress = float(current_step - num_warmup_steps) / max(
                1, num_training_steps - num_warmup_steps
            )
            progress = min(progress, 1.0)
            cosine_ratio = 0.5 * (1.0 + cos(pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_ratio

        return lr_lambda

    return LambdaLR(
        optimizer,
        lr_lambda=[make_lr_lambda(base_lr) for base_lr in base_lrs],
    )
