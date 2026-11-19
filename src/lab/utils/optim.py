from collections.abc import Callable
from math import cos, pi

from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

_SCHEDULER_NAMES = ('constant', 'constant_warmup', 'cosine', 'cosine_warmup')


def get_lr_scheduler(
    config: DictConfig,
    optimizer: Optimizer,
    total_steps: int,
) -> LRScheduler:
    """
    Get a learning rate scheduler from config.

    Supported schedulers:
        - 'constant':        constant LR.
        - 'constant_warmup': linear warmup, then constant LR.
        - 'cosine':          cosine decay from the initial LR to min_lr.
        - 'cosine_warmup':   linear warmup, then cosine decay to min_lr.

    Config fields (config.lr_scheduler):
        name : str
            One of the names above.
        warmup_ratio : float | None, default None
            Fraction of total steps used for warmup, in [0, 1).
        warmup_steps : int | None, default None
            Number of warmup steps, in [0, total_steps).
            Exactly one of 'warmup_ratio' / 'warmup_steps' must be set
            for '*_warmup' schedulers (both are ignored by the others).
        warmup_start_lr : float, default 0.0
            LR at step 0 of the warmup, in [0, initial LR of every param group].
            Used only by '*_warmup' schedulers.
        min_lr : float, default 0.0
            Final LR of the cosine decay. Used only by 'cosine*' schedulers.

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

    if scheduler_name not in _SCHEDULER_NAMES:
        raise ValueError(
            f'Unknown LR scheduler: {scheduler_name}. Supported: {", ".join(_SCHEDULER_NAMES)}.'
        )

    if total_steps <= 0:
        raise ValueError('The number of training steps must be positive.')

    use_warmup = scheduler_name.endswith('_warmup')
    use_cosine = scheduler_name.startswith('cosine')

    # Warmup
    num_warmup_steps = 0
    warmup_start_lr = 0.0
    base_lrs = [float(param_group['lr']) for param_group in optimizer.param_groups]
    if use_warmup:
        warmup_ratio = config.lr_scheduler.get('warmup_ratio', None)
        warmup_steps = config.lr_scheduler.get('warmup_steps', None)

        if warmup_ratio is not None and warmup_steps is not None:
            raise ValueError('Specify only one of warmup_ratio or warmup_steps, not both.')
        if warmup_ratio is None and warmup_steps is None:
            raise ValueError(f"'{scheduler_name}' requires either warmup_ratio or warmup_steps.")

        if warmup_ratio is not None:
            warmup_ratio = float(warmup_ratio)
            if not 0.0 <= warmup_ratio < 1.0:
                raise ValueError('warmup_ratio must be in the range [0, 1).')
            num_warmup_steps = min(round(total_steps * warmup_ratio), total_steps - 1)
        else:
            num_warmup_steps = int(warmup_steps)
            if not 0 <= num_warmup_steps < total_steps:
                raise ValueError('warmup_steps must be in the range [0, total_steps).')

        warmup_start_lr = float(config.lr_scheduler.get('warmup_start_lr', 0.0))
        if warmup_start_lr < 0.0:
            raise ValueError('warmup_start_lr must be non-negative.')
        if any(warmup_start_lr > base_lr for base_lr in base_lrs):
            raise ValueError(
                'warmup_start_lr must not exceed the initial LR of any parameter group.'
            )

    # Cosine decay
    min_lr = 0.0
    if use_cosine:
        min_lr = float(config.lr_scheduler.get('min_lr', 0.0))
        if min_lr < 0.0:
            raise ValueError('min_lr must be non-negative.')
        if any(min_lr > base_lr for base_lr in base_lrs):
            raise ValueError('min_lr must not exceed the initial LR of any parameter group.')

    def make_lr_lambda(base_lr: float) -> Callable[[int], float]:
        min_lr_ratio = min_lr / base_lr if base_lr > 0.0 else 0.0
        start_lr_ratio = warmup_start_lr / base_lr if base_lr > 0.0 else 0.0

        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                warmup_progress = current_step / num_warmup_steps
                return start_lr_ratio + (1.0 - start_lr_ratio) * warmup_progress

            if not use_cosine:
                return 1.0

            progress = (current_step - num_warmup_steps) / max(1, total_steps - num_warmup_steps)
            progress = min(progress, 1.0)
            cosine_ratio = 0.5 * (1.0 + cos(pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_ratio

        return lr_lambda

    return LambdaLR(
        optimizer,
        lr_lambda=[make_lr_lambda(base_lr) for base_lr in base_lrs],
    )
