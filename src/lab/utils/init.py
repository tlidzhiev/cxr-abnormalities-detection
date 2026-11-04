import logging
import random
import secrets
import shutil
import string
import subprocess
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from lab.logger import setup_logging

from .io import get_root


def set_worker_seed(worker_id) -> None:
    """
    Set the seed of a dataloader worker.

    See https://pytorch.org/docs/stable/notes/randomness.html

    Parameters
    ----------
    worker_id : int
        Id of the worker (unused, the seed is derived from torch).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_random_seed(seed: int) -> None:
    """
    Set the random seed for training or inference.

    Parameters
    ----------
    seed : int
        Seed value.
    """
    # fix random seeds for reproducibility
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark=True works faster but reproducibility decreases
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def generate_id(length: int = 8) -> str:
    """
    Generate a random base-36 run id for the experiment tracker.

    See https://github.com/wandb/wandb/blob/main/wandb/sdk/lib/runid.py.

    Parameters
    ----------
    length : int, optional
        Length of the id.

    Returns
    -------
    str
        Random string of lowercase letters and digits.
    """
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def log_git_commit_and_patch(log_dir: Path) -> None:
    """
    Save the current git commit and uncommitted changes to the run directory.

    Restore the code with 'git checkout <hash>' and 'git apply git_diff.patch'.
    New files are in the patch only if staged via 'git add'.

    Parameters
    ----------
    log_dir : Path
        Run directory.
    """
    commands = {
        'git_commit.txt': ['git', 'rev-parse', 'HEAD'],
        'git_diff.patch': ['git', 'diff', 'HEAD'],
    }
    for filename, command in commands.items():
        # Bytes are kept as is, so the patch still applies to files with CRLF or non-UTF-8 text.
        result = subprocess.run(command, cwd=get_root(), capture_output=True, check=False)
        if result.returncode == 0:
            (log_dir / filename).write_bytes(result.stdout)


def saving_init(log_dir: Path, cfg: DictConfig) -> Literal['new', 'resume', 'override']:
    """
    Create the run directory and save the config into it.

    On resume the run id is taken from the saved config, so the tracker
    continues the same run.

    Parameters
    ----------
    log_dir : Path
        Run directory.
    cfg : DictConfig
        Project config; 'writer.run_id' is filled in place.

    Returns
    -------
    Literal['new', 'resume', 'override']
        What was done with the run directory (logged after the logging is set up).

    Raises
    ------
    ValueError
        If the run directory holds a run and neither resume nor override is set.
    """
    status = 'new'
    run_id = None
    # Hydra creates the run directory and its own log before the job starts,
    # so a previous run is recognized by its saved config.
    if (log_dir / 'config.yaml').exists():
        if cfg.trainer.resume_from is not None:
            run_id = OmegaConf.load(log_dir / 'config.yaml').writer.run_id
            status = 'resume'
        elif cfg.trainer.override:
            shutil.rmtree(log_dir)
            status = 'override'
        else:
            raise ValueError(
                f"Log directory '{log_dir}' exists. "
                'Change writer.run_name, set trainer.override=True or trainer.resume_from.'
            )
    elif cfg.trainer.resume_from is not None:
        raise ValueError(f"Cannot resume: log directory '{log_dir}' has no run.")

    log_dir.mkdir(exist_ok=True, parents=True)
    OmegaConf.update(cfg, 'writer.run_id', run_id or generate_id(cfg.writer.id_length))
    OmegaConf.save(cfg, log_dir / 'config.yaml')
    log_git_commit_and_patch(log_dir)
    return status


def setup_saving_and_logging(cfg: DictConfig) -> logging.Logger:
    """
    Initialize the run directory, the checkpoint directory and logging.

    Parameters
    ----------
    cfg : DictConfig
        Config of the current experiment.

    Returns
    -------
    logging.Logger
        Logger of the training script.
    """
    log_dir = get_root() / cfg.writer.log_dir
    status = saving_init(log_dir, cfg)
    (get_root() / cfg.trainer.checkpoint_dir).mkdir(exist_ok=True, parents=True)
    setup_logging(log_dir, append=cfg.trainer.resume_from is not None)

    logger = logging.getLogger('train')
    if status == 'resume':
        logger.info(f'Resuming training from run {cfg.writer.run_id}...')
    elif status == 'override':
        logger.info(f"Overriding log directory '{log_dir}'...")
    return logger
