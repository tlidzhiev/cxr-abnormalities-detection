import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from lab.dataset.utils import get_dataloaders
from lab.training.trainer import Trainer
from lab.utils.init import set_random_seed, setup_saving_and_logging
from lab.utils.io import get_root
from lab.utils.optim import get_lr_scheduler


@hydra.main(
    version_base='1.3',
    config_path=str(get_root() / 'src/lab/configs'),
    config_name='train',
)
def main(config: DictConfig) -> None:
    """
    Train the model: instantiate everything from the config and run Trainer.

    Parameters
    ----------
    config : DictConfig
        Hydra experiment config.
    """
    set_random_seed(config.trainer.seed)
    logger = setup_saving_and_logging(config)

    project_config = OmegaConf.to_container(config, resolve=True)
    writer = instantiate(config.writer, project_config)

    logger.info(f'Config:\n{OmegaConf.to_yaml(config, resolve=True)}')
    if config.trainer.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = config.trainer.device
    logger.info(f'Using device: {device}')

    dataloaders, batch_transforms = get_dataloaders(config, device)

    model = instantiate(config.model).to(device)
    logger.info(f'Model:\n{model}')

    # Hydra turns dict arguments into configs, so the optimizer is created as a partial and
    # called with the parameter groups (dicts) afterwards
    optimizer = instantiate(config.optimizer, _partial_=True)(
        params=model.get_param_groups(base_lr=config.optimizer.lr),
    )
    logger.info(f'Optimizer:\n{optimizer}')
    lr_scheduler = get_lr_scheduler(
        config=config,
        optimizer=optimizer,
        total_steps=config.trainer.total_steps,
    )
    logger.info(f'LR Scheduler: {lr_scheduler}')

    criterion = instantiate(config.criterion).to(device)
    logger.info(f'Criterion: {criterion}')
    metrics = instantiate(config.metrics)
    logger.info(f'Metrics: {metrics}')

    trainer = Trainer(
        config=config,
        device=device,
        dataloaders=dataloaders,
        model=model,
        criterion=criterion,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        writer=writer,
        skip_oom=config.trainer.get('skip_oom', False),
        batch_transforms=batch_transforms,
    )
    trainer.train()


if __name__ == '__main__':
    main()
