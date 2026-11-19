import logging
import math
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from lab.dataset.utils import inf_loop
from lab.logger.base import BaseWriter
from lab.loss.base import BaseLoss
from lab.metrics.base import BaseMetric
from lab.metrics.tracker import MetricTracker
from lab.model.base import BaseModel
from lab.utils.io import get_root

from .config import TrainerConfig
from .runner import BaseRunner
from .utils import EarlyStopping, StepTimer

logger = logging.getLogger(__name__)


def get_ema_multi_avg_fn_with_linear_warmup(
    decay: float = 0.999,
    warmup_steps: int = 0,
    update_every: int = 1,
):
    """
    EMA update with the decay linearly increasing during the first 'warmup_steps' optimizer
    steps; the EMA is updated every 'update_every' steps, so the number of updates is
    multiplied by it to get the number of steps.
    """

    @torch.no_grad()
    def ema_update(
        ema_param_list: tuple[torch.Tensor, ...] | list[torch.Tensor],
        current_param_list: tuple[torch.Tensor, ...] | list[torch.Tensor],
        num_averaged: torch.Tensor | int,
    ) -> None:
        d = decay
        if warmup_steps > 0:
            n = num_averaged.item() if isinstance(num_averaged, torch.Tensor) else num_averaged
            n *= update_every
            if n < warmup_steps:
                d = decay * (n / warmup_steps)

        if torch.is_floating_point(ema_param_list[0]) or torch.is_complex(ema_param_list[0]):
            torch._foreach_lerp_(ema_param_list, current_param_list, 1 - d)
        else:
            for p_ema, p_model in zip(ema_param_list, current_param_list, strict=True):
                p_ema.copy_(p_model)

    return ema_update


class BaseTrainer(BaseRunner):
    """
    Base class for all trainers.

    Provides step management, metric tracking, checkpointing and evaluation.
    """

    def __init__(
        self,
        config: DictConfig,
        device: str,
        dataloaders: dict[str, DataLoader],
        model: BaseModel,
        criterion: BaseLoss,
        metrics: dict[str, list[BaseMetric]],
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        writer: BaseWriter,
        batch_transforms: dict[str, dict[str, nn.Sequential]],
    ) -> None:
        """
        Parameters
        ----------
        config : DictConfig
            Experiment config.
        device : str
            Device for tensors and model.
        dataloaders : dict[str, DataLoader]
            Dataloader of each partition.
        model : BaseModel
            PyTorch model.
        criterion : BaseLoss
            Loss function.
        metrics : dict[str, list[BaseMetric]]
            Metrics for training ('train') and inference ('inference').
        optimizer : Optimizer
            Optimizer of the model.
        lr_scheduler : LRScheduler
            Learning rate scheduler.
        writer : BaseWriter
            Experiment tracker.
        batch_transforms : dict[str, dict[str, nn.Sequential]]
            Transforms applied to the whole batch, by tensor name.
        """
        self.config = config
        self.cfg_trainer = TrainerConfig.from_config(config.trainer)
        cfg = self.cfg_trainer

        super().__init__(
            device=device,
            model=model,
            metrics=metrics,
            batch_transforms=batch_transforms,
            device_tensors=cfg.device_tensors,
            use_bfloat16=cfg.use_bfloat16,
        )
        self.timer = StepTimer(enabled=cfg.debug, device=device)
        # clip_grad_norm_ with an infinite threshold only measures the norm
        self.max_grad_norm = cfg.max_grad_norm or math.inf
        if self.max_grad_norm != math.inf:
            logger.info(f'Gradient clipping enabled with max_grad_norm={self.max_grad_norm}')

        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.ema: AveragedModel | None = None
        if cfg.ema is not None:
            self.ema_update_every = cfg.ema.update_every
            self.ema = AveragedModel(
                self._ema_module(),
                multi_avg_fn=get_ema_multi_avg_fn_with_linear_warmup(
                    decay=cfg.ema.decay,
                    warmup_steps=cfg.ema.warmup_steps,
                    update_every=cfg.ema.update_every,
                ),
                use_buffers=True,
            )
            logger.info(f'EMA enabled: {cfg.ema}')

        # Define metrics
        self.train_metric_tracker = MetricTracker(
            *self.criterion.loss_names,
            'grad_norm',
            *self.timer.metric_names,
            *[m.name for m in self.metrics['train']],
        )
        self.eval_metric_tracker = MetricTracker(
            *self.criterion.loss_names,
            *[m.name for m in self.metrics['inference']],
        )

        # Define dataloaders
        self.train_loader = dataloaders['train']
        # Keep one iterator so evaluation windows do not restart the training sampler
        self._train_iter = iter(inf_loop(self.train_loader))
        # 'test' has no ground truth: it is only for inference
        self.eval_dataloaders = {k: v for k, v in dataloaders.items() if k not in ('train', 'test')}

        self.global_step = 0  # optimizer steps done
        # Reference point of the 'steps_per_sec' of the train logs
        self._last_log_step = 0
        self._last_log_time = time.perf_counter()

        self.writer = writer
        self.early_stopping = EarlyStopping(monitor=cfg.monitor, patience=cfg.early_stop)
        self.checkpoint_dir = get_root() / cfg.checkpoint_dir

        if cfg.resume_from is not None:
            self._resume_from(self.checkpoint_dir / cfg.resume_from)
        if cfg.from_pretrained is not None:
            self.load_weights(cfg.from_pretrained)

        self.compile_model(cfg.compile)

    def train(self) -> None:
        """
        Wrapper around training process to save model on keyboard interrupt and always
        close the writer.
        """
        try:
            self._train_process()
        except KeyboardInterrupt:
            logger.info('Saving model on keyboard interrupt.')
            self._save_checkpoint('interrupted')
            raise
        finally:
            self.writer.close()

    def _train_process(self) -> None:
        """
        Train for a fixed number of optimizer steps, evaluate after each window, and monitor
        performance (early stopping and the best checkpoint).
        """
        cfg = self.cfg_trainer
        while self.global_step < cfg.total_steps:
            # Windows end on multiples of 'eval_every_steps', so after resuming from a step
            # inside a window, the evaluations and checkpoints stay on the same steps
            steps_to_eval = cfg.eval_every_steps - self.global_step % cfg.eval_every_steps
            window_steps = min(steps_to_eval, cfg.total_steps - self.global_step)
            train_logs = self._train_window(window_steps)
            logs = {
                'step': self.global_step,
                'data_passes': self.global_step * cfg.grad_accum_steps / len(self.train_loader),
                **train_logs,
                **self._evaluate(),
            }
            logger.info(f'\nMetrics:\n{yaml.dump(logs)}')

            if self.early_stopping.update(logs):
                self._save_checkpoint('best')

            if not cfg.only_best and self.global_step % cfg.checkpoint_every_steps == 0:
                self._save_checkpoint()
            if self.early_stopping.should_stop:
                logger.info(
                    f"Validation performance didn't improve for "
                    f'{self.early_stopping.patience} '
                    'evaluations. Training stops.'
                )
                break

        # Always keep the final state (unless it was just saved by the periodic checkpoint)
        if cfg.only_best or self.global_step % cfg.checkpoint_every_steps != 0:
            self._save_checkpoint()

    def _train_window(self, steps: int) -> dict[str, Any]:
        """
        Train for a window of 'steps' optimizer steps and return the train metrics averaged
        over the last logging interval of the window (about 'log_step' steps, not the whole
        window).
        """
        self.model.train()
        self.train_metric_tracker.reset()

        start_step = self.global_step
        self.writer.set_step(start_step)
        self._last_log_step, self._last_log_time = start_step, time.perf_counter()

        last_train_metrics: dict[str, Any] = {}
        for _ in tqdm(range(steps), desc=f'Train Steps {start_step}–{start_step + steps}'):
            batch = self._train_step()
            if batch is None:  # skipped on OOM
                continue
            if self.global_step % self.cfg_trainer.log_step == 0:
                last_train_metrics = self._log_train(batch)

        # Steps after the last logging point have not been averaged yet
        if self.global_step != self._last_log_step:
            last_train_metrics = self.train_metric_tracker.result()

        return {f'train_{name}': value for name, value in last_train_metrics.items()}

    def _train_step(self) -> dict[str, Any] | None:
        """
        Do an optimizer step and the bookkeeping after it: step counter, EMA, timing and the
        periodic checkpoint.

        Returns
        -------
        dict[str, Any] or None
            The last micro-batch after 'process_batch', or None if the step was skipped
            because of OOM (with 'skip_oom').
        """
        self.timer.start_step()
        try:
            batch = self._accumulate_and_step()
        except torch.cuda.OutOfMemoryError:
            if not self.cfg_trainer.skip_oom:
                raise
            # The unfetched batches of the step go to the next steps
            logger.warning('OOM on batch. Skipping optimizer step.')
            self._zero_grad()  # Drop the partially accumulated gradients
            torch.cuda.empty_cache()  # Free some memory
            return None

        self.global_step += 1
        if self.ema is not None and self.global_step % self.ema_update_every == 0:
            self.ema.update_parameters(self._ema_module())

        self.timer.end_step()
        self.timer.record(self.train_metric_tracker)
        return batch

    def _accumulate_and_step(self) -> dict[str, Any]:
        """
        Accumulate the gradients over the micro-batches of the step, do an optimizer step,
        and return the last micro-batch after 'process_batch'.

        The micro-batches are fetched one at a time, so loading the next one overlaps with
        the asynchronous (CUDA) computation of the current one.
        """
        self._zero_grad()
        batch: dict[str, Any] = {}
        for _ in range(self.cfg_trainer.grad_accum_steps):
            # Rebinding releases the processed batch (it holds device tensors and outputs)
            with self.timer.data():
                batch = next(self._train_iter)
            batch = self.process_batch(batch, 'train')
        self._optimizer_step()
        return batch

    def _log_train(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Log the train metrics averaged since the previous logging point (and reset them),
        the learning rate, speed and the batch. Returns the logged metrics.
        """
        now = time.perf_counter()
        self.writer.set_step(self.global_step)
        self.writer.add_scalars(self._learning_rates())
        self.writer.add_scalar(
            'steps_per_sec',
            (self.global_step - self._last_log_step) / (now - self._last_log_time),
        )
        self._last_log_step, self._last_log_time = self.global_step, now

        metrics = self.train_metric_tracker.result()
        self.train_metric_tracker.reset()
        self.writer.add_scalars(metrics)
        if self.timer.enabled:
            data_time, step_time = metrics['data_time'], metrics['step_time']
            self.writer.add_scalar('data_wait_ratio', data_time / (data_time + step_time))
        self._log_batch(batch, 'train')
        return metrics

    def _evaluate(self) -> dict[str, Any]:
        """
        Evaluate on all eval dataloaders (with EMA weights if available); the logs are
        prefixed with the dataloader name.
        """
        logs: dict[str, Any] = {}
        with self._ema_scope():
            for part, dataloader in self.eval_dataloaders.items():
                part_logs = self._evaluate_part(part, dataloader)
                logs.update({f'{part}_{name}': value for name, value in part_logs.items()})
        return logs

    @contextmanager
    def _ema_scope(self) -> Generator[None]:
        """
        Temporarily swap 'self.model' for the EMA-averaged model (no-op without EMA).

        '.eval()' is not needed here: '_evaluate_part' calls it on whatever
        'self.model' points to.
        """
        if self.ema is None:
            yield
            return

        logger.info('Switching to EMA model...')
        original_model = self.model
        self.model = self.ema.module
        try:
            yield
        finally:
            self.model = original_model

    @torch.no_grad()
    def _evaluate_part(self, part: str, dataloader: DataLoader) -> dict[str, Any]:
        """
        Evaluate the model on a partition and return its logs.
        """
        self.model.eval()
        self.eval_metric_tracker.reset()

        batch = None
        start_time = time.perf_counter()
        for batch in tqdm(
            dataloader,
            desc=f'{part.capitalize()} Step {self.global_step}',
            total=len(dataloader),
        ):
            batch = self.process_batch(batch, part)

        for met in self.metrics['inference']:
            # All the values of the metric (e.g. mAP and the AP of each class)
            for name, value in met.compute_all().items():
                self.eval_metric_tracker.update(name, value)
            met.reset()
        elapsed = time.perf_counter() - start_time

        self.writer.set_step(self.global_step, part)
        logs = self.eval_metric_tracker.result()
        self.writer.add_scalars(logs)
        self.writer.add_scalar('steps_per_sec', len(dataloader) / elapsed)
        if batch is not None:
            self._log_batch(batch, part)
        return logs

    def _ema_module(self) -> nn.Module:
        """
        The module averaged by EMA (the whole model by default).
        """
        return self.model

    def _zero_grad(self) -> None:
        """
        Reset the gradients of all optimizers.
        """
        self.optimizer.zero_grad(set_to_none=True)

    def _learning_rates(self) -> dict[str, float]:
        """
        Current learning rates to log, by name.
        """
        return {'lr': float(self.lr_scheduler.get_last_lr()[0])}

    def _optim_state_dict(self) -> dict[str, Any]:
        """
        State of the optimizers and LR schedulers (and anything else trainable besides the
        model) to save in the checkpoint.
        """
        return {
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
        }

    def _load_optim_state_dict(self, checkpoint: dict[str, Any]) -> None:
        """
        Restore the state saved by '_optim_state_dict' from the checkpoint.
        """
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    def backward(self, loss: torch.Tensor) -> None:
        """
        Backpropagate the loss of a micro-batch, accumulating its gradients.

        The loss is divided by the number of micro-batches, so the accumulated gradient is
        the mean over them.
        """
        (loss / self.cfg_trainer.grad_accum_steps).backward()

    def _optimizer_step(self) -> None:
        """
        Clip the accumulated gradients, step the optimizer and the lr_scheduler.
        """
        # Returns the total norm before clipping
        grad_norm = clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.train_metric_tracker.update('grad_norm', grad_norm.detach())
        self.optimizer.step()
        self.lr_scheduler.step()

    def process_batch(self, batch: dict[str, Any], part: str) -> dict[str, Any]:
        """
        Run a batch through the model, compute loss and metrics, and backpropagate the loss
        with 'self.backward' (in train mode; the optimizer step is done in '_accumulate_and_step').

        The criterion must aggregate all losses into a single 'loss' key. The metrics are
        tracked with 'self.train_metric_tracker' for the 'train' partition and with
        'self.eval_metric_tracker' otherwise.

        Parameters
        ----------
        batch : dict[str, Any]
            Dict-based batch from the dataloader.
        part : str
            Partition name ('train' in train mode).

        Returns
        -------
        dict[str, Any]
            The (transformed) batch with model outputs and losses.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement process_batch method.')

    def _log_batch(self, batch: dict[str, Any], part: str) -> None:
        """
        Log data from a batch (after 'process_batch') to the experiment tracker via
        self.writer.add_*; the writer step is already set to the current step and 'part'.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement _log_batch method.')

    def _save_checkpoint(self, kind: Literal['step', 'best', 'interrupted'] = 'step') -> None:
        """
        Save the training state of the current step: model, optimizer, lr_scheduler, EMA,
        monitor.
        """
        filenames = {
            'step': f'checkpoint-step-{self.global_step}.pth',
            'best': 'model-best.pth',
            'interrupted': 'checkpoint-interrupted.pth',
        }
        path = self.checkpoint_dir / filenames[kind]
        logger.info(f'Saving {kind} checkpoint: {path}...')
        state = {
            'arch': type(self.model).__name__,
            'step': self.global_step,
            'state_dict': self.model.state_dict(),
            **self._optim_state_dict(),
            'early_stopping': self.early_stopping.state_dict(),
            'config': OmegaConf.to_container(self.config, resolve=True),
        }
        if self.ema is not None:
            state['ema'] = self.ema.state_dict()

        # Write to a temporary file first, so an interruption never leaves a truncated checkpoint
        tmp_path = path.with_name(path.name + '.tmp')
        torch.save(state, tmp_path)
        tmp_path.replace(path)
        self.writer.add_checkpoint(str(path), str(self.checkpoint_dir.parent))

    def _resume_from(self, path: Path) -> None:
        """
        Restore the training state saved by '_save_checkpoint' in place.

        Raises
        ------
        RuntimeError
            If the model, optimizer or lr_scheduler config differs from the checkpoint one.
        """
        checkpoint = self.load_checkpoint(path)

        saved, current = checkpoint['config'], OmegaConf.to_container(self.config, resolve=True)
        assert isinstance(current, dict)
        if saved['model'] != current['model']:
            raise RuntimeError(
                'Architecture configuration given in the config file is different '
                'from that of the checkpoint.'
            )
        if (
            saved['optimizer'] != current['optimizer']
            or saved['lr_scheduler'] != current['lr_scheduler']
        ):
            raise RuntimeError(
                'Optimizer or lr_scheduler given in the config file is different '
                'from that of the checkpoint. Cannot resume optimizer and scheduler states.'
            )

        self.model.load_state_dict(checkpoint['state_dict'])
        self._load_optim_state_dict(checkpoint)
        if self.ema is not None and 'ema' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema'])
        self.early_stopping.load_state_dict(checkpoint.get('early_stopping', {}))
        self.global_step = checkpoint.get('step', checkpoint.get('global_step', 0))
        logger.info(f'Resuming from optimizer step {self.global_step}')
