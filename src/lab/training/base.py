import logging
import math
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Literal

import torch
import yaml
from omegaconf import DictConfig
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

from .checkpoint import Checkpointer, Monitor
from .timing import StepTimer
from .utils import autocast, compile_model, get_amp_dtype

logger = logging.getLogger(__name__)


def get_ema_multi_avg_fn_with_linear_warmup(
    decay: float = 0.999, warmup_steps: int = 0, update_every: int = 1
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


class BaseTrainer:
    """
    Base class for all trainers.

    Provides step management, metric tracking, checkpointing and evaluation.
    """

    def _validate_args(self) -> None:
        """Validate trainer configuration values and option combinations."""
        cfg_trainer = self.cfg_trainer
        grad_accum_steps = cfg_trainer.get('grad_accum_steps', 1)
        if grad_accum_steps < 1:
            raise ValueError(f'trainer.grad_accum_steps must be positive, got {grad_accum_steps}.')
        ema_cfg = cfg_trainer.get('ema')
        if ema_cfg is not None and ema_cfg.get('update_every', 1) < 1:
            raise ValueError('trainer.ema.update_every must be >= 1.')
        if cfg_trainer.total_steps < 1:
            raise ValueError('trainer.total_steps must be positive.')
        if cfg_trainer.eval_every_steps < 1:
            raise ValueError('trainer.eval_every_steps must be positive.')
        if cfg_trainer.checkpoint_every_steps < 1:
            raise ValueError('trainer.checkpoint_every_steps must be positive.')
        if cfg_trainer.get('only_best', False) and cfg_trainer.get('monitor', 'off') == 'off':
            raise ValueError("trainer.only_best requires trainer.monitor to be set (not 'off').")
        if cfg_trainer.get('resume_from') is not None and cfg_trainer.get('from_pretrained'):
            raise ValueError(
                'trainer.resume_from and trainer.from_pretrained are mutually exclusive.'
            )

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
        skip_oom: bool = False,
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
        skip_oom : bool, optional
            Skip optimizer steps with the OutOfMemory error.
        """
        self.config = config
        self.cfg_trainer = self.config.trainer
        self._validate_args()

        self.device = device
        self.skip_oom = skip_oom

        self.log_step = self.cfg_trainer.get('log_step', 50)
        self.timer = StepTimer(enabled=self.cfg_trainer.get('debug', False), device=device)
        # clip_grad_norm_ with an infinite threshold only measures the norm
        self.max_grad_norm = self.cfg_trainer.get('max_grad_norm') or math.inf
        # Mixed precision: bfloat16 needs no loss scaling, float16 does
        self.amp_dtype = get_amp_dtype(self.cfg_trainer.get('amp'))
        self.scaler = torch.amp.GradScaler(
            torch.device(device).type,
            enabled=self.amp_dtype == torch.float16,
        )
        if self.amp_dtype is not None:
            logger.info(f'AMP enabled with dtype={self.amp_dtype}')
        # Number of batches (micro-batches) whose gradients are summed before an optimizer step
        self.grad_accum_steps = self.cfg_trainer.get('grad_accum_steps', 1)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_transforms = batch_transforms

        # Initialize EMA if configured
        self.ema: AveragedModel | None = None
        if (ema_cfg := self.cfg_trainer.get('ema')) is not None:
            ema_decay = ema_cfg.get('decay', 0.999)
            ema_warmup_steps = ema_cfg.get('warmup_steps', 0)
            self.ema_update_every = ema_cfg.get('update_every', 1)
            self.ema = AveragedModel(
                model,
                multi_avg_fn=get_ema_multi_avg_fn_with_linear_warmup(
                    decay=ema_decay,
                    warmup_steps=ema_warmup_steps,
                    update_every=self.ema_update_every,
                ),
                use_buffers=True,
            )
            logger.info(
                f'EMA enabled with decay={ema_decay}, warmup_steps={ema_warmup_steps}, '
                f'update_every={self.ema_update_every}'
            )

        # Define metrics
        self.metrics = metrics
        self.train_metric_tracker = MetricTracker(
            *self.criterion.loss_names,
            'grad_norm',
            *self.timer.metric_names,
            *[n for m in self.metrics['train'] for n in m.result_names],
        )
        self.eval_metric_tracker = MetricTracker(
            *self.criterion.loss_names,
            *[n for m in self.metrics['inference'] for n in m.result_names],
        )

        # Define dataloaders
        self.train_loader = dataloaders['train']
        # Keep one iterator so evaluation windows do not restart the training sampler
        self._train_iter = iter(inf_loop(self.train_loader))
        self.eval_every_steps = self.cfg_trainer.eval_every_steps
        self.eval_dataloaders = {k: v for k, v in dataloaders.items() if k != 'train'}

        # Define optimizer step limits
        self.global_step = 0
        self.total_steps = self.cfg_trainer.total_steps

        # Configuration to monitor model performance and save best
        self.checkpoint_every_steps = self.cfg_trainer.checkpoint_every_steps
        self.only_best = self.cfg_trainer.get('only_best', False)  # Save only model-best.pth
        self.monitor = Monitor.from_string(
            self.cfg_trainer.get('monitor', 'off'),  # Format: "mnt_mode mnt_metric"
            early_stop=self.cfg_trainer.get('early_stop'),
        )
        # Setup visualization writer instance
        self.writer = writer

        # Define checkpointer and init everything if required
        self.checkpointer = Checkpointer(
            checkpoint_dir=get_root() / self.cfg_trainer.checkpoint_dir,
            config=config,
            device=device,
            monitor=self.monitor,
            writer=writer,
        )

        if (resume_from := self.cfg_trainer.get('resume_from')) is not None:
            self.global_step = self.checkpointer.restore(
                path=self.checkpointer.directory / resume_from,
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                ema=self.ema,
                scaler=self.scaler,
            )
            logger.info(f'Resuming from optimizer step {self.global_step}')

        if (pretrained_path := self.cfg_trainer.get('from_pretrained')) is not None:
            logger.info(f'Loading model weights from: {pretrained_path}...')
            self.model = Checkpointer.from_pretrained(
                model=self.model,
                path=pretrained_path,
                device=self.device,
            )

        if (compile_cfg := self.cfg_trainer.get('compile')) is not None:
            compile_model(self.model, compile_cfg)
            logger.info(f'Compiling model with options: {compile_cfg}')

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
        while self.global_step < self.total_steps:
            window_steps = min(self.eval_every_steps, self.total_steps - self.global_step)
            train_logs = self._train_window(window_steps)
            logs = {
                'step': self.global_step,
                'data_passes': self.global_step * self.grad_accum_steps / len(self.train_loader),
                **train_logs,
                **self._evaluate(),
            }
            logger.info(f'\nMetrics:\n{yaml.dump(logs)}')

            if self.monitor.update(logs):
                self._save_checkpoint('best')
            if self.monitor.should_stop:
                logger.info(
                    f"Validation performance didn't improve for {self.monitor.early_stop} "
                    'evaluations. Training stops.'
                )
                break

        # Always keep the final state (unless it was just saved by the periodic checkpoint)
        if self.only_best or self.global_step % self.checkpoint_every_steps != 0:
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

        last_train_metrics: dict[str, Any] = {}
        last_log_step = start_step
        last_log_time = time.perf_counter()
        for _ in tqdm(range(steps), desc=f'Train Steps {start_step}–{start_step + steps}'):
            self.timer.start_step()
            try:
                batch = self._train_step()
            except torch.cuda.OutOfMemoryError:
                if not self.skip_oom:
                    raise
                # The unfetched batches of the step go to the next steps
                logger.warning('OOM on batch. Skipping optimizer step.')
                self.optimizer.zero_grad()  # Drop the partially accumulated gradients
                torch.cuda.empty_cache()  # Free some memory
                continue

            self.global_step += 1
            if self.ema is not None and self.global_step % self.ema_update_every == 0:
                self.ema.update_parameters(self.model)

            self.timer.end_step()
            self.timer.record(self.train_metric_tracker)
            if not self.only_best and self.global_step % self.checkpoint_every_steps == 0:
                self._save_checkpoint()

            # Log current results
            if self.global_step % self.log_step == 0:
                self.writer.set_step(self.global_step)
                self.writer.add_scalar('lr', float(self.lr_scheduler.get_last_lr()[0]))
                now = time.perf_counter()
                self.writer.add_scalar(
                    'steps_per_sec', (self.global_step - last_log_step) / (now - last_log_time)
                )
                last_log_step, last_log_time = self.global_step, now
                # Keep recent train metrics between logging points.
                last_train_metrics = self.train_metric_tracker.result()
                self.train_metric_tracker.reset()
                self.writer.add_scalars(last_train_metrics)
                if self.timer.enabled:
                    data_time = last_train_metrics['data_time']
                    step_time = last_train_metrics['step_time']
                    self.writer.add_scalar('data_wait_ratio', data_time / (data_time + step_time))
                self._log_batch(batch, 'train')

        # Steps after the last logging point have not been averaged yet
        if self.global_step != last_log_step:
            last_train_metrics = self.train_metric_tracker.result()

        return {f'train_{name}': value for name, value in last_train_metrics.items()}

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
            dataloader, desc=f'{part.capitalize()} Step {self.global_step}', total=len(dataloader)
        ):
            batch = self.process_batch(batch, part)

        # Metrics are accumulated over all batches, so batch sizes are weighted correctly
        for met in self.metrics['inference']:
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

    def _train_step(self) -> dict[str, Any]:
        """
        Accumulate the gradients over the micro-batches of the step, do an optimizer step,
        and return the last micro-batch after 'process_batch'.

        The micro-batches are fetched one at a time, so loading the next one overlaps with
        the asynchronous (CUDA) computation of the current one.
        """
        self.optimizer.zero_grad(set_to_none=True)
        batch: dict[str, Any] = {}
        for _ in range(self.grad_accum_steps):
            # Rebinding releases the processed batch (it holds device tensors and outputs)
            with self.timer.data():
                batch = next(self._train_iter)
            batch = self.process_batch(batch, 'train')
        self._optimizer_step()
        return batch

    def backward(self, loss: torch.Tensor) -> None:
        """
        Backpropagate the loss of a micro-batch, accumulating its gradients.

        The loss is divided by the number of micro-batches, so the accumulated gradient is
        the mean over them; loss scaling is a no-op unless AMP is float16.
        """
        self.scaler.scale(loss / self.grad_accum_steps).backward()

    def _optimizer_step(self) -> None:
        """
        Clip the accumulated gradients, step the optimizer and the lr_scheduler.
        """
        # Clip the true (unscaled) gradients
        self.scaler.unscale_(self.optimizer)
        # Returns the total norm before clipping
        grad_norm = clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.train_metric_tracker.update('grad_norm', grad_norm.detach())
        # Skips the step if float16 gradients overflowed
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.lr_scheduler.step()

    def autocast(self) -> torch.autocast:
        """
        Autocast context for the forward pass and loss (a no-op without AMP).
        """
        return autocast(self.device, self.amp_dtype)

    def process_batch(self, batch: dict[str, Any], part: str) -> dict[str, Any]:
        """
        Run a batch through the model, compute loss and metrics, and backpropagate the loss
        with 'self.backward' (in train mode; the optimizer step is done in '_train_step').

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
        Save a checkpoint of the current step ('model-best.pth' for 'best').
        """
        action = {
            'step': 'checkpoint',
            'best': 'current best',
            'interrupted': 'interrupted checkpoint',
        }
        step = self.global_step
        logger.info(f'Saving {action[kind]}: {self.checkpointer.path(step, kind)}...')
        self.checkpointer.save(
            step=step,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            ema=self.ema,
            scaler=self.scaler,
            kind=kind,
        )
