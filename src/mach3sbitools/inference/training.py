"""
Custom SBI training loop.

Implements a high-performance training loop for ``sbi`` density estimators
with linear warm-up, :class:`~torch.optim.lr_scheduler.ReduceLROnPlateau`
scheduling, EMA-based early stopping, AMP support, and periodic checkpointing.
"""

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, random_split

from mach3sbitools.exceptions import (
    DensityEstimatorError,
    OptimizerNotSpecified,
    SBITrainingException,
    ScalarNotSpecified,
)
from mach3sbitools.utils.config import TrainingConfig
from mach3sbitools.utils.logger import create_progress, get_logger

from .tensorboard_writer import TensorBoardWriter

logger = get_logger()


def save_checkpoint(
    epoch: int,
    density_estimator: nn.Module,
    optimizer: torch.optim.Optimizer,
    warmup_scheduler: LinearLR,
    plateau_scheduler: ReduceLROnPlateau,
    scaler: GradScaler,
    best_val_loss: float,
    epochs_no_improve: int,
    save_path: Path,
    training_config: TrainingConfig | None = None,
    use_unique_path: bool = True,
) -> None:
    """
    Serialise a full training checkpoint to disk.

    The checkpoint dict contains model weights, optimiser state, both
    scheduler states, the gradient scaler, and training metadata — enough
    to resume training exactly.

    :param epoch: Current epoch number (written into the filename when
        *use_unique_path* is ``True``).
    :param density_estimator: The density estimator module to save.
    :param optimizer: Current optimiser.
    :param warmup_scheduler: Linear warm-up scheduler.
    :param plateau_scheduler: ReduceLROnPlateau scheduler.
    :param scaler: AMP gradient scaler.
    :param best_val_loss: Best EMA validation loss seen so far.
    :param epochs_no_improve: Current early-stopping counter.
    :param save_path: Base file path. The epoch number is appended when
        *use_unique_path* is ``True``.
    :param training_config: Optionally embed the :class:`~mach3sbitools.utils.TrainingConfig`
        in the checkpoint for provenance.
    :param use_unique_path: If ``True`` (default), append ``_epoch{N}`` to
        the stem of *save_path* to avoid overwriting previous checkpoints.
    """
    ckpt = {
        "epoch": epoch,
        "model_state": {
            k: v.cpu().clone() for k, v in density_estimator.state_dict().items()
        },
        "optimizer_state": optimizer.state_dict(),
        "warmup_scheduler_state": warmup_scheduler.state_dict(),
        "plateau_scheduler_state": plateau_scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_val_loss": best_val_loss,
        "epochs_no_improve": epochs_no_improve,
        "training_config": training_config,
    }

    ckpt_path = (
        save_path.with_stem(f"{save_path.stem}_epoch{epoch}")
        if use_unique_path
        else save_path
    )
    torch.save(ckpt, ckpt_path)
    logger.debug(f"Autosaved checkpoint → [cyan]{ckpt_path}[/]")


class SBITrainer:
    """
    Training loop for ``sbi`` density estimators.

    Handles data splitting, DataLoader construction, optimiser and scheduler
    setup, AMP, gradient clipping, EMA-based early stopping, TensorBoard
    logging, and checkpointing.

    Typical usage via :class:`~mach3sbitools.inference.InferenceHandler`::

        trainer = SBITrainer(dataset, config, device)
        best_model = trainer.train(density_estimator)
    """

    def __init__(self, dataset: TensorDataset, config: TrainingConfig, device: str):
        """
        Initialise the trainer and build DataLoaders.

        :param dataset: Pre-loaded :class:`~torch.utils.data.TensorDataset`
            of ``(theta, x)`` pairs.
        :param config: Training hyperparameters.
            See :class:`~mach3sbitools.utils.TrainingConfig`.
        :param device: PyTorch device string (e.g. ``"cuda"`` or ``"cpu"``).
        """
        self.device = device
        self.device_type = device.split(":")[0]
        self.use_amp = False
        self.config = config

        self.optimizer: torch.optim.Optimizer | None = None
        self.warmup: LinearLR | None = None
        self.plateau: ReduceLROnPlateau | None = None
        self.scaler: GradScaler | None = None

        self.best_val_loss: float = float("inf")
        self.best_state: dict[str, Any] | None = None
        self.epochs_no_improve: int = 0
        self.ema_val_loss: float = float("inf")

        self.writer = self._init_tensorboard()
        self.train_loader, self.val_loader = self._build_dataloaders(dataset)

    # ── Init helpers ──────────────────────────────────────────────────────────

    def _init_tensorboard(self) -> TensorBoardWriter | None:
        """
        Initialise TensorBoard logging if a log directory is configured.

        :returns: A :class:`~mach3sbitools.inference.tensorboard_writer.TensorBoardWriter`,
            or ``None`` if ``config.tensorboard_dir`` is unset.
        """
        if self.config.tensorboard_dir is None:
            return None

        tb_dir = Path(self.config.tensorboard_dir)
        tb_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"TensorBoard logging → [cyan]{tb_dir}[/]")
        logger.info(f"  Run: [bold]tensorboard --logdir {tb_dir}[/]")
        return TensorBoardWriter(str(tb_dir), self.device_type)

    def _build_dataloaders(
        self, dataset: TensorDataset
    ) -> tuple[DataLoader, DataLoader]:
        """
        Split *dataset* into train/val subsets and build DataLoaders.

        Uses ``config.validation_fraction`` for the split, pinned memory when
        data is on CPU, and persistent workers when ``num_workers > 0``.

        :param dataset: Source dataset.
        :returns: Tuple of ``(train_loader, val_loader)``.
        """
        n_val = int(len(dataset) * self.config.validation_fraction)
        n_train = len(dataset) - n_val

        train_ds, val_ds = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        data_on_cpu = dataset.tensors[0].device.type == "cpu"
        workers = self.config.num_workers if data_on_cpu else 0

        loader_kwargs: dict[str, Any] = dict(
            num_workers=workers,
            pin_memory=data_on_cpu,
            persistent_workers=data_on_cpu and workers > 0,
            prefetch_factor=2 if data_on_cpu and workers > 0 else None,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size * 4,
            shuffle=False,
            **loader_kwargs,
        )

        logger.info(
            f"Trainer ready | train: [bold]{n_train:,}[/]  val: [bold]{n_val:,}[/] | "
            f"steps/epoch: [bold]{len(train_loader):,}[/] | "
            f"AMP: [cyan]{self.use_amp}[/] | pin_memory: [cyan]{data_on_cpu}[/] | "
            f"num_workers: [cyan]{workers}[/]"
        )
        return train_loader, val_loader

    # ── Public API ────────────────────────────────────────────────────────────

    def train(
        self,
        density_estimator: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        resume_checkpoint: Path | None = None,
    ) -> nn.Module:
        """
        Run the full training loop.

        Trains for up to ``config.max_epochs`` epochs, applying EMA early
        stopping and saving periodic checkpoints. Returns the model restored
        to its best-validation-loss state.

        :param density_estimator: The network to train (built by ``sbi``).
        :param optimizer: Optional pre-built optimiser. Defaults to
            :class:`~torch.optim.Adam` with ``config.learning_rate``.
        :param resume_checkpoint: Path to a checkpoint file from which to
            restore model weights, optimiser state, and scheduler state.
        :returns: The density estimator at its best validation loss.
        :raises DensityEstimatorError: If *density_estimator* is ``None``.
        """
        density_estimator = self._maybe_compile(density_estimator)

        if density_estimator is None:
            raise DensityEstimatorError("No density estimator found")

        self.optimizer = optimizer or torch.optim.Adam(
            density_estimator.parameters(), lr=self.config.learning_rate
        )
        self.warmup, self.plateau = self._build_schedulers()
        self.scaler = GradScaler(device=self.device, enabled=self.use_amp)

        start_epoch = self._resume_if_requested(density_estimator, resume_checkpoint)

        density_estimator.to(self.device)
        self.best_state = self._clone_state(density_estimator)
        self.ema_val_loss = float("inf")

        progress, epoch_task = create_progress(
            show_epoch=self.config.show_epoch_progress,
            total_epochs=self.config.max_epochs - (start_epoch - 1),
            description=f"Training (epoch {start_epoch}→{self.config.max_epochs})",
        )

        with progress:
            for epoch in range(start_epoch, self.config.max_epochs + 1):
                t0 = time.perf_counter()

                train_loss = self._train_one_epoch(density_estimator)
                val_loss = self._validate(density_estimator)
                elapsed = time.perf_counter() - t0

                self.ema_val_loss = (
                    val_loss
                    if self.ema_val_loss is None
                    else self.config.ema_alpha * val_loss
                    + (1 - self.config.ema_alpha) * self.ema_val_loss
                )

                self._step_schedulers(epoch)

                if epoch % self.config.print_interval == 0:
                    self._log_epoch(epoch, train_loss, val_loss, elapsed)

                self._log_tensorboard(epoch, train_loss, val_loss, elapsed)
                self._update_best_state(density_estimator)
                self._maybe_save(epoch, density_estimator)

                if self.epochs_no_improve >= self.config.stop_after_epochs:
                    logger.warning(f"Early stopping at epoch {epoch}")
                    break

                if epoch_task and not isinstance(progress, nullcontext):
                    progress.update(task_id=epoch_task, advance=1)

        if self.best_state:
            density_estimator.load_state_dict(self.best_state)

        if self.writer:
            self.writer.close()

        logger.info(f"Training complete. Best val loss: {self.best_val_loss:.4f}")
        return density_estimator

    # ── Training internals ────────────────────────────────────────────────────

    def _train_one_epoch(self, model: nn.Module) -> float:
        """
        Run one full pass over the training set.

        :param model: The density estimator.
        :returns: Mean training loss for the epoch.
        :raises OptimizerNotSpecified: If the optimiser has not been set.
        :raises ScalarNotSpecified: If the AMP scaler has not been set.
        """
        model.train()
        total_loss = 0.0

        if self.optimizer is None:
            raise OptimizerNotSpecified("Optimizer not provided")
        if self.scaler is None:
            raise ScalarNotSpecified("Scaler not provided")

        for theta, x in self.train_loader:
            theta = theta.to(self.device, non_blocking=True)
            x = x.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(
                device_type=self.device_type, dtype=torch.bfloat16, enabled=self.use_amp
            ):
                loss = model.loss(theta, x).mean()

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.detach().item()

        return total_loss / len(self.train_loader)

    def _validate(self, model: nn.Module) -> float:
        """
        Evaluate the model on the validation set.

        :param model: The density estimator.
        :returns: Mean validation loss for the epoch.
        """
        model.eval()
        total_loss = 0.0

        with torch.inference_mode():
            for theta, x in self.val_loader:
                theta = theta.to(self.device, non_blocking=True)
                x = x.to(self.device, non_blocking=True)

                with autocast(
                    device_type=self.device_type,
                    dtype=torch.bfloat16,
                    enabled=self.use_amp,
                ):
                    total_loss += model.loss(theta, x).mean().item()

        return total_loss / len(self.val_loader)

    # ── State management ──────────────────────────────────────────────────────

    def _set_best_state(self, model: nn.Module) -> None:
        """Record *model* as the new best and reset the no-improvement counter."""
        self.best_val_loss = self.ema_val_loss
        self.best_state = self._clone_state(model)
        self.epochs_no_improve = 0

    def _update_best_state(self, model: nn.Module) -> None:
        """Update best state if the current EMA validation loss is lower."""
        if self.best_val_loss is None:
            self._set_best_state(model)
        elif self.ema_val_loss < self.best_val_loss:
            self._set_best_state(model)
        else:
            self.epochs_no_improve += 1

    def _resume_if_requested(
        self, model: nn.Module, resume_checkpoint: Path | None
    ) -> int:
        """
        Load checkpoint state into all components and return the start epoch.

        :param model: The density estimator to restore weights into.
        :param resume_checkpoint: Path to checkpoint, or ``None`` to start fresh.
        :returns: Epoch number to start training from.
        :raises OptimizerNotSpecified: If the optimiser is not initialised.
        :raises SBITrainingException: If schedulers are not initialised.
        """
        if not resume_checkpoint:
            return 1

        if self.optimizer is None:
            raise OptimizerNotSpecified("Optimizer not provided")
        if self.warmup is None or self.plateau is None or self.scaler is None:
            raise SBITrainingException("Schedulers not provided")

        ckpt = torch.load(resume_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.warmup.load_state_dict(ckpt["warmup_scheduler_state"])
        self.plateau.load_state_dict(ckpt["plateau_scheduler_state"])
        self.scaler.load_state_dict(ckpt["scaler_state"])
        self.best_val_loss = ckpt["best_val_loss"]
        self.epochs_no_improve = ckpt["epochs_no_improve"]

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        start_epoch = cast(int, ckpt["epoch"] + 1)
        logger.info(f"Resumed from {resume_checkpoint} | epoch {start_epoch}")
        return start_epoch

    # ── Scheduler helpers ─────────────────────────────────────────────────────

    def _build_schedulers(self) -> tuple[LinearLR, ReduceLROnPlateau]:
        """
        Construct the warm-up and plateau learning-rate schedulers.

        :returns: Tuple of ``(warmup_scheduler, plateau_scheduler)``.
        :raises OptimizerNotSpecified: If the optimiser is not initialised.
        """
        if self.optimizer is None:
            raise OptimizerNotSpecified("Optimizer not provided")

        warmup = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.config.warmup_epochs,
        )
        plateau = ReduceLROnPlateau(
            self.optimizer,
            patience=self.config.scheduler_patience,
            factor=0.5,
            min_lr=1e-8,
            threshold=1e-3,
            threshold_mode="abs",
        )
        return warmup, plateau

    def _step_schedulers(self, epoch: int) -> None:
        """
        Advance the appropriate scheduler for the current epoch.

        Uses the warm-up scheduler during the warm-up phase, then hands over
        to the plateau scheduler.

        :param epoch: Current epoch number.
        :raises SBITrainingException: If schedulers are not initialised.
        """
        if self.warmup is None or self.plateau is None:
            raise SBITrainingException("Schedulers not provided")

        if epoch <= self.config.warmup_epochs:
            self.warmup.step()
        else:
            self.plateau.step(self.ema_val_loss)

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_epoch(
        self, epoch: int, train_loss: float, val_loss: float, elapsed: float
    ) -> None:
        """Log a one-line training summary for *epoch*."""
        if self.optimizer is None:
            raise OptimizerNotSpecified("Optimizer not provided")

        logger.info(
            f"Epoch {epoch:4d}/{self.config.max_epochs} | "
            f"train: {train_loss:.4f} | val: {val_loss:.4f} | ema_val: {self.ema_val_loss:.4f} | "
            f"lr: {self.optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
        )

    def _log_tensorboard(
        self, epoch: int, train_loss: float, val_loss: float, elapsed: float
    ) -> None:
        """Write scalars to TensorBoard for *epoch* (no-op if writer is absent)."""
        if not self.writer:
            return
        if self.optimizer is None:
            raise OptimizerNotSpecified("Optimizer not provided")

        self.writer.add_to_writer(
            epoch,
            train_loss,
            val_loss,
            self.ema_val_loss,
            self.best_val_loss,
            self.optimizer,
            elapsed,
            self.epochs_no_improve,
            len(self.train_loader),
        )

    # ── Checkpoint saving ─────────────────────────────────────────────────────

    def _maybe_save(self, epoch: int, model: nn.Module) -> None:
        """
        Save checkpoints according to the configured schedule.

        Saves a best-model checkpoint whenever validation improves
        (``epochs_no_improve == 0``) and a periodic autosave every
        ``config.autosave_every`` epochs.

        :param epoch: Current epoch number.
        :param model: Current density estimator state.
        """
        if not self.config.save_path:
            return
        if self.optimizer is None:
            raise OptimizerNotSpecified("Optimizer not provided")
        if self.warmup is None or self.plateau is None or self.scaler is None:
            raise SBITrainingException("Schedulers not provided")

        if self.epochs_no_improve == 0:
            save_checkpoint(
                epoch,
                model,
                self.optimizer,
                self.warmup,
                self.plateau,
                self.scaler,
                self.best_val_loss,
                self.epochs_no_improve,
                self.config.save_path,
                training_config=self.config,
                use_unique_path=False,
            )

        if epoch % self.config.autosave_every == 0:
            save_checkpoint(
                epoch,
                model,
                self.optimizer,
                self.warmup,
                self.plateau,
                self.scaler,
                self.best_val_loss,
                self.epochs_no_improve,
                self.config.save_path,
                training_config=self.config,
            )

    # ── Static utilities ──────────────────────────────────────────────────────

    @staticmethod
    def _clone_state(model: nn.Module) -> dict[str, Any]:
        """Return a CPU copy of *model*'s state dict."""
        return {k: v.cpu().clone() for k, v in model.state_dict().items()}

    def _maybe_compile(self, model: nn.Module) -> Any:
        """
        Optionally compile *model* with ``torch.compile``.

        Falls back to the original model if compilation is unavailable.

        :param model: The model to (maybe) compile.
        :returns: The compiled model, or the original if compilation is
            disabled or fails.
        """
        if not self.config.compile:
            return model
        try:
            logger.info("torch.compile() enabled")
            return torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            logger.warning(f"torch.compile() unavailable: {e}")
            return model
