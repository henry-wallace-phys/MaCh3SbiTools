import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset, random_split

from mach3sbitools.utils.config import TrainingConfig
from mach3sbitools.utils.logger import create_progress, get_logger

from .tensorboard_writer import TensorBoardWriter

logger = get_logger()


# ──────────────────────────────
# Helper functions (pure)
# ──────────────────────────────


def _log_epoch(
    epoch, max_epochs, train_loss, val_loss, ema_val_loss, optimizer, elapsed
):
    logger.info(
        f"Epoch {epoch:4d}/{max_epochs} | "
        f"train: {train_loss:.4f} | val: {val_loss:.4f} | ema_val: {ema_val_loss:.4f} | "
        f"lr: {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
    )


def _update_best_state(
    ema_val_loss: float,
    best_val_loss: float,
    epochs_no_improve: int,
    best_state: dict | None,
    model: nn.Module,
) -> tuple[dict | None, float, int]:
    improved = ema_val_loss < best_val_loss

    if improved:
        return (
            {k: v.cpu().clone() for k, v in model.state_dict().items()},
            ema_val_loss,
            0,
        )

    return best_state, best_val_loss, epochs_no_improve + 1


def save_checkpoint(
    epoch: int,
    density_estimator: nn.Module,
    optimizer: torch.optim.Optimizer,
    warmup_scheduler: torch.optim.lr_scheduler.LRScheduler,
    plateau_scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    best_val_loss: float,
    epochs_no_improve: int,
    save_path: Path,
    training_config: TrainingConfig | None = None,
    use_unique_path: bool = True,
) -> None:
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


# ──────────────────────────────
# Trainer
# ──────────────────────────────


class SBITrainer:
    def __init__(self, dataset: TensorDataset, config: TrainingConfig, device: str):
        self.device = device
        self.device_type = device.split(":")[0]
        self.use_amp = False

        self.writer = self._init_tensorboard(config)
        self.train_loader, self.val_loader = self._build_dataloaders(dataset, config)

    # ── Init helpers ─────────────────────────────────────────────

    def _init_tensorboard(self, config: TrainingConfig) -> TensorBoardWriter | None:
        if config.tensorboard_dir is None:
            return None

        tb_dir = Path(config.tensorboard_dir)
        tb_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"TensorBoard logging → [cyan]{tb_dir}[/]")
        logger.info(f"  Run: [bold]tensorboard --logdir {tb_dir}[/]")

        return TensorBoardWriter(tb_dir, self.device_type)

    def _build_dataloaders(self, dataset: TensorDataset, config: TrainingConfig):
        n_val = int(len(dataset) * config.validation_fraction)
        n_train = len(dataset) - n_val

        train_ds, val_ds = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        data_on_cpu = dataset.tensors[0].device.type == "cpu"
        workers = config.num_workers if data_on_cpu else 0

        loader_kwargs = dict(
            num_workers=workers,
            pin_memory=data_on_cpu,
            persistent_workers=data_on_cpu and workers > 0,
            prefetch_factor=2 if data_on_cpu and workers > 0 else None,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size * 4,
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

    # ── Public API ───────────────────────────────────────────────

    def train(
        self,
        density_estimator: nn.Module,
        config: TrainingConfig,
        optimizer: torch.optim.Optimizer | None = None,
        resume_checkpoint: Path | None = None,
    ) -> nn.Module:

        density_estimator = self._maybe_compile(density_estimator, config)
        optimizer = optimizer or torch.optim.Adam(
            density_estimator.parameters(), lr=config.learning_rate
        )

        schedulers = self._build_schedulers(optimizer, config)
        scaler = GradScaler(device=self.device, enabled=self.use_amp)

        start_epoch, best_val_loss, epochs_no_improve = self._resume_if_requested(
            density_estimator, optimizer, *schedulers, scaler, resume_checkpoint
        )

        density_estimator.to(self.device)
        best_state = self._clone_state(density_estimator)

        ema_val_loss = None
        progress, epoch_task = create_progress(
            show_epoch=config.show_epoch_progress,
            total_epochs=config.max_epochs - (start_epoch - 1),
            description=f"Training (epoch {start_epoch}→{config.max_epochs})",
        )

        with progress:
            for epoch in range(start_epoch, config.max_epochs + 1):
                t0 = time.perf_counter()

                train_loss = self._train_one_epoch(
                    density_estimator, optimizer, scaler, config, epoch
                )
                val_loss = self._validate(density_estimator)

                elapsed = time.perf_counter() - t0
                ema_val_loss = (
                    val_loss
                    if ema_val_loss is None
                    else config.ema_alpha * val_loss
                    + (1 - config.ema_alpha) * ema_val_loss
                )

                self._step_schedulers(epoch, ema_val_loss, config, *schedulers)

                if epoch % config.print_interval == 0:
                    _log_epoch(
                        epoch,
                        config.max_epochs,
                        train_loss,
                        val_loss,
                        ema_val_loss,
                        optimizer,
                        elapsed,
                    )

                self._log_tensorboard(
                    epoch,
                    train_loss,
                    val_loss,
                    ema_val_loss,
                    best_val_loss,
                    optimizer,
                    elapsed,
                    epochs_no_improve,
                )

                best_state, best_val_loss, epochs_no_improve = _update_best_state(
                    ema_val_loss,
                    best_val_loss,
                    epochs_no_improve,
                    best_state,
                    density_estimator,
                )

                self._maybe_save(
                    epoch,
                    density_estimator,
                    optimizer,
                    schedulers,
                    scaler,
                    best_val_loss,
                    epochs_no_improve,
                    config,
                )

                if epochs_no_improve >= config.stop_after_epochs:
                    logger.warning(f"Early stopping at epoch {epoch}")
                    break

                if epoch_task:
                    progress.update(epoch_task, advance=1)

        if best_state:
            density_estimator.load_state_dict(best_state)

        if self.writer:
            self.writer.close()

        logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
        return density_estimator

    # ── Training internals ───────────────────────────────────────

    def _train_one_epoch(self, model, optimizer, scaler, config, epoch):
        model.train()
        total_loss = 0.0

        for theta, x in self.train_loader:
            theta = theta.to(self.device, non_blocking=True)
            x = x.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(
                device_type=self.device_type, dtype=torch.bfloat16, enabled=self.use_amp
            ):
                loss = model.loss(theta, x).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.detach().item()

        return total_loss / len(self.train_loader)

    def _validate(self, model):
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

    # ── Utility helpers ─────────────────────────────────────────

    def _clone_state(self, model):
        return {k: v.cpu().clone() for k, v in model.state_dict().items()}

    def _maybe_compile(self, model, config):
        if not config.compile:
            return model
        try:
            logger.info("torch.compile() enabled")
            return torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            logger.warning(f"torch.compile() unavailable: {e}")
            return model

    def _build_schedulers(self, optimizer, config):
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=config.warmup_epochs,
        )

        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=config.scheduler_patience,
            factor=0.5,
            min_lr=1e-8,
            threshold=1e-3,
            threshold_mode="abs",
        )

        return warmup, plateau

    def _step_schedulers(self, epoch, ema_val_loss, config, warmup, plateau):
        if epoch <= config.warmup_epochs:
            warmup.step()
        else:
            plateau.step(ema_val_loss)

    def _log_tensorboard(
        self,
        epoch,
        train_loss,
        val_loss,
        ema_val_loss,
        best_val_loss,
        optimizer,
        elapsed,
        epochs_no_improve,
    ):
        if not self.writer:
            return

        self.writer.add_to_writer(
            epoch,
            train_loss,
            val_loss,
            ema_val_loss,
            best_val_loss,
            optimizer,
            elapsed,
            epochs_no_improve,
        )

    def _maybe_save(
        self,
        epoch,
        model,
        optimizer,
        schedulers,
        scaler,
        best_val_loss,
        epochs_no_improve,
        config,
    ):
        if not config.save_path:
            return

        warmup, plateau = schedulers

        if epochs_no_improve == 0:
            save_checkpoint(
                epoch,
                model,
                optimizer,
                warmup,
                plateau,
                scaler,
                best_val_loss,
                epochs_no_improve,
                config.save_path,
                training_config=config,
                use_unique_path=False,
            )

        if epoch % config.autosave_every == 0:
            save_checkpoint(
                epoch,
                model,
                optimizer,
                warmup,
                plateau,
                scaler,
                best_val_loss,
                epochs_no_improve,
                config.save_path,
                training_config=config,
            )

    def _resume_if_requested(
        self,
        model,
        optimizer,
        warmup,
        plateau,
        scaler,
        resume_checkpoint,
    ):
        if not resume_checkpoint:
            return 1, float("inf"), 0

        ckpt = torch.load(resume_checkpoint, map_location="cpu")

        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        warmup.load_state_dict(ckpt["warmup_scheduler_state"])
        plateau.load_state_dict(ckpt["plateau_scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        start_epoch = ckpt["epoch"] + 1

        logger.info(f"Resumed from {resume_checkpoint} | epoch {start_epoch}")

        return start_epoch, ckpt["best_val_loss"], ckpt["epochs_no_improve"]
