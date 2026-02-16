import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from contextlib import nullcontext
from rich.progress import Progress
from mach3sbitools.utils.config import TrainingConfig
from mach3sbitools.utils.logger import get_logger, create_progress

logger = get_logger(__name__)


class SBITrainer:
    """
    Custom training loop for NPE_C (single-round NPE) with:
      - Pre-loaded TensorDataset (no per-epoch disk reads)
      - Pinned memory + prefetching for CPU→GPU transfers
      - BF16 mixed precision (CUDA only)
      - ReduceLROnPlateau scheduler
      - Gradient clipping
      - Best-model checkpointing
      - Periodic autosave with full resumable state
    """

    def __init__(
        self,
        dataset: TensorDataset,
        config: TrainingConfig,
        device: str,
    ):
        self.device = device
        # nflows' rational quadratic spline does in-place index assignment that
        # requires source/destination dtypes to match — BF16 autocast breaks this.
        # AMP is therefore disabled regardless of the use_amp flag.
        self.use_amp = False

        # TensorBoard writer — None if no log dir provided
        if config.tensorboard_dir is not None:
            tensorboard_dir = Path(config.tensorboard_dir)
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.writer: Optional[SummaryWriter] = SummaryWriter(log_dir=tensorboard_dir)
            logger.info(f"TensorBoard logging → [cyan]{tensorboard_dir}[/]")
            logger.info(f"  Run: [bold]tensorboard --logdir {tensorboard_dir}[/]")
        else:
            self.writer = None

        # Split train / val
        n_val = int(len(dataset) * config.validation_fraction)
        n_train = len(dataset) - n_val
        self.train_dataset, self.val_dataset = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        # Pin memory only when data lives on CPU (enables async DMA transfers)
        data_on_cpu = dataset.tensors[0].device.type == "cpu"
        _workers = config.num_workers if data_on_cpu else 0
        _pin = data_on_cpu
        _prefetch = 2 if data_on_cpu and _workers > 0 else None
        _persistent = data_on_cpu and _workers > 0

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=_workers,
            pin_memory=_pin,
            prefetch_factor=_prefetch,
            persistent_workers=_persistent,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size * 4,  # no grad → can use larger batches
            shuffle=False,
            num_workers=_workers,
            pin_memory=_pin,
            prefetch_factor=_prefetch,
            persistent_workers=_persistent,
        )

        logger.info(
            f"Trainer ready | "
            f"train: [bold]{n_train:,}[/]  val: [bold]{n_val:,}[/] | "
            f"steps/epoch: [bold]{len(self.train_loader):,}[/] | "
            f"AMP: [cyan]{self.use_amp}[/] | "
            f"pin_memory: [cyan]{_pin}[/] | "
            f"num_workers: [cyan]{_workers}[/]"
        )

    def train(
        self,
        density_estimator: nn.Module,
        config: TrainingConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> nn.Module:

        # ── Optimizer / scheduler / scaler ──────────────────────────────
        optimizer = optimizer or torch.optim.Adam(density_estimator.parameters(), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=config.scheduler_patience, factor=0.5, min_lr=1e-8
        )
        scaler = GradScaler(device=self.device, enabled=self.use_amp)

        # ── Resume checkpoint if provided ──────────────────────────────
        start_epoch, best_val_loss, epochs_no_improve = self._resume_if_requested(
            density_estimator, optimizer, scheduler, scaler, config.resume_checkpoint
        )

        density_estimator.to(self.device)
        best_state = None

        # ── Create epoch-level progress ────────────────────────────────
        progress, epoch_task = create_progress(
            show_epoch=config.show_epoch_progress,
            total_epochs=config.max_epochs,
            description="Training",
        )

        with progress:
            for epoch in range(start_epoch, int(config.max_epochs + 1)):
                t0 = time.time()

                # ── Training step ───────────────────────────────────────
                train_loss = self._train_one_epoch(
                    density_estimator,
                    optimizer,
                    scaler,
                    progress=progress if config.show_progress_bar else nullcontext(),
                    epoch=epoch,
                    max_epochs=config.max_epochs,
                )

                # ── Validation step ───────────────────────────────────
                val_loss = self._validate(density_estimator)
                scheduler.step(val_loss)
                elapsed = time.time() - t0

                # ── Logging ──────────────────────────────────────────
                if epoch % config.print_interval == 0:
                    self._log_epoch(epoch, config.max_epochs, train_loss, val_loss, optimizer, elapsed)

                self.add_to_writer(
                    epoch, train_loss, val_loss, best_val_loss, optimizer, elapsed, epochs_no_improve
                )

                # ── Checkpoints ───────────────────────────────────────
                best_state, best_val_loss, epochs_no_improve = self._handle_checkpoints(
                    val_loss, best_val_loss, epochs_no_improve, density_estimator, config.save_path
                )

                if config.save_path and epoch % config.autosave_every == 0:
                    self.save_checkpoint(
                        epoch, density_estimator, optimizer, scheduler, scaler,
                        best_val_loss, epochs_no_improve, config.save_path
                    )

                # ── Early stopping ────────────────────────────────────
                if epochs_no_improve >= config.stop_after_epochs:
                    logger.warning(
                        f"Early stopping at epoch {epoch} (no improvement for {config.stop_after_epochs} epochs)"
                    )
                    break

                # ── Advance epoch progress ─────────────────────────────
                if epoch_task is not None:
                    progress.update(epoch_task, advance=1)

        # ── Restore best model weights ───────────────────────────────
        if best_state is not None:
            density_estimator.load_state_dict(best_state)

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
        return density_estimator


    def _resume_if_requested(self, model, optimizer, scheduler, scaler, resume_checkpoint):
        start_epoch = 1
        best_val_loss = float("inf")
        epochs_no_improve = 0

        if resume_checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
            ckpt = torch.load(resume_checkpoint, map_location="cpu")
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            scaler.load_state_dict(ckpt["scaler_state"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt["best_val_loss"]
            epochs_no_improve = ckpt["epochs_no_improve"]
            logger.info(f"Resumed at epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
        return start_epoch, best_val_loss, epochs_no_improve

    def _train_one_epoch(
        self, model, optimizer, scaler, progress: Progress | nullcontext = nullcontext(), epoch: int = 0, max_epochs: int = 1
    ):
        model.train()
        total_loss = 0.0

        # Create batch-level task inside this method
        batch_task = None
        if isinstance(progress, Progress):
            batch_task = progress.add_task(f"Epoch {epoch}/{max_epochs}", total=len(self.train_loader),
                                        loss=0.0, lr=optimizer.param_groups[0]["lr"])

        for theta, x in self.train_loader:
            theta, x = theta.to(self.device, non_blocking=True), x.to(self.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=self.device.split(":")[0], dtype=torch.bfloat16, enabled=self.use_amp):
                loss = model.loss(theta, x).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.detach()

            if batch_task is not None:
                progress.update(batch_task, advance=1, loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        if batch_task is not None:
            progress.remove_task(batch_task)

        return total_loss / len(self.train_loader)

    def _validate(self, model):
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for theta, x in self.val_loader:
                theta, x = theta.to(self.device, non_blocking=True), x.to(self.device, non_blocking=True)
                with autocast(device_type=self.device.split(":")[0], dtype=torch.bfloat16, enabled=self.use_amp):
                    val_loss += model.loss(theta, x).mean()
        return val_loss / len(self.val_loader)

    def _log_epoch(self, epoch, max_epochs, train_loss, val_loss, optimizer, elapsed):
        logger.info(
            f"Epoch {epoch:4d}/{max_epochs} | "
            f"train: {train_loss:.4f} | val: {val_loss:.4f} | "
            f"lr: {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
        )

    def _handle_checkpoints(self, val_loss, best_val_loss, epochs_no_improve, model, save_path):
        best_state = None
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if save_path: torch.save(best_state, save_path)
        else:
            epochs_no_improve += 1
        return best_state, best_val_loss, epochs_no_improve

    
    def save_checkpoint(self, epoch, density_estimator, optimizer, scheduler, scaler, best_val_loss, epochs_no_improve, save_path):
        ckpt = {
            "epoch": epoch,
            "model_state": {
                k: v.cpu().clone()
                for k, v in density_estimator.state_dict().items()
            },
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "epochs_no_improve": epochs_no_improve,
        }
        ckpt_path = save_path.with_stem(f"{save_path.stem}_epoch{epoch}")
        torch.save(ckpt, ckpt_path)
        logger.info(f"Autosaved checkpoint → [cyan]{ckpt_path}[/]")

    def add_to_writer(self, epoch, train_loss, val_loss, best_val_loss, optimizer, elapsed, epochs_no_improve):
        if self.writer is not None:
            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("loss/val", val_loss, epoch)
            self.writer.add_scalar("loss/best_val", best_val_loss, epoch)
            self.writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
            self.writer.add_scalar("time/epoch_seconds", elapsed, epoch)
            self.writer.add_scalar("early_stopping/epochs_no_improve", epochs_no_improve, epoch)

            gpu_stats = self.get_gpu_stats()
            self.writer.add_scalar("gpu/allocated_mb", gpu_stats["allocated_mb"], epoch)
            self.writer.add_scalar("gpu/reserved_mb", gpu_stats["reserved_mb"], epoch)
            self.writer.add_scalar("gpu/max_reserved_mb", gpu_stats["max_reserved_mb"], epoch)
            self.writer.add_scalar("gpu/utilization_pct", gpu_stats["utilization_pct"], epoch)

    def get_gpu_stats(self) -> dict:
        """
        Return current GPU stats for monitoring.
        """
        
        if not torch.cuda.is_available() or self.device == "cpu":
            return {
                "allocated_mb": 0,
                "reserved_mb": 0,
                "max_reserved_mb": 0,
                "utilization_pct": 0,
            }

        allocated = torch.cuda.memory_allocated(self.device) / 1024**2
        reserved = torch.cuda.memory_reserved(self.device) / 1024**2
        max_reserved = torch.cuda.max_memory_reserved(self.device) / 1024**2
        util = (allocated / max_reserved * 100) if max_reserved > 0 else 0

        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "max_reserved_mb": max_reserved,
            "utilization_pct": util,
        }
