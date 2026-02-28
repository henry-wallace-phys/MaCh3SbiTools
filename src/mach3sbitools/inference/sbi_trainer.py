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

try:
    from pynvml import nvml as pynvml
    pynvml.nvmlInit()
    _PYNVML_AVAILABLE = True
except Exception:
    _PYNVML_AVAILABLE = False

logger = get_logger(__name__)


class SBITrainer:
    """
    Custom training loop for NPE_C (single-round NPE) with:
      - Pre-loaded TensorDataset (no per-epoch disk reads)
      - Pinned memory + prefetching for CPU→GPU transfers (or full GPU dataset)
      - BF16 mixed precision (CUDA only)
      - ReduceLROnPlateau scheduler on EMA-smoothed val loss
      - LR warmup for the first `warmup_epochs` epochs
      - Gradient clipping
      - Best-model checkpointing
      - Periodic autosave with full resumable state
      - torch.compile() for faster execution
      - Rich TensorBoard metrics (loss, LR, gradients, GPU, throughput)
    """

    def __init__(
        self,
        dataset: TensorDataset,
        config: TrainingConfig,
        device: str,
    ):
        self.device = device
        self.device_type = device.split(":")[0]
        self.use_amp = False

        if config.tensorboard_dir is not None:
            tensorboard_dir = Path(config.tensorboard_dir)
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.writer: Optional[SummaryWriter] = SummaryWriter(log_dir=tensorboard_dir)
            logger.info(f"TensorBoard logging → [cyan]{tensorboard_dir}[/]")
            logger.info(f"  Run: [bold]tensorboard --logdir {tensorboard_dir}[/]")
        else:
            self.writer = None

        n_val = int(len(dataset) * config.validation_fraction)
        n_train = len(dataset) - n_val
        self.train_dataset, self.val_dataset = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

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
            batch_size=config.batch_size * 4,
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
        resume_checkpoint: Optional[Path] = None,
    ) -> nn.Module:
        """
        Train the density estimator.

        Args:
            density_estimator: The network to train.
            config:            Hyperparameters and I/O settings.
            optimizer:         Optional pre-built optimizer; Adam is created if omitted.
            resume_checkpoint: Path to a checkpoint saved by save_checkpoint().
                               Restores model weights, optimizer, scheduler, scaler,
                               epoch counter, best val loss, and early-stop counter.
            posterior_config:  Architecture config to embed in every checkpoint so
                               the network can be fully reconstructed without user
                               input on resume. Pass the same PosteriorConfig used
                               to build the density estimator.
        """

        if config.compile:
            try:
                density_estimator = torch.compile(density_estimator, mode='reduce-overhead')
                logger.info("torch.compile() enabled")
            except Exception as e:
                logger.warning(f"torch.compile() unavailable, skipping: {e}")
        else:
            logger.info("torch.compile() disabled (set config.compile=True to enable)")

        optimizer = optimizer or torch.optim.Adam(density_estimator.parameters(), lr=config.learning_rate)

        warmup_epochs = config.warmup_epochs
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=config.scheduler_patience,
            factor=0.5,
            min_lr=1e-8,
            threshold=1e-3,
            threshold_mode="abs",
        )

        scaler = GradScaler(device=self.device, enabled=self.use_amp)

        ema_alpha = config.ema_alpha
        ema_val_loss: Optional[float] = None

        start_epoch, best_val_loss, epochs_no_improve = self._resume_if_requested(
            density_estimator, optimizer, warmup_scheduler, plateau_scheduler, scaler, resume_checkpoint
        )

        density_estimator.to(self.device)

        best_state = {k: v.cpu().clone() for k, v in density_estimator.state_dict().items()}

        remaining_epochs = config.max_epochs - (start_epoch - 1)
        progress, epoch_task = create_progress(
            show_epoch=config.show_epoch_progress,
            total_epochs=remaining_epochs,
            description=f"Training (epoch {start_epoch}→{config.max_epochs})",
        )

        with progress:
            for epoch in range(start_epoch, int(config.max_epochs + 1)):
                t0 = time.perf_counter()

                train_loss = self._train_one_epoch(
                    density_estimator, optimizer, scaler,
                    progress=progress if config.show_progress_bar else nullcontext(),
                    epoch=epoch,
                    max_epochs=config.max_epochs,
                )

                val_loss = self._validate(density_estimator)
                elapsed = time.perf_counter() - t0

                if ema_val_loss is None:
                    ema_val_loss = val_loss
                else:
                    ema_val_loss = ema_alpha * val_loss + (1 - ema_alpha) * ema_val_loss

                if epoch <= warmup_epochs:
                    warmup_scheduler.step()
                else:
                    plateau_scheduler.step(ema_val_loss)

                if epoch % config.print_interval == 0:
                    self._log_epoch(
                        epoch, config.max_epochs, train_loss, val_loss, ema_val_loss, optimizer, elapsed
                    )

                self.add_to_writer(
                    epoch, train_loss, val_loss, ema_val_loss, best_val_loss,
                    optimizer, elapsed, epochs_no_improve,
                )

                best_state, best_val_loss, epochs_no_improve = self._handle_best_state(
                    ema_val_loss, best_val_loss, epochs_no_improve, best_state, density_estimator,
                )
                
                if epochs_no_improve == 0 and config.save_path is not None: 
                    # Save a checkpoint of the best model whenever we get a new best val loss.
                    self.save_checkpoint(
                        epoch, density_estimator, optimizer,
                        warmup_scheduler, plateau_scheduler, scaler,
                        best_val_loss, epochs_no_improve, config.save_path,
                        training_config=config,
                        use_unique_path=False,  # overwrite previous best
                    )

                # Also
                if config.save_path and epoch % config.autosave_every == 0:
                    self.save_checkpoint(
                        epoch, density_estimator, optimizer,
                        warmup_scheduler, plateau_scheduler, scaler,
                        best_val_loss, epochs_no_improve, config.save_path,
                        training_config=config,
                    )

                if epochs_no_improve >= config.stop_after_epochs:
                    logger.warning(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement for {config.stop_after_epochs} epochs)"
                    )
                    break

                if epoch_task is not None:
                    progress.update(epoch_task, advance=1)

        if best_state is not None:
            density_estimator.load_state_dict(best_state)

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        self.save_checkpoint(
            epoch, density_estimator, optimizer,
            warmup_scheduler, plateau_scheduler, scaler,
            best_val_loss, epochs_no_improve, config.save_path,
            training_config=config,
            use_unique_path=False,  # final checkpoint with best model
        )

        logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
        return density_estimator

    # ── Private helpers ───────────────────────────────────────────────────────

    def _resume_if_requested(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        warmup_scheduler: torch.optim.lr_scheduler.LRScheduler,
        plateau_scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: GradScaler,
        resume_checkpoint: Optional[Path],
    ) -> tuple[int, float, int]:
        if resume_checkpoint is None:
            return 1, float("inf"), 0

        resume_checkpoint = Path(resume_checkpoint)
        if not resume_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_checkpoint}")

        logger.info(f"Resuming from checkpoint: [cyan]{resume_checkpoint}[/]")
        ckpt = torch.load(resume_checkpoint, map_location="cpu")

        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        warmup_scheduler.load_state_dict(ckpt["warmup_scheduler_state"])
        plateau_scheduler.load_state_dict(ckpt["plateau_scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        epochs_no_improve = ckpt["epochs_no_improve"]

        logger.info(
            f"Resumed | start epoch: [bold]{start_epoch}[/] | "
            f"best val loss: [bold]{best_val_loss:.4f}[/] | "
            f"epochs without improvement: [bold]{epochs_no_improve}[/]"
        )
        return start_epoch, best_val_loss, epochs_no_improve

    def _train_one_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: GradScaler,
        progress: Progress | nullcontext = nullcontext(),
        epoch: int = 0,
        max_epochs: int = 1,
    ) -> float:
        model.train()
        total_loss = 0.0

        batch_task = None
        if isinstance(progress, Progress):
            batch_task = progress.add_task(
                f"Epoch {epoch}/{max_epochs}",
                total=len(self.train_loader),
                loss=0.0,
                lr=optimizer.param_groups[0]["lr"],
            )

        for theta, x in self.train_loader:
            theta = theta.to(self.device, non_blocking=True)
            x = x.to(self.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=self.device_type, dtype=torch.bfloat16, enabled=self.use_amp):
                loss = model.loss(theta, x).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            # FIX: .item() forces a float — avoids accumulating a GPU tensor
            # which would serialise the pipeline with a sync on every batch.
            total_loss += loss.detach().item()

            if batch_task is not None:
                progress.update(batch_task, advance=1, loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        if batch_task is not None:
            progress.remove_task(batch_task)

        return total_loss / len(self.train_loader)

    def _validate(self, model: nn.Module) -> float:
        model.eval()
        total_loss = 0.0
        with torch.inference_mode():
            for theta, x in self.val_loader:
                theta = theta.to(self.device, non_blocking=True)
                x = x.to(self.device, non_blocking=True)
                with autocast(device_type=self.device_type, dtype=torch.bfloat16, enabled=self.use_amp):
                    # FIX: .item() forces a sync once per batch and returns a plain
                    # Python float, so total_loss never becomes a GPU tensor.
                    # Without this, ema_val_loss stays on-device and
                    # plateau_scheduler.step() silently syncs every epoch.
                    total_loss += model.loss(theta, x).mean().item()
        return total_loss / len(self.val_loader)

    def _log_epoch(self, epoch, max_epochs, train_loss, val_loss, ema_val_loss, optimizer, elapsed):
        logger.info(
            f"Epoch {epoch:4d}/{max_epochs} | "
            f"train: {train_loss:.4f} | val: {val_loss:.4f} | ema_val: {ema_val_loss:.4f} | "
            f"lr: {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
        )

    def _handle_best_state(
        self,
        ema_val_loss: float,
        best_val_loss: float,
        epochs_no_improve: int,
        best_state: Optional[dict],
        model: nn.Module,
    ) -> tuple[Optional[dict], float, int]:
        if ema_val_loss < best_val_loss:
            best_val_loss = ema_val_loss
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
        return best_state, best_val_loss, epochs_no_improve

    def save_checkpoint(
        self,
        epoch: int,
        density_estimator: nn.Module,
        optimizer: torch.optim.Optimizer,
        warmup_scheduler: torch.optim.lr_scheduler.LRScheduler,
        plateau_scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: GradScaler,
        best_val_loss: float,
        epochs_no_improve: int,
        save_path: Path,
        training_config: TrainingConfig|None = None,
        use_unique_path: bool = True,
    ) -> None:
        ckpt = {
            "epoch": epoch,
            "model_state": {k: v.cpu().clone() for k, v in density_estimator.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "warmup_scheduler_state": warmup_scheduler.state_dict(),
            "plateau_scheduler_state": plateau_scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "epochs_no_improve": epochs_no_improve,
            "training_config": training_config,
        }
        if use_unique_path:
            ckpt_path = save_path.with_stem(f"{save_path.stem}_epoch{epoch}")
        else:
            ckpt_path = save_path
        torch.save(ckpt, ckpt_path)
        logger.info(f"Autosaved checkpoint → [cyan]{ckpt_path}[/]")

    def add_to_writer(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        ema_val_loss: float,
        best_val_loss: float,
        optimizer: torch.optim.Optimizer,
        elapsed: float,
        epochs_no_improve: int,
    ) -> None:
        if self.writer is None:
            return

        self.writer.add_scalar("loss/train", train_loss, epoch)
        self.writer.add_scalar("loss/val", val_loss, epoch)
        self.writer.add_scalar("loss/val_ema", ema_val_loss, epoch)
        self.writer.add_scalar("loss/best_val_ema", best_val_loss, epoch)
        self.writer.add_scalar("loss/train_val_gap", train_loss - val_loss, epoch)

        for i, pg in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f"lr/group_{i}", pg["lr"], epoch)

        self.writer.add_scalar("throughput/samples_per_sec", len(self.train_dataset) / elapsed, epoch)
        self.writer.add_scalar("throughput/epoch_seconds", elapsed, epoch)

        self.writer.add_scalar("early_stopping/epochs_no_improve", epochs_no_improve, epoch)

        gpu_stats = self.get_gpu_stats()
        self.writer.add_scalar("gpu/allocated_mb", gpu_stats["allocated_mb"], epoch)
        self.writer.add_scalar("gpu/reserved_mb", gpu_stats["reserved_mb"], epoch)
        self.writer.add_scalar("gpu/max_reserved_mb", gpu_stats["max_reserved_mb"], epoch)
        self.writer.add_scalar("gpu/memory_utilization_pct", gpu_stats["memory_utilization_pct"], epoch)
        if "sm_utilization_pct" in gpu_stats:
            self.writer.add_scalar("gpu/sm_utilization_pct", gpu_stats["sm_utilization_pct"], epoch)
            self.writer.add_scalar(
                "gpu/memory_bandwidth_utilization_pct", gpu_stats["memory_bandwidth_utilization_pct"], epoch
            )

    def get_gpu_stats(self) -> dict:
        """Return GPU memory and SM utilisation stats for TensorBoard monitoring."""
        if not torch.cuda.is_available() or self.device_type == "cpu":
            return {"allocated_mb": 0, "reserved_mb": 0, "max_reserved_mb": 0, "memory_utilization_pct": 0}

        allocated = torch.cuda.memory_allocated(self.device) / 1024**2
        reserved = torch.cuda.memory_reserved(self.device) / 1024**2
        max_reserved = torch.cuda.max_memory_reserved(self.device) / 1024**2
        mem_util = (allocated / reserved * 100) if reserved > 0 else 0

        stats = {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "max_reserved_mb": max_reserved,
            "memory_utilization_pct": mem_util,
        }

        if _PYNVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                stats["sm_utilization_pct"] = util.gpu
                stats["memory_bandwidth_utilization_pct"] = util.memory
            except Exception:
                pass

        return stats