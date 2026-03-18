"""
TensorBoard logging wrapper for the SBI training loop.
"""

import torch
from torch.utils.tensorboard import SummaryWriter


class TensorBoardWriter:
    """
    Thin wrapper around :class:`~torch.utils.tensorboard.SummaryWriter`.

    Writes training scalars (losses, learning rates, throughput, GPU stats)
    to a TensorBoard event file each epoch.
    """

    def __init__(self, log_dir: str, device_type: str):
        """
        :param log_dir: Directory for TensorBoard event files.
        :param device_type: PyTorch device type string, e.g. ``"cuda"`` or ``"cpu"``.
        """
        self.writer = SummaryWriter(log_dir=log_dir)
        self.device_type = device_type

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
        total_samples: int,
    ) -> None:
        """
        Write all training scalars for one epoch.

        Records loss curves, per-group learning rates, throughput metrics,
        early-stopping state, and GPU memory statistics.

        :param epoch: Current epoch number (x-axis value).
        :param train_loss: Mean training loss for the epoch.
        :param val_loss: Mean validation loss for the epoch.
        :param ema_val_loss: EMA-smoothed validation loss.
        :param best_val_loss: Best EMA validation loss seen so far.
        :param optimizer: Current optimiser (used to read learning rates).
        :param elapsed: Wall-clock seconds for the epoch.
        :param epochs_no_improve: Current early-stopping counter.
        :param total_samples: Number of training steps in the epoch
            (used for throughput calculation).
        """
        self.writer.add_scalar("loss/train", train_loss, epoch)
        self.writer.add_scalar("loss/val", val_loss, epoch)
        self.writer.add_scalar("loss/val_ema", ema_val_loss, epoch)
        self.writer.add_scalar("loss/best_val_ema", best_val_loss, epoch)
        self.writer.add_scalar("loss/train_val_gap", train_loss - val_loss, epoch)

        for i, pg in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f"lr/group_{i}", pg["lr"], epoch)

        self.writer.add_scalar(
            "throughput/samples_per_sec", total_samples / elapsed, epoch
        )
        self.writer.add_scalar("throughput/epoch_seconds", elapsed, epoch)
        self.writer.add_scalar(
            "early_stopping/epochs_no_improve", epochs_no_improve, epoch
        )

        gpu_stats = self.get_gpu_stats()
        self.writer.add_scalar("gpu/allocated_mb", gpu_stats["allocated_mb"], epoch)
        self.writer.add_scalar("gpu/reserved_mb", gpu_stats["reserved_mb"], epoch)
        self.writer.add_scalar(
            "gpu/max_reserved_mb", gpu_stats["max_reserved_mb"], epoch
        )
        self.writer.add_scalar(
            "gpu/memory_utilization_pct", gpu_stats["memory_utilization_pct"], epoch
        )
        if "sm_utilization_pct" in gpu_stats:
            self.writer.add_scalar(
                "gpu/sm_utilization_pct", gpu_stats["sm_utilization_pct"], epoch
            )
            self.writer.add_scalar(
                "gpu/memory_bandwidth_utilization_pct",
                gpu_stats["memory_bandwidth_utilization_pct"],
                epoch,
            )

    def get_gpu_stats(self) -> dict:
        """
        Return GPU memory and utilisation statistics.

        Returns zeros for all fields when running on CPU or when CUDA is
        unavailable.

        :returns: Dict with keys ``allocated_mb``, ``reserved_mb``,
            ``max_reserved_mb``, ``memory_utilization_pct``, and optionally
            ``sm_utilization_pct`` and ``memory_bandwidth_utilization_pct``
            when CUDA is active.
        """
        if not torch.cuda.is_available() or self.device_type == "cpu":
            return {
                "allocated_mb": 0,
                "reserved_mb": 0,
                "max_reserved_mb": 0,
                "memory_utilization_pct": 0,
            }

        allocated = torch.cuda.memory_allocated(self.device_type) / 1024**2
        reserved = torch.cuda.memory_reserved(self.device_type) / 1024**2
        max_reserved = torch.cuda.max_memory_reserved(self.device_type) / 1024**2
        mem_util = (allocated / reserved * 100) if reserved > 0 else 0

        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "max_reserved_mb": max_reserved,
            "memory_utilization_pct": mem_util,
        }

    def close(self) -> None:
        """Flush and close the underlying :class:`~torch.utils.tensorboard.SummaryWriter`."""
        self.writer.flush()
        self.writer.close()
