import torch
from torch.utils.tensorboard import SummaryWriter


class TensorBoardWriter:
    def __init__(self, log_dir: str, device_type: str):
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

        self.writer.add_scalar(
            "throughput/samples_per_sec", len(self.train_dataset) / elapsed, epoch
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
        """Return GPU memory and SM utilisation stats for TensorBoard monitoring."""
        if not torch.cuda.is_available() or self.device_type == "cpu":
            return {
                "allocated_mb": 0,
                "reserved_mb": 0,
                "max_reserved_mb": 0,
                "memory_utilization_pct": 0,
            }

        allocated = torch.cuda.memory_allocated(self.device) / 1024**2
        reserved = torch.cuda.memory_reserved(self.device) / 1024**2
        max_reserved = torch.cuda.max_memory_reserved(self.device) / 1024**2
        mem_util = (allocated / reserved * 100) if reserved > 0 else 0

        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "max_reserved_mb": max_reserved,
            "memory_utilization_pct": mem_util,
        }

    def close(self):
        self.writer.flush()
        self.writer.close()
