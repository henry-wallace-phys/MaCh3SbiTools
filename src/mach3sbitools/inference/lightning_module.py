"""
PyTorch Lightning module for SBI density estimator training.

Wraps an ``sbi`` density estimator in a :class:`~lightning.LightningModule`
to enable multi-GPU and multi-node training via DDP.
"""

from typing import Any, Protocol, runtime_checkable

import lightning as L
import torch
import torch.nn as nn

from mach3sbitools.utils.config import PosteriorConfig, TrainingConfig


@runtime_checkable
class HasLoss(Protocol):
    """
    Minimal protocol for objects implementing a .loss() method.
    """

    def loss(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor: ...


class SBILightningModule(L.LightningModule):
    """
    Lightning wrapper for an ``sbi`` density estimator.

    Handles the training and validation steps, EMA-smoothed validation loss
    tracking, and the learning rate scheduler.

    :ivar model: The density estimator module.
    :vartype model: torch.nn.Module
    :ivar config: Training configuration object.
    :ivar model_config: Optional architecture configuration.
    :ivar ema_val_loss: Exponential Moving Average of the validation loss.
    """

    def __init__(
        self,
        density_estimator: nn.Module,
        config: TrainingConfig,
        model_config: PosteriorConfig | None = None,
    ):
        """
        :param density_estimator: The ``sbi`` density estimator network.
            Must implement a ``.loss(theta, x)`` method.
        :param config: Training loop hyperparameters.
        :param model_config: Architecture configuration embedded in checkpoints.
        """
        super().__init__()

        # Runtime check to ensure the model is valid
        if not hasattr(density_estimator, "loss"):
            raise TypeError(
                f"Model of type {type(density_estimator)} must implement .loss(theta, x)"
            )

        # Typing as Any prevents Mypy from complaining about .loss() not existing
        # on nn.Module, while still allowing .eval(), .train(), etc.
        self.model: Any = density_estimator
        self.config = config
        self.model_config = model_config
        self.save_hyperparameters(ignore=["density_estimator"])

        self.ema_val_loss: float = float("inf")

    def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass — delegates to the density estimator's loss method.

        :param theta: Parameter tensor of shape ``(batch_size, n_params)``.
        :param x: Observable tensor of shape ``(batch_size, n_bins)``.
        :returns: Per-sample loss tensor of shape ``(batch_size,)``.
        """
        return self.model.loss(theta, x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Compute and log the mean training loss for one batch.

        :param batch: Tuple of ``(theta, x)`` tensors.
        :param batch_idx: Index of the current batch within the epoch.
        :returns: Scalar mean loss for this batch.
        """
        theta, x = batch
        loss = self.model.loss(theta, x).mean()
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Compute and log the mean validation loss for one batch.

        :param batch: Tuple of ``(theta, x)`` tensors.
        :param batch_idx: Index of the current batch within the epoch.
        :returns: Scalar mean loss for this batch.
        """
        theta, x = batch
        loss = self.model.loss(theta, x).mean()
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        """
        Update and log the EMA-smoothed validation loss after each epoch.
        """
        val_loss_raw = self.trainer.callback_metrics.get(
            "val_loss", torch.tensor(float("inf"))
        )
        val_loss = float(val_loss_raw)

        if self.ema_val_loss == float("inf"):
            self.ema_val_loss = val_loss
        else:
            alpha = self.config.ema_alpha
            self.ema_val_loss = (alpha * val_loss) + (1 - alpha) * self.ema_val_loss

        self.log("ema_val_loss", self.ema_val_loss, sync_dist=True)

    def on_train_epoch_end(self) -> None:
        """
        Log per-rank GPU memory statistics at the end of each training epoch.
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            self.log("gpu/allocated_mb", allocated, sync_dist=False)
            self.log("gpu/reserved_mb", reserved, sync_dist=False)

    def configure_optimizers(self) -> dict[str, Any]:
        """
        Initialize the Adam optimizer and ReduceLROnPlateau scheduler.

        :returns: A dictionary containing the optimizer and LR scheduler configuration.
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.config.scheduler_patience,
            factor=0.5,
            min_lr=1e-8,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "ema_val_loss",
            },
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """
        Custom checkpoint logic to explicitly save model weights and configuration.

        :param checkpoint: The checkpoint dictionary to be saved.
        """
        checkpoint["model_state"] = self.model.state_dict()

        if self.model_config is not None:
            checkpoint["model_config"] = self.model_config

        checkpoint["epoch"] = self.current_epoch
