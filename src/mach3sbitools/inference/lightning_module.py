"""
PyTorch Lightning module for SBI density estimator training.

Wraps an ``sbi`` density estimator in a :class:`~lightning.LightningModule`
to enable multi-GPU and multi-node training via DDP. The typical workflow
is to construct this module and pass it to a
:class:`~lightning.pytorch.trainer.Trainer` alongside an
:class:`SBIDataModule`::

    module = SBILightningModule(density_estimator, config, model_config)
    datamodule = SBIDataModule(dataset, config)
    trainer = L.Trainer(strategy="ddp", accelerator="auto", devices="auto")
    trainer.fit(module, datamodule=datamodule)
"""

import lightning as L
import torch
from sbi.neural_nets.estimators.base import ConditionalEstimator

from mach3sbitools.utils.config import PosteriorConfig, TrainingConfig


class SBILightningModule(L.LightningModule):
    """
    Lightning wrapper for an ``sbi`` density estimator.

    Handles the training and validation steps, EMA-smoothed validation loss
    tracking, and the learning rate scheduler.

    All logged metrics use ``sync_dist=True`` so that early stopping and
    checkpointing behave correctly across ranks in DDP training.

    .. note::

        The density estimator is **not** passed to
        :meth:`~lightning.LightningModule.save_hyperparameters` because
        ``sbi`` networks are not always cleanly serialisable as hyperparameters.
        The ``model_config`` is saved instead so the architecture can be
        reconstructed at load time.
    """

    def __init__(
        self,
        density_estimator: ConditionalEstimator,
        config: TrainingConfig,
        model_config: PosteriorConfig | None = None,
    ):
        """
        :param density_estimator: The ``sbi`` density estimator network to
            train. Must expose a ``.loss(theta, x)`` method that returns a
            per-sample loss tensor of shape ``(batch_size,)``.
        :param config: Training loop hyperparameters. See
            :class:`~mach3sbitools.utils.config.TrainingConfig`.
        :param model_config: Architecture configuration embedded in every
            checkpoint so the model can be reconstructed without supplying
            the config again at load time. ``None`` disables this embedding.
        """
        super().__init__()
        self.model = density_estimator
        self.config = config
        self.model_config = model_config
        self.save_hyperparameters(ignore=["density_estimator"])

        self.ema_val_loss: float = float("inf")

    def forward(self, theta: torch.Tensor, x: torch.Tensor):
        """
        Forward pass — delegates to the density estimator's loss method.

        :param theta: Parameter tensor of shape ``(batch_size, n_params)``.
        :param x: Observable tensor of shape ``(batch_size, n_bins)``.
        :returns: Per-sample loss tensor of shape ``(batch_size,)``.
        """
        return self.model.loss(theta, x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """
        Compute and log the mean training loss for one batch.

        Logged as ``train_loss`` — aggregated per epoch, shown in the
        progress bar, and synchronised across DDP ranks.

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

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """
        Compute and log the mean validation loss for one batch.

        Logged as ``val_loss`` — aggregated per epoch, shown in the
        progress bar, and synchronised across DDP ranks.

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

        The EMA is initialised to ``inf`` on the first epoch so the first
        real validation loss is always accepted as the best. Logged as
        ``ema_val_loss`` and used by
        :class:`~lightning.pytorch.callbacks.EarlyStopping` and
        :class:`~lightning.pytorch.callbacks.ModelCheckpoint`.
        """
        val_loss = self.trainer.callback_metrics.get("val_loss", float("inf"))
        self.ema_val_loss = (
            float(val_loss)
            if self.ema_val_loss == float("inf")
            else self.config.ema_alpha * float(val_loss)
            + (1 - self.config.ema_alpha) * self.ema_val_loss
        )
        self.log("ema_val_loss", self.ema_val_loss, sync_dist=True)

    def on_train_epoch_end(self) -> None:
        """
        Log per-rank GPU memory statistics at the end of each training epoch.

        Records allocated and reserved VRAM in megabytes. Logged with
        ``sync_dist=False`` because GPU memory is a per-rank quantity —
        averaging across ranks is not meaningful.
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            self.log("gpu/allocated_mb", allocated, sync_dist=False)
            self.log("gpu/reserved_mb", reserved, sync_dist=False)

    def configure_optimizers(self):
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

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # Save only the model weights explicitly
        checkpoint["model_state"] = self.model.state_dict()

        # Optionally store config (small, safe)
        if self.model_config is not None:
            checkpoint["model_config"] = self.model_config

        # You can also store epoch for convenience
        checkpoint["epoch"] = self.current_epoch
