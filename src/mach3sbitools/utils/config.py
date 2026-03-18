"""
Configuration dataclasses for model architecture and training.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    """
    Configuration for the SBI training loop.

    :param save_path: Directory to write model checkpoints. Set to ``None``
        to disable checkpointing.
    :param batch_size: Number of samples per training batch.
    :param learning_rate: Initial learning rate for the Adam optimiser.
    :param max_epochs: Hard upper limit on training epochs.
    :param stop_after_epochs: Stop training if the EMA validation loss has not
        improved for this many consecutive epochs.
    :param scheduler_patience: Epochs without improvement before
        :class:`~torch.optim.lr_scheduler.ReduceLROnPlateau` halves the
        learning rate.
    :param validation_fraction: Fraction of data held out for validation.
    :param num_workers: Number of DataLoader worker processes.
    :param autosave_every: Save a periodic checkpoint every *N* epochs,
        in addition to best-model saves.
    :param resume_checkpoint: Path to a checkpoint from which to resume
        training. ``None`` starts from scratch.
    :param use_amp: Enable automatic mixed precision. May not improve
        performance on all hardware.
    :param print_interval: Log a training summary every *N* epochs.
    :param show_progress_bar: Show a global epoch progress bar. Recommended
        for interactive / Jupyter use.
    :param show_epoch_progress: Show a per-epoch step progress bar.
        Recommended for interactive / Jupyter use.
    :param tensorboard_dir: Directory for TensorBoard event files.
        ``None`` disables TensorBoard logging.
    :param warmup_epochs: Number of epochs for linear LR warm-up from 1% to
        100% of ``learning_rate``.
    :param ema_alpha: Smoothing factor for the EMA of validation loss used
        in early stopping. Smaller values are smoother.
    :param compile: Compile the model with ``torch.compile``. Can reduce
        per-step time on supported hardware but increases startup time.
        Not recommended in most cases.
    """

    save_path: Path | None = None
    batch_size: int = 2048
    learning_rate: float = 5e-4
    max_epochs: int = 500
    stop_after_epochs: int = 100
    scheduler_patience: int = 20
    validation_fraction: float = 0.1
    num_workers: int = 1
    autosave_every: int = 10
    resume_checkpoint: Path | None = None
    use_amp: bool = False
    print_interval: int = 10
    show_progress_bar: bool = False
    show_epoch_progress: bool = False
    tensorboard_dir: Path | None = None
    warmup_epochs: int = 50
    ema_alpha: float = 0.05
    compile: bool = False


@dataclass
class PosteriorConfig:
    """
    Configuration for the NPE density estimator architecture.

    :param model: Density estimator type. One of ``"maf"`` (Masked
        Autoregressive Flow) or ``"nse"`` (Neural Spline Flow).
    :param hidden_features: Number of hidden units per layer.
    :param num_transforms: Number of autoregressive transforms (MAF only).
    :param dropout_probability: Dropout probability applied during training.
    :param num_blocks: Number of residual blocks.
    :param num_bins: Number of spline bins (NSF only).
    """

    model: str = "maf"
    hidden_features: int = 128
    num_transforms: int = 6
    dropout_probability: float = 0.1
    num_blocks: int = 2
    num_bins: int = 10
