from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for training the posterior density estimator."""
    save_path: Path | None = None
    batch_size: int = 2048
    learning_rate: float = 5e-4
    max_epochs: int = 500
    stop_after_epochs: int = 100
    validation_fraction: float = 0.1
    num_workers: int = 1
    autosave_every: int = 10
    resume_checkpoint: Path | None = None
    use_amp: bool = True
    print_interval: int = 10
    show_progress_bar: bool = True
    tensorboard_dir: Path | None = None
    scheduler_patience: int = 100
    show_epoch_progress: bool = True
    compile: bool = False
    warmup_epochs: int = 50
    ema_alpha: float = 0.05


@dataclass
class PosteriorConfig:
    """Configuration for creating the posterior density estimator."""

    model: str = "maf"
    hidden_features: int = 128
    num_transforms: int = 6
    dropout_probability: float = 0.1
    num_blocks: int = 2
    num_bins: int = 10
