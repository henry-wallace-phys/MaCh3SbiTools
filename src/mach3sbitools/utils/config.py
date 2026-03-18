from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for training the posterior density estimator."""
    save_path: Path | None = None # Path to save the density estimator to
    batch_size: int = 2048 # Batch size for training
    learning_rate: float = 5e-4 # Learning rate for training
    max_epochs: int = 500 # Max number of epochs
    stop_after_epochs: int = 100 # How many epochs to stop after plateauing
    scheduler_patience: int = 20 # How long to wait until lowering the learning rate
    validation_fraction: float = 0.1 # Validation ratio
    num_workers: int = 1 # Number of workers to use for dataloaders
    autosave_every: int = 10 # How often to save the model
    resume_checkpoint: Path | None = None # Where to resume training from
    use_amp: bool = False # Use AMP? Not necessarily recommended
    print_interval: int = 10 # How often to print
    show_progress_bar: bool = False # Show a full progress bar (only works in Jupyter)
    show_epoch_progress: bool = False # Show an epoch/epoch progress bar (only works in Jupyter)
    tensorboard_dir: Path | None = None # Path to tensorboard directory
    warmup_epochs: int = 50 # Number of epochs to warm up with
    ema_alpha: float = 0.05 # Alpha value for smoothing
    compile: bool = False # Compile the model [NOT RECOMMENDED]


@dataclass
class PosteriorConfig:
    """Configuration for creating the posterior density estimator."""

    model: str = "maf" # Model to use
    hidden_features: int = 128 # Number of hidden features
    num_transforms: int = 6 # Number of transforms (for MAF)
    dropout_probability: float = 0.1 # Dropout probability
    num_blocks: int = 2 # Number of recurssive blocks
    num_bins: int = 10 # Number of bins (for NSE)
