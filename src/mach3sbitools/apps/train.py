from pathlib import Path

from mach3sbitools.inference import InferenceHandler
from mach3sbitools.utils import PosteriorConfig, TrainingConfig, get_logger


def train_module(
    save_file: Path,
    prior_path: Path,
    dataset: Path,
    nuisance_pars: list[str],
    model: str,
    hidden: int,
    dropout: float,
    num_blocks: int,
    transforms: int,
    num_bins: int,
    batch_size: int,
    max_epochs: int,
    ema_alpha: float,
    learning_rate: float,
    stop_after_epochs: int,
    validation_fraction: float,
    num_workers: int,
    autosave_every: int,
    resume_checkpoint: Path | None,
    use_amp: bool,
    print_interval: int,
    tensorboard_dir: Path | None,
    scheduler_patience: int,
    show_progress: bool,
    compile_model: bool,
) -> None:
    """Train a Neural Posterior Estimation (NPE) density estimator.

    Loads simulations from ``--dataset``, builds an NPE model with the
    specified architecture, and trains it with a custom loop featuring
    linear warm-up, ReduceLROnPlateau scheduling, EMA-based early stopping,
    and periodic checkpointing.

    The full model configuration (architecture + hyperparameters) is embedded
    in every checkpoint, so ``inference`` requires no architecture flags.

    Example::

        mach3sbi train \\
            -r prior.pkl -d sims/ -s models/run.pt \\
            --model maf --hidden 128 --transforms 8 \\
            --max_epochs 50000 --stop_after_epochs 200
    """
    logger = get_logger()
    if compile_model:
        logger.warning(
            "Requested model compilation. In testing this has been shown to be slower."
        )

    posterior_config = PosteriorConfig(
        model=model,
        hidden_features=hidden,
        num_transforms=transforms,
        dropout_probability=dropout,
        num_blocks=num_blocks,
        num_bins=num_bins,
    )

    save_file = Path(save_file)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    training_config = TrainingConfig(
        save_path=save_file,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        stop_after_epochs=stop_after_epochs,
        validation_fraction=validation_fraction,
        num_workers=num_workers,
        autosave_every=autosave_every,
        resume_checkpoint=Path(resume_checkpoint) if resume_checkpoint else None,
        use_amp=use_amp,
        print_interval=print_interval,
        show_progress=show_progress,
        tensorboard_dir=Path(tensorboard_dir) if tensorboard_dir else None,
        scheduler_patience=scheduler_patience,
        compile=compile_model,
        ema_alpha=ema_alpha,
    )

    if resume_checkpoint:
        get_logger().info(f"Resuming from {resume_checkpoint}")

    inference_handler = InferenceHandler(Path(prior_path), nuisance_pars)
    inference_handler.set_dataset(Path(dataset))
    inference_handler.load_training_data()
    inference_handler.create_posterior(posterior_config)
    # model_config is passed through so every checkpoint is self-contained
    inference_handler.train_posterior(training_config, model_config=posterior_config)
