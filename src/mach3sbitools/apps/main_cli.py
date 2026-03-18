from pathlib import Path

import click
import pandas as pd
from pyarrow import parquet as pq

from mach3sbitools.inference import InferenceHandler
from mach3sbitools.simulator import Simulator, create_prior, get_simulator
from mach3sbitools.utils import MaCh3Logger, PosteriorConfig, TrainingConfig, get_logger

# Helpers and common options
nuisance_par_opt = click.option(
    "--nuisance_pars",
    "-p",
    multiple=True,
    help="Parameter name patterns (fnmatch-style, e.g. 'syst_*') to exclude from the prior and training.",
)

_LOGGER_OPTIONS = [
    click.option(
        "--log-level",
        help="Console logging level. One of DEBUG, INFO, WARNING, ERROR.",
        default="INFO",
    ),
    click.option("--log_file", help="Optional path to write a plain-text log file."),
]

_SIMULATOR_OPTIONS = [
    click.option(
        "--simulator_module",
        "-m",
        help="Dotted Python module path containing the simulator class (e.g. 'mypackage.simulator').",
        required=True,
    ),
    click.option(
        "--simulator_class",
        "-s",
        help="Name of the simulator class within the module. Must implement SimulatorProtocol.",
        required=True,
    ),
    click.option(
        "--config",
        "-c",
        help="Path to the simulator configuration file (e.g. a MaCh3 fitter YAML).",
        required=True,
    ),
    click.option("--output_file", "-o", help="Path to write the output file."),
    nuisance_par_opt,
    click.option(
        "--cyclical_pars",
        "-s",
        multiple=True,
        help="Parameter name patterns (fnmatch-style) that should use a cyclical sinusoidal prior over [-2π, 2π].",
    ),
]

_INFERENCE_OPTIONS = [
    click.option(
        "--prior_path", "-r", help="Path to a saved Prior .pkl file.", required=True
    ),
    nuisance_par_opt,
]

_MODEL_OPTIONS = [
    click.option(
        "--model",
        help="Density estimator architecture. One of: 'maf' (Masked Autoregressive Flow) or 'nse' (Neural Spline Flow).",
        default="maf",
    ),
    click.option(
        "--hidden",
        help="Number of hidden units per layer in the density estimator.",
        default=64,
    ),
    click.option(
        "--dropout", help="Dropout probability applied during training.", default=0.2
    ),
    click.option(
        "--num_blocks",
        help="Number of residual blocks in the density estimator.",
        default=2,
    ),
    click.option(
        "--transforms",
        help="Number of autoregressive transforms (MAF only).",
        default=5,
    ),
    click.option("--num_bins", help="Number of spline bins (NSF only).", default=10),
]

_TRAINING_OPTIONS = [
    click.option(
        "--batch_size",
        help="Number of samples per training batch.",
        default=2048,
        type=int,
    ),
    click.option(
        "--max_epochs",
        help="Maximum number of training epochs.",
        default=int(1e5),
        type=int,
    ),
    click.option(
        "--warmup_epochs",
        help="Number of epochs for linear learning-rate warm-up from 1% to 100%.",
        default=50,
        type=int,
    ),
    click.option(
        "--ema_alpha",
        help="Smoothing factor for the exponential moving average of validation loss used in early stopping.",
        default=0.01,
        type=float,
    ),
    click.option(
        "--learning_rate",
        help="Initial learning rate for the Adam optimiser.",
        default=5e-4,
        type=float,
    ),
    click.option(
        "--stop_after_epochs",
        help="Stop training if the EMA validation loss has not improved for this many epochs.",
        type=int,
        default=50,
    ),
    click.option(
        "--validation_fraction",
        help="Fraction of data held out for validation.",
        type=float,
        default=0.1,
    ),
    click.option(
        "--num_workers",
        help="Number of DataLoader worker processes for data loading.",
        default=1,
        type=int,
    ),
    click.option(
        "--autosave_every",
        help="Save a checkpoint every N epochs (in addition to best-model saves).",
        default=1,
        type=int,
    ),
    click.option(
        "--resume_checkpoint",
        type=click.Path(exists=True),
        help="Path to a checkpoint file to resume training from.",
    ),
    click.option(
        "--use_amp",
        help="Enable automatic mixed precision (AMP) training. May not improve performance on all hardware.",
        is_flag=True,
        default=False,
    ),
    click.option(
        "--print_interval",
        help="Log training progress every N epochs.",
        default=1,
        type=int,
    ),
    click.option(
        "--tensorboard_dir",
        help="Directory for TensorBoard event files. Omit to disable TensorBoard logging.",
        default=None,
    ),
    click.option(
        "--scheduler_patience",
        help="Number of epochs without improvement before the ReduceLROnPlateau scheduler halves the learning rate.",
        default=20,
        type=int,
    ),
    click.option(
        "--show_progress_bar",
        help="Show a global epoch progress bar (recommended for interactive/Jupyter use).",
        is_flag=True,
        default=False,
    ),
    click.option(
        "--show_epoch_progress",
        help="Show a per-epoch step progress bar (recommended for interactive/Jupyter use).",
        is_flag=True,
        default=False,
    ),
    click.option(
        "--compile_model",
        help="Compile the model with torch.compile. Can reduce per-step time on supported hardware but increases startup time.",
        is_flag=True,
        default=False,
    ),
]


def apply_options(options):
    """Generic decorator factory for any list of click options."""

    def decorator(f):
        for option in reversed(options):
            f = option(f)
        return f

    return decorator


@click.group()
@apply_options(_LOGGER_OPTIONS)
def cli(log_file: Path, log_level: str):
    """mach3sbi — simulation-based inference tools for MaCh3.

    Run ``mach3sbi COMMAND --help`` for detailed usage of each subcommand.
    """
    MaCh3Logger(
        name=__name__,
        level=log_level,
        log_file=log_file,
    )


@cli.command(
    "create_prior", short_help="Generate and save a prior from a simulator instance."
)
@apply_options(_SIMULATOR_OPTIONS)
def save_prior(
    simulator_module: str,
    simulator_class: str,
    config: Path,
    output_file: Path,
    nuisance_pars: list[str] | None,
    cyclical_pars: list[str] | None,
):
    """Generate a Prior from a simulator and save it to disk.

    Instantiates the simulator, reads its parameter names, bounds, nominals,
    and covariance, then constructs and pickles a Prior object ready for use
    in training and inference.

    Example::

        mach3sbi create_prior \\
            -m mypackage.simulator -s MySimulator \\
            -c config.yaml -o prior.pkl
    """
    injector = get_simulator(simulator_module, simulator_class, config)
    prior = create_prior(injector, nuisance_pars, cyclical_pars)
    prior.save(output_file)


@cli.command("simulate", short_help="Run simulations and save to feather files.")
@apply_options(_SIMULATOR_OPTIONS)
@click.option(
    "--n_simulations",
    "-n",
    required=True,
    help="Number of simulation samples to generate.",
)
@click.option(
    "--prior_file", "-r", help="Optional path to also save the prior used for this run."
)
def simulate(
    simulator_module: str,
    simulator_class: str,
    config: Path,
    n_simulations: int,
    output_file: Path,
    nuisance_pars: list[str] | None,
    cyclical_pars: list[str] | None,
    prior_file: Path | None,
):
    """Draw samples from the prior and run the simulator for each.

    Samples ``n_simulations`` parameter vectors θ from the prior, passes each
    through the simulator, applies Poisson smearing to the output, and saves
    the (θ, x) pairs as a feather file. Failed simulations are skipped with a
    warning.

    Example::

        mach3sbi simulate \\
            -m mypackage.simulator -s MySimulator \\
            -c config.yaml -n 100000 -o sims.feather
    """
    simulator = Simulator(
        simulator_module,
        simulator_class,
        config,
        nuisance_pars=nuisance_pars,
        cyclical_pars=cyclical_pars,
    )
    x, theta = simulator.simulate(n_simulations)
    simulator.save(output_file, x, theta, prior_file)


@cli.command(
    "save_data",
    short_help="Save observed data bins from the simulator to parquet.",
)
@apply_options(_SIMULATOR_OPTIONS)
def save_data(
    simulator_module: str,
    simulator_class: str,
    config: Path,
    output_file: Path,
    nuisance_pars: list[str] | None,
    cyclical_pars: list[str] | None,
):
    """Extract and save the observed data bins from the simulator.

    Calls ``get_data_bins()`` on the simulator and writes the result to a
    parquet file. Useful for producing the observed data vector ``x_o``
    that is passed to ``inference``.

    Example::

        mach3sbi save_data \\
            -m mypackage.simulator -s MySimulator \\
            -c config.yaml -o observed.parquet
    """
    simulator = Simulator(
        simulator_module,
        simulator_class,
        config,
        nuisance_pars=nuisance_pars,
        cyclical_pars=cyclical_pars,
    )
    simulator.save_data(output_file)


@cli.command("train", short_help="Train the NPE density estimator.")
@click.option(
    "--save-dir", "-s", help="Directory to write model checkpoints to.", required=True
)
@click.option(
    "--dataset",
    "-d",
    type=click.Path(exists=True),
    required=True,
    help="Path to folder of .feather simulation files.",
)
@apply_options(_INFERENCE_OPTIONS)
@apply_options(_MODEL_OPTIONS)
@apply_options(_TRAINING_OPTIONS)
def train(
    save_dir: Path,
    prior_path: Path,
    dataset: Path,
    nuisance_pars: list[str] | None,
    model: str,
    hidden: int,
    dropout: float,
    num_blocks: int,
    transforms: int,
    num_bins: int,
    batch_size: int,
    max_epochs: int,
    warmup_epochs: int,
    ema_alpha: float,
    learning_rate: float,
    stop_after_epochs: int,
    validation_fraction: float,
    num_workers: int,
    autosave_every: int,
    resume_checkpoint: Path,
    use_amp: bool,
    print_interval: int,
    tensorboard_dir: Path,
    scheduler_patience: int,
    show_progress_bar: bool,
    show_epoch_progress: bool,
    compile_model: bool,
):
    """Train a Neural Posterior Estimation (NPE) density estimator.

    Loads simulations from ``--dataset``, builds an NPE model with the
    specified architecture, and trains it with a custom loop featuring
    linear warm-up, ReduceLROnPlateau scheduling, EMA-based early stopping,
    and periodic checkpointing.

    Example::

        mach3sbi train \\
            -r prior.pkl -d sims/ -s models/ \\
            --model maf --hidden 128 --transforms 8 \\
            --max_epochs 50000 --stop_after_epochs 200
    """
    logger = get_logger()
    if compile_model:
        logger.warning(
            "Request model compilation. In testing this has been shown to be slower"
        )

    posterior_config = PosteriorConfig(
        model=model,
        hidden_features=hidden,
        num_transforms=transforms,
        dropout_probability=dropout,
        num_blocks=num_blocks,
        num_bins=num_bins,
    )

    training_config = TrainingConfig(
        save_path=save_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        stop_after_epochs=stop_after_epochs,
        validation_fraction=validation_fraction,
        num_workers=num_workers,
        autosave_every=autosave_every,
        resume_checkpoint=resume_checkpoint,
        use_amp=use_amp,
        print_interval=print_interval,
        show_progress_bar=show_progress_bar,
        tensorboard_dir=tensorboard_dir,
        scheduler_patience=scheduler_patience,
        show_epoch_progress=show_epoch_progress,
        compile=compile_model,
        warmup_epochs=warmup_epochs,
        ema_alpha=ema_alpha,
    )

    inference_handler = InferenceHandler(prior_path, nuisance_pars)
    inference_handler.set_dataset(dataset)
    inference_handler.load_training_data()
    inference_handler.create_posterior(posterior_config)
    inference_handler.train_posterior(training_config)


@cli.command(short_help="Sample the posterior given observed data.")
@click.option(
    "--posterior",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Path to a saved density estimator checkpoint (.pt).",
)
@click.option(
    "--n_samples", "-n", required=True, help="Number of posterior samples to draw."
)
@click.option(
    "--observed_data-file",
    "-o",
    type=click.Path(exists=True),
    help="Path to the observed data parquet file (produced by save_data).",
)
@apply_options(_MODEL_OPTIONS)
@apply_options(_INFERENCE_OPTIONS)
def inference(
    posterior_path: Path,
    prior_path,
    save_dir,
    n_samples: int,
    observed_data_file: Path,
    nuisance_pars,
    model: str,
    hidden: int,
    dropout: float,
    num_blocks: int,
    transforms: int,
    num_bins: int,
):
    """Sample the posterior distribution conditioned on observed data.

    Loads a trained density estimator, conditions it on the observed data
    vector from ``--observed_data-file``, draws ``--n_samples`` posterior
    samples, and writes them as a parquet file with one column per parameter.

    The model architecture flags must match those used during ``train``.

    Example::

        mach3sbi inference \\
            -i models/best.pt -r prior.pkl \\
            -n 100000 -o observed.parquet
    """
    posterior_config = PosteriorConfig(
        model=model,
        hidden_features=hidden,
        num_transforms=transforms,
        dropout_probability=dropout,
        num_blocks=num_blocks,
        num_bins=num_bins,
    )

    inference_handler = InferenceHandler(prior_path, nuisance_pars)
    inference_handler.load_posterior(posterior_path, posterior_config)

    parameter_names = inference_handler.prior.prior_data.parameter_names
    observed_data = pq.read_table(observed_data_file)
    samples = inference_handler.sample_posterior(n_samples, observed_data).cpu()

    data_table = pd.DataFrame({p: samples[:, i] for i, p in enumerate(parameter_names)})
    pq.write_table(data_table, save_dir)
