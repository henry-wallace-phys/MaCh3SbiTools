import click

from pathlib import Path

from mach3sbitools.simulator import (create_prior,
                                     get_simulator,
                                     Simulator)

from mach3sbitools.utils import (MaCh3Logger,
                                 get_logger,
                                 TrainingConfig,
                                 PosteriorConfig)
from mach3sbitools.inference import InferenceHandler
from pyarrow import parquet as pq
import pandas as pd
import uproot as ur

# Helpers and common options
nuisance_par_opt = click.option('--nuisance_pars', '-p', multiple=True, help="List of parameter names to exclude from the simulator")

_LOGGER_OPTIONS = [click.option('--log-level', help="Logging level", default='INFO'),
                   click.option('--log_file', help="Save the log file"),]

_SIMULATOR_OPTIONS = [
    click.option('--simulator_module', '-m', help="Name of the simulator module (i.e. Simulator.simulator_module)", required=True),
    click.option("--simulator_class", '-s', help="Name of the simulator class (i.e. SimulatorClass)", required=True),
    click.option("--config", "-c", help="Name of the simulator configuration file (i.e. FitterConfig.yaml)", required=True),
    click.option("--output_file", '-o', help="Name of the output file", required=True),
    nuisance_par_opt,
    # Optional params,
    click.option('--cyclical_pars', '-s', multiple=True, help="List of cyclical parameters")
]

_INFERENCE_OPTIONS = [
    click.option('--prior_path', '-r', help="Path to the prior file", required=True),
    nuisance_par_opt
]

_MODEL_OPTIONS = [
    click.option("--model", help="Name of the model", default="maf"),
    click.option("--hidden", help="number of hidden neurons", default=64),
    click.option('--dropout', help="dropout rate", default=0.2),
    click.option('--num_blocks', help="number of blocks", default=2),
    click.option('--transforms', help="number of transformations for (only for MAF)", default=5),
    click.option('--num_bins', help="number of bins for (only for NSE)", default=10),
]

_TRAINING_OPTIONS = [
    click.option("--batch_size", help="Batch size", default=2048, type=int),
    click.option("--max_epochs", help="Number of epochs", default=int(1e5), type=int),
    click.option("--warmup_epochs", help="Number of epochs to warm up", default=50, type=int),
    click.option("--ema_alpha", help="EMA smoothing parameter", default=0.01, type=float),
    click.option("--learning_rate", help="Learning rate", default=5e-4, type=float),
    click.option('--stop_after_epochs', help="Number of epochs to stop after plateauing", type=int, default=50),
    click.option("--validation_fraction", help="Fraction of data to use for validation", type=float, default=0.1),
    click.option('--num_workers', help="Number of workers", default=1, type=int),
    click.option("--autosave_every", help="Number of epochs to save the model", default=1, type=int),
    click.option("--resume_checkpoint", type=click.Path(exists=True), help="Path to checkpoint to resume training"),
    click.option("--use_amp", help="Use AMP", is_flag=True, default=False),
    click.option("--print_interval", help="Print training progress every this many steps", default=1, type=int),
    click.option("--tensorboard_dir", help="Tensorboard directory", default=None),
    click.option("--scheduler_patience", help="Number of epochs to wait before reducing lr", default=20, type=int),
    click.option("--show_progress_bar", help="Show global progress bar", is_flag=True, default=False),
    click.option("--show_epoch_progress", help="Show epoch progress bar", is_flag=True, default=False),
    click.option("--compile_model", help="Compile the model. WARNING: This can be slow!", is_flag=True, default=False),
]


def apply_options(options):
    """Generic decorator factory for any list of options."""
    def decorator(f):
        for option in reversed(options):
            f = option(f)
        return f
    return decorator


@click.group()
@apply_options(_LOGGER_OPTIONS)
def cli(log_file: Path,
        log_level: str):
     MaCh3Logger(
        name=__name__,
        level=log_level,
        log_file=log_file,
    )

@cli.command('create_prior', short_help="Generate prior from simulator instance")
@apply_options(_SIMULATOR_OPTIONS)
# Mandatory params
def create_prior(simulator_module: str,
                 simulator_class: str,
                 config: Path,
                 output_file: Path,
                 nuisance_pars: list[str] | None,
                 cyclical_pars: list[str] | None,
                 ):
    # Creates a static prior object
    injector = get_simulator(simulator_module, simulator_class, config)
    prior = create_prior(injector, nuisance_pars, cyclical_pars)
    prior.save(output_file)

@cli.command('simulate', short_help="Produce simulations from the simulator")
@apply_options(_SIMULATOR_OPTIONS)
@click.option('--n_simulations', '-n', required=True, help="Number of simulations to run")
@click.option('--prior_file', '-r', help="File to save the prior from this process to")
def simulate(simulator_module: str,
             simulator_class: str,
             config: Path,
             n_simulations: int,
             output_file: Path,
             nuisance_pars: list[str] | None,
             cyclical_pars: list[str] | None,
             prior_file: Path | None,
             ):

    # Runs the simulator
    simulator = Simulator(simulator_module,
                          simulator_class,
                          config,
                          nuisance_pars=nuisance_pars,
                          cyclical_pars=cyclical_pars)

    x, theta = simulator.simulate(n_simulations)
    simulator.save(output_file, x, theta, prior_file)

@cli.command('save_data', short_help="When accessible from the simulator, get data + save to paraquet")
@apply_options(_SIMULATOR_OPTIONS)
def save_data(simulator_module: str,
             simulator_class: str,
             config: Path,
             output_file: Path,
             nuisance_pars: list[str] | None,
             cyclical_pars: list[str] | None,
             ):

    # Runs the simulator
    simulator = Simulator(simulator_module,
                          simulator_class,
                          config,
                          nuisance_pars=nuisance_pars,
                          cyclical_pars=cyclical_pars)
    simulator.save_data(output_file)


@cli.command('train', short_help="Train the model")
@click.option('--save-dir', '-s', help="Directory to save the trained model", required=True)
@click.option("--dataset", "-d", type=click.Path(exists=True), required=True, help="Path to the data folder")
@apply_options(_INFERENCE_OPTIONS)
@apply_options(_MODEL_OPTIONS)
@apply_options(_TRAINING_OPTIONS)
def train(
    save_dir: Path,
    prior_path: Path,
    dataset: Path,
    nuisance_pars: list[str] | None,

    # Training pars
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
    logger = get_logger(__name__)
    if compile_model:
        logger.warning("Request model compilation. In testing this has been shown to be slower")

    # Painful but needed
    posterior_config = PosteriorConfig(
        model = model,
        hidden_features = hidden,
        num_transforms = transforms,
        dropout_probability = dropout,
        num_blocks = num_blocks,
        num_bins = num_bins,
    )

    training_config = TrainingConfig(
        save_path = save_dir,
        batch_size = batch_size,
        learning_rate = learning_rate,
        max_epochs = max_epochs,
        stop_after_epochs = stop_after_epochs,
        validation_fraction = validation_fraction,
        num_workers = num_workers,
        autosave_every = autosave_every,
        resume_checkpoint = resume_checkpoint,
        use_amp = use_amp,
        print_interval = print_interval,
        show_progress_bar = show_progress_bar,
        tensorboard_dir = tensorboard_dir,
        scheduler_patience = scheduler_patience,
        show_epoch_progress = show_epoch_progress,
        compile = compile_model,
        warmup_epochs = warmup_epochs,
        ema_alpha = ema_alpha
    )

    inference_handler = InferenceHandler(prior_path, nuisance_pars)
    inference_handler.set_dataset(dataset)
    inference_handler.load_training_data()

    inference_handler.create_posterior(posterior_config)
    inference_handler.train_posterior(training_config)


@cli.command(short_help="Perform inference with the SBI")
@click.option('--posterior', '-i', type=click.Path(exists=True), required=True, help="Path to the posterior file")
@click.option('--n_samples', '-n', required=True, help="Number of samples")
@click.option('--observed_data', '-o', type=click.Path(exists=True), help="Path to the observed data")
@apply_options(_MODEL_OPTIONS)
@apply_options(_INFERENCE_OPTIONS)
def inference(posterior_path: Path,
              prior_path,
              save_dir,
              n_samples: int,
              nuisance_pars,
              model: str,
              hidden: int,
              dropout: float,
              num_blocks: int,
              transforms: int,
              num_bins: int,
              ):
    # Performs inference using a trained SBI model and saves to a paraquet file
    posterior_config = PosteriorConfig(
        model = model,
        hidden_features = hidden,
        num_transforms = transforms,
        dropout_probability = dropout,
        num_blocks = num_blocks,
        num_bins = num_bins,
    )



    inference_handler = InferenceHandler(prior_path, nuisance_pars)
    inference_handler.load_posterior(posterior_path, posterior_config)

    parameter_names = inference_handler.prior.prior_data.parameter_names
    samples = inference_handler.sample_posterior(n_samples).cpu()
    # Write to dataframe
    data_table = pd.DataFrame({p: samples[:,i] for i,p in enumerate(parameter_names)})
    pq.write_table(data_table, save_dir)


if __name__ == "__main__":
    cli()