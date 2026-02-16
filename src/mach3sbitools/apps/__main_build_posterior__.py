from pathlib import Path
import fnmatch

import click
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from mach3sbitools.utils.config import TrainingConfig, PosteriorConfig, PosteriorConfig
from mach3sbitools.inference.sbi_interface import MaCh3SBIInterface
from mach3sbitools.utils.logger import MaCh3Logger, get_logger

logger = get_logger("mach3sbitools")


@click.command()
# ── Simulation ──────────────────────────────────────────────────────────────
@click.option('--mach3-type',    type=str,  required=True, help='Type of MaCh3 simulator to use.')
@click.option('--config-file',   type=Path, required=True, help='Path to the MaCh3 configuration file.')
@click.option('--mach3-dataset', type=Path, required=True, help='Path to the MaCh3 dataset folder.')
@click.option('--nuisance-pars', type=str,  multiple=True, help='Nuisance parameter name patterns to exclude (supports wildcards).')
@click.option('--cyclical-pars', type=str,  multiple=True, help='Nuisance parameter name patterns to exclude (supports wildcards).')

# ── Persistence ──────────────────────────────────────────────────────────────
@click.option('--save-file',      type=Path, required=True, help='Path to save best model weights (.pt).')
@click.option('--inference-file', type=Path, default=None,  help='Path to resume from a saved checkpoint (.pt).')

# ── Network architecture ─────────────────────────────────────────────────────
@click.option('--hidden-features',      type=int,   default=128,      show_default=True, help='Flow hidden layer width.')
@click.option('--num-transforms',       type=int,   default=6,        show_default=True, help='Number of NSF transforms.')
@click.option('--dropout-probability',  type=float, default=0.1,      show_default=True, help='Dropout probability.')
@click.option('--num-blocks',           type=int,   default=2,        show_default=True, help='Residual blocks per transform.')
@click.option('--num-bins',             type=int,   default=10,       show_default=True, help='Spline bins per transform (NSF only).')

# ── Training ─────────────────────────────────────────────────────────────────
@click.option('--batch-size',           type=int,   default=2048,    show_default=True, help='DataLoader batch size.')
@click.option('--learning-rate',        type=float, default=5e-4,    show_default=True, help='Initial Adam learning rate.')
@click.option('--max-epochs',           type=int,   default=50_000,  show_default=True, help='Hard epoch ceiling.')
@click.option('--stop-after-epochs',    type=int,   default=50,      show_default=True, help='Early-stopping patience (epochs).')
@click.option('--validation-fraction',  type=float, default=0.1,     show_default=True, help='Fraction of data held out for validation.')
@click.option('--num-workers',          type=int,   default=4,       show_default=True, help='DataLoader worker processes.')
@click.option('--autosave-every',       type=int,   default=100,     show_default=True, help='Save a resumable checkpoint every N epochs.')
@click.option('--print-interval',       type=int,   default=1,       show_default=True, help='How often to print info.')
@click.option('--show-epoch-progress',  is_flag=True, default=False, help='How often to print info.')
@click.option('--show-progress-bar',    is_flag=True, default=False, help='How often to print info.')
@click.option('--no-amp',               is_flag=True, default=False, help='Disable BF16 mixed precision (enabled by default on CUDA).')


# ── Output ───────────────────────────────────────────────────────────────────
@click.option('--n-samples',       type=int,  default=1_000_000, show_default=True, help='Posterior samples to draw for plots.')
@click.option('--log-file',        type=Path, default=None, help='Optional path to write logs to file.')
@click.option('--log-level',       type=click.Choice(['DEBUG', 'INFO', 'WARNING'], case_sensitive=False), default='INFO', show_default=True)
@click.option('--tensorboard-dir', type=Path, default=None, help='Directory for TensorBoard logs. Omit to disable.')
def main(
    mach3_type: str,
    config_file: Path,
    mach3_dataset: Path,
    nuisance_pars: tuple,
    cyclical_pars: tuple,
    save_file: Path,
    inference_file: Path,
    hidden_features: int,
    num_transforms: int,
    dropout_probability: float,
    num_blocks: int,
    num_bins: int,
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    stop_after_epochs: int,
    validation_fraction: float,
    num_workers: int,
    autosave_every: int,
    print_interval: int,
    show_epoch_progress: bool,
    show_progress_bar: bool,
    no_amp: bool,
    n_samples: int,
    log_file: Path,
    log_level: str,
    tensorboard_dir: Path,
):
    # ── Initialise logger ────────────────────────────────────────────────────
    MaCh3Logger("mach3sbitools", level=log_level, log_file=log_file)
    logger.info(f"MaCh3 SBI | type=[cyan]{mach3_type}[/] | config=[cyan]{config_file}[/]")

    # ── Build interface ──────────────────────────────────────────────────────
    inference = MaCh3SBIInterface(
        mach3_type,
        config_file,
        nuisance_pars=list(nuisance_pars) or None,
        cyclical_pars=list(cyclical_pars) or None
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    inference.set_dataset(mach3_dataset)
    inference.load_training_data()

    # ── Model ─────────────────────────────────────────────────────────────────
    posterior_config = PosteriorConfig(
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        dropout_probability=dropout_probability,
        num_blocks=num_blocks,
        num_bins=num_bins,
    )
    inference.create_posterior(posterior_config)

    # ── Train ─────────────────────────────────────────────────────────────────
    config = TrainingConfig(
        save_path=save_file,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        stop_after_epochs=stop_after_epochs,
        validation_fraction=validation_fraction,
        num_workers=num_workers,
        autosave_every=autosave_every,
        resume_checkpoint=inference_file,
        use_amp=not no_amp,
        print_interval=print_interval,
        show_progress_bar=show_progress_bar,
        show_epoch_progress=show_epoch_progress,
        tensorboard_dir=tensorboard_dir,
    )
    inference.train_posterior(config)

    # ── Sample ───────────────────────────────────────────────────────────────
    samples = inference.sample_posterior(n_samples)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_path = save_file.with_name(f"{save_file.stem}_posterior_plots.pdf")
    logger.info(f"Writing posterior plots → [cyan]{plot_path}[/]")

    parameter_names = inference.simulator.mach3_wrapper.get_parameter_names()

    # Filter to only the parameters that weren't masked as nuisance
    active_params = [
        (i, name) for i, name in enumerate(parameter_names)
        if not any(fnmatch.fnmatch(name, pat) for pat in nuisance_pars)
    ]

    with PdfPages(plot_path) as pdf:
        for i, name in active_params:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(samples[:, i].cpu().numpy(), bins=100, density=True)
            ax.set_title(f'Posterior: {name}')
            ax.set_xlabel('Parameter value')
            ax.set_ylabel('Density')
            pdf.savefig(fig)
            plt.close(fig)

    logger.info(f"Saved [bold]{len(active_params)}[/] posterior plots to [cyan]{plot_path}[/]")


if __name__ == '__main__':
    main()