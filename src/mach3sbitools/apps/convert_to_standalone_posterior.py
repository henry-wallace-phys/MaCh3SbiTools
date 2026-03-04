'''
Simple app that merely converts a model file to posterior
'''
from mach3sbitools.utils.config import PosteriorConfig, PosteriorConfig
from mach3sbitools.inference.sbi_interface import MaCh3SBIInterface
from mach3sbitools.utils.logger import MaCh3Logger, get_logger

from pathlib import Path
import pickle as pkl
import click


logger = get_logger("mach3sbitools")

@click.command()
# ── Simulation ──────────────────────────────────────────────────────────────
@click.option('--mach3-type',    type=str,  required=True, help='Type of MaCh3 simulator to use.')
@click.option('--config-file',   type=Path, required=True, help='Path to the MaCh3 configuration file.')
@click.option('--nuisance-pars', type=str,  multiple=True, help='Nuisance parameter name patterns to exclude (supports wildcards).')
@click.option('--cyclical-pars', type=str,  multiple=True, help='Nuisance parameter name patterns to exclude (supports wildcards).')

# ── Persistence ──────────────────────────────────────────────────────────────
@click.option('--save-file',      type=Path, required=True, help='Path to save best model weights (.pkl).')
@click.option('--inference-file', type=Path, default=None,  help='Path to resume from a saved checkpoint (.ts).')

# ── Network architecture ─────────────────────────────────────────────────────
@click.option('--hidden-features',      type=int,   default=128,      show_default=True, help='Flow hidden layer width.')
@click.option('--num-transforms',       type=int,   default=6,        show_default=True, help='Number of NSF transforms.')
@click.option('--dropout-probability',  type=float, default=0.1,      show_default=True, help='Dropout probability.')
@click.option('--num-blocks',           type=int,   default=2,        show_default=True, help='Residual blocks per transform.')
@click.option('--num-bins',             type=int,   default=10,       show_default=True, help='Spline bins per transform (NSF only).')
# ── Output ───────────────────────────────────────────────────────────────────
@click.option('--log-file',        type=Path, default=None, help='Optional path to write logs to file.')
@click.option('--log-level',       type=click.Choice(['DEBUG', 'INFO', 'WARNING'], case_sensitive=False), default='INFO', show_default=True)
def convert_to_standalone_posterior(
    mach3_type: str,
    config_file: Path,
    inference_file: Path,
    save_file: Path,
    log_file: Path,
    log_level: str,    
    nuisance_pars: tuple,
    cyclical_pars: tuple,
    hidden_features: int,
    num_transforms: int,
    dropout_probability: float,
    num_blocks: int,
    num_bins: int
):
    MaCh3Logger("mach3sbitools", level=log_level, log_file=log_file)
    logger.info(f"MaCh3 SBI | type=[cyan]{mach3_type}[/] | config=[cyan]{config_file}[/]")
    
    inference = MaCh3SBIInterface(
        mach3_type,
        config_file,
        nuisance_pars=list(nuisance_pars) or None,
        cyclical_pars=list(cyclical_pars) or None
    )

    posterior_config = PosteriorConfig(
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        dropout_probability=dropout_probability,
        num_blocks=num_blocks,
        num_bins=num_bins,
    )
    
    # Now we can now
    logger.info(f"Loading {inference_file}")
    if not inference_file.exists():
        raise FileNotFoundError(f"Cannot find {inference_file}")
    
    inference.load_posterior(inference_file, posterior_config)
    inference.sample_posterior(1)
    
    if inference.posterior is None:
        raise FileExistsError("Cannot find posterior!")
    
    
    logger.info(f"Saving to {save_file}")
    with open(save_file, 'wb') as handle:
        pkl.dump(inference.posterior, handle)
    

if __name__=='__main__':
    convert_to_standalone_posterior()