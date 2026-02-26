from mach3sbitools.mach3_interface.mach3_simulator import MaCh3Simulator
from pathlib import Path
import click

@click.command()
@click.option('--mach3-type', type=str, required=True, help='Type of MaCh3 simulator to use.')
@click.option('--config-file', type=Path, required=True, help='Path to the MaCh3 configuration file.')
@click.option('--n-samples', type=int, required=True, help='Number of samples to simulate.')
@click.option('--output-file', type=Path, required=True, help='Path to the output Arrow file.')
@click.option('--nuisance-pars', type=str,  multiple=True, help='Nuisance parameter name patterns to exclude (supports wildcards).')
@click.option('--cyclical-pars', type=str,  multiple=True, help='Nuisance parameter name patterns to exclude (supports wildcards).')

def main(mach3_type: str,
         config_file: Path,
         n_samples: int,
         output_file: Path,
         nuisance_pars: tuple,
         cyclical_pars: tuple,
    ):
    simulator = MaCh3Simulator(mach3_type,
                               config_file,
                               nuisance_pars=list(nuisance_pars) or None,
                               cyclical_pars=list(cyclical_pars) or None
    )

    theta, x = simulator.simulate_mach3(n_samples)
    simulator.save_to_arrow(output_file, theta, x)
    
if __name__ == '__main__':
    main()