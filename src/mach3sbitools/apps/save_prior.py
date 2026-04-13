from pathlib import Path

from mach3sbitools.simulator import create_prior, get_simulator


def save_prior_module(
    simulator_module: str,
    simulator_class: str,
    config: Path,
    output_file: Path,
    nuisance_pars: list[str],
    cyclical_pars: list[str],
) -> None:
    """Generate a Prior from a simulator and save it to disk.

    Instantiates the simulator, reads its parameter names, bounds, nominals,
    and covariance, then constructs and pickles a Prior object ready for use
    in training and inference.

    Example::

        mach3sbi create_prior \\
            -m mypackage.simulator -s MySimulator \\
            -c config.yaml -o prior.pkl
    """
    injector = get_simulator(simulator_module, simulator_class, Path(config))
    prior = create_prior(injector, nuisance_pars, cyclical_pars)
    prior.save(Path(output_file))
