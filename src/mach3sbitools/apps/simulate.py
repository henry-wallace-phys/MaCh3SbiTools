from pathlib import Path

from mach3sbitools.simulator import Simulator


def simulate_module(
    simulator_module: str,
    simulator_class: str,
    config: Path,
    n_simulations: int,
    output_file: Path,
    nuisance_pars: list[str],
    cyclical_pars: list[str],
    prior_file: Path | None,
) -> None:
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
    simulator.save(
        Path(output_file), x, theta, Path(prior_file) if prior_file else None
    )
