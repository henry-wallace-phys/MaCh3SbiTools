"""
High-level simulator wrapper.

Combines prior construction and forward simulation into a single interface,
handling bad simulations gracefully and persisting results as Arrow feather
files.
"""

from pathlib import Path

import numpy as np
from pyarrow import parquet as pq
from tqdm import tqdm

from mach3sbitools.types import SimulatorData
from mach3sbitools.utils import get_logger, to_feather

from .priors import Prior, create_prior
from .simulator_injector import SimulatorProtocol, get_simulator

logger = get_logger()


class Simulator:
    """
    Wraps a :class:`~mach3sbitools.simulator.simulator_injector.SimulatorProtocol`
    with prior construction and simulation utilities.

    The simulator draws parameter vectors from the prior, passes each through
    the underlying simulator, and applies Poisson smearing to the output.
    Failed simulations are skipped with a warning.
    """

    def __init__(
        self,
        module_name: str,
        class_name: str,
        config: Path,
        nuisance_pars: list[str] | None = None,
        cyclical_pars: list[str] | None = None,
    ):
        """
        Instantiate the simulator and build its prior.

        :param module_name: Dotted Python module path containing the simulator
            class (e.g. ``'mypackage.simulator'``).
        :param class_name: Name of the simulator class. Must implement
            :class:`~mach3sbitools.simulator.simulator_injector.SimulatorProtocol`.
        :param config: Path to the simulator configuration file.
        :param nuisance_pars: fnmatch patterns for parameters to exclude from
            the prior and saved outputs.
        :param cyclical_pars: fnmatch patterns for parameters that use a
            cyclical sinusoidal prior over ``[-2π, 2π]``.
        """
        self.simulator_wrapper: SimulatorProtocol = get_simulator(
            module_name, class_name, config
        )
        self.prior: Prior = create_prior(
            self.simulator_wrapper,
            nuisance_pars=nuisance_pars,
            cyclical_pars=cyclical_pars,
        )

    def simulate(self, n_samples: int) -> tuple[SimulatorData, SimulatorData]:
        """
        Draw *n_samples* from the prior and run the forward simulator.

        Each successful simulation draws ``θ ~ prior``, calls
        ``simulator.simulate(θ)``, then applies Poisson smearing
        ``x ~ Poisson(simulator_output)``. Samples that raise an exception
        are silently skipped.

        :param n_samples: Number of simulation attempts.
        :returns: Tuple of ``(theta, x)`` arrays, each of length ≤ *n_samples*.
            Fewer samples are returned if any simulations failed.
        """
        samples = self.prior.sample((n_samples,))
        theta = samples.cpu().numpy()

        valid_theta = np.empty_like(theta)
        valid_x = None
        count = 0

        for i, t in enumerate(tqdm(theta, desc="Simulating")):
            try:
                x = self.simulator_wrapper.simulate(t)
                x_sample = np.random.poisson(x)

                if valid_x is None:
                    valid_x = np.empty(
                        (n_samples, *x_sample.shape), dtype=x_sample.dtype
                    )
                valid_theta[count] = t
                valid_x[count] = x_sample
                count += 1
            except Exception:
                logger.warning("Error: Bad simulation! Skipping sample.")

        return valid_theta[:count], valid_x[
            :count
        ] if valid_x is not None else np.array([])

    def save(
        self,
        file_path: Path,
        theta: SimulatorData,
        x: SimulatorData,
        prior_path: Path | None = None,
    ) -> None:
        """
        Save simulation outputs to a feather file.

        Optionally also pickles the prior alongside the data.

        :param file_path: Destination ``.feather`` file path.
        :param theta: Sampled parameter arrays, shape ``(n, n_params)``.
        :param x: Simulated observable arrays, shape ``(n, n_bins)``.
        :param prior_path: If provided, the prior is also saved here as a
            ``.pkl`` file.
        """
        to_feather(file_path, theta, x)
        if prior_path is not None:
            prior_path.parent.mkdir(parents=True, exist_ok=True)
            self.prior.save(prior_path)

    def save_data(self, file_path: Path) -> None:
        """
        Save the observed data bins from the simulator to a parquet file.

        Calls :meth:`~mach3sbitools.simulator.simulator_injector.SimulatorProtocol.get_data_bins`
        and writes the result under the key ``"data"``. Useful for producing
        the observation vector *x_o* used during inference.

        :param file_path: Destination ``.parquet`` file path. Parent
            directories are created automatically.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data_table = {"data": self.simulator_wrapper.get_data_bins()}
        pq.write_table(data_table, str(file_path))
