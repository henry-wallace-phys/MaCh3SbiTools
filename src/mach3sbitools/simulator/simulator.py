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
    Wraps around a simulator protocol object.
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
        The main simulator interface
        :param module_name: The name of the module to import i.e. mymodule.submodule. ...
        :param class_name: The name of the class in the module
        :param config: The path to the config file (all simulators are required to be configurable)
        :param nuisance_pars: The parameters to filter out
        :param cyclical_pars: Parameters which use a cyclical distribution (±2pi)
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
        Generate up to n_samples samples from the simulator.

        :param n_samples: Number of samples to generate
        :return: theta, x
        """
        samples = self.prior.sample((n_samples,))
        theta = samples.cpu().numpy()

        valid_theta = np.empty_like(theta)
        valid_x = None
        count = 0 # Keep track of good simulations

        for i, t in enumerate(tqdm(theta, desc="Simulating")):
            try:
                x = self.simulator_wrapper.simulate(t)
                x_sample = np.random.poisson(x)

                if valid_x is None:
                    valid_x = np.empty((n_samples, *x_sample.shape),
                                       dtype=x_sample.dtype)
                valid_theta[count] = t
                valid_x[count] = x_sample
                count += 1
            except Exception:
                logger.warning("Error: Bad simulation! Skipping sample.")

        return valid_theta[:count], valid_x[:count] if valid_x is not None else np.array([])

    def save(
        self,
        file_path: Path,
        theta: SimulatorData,
        x: SimulatorData,
        prior_path: Path | None = None,
    ) -> None:
        """
        Saves the sampled data to an Arrow file.
        :param file_path: Path to the output Arrow file
        :param theta: Sampled theta values
        :param x: Sampled x values
        :param prior_path: Where to save the prior data.
        """
        # Save the simulations to a feather file
        to_feather(file_path, theta, x)
        # We can also save our prior
        if prior_path is not None:
            prior_path.parent.mkdir(parents=True, exist_ok=True)
            self.prior.save(prior_path)

    def save_data(self, file_path: Path):
        """
        Save "data" generated from the simulator. Useful for testing many data points

        :param file_path: The path to save to
        :return:
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data_table = {"data": self.simulator_wrapper.get_data_bins()}
        pq.write_table(data_table, str(file_path))