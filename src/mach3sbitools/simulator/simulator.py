from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
from pyarrow import parquet as pq
from tqdm import tqdm

from mach3sbitools.utils import get_logger

from .priors import Prior, create_prior
from .simulator_injector import SimulatorProtocol, get_simulator

logger = get_logger(__name__)


class Simulator:
    """
    Wraps around a simulator protocol object.
    """

    def __init__(
        self,
        module_name: str,
        class_name: str,
        config: Path | str,
        nuisance_pars: list[str] | None = None,
        cyclical_pars: list[str] | None = None,
    ):

        self.simulator_wrapper: SimulatorProtocol = get_simulator(
            module_name, class_name, config
        )
        self.prior: Prior = create_prior(
            self.simulator_wrapper,
            nuisance_pars=nuisance_pars,
            cyclical_pars=cyclical_pars,
        )

    def simulate(self, n_samples: int) -> tuple[Iterable, Iterable]:
        """
        Samples data from the specified simulator using the provided configuration and parameters.
        Args:
            :n_samples (int): Number of samples to generate.

        :returns: Tuple of (theta, x)
        """

        samples = self.prior.sample((n_samples,))
        theta = samples.cpu().numpy()

        valid_theta = []
        valid_x = []
        for t in tqdm(theta, desc="Simulating"):
            try:
                x = self.simulator_wrapper.simulate(t)
                valid_theta.append(t)
                valid_x.append(np.random.poisson(x))

            # If anything raises an exception we continue and skip the error
            except Exception:
                logger.warning("Error: Bad simulation! Skipping sample.")

        return valid_theta, valid_x

    def save(
        self,
        file_path: Path,
        theta: Iterable,
        x: Iterable,
        prior_path: Path | None = None,
    ) -> None:
        """
        Saves the sampled data to an Arrow file.

        Args:
            file_path (Path): Path to the output Arrow file.
            theta (Iterable): Sampled theta values.
            x (Iterable): Sampled x values.

            prior_path (Path, Optional): Where to save the prior data.
        """
        table = pa.Table.from_pydict(
            {
                "theta": theta,
                "x": x,
                "parameter_names": self.simulator_wrapper.get_parameter_names(),
            }
        )
        feather.write_feather(table, str(file_path))

        # We can also save our prior
        if prior_path is not None:
            prior_path.parent.mkdir(parents=True, exist_ok=True)
            self.prior.save(prior_path)

    def save_data(self, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data_table = {"data": self.simulator_wrapper.get_data_bins()}
        pq.write_table(data_table, str(file_path))

    def __call__(self, n_samples: int, file_path: Path) -> None:
        logger.info(f"Starting simulation of {n_samples} samples")
        theta, x = self.simulate(n_samples)
        logger.info("Simulation complete. Saving to Arrow file...")
        self.save(file_path, theta, x)
