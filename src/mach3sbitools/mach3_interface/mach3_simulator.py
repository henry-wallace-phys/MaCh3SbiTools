from pathlib import Path
from typing import Iterable, Tuple, Optional, List

from tqdm import tqdm
import pyarrow as pa
import pyarrow.feather as feather
import numpy as np

from mach3sbitools.mach3_interface.mach3_importer import get_mach3_wrapper
from mach3sbitools.mach3_interface.mach3_prior import create_mach3_prior
from mach3sbitools.utils.device_handler import TorchDeviceHander


class MaCh3Simulator:
    def __init__(self, mach3_name: str, config_file: Path, nuisance_pars: Optional[List[str]]=None):
        device_handler = TorchDeviceHander()
        self._mach3_type = mach3_name
        self.mach3_wrapper = get_mach3_wrapper(mach3_name, config_file)
        self.prior = create_mach3_prior(self.mach3_wrapper, device_handler.device, nuisance_pars=nuisance_pars)
    
    def simulate_mach3(self, n_samples: int) -> Tuple[Iterable, Iterable]:
        """
        Samples data from the specified MaCh3 using the provided configuration and parameters.

        Args:
            mach3_name (str): The name of the MaCh3 to use.
            config_file (Path): Path to the configuration file for the MaCh3.
            sample_params (dict): Parameters for sampling.

        Returns:
            Tuple[Iterable, Iterable]: Valid theta and x values.
        """
        
        samples = self.prior.sample((n_samples,))
        theta = samples.cpu().numpy()
        
        valid_theta = []
        valid_x = []
        for t in tqdm(theta, desc=f"Simulating from MaCh3: {self._mach3_type}"):
            try:
                x = self.mach3_wrapper.simulate(t)
                valid_theta.append(t)
                valid_x.append(np.random.poisson(x))
            except Exception:
                print("Error: Bad simulation! Skipping sample.")

        return valid_theta, valid_x

    def save_to_arrow(self, file_path: Path, theta: Iterable, x: Iterable) -> None:
        """
        Saves the sampled data to an Arrow file.

        Args:
            file_path (Path): Path to the output Arrow file.
            theta (Iterable): Sampled theta values.
            x (Iterable): Sampled x values.
        """

        table = pa.Table.from_pydict({"theta": theta, "x": x})
        feather.write_feather(table, str(file_path))

    def __call__(self, n_samples: int, file_path: Path) -> None:
        print(f"Starting simulation of {n_samples} samples from MaCh3: {self._mach3_type}")
        theta, x = self.simulate_mach3(n_samples)
        print("Simulation complete. Saving to Arrow file...")
        self.save_to_arrow(file_path, theta, x)