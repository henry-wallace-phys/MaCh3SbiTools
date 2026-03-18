from typing import Sequence
import numpy as np

from mach3sbitools.types import NominalError, BoundaryConditions

class DummySimulator:
    def __init__(self, _: str) -> None:
        # Dummy config will do nothing
        self.n_params = 10

    def simulate(self, theta: Sequence[float]) -> Sequence[float]:
        return np.ones(self.n_params)

    def get_parameter_names(self)-> Sequence[str]:
        return [f'theta_{i}' for i in range(self.n_params)]

    def get_is_flat(self, i: int) -> bool:
        # Have some flat params
        return i<3

    def get_data_bins(self)->Sequence[float]:
        return np.ones(self.n_params)

    def get_bounds(self) -> BoundaryConditions:
        return np.zeros(self.n_params), np.ones(self.n_params)*2

    def get_nominal_error(self)->NominalError:
        return np.ones(self.n_params), np.ones(self.n_params)

    def get_covariance_matrix(self)->np.ndarray:
        return np.identity(self.n_params, dtype=np.float32)