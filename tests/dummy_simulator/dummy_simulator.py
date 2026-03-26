from collections.abc import Sequence
from typing import cast

import numpy as np

from mach3sbitools.types import BoundaryConditions

THETA_DIM = 30
X_DIM = 12


class DummySimulator:
    def __init__(self, _: str) -> None:
        # Dummy config will do nothing
        ...

    @property
    def n_params(self):
        return THETA_DIM

    def simulate(self, _: Sequence[float]) -> Sequence[float]:
        return cast(Sequence[float], np.ones(X_DIM).tolist())

    def get_parameter_names(self) -> Sequence[str]:
        return [f"theta_{i}" for i in range(THETA_DIM)]

    def get_is_flat(self, i: int) -> bool:
        # Have some flat params
        return i < 3

    def get_data_bins(self) -> Sequence[float]:
        return cast(Sequence[float], np.ones(X_DIM).tolist())

    def get_parameter_bounds(self) -> BoundaryConditions:
        lower = -5 * np.ones(THETA_DIM)
        upper = 5 * np.ones(THETA_DIM)

        # Keep cyclical param (index 9) at ±2π
        lower[9] = -2 * np.pi
        upper[9] = 2 * np.pi

        # Flat params (0-2) need finite bounds for Uniform
        lower[:3] = 0.0
        upper[:3] = 2.0

        return cast(BoundaryConditions, (lower.tolist(), upper.tolist()))

    def get_parameter_nominals(self) -> Sequence[float]:
        return cast(Sequence[float], np.ones(THETA_DIM).tolist())

    def get_parameter_errors(self) -> Sequence[float]:
        return cast(Sequence[float], np.ones(THETA_DIM).tolist())

    def get_covariance_matrix(self) -> np.ndarray:
        return np.identity(THETA_DIM, dtype=np.float32)

    def get_log_likelihood(self, theta) -> float:
        np_theta = np.array(theta, dtype=float)
        np_data = np.array(self.get_data_bins(), dtype=float)

        return float(np.sum(np_data * (1 - np.log(np_data / np_theta)) - np_theta))
