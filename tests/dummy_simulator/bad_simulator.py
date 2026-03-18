from collections.abc import Sequence

import numpy as np


class PoorlyDefinedSimulator:
    def __init__(self, config: str) -> None:
        # Dummy config will do nothing
        self.n_params = 10

    def simulate(self, theta: Sequence[float]) -> None:
        return None

    def get_parmeter_names(self) -> Sequence[str]:
        return [f"theta_{i}" for i in range(self.n_params)]

    def get_is_flat(self, i: int) -> bool:
        # Have some flat params
        return i < 3

    def get_data_bins(self) -> str:
        return "no"

    def get_bounds(self) -> int:
        return 0

    def get_nominals(self):
        return np.ones(self.n_params), np.ones(self.n_params)

    def get_corelation_matrix(self) -> np.ndarray:
        return np.identity(self.n_params, dtype=np.float32)
