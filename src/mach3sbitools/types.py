from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

# Basically for x, theta pairs
SimulatorData: TypeAlias = NDArray[np.float32]
SimulatorDataGrouped: TypeAlias = tuple[SimulatorData, SimulatorData]

# For lower, upper bounds
BoundaryConditions: TypeAlias = tuple[list[float], list[float]]
