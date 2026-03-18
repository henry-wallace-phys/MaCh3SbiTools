from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

SimulatorData = NDArray[np.float32]
SimulatorDataGrouped = tuple[SimulatorData, SimulatorData]

# Boundary condition type
BoundaryConditions = tuple[list[float], list[float]]
