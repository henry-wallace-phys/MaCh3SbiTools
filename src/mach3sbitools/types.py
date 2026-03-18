from collections.abc import Sequence

import pandas as pd

# Boundary condition type
BoundaryConditions: type[tuple]
NominalError: type[tuple]
BoundaryConditions = NominalError = tuple[Sequence[float], Sequence[float]]

PlotDict = dict[str, pd.DataFrame]
