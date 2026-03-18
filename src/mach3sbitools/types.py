from typing import Tuple, Sequence
import pandas as pd

# Boundary condition type
BoundaryConditions: type[tuple]
NominalError: type[tuple]
BoundaryConditions = NominalError = Tuple[Sequence[float], Sequence[float]]

PlotDict = dict[str, pd.DataFrame]