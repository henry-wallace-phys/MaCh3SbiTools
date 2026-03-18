from fnmatch import fnmatch
from pathlib import Path
from typing import TypedDict

import numpy as np
from pyarrow import Table, feather

from mach3sbitools.types import SimulatorData, SimulatorDataGrouped


class FeatherOutput(TypedDict):
    x: SimulatorData
    theta: SimulatorData


def filter_nuisance(
    parameter_names: list[str], nuisance_pars: list[str], theta: SimulatorData
) -> SimulatorData:
    """
    Filters out nuisance parameters + reduces the size of input parameter array

    :param parameter_names: Names of parameters
    :param nuisance_pars: Parameters to filter out (works with blah*)
    :param theta: Theta values
    :return:
    """
    if len(theta[0]) != len(parameter_names):
        raise ValueError("Parameter names and theta must have same length")

    if nuisance_pars is None:
        return theta

    param_filter = np.array(
        [
            not any(fnmatch(param, nuis) for nuis in nuisance_pars)
            for param in parameter_names
        ],
        dtype=bool,
    )

    return theta[:, param_filter].copy()


def from_feather(
    file_name: Path, parameter_names: list[str], nuisance_pars: list[str] | None = None
) -> SimulatorDataGrouped:
    """
    Gets parameters from a feather file
    :param file_name: File to load
    :param parameter_names: Parameter names
    :param nuisance_pars: Parameters to filter out (works with blah*)
    :return:
    """
    if not file_name.exists():
        raise FileNotFoundError(file_name)

    table = feather.read_feather(str(file_name))
    # Get parameters
    theta = np.array(table["theta"].to_list(), dtype=np.float32)
    x = np.array(table["x"].to_list(), dtype=np.float32)

    # If nuisance parameters are defined we need to filter them out
    if nuisance_pars is not None:
        theta = filter_nuisance(parameter_names, nuisance_pars, theta)

    # Now get parameter_names

    return theta, x


def to_feather(
    file_name: Path,
    theta_values: SimulatorData,
    x_values: SimulatorData,
) -> None:
    """
    Stores files in a feather format
    :param file_name: File to store in
    :param theta_values: Parameter values
    :param x_values: Bin values
    :return:
    """

    if file_name.suffix != ".feather":
        raise ValueError("Must store outputs files with the *.feather extension")

    param_dict: FeatherOutput = {"x": x_values.tolist(), "theta": theta_values.tolist()}

    param_table = Table.from_pydict(param_dict)

    file_name.parent.mkdir(parents=True, exist_ok=True)
    feather.write_feather(param_table, str(file_name))
