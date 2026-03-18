import importlib
import inspect
import pkgutil
from collections.abc import Callable
from difflib import get_close_matches
from importlib.util import find_spec
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

import numpy as np

from mach3sbitools.types import BoundaryConditions
from mach3sbitools.utils.logger import get_logger

logger = get_logger()

"""
A simulator injector. Simulators are expected to follow the SimulatorProtocol contract. Additionally they require
setup by some input file. This should set up the simulator in full. For MaCh3 this is the fitter YAML config.
"""

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SimulatorException(Exception):
    pass


class SimulatorImportError(SimulatorException):
    pass


class SimulatorImplementationError(SimulatorException):
    pass


class SimulatorSetupError(SimulatorException):
    pass


# ---------------------------------------------------------------------------
# Types & Protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class SimulatorProtocol(Protocol):
    """
    Any simulator requires
    1. To be set up via configuration file
    2. Have AT LEAST these methods
    """

    def __init__(self, simulator_config: Path | str) -> None: ...

    # Get the simulation for a single input
    def simulate(self, theta: list[float]) -> list[float]: ...

    # Get the names for each theta
    def get_parameter_names(self) -> list[str]: ...

    # Get the bounds as a [lower, upper]
    def get_parameter_bounds(self) -> BoundaryConditions: ...

    # Check if a given parameter is flat
    def get_is_flat(self, i: int) -> bool: ...

    # Get the data bins (xo)
    def get_data_bins(self) -> list[float]: ...

    # Get the nominal (mean) values
    def get_parameter_nominals(self) -> list[float]: ...

    # Get the nominal (mean) values
    def get_parameter_errors(self) -> list[float]: ...

    # Get the covariance matrix
    def get_covariance_matrix(self) -> np.ndarray: ...


def _implements(proto: type) -> Callable[[type], type]:
    # Thanks stack overflow
    # https://stackoverflow.com/questions/62922935/python-check-if-class-implements-unrelated-interface
    """Creates a decorator for classes that checks that the decorated class implements the runtime protocol `proto`"""

    def _deco(cls_def):
        if issubclass(cls_def, proto):
            return cls_def
        raise SimulatorImplementationError(
            f"{cls_def} does not implement protocol {proto}. Please see {__file__} for the implmentation."
        )

    return _deco


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _closest_match(name: str, candidates: list[str]) -> str | None:
    # Get the closest match for 'name' and a list of candidate names
    matches = get_close_matches(name, candidates, n=1, cutoff=0.6)
    return matches[0] if matches else None


def _hint(name: str, candidates: list[str]) -> str:
    # Generate hint text or errors
    match = _closest_match(name, candidates)
    return f" Did you mean: {match}?" if match else ""


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def get_simulator(module_name: str, class_name: str, config: Path) -> SimulatorProtocol:
    """
    Dynamically injects a simulator into the package. NOTE it must follow SimulatorProtocol.
    :param module_name: The name of the module i.e. mymodule.myclass...
    :param class_name: The name of the simulator class in the module (same as doing from <module_name> import <class_name>
    :param config: The config file for the simulator. All simulators are required have a config
    :return:
    """
    if find_spec(module_name) is None:
        installed = [m.name for m in pkgutil.iter_modules()]
        raise SimulatorImportError(
            f"Module '{module_name}' not found.{_hint(module_name, installed)}"
        )

    module = importlib.import_module(module_name)
    logger.info("Found simulator '%s'", module_name)

    if not hasattr(module, class_name):
        all_classes = [n for n, _ in inspect.getmembers(module, inspect.isclass)]
        raise SimulatorImportError(
            f"Class '{class_name}' not found in '{module_name}'.{_hint(class_name, all_classes)}"
        )

    simulator_cls = getattr(module, class_name)
    simulator_cls = _implements(SimulatorProtocol)(simulator_cls)
    logger.info("Imported simulator '%s' from '%s'", class_name, module_name)

    if not config.exists():
        raise SimulatorSetupError(f"Config file not found: {config}")

    logger.info("Found simulator config '%s'", config)
    return cast(SimulatorProtocol, simulator_cls(str(config)))
