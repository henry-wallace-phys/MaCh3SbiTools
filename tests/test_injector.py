import pkgutil

import pytest

from mach3sbitools.simulator.simulator_injector import (
    SimulatorImplementationError,
    SimulatorImportError,
    SimulatorProtocol,
    _hint,
    get_simulator,
)
from mach3sbitools.utils import get_logger

logger = get_logger()

# Tests the injector
def test_import(simulator_module, simulator_class, dummy_config):
    simulator = get_simulator(simulator_module,
                              simulator_class,
                              dummy_config)
    assert isinstance(simulator, SimulatorProtocol)

def test_relative_import(simulator_module, simulator_class, dummy_config):
    # Check relative import
    simulator = get_simulator(f"{simulator_module}.dummy_simulator",
                            simulator_class,
                            dummy_config)
    assert isinstance(simulator, SimulatorProtocol)


def test_protocol_followed(simulator_module, dummy_config):
    # Checks a pre-built class that doesn't follow protocol
    bad_sim_class = "PoorlyDefinedSimulator"
    with pytest.raises(SimulatorImplementationError):
        get_simulator(simulator_module,
                      bad_sim_class,
                      dummy_config)

def test_importing(simulator_module):
    with pytest.raises(SimulatorImportError):
        get_simulator("ABadPythonSimulator", "NotAClass", "")

    with pytest.raises(SimulatorImportError):
        get_simulator(simulator_module, "NotAClass", "")

def test_hint(simulator_module, simulator_class):
    # Check our hints are doing what we expect
    installed = [m.name for m in pkgutil.iter_modules()]

    sim_copy = simulator_module.capitalize()
    hint = _hint(sim_copy, installed)
    assert hint == f" Did you mean: {simulator_module}?"

    sim_copy = simulator_module[:-1]+"Z"
    hint = _hint(sim_copy, installed)
    assert hint == f" Did you mean: {simulator_module}?"