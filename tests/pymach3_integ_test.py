import os
from pathlib import Path

import pytest

from mach3sbitools.examples import pyMaCh3Simulator
from mach3sbitools.simulator import Simulator, SimulatorProtocol

test_possible = pyMaCh3Simulator is not None


@pytest.fixture(scope="module")
def pymach3_simulator():
    return Simulator(
        module_name="mach3sbitools.examples",
        class_name="pyMach3Simulator",
        config=Path(os.getenv("MACH3")) / "TutorialConfigs" / "FitterConfig.yaml",
    )


@pytest.fixture(scope="module")
def pymach3_instance():
    return pyMaCh3Simulator(
        Path(os.getenv("MACH3")) / "TutorialConfigs" / "FitterConfig.yaml"
    )


@pytest.mark.skipif(not test_possible, reason="pyMach3 is not installed")
@pytest.mark.slow
def test_protocol(pymach3_simulator):
    assert isinstance(pymach3_simulator.simulator_wrapper, SimulatorProtocol)


@pytest.mark.skipif(not test_possible, reason="pyMach3 is not installed")
@pytest.mark.slow
def test_simulator(pymach3_simulator):
    samples = pymach3_simulator.simulate(10)
    assert len(samples) == 10
