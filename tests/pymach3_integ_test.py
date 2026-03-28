import pytest

pytest.importorskip("pyMaCh3_tutorial")

import os
from pathlib import Path

from mach3sbitools.examples.pyMaCh3 import pyMaCh3Simulator
from mach3sbitools.simulator import Simulator, SimulatorProtocol


# The importer will return none
@pytest.fixture(scope="module")
def pymach3_simulator():
    return Simulator(
        module_name="mach3sbitools.examples.pyMaCh3",
        class_name="pyMach3Simulator",
        config=Path(os.getenv("MACH3")) / "TutorialConfigs" / "FitterConfig.yaml",
    )


@pytest.fixture(scope="module")
def pymach3_instance():
    return pyMaCh3Simulator(
        Path(os.getenv("MACH3")) / "TutorialConfigs" / "FitterConfig.yaml"
    )


@pytest.mark.slow
@pytest.mark.mach3_tutorial
def test_protocol(pymach3_simulator):
    assert isinstance(pymach3_simulator.simulator_wrapper, SimulatorProtocol)


@pytest.mark.slow
@pytest.mark.mach3_tutorial
def test_simulator(pymach3_simulator):
    samples = pymach3_simulator.simulate(10)
    assert len(samples) == 10
