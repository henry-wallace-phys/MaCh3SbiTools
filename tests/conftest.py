from pathlib import Path

import pytest

@pytest.fixture(scope="session")
def simulator_module()->str:
    return "dummy_simulator"

@pytest.fixture(scope="session")
def simulator_class()->str:
    return "DummySimulator"

@pytest.fixture(scope="session")
def dummy_config()->Path:
    return Path(__file__).parent/"dummy_simulator"/"dummy_config.yaml"