import os
import sys
from pathlib import Path

import yaml

from mach3sbitools.utils import get_logger

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "physics_engine"))
from physics_engine import ParameterHandler, SampleHandler

logger = get_logger("tutorial")


class MySimulator:
    def __init__(self, config_path: Path):
        if not isinstance(config_path, Path):
            config_path = Path(config_path)

        self.parameter_handler = ParameterHandler(config_path)
        sample_config = self._get_sample_yaml(config_path)
        self.sample_handler = SampleHandler(sample_config, self.parameter_handler)

    # Helper method
    @classmethod
    def _get_sample_yaml(cls, physics_config: Path):
        # First we get the config and check it exists

        if not physics_config.is_file() and not physics_config.exists():
            raise FileNotFoundError("Config file not found.")

        # We extract the sample config from the physics config. This is a tad hacky but
        # is what's currently done in MaCh3
        with open(physics_config) as f:
            physics_open = yaml.safe_load(f)
            sample_config = Path(physics_open["Sample"]["sample_config"])

        # We also need to get the SAMPLE config from the physics config!
        if not sample_config.is_file():
            raise ValueError("Config file not found.")

        return sample_config
