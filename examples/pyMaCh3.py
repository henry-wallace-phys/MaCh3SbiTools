import yaml
from pathlib import Path
import os

import pyMaCh3_tutorial as m3
'''
Code designed to work with MaCh3 Tutorial!
'''

class MaCh3Error(Exception):
    pass

class MaCh3ConfigError(MaCh3Error):
    pass

class MaCh3EnvironmentError(MaCh3Error):
    pass

def parse_mach3_yaml(mach3_config: Path)->tuple[list[str], list[str]]:
    if not mach3_config.is_file():
        raise MaCh3ConfigError(f"Cannot find {mach3_config}")

    with open(mach3_config, "r") as f:
        mach3_yaml = yaml.safe_load(f)

    # We now extract the systematics from the yaml
    systematics_info = mach3_yaml.get('General', {}).get('Systematics', {})
    systematics = systematics_info.get('XsecCovFile')

    if not systematics:
        raise MaCh3ConfigError(f"Cannot find systematics in {mach3_config}")

    for fixed_par in systematics_info.get('XsecFix'):
        systematics.toggle_fix_parameter(fixed_par)

    # First extract the samples
    samples: list[str] = mach3_yaml.get('General', {}).get('TutorialSamples', [])

    if not samples:
        raise MaCh3ConfigError(f"Cannot find any samples (General::TutorialSamples) in {mach3_config}")

    return systematics, samples

def create_mach3_handlers(mach3_config: Path)->tuple[m3.parameters.ParameterHandlerGeneric, m3.samples.SampleHandlerTutorial]:
    systematics, samples = parse_mach3_yaml(mach3_config)
    parameter_handler = m3.parameters.ParameterHandlerGeneric(systematics)
    sample_handler = m3.samples.SampleHandlerTutorial(systematics)

    return parameter_handler, sample_handler

class pyMaCh3Wrapper:
    def __init__(self, mach3_config: Path):
        if os.getenv("MACH3") is None:
            raise MaCh3EnvironmentError("Need to set MACH3 environment variable to point to MaCh3 directory")

        self._parameter_handler, self._sample_handler = create_mach3_handlers(mach3_config)

    def get_data_bins(self):
        return self._sample_handler.get_data_hist()[0]

    def