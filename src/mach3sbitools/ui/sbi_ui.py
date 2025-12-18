from pathlib import Path

from mach3sbitools.sbi.sbi_factory import sbi_factory
from mach3sbitools.mach3_interface.mach3_interface import MaCh3Interface

    
class MaCh3SbiUI:
    def __init__(self, input_config_path: Path):
        self._mach3 = MaCh3Interface(str(input_config_path))
        self._fitter = None
        
    @property
    def mach3(self):
        return self._mach3
        
    def run_fit(self, fit_type, n_rounds, samples_per_round, sampling_settings={}, training_settings={}, autosave_interval: int=-1, output_file: Path = Path("model_output.pkl")):
        self._fitter = sbi_factory(fit_type, self._mach3, n_rounds, samples_per_round, autosave_interval=autosave_interval, output_file=output_file)
        self._fitter.train(sampling_settings, training_settings)
        
    @property
    def fitter(self):
        return self._fitter