from pathlib import Path

from mach3sbitools.file_io.file_handler_factory import file_handler_factory
from mach3sbitools.sbi.sbi_factory import sbi_factory

from pyMaCh3Tutorial import MaCh3TutorialWrapper

class FileSbiUI:
    def __init__(self, input_file_path: Path, file_type: str='root'):
        self._file_handler = file_handler_factory(file_type, input_file_path)
        self._fitter = None
        
    @property
    def file_handler(self):
        return self._file_handler
    
    def set_prior(self, prior):
        self._prior = prior
    
    def run_fit(self, fit_type, n_rounds, x0, **kwargs):
        self._fitter = sbi_factory(fit_type, self._file_handler, n_rounds, self._prior)
        self._fitter.x0 = x0
        self._fitter.train(**kwargs)
        
    @property
    def fitter(self):
        return self._fitter
    
class MaCh3SbiUI:
    def __init__(self, input_config_path: Path):
        self._mach3 = MaCh3TutorialWrapper(input_config_path)
        self._fitter = None
        
    @property
    def mach3(self):
        return self._mach3
        
    def run_fit(self, fit_type, n_rounds, samples_per_round, sampling_settings={}, training_settings={}):
        self._fitter = sbi_factory(fit_type, self._mach3, n_rounds, samples_per_round)
        self._fitter.train(sampling_settings, training_settings)
        
    @property
    def fitter(self):
        return self._fitter