from abc import ABC, abstractmethod
from typing import Optional

from sbi.inference.trainers.npe.npe_base import PosteriorEstimatorTrainer
import torch
from tqdm.rich import tqdm

from mach3sbitools.file_io.file_handler_base import FileHandlerBase
from mach3sbitools.utils.device_handler import TorchDeviceHander

class SbiInterface(ABC):
    _inference: PosteriorEstimatorTrainer
    def __init__(self, file_handler: FileHandlerBase, n_rounds: int, prior):
        self._file_handler = file_handler
        self._n_rounds = n_rounds
        self._prior = prior
        self._x0 = None
        self._posterior = None
        
        self.device_handler = TorchDeviceHander()

    @property
    def posterior(self):
        return self._posterior
    
    @property
    def inference(self):
        return self._inference
    
    # Useful helper function
    @property
    def x(self):
        return self._file_handler.x
    
    @property
    def theta(self):
        return self._file_handler.theta
            
    @property
    def x0(self):
        return self._x0
    
    @x0.setter
    def x0(self, x0: torch.Tensor):
        self._x0 = x0
        
    
    def load_x_theta(self, batch_num: Optional[int] = None, **kwargs):
        if batch_num is not None:
            self._file_handler.set_batch_mode(True)

            kwargs['batch_num'] = batch_num
            kwargs['n_batches'] = self._n_rounds 
            
            # Now it can load
            self._file_handler.load_x_theta(**kwargs)
           
    @abstractmethod 
    def training_iter(self, iter: int, file_args: dict, **kwargs):
        ...        

    def train(self, **kwargs):
        file_args = kwargs.get('TTreeSettings', {})

        if self.x0 is None:
            raise ValueError("Cannot perform inference without a set x0!")
        
        for r in range(self._n_rounds):
            self.training_iter(r, file_args, **kwargs)
    
    def sample(self, n_samples: int, **kwargs):
        return self._posterior.sample((n_samples, ), **kwargs)
        