from pyMaCh3Tutorial import MaCh3TutorialWrapper
# from mach3sbitools.sbi.mach3_prior import MaCh3TorchPrior
from mach3sbitools.utils.device_handler import TorchDeviceHander

from sbi.utils import BoxUniform

import torch
from abc import ABC, abstractmethod
from tqdm.autonotebook import tqdm

class MaCh3SBIInterface(ABC):
    device_handler = TorchDeviceHander()
    def __init__(self, handler: MaCh3TutorialWrapper, n_rounds: int, samples_per_round: int):
        self._simulator = handler
        self._n_rounds = n_rounds
        # self._prior = MaCh3TorchPrior(handler, self.device_handler)
        
        low, high = handler.get_bounds()
        
        self._prior = BoxUniform(self.device_handler.to_tensor(low),
                                 self.device_handler.to_tensor(high))
        
        self._posterior = self._proposal = self._prior
        self._samples_per_round = samples_per_round

    @property
    def posterior(self):
        return self._posterior

    @property
    def x0(self):
        return self.device_handler.to_tensor(self._simulator.get_data_bins())
    
    def sample(self, n_samples: int, **kwargs):
        return self._posterior.sample((n_samples, ), **kwargs)

    def get_x_vals(self, theta: torch.Tensor):
        # Convert to numpy array on  CPU
        theta_cpu = theta.cpu().numpy()
        sims = self._simulator.simulate(theta_cpu)
        return self.device_handler.to_tensor(sims)
    
    def simulate(self, **kwargs):
        theta = self._proposal.sample((self._samples_per_round, ), **kwargs)
        x = self.get_x_vals(theta)
        return x, theta
    
    def train(self, sampling_settings, training_settings):
        self._proposal = self._prior
        for r in tqdm(range(self._n_rounds)):
            self.training_iter(sampling_settings, training_settings)
            
    @abstractmethod
    def training_iter(self, sampling_settings, training_settings):
        ...