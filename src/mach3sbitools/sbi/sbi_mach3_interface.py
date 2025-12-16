from pyMaCh3Tutorial import MaCh3TutorialWrapper
import torch
from abc import ABC, abstractmethod
from mach3sbitools.utils.device_handler import TorchDeviceHander

class MaCh3SBIInterface(ABC):
    device_handler = TorchDeviceHander()
    def __init__(self, handler: MaCh3TutorialWrapper, n_rounds: int, samples_per_round: int, prior):
        self._simulator = handler
        self._n_rounds = n_rounds
        self._prior = prior
        self._posterior = self._proposal = prior
        self._samples_per_round = samples_per_round

    @property
    def posterior(self):
        return self._posterior

    @property
    def x0(self):
        return self._simulator.get_data_bins()
    
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
        for r in range(self._n_rounds):
            self.training_iter(sampling_settings, training_settings)
            
    @abstractmethod
    def training_iter(sampling_settings, training_settings):
        ...